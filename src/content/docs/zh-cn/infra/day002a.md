---
title: Day 2 上午 - 计算调度与存储层级优化
description: 计算调度与存储层级优化
time: 2026-02-03
---

## 计算调度与存储层级优化

**学习阶段**：Day 2 上午
**核心议题**：Warp 调度机制、资源占用率与指令级并行、内存访问模式优化
**理论基石**：CUDA 硬件架构原理、Roofline Model、延迟掩盖策略

---

## 1. 性能优化的核心误区与重构

在着手具体的硬件参数计算前，我们必须首先修正对于 GPU 性能指标的根本性认知。在 AI Infra 的底层开发中，存在一个普遍的误区，即盲目追求 **占用率（Occupancy）** 的最大化。

### 1.1 占用率与吞吐量的辩证关系
Occupancy 定义为当前 **流式多处理器（Streaming Multiprocessor, SM）** 上活跃的 Warp 数量与硬件支持的最大 Warp 数量之比。然而，高 Occupancy 并不等同于高 **指令吞吐量（Instruction Throughput）**。
* **延迟掩盖（Latency Hiding）的本质**：提升 Occupancy 的初衷是为了利用 **线程级并行（Thread-Level Parallelism, TLP）** 来掩盖内存访问或算术指令的延迟。
* **边际递减效应**：当 Occupancy 达到一定阈值（通常在 30%-50% 左右，视具体算子特性而定），足以掩盖大部分流水线停顿时，继续提升 Occupancy 不会带来性能收益。
* **寄存器溢出风险**：过度追求高并行度往往意味着限制每个线程可用的寄存器数量。一旦发生 **寄存器溢出（Register Spilling）**，数据将被迫存储至 L1 缓存甚至本地内存（Local Memory），导致 **本地内存流量（Local Memory Traffic）** 激增，直接抢占全局内存带宽，从而导致性能断崖式下跌。

**结论**：性能优化的终极目标是指令吞吐量，Occupancy 仅是达成此目标的手段之一，而非结果。

---

## 2. Warp 调度机制与延迟掩盖

SM 的核心是一个大规模多线程调度器。理解其工作原理是掌握 GPU 并行计算的关键。

### 2.1 候选 Warp（Eligible Warps）判定标准
在每个时钟周期，调度器从驻留在 SM 上的众多 Warp 中选择指令发射。一个 Warp 要成为 Eligible Warp，必须同时满足以下三个微架构层面的条件：
1.  **指令获取就绪（Instruction Fetch Ready）**：下一条指令已从指令缓存（I-Cache）中取出并解码。
2.  **操作数就绪（Arguments Ready）**：指令所需的源操作数（寄存器或共享内存数据）已准备完毕，没有未完成的依赖项（如等待内存回传）。
3.  **执行单元就绪（Execution Unit Ready）**：目标执行单元（如 FP32 Core, Tensor Core, LSU）处于空闲状态。

### 2.2 分支发散（Branch Divergence）的代价
调度器以 Warp（32 个线程）为原子单位进行调度。如果 Kernel 逻辑中存在大量条件分支，导致同一 Warp 内的线程走向不同的执行路径，硬件将序列化执行这些路径（Masking 机制）。此时，即便调度器全速运转，**有效指令吞吐量（Effective Instruction Throughput）** 也会因活跃线程掩码（Active Mask）的减少而大幅降低。

---

## 3. 资源分配粒度与并行度限制

计算 GPU 的理论并行度时，必须考虑硬件资源的 **分配粒度（Allocation Granularity）**。

### 3.1 资源硬性分区（Hard Partitioning）
SM 的资源（寄存器文件、共享内存）并非以线程为单位分配，而是以 **线程块（Thread Block）** 为最小粒度进行预留。
* **木桶效应**：限制 SM 上驻留 Block 数量的瓶颈，通常取决于最紧缺的那一种资源（通常是共享内存或寄存器总量）。
* **粒度损耗**：例如，若 SM 有 64KB 共享内存，而一个 Block 申请了 33KB。虽然剩余 31KB，但由于无法容纳第二个完整的 Block，这 31KB 空间将被闲置。这种物理上的硬性分区直接决定了 **理论最大 Occupancy**。

### 3.2 带宽墙与 MSHRs 限制
通过增加 Block 数量（即提升 TLP）来掩盖延迟并非无限可扩展，其物理极限受限于：
1.  **内存带宽饱和（Memory Bandwidth Saturation）**：过多的并发 Warp 同时发起内存请求会迅速填满内存控制器的事务队列。此时瓶颈由“延迟”转变为“带宽”。
2.  **MSHRs 耗尽**：每个 SM 用于追踪在途内存请求（Outstanding Loads）的 **缺失状态保持寄存器（Miss Status Holding Registers, MSHRs）** 数量有限。一旦耗尽，调度器将无法发射新的内存指令，导致流水线停顿。

### 3.3 指令级并行（Instruction-Level Parallelism, ILP）
当 TLP 受限（如 Shared Memory 占用过高导致 Occupancy 无法提升）时，我们应转向开发 ILP：
* **技术手段**：循环展开（Loop Unrolling）、增加单个线程的计算密度（每个线程处理更多数据）。
* **原理**：利用寄存器重命名和乱序执行（在编译器层面），在单线程内部发起多个独立的内存请求或算术指令，不依赖 Warp 切换即可填满流水线气泡。

---

## 4. 全局内存访问模式：合并访问（Memory Coalescing）

全局内存（Global Memory）的高延迟特性要求我们必须高效利用总线带宽。

### 4.1 合并访问黄金法则
硬件内存控制器以 **缓存行（Cache Line，通常 128 Bytes）** 为单位与 DRAM 交互。Warp 内 32 个线程的访存请求能否合并，取决于：
1.  **连续性（Contiguity）**：线程 $k$ 访问地址 $Address_k$，线程 $k+1$ 访问 $Address_k + sizeof(data)$。
2.  **对齐（Alignment）**：首地址应为 32、64 或 128 字节的倍数。

### 4.2 跨步访问（Strided Access）的性能惩罚
以矩阵处理为例，**行优先（Row-Major）** 存储的矩阵在进行列读取（`A[i][col]`）时会触发最差访问模式：
* **现象**：相邻线程访问的物理地址间隔一整行（Stride），远超 Cache Line 范围。
* **后果**：硬件必须为每个线程发起独立的内存事务。带宽利用率可能跌至 1/32（即 3.125%），且会导致 L1/L2 缓存被低局部性的数据频繁刷出（Cache Thrashing）。

### 4.3 性能指标分析：Sectors per Request
在使用 Nsight Compute 进行分析时，`l1tex__t_sectors_pipe_lsu_mem_global_op_ld` 与 requests 的比值是核心指标。
* **物理定义**：CUDA 架构中，一个 **扇区（Sector）** 为 32 Bytes。
* **理想值（Float32）**：一个 Warp 请求 $32 \times 4B = 128B$，对应 4 个 Sectors。因此，理想比值为 **4.0**。
* **异常值解读**：若比值达到 **8.0**（即每个请求传输 256B），说明平均每个请求传输了 8 个扇区，这通常意味着严重的非合并访问或未对齐访问，导致了 50% 以上的带宽浪费。

---

## 5. 共享内存优化：Bank Conflict 机制

**共享内存（Shared Memory）** 是片上高速缓存，被划分为 32 个独立的 **存储体（Banks）**，用于支持高并发访问。

### 5.1 冲突与广播机制
* **Bank Conflict**：当 Warp 内多个线程试图同时访问同一个 Bank 的 **不同地址** 时，硬件必须将这些请求 **串行化（Serialize）**，导致有效带宽成倍下降（最差情况下降至 1/32）。
* **广播（Broadcast）**：当 Warp 内多个线程访问同一个 Bank 的 **完全相同地址** 时，硬件触发广播机制，一次传输即可满足所有请求，不产生冲突。

### 5.2 消除冲突的策略：Padding vs. Swizzling
在处理矩阵转置或卷积 Tiling 时，常遇到 Bank Conflict。
* **Padding（填充）**：
    * *原理*：将声明从 `A[32][32]` 改为 `A[32][33]`。
    * *数学逻辑*：利用模运算特性。原索引映射为 `(row * 32 + col) % 32 = col`（列相同则 Bank 相同）；Padding 后映射为 `(row * 33 + col) % 32 = (row + col) % 32`。引入行号变量后，列元素的 Bank 索引发生错位，从而消除冲突。
    * *缺点*：破坏了内存布局的连续性，可能降低 L1 Cache 命中率并浪费物理空间。
* **Swizzling（重配/混洗）**：
    * *原理*：现代架构（如 NVIDIA Hopper）及高级库（CUTLASS）更倾向于使用异或（XOR）哈希。通过 `bank_idx = (row ^ col) % 32` 这种低开销的位运算重新映射地址。
    * *优势*：在保持数据紧凑存储（无需 Padding）的同时，利用 XOR 的随机散列特性打散访问模式，彻底消除冲突。

---

## 6. 总结：AI Infra 性能坐标系

通过 Day 2 上午的学习，我们建立了一个基于 **Roofline Model** 的多维性能分析坐标系：

| 维度 | 关键指标 | 优化方向 | 物理瓶颈 |
| :--- | :--- | :--- | :--- |
| **计算调度** | Warp Occupancy | 提升 TLP (适度) 提升 ILP (循环展开/寄存器复用) | 寄存器文件大小 SM 调度器吞吐 |
| **全局内存** | Sectors/Requests | 内存合并 (Coalescing) 对齐优化 | DRAM 带宽 MSHRs 容量 |
| **片上内存** | Bank Conflict | Padding (空间换时间) Swizzling (逻辑映射) | Shared Memory 带宽 Bank 串行化 |

接下来的研习将运用上述理论，深入 **Flash Attention** 等前沿算子的代码级优化实战。