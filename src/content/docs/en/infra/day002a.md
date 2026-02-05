---
title: Day 2 morning - Computational Scheduling and Memory Hierarchy Optimization
description: Computational Scheduling and Memory Hierarchy Optimization
time: 2026-02-03
---

## Computational Scheduling and Memory Hierarchy Optimization

**Learning Phase**: Day 2 morning
**Core Topics**: Warp Scheduling Mechanisms, Occupancy vs. ILP, Memory Access Pattern Optimization
**Theoretical Foundations**: CUDA Hardware Architecture, Roofline Model, Latency Hiding Strategies

---

## 1. Core Misconceptions and Refactoring of Performance Optimization

Before diving into specific hardware parameter calculations, we must first correct the fundamental understanding of GPU performance metrics. In AI Infra low-level development, a common pitfall is the blind pursuit of maximizing **Occupancy**.

### 1.1 The Dialectical Relationship between Occupancy and Throughput
Occupancy is defined as the ratio of active **Warps** on a **Streaming Multiprocessor (SM)** to the maximum number of Warps supported by the hardware. However, high occupancy does not necessarily equate to high **Instruction Throughput**.
* **The Essence of Latency Hiding**: The original purpose of increasing occupancy is to leverage **Thread-Level Parallelism (TLP)** to hide the latency of memory accesses or arithmetic instructions.
* **Diminishing Marginal Returns**: Once occupancy reaches a certain threshold (typically around 30%-50%, depending on the operator's characteristics), it is sufficient to hide most pipeline stalls. Further increases in occupancy yield no performance gain.
* **Register Spilling Risk**: Over-pursuing high parallelism often means restricting the number of registers available to each thread. Once **Register Spilling** occurs, data is forced into the L1 cache or even Local Memory. This causes a surge in **Local Memory Traffic**, which directly competes for Global Memory bandwidth, leading to a performance cliff.

**Conclusion**: The ultimate goal of performance optimization is instruction throughput; Occupancy is merely a means to an end, not the end itself.

---

## 2. Warp Scheduling and Latency Hiding

The core of an SM is a massive multi-threaded scheduler. Understanding how it works is key to mastering GPU parallel computing.

### 2.1 Criteria for "Eligible Warps"
In every clock cycle, the scheduler selects instructions to issue from the numerous Warps residing on the SM. For a Warp to become an "Eligible Warp," it must simultaneously satisfy three micro-architectural conditions:
1.  **Instruction Fetch Ready**: The next instruction has been fetched from the Instruction Cache (I-Cache) and decoded.
2.  **Arguments Ready**: The source operands (registers or shared memory data) required by the instruction are ready, with no outstanding dependencies (e.g., waiting for memory returns).
3.  **Execution Unit Ready**: The target execution unit (e.g., FP32 Core, Tensor Core, LSU) is currently idle.

### 2.2 The Cost of Branch Divergence
The scheduler treats the Warp (32 threads) as an atomic unit. If the kernel logic contains significant conditional branching that causes threads within the same Warp to take different execution paths, the hardware will serialize the execution of these paths (via a Masking mechanism). In this scenario, even if the scheduler runs at full speed, the **Effective Instruction Throughput** drops significantly due to the reduction in the Active Mask.

---

## 3. Resource Allocation Granularity and Parallelism Limits

When calculating the theoretical parallelism of a GPU, one must consider the **Allocation Granularity** of hardware resources.

### 3.1 Hard Partitioning of Resources
SM resources (register files, shared memory) are not allocated per thread, but are reserved using the **Thread Block** as the minimum granularity.
* **The Cannikin Law (Shortest-Slab Effect)**: The bottleneck limiting the number of resident blocks on an SM is usually determined by the most scarce resource (typically shared memory or the total number of registers).
* **Granularity Loss**: For example, if an SM has 64KB of shared memory and one block requests 33KB, the remaining 31KB will sit idle because it cannot accommodate a second complete block. This physical hard partitioning directly determines the **Theoretical Maximum Occupancy**.

### 3.2 Bandwidth Walls and MSHR Limits
Increasing the number of blocks (i.e., boosting TLP) to hide latency is not infinitely scalable. Its physical limits are constrained by:
1.  **Memory Bandwidth Saturation**: Too many concurrent Warps initiating memory requests simultaneously will quickly fill the memory controller's transaction queue. At this point, the bottleneck shifts from "latency" to "bandwidth."
2.  **MSHR Exhaustion**: Each SM has a limited number of **Miss Status Holding Registers (MSHRs)** used to track outstanding memory loads. Once exhausted, the scheduler cannot issue new memory instructions, causing pipeline stalls.

### 3.3 Instruction-Level Parallelism (ILP)
When TLP is limited (e.g., high shared memory usage prevents occupancy from rising), we should pivot to exploiting ILP:
* **Technical Methods**: Loop unrolling, increasing computational density per thread (processing more data per thread).
* **Principle**: By utilizing register renaming and out-of-order execution (at the compiler level), multiple independent memory requests or arithmetic instructions can be issued within a single thread to fill pipeline bubbles without relying on Warp switching.

---

## 4. Global Memory Access: Memory Coalescing

The high-latency nature of Global Memory requires us to utilize bus bandwidth efficiently.

### 4.1 The Golden Rule of Coalescing
The hardware memory controller interacts with DRAM in units of **Cache Lines (typically 128 Bytes)**. Whether memory requests from 32 threads in a Warp can be coalesced depends on:
1.  **Contiguity**: Thread $k$ accesses $Address_k$, and thread $k+1$ accesses $Address_k + sizeof(data)$.
2.  **Alignment**: The starting address should be a multiple of 32, 64, or 128 bytes.

### 4.2 The Penalty of Strided Access
Taking matrix processing as an example, a matrix stored in **Row-Major** order will trigger the worst access pattern during column-wise reads (`A[i][col]`):
* **Phenomenon**: The physical addresses accessed by adjacent threads are separated by an entire row (Stride), far exceeding the Cache Line range.
* **Consequence**: The hardware must initiate an independent memory transaction for every single thread. Bandwidth utilization may drop to 1/32 (3.125%), and the L1/L2 caches will be frequently flushed by data with low locality (**Cache Thrashing**).

### 4.3 Metric Analysis: Sectors per Request
When using Nsight Compute, the ratio of `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` to `requests` is a core indicator.
* **Physical Definition**: In CUDA architecture, one **Sector** is 32 Bytes.
* **Ideal Value (Float32)**: A Warp requests $32 \times 4B = 128B$, corresponding to 4 Sectors. Thus, the ideal ratio is **4.0**.
* **Interpreting Anomalies**: If the ratio reaches **8.0** (meaning 256B per request), it indicates that each request is transferring 8 sectors on average. This usually signifies severe non-coalesced or unaligned access, resulting in over 50% bandwidth waste.

---

## 5. Shared Memory Optimization: Bank Conflict Mechanism

**Shared Memory** is on-chip high-speed cache divided into 32 independent **Banks** to support high-concurrency access.

### 5.1 Conflicts and Broadcast Mechanism
* **Bank Conflict**: When multiple threads within a Warp attempt to access **different addresses** within the same Bank simultaneously, the hardware must **serialize** these requests, causing effective bandwidth to drop (down to 1/32 in the worst case).
* **Broadcast**: When multiple threads within a Warp access the **exact same address** in the same Bank, the hardware triggers a broadcast mechanism, satisfying all requests in a single cycle without conflict.

### 5.2 Conflict Mitigation Strategies: Padding vs. Swizzling
Bank conflicts are common in matrix transposition or Convolution Tiling.
* **Padding**:
    * *Principle*: Changing a declaration from `A[32][32]` to `A[32][33]`.
    * *Mathematical Logic*: Leveraging modulo arithmetic. The original index maps to `(row * 32 + col) % 32 = col` (same column results in same bank). With padding, it maps to `(row * 33 + col) % 32 = (row + col) % 32`. By introducing the row variable, the bank indices of column elements are shifted, eliminating conflicts.
    * *Drawback*: It breaks memory layout continuity, potentially lowering L1 Cache hit rates and wasting physical space.
* **Swizzling**:
    * *Principle*: Modern architectures (like NVIDIA Hopper) and advanced libraries (CUTLASS) prefer using XOR hashing. Addresses are remapped via low-overhead bitwise operations like `bank_idx = (row ^ col) % 32`.
    * *Advantage*: It maintains compact storage (no padding needed) while utilizing the random hashing property of XOR to scatter access patterns and eliminate conflicts entirely.

---

## 6. Summary: The AI Infra Performance Coordinate System

Through the Day 2 morning session, we have established a multi-dimensional performance analysis coordinate system based on the **Roofline Model**:

| Dimension | Key Metric | Optimization Direction | Physical Bottleneck |
| :--- | :--- | :--- | :--- |
| **Computation Scheduling** | Warp Occupancy | Increase TLP (moderately) Increase ILP (Unrolling/Register Reuse) | Register File Size SM Scheduler Throughput |
| **Global Memory** | Sectors/Requests | Memory Coalescing Alignment Optimization | DRAM Bandwidth MSHR Capacity |
| **On-chip Memory** | Bank Conflict | Padding (Space for Time) Swizzling (Logical Mapping) | Shared Memory Bandwidth Bank Serialization |

The next session will apply these theories to code-level optimization practice for cutting-edge operators like **Flash Attention**.