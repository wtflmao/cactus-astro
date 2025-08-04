// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightThemeRapide from 'starlight-theme-rapide'

import cloudflare from '@astrojs/cloudflare';

// https://astro.build/config
export default defineConfig({
	output: 'server',
	site: "https://cactus.hhzm.win",
	integrations: [
		starlight({
			title: {
				'en': 'Cactus',
				'zh-CN': 'Cactus',
			},
			description: "Cactus is a platform for sharing and discovering AI related topics.",
			plugins: [starlightThemeRapide()],
			customCss: [
				'./src/styles/custom.css',
			],
			components: {
				LastUpdated: './src/components/LastUpdated.astro',
			},
			defaultLocale: 'zh-cn',
			locales: {
				'en': {
					label: 'English',
					lang: 'en',
				},
				'zh-cn': {
					label: '简体中文',
					lang: 'zh-CN',
				},
			},
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/withastro/starlight' }],
			sidebar: [
				{
					label: '',
					translations: {
						'en': 'Topics',
						'zh-cn': '有趣的话题',
						'zh-CN': '有趣的话题',
					},
					autogenerate: { directory: 'topics' },
				},
			],
		}),
	],
	adapter: cloudflare({
		platformProxy: { enabled: true },
		sessionKVBindingName: 'cactus',
		cloudflareModules: true
	}),
});
