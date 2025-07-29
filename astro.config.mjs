// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

import cloudflare from '@astrojs/cloudflare';

import auth from 'auth-astro';

// https://astro.build/config
export default defineConfig({
	output: 'server',
    site: "https://cactus.hhzm.win",
	integrations: [starlight({
		title: 'Cactus',
		social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/withastro/starlight' }],
		sidebar: [
			{
				label: 'Guides',
				items: [
					// Each item here is one entry in the navigation menu.
					{ label: 'Example Guide', slug: 'guides/example' },
				],
			},
			{
				label: 'Reference',
				autogenerate: { directory: 'reference' },
			},
		],
	}), auth()],
	adapter: cloudflare({
		routes: {
			extend: {
				include: [
					{ pattern: '/api/auth/*' },
				],
			},
		},
		platformProxy: { enabled: true },
		sessionKVBindingName: 'ai-learn-astro',
		cloudflareModules: true
	}),
});