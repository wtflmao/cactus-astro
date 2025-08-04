import { defineCollection } from 'astro:content';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';
import { z } from 'astro/zod';

export const collections = {
	docs: defineCollection({
		loader: docsLoader(),
		schema: docsSchema({
			extend: z.object({
				time: z.union([z.string(), z.date()]).optional(), 
			}),
		}),
	}),
	i18n: defineCollection({
		loader: i18nLoader(),
		schema: i18nSchema({
			extend: z.object({
				'customUi.welcome': z.string().optional(),
				'customUi.topics': z.string().optional(),
				'customUi.readMore': z.string().optional(),
				'customUi.backToTop': z.string().optional(),
			}),
		}),
	}),
};
