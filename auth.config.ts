// auth.config.ts
import { defineConfig } from 'auth-astro';
import Discord from '@auth/core/providers/discord';

export default defineConfig({
    providers: [
        Discord({
            clientId: process.env.DISCORD_CLIENT_ID!,
            clientSecret: process.env.DISCORD_CLIENT_SECRET!,
        }),
    ],
    callbacks: {
        async session({ session, user }) {
        session.user.id = user.id;
        return session;
        },
    },
});