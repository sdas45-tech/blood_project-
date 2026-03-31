/**
 * Configuration Manager for Blood Health Advisor
 * Handles environment-specific API base URLs
 */

// ─── ENVIRONMENT DETECTION ──────────────────────
const ENVIRONMENT = {
    LOCAL: {
        name: "local",
        baseUrl: "http://127.0.0.1:8022",
        description: "Local development server"
    },
    PRODUCTION: {
        name: "production",
        baseUrl: "https://server.uemcseaiml.org/blood",
        description: "UEM Server (Production) via nginx proxy"
    }
};

// ─── ACTIVE CONFIGURATION ───────────────────────
// MANUALLY TOGGLE WHICH ENVIRONMENT TO USE:
// 
// ⬇️ UNCOMMENT ONE OF THESE TWO LINES:
//
// For LOCAL development (test locally on your machine):
//const ACTIVE_ENV = ENVIRONMENT.LOCAL;
//
// For PRODUCTION deployment (Vercel → Your UEM Server):
const ACTIVE_ENV = ENVIRONMENT.PRODUCTION;
//
// ─────────────────────────────────────────────────

// ─── DERIVED CONFIGURATION ──────────────────────
const BASE_URL = ACTIVE_ENV.baseUrl;

// Debug logging (remove in production if needed)
console.log(`🔗 Backend Environment: ${ACTIVE_ENV.name.toUpperCase()} (${ACTIVE_ENV.description})`);
console.log(`📍 Base URL: ${BASE_URL}`);
