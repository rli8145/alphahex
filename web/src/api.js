// Thin client for the Catan FastAPI backend. In dev, calls go through the Vite
// proxy at /api -> http://127.0.0.1:8000.
const BASE = import.meta.env.VITE_API_BASE ?? "/api";

// Set by App.jsx from the Supabase session (see ../supabase.js). Attached to
// every request when present so the backend can tie a finished game to the
// signed-in user; null (the default, and always the case if Supabase isn't
// configured) means requests go out exactly as before - fully anonymous.
let authToken = null;
export function setAuthToken(token) {
  authToken = token ?? null;
}

function authHeaders() {
  return authToken ? { Authorization: `Bearer ${authToken}` } : {};
}

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${path} failed (${res.status}): ${text}`);
  }
  return res.json();
}

async function get(path) {
  const res = await fetch(`${BASE}${path}`, { headers: authHeaders() });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${path} failed (${res.status}): ${text}`);
  }
  return res.json();
}

export const newGame = (seed = 0) => post("/games/new", { seed });
export const applyAction = (state, action) => post("/games/action", { state, action });
export const botStep = (state) => post("/games/bot-step", { state });

// Recent completed games for the signed-in user (requires an auth token, see
// setAuthToken above); returns { games: [] } if signed out or unconfigured.
export const myGameHistory = (limit = 20) => get(`/games/history/me?limit=${limit}`);
