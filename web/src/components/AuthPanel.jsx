import React, { useCallback, useEffect, useState } from "react";
import { supabase } from "../supabase.js";
import * as api from "../api.js";
import { HUMAN_ID } from "../format.js";

function GitHubIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M8 0a8 8 0 0 0-2.53 15.59c.4.07.55-.17.55-.38l-.01-1.49c-2.22.48-2.69-1.07-2.69-1.07-.36-.93-.88-1.17-.88-1.17-.72-.49.05-.48.05-.48.8.06 1.22.82 1.22.82.71 1.21 1.86.87 2.31.66.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.22 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48l-.01 2.2c0 .21.14.46.55.38A8 8 0 0 0 8 0Z" />
    </svg>
  );
}

function GoogleIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 18 18" aria-hidden="true">
      <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.9c1.7-1.57 2.7-3.87 2.7-6.62Z" />
      <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.9-2.26c-.81.54-1.85.86-3.06.86-2.35 0-4.34-1.59-5.05-3.72H.96v2.33A9 9 0 0 0 9 18Z" />
      <path fill="#FBBC05" d="M3.95 10.7A5.4 5.4 0 0 1 3.67 9c0-.59.1-1.16.28-1.7V4.97H.96A9 9 0 0 0 0 9c0 1.45.35 2.83.96 4.03l2.99-2.33Z" />
      <path fill="#EA4335" d="M9 3.58c1.32 0 2.51.46 3.44 1.35l2.58-2.58A9 9 0 0 0 .96 4.97l2.99 2.33C4.66 5.17 6.65 3.58 9 3.58Z" />
    </svg>
  );
}

function UserGlyph() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <circle cx="8" cy="5" r="3" />
      <path d="M2 14c0-3.3 2.7-5.5 6-5.5s6 2.2 6 5.5v.5H2v-.5Z" />
    </svg>
  );
}

function formatWhen(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const today = new Date();
  const sameYear = d.getFullYear() === today.getFullYear();
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: sameYear ? undefined : "numeric" });
}

// A single "user button" that expands into a panel: sign-in (with provider
// logos) when signed out, or account info + recent game history when signed
// in. Renders nothing if Supabase isn't configured (see ../supabase.js) -
// sign-in is entirely optional and games still play/record anonymously
// without it.
export default function AuthPanel({ session }) {
  const [open, setOpen] = useState(false);
  const [history, setHistory] = useState(null); // null = not loaded yet
  const [historyError, setHistoryError] = useState(null);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const userId = session?.user?.id ?? null;
  useEffect(() => {
    setHistory(null);
    setHistoryError(null);
  }, [userId]);

  const loadHistory = useCallback(() => {
    if (!userId || history != null || loadingHistory) return;
    setLoadingHistory(true);
    api
      .myGameHistory()
      .then((data) => setHistory(data.games ?? []))
      .catch((err) => setHistoryError(String(err.message ?? err)))
      .finally(() => setLoadingHistory(false));
  }, [userId, history, loadingHistory]);

  if (!supabase) return null;

  const toggle = () => {
    setOpen((cur) => {
      const next = !cur;
      if (next) loadHistory();
      return next;
    });
  };

  const user = session?.user ?? null;
  const label = user ? user.user_metadata?.user_name ?? user.user_metadata?.full_name ?? user.email ?? user.id : null;
  const avatarUrl = user?.user_metadata?.avatar_url ?? user?.user_metadata?.picture ?? null;

  return (
    <div className="panel user-panel">
      <button className="user-button" onClick={toggle} aria-expanded={open}>
        <span className="user-avatar">
          {avatarUrl ? <img src={avatarUrl} alt="" /> : user ? label[0]?.toUpperCase() : <UserGlyph />}
        </span>
        <span className="user-button-label">{user ? label : "Sign in"}</span>
        <span className={`user-caret${open ? " open" : ""}`}>▾</span>
      </button>

      {open && !user && (
        <div className="user-dropdown">
          <p className="auth-hint">Sign in to save your game history.</p>
          <div className="auth-buttons">
            <button className="btn-oauth" onClick={() => supabase.auth.signInWithOAuth({ provider: "github" })}>
              <GitHubIcon /> Continue with GitHub
            </button>
            <button className="btn-oauth" onClick={() => supabase.auth.signInWithOAuth({ provider: "google" })}>
              <GoogleIcon /> Continue with Google
            </button>
          </div>
        </div>
      )}

      {open && user && (
        <div className="user-dropdown">
          <div className="user-info">
            <span className="user-info-name">{label}</span>
            {user.email && user.email !== label && <span className="user-info-email">{user.email}</span>}
          </div>

          <div className="history-section">
            <h3>Game history</h3>
            {loadingHistory && <p className="muted">Loading…</p>}
            {historyError && <p className="muted">Couldn't load history.</p>}
            {!loadingHistory && !historyError && history?.length === 0 && <p className="muted">No finished games yet.</p>}
            {!loadingHistory && !historyError && history?.length > 0 && (
              <ul className="history-list">
                {history.map((g) => (
                  <li key={g.id} className="history-row">
                    <span className={`history-result ${g.winner === HUMAN_ID ? "win" : "lose"}`}>
                      {g.winner === HUMAN_ID ? "Win" : "Loss"}
                    </span>
                    <span className="history-meta">
                      {formatWhen(g.finished_at)} · {g.turn_count} turns
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <button className="btn-secondary" onClick={() => supabase.auth.signOut()}>
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
