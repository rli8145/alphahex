import React from "react";
import { supabase } from "../supabase.js";

// Renders nothing if Supabase isn't configured (see ../supabase.js) - sign-in
// is entirely optional and games still play/record anonymously without it.
export default function AuthPanel({ session }) {
  if (!supabase) return null;

  if (!session) {
    return (
      <div className="panel auth-panel">
        <h2>Account</h2>
        <p className="auth-hint">Sign in to save your game history.</p>
        <div className="auth-buttons">
          <button className="btn-secondary" onClick={() => supabase.auth.signInWithOAuth({ provider: "github" })}>
            Sign in with GitHub
          </button>
          <button className="btn-secondary" onClick={() => supabase.auth.signInWithOAuth({ provider: "google" })}>
            Sign in with Google
          </button>
        </div>
      </div>
    );
  }

  const user = session.user;
  const label = user.user_metadata?.user_name ?? user.user_metadata?.full_name ?? user.email ?? user.id;
  return (
    <div className="panel auth-panel">
      <h2>Account</h2>
      <p className="auth-hint">Signed in as {label}</p>
      <button className="btn-secondary" onClick={() => supabase.auth.signOut()}>
        Sign out
      </button>
    </div>
  );
}
