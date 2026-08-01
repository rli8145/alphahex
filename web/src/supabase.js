// Optional: sign-in via Supabase Auth so completed games can be tied to an
// account (see packages/catan_api/auth.py on the backend). Without
// VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY set, `supabase` is null and the
// app plays on exactly as it did before this file existed - fully anonymous,
// nothing recorded against a user.
import { createClient } from "@supabase/supabase-js";

const url = import.meta.env.VITE_SUPABASE_URL;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const supabase = url && anonKey ? createClient(url, anonKey) : null;
