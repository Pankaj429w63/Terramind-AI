import { AuthResponse, Session, User } from "@supabase/supabase-js";
import { requireSupabase } from "./supabase";

export async function signIn(email: string, password: string): Promise<AuthResponse> {
  return requireSupabase().auth.signInWithPassword({ email, password });
}

export async function signUp(email: string, password: string, displayName?: string): Promise<AuthResponse> {
  return requireSupabase().auth.signUp({ email, password, options: { data: { full_name: displayName } } });
}

export async function signOut(): Promise<void> {
  const { error } = await requireSupabase().auth.signOut();
  if (error) throw error;
}

export async function getSession(): Promise<Session | null> {
  const { data, error } = await requireSupabase().auth.getSession();
  if (error) throw error;
  return data.session;
}

export function onAuthStateChange(callback: (session: Session | null, user: User | null) => void): () => void {
  const client = requireSupabase();
  const { data } = client.auth.onAuthStateChange((_event, session) => callback(session, session?.user ?? null));
  return () => data.subscription.unsubscribe();
}
