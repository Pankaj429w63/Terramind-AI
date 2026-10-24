// Production always uses same-origin rewrites. A direct API URL is only honored
// in development so a stale localhost variable can never leak into a build.
export const API_BASE_URL = process.env.NODE_ENV === "development"
  ? (process.env.NEXT_PUBLIC_API_URL || "")
  : "";

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  let token: string | undefined;
  try {
    const { supabase } = await import("./supabase");
    const session = await supabase?.auth.getSession();
    token = session?.data.session?.access_token;
  } catch { /* Public endpoints and local mode do not require a session. */ }
  const headers = new Headers(init?.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const rawDetail = body?.detail;
    const detail = typeof rawDetail === "string"
      ? rawDetail
      : Array.isArray(rawDetail)
        ? rawDetail.map((item) => `${Array.isArray(item.loc) ? item.loc.join(".") : "request"}: ${item.msg || "invalid value"}`).join("; ")
        : typeof body?.error === "string" ? body.error : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return body as T;
}
