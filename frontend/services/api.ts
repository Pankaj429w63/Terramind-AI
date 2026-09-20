export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8001";

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body && typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return body as T;
}
