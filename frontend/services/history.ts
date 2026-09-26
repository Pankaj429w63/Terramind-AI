import { apiRequest } from "./api";
import { supabase } from "./supabase";
import { Diagnosis } from "./diagnosis";

export type LocalHistoryItem = Diagnosis & {
  timestamp?: string;
  prediction?: { label: string; score: number; confidence_pct?: number };
  low_confidence?: boolean;
  inference_ms?: number;
  source_filename?: string;
};

export type HistoryResponse = { count: number; items: LocalHistoryItem[]; source?: "supabase" | "local" };

export async function getDiagnosisHistory(limit = 50): Promise<Diagnosis[]> {
  if (!supabase) throw new Error("Sign in to view your diagnosis history.");
  const { data, error } = await supabase
    .from("diagnoses")
    .select("*, predictions(*)")
    .order("created_at", { ascending: false })
    .limit(limit);
  if (error) throw error;
  return (data ?? []) as Diagnosis[];
}

export async function getLocalDiagnosisHistory(limit = 50): Promise<HistoryResponse> {
  return apiRequest<HistoryResponse>(`/api/diagnosis/history?limit=${limit}`);
}

export async function getSupabaseDiagnosisHistory(_userId: string, limit = 50): Promise<HistoryResponse> {
  return apiRequest<HistoryResponse>(`/api/diagnosis/history/database?limit=${limit}`);
}

export async function listDiagnosisHistory(userId?: string, limit = 50): Promise<{ items: LocalHistoryItem[]; source: "supabase" | "local"; count: number }> {
  const session = await supabase?.auth.getSession();
  const authenticatedId = session?.data.session?.user.id;
  if (!authenticatedId) throw new Error("Sign in to view your diagnosis history.");
  try {
    const res = await getSupabaseDiagnosisHistory(userId || authenticatedId, limit);
    return { items: res.items, source: res.source === "local" ? "local" : "supabase", count: res.count };
  } catch (error) {
    throw error;
  }
}

export async function getDiagnosisById(diagnosisId: string): Promise<LocalHistoryItem> {
  return apiRequest<LocalHistoryItem>(`/api/diagnosis/${encodeURIComponent(diagnosisId)}`);
}
