import { apiRequest } from "./api";
import { supabase, requireSupabase } from "./supabase";
import { Diagnosis } from "./diagnosis";

export type LocalHistoryItem = Diagnosis & {
  timestamp?: string;
  prediction?: { label: string; score: number; confidence_pct?: number };
  low_confidence?: boolean;
  inference_ms?: number;
  source_filename?: string;
};

export type HistoryResponse = { count: number; items: LocalHistoryItem[] };

export async function getDiagnosisHistory(limit = 50): Promise<Diagnosis[]> {
  const { data, error } = await requireSupabase()
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

export async function getSupabaseDiagnosisHistory(userId: string, limit = 50): Promise<HistoryResponse> {
  return apiRequest<HistoryResponse>(`/api/diagnosis/history/database?user_id=${encodeURIComponent(userId)}&limit=${limit}`);
}

export async function listDiagnosisHistory(userId?: string, limit = 50): Promise<{ items: LocalHistoryItem[]; source: "supabase" | "local"; count: number }> {
  if (userId && supabase) {
    try {
      const res = await getSupabaseDiagnosisHistory(userId, limit);
      return { items: res.items, source: "supabase", count: res.count };
    } catch {
      // fall through to local history
    }
  }
  const res = await getLocalDiagnosisHistory(limit);
  return { items: res.items, source: "local", count: res.count };
}

export async function getDiagnosisById(diagnosisId: string): Promise<LocalHistoryItem> {
  return apiRequest<LocalHistoryItem>(`/api/diagnosis/${encodeURIComponent(diagnosisId)}`);
}
