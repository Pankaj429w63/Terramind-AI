import { apiRequest } from "./api";
import { supabase, requireSupabase } from "./supabase";
import { LocalHistoryItem } from "./history";

export type ReportRecord = { id: string; diagnosis_id: string | null; bucket: string; storage_path: string | null; report_type: string; created_at: string };
export type ReportResponse = { count: number; items: ReportRecord[] };

export async function getReports(limit = 50): Promise<ReportRecord[]> {
  const { data, error } = await requireSupabase()
    .from("reports")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(limit);
  if (error) throw error;
  return (data ?? []) as ReportRecord[];
}

export async function getSupabaseReports(userId: string, limit = 50): Promise<ReportResponse> {
  return apiRequest<ReportResponse>(`/api/reports/database?user_id=${encodeURIComponent(userId)}&limit=${limit}`);
}

export async function listReports(userId?: string, limit = 50): Promise<{ items: (ReportRecord | LocalHistoryItem)[]; source: "supabase" | "local"; count: number }> {
  if (userId && supabase) {
    try {
      const res = await getSupabaseReports(userId, limit);
      return { items: res.items, source: "supabase", count: res.count };
    } catch {
      // fall through to local reports (diagnosis json files)
    }
  }
  const res = await apiRequest<{ count: number; items: LocalHistoryItem[] }>(`/api/diagnosis/history?limit=${limit}`);
  return { items: res.items, source: "local", count: res.count };
}

export async function getReportDownloadUrl(report: ReportRecord, expiresIn = 3600): Promise<string> {
  if (!report.storage_path) throw new Error("This report does not have a storage object.");
  const { data, error } = await requireSupabase().storage.from(report.bucket).createSignedUrl(report.storage_path, expiresIn);
  if (error) throw error;
  return data.signedUrl;
}

export function formatTimestamp(iso?: string): { date: string; time: string } {
  if (!iso) return { date: "—", time: "—" };
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return { date: iso.slice(0, 10), time: iso.slice(11, 16) };
    const date = d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
    const time = d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
    return { date, time };
  } catch {
    return { date: iso.slice(0, 10), time: iso.slice(11, 16) };
  }
}

export function formatBytes(bytes?: number): string {
  if (!bytes || bytes <= 0) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let n = bytes;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(n >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}
