import { apiRequest } from "./api";
import { LocalHistoryItem } from "./history";

export type AgentEvent = {
  agent_name: string;
  status: "running" | "completed" | "failed" | string;
  started_at?: string;
  completed_at?: string;
  duration_ms?: number;
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
  error?: string;
};

export type AgentDiagnosis = LocalHistoryItem & {
  agent_workflow_id: string;
  agent_events: AgentEvent[];
  agent_results: Record<string, Record<string, unknown>>;
};

export async function runAgentDiagnosis(file: File, query = ""): Promise<AgentDiagnosis> {
  const form = new FormData();
  form.append("file", file);
  form.append("query", query);
  return apiRequest<AgentDiagnosis>("/api/agents/diagnose", { method: "POST", body: form });
}

export function listLocalHistory(limit = 50): Promise<{ count: number; items: LocalHistoryItem[] }> {
  return apiRequest<{ count: number; items: LocalHistoryItem[] }>(`/api/diagnosis/history?limit=${limit}`);
}
