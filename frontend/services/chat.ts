import { apiRequest } from "./api";

export type ChatMessage = { role: "user" | "assistant"; content: string };
export type ChatSource = { title?: string; source?: string; category?: string; text?: string; excerpt?: string; retrieval_score?: number; citation_id?: string };
export type ChatWorkflowEvent = { agent_name: string; status: string; output?: Record<string, unknown>; error?: string };
export type ChatResponse = {
  response: string | null;
  workflow_id: string;
  workflow: ChatWorkflowEvent[];
  sources: ChatSource[];
  grounded: boolean;
  specialist_agent: string;
  generation_mode: "llm" | "evidence_template" | "no_evidence" | string;
  generation_status: string;
  diagnosis_context?: Record<string, unknown> | null;
};

export function askTerraMind(messages: ChatMessage[], diagnosisId?: string): Promise<ChatResponse> {
  return apiRequest<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      diagnosis_id: diagnosisId || null,
      enable_agents: true,
      enable_rag: true,
    }),
  });
}
