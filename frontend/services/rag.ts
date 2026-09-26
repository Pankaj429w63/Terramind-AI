import { apiRequest } from "./api";

export type RetrievedSource = {
  title?: string;
  source?: string;
  category?: string;
  text?: string;
  excerpt?: string;
  score?: number;
  retrieval_score?: number;
  metadata?: Record<string, string>;
};

export type RagRetrieval = {
  retrieved: number;
  sources: RetrievedSource[];
  error?: string | null;
  query: string;
  category?: string | null;
  grounded: boolean;
};

export function retrieveKnowledge(query: string, category?: string, limit = 5): Promise<RagRetrieval> {
  return apiRequest<RagRetrieval>("/api/rag/retrieve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, category: category || null, limit }),
  });
}
