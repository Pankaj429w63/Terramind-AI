import { apiRequest } from "./api";
import { ModelInfo } from "./diagnosis";

export type HealthResponse = {
  status: "ok" | "degraded" | string;
  service: string;
  version: string;
  timestamp: string;
  inference_available: boolean;
  model: ModelInfo | null;
};

export type ServingModelInfo = ModelInfo & {
  test_macro_f1?: number | null;
  best_val_macro_f1?: number | null;
  best_epoch?: number | null;
  preprocessing?: string;
  labels?: string[];
};

export type SupabaseStatus = { configured: boolean; missing: string[] };
export type LlmStatus = { configured: boolean; provider: string | null; model: string | null; generation_fallback: string; api_key_configured: boolean };

export type ModelAvailability = {
  name: string; kind: string; implemented?: boolean; trained: boolean; loaded?: boolean;
  available?: boolean; available_for_inference?: boolean; requires_text?: boolean;
  status: string; reason?: string | null; promoted?: boolean;
  test_evaluation_completed?: boolean; evaluation_subset?: string | null;
  validation_macro_f1?: number | null; metrics?: Record<string, number | null> | null;
};

export type ModelRegistryStatus = {
  production: { name: string; status: string; available_for_inference: boolean; metrics: ServingModelInfo | null };
  experimental_candidates: ModelAvailability[];
  fusion_models: ModelAvailability[];
  text_only_baseline?: ModelAvailability | null;
  experimental_weights_loaded: boolean;
};

export function getHealth(): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/health");
}

export async function getServingModelInfo(): Promise<ServingModelInfo> {
  const result = await apiRequest<{ info: ServingModelInfo }>("/api/model/info");
  return result.info;
}

export function getSupabaseStatus(): Promise<SupabaseStatus> {
  return apiRequest<SupabaseStatus>("/api/supabase/status");
}

export function getModelRegistryStatus(): Promise<ModelRegistryStatus> {
  return apiRequest<ModelRegistryStatus>("/api/models/status");
}

export function getLlmStatus(): Promise<LlmStatus> {
  return apiRequest<LlmStatus>("/api/llm/status");
}

export async function getModelLabels(): Promise<string[]> {
  const result = await apiRequest<{ num_classes: number; labels: string[] }>("/api/model/labels");
  return result.labels;
}
