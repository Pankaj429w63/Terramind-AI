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

export async function getModelLabels(): Promise<string[]> {
  const result = await apiRequest<{ num_classes: number; labels: string[] }>("/api/model/labels");
  return result.labels;
}
