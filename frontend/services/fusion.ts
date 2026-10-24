import { apiRequest } from "./api";

export type FusionKind = "early" | "intermediate" | "late";

export type FusionModelAvailability = {
  kind: FusionKind;
  implemented: boolean;
  trained: boolean;
  available: boolean;
  loaded: boolean;
  requires_text: boolean;
  status: string;
  reason?: string | null;
};

export type FusionStatus = {
  supported_kinds: FusionKind[];
  fusion_models: FusionModelAvailability[];
  loaded_kinds: FusionKind[];
  experimental_weights_loaded: boolean;
  production_image_only_available: boolean;
};

export type FusionPrediction = { class_id: number; label: string; confidence: number };

export type FusionAnalysis = {
  fusion_kind: FusionKind;
  modalities_used: ("image" | "text")[];
  multimodal: boolean;
  image_only_fallback: boolean;
  text_provided: boolean;
  predicted_class: FusionPrediction;
  confidence: number;
  top_k: FusionPrediction[];
  inference_ms: number;
  model: {
    fusion_kind: FusionKind;
    requires_text: boolean;
    supports_image_only_fallback: boolean;
    class_count: number;
  };
  notice: string;
};

export function getFusionStatus(): Promise<FusionStatus> {
  return apiRequest<FusionStatus>("/api/fusion/status");
}

export function analyzeFusionImage(
  kind: FusionKind,
  file: File,
  pairedText?: string,
  topK = 5,
): Promise<FusionAnalysis> {
  const form = new FormData();
  form.append("file", file);
  form.append("kind", kind);
  if (pairedText && pairedText.trim()) form.append("paired_text", pairedText);
  form.append("top_k", String(topK));
  return apiRequest<FusionAnalysis>("/api/fusion/analyze", { method: "POST", body: form });
}
