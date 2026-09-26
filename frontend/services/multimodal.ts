import { apiRequest } from "./api";
import { Diagnosis } from "./diagnosis";
import { RetrievedSource } from "./rag";

export type MultimodalStatus = {
  module: string;
  status: "ready" | "unavailable" | string;
  trained: boolean;
  checkpoint_available: boolean;
  checkpoint_valid: boolean;
  training_completed: boolean;
  architecture: Record<string, unknown>;
  data: {
    class_mapping_available: boolean;
    train_split_entries: number;
    validation_split_entries: number;
    text_source: string;
    paired_natural_language_descriptions: boolean;
  };
  production_diagnosis: string;
  message: string | null;
};

export type MultimodalAnalysis = {
  diagnosis: Diagnosis;
  metadata_text: string;
  fused_representation: { dimension: number; vector: number[] };
  guidance: { retrieved?: number; sources?: RetrievedSource[]; error?: string | null };
  production_classifier: string;
  multimodal_training_state: string;
};

export function getMultimodalStatus(): Promise<MultimodalStatus> {
  return apiRequest<MultimodalStatus>("/api/multimodal/status");
}

export function analyzeMultimodalImage(file: File): Promise<MultimodalAnalysis> {
  const form = new FormData();
  form.append("file", file);
  return apiRequest<MultimodalAnalysis>("/api/multimodal/analyze", { method: "POST", body: form });
}
