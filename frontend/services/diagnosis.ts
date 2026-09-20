import { apiRequest } from "./api";

export type Prediction = { class_id: number; label: string; confidence: number; score?: number; confidence_pct?: number };
export type ModelInfo = { backbone: string; num_classes: number; img_size: number; mean: number[]; std: number[]; model_path: string; test_acc?: number; best_val_acc?: number };
export type Diagnosis = { id: string; source_filename?: string; predicted_class: Prediction; top5: Prediction[]; model_info: ModelInfo; inference_ms: number; low_confidence?: boolean; low_confidence_threshold?: number };
export type BatchResult = { id: string; filename: string; error: string | null; prediction: Diagnosis | null };
export type BatchResponse = { batch_id: string; count: number; errors: number; results: BatchResult[] };

function finiteConfidence(prediction: Prediction): number {
  const candidates = [prediction.confidence, prediction.score, prediction.confidence_pct];
  for (const candidate of candidates) {
    const value = Number(candidate);
    if (Number.isFinite(value)) {
      return value > 1 ? value / 100 : Math.max(0, value);
    }
  }
  return 0;
}

function normalizePrediction(prediction: Prediction): Prediction {
  return { ...prediction, confidence: finiteConfidence(prediction) };
}

function normalizeDiagnosis(value: Diagnosis): Diagnosis {
  return {
    ...value,
    predicted_class: normalizePrediction(value.predicted_class),
    top5: (value.top5 || []).map(normalizePrediction),
  };
}

export function predictImage(file: File, topK = 5): Promise<Diagnosis> {
  const form = new FormData();
  form.append("file", file);
  form.append("top_k", String(topK));
  return apiRequest<Diagnosis>("/api/diagnosis/predict", { method: "POST", body: form }).then(normalizeDiagnosis);
}

export function predictBatch(files: File[], topK = 5): Promise<BatchResponse> {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  form.append("top_k", String(topK));
  return apiRequest<BatchResponse>("/api/diagnosis/batch", { method: "POST", body: form }).then((response) => ({
    ...response,
    results: response.results.map((item) => ({ ...item, prediction: item.prediction ? normalizeDiagnosis(item.prediction) : null })),
  }));
}
