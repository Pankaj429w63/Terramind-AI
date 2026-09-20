import { API_BASE_URL, apiRequest } from "./api";

export type ModelSummary = {
  model_name?: string;
  backbone?: string;
  num_classes?: number;
  img_size?: number;
  train_accuracy?: number;
  val_accuracy?: number;
  test_accuracy?: number;
  train_loss?: number;
  val_loss?: number;
  test_loss?: number;
  train_macro_f1?: number;
  val_macro_f1?: number;
  test_macro_f1?: number;
  train_weighted_f1?: number;
  val_weighted_f1?: number;
  test_weighted_f1?: number;
  epochs?: number;
  best_epoch?: number;
  train_samples?: number;
  val_samples?: number;
  test_samples?: number;
  dataset?: string;
  timestamp?: string;
  total_trainable_params?: number;
  [key: string]: unknown;
};

export type ClassMetric = {
  class_id?: number;
  class: string;
  label?: string;
  precision?: number;
  recall?: number;
  f1?: number;
  f1_score?: number;
  support?: number;
  samples?: number;
};

export type FullReport = {
  summary?: Partial<ModelSummary>;
  per_class?: ClassMetric[];
  classification_report?: { [className: string]: Omit<ClassMetric, "class"> & { support?: number } };
  confusion_matrix?: number[][];
  labels?: string[];
  [key: string]: unknown;
};

export type FinalConfig = {
  model_name?: string;
  backbone?: string;
  num_classes?: number;
  img_size?: number;
  batch_size?: number;
  epochs?: number;
  learning_rate?: number;
  optimizer?: string;
  scheduler?: string;
  loss_function?: string;
  weight_decay?: number;
  dropout?: number;
  use_augmentation?: boolean;
  dataset?: string;
  seed?: number;
  device?: string;
  [key: string]: unknown;
};

export async function getAnalyticsSummary(): Promise<ModelSummary> {
  const raw = await apiRequest<any>("/api/analytics/summary");
  const m = raw?.metrics ?? raw ?? {};
  const conv = (v: unknown, isPct: boolean) => {
    if (typeof v !== "number" || Number.isNaN(v)) return undefined;
    return isPct ? v / 100 : v;
  };
  return {
    model_name: raw?.config?.model_name ?? m?.model_name,
    backbone: raw?.config?.backbone ?? m?.backbone,
    num_classes: m?.num_classes ?? raw?.metrics?.num_classes,
    img_size: raw?.config?.img_size ?? m?.img_size,
    train_accuracy: conv(m?.train_accuracy, false) ?? conv(m?.train_accuracy_pct, true),
    val_accuracy: conv(m?.validation_accuracy ?? m?.val_accuracy, false) ?? conv(m?.validation_accuracy_pct ?? m?.val_accuracy_pct, true),
    test_accuracy: conv(m?.test_accuracy, false) ?? conv(m?.test_accuracy_pct, true),
    train_loss: m?.train_loss,
    val_loss: m?.best_val_loss ?? m?.val_loss,
    test_loss: m?.test_loss,
    train_macro_f1: conv(m?.train_macro_f1, false) ?? conv(m?.train_macro_f1_pct, true),
    val_macro_f1: conv(m?.val_macro_f1, false) ?? conv(m?.val_macro_f1_pct, true),
    test_macro_f1: conv(m?.test_macro_f1, false) ?? conv(m?.test_macro_f1_pct, true),
    train_weighted_f1: conv(m?.train_weighted_f1, false) ?? conv(m?.train_weighted_f1_pct, true),
    val_weighted_f1: conv(m?.val_weighted_f1, false) ?? conv(m?.val_weighted_f1_pct, true),
    test_weighted_f1: conv(m?.test_weighted_f1, false) ?? conv(m?.test_weighted_f1_pct, true),
    epochs: m?.epochs_ran ?? raw?.config?.epochs,
    best_epoch: m?.best_epoch,
    train_samples: m?.train_samples,
    val_samples: m?.val_samples,
    test_samples: m?.test_samples,
    dataset: raw?.config?.dataset ?? m?.dataset,
    timestamp: raw?.timestamp,
    total_trainable_params: raw?.config?.total_trainable_params ?? m?.total_trainable_params,
    ...raw,
  };
}

export async function getAnalyticsFullReport(): Promise<FullReport> {
  return apiRequest<FullReport>("/api/analytics/full_report");
}

export async function getAnalyticsFinalConfig(): Promise<FinalConfig> {
  return apiRequest<FinalConfig>("/api/analytics/final_config");
}

export function getGraphUrl(name: "training_curves" | "confusion_matrix"): string {
  return `${API_BASE_URL}/api/analytics/graph/${name}`;
}

export function pct(value?: number, digits = 2): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  const v = value <= 1 ? value * 100 : value;
  return `${v.toFixed(digits)}%`;
}

export function formatScalar(value?: number, digits = 4): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}
