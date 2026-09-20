import { apiRequest } from "./api";
import { formatTimestamp } from "./reports";
import { LocalHistoryItem } from "./history";

export type RagStats = {
  indexed: boolean;
  backend?: string;
  collection?: string;
  total_points: number;
  sources_indexed: number;
  tfidf_size_bytes: number;
  by_category: Record<string, number>;
};

export type AgentStatus = {
  supervisor: {
    state: "active" | "standby" | "inactive";
    ready: boolean;
    chat_ready: boolean;
    inference_ready: boolean;
  };
  progress: {
    completed_steps: number;
    total_steps: number;
    percent: number;
  };
  agent_runs: { recorded_supabase: number };
  diagnoses: { local_reports: number; supabase_diagnoses: number; total: number };
  reports: { supabase_reports: number };
  backend: { supabase_enabled: boolean; rag_backend?: string };
};

export type DashboardAggregate = {
  total: number;
  healthy: number;
  diseased: number;
  low_confidence: number;
  avg_confidence: number;
  reports_count: number;
  recent: LocalHistoryItem[];
  trend: { days: string[]; healthy: number[]; diseased: number[] };
};

export async function getRagStats(): Promise<RagStats> {
  return apiRequest<RagStats>("/api/rag/stats");
}

export async function getAgentStatus(): Promise<AgentStatus> {
  return apiRequest<AgentStatus>("/api/agents/status");
}

const CAT_LABELS = ["diseases", "treatment", "fertilizer", "plant care", "diseases", "plantcare", "plant_care"] as const;
const CAT_MAP: Record<string, string> = {
  disease: "diseases",
  diseases: "diseases",
  condition: "diseases",
  conditions: "diseases",
  treatment: "treatments",
  treatments: "treatments",
  fertilizer: "fertilizers",
  fertilizers: "fertilizers",
  nutrient: "fertilizers",
  nutrients: "fertilizers",
  care: "plant care",
  "plant care": "plant care",
  plantcare: "plant care",
  plant_care: "plant care",
  farming: "plant care",
  crop: "plant care",
  crops: "plant care",
};

export function mapCategoryCounts(
  byCategory: Record<string, number> | undefined
): { diseases: number; treatments: number; fertilizers: number; plantCare: number; other: number } {
  const out = { diseases: 0, treatments: 0, fertilizers: 0, plantCare: 0, other: 0 };
  if (!byCategory) return out;
  for (const [raw, count] of Object.entries(byCategory)) {
    const key = String(raw || "").trim().toLowerCase();
    const mapped = CAT_MAP[key];
    if (mapped === "diseases") out.diseases += count;
    else if (mapped === "treatments") out.treatments += count;
    else if (mapped === "fertilizers") out.fertilizers += count;
    else if (mapped === "plant care") out.plantCare += count;
    else if (CAT_LABELS.some((l) => key.includes(l.replace(" ", "")))) {
      // fuzzy includes
      if (key.includes("disease") || key.includes("condition")) out.diseases += count;
      else if (key.includes("treatment")) out.treatments += count;
      else if (key.includes("fertilizer") || key.includes("nutrient")) out.fertilizers += count;
      else if (key.includes("care") || key.includes("plant")) out.plantCare += count;
      else out.other += count;
    } else {
      out.other += count;
    }
  }
  return out;
}

export function aggregateHistory(items: LocalHistoryItem[]): DashboardAggregate {
  let healthy = 0;
  let diseased = 0;
  let low = 0;
  let confSum = 0;
  for (const it of items) {
    const top =
      it.predicted_class ??
      (it.prediction
        ? {
            class_id: -1,
            label: it.prediction.label,
            confidence: it.prediction.score ?? it.prediction.confidence_pct ?? 0,
          }
        : (Array.isArray(it.top5) && it.top5.length > 0 ? it.top5[0] : undefined));
    if (!top) continue;
    const conf = typeof top.confidence === "number" ? top.confidence : 0;
    confSum += conf;
    if (it.low_confidence || conf < 0.35) low++;
    const lbl = (top.label || "").toLowerCase();
    if (lbl.includes("healthy") || lbl.includes("normal") || lbl.includes("good")) {
      healthy++;
    } else {
      diseased++;
    }
  }
  // 7-day trend aggregation
  const buckets = new Map<
    string,
    { date: string; dayLabel: string; healthy: number; diseased: number }
  >();
  const now = new Date();
  for (let i = 6; i >= 0; i--) {
    const d = new Date(now);
    d.setDate(now.getDate() - i);
    const isoKey = d.toISOString().slice(0, 10);
    const dayLabel = d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
    buckets.set(isoKey, { date: isoKey, dayLabel, healthy: 0, diseased: 0 });
  }
  for (const it of items) {
    const ts = it.timestamp;
    if (!ts) continue;
    try {
      const iso = new Date(ts).toISOString().slice(0, 10);
      const bucket = buckets.get(iso);
      if (!bucket) continue;
      const top =
        it.predicted_class ??
        (it.prediction
          ? {
              class_id: -1,
              label: it.prediction.label,
              confidence: it.prediction.score ?? it.prediction.confidence_pct ?? 0,
            }
          : (Array.isArray(it.top5) && it.top5.length > 0 ? it.top5[0] : undefined));
      const lbl = ((top?.label) || "").toLowerCase();
      if (lbl.includes("healthy") || lbl.includes("normal") || lbl.includes("good")) {
        bucket.healthy++;
      } else {
        bucket.diseased++;
      }
    } catch {
      // skip
    }
  }
  const arr = Array.from(buckets.values());
  const days = arr.map((b) => b.dayLabel);
  const healthyArr = arr.map((b) => b.healthy);
  const diseasedArr = arr.map((b) => b.diseased);

  // Recent top 5 for display: use the order we got (already most recent first)
  const recent = items.slice(0, 5);
  // Augment each recent item with display helpers
  for (const r of recent) {
    if (r.timestamp) void formatTimestamp(r.timestamp);
  }

  return {
    total: items.length,
    healthy,
    diseased,
    low_confidence: low,
    avg_confidence: items.length ? confSum / items.length : 0,
    reports_count: items.length,
    recent,
    trend: { days, healthy: healthyArr, diseased: diseasedArr },
  };
}
