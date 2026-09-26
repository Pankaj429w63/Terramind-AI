"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import { Diagnosis, Prediction, ExpertReview, submitExpertReview } from "../services/diagnosis";
import { useDiagnosis } from "../hooks/useDiagnosis";
import { askTerraMind, ChatMessage, ChatResponse } from "../services/chat";
import { getDiagnosisById, listDiagnosisHistory, LocalHistoryItem } from "../services/history";
import { ReportRecord, formatBytes, formatTimestamp, listReports, getReportDownloadUrl } from "../services/reports";
import { getGraphUrl, pct } from "../services/analytics";
import {
  AgentStatus,
  DashboardAggregate,
  RagStats,
  aggregateHistory,
  getAgentStatus,
  getRagStats,
  mapCategoryCounts,
} from "../services/dashboard";
import { getHealth, getServingModelInfo, getSupabaseStatus, HealthResponse, ServingModelInfo, SupabaseStatus } from "../services/system";
import { supabase } from "../services/supabase";
import { signOut } from "../services/auth";
import {
  AgentToolsPage,
  MultimodalPage,
  ProfilePage,
  RagKnowledgeBasePage,
  RecommendationPage,
  ServedModelInsightsPage,
  ServedModelPerformancePage,
  SettingsPage,
  SystemStatusPage,
} from "./live-pages";

const nav = [
  { group: "MAIN", items: [["Dashboard", "▦"], ["Plant Diagnosis", "⌁"]] },
  { group: "DIAGNOSIS", items: [["Upload Image", "↥"], ["Batch Detection", "▤"], ["Live Detection", "◉"]] },
  { group: "AI ASSISTANT", items: [["AI Chat Assistant", "✧"]] },
  { group: "RECOMMENDATIONS", items: [["Treatment Advisor", "♧"], ["Fertilizer Guide", "✚"], ["Care Recommendations", "♡"]] },
  { group: "KNOWLEDGE & AI", items: [["RAG Knowledge Base", "▱"], ["Agent Orchestrator", "⌘"], ["Agent Workflow", "⌘"], ["Multimodal AI", "?"]] },
  { group: "ANALYTICS", items: [["Model Performance", "◒"], ["Model Insights", "⌗"], ["System Status", "◇"]] },
  { group: "HISTORY & REPORTS", items: [["History", "◴"], ["Reports", "▤"]] },
  { group: "SETTINGS", items: [["Settings", "⚙"], ["Profile", "♙"]] },
];

function Icon({ children }: { children: React.ReactNode }) { return <span className="icon">{children}</span>; }
function Button({ children, primary = false, onClick, type = "button", disabled = false }: { children: React.ReactNode; primary?: boolean; onClick?: () => void; type?: "button" | "submit"; disabled?: boolean }) {
  return <button type={type} onClick={onClick} disabled={disabled} className={primary ? "button primary" : "button"}>{children}</button>;
}
function Card({ children, className = "", style }: { children: React.ReactNode; className?: string; style?: React.CSSProperties }) { return <section className={`card ${className}`} style={style}>{children}</section>; }
function Metric({ title, value, change, tone, icon }: { title: string; value: string; change: string; tone: string; icon: string }) {
  return <Card className="metric"><div className="metric-top"><span>{title}</span><span className={`metric-icon ${tone}`}>{icon}</span></div><strong>{value}</strong><small className={change.startsWith("↓") ? "down" : ""}>{change}</small></Card>;
}

function buildTrendPath(
  values: number[],
  width: number,
  height: number,
  paddingX = 20,
  paddingBottom = 20
): string {
  if (values.length === 0) return "";
  const innerHeight = height - paddingBottom;
  const maxV = Math.max(1, ...values);
  const stepX = values.length > 1 ? (width - paddingX * 2) / (values.length - 1) : 0;
  return values
    .map((v, i) => {
      const x = paddingX + stepX * i;
      const y = innerHeight - (v / maxV) * (innerHeight - 40) - 20;
      return `${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

function buildAreaPath(values: number[], width: number, height: number): string {
  if (values.length === 0) return "";
  const line = buildTrendPath(values, width, height);
  const paddingX = 20;
  const paddingBottom = 20;
  const first = paddingX;
  const last = width - paddingX;
  const bottom = height - paddingBottom;
  return `M ${first} ${bottom} L ${line.split(" ")[0]} L ${line
    .split(" ")
    .filter((_, i) => i % 2 === 0)
    .map((x, i) => {
      const pts = line.split(" ");
      return `${pts[i * 2]} ${pts[i * 2 + 1]}`;
    })
    .join(" L ")} L ${last} ${bottom} Z`;
}

function Dashboard({ go }: { go: (page: string) => void }) {
  const [modelSummary, setModelSummary] = useState<ServingModelInfo | null>(null);
  const [modelLoading, setModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);

  const [historyItems, setHistoryItems] = useState<LocalHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState<string | null>(null);

  const [ragStats, setRagStats] = useState<RagStats | null>(null);
  const [ragLoading, setRagLoading] = useState(true);
  const [ragError, setRagError] = useState<string | null>(null);

  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [agentLoading, setAgentLoading] = useState(true);
  const [agentError, setAgentError] = useState<string | null>(null);
  const [apiHealth, setApiHealth] = useState<HealthResponse | null>(null);
  const [supabaseStatus, setSupabaseStatus] = useState<SupabaseStatus | null>(null);
  const [matrixUnavailable, setMatrixUnavailable] = useState(false);

  const [todayLabel] = useState(() =>
    new Date().toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" })
  );
  const [lastUpdated, setLastUpdated] = useState<string>(() =>
    new Date().toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })
  );

  async function loadDashboard(abort?: AbortSignal) {
    setLastUpdated(new Date().toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }));
    const pModel = getServingModelInfo();
    const pHistory = listDiagnosisHistory(undefined, 500);
    const pRag = getRagStats();
    const pAgent = getAgentStatus();
    const pHealth = getHealth();
    const pSupabase = getSupabaseStatus();
    setModelLoading(true);
    setHistoryLoading(true);
    setRagLoading(true);
    setAgentLoading(true);
    try {
      const [m, h, r, a, liveHealth, dbStatus] = await Promise.all([
        pModel.catch((cause) => {
          if (!abort?.aborted) setModelError(cause instanceof Error ? cause.message : "Unavailable");
          return null;
        }),
        pHistory.catch((cause) => {
          if (!abort?.aborted) setHistoryError(cause instanceof Error ? cause.message : "Unavailable");
          return { items: [] as LocalHistoryItem[], count: 0, source: "local" as const };
        }),
        pRag.catch((cause) => {
          if (!abort?.aborted) setRagError(cause instanceof Error ? cause.message : "Unavailable");
          return null;
        }),
        pAgent.catch((cause) => {
          if (!abort?.aborted) setAgentError(cause instanceof Error ? cause.message : "Unavailable");
          return null;
        }),
        pHealth.catch(() => null),
        pSupabase.catch(() => null),
      ]);
      if (abort?.aborted) return;
      if (m) {
        setModelSummary(m);
        setModelError(null);
      }
      if (h) {
        setHistoryItems(h.items);
        setHistoryError(null);
      }
      if (r) {
        setRagStats(r);
        setRagError(null);
      }
      if (a) {
        setAgentStatus(a);
        setAgentError(null);
      }
      if (liveHealth) setApiHealth(liveHealth);
      if (dbStatus) setSupabaseStatus(dbStatus);
    } finally {
      if (!abort?.aborted) {
        setModelLoading(false);
        setHistoryLoading(false);
        setRagLoading(false);
        setAgentLoading(false);
      }
    }
  }

  useEffect(() => {
    const ctrl = new AbortController();
    void loadDashboard(ctrl.signal);
    return () => ctrl.abort();
  }, []);

  const agg: DashboardAggregate = useMemo(() => aggregateHistory(historyItems), [historyItems]);
  const ragCounts = useMemo(() => mapCategoryCounts(ragStats?.by_category), [ragStats]);
  const latestAgentRun = historyItems.find((item) => Array.isArray((item as LocalHistoryItem & { agent_events?: unknown[] }).agent_events));
  const latestAgentEvents = Array.isArray((latestAgentRun as (LocalHistoryItem & { agent_events?: unknown[] }) | undefined)?.agent_events)
    ? (latestAgentRun as LocalHistoryItem & { agent_events: { agent_name: string; status: string }[] }).agent_events
    : [];
  const readyServices = [agentStatus?.supervisor.ready, agentStatus?.supervisor.chat_ready, agentStatus?.supervisor.inference_ready, ragStats?.indexed].filter(Boolean).length;

  const ringValue = modelSummary?.test_acc ?? 0;
  const ringPercent = Math.round(ringValue * 10000) / 100;

  const totalDiagnoses = agg.total;
  const totalReports = agg.reports_count;
  const avgConfidencePct = Math.round(agg.avg_confidence * 10000) / 100;

  const healthyChange = historyLoading || !agg.total ? "—" : `${agg.healthy} tracked`;
  const diseasedChange = historyLoading || !agg.total ? "—" : `${agg.diseased} tracked`;
  const confChange = historyLoading || agg.low_confidence === 0 ? "Stable" : `${agg.low_confidence} low conf.`;
  const confTone = historyLoading || agg.low_confidence === 0 ? "green" : "gold";

  const trendMaxY = Math.max(1, ...agg.trend.healthy, ...agg.trend.diseased);
  const trendWidth = 500;
  const trendHeight = 200;
  const healthyPath = buildTrendPath(agg.trend.healthy, trendWidth, trendHeight);
  const diseasedPath = buildTrendPath(agg.trend.diseased, trendWidth, trendHeight);
  const areaPath = buildAreaPath(agg.trend.healthy, trendWidth, trendHeight);

  return (
    <div className="page-content">
      <div className="welcome">
        <div>
          <h1>
            Welcome back! <span>👋</span>
          </h1>
          <p>Here&apos;s what&apos;s happening with your farm today.</p>
        </div>
        <div className="welcome-actions">
          <Button>
            {todayLabel}　▣
          </Button>
          <Button onClick={() => void loadDashboard()}>⟳ Refresh</Button>
          <Button primary onClick={() => go("Upload Image")}>＋ New Diagnosis</Button>
        </div>
      </div>

      <div className="metrics">
        <Metric
          title="Total Diagnoses"
          value={historyLoading ? "…" : String(totalDiagnoses)}
          change={historyLoading || !historyError ? "Tracked diagnoses" : "Offline"}
          tone={historyError ? "gold" : "green"}
          icon="◴"
        />
        <Metric
          title="Healthy Plants"
          value={historyLoading ? "…" : String(agg.healthy)}
          change={healthyChange}
          tone={historyError ? "gold" : "green"}
          icon="♧"
        />
        <Metric
          title="Diseased Plants"
          value={historyLoading ? "…" : String(agg.diseased)}
          change={diseasedChange}
          tone={historyError ? "gold" : "red"}
          icon="✣"
        />
        <Metric
          title="Avg. Confidence"
          value={historyLoading ? "…" : agg.total ? `${avgConfidencePct.toFixed(1)}%` : "—"}
          change={agg.total ? confChange : "No diagnoses yet"}
          tone={historyError ? "gold" : confTone}
          icon="▥"
        />
        <Metric
          title="Reports Generated"
          value={historyLoading ? "…" : String(totalReports)}
          change={historyError ? "Using local" : "JSON reports"}
          tone={historyError ? "gold" : "blue"}
          icon="▤"
        />
      </div>

      <div className="dashboard-grid">
        {/* Recent Diagnoses — REAL */}
        <Card className="recent">
          <div className="card-heading">
            <h3>Recent Diagnoses</h3>
            <button onClick={() => go("History")}>View All</button>
          </div>
          {historyLoading ? (
            <p className="loading-state">Loading recent diagnoses…</p>
          ) : historyError ? (
            <><p className="error">Could not load history: {historyError}</p>{historyError.startsWith("Sign in") && <Button onClick={() => go("Profile")}>Sign in</Button>}</>
          ) : agg.recent.length === 0 ? (
            <p className="subtle" style={{ padding: 24 }}>
              No diagnoses yet. Run a plant image upload to populate history.
            </p>
          ) : (
            <div className="table">
              <div className="table-head">
                <span>Plant / Disease</span>
                <span>Confidence</span>
                <span>Date</span>
                <span>Action</span>
              </div>
              {agg.recent.map((r) => {
                const top =
                  r.predicted_class ??
                  (r.prediction
                    ? {
                        class_id: -1,
                        label: r.prediction.label,
                        confidence: r.prediction.score ?? r.prediction.confidence_pct ?? 0,
                      }
                    : (Array.isArray(r.top5) && r.top5.length > 0 ? r.top5[0] : undefined));
                const lbl = LabelDisplay(r);
                const conf = top?.confidence ?? 0;
                const ts = formatTimestamp(r.timestamp);
                const pctVal = (conf * 100).toFixed(2);
                return (
                  <div className="table-row" key={r.id} style={{ gridTemplateColumns: "1.8fr 1fr 1fr 40px" }}>
                    <span>
                      <b style={{ color: r.low_confidence ? "#e5b438" : undefined }}>{lbl.label}</b>
                      <small>{lbl.secondary || r.source_filename || r.id.slice(0, 8)}</small>
                    </span>
                    <span>
                      <ConfidenceBadge value={conf} />
                    </span>
                    <span className="date">
                      {ts.date}
                      <small>{ts.time}</small>
                    </span>
                    <span style={{ display: "flex", gap: 4 }}>
                      <button className="view" title="View in History" onClick={() => go("History")}>
                        ⊙
                      </button>
                    </span>
                  </div>
                );
              })}
              {agg.total > agg.recent.length && (
                <p className="subtle" style={{ marginTop: 6, textAlign: "right" }}>
                  Showing {agg.recent.length} of {agg.total} diagnoses
                </p>
              )}
            </div>
          )}
        </Card>

        {/* Diagnosis Trend — REAL 7-day aggregation */}
        <Card className="trend">
          <div className="card-heading">
            <h3>Diagnosis Trend</h3>
            <button>Last 7 Days⌄</button>
          </div>
          <div className="legend">
            <span>
              <i className="dot green-dot" /> Healthy ({agg.trend.healthy.reduce((a, b) => a + b, 0)})
            </span>
            <span>
              <i className="dot red-dot" /> Diseased ({agg.trend.diseased.reduce((a, b) => a + b, 0)})
            </span>
          </div>
          {historyLoading ? (
            <p className="loading-state">Computing 7-day trend…</p>
          ) : !agg.total ? (
            <p className="subtle" style={{ padding: 24 }}>
              Not enough diagnoses yet to build a trend.
            </p>
          ) : (
            <>
              <svg className="chart" viewBox={`0 0 ${trendWidth} ${trendHeight}`}>
                <defs>
                  <linearGradient id="dashGreenFill" x1="0" x2="0" y1="0" y2="1">
                    <stop stopColor="#5ed36f" stopOpacity=".25" />
                    <stop offset="1" stopColor="#5ed36f" stopOpacity="0" />
                  </linearGradient>
                </defs>
                {areaPath && <path d={areaPath} fill="url(#dashGreenFill)" />}
                {healthyPath && <path d={`M ${healthyPath}`} fill="none" stroke="#5ed36f" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />}
                {diseasedPath && <path d={`M ${diseasedPath}`} fill="none" stroke="#e85b58" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />}
              </svg>
              <div className="axis">
                {agg.trend.days.map((d) => (
                  <span key={d}>{d}</span>
                ))}
              </div>
            </>
          )}
        </Card>

        {/* Model Performance — REAL (already connected) */}
        <Card className="performance">
          <div className="card-heading">
            <h3>Model Performance</h3>
            <button onClick={() => go("Model Performance")}>View Details</button>
          </div>
          {modelLoading ? (
            <p className="loading-state">Loading model metrics…</p>
          ) : modelError ? (
            <p className="error">Could not load metrics: {modelError}</p>
          ) : (
            <>
              <div className="ring">
                <span>
                  {ringPercent.toFixed(2)}%
                  <small>Test Accuracy</small>
                </span>
              </div>
              <div className="stats">
                <p>
                  <span>Validation Accuracy</span>
                  <b>{pct(modelSummary?.best_val_acc ?? undefined, 2)}</b>
                </p>
                <p>
                  <span>Test Macro F1</span>
                  <b>{pct(modelSummary?.test_macro_f1 ?? undefined, 2)}</b>
                </p>
                <p>
                  <span>Val Macro F1</span>
                  <b>{pct(modelSummary?.best_val_macro_f1 ?? undefined, 2)}</b>
                </p>
                <p>
                  <span>Total Classes</span>
                <b>{String(modelSummary?.num_classes ?? "—")}</b>
                </p>
              </div>
              <small className="model-label">
                ● Model: {modelSummary?.backbone ?? "Unavailable"}
              </small>
            </>
          )}
        </Card>

        {/* Confusion Matrix — REAL PNG */}
        <Card className="matrix">
          <div className="card-heading">
            <h3>Confusion Matrix</h3>
            <button onClick={() => go("Model Insights")}>View Matrix</button>
          </div>
          <img
            className="matrix-image"
            alt="Confusion Matrix"
            src={getGraphUrl("confusion_matrix")}
            style={{ width: "100%", height: "auto", borderRadius: 6, background: "transparent", padding: 4 }}
            onError={() => setMatrixUnavailable(true)}
          />
          {matrixUnavailable && <p className="subtle">Confusion matrix is unavailable from the analytics API.</p>}
          <div className="matrix-axis">
            {modelSummary?.num_classes ? `${modelSummary.num_classes} classes` : ""}
          </div>
        </Card>

        {/* Agent Orchestrator — REAL status/progress */}
        <Card className="orchestrator">
          <div className="card-heading">
            <h3>Agent Orchestrator</h3>
            {agentLoading ? (
              <span className="live">…</span>
            ) : agentError ? (
              <span className="low-confidence">Offline</span>
            ) : agentStatus?.supervisor.state === "active" ? (
              <span className="live">Live</span>
            ) : agentStatus?.supervisor.state === "standby" ? (
              <span style={{ color: "#e5b438", fontSize: 10 }}>Standby</span>
            ) : (
              <span style={{ color: "#a1a1a1", fontSize: 10 }}>Idle</span>
            )}
          </div>
          {agentLoading ? (
            <p className="loading-state">Loading agent system status…</p>
          ) : (
            <>
              <p className="subtle">Supervisor: <b>{agentStatus?.supervisor.state ?? "Unavailable"}</b></p>
              {latestAgentEvents.length ? <div className="agent-event-list">{latestAgentEvents.map((event, index) => <div className="agent-event" key={`${event.agent_name}-${index}`}>
                <span className={event.status === "completed" ? "event-ok" : event.status === "failed" ? "event-failed" : "event-running"}>{event.status === "completed" ? "?" : event.status === "failed" ? "!" : "?"}</span>
                <div><b>{event.agent_name}</b><small>{event.status}</small></div>
              </div>)}</div> : <p className="subtle">No executed agent workflow is recorded in recent diagnosis history.</p>}
              <div className="progress-line">
                <span>Services Ready</span>
                <i><b style={{ width: `${(readyServices / 4) * 100}%` }} /></i>
                <strong>{readyServices}/4</strong>
              </div>
              {agentStatus?.agent_runs.recorded_supabase ? (
                <small className="subtle">{agentStatus.agent_runs.recorded_supabase} agent runs recorded</small>
              ) : null}
            </>
          )}
        </Card>

        {/* System Status — REAL */}
        <Card className="status">
          <div className="card-heading">
            <h3>System Status</h3>
          </div>
          <p>
            <span>◉　ML Model</span>
            <small
              className={
                !modelLoading && !modelError && modelSummary ? "connected" : undefined
              }
            >
              {modelLoading ? "Loading…" : modelError ? "Unavailable" : modelSummary?.backbone ?? "Unavailable"}
            </small>
          </p>
          <p>
            <span>◉　API Server</span>
            <small className={apiHealth?.status === "ok" ? "connected" : undefined}>
              {apiHealth ? (apiHealth.status === "ok" ? "Operational" : apiHealth.status) : "Unavailable"}
            </small>
          </p>
          <p>
            <span>◉　Database</span>
            <small className={supabaseStatus?.configured ? "connected" : undefined}>
              {agentLoading ? "…" : supabaseStatus ? (supabaseStatus.configured ? "Configured" : "Local only") : "Unavailable"}
            </small>
          </p>
          <p>
            <span>◉　Vector DB</span>
            <small className={ragStats?.indexed ? "connected" : undefined}>
              {ragLoading ? "…" : ragError ? "Offline" : ragStats?.backend === "qdrant" ? "Qdrant" : ragStats?.backend === "local-memory" ? "In-memory" : ragStats?.indexed ? "Indexed" : "Empty"}
            </small>
          </p>
          <p>
            <span>◉　Agent System</span>
            <small className={agentStatus?.supervisor.state === "active" ? "connected" : undefined}>
              {agentLoading ? "…" : agentError ? "Offline" : agentStatus?.supervisor.state === "active" ? "Active" : agentStatus?.supervisor.state === "standby" ? "Standby" : "Idle"}
            </small>
          </p>
          <small className="updated">
            Last checked: {todayLabel} {lastUpdated}　⟳
          </small>
        </Card>

        {/* Quick Actions */}
        <Card className="quick">
          <h3>Quick Actions</h3>
          <div>
            {[
              ["↥", "Upload Image", "Single Image"],
              ["▱", "Batch Detection", "Multiple Images"],
              ["◉", "Live Detection", "Camera Real-time"],
              ["✧", "AI Chat", "Ask Assistant"],
              ["▤", "Reports", "All Reports"],
            ].map((x) => (
              <button
                key={x[1]}
                onClick={() => go(x[1])}
              >
                <strong>{x[0]}</strong>
                <b>{x[1]}</b>
                <small>{x[2]}</small>
              </button>
            ))}
          </div>
        </Card>

        {/* Recent Reports — REAL top history report */}
        <Card className="reports">
          <div className="card-heading">
            <h3>Recent Reports</h3>
            <button onClick={() => go("Reports")}>View All</button>
          </div>
          {historyLoading ? (
            <p className="loading-state">Loading reports…</p>
          ) : historyError ? (
            <><p className="error">Could not load reports: {historyError}</p>{historyError.startsWith("Sign in") && <Button onClick={() => go("Profile")}>Sign in</Button>}</>
          ) : agg.recent.length === 0 ? (
            <p className="subtle" style={{ padding: 24 }}>
              Run a diagnosis to generate a report.
            </p>
          ) : (
            (() => {
              const top = agg.recent[0];
              const lbl = LabelDisplay(top);
              const topConf =
                top.predicted_class?.confidence ??
                top.prediction?.score ??
                top.prediction?.confidence_pct ??
                (Array.isArray(top.top5) && top.top5.length ? top.top5[0].confidence : 0);
              const ts = formatTimestamp(top.timestamp);
              return (
                <div className="report-preview">
                  <span>{lbl.label.toLowerCase().includes("healthy") ? "🌿" : "🍂"}</span>
                  <div>
                    <b>{lbl.label} Report</b>
                    <small>
                      {ts.date} · {ts.time}
                    </small>
                    <small>
                      JSON · {(topConf * 100).toFixed(1)}% confidence
                    </small>
                  </div>
                  <button title="Download JSON" onClick={() => go("Reports")}>
                    ↓
                  </button>
                </div>
              );
            })()
          )}
        </Card>

        {/* Knowledge Base (RAG) — REAL counts */}
        <Card className="knowledge">
          <div className="card-heading">
            <h3>Knowledge Base (RAG)</h3>
            <button onClick={() => go("RAG Knowledge Base")}>Explore</button>
          </div>
          {ragLoading ? (
            <p className="loading-state">Checking RAG knowledge index…</p>
          ) : ragError ? (
            <p className="error">Could not load RAG stats: {ragError}</p>
          ) : !ragStats?.indexed && !ragStats?.total_points ? (
            <p className="subtle" style={{ padding: 16 }}>
              Knowledge base is empty — ingest agricultural documents to begin RAG.
            </p>
          ) : (
            <>
              {[
                { key: "Diseases", count: ragCounts.diseases, icon: "✣" },
                { key: "Fertilizers", count: ragCounts.fertilizers, icon: "✦" },
                { key: "Treatments", count: ragCounts.treatments, icon: "♧" },
                { key: "Plant Care", count: ragCounts.plantCare, icon: "♡" },
              ].map((item, i) => (
                <div className="knowledge-item" key={item.key}>
                  <span>{item.icon}</span>
                  <b>
                    {item.key}
                    <small>
                      {`${item.count + (i === 3 ? ragCounts.other : 0)} chunks`}
                    </small>
                  </b>
                </div>
              ))}
              <p className="subtle" style={{ marginTop: 6, padding: "0 4px" }}>
                {ragStats?.total_points ?? 0} chunks · {ragStats?.sources_indexed ?? 0} sources ·{" "}
                {formatBytes(ragStats?.tfidf_size_bytes)}
              </p>
            </>
          )}
        </Card>
      </div>
    </div>
  );
}

function Upload({ batch = false, live = false }: { batch?: boolean; live?: boolean }) {
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const { result, loading, error, partialErrors, run } = useDiagnosis();

  function select(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files || []);
    previews.forEach((preview) => URL.revokeObjectURL(preview));
    setFiles(selected);
    setPreviews(selected.map((file) => URL.createObjectURL(file)));
  }

  useEffect(() => () => previews.forEach((preview) => URL.revokeObjectURL(preview)), [previews]);
  function submit(event: FormEvent) { event.preventDefault(); void run(files, batch); }
  const list = result ? (Array.isArray(result) ? result : [result]) : [];
  const title = live ? "Live Detection" : batch ? "Batch Detection" : "Plant Diagnosis";

  return (
    <div className="page-content inner-page">
      <div className="section-title"><div><h1>{title}</h1><p>{batch ? "Analyze up to 64 images with the served model." : live ? "Capture or choose a plant image and send it through live model inference." : "Upload a clear plant image for an AI-powered diagnosis."}</p></div></div>
      <Card className="upload-card">
        <form onSubmit={submit}>
          <label className="dropzone">
            <input type="file" accept="image/*" capture={live ? "environment" : undefined} multiple={batch} onChange={select} />
            <span className="upload-icon">↥</span>
            <b>{files.length ? `${files.length} image${files.length > 1 ? "s" : ""} selected` : "Drop your plant image here"}</b>
            <small>{batch ? "Up to 64 JPG, PNG, WEBP images · 16 MB each" : "or click to browse · JPG, PNG, WEBP up to 16 MB"}</small>
          </label>
          {files.length > 0 && <div className="file-list">{files.map((file, index) => <span key={`${file.name}-${index}`}>{previews[index] && <img src={previews[index]} alt="" />} ◉ {file.name}</span>)}</div>}
          <Button primary type="submit">{loading ? "Analyzing…" : batch ? "Run Batch Detection" : "Analyze Image"}</Button>
          {error && <p className="error">API error: {error}</p>}
        </form>
      </Card>
      {loading && <Card className="loading-state">Running image inference through the active model…</Card>}
      {partialErrors.map((message) => <Card key={message}><p className="error">{message}</p></Card>)}
      {!loading && !error && list.length === 0 && files.length > 0 && <Card className="subtle">No prediction was returned for the selected image.</Card>}
      {list.length > 0 && <div className="result-grid">{list.map((diagnosis, index) => (
        <Card className="prediction-card" key={`${diagnosis.id}-${index}`}>
          {previews[index] && <img className="result-image" src={previews[index]} alt={diagnosis.source_filename || "Uploaded plant"} />}
          <div className="card-heading"><h3>{diagnosis.source_filename || `Image ${index + 1}`}</h3><span className="confidence">{(diagnosis.predicted_class.confidence * 100).toFixed(2)}%</span></div>
          {diagnosis.low_confidence && <><p className="low-confidence">Low confidence result — try a clearer, closer leaf image.</p><ExpertReviewForm diagnosis={diagnosis} /></>}
          <h2>{diagnosis.predicted_class.label}</h2>
          <p className="subtle">Inference time · {diagnosis.inference_ms.toFixed(1)} ms</p>
          <div className="model-info"><b>{diagnosis.model_info.backbone}</b><span>{diagnosis.model_info.num_classes} classes · {diagnosis.model_info.img_size}px input</span></div>
          <div className="top-list">{diagnosis.top5.map((prediction, rank) => <p key={`${prediction.class_id}-${prediction.label}`}><span>{rank + 1}. {prediction.label}</span><b>{(prediction.confidence * 100).toFixed(2)}%</b><i><em style={{ width: `${prediction.confidence * 100}%` }} /></i></p>)}</div>
        </Card>
      ))}</div>}
    </div>
  );
}

function ExpertReviewForm({ diagnosis }: { diagnosis: Diagnosis }) {
  const [decision, setDecision] = useState("confirmed");
  const [corrected, setCorrected] = useState("");
  const [notes, setNotes] = useState("");
  const [review, setReview] = useState<ExpertReview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function submit(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError(null);
    try { setReview(await submitExpertReview({ diagnosis_id: diagnosis.id, reviewer_decision: decision, corrected_disease: corrected.trim() || null, notes: notes.trim() || null })); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Could not save expert review."); }
    finally { setBusy(false); }
  }
  return <div className="expert-review"><h3>Expert Review Recommended</h3><p className="subtle">Record a human review. This feedback is stored for review and does not change the EfficientNet-B0 model.</p>
    {review ? <p className="live">Review saved · {new Date(review.created_at).toLocaleString()}</p> : <form className="query-form" onSubmit={submit}>
      <label>Diagnosis<input value={diagnosis.predicted_class.label} readOnly /></label><label>Confidence<input value={`${(diagnosis.predicted_class.confidence * 100).toFixed(2)}%`} readOnly /></label>
      <label>Reviewer decision<select value={decision} onChange={(event) => setDecision(event.target.value)}><option value="confirmed">Confirmed</option><option value="incorrect">Incorrect</option><option value="uncertain">Uncertain</option></select></label>
      <label>Corrected disease / class<input value={corrected} onChange={(event) => setCorrected(event.target.value)} placeholder="Optional correction" /></label>
      <label>Notes<textarea value={notes} onChange={(event) => setNotes(event.target.value)} maxLength={5000} /></label>
      <Button primary type="submit" disabled={busy}>{busy ? "Saving review…" : "Save review"}</Button>
    </form>}{error && <p className="error">{error}</p>}</div>;
}

function ChatAssistant() {
  const [question, setQuestion] = useState("");
  const [diagnosisId, setDiagnosisId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [result, setResult] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit(e: FormEvent) {
    e.preventDefault();
    const content = question.trim();
    if (!content || loading) return;
    const next = [...messages, { role: "user" as const, content }];
    setMessages(next);
    setQuestion("");
    setLoading(true);
    setError("");
    try {
      const response = await askTerraMind(next, diagnosisId.trim() || undefined);
      setResult(response);
      setMessages([...next, { role: "assistant", content: response.response || "No grounded response was returned for this question." }]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Chat request failed.");
    } finally {
      setLoading(false);
    }
  }

  return <div className="page-content inner-page"><div className="section-title"><div><h1>AI Chat Assistant</h1><p>Ask questions grounded in TerraMind&apos;s indexed agricultural knowledge.</p></div></div><div className="dashboard-grid"><Card className="chat-card"><div className="chat-messages">{messages.length === 0 && <p className="subtle">Ask about a disease, treatment, fertilizer, or plant care.</p>}{messages.map((message, index) => <div className={`chat-message ${message.role}`} key={`${message.role}-${index}`}><b>{message.role === "user" ? "You" : "TerraMind"}</b><p>{message.content}</p></div>)}{loading && <p className="loading-state">Supervisor is routing the question through RAG and a specialist agent…</p>}</div><form onSubmit={submit}><input className="chat-input" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a plant health question…" disabled={loading} /><input className="chat-context-input" value={diagnosisId} onChange={(event) => setDiagnosisId(event.target.value)} placeholder="Optional diagnosis ID for context" disabled={loading} /><Button primary type="submit">{loading ? "Researching…" : "Ask TerraMind"}</Button></form>{error && <p className="error">API error: {error}</p>}</Card>{result && <Card className="chat-status"><div className="card-heading"><h3>Workflow status</h3><span className={result.grounded ? "live" : "low-confidence"}>{result.grounded ? "Grounded" : "No sources"}</span></div><p className="subtle">Workflow: {result.workflow_id}</p><p>Specialist: <b>{result.specialist_agent}</b></p>{result.workflow.map((event, index) => <p key={`${event.agent_name}-${index}`}><span>{event.status === "completed" ? "●" : "!"} {event.agent_name}</span><small>{event.status}</small></p>)}<h3>Retrieved sources ({result.sources.length})</h3>{result.sources.map((source, index) => <div className="source-item" key={`${source.source}-${index}`}><b>{source.title || "Knowledge source"}</b><small>{source.source}</small><p>{source.text?.slice(0, 240)}</p></div>)}</Card>}</div></div>;
}

function ConfidenceBadge({ value }: { value: number }) {
  const pct = Math.round(value * 10000) / 100;
  const tone = pct >= 75 ? "confidence" : pct >= 50 ? "confidence warn" : "low-confidence";
  return <strong className={tone}>{pct.toFixed(2)}%</strong>;
}

function TopPrediction(item: LocalHistoryItem): Prediction | undefined {
  if (item.predicted_class) return item.predicted_class;
  if (item.prediction) return { class_id: -1, label: item.prediction.label, confidence: item.prediction.score ?? item.prediction.confidence_pct ?? 0 };
  if (Array.isArray(item.top5) && item.top5.length > 0) return item.top5[0];
  return undefined;
}

function LabelDisplay(item: LocalHistoryItem): { label: string; secondary: string } {
  const p = TopPrediction(item);
  if (!p) return { label: "Unknown", secondary: "—" };
  const label = p.label.replace(/[_-]/g, " ");
  const nameParts = label.split(/\s*-\s*|\s*:\s*/).filter(Boolean);
  if (nameParts.length >= 2) return { label: nameParts[0].trim(), secondary: nameParts.slice(1).join(" — ").trim() };
  return { label: label, secondary: "" };
}

function HistoryPage({ go }: { go: (page: string) => void }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [items, setItems] = useState<LocalHistoryItem[]>([]);
  const [source, setSource] = useState<"supabase" | "local">("local");
  const [total, setTotal] = useState(0);
  const [limit, setLimit] = useState(50);
  const [selected, setSelected] = useState<LocalHistoryItem | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  async function load(limitOverride?: number) {
    setLoading(true);
    setError(null);
    try {
      const res = await listDiagnosisHistory(undefined, limitOverride ?? limit);
      setItems(res.items);
      setSource(res.source);
      setTotal(res.count);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load diagnosis history.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void load(); }, []);

  async function openDetail(item: LocalHistoryItem) {
    setDetailLoading(true);
    try {
      setSelected(await getDiagnosisById(item.id));
    } catch {
      setSelected(item);
    } finally {
      setDetailLoading(false);
    }
  }

  const stats = useMemo(() => {
    let lowConf = 0;
    let healthy = 0;
    let diseased = 0;
    let confSum = 0;
    for (const item of items) {
      const p = TopPrediction(item);
      if (!p) continue;
      const c = p.confidence ?? 0;
      confSum += c;
      if (item.low_confidence || c < 0.35) lowConf++;
      const lbl = p.label.toLowerCase();
      if (lbl.includes("healthy") || lbl.includes("normal") || lbl.includes("good")) healthy++;
      else diseased++;
    }
    return {
      lowConf,
      healthy,
      diseased,
      avgConf: items.length ? (confSum / items.length) * 100 : 0,
    };
  }, [items]);

  return (
    <div className="page-content inner-page">
      <div className="section-title">
        <div>
          <h1>Diagnosis History</h1>
          <p>
            {source === "supabase" ? "Stored diagnoses loaded from Supabase." : "Local diagnosis records from the TerraMind backend."} Showing up to {limit} records.
          </p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <select
            className="button"
            value={String(limit)}
            onChange={(e) => { const v = Number(e.target.value); setLimit(v); void load(v); }}
            style={{ cursor: "pointer" }}
          >
            <option value="20">20 / page</option>
            <option value="50">50 / page</option>
            <option value="100">100 / page</option>
            <option value="500">500 / page</option>
          </select>
          <Button onClick={() => void load()}>⟳ Refresh</Button>
          <Button primary onClick={() => go("Upload Image")}>＋ New Diagnosis</Button>
        </div>
      </div>

      {items.length > 0 && (
        <div className="metrics" style={{ gridTemplateColumns: "repeat(4,1fr)", marginBottom: 10 }}>
          <Metric title="Total Diagnoses" value={String(total || items.length)} change={`↑ ${items.length}`} tone="green" icon="◴" />
          <Metric title="Healthy Plants" value={String(stats.healthy)} change={stats.healthy ? "Tracked" : "—"} tone="green" icon="♧" />
          <Metric title="Diseased Plants" value={String(stats.diseased)} change={stats.diseased ? "Tracked" : "—"} tone="red" icon="✣" />
          <Metric title="Avg. Confidence" value={`${stats.avgConf.toFixed(1)}%`} change={stats.lowConf ? `${stats.lowConf} low conf.` : "Stable"} tone={stats.lowConf ? "gold" : "green"} icon="▥" />
        </div>
      )}

      {loading && (
        <Card className="loading-state">
          <p>Loading diagnosis history…</p>
        </Card>
      )}

      {!loading && error && (
        <Card>
          <p className="error">Failed to load history: {error}</p>
          <Button onClick={() => void load()}>Retry</Button>
        </Card>
      )}

      {!loading && !error && items.length === 0 && (
        <Card className="empty-state">
          <span className="empty-icon">◴</span>
          <h2>No diagnoses yet</h2>
          <p>Run a plant image diagnosis to build your history. Your local records appear here automatically.</p>
          <Button primary onClick={() => go("Upload Image")}>Run Your First Diagnosis</Button>
        </Card>
      )}

      {!loading && !error && items.length > 0 && !selected && (
        <Card>
          <div className="card-heading"><h3>{items.length} Records</h3><small className="subtle">Source: {source}</small></div>
          <div className="table">
            <div className="table-head">
              <span>Plant / Disease</span>
              <span>Confidence</span>
              <span>Model</span>
              <span>Inference</span>
              <span>Date</span>
              <span>Action</span>
            </div>
            {items.map((item) => {
              const lbl = LabelDisplay(item);
              const top = TopPrediction(item);
              const ts = formatTimestamp(item.timestamp);
              return (
                <div className="table-row" key={item.id} style={{ gridTemplateColumns: "1.8fr .8fr 1fr .8fr 1fr 40px" }}>
                  <span>
                    <b style={{ color: item.low_confidence ? "#e5b438" : undefined }}>{lbl.label}</b>
                    <small>{lbl.secondary || item.source_filename || item.id.slice(0, 8)}</small>
                  </span>
                  {top ? <span><ConfidenceBadge value={top.confidence} /></span> : <span>—</span>}
                  <span>
                    <small>{String(item.model_info?.backbone ?? "—")}</small>
                    <b>{String(item.model_info?.img_size ?? "")}px</b>
                  </span>
                  <span>
                    <b>{typeof item.inference_ms === "number" ? `${item.inference_ms.toFixed(0)} ms` : "—"}</b>
                    {item.low_confidence && <small style={{ color: "#e5b438" }}>⚠ low conf.</small>}
                  </span>
                  <span className="date">{ts.date}<small>{ts.time}</small></span>
                  <span style={{ display: "flex", gap: 4 }}>
                    <button className="view" title="View details" onClick={() => void openDetail(item)}>⊙</button>
                  </span>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {!loading && selected && (
        <>
          <Card style={{ marginBottom: 10 }}>
            <div className="card-heading">
              <h3>Diagnosis Details — {selected.id.slice(0, 8)}</h3>
              <Button onClick={() => setSelected(null)}>← Back to list</Button>
            </div>
            {detailLoading && <p className="loading-state">Loading full diagnosis…</p>}
            {!detailLoading && (
              <div className="result-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                <div>
                  {(() => {
                    const top = TopPrediction(selected);
                    if (!top) return null;
                    const lbl = LabelDisplay(selected);
                    return (
                      <>
                        <h2 style={{ color: "#72dd7d", marginTop: 0 }}>{lbl.label}</h2>
                        {lbl.secondary && <p className="subtle">{lbl.secondary}</p>}
                        <div className="model-info">
                          <b>{String(selected.model_info?.backbone ?? "Model")}</b>
                          <span>{String(selected.model_info?.num_classes ?? "?")} classes · {String(selected.model_info?.img_size ?? "?")}px</span>
                        </div>
                        <p><b>Confidence:</b> <ConfidenceBadge value={top.confidence} /></p>
                        <p><b>Inference:</b> {typeof selected.inference_ms === "number" ? `${selected.inference_ms.toFixed(1)} ms` : "—"}</p>
                        <p><b>Filename:</b> {selected.source_filename ?? "—"}</p>
                        <p><b>Timestamp:</b> {formatTimestamp(selected.timestamp).date} {formatTimestamp(selected.timestamp).time}</p>
                        {selected.low_confidence && (
                          <p className="low-confidence">Low confidence result — try a clearer, closer leaf image.</p>
                        )}
                      </>
                    );
                  })()}
                </div>
                <div>
                  <h3 style={{ fontSize: 12, marginBottom: 8 }}>Top-5 Predictions</h3>
                  <div className="top-list">
                    {(selected.top5 || []).map((p, n) => {
                      const conf = typeof p.confidence === "number" ? p.confidence : (typeof p.score === "number" ? p.score : 0);
                      return (
                        <p key={`${p.label}-${n}`}>
                          <span>{n + 1}. {p.label.replace(/[_-]/g, " ")}</span>
                          <b>{(conf * 100).toFixed(2)}%</b>
                          <i><em style={{ width: `${Math.max(2, conf * 100)}%` }} /></i>
                        </p>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  );
}

function ReportsPage({ go }: { go: (page: string) => void }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [items, setItems] = useState<(ReportRecord | LocalHistoryItem)[]>([]);
  const [source, setSource] = useState<"supabase" | "local">("local");
  const [total, setTotal] = useState(0);
  const [limit, setLimit] = useState(50);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  async function load(limitOverride?: number) {
    setLoading(true);
    setError(null);
    setDownloadError(null);
    try {
      const res = await listReports(undefined, limitOverride ?? limit);
      setItems(res.items);
      setSource(res.source);
      setTotal(res.count);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load reports.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void load(); }, []);

  function isReportRecord(x: ReportRecord | LocalHistoryItem): x is ReportRecord {
    return "bucket" in x && "report_type" in x;
  }

  function reportLabel(item: ReportRecord | LocalHistoryItem): { title: string; sub: string; type: string; size?: number } {
    if (isReportRecord(item)) {
      return {
        title: `Report ${item.id.slice(0, 8)}`,
        sub: `Diagnosis: ${item.diagnosis_id?.slice(0, 8) ?? "—"}`,
        type: item.report_type.toUpperCase(),
      };
    }
    const d = item as LocalHistoryItem;
    const top = (d.predicted_class?.label) || (d.prediction?.label) || (d.top5?.[0]?.label) || "Diagnosis";
    const pct = ((d.predicted_class?.confidence ?? d.prediction?.score ?? d.top5?.[0]?.confidence ?? 0) * 100);
    return {
      title: top.replace(/[_-]/g, " "),
      sub: `${d.source_filename || d.id.slice(0, 8)} · ${pct.toFixed(1)}%`,
      type: "JSON",
    };
  }

  async function downloadReport(item: ReportRecord | LocalHistoryItem) {
    setDownloading(item.id);
    setDownloadError(null);
    try {
      if (isReportRecord(item)) {
        const url = await getReportDownloadUrl(item);
        window.open(url, "_blank", "noopener,noreferrer");
      } else {
        const blob = new Blob([JSON.stringify(item, null, 2)], { type: "application/json" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `terramind-report-${item.id}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      }
    } catch (cause) {
      setDownloadError(cause instanceof Error ? cause.message : "Could not prepare report download.");
    } finally {
      setDownloading(null);
    }
  }

  function viewInHistory(item: ReportRecord | LocalHistoryItem) {
    const diagId = isReportRecord(item) ? (item.diagnosis_id ?? item.id) : item.id;
    void diagId;
    go("History");
  }

  const counts = useMemo(() => {
    let json = 0, pdf = 0, other = 0;
    for (const item of items) {
      const type = (isReportRecord(item) ? item.report_type : "json").toLowerCase();
      if (type === "json") json++;
      else if (type === "pdf") pdf++;
      else other++;
    }
    return { json, pdf, other };
  }, [items]);

  return (
    <div className="page-content inner-page">
      <div className="section-title">
        <div>
          <h1>Reports</h1>
          <p>
            {source === "supabase" ? "Generated reports from Supabase storage." : "Locally persisted diagnosis reports from the TerraMind backend."}
          </p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <select
            className="button"
            value={String(limit)}
            onChange={(e) => { const v = Number(e.target.value); setLimit(v); void load(v); }}
            style={{ cursor: "pointer" }}
          >
            <option value="20">20 / page</option>
            <option value="50">50 / page</option>
            <option value="100">100 / page</option>
            <option value="500">500 / page</option>
          </select>
          <Button onClick={() => void load()}>⟳ Refresh</Button>
          <Button primary onClick={() => go("Upload Image")}>＋ Generate Report</Button>
        </div>
      </div>

      {items.length > 0 && (
        <div className="metrics" style={{ gridTemplateColumns: "repeat(4,1fr)", marginBottom: 10 }}>
          <Metric title="Total Reports" value={String(total || items.length)} change={`↑ ${items.length}`} tone="green" icon="▤" />
          <Metric title="JSON Reports" value={String(counts.json)} change={counts.json ? "Downloadable" : "—"} tone="blue" icon="▱" />
          <Metric title="PDF Reports" value={String(counts.pdf)} change={counts.pdf ? "Downloadable" : "—"} tone="gold" icon="▤" />
          <Metric title="Other Formats" value={String(counts.other)} change={counts.other ? "Available" : "—"} tone="green" icon="◇" />
        </div>
      )}

      {loading && (
        <Card className="loading-state">
          <p>Loading reports…</p>
        </Card>
      )}

      {!loading && error && (
        <Card>
          <p className="error">Failed to load reports: {error}</p>
          <Button onClick={() => void load()}>Retry</Button>
        </Card>
      )}

      {downloadError && (
        <Card style={{ marginBottom: 10 }}>
          <p className="error">Download failed: {downloadError}</p>
        </Card>
      )}

      {!loading && !error && items.length === 0 && (
        <Card className="empty-state">
          <span className="empty-icon">▤</span>
          <h2>No reports available</h2>
          <p>Diagnoses generate reports automatically. Run a new plant image diagnosis to create your first downloadable report.</p>
          <Button primary onClick={() => go("Upload Image")}>Run Diagnosis &amp; Generate Report</Button>
        </Card>
      )}

      {!loading && !error && items.length > 0 && (
        <Card>
          <div className="card-heading"><h3>{items.length} Reports</h3><small className="subtle">Source: {source}</small></div>
          <div className="table">
            <div className="table-head">
              <span>Report</span>
              <span>Type</span>
              <span>Size</span>
              <span>Generated</span>
              <span>Actions</span>
            </div>
            {items.map((item) => {
              const info = reportLabel(item);
              const ts = formatTimestamp(isReportRecord(item) ? item.created_at : (item as LocalHistoryItem).timestamp);
              return (
                <div className="table-row" key={item.id} style={{ gridTemplateColumns: "2fr .6fr .8fr 1fr 120px" }}>
                  <span>
                    <b>{info.title}</b>
                    <small>{info.sub}</small>
                  </span>
                  <span>
                    <b style={{ background: "#153526", padding: "3px 8px", borderRadius: 4, color: "#79bd84", fontSize: 10 }}>{info.type}</b>
                  </span>
                  <span>
                    <b>{formatBytes(info.size)}</b>
                    {downloading === item.id ? <small className="loading-state" style={{ display: "inline", padding: 0, margin: 0, background: "transparent" }}>preparing…</small> : null}
                  </span>
                  <span className="date">{ts.date}<small>{ts.time}</small></span>
                  <span style={{ display: "flex", gap: 4 }}>
                    <button
                      className="view"
                      title="Download report"
                      onClick={() => void downloadReport(item)}
                      disabled={downloading === item.id}
                      style={{ opacity: downloading === item.id ? 0.6 : 1 }}
                    >↓</button>
                    {!isReportRecord(item) && (
                      <button
                        className="view"
                        title="View diagnosis"
                        onClick={() => go("History")}
                      >⊙</button>
                    )}
                  </span>
                </div>
              );
            })}
          </div>
        </Card>
      )}
    </div>
  );
}

export default function Home() {
  const [page, setPage] = useState("Dashboard");
  const [menu, setMenu] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [userName, setUserName] = useState("Guest");
  const [signedIn, setSignedIn] = useState(false);
  const [search, setSearch] = useState("");

  useEffect(() => {
    let active = true;
    const refreshHealth = () => getHealth().then((value) => {
      if (active) { setHealth(value); setHealthError(null); }
    }).catch((error: unknown) => {
      if (active) setHealthError(error instanceof Error ? error.message : "Backend unavailable");
    });
    void refreshHealth();
    const timer = window.setInterval(() => void refreshHealth(), 30000);
    if (supabase) {
      void supabase.auth.getSession().then(({ data }) => {
        if (active && data.session?.user) {
          const user = data.session.user;
          setSignedIn(true);
          setUserName(String(user.user_metadata?.full_name || user.user_metadata?.name || user.email || "Guest"));
        }
      }).catch(() => undefined);
      const { data } = supabase.auth.onAuthStateChange((_event, session) => {
        const user = session?.user;
        setSignedIn(Boolean(user));
        setUserName(user ? String(user.user_metadata?.full_name || user.user_metadata?.name || user.email || "Guest") : "Guest");
      });
      return () => { active = false; window.clearInterval(timer); data.subscription.unsubscribe(); };
    }
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  const content = useMemo(() => {
    if (page === "Dashboard") return <Dashboard go={setPage} />;
    if (page === "Upload Image" || page === "Plant Diagnosis") return <Upload />;
    if (page === "Live Detection") return <Upload live />;
    if (page === "Batch Detection") return <Upload batch />;
    if (page === "AI Chat Assistant") return <ChatAssistant />;
    if (page === "History") return <HistoryPage go={setPage} />;
    if (page === "Reports") return <ReportsPage go={setPage} />;
    if (page === "Treatment Advisor" || page === "Fertilizer Guide" || page === "Care Recommendations") return <RecommendationPage name={page} />;
    if (page === "RAG Knowledge Base") return <RagKnowledgeBasePage />;
    if (page === "Agent Orchestrator") return <AgentToolsPage mode="orchestrator" />;
    if (page === "Agent Workflow") return <AgentToolsPage mode="workflow" />;
    if (page === "Multimodal AI") return <MultimodalPage />;
    if (page === "Model Performance") return <ServedModelPerformancePage go={setPage} />;
    if (page === "Model Insights") return <ServedModelInsightsPage />;
    if (page === "System Status") return <SystemStatusPage />;
    if (page === "Settings") return <SettingsPage />;
    if (page === "Profile") return <ProfilePage />;
    return <Dashboard go={setPage} />;
  }, [page]);

  const executeSearch = (event: FormEvent) => {
    event.preventDefault();
    const query = search.trim().toLowerCase();
    if (!query) return;
    const item = nav.flatMap((section) => section.items).find(([name]) => name.toLowerCase().includes(query));
    if (item) { setPage(item[0] === "Plant Diagnosis" ? "Upload Image" : item[0]); setSearch(""); }
  };
  const healthy = health?.status === "ok";
  const avatar = userName.trim().charAt(0).toUpperCase() || "G";
  async function logout() { try { await signOut(); setPage("Profile"); } catch { setPage("Profile"); } }

  return <main className="app-shell">
    <aside className={`sidebar ${menu ? "open" : ""}`}>
      <div className="brand"><span>?</span><b>TerraMind <em>AI</em></b></div>
      <div className="nav">{nav.map((section) => <div key={section.group}><label>{section.group}</label>{section.items.map(([name, icon]) => <button className={page === name || (name === "Plant Diagnosis" && page === "Upload Image") ? "active" : ""} key={name} onClick={() => { setPage(name === "Plant Diagnosis" ? "Upload Image" : name); setMenu(false); }}><Icon>{icon}</Icon>{name}{name === "Plant Diagnosis" && <small>?</small>}</button>)}</div>)}</div>
      <div className="sidebar-status"><span>?</span><div><b>System Status</b><small>{healthError ? "Backend unavailable" : healthy ? "All Systems Operational" : health ? `Service ${health.status}` : "Checking backend?"}</small></div></div>
      <footer>{health?.version ? `API ${health.version}` : "TerraMind AI"}</footer>
    </aside>
    {menu && <button className="sidebar-backdrop" aria-label="Close navigation" onClick={() => setMenu(false)} />}
    <div className="workspace">
      <header className="topbar"><button className="hamburger" aria-label="Toggle navigation" aria-expanded={menu} onClick={() => setMenu(!menu)}>?</button>
        <form className="search" onSubmit={executeSearch}><span aria-hidden="true">?</span><input aria-label="Search pages" placeholder="Search diseases, treatments, fertilizers, or pages?" value={search} onChange={(event) => setSearch(event.target.value)} /><kbd>Enter</kbd></form>
        <div className="top-actions"><div className="user-avatar" aria-hidden="true">{avatar}</div><div className="user-name"><b>{userName}</b><small>{supabase ? (signedIn ? "TerraMind account" : "Sign in for saved data") : "Local workspace"}</small></div>{signedIn && <button className="button" onClick={() => void logout()}>Sign out</button>}{!signedIn && <button className="button" onClick={() => setPage("Profile")}>Sign in</button>}</div>
      </header>
      {content}
      <div className="copyright">? {new Date().getFullYear()} TerraMind AI<span>Plant health and agricultural guidance</span></div>
    </div>
  </main>;
}
