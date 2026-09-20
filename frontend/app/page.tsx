"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import { API_BASE_URL } from "../services/api";
import { Diagnosis, Prediction } from "../services/diagnosis";
import { useDiagnosis } from "../hooks/useDiagnosis";
import { askTerraMind, ChatMessage, ChatResponse } from "../services/chat";
import { getDiagnosisById, listDiagnosisHistory, LocalHistoryItem } from "../services/history";
import { ReportRecord, formatBytes, formatTimestamp, listReports, getReportDownloadUrl } from "../services/reports";
import {
  FinalConfig,
  FullReport,
  ModelSummary,
  ClassMetric,
  formatScalar,
  getAnalyticsFinalConfig,
  getAnalyticsFullReport,
  getAnalyticsSummary,
  getGraphUrl,
  pct,
} from "../services/analytics";
import {
  AgentStatus,
  DashboardAggregate,
  RagStats,
  aggregateHistory,
  getAgentStatus,
  getRagStats,
  mapCategoryCounts,
} from "../services/dashboard";

const nav = [
  { group: "MAIN", items: [["Dashboard", "▦"], ["Plant Diagnosis", "⌁"]] },
  { group: "DIAGNOSIS", items: [["Upload Image", "↥"], ["Batch Detection", "▤"], ["Live Detection", "◉"]] },
  { group: "AI ASSISTANT", items: [["AI Chat Assistant", "✧"]] },
  { group: "RECOMMENDATIONS", items: [["Treatment Advisor", "♧"], ["Fertilizer Guide", "✚"], ["Care Recommendations", "♡"]] },
  { group: "KNOWLEDGE & AI", items: [["RAG Knowledge Base", "▱"], ["Agent Orchestrator", "⌘"], ["Agent Workflow", "⌘"]] },
  { group: "ANALYTICS", items: [["Model Performance", "◒"], ["Model Insights", "⌗"], ["System Architecture", "◇"]] },
  { group: "HISTORY & REPORTS", items: [["History", "◴"], ["Reports", "▤"]] },
  { group: "SETTINGS", items: [["Settings", "⚙"], ["Profile", "♙"]] },
];

function Icon({ children }: { children: React.ReactNode }) { return <span className="icon">{children}</span>; }
function Button({ children, primary = false, onClick, type = "button" }: { children: React.ReactNode; primary?: boolean; onClick?: () => void; type?: "button" | "submit" }) {
  return <button type={type} onClick={onClick} className={primary ? "button primary" : "button"}>{children}</button>;
}
function Card({ children, className = "", style }: { children: React.ReactNode; className?: string; style?: React.CSSProperties }) { return <section className={`card ${className}`} style={style}>{children}</section>; }
function Sparkline({ color = "#65d879", points = "0,28 18,23 35,26 51,12 68,19 84,8 102,17" }: { color?: string; points?: string }) {
  return <svg className="sparkline" viewBox="0 0 102 32" preserveAspectRatio="none"><polyline points={points} fill="none" stroke={color} strokeWidth="2" /></svg>;
}

function Metric({ title, value, change, tone, icon }: { title: string; value: string; change: string; tone: string; icon: string }) {
  return <Card className="metric"><div className="metric-top"><span>{title}</span><span className={`metric-icon ${tone}`}>{icon}</span></div><strong>{value}</strong><small className={change.startsWith("↓") ? "down" : ""}>{change} <em>from last month</em></small><Sparkline color={tone === "red" ? "#ef645d" : tone === "gold" ? "#e9b83e" : tone === "blue" ? "#5ba7e6" : "#65d879"} /></Card>;
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
  const [modelSummary, setModelSummary] = useState<ModelSummary | null>(null);
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

  const [todayLabel] = useState(() =>
    new Date().toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" })
  );
  const [lastUpdated, setLastUpdated] = useState<string>(() =>
    new Date().toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })
  );

  async function loadDashboard(abort?: AbortSignal) {
    setLastUpdated(new Date().toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }));
    const pModel = getAnalyticsSummary();
    const pHistory = listDiagnosisHistory(undefined, 500);
    const pRag = getRagStats();
    const pAgent = getAgentStatus();
    setModelLoading(true);
    setHistoryLoading(true);
    setRagLoading(true);
    setAgentLoading(true);
    try {
      const [m, h, r, a] = await Promise.all([
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

  const ringValue = modelSummary
    ? (modelSummary.test_accuracy ?? modelSummary.val_accuracy ?? modelSummary.train_accuracy ?? 0)
    : 0.6002;
  const ringPercent = Math.round(ringValue * 10000) / 100;

  const totalDiagnoses = Math.max(agg.total, agentStatus?.diagnoses.total ?? 0, agentStatus?.diagnoses.local_reports ?? 0);
  const totalReports = Math.max(agg.reports_count, agentStatus?.reports.supabase_reports ?? 0, totalDiagnoses);
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
            Welcome back, Pankaj! <span>👋</span>
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
          value={historyLoading ? "…" : `${avgConfidencePct.toFixed(1)}%`}
          change={confChange}
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
            <p className="error">Could not load history: {historyError}</p>
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
                  <b>{pct(modelSummary?.val_accuracy, 2)}</b>
                </p>
                <p>
                  <span>Test Macro F1</span>
                  <b>{pct(modelSummary?.test_macro_f1 ?? modelSummary?.test_weighted_f1, 2)}</b>
                </p>
                <p>
                  <span>Val Macro F1</span>
                  <b>{pct(modelSummary?.val_macro_f1 ?? modelSummary?.val_weighted_f1, 2)}</b>
                </p>
                <p>
                  <span>Total Classes</span>
                  <b>{String(modelSummary?.num_classes ?? 89)}</b>
                </p>
              </div>
              <small className="model-label">
                ● Model: {modelSummary?.backbone ?? modelSummary?.model_name ?? "EfficientNet-B0 v3"}
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
            onError={(e) => {
              (e.currentTarget as HTMLImageElement).style.display = "none";
              const fallback = (e.currentTarget as HTMLImageElement).nextElementSibling as HTMLDivElement | null;
              if (fallback) fallback.style.display = "grid";
            }}
          />
          <div className="matrix-box" style={{ display: "none" }}>
            {Array.from({ length: 64 }).map((_, i) => (
              <i key={i} style={{ opacity: ((i * 17) % 10) / 12 + 0.12 }} />
            ))}
          </div>
          <div className="matrix-axis">
            1　 2　 3　 4　 5　　　　　　　　　{modelSummary?.num_classes ?? 89}
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
              <p className="subtle">
                Supervisor Agent Status: <b>{agentStatus?.supervisor.state === "active" ? "Active" : agentStatus?.supervisor.state === "standby" ? "Standby" : agentError ? "Error" : "Idle"}</b>
              </p>
              <div className="agent-flow">
                <b>♧ Supervisor Agent</b>
                <div>
                  <span>
                    ◉
                    <small>
                      Vision Agent
                      <br />
                      {agentStatus?.supervisor.inference_ready ? "Active" : "Waiting"}
                    </small>
                  </span>
                  <span>
                    ▱
                    <small>
                      Research Agent
                      <br />
                      {ragLoading ? "…" : ragStats?.indexed ? "Active" : "Idle"}
                    </small>
                  </span>
                  <span>
                    ♧
                    <small>
                      Treatment Agent
                      <br />
                      {agentStatus?.supervisor.ready ? "Active" : "Idle"}
                    </small>
                  </span>
                  <span>
                    ▤
                    <small>
                      Report Agent
                      <br />
                      {totalReports > 0 ? "Active" : "Idle"}
                    </small>
                  </span>
                </div>
              </div>
              <div className="progress-line">
                <span>Workflow Status</span>
                <i>
                  <b style={{ width: `${agentStatus?.progress.percent ?? 0}%` }} />
                </i>
                <strong>{agentStatus?.progress.percent ?? 0}%</strong>
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
              {modelLoading ? "Loading…" : modelError ? "Unavailable" : modelSummary?.backbone ?? modelSummary?.model_name ?? "EfficientNet-B0"}
            </small>
          </p>
          <p>
            <span>◉　API Server</span>
            <small className="connected">Active</small>
          </p>
          <p>
            <span>◉　Database</span>
            <small className={agentStatus?.backend.supabase_enabled ? "connected" : undefined}>
              {agentLoading ? "…" : agentStatus?.backend.supabase_enabled ? "Connected" : "Local only"}
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
            Last Updated: {todayLabel} {lastUpdated}　⟳
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
            <p className="error">Could not load reports: {historyError}</p>
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
              const rough = Math.max(1024, Math.round(JSON.stringify(top).length));
              const sizeKb = Math.round(rough / 1024);
              const sizeStr = sizeKb > 1024 ? `${(sizeKb / 1024).toFixed(1)} MB` : `${sizeKb} KB`;
              return (
                <div className="report-preview">
                  <span>{lbl.label.toLowerCase().includes("healthy") ? "🌿" : "🍂"}</span>
                  <div>
                    <b>{lbl.label} Report</b>
                    <small>
                      {ts.date} · {ts.time}
                    </small>
                    <small>
                      JSON · {sizeStr} · {(topConf * 100).toFixed(1)}% conf.
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
                      {item.count || ragCounts.other ? `${item.count + (i === 3 ? ragCounts.other : 0)} Chunks　›` : "Indexed　›"}
                    </small>
                  </b>
                </div>
              ))}
              <p className="subtle" style={{ marginTop: 6, padding: "0 4px" }}>
                {ragStats?.total_points || 0} chunks · {ragStats?.sources_indexed || 0} sources ·{" "}
                {formatBytes(ragStats?.tfidf_size_bytes)}
              </p>
            </>
          )}
        </Card>
      </div>
    </div>
  );
}

function Upload({ batch = false }: { batch?: boolean }) {
  const [files, setFiles] = useState<File[]>([]); const [previews, setPreviews] = useState<string[]>([]);
  const { result, loading, error, run } = useDiagnosis();
  const select = (e: ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(e.target.files || []);
    previews.forEach((preview) => URL.revokeObjectURL(preview));
    setFiles(selected);
    setPreviews(selected.map((file) => URL.createObjectURL(file)));
  };
  useEffect(() => () => previews.forEach((preview) => URL.revokeObjectURL(preview)), [previews]);
  function submit(e: FormEvent) { e.preventDefault(); void run(files, batch); }
  const list = result ? (Array.isArray(result) ? result : [result]) : [];
  return <div className="page-content inner-page"><div className="section-title"><div><h1>{batch ? "Batch Detection" : "Plant Diagnosis"}</h1><p>{batch ? "Analyze multiple plant images in one real inference request." : "Upload a clear plant image for an AI-powered diagnosis."}</p></div></div><Card className="upload-card"><form onSubmit={submit}><label className="dropzone"><input type="file" accept="image/*" multiple={batch} onChange={select} /><span className="upload-icon">↥</span><b>{files.length ? `${files.length} image${files.length > 1 ? "s" : ""} selected` : "Drop your plant image here"}</b><small>or click to browse · JPG, PNG, WEBP up to 16 MB</small></label>{files.length > 0 && <div className="file-list">{files.map((f, i) => <span key={`${f.name}-${i}`}>{previews[i] && <img src={previews[i]} alt="" />} ◉ {f.name}</span>)}</div>}<Button primary type="submit">{loading ? "Analyzing…" : batch ? "Run Batch Detection" : "Analyze Image"}</Button>{error && <p className="error">API error: {error}</p>}</form></Card>{loading && <Card className="loading-state">Running real MobileNetV3 inference…</Card>}{list.length > 0 && <div className="result-grid">{list.map((d, i) => <Card className="prediction-card" key={`${d.id}-${i}`}>{previews[i] && <img className="result-image" src={previews[i]} alt={d.source_filename || "Uploaded plant"} />}<div className="card-heading"><h3>{d.source_filename || `Image ${i + 1}`}</h3><span className="confidence">{(d.predicted_class.confidence * 100).toFixed(2)}%</span></div>{d.low_confidence && <p className="low-confidence">Low confidence result — try a clearer, closer leaf image.</p>}<h2>{d.predicted_class.label}</h2><p className="subtle">Real model inference · {d.inference_ms} ms</p><div className="model-info"><b>{d.model_info.backbone}</b><span>{d.model_info.num_classes} classes · {d.model_info.img_size}px input</span></div><div className="top-list">{d.top5.map((p, n) => <p key={p.label}><span>{n + 1}. {p.label}</span><b>{(p.confidence * 100).toFixed(2)}%</b><i><em style={{ width: `${p.confidence * 100}%` }} /></i></p>)}</div></Card>)}</div>}</div>;
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
      size: 4096,
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

function ModelPerformancePage({ go }: { go: (page: string) => void }) {
  const [summary, setSummary] = useState<ModelSummary | null>(null);
  const [config, setConfig] = useState<FinalConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [graphError, setGraphError] = useState<{ training?: boolean; confusion?: boolean }>({});

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [s, c] = await Promise.all([getAnalyticsSummary(), getAnalyticsFinalConfig()]);
      setSummary(s);
      setConfig(c);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load analytics data.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void load(); }, []);

  const backbone = summary?.backbone ?? config?.backbone ?? summary?.model_name ?? config?.model_name ?? "EfficientNet-B0 v3";
  const trainAcc = summary?.train_accuracy;
  const valAcc = summary?.val_accuracy;
  const testAcc = summary?.test_accuracy;
  const trainLoss = summary?.train_loss;
  const valLoss = summary?.val_loss;
  const testLoss = summary?.test_loss;
  const trainF1 = summary?.train_macro_f1 ?? summary?.train_weighted_f1;
  const valF1 = summary?.val_macro_f1 ?? summary?.val_weighted_f1;
  const testF1 = summary?.test_macro_f1 ?? summary?.test_weighted_f1;

  return (
    <div className="page-content inner-page">
      <div className="section-title">
        <div>
          <h1>Model Performance</h1>
          <p>Real evaluation metrics from the archived {backbone} model. All values are computed from the test/validation splits.</p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <Button onClick={() => void load()}>⟳ Refresh</Button>
          <Button onClick={() => go("Model Insights")}>⌗ Class-level Insights</Button>
        </div>
      </div>

      {loading && (
        <Card className="loading-state"><p>Loading model performance analytics…</p></Card>
      )}

      {!loading && error && (
        <Card>
          <p className="error">Failed to load analytics: {error}</p>
          <Button onClick={() => void load()}>Retry</Button>
        </Card>
      )}

      {!loading && !error && summary && (
        <>
          <div className="metrics" style={{ gridTemplateColumns: "repeat(3,1fr)", marginBottom: 10 }}>
            <Metric title="Train Accuracy" value={pct(trainAcc)} change={formatScalar(trainLoss, 4)} tone="green" icon="▥" />
            <Metric title="Validation Accuracy" value={pct(valAcc)} change={formatScalar(valLoss, 4)} tone="gold" icon="▥" />
            <Metric title="Test Accuracy" value={pct(testAcc)} change={formatScalar(testLoss, 4)} tone="blue" icon="▥" />
            <Metric title="Train Macro F1" value={pct(trainF1)} change={trainF1 ? "Macro averaged" : "—"} tone="green" icon="✦" />
            <Metric title="Validation Macro F1" value={pct(valF1)} change={valF1 ? "Macro averaged" : "—"} tone="gold" icon="✦" />
            <Metric title="Test Macro F1" value={pct(testF1)} change={testF1 ? "Macro averaged" : "—"} tone="blue" icon="✦" />
          </div>

          <div className="dashboard-grid">
            <Card>
              <div className="card-heading">
                <h3>Training &amp; Validation Curves</h3>
                <small className="subtle">{config?.epochs ? `${config.epochs} epochs` : "—"} · best epoch {String(summary?.best_epoch ?? "—")}</small>
              </div>
              {graphError.training
                ? <p className="error">Training curves image is unavailable.</p>
                : (
                  <img
                    alt="Training Curves"
                    src={getGraphUrl("training_curves")}
                    style={{ width: "100%", height: "auto", borderRadius: 6, display: "block", marginTop: 4 }}
                    onError={() => setGraphError((g) => ({ ...g, training: true }))}
                  />
                )}
            </Card>

            <Card>
              <div className="card-heading">
                <h3>Confusion Matrix (89 Classes)</h3>
                <small className="subtle">Dataset: {summary?.dataset ?? config?.dataset ?? "PlantWild"}</small>
              </div>
              {graphError.confusion
                ? <p className="error">Confusion matrix image is unavailable.</p>
                : (
                  <img
                    alt="Confusion Matrix"
                    src={getGraphUrl("confusion_matrix")}
                    style={{ width: "100%", height: "auto", borderRadius: 6, display: "block", marginTop: 4 }}
                    onError={() => setGraphError((g) => ({ ...g, confusion: true }))}
                  />
                )}
            </Card>

            <Card style={{ gridColumn: "1 / -1" }}>
              <div className="card-heading"><h3>Summary Metrics</h3><small className="subtle">All splits combined</small></div>
              <div className="table">
                <div className="table-head">
                  <span>Split</span>
                  <span>Accuracy</span>
                  <span>Macro F1</span>
                  <span>Weighted F1</span>
                  <span>Loss</span>
                  <span>Samples</span>
                </div>
                {[
                  ["Train", trainAcc, summary?.train_macro_f1, summary?.train_weighted_f1, trainLoss, summary?.train_samples],
                  ["Validation", valAcc, summary?.val_macro_f1, summary?.val_weighted_f1, valLoss, summary?.val_samples],
                  ["Test", testAcc, summary?.test_macro_f1, summary?.test_weighted_f1, testLoss, summary?.test_samples],
                ].map((row) => (
                  <div className="table-row" key={String(row[0])} style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr 1fr 1fr" }}>
                    <b>{String(row[0])}</b>
                    <span>{pct(row[1] as number | undefined)}</span>
                    <span>{pct(row[2] as number | undefined)}</span>
                    <span>{pct(row[3] as number | undefined)}</span>
                    <span>{formatScalar(row[4] as number | undefined, 4)}</span>
                    <span>{row[5] !== undefined ? String(row[5]) : "—"}</span>
                  </div>
                ))}
              </div>
            </Card>

            <Card>
              <div className="card-heading"><h3>Model &amp; Dataset</h3></div>
              <p><span>Backbone</span><b>{backbone}</b></p>
              <p><span>Total Classes</span><b>{String(summary?.num_classes ?? config?.num_classes ?? 89)}</b></p>
              <p><span>Input Size</span><b>{String(summary?.img_size ?? config?.img_size ?? "?")} px</b></p>
              <p><span>Best Epoch</span><b>{String(summary?.best_epoch ?? "—")}</b></p>
              <p><span>Train Samples</span><b>{String(summary?.train_samples ?? "—")}</b></p>
              <p><span>Val Samples</span><b>{String(summary?.val_samples ?? "—")}</b></p>
              <p><span>Test Samples</span><b>{String(summary?.test_samples ?? "—")}</b></p>
              <p><span>Trainable Params</span><b>{summary?.total_trainable_params ? `${(summary.total_trainable_params / 1_000_000).toFixed(2)} M` : "—"}</b></p>
            </Card>

            <Card>
              <div className="card-heading"><h3>Training Config</h3></div>
              <p><span>Optimizer</span><b>{config?.optimizer ?? "—"}</b></p>
              <p><span>Learning Rate</span><b>{config?.learning_rate ? formatScalar(config.learning_rate, 6) : "—"}</b></p>
              <p><span>Batch Size</span><b>{String(config?.batch_size ?? "—")}</b></p>
              <p><span>Max Epochs</span><b>{String(config?.epochs ?? "—")}</b></p>
              <p><span>Loss Function</span><b>{config?.loss_function ?? "—"}</b></p>
              <p><span>Scheduler</span><b>{config?.scheduler ?? "—"}</b></p>
              <p><span>Weight Decay</span><b>{config?.weight_decay ? formatScalar(config.weight_decay, 6) : "—"}</b></p>
              <p><span>Augmentation</span><b>{config?.use_augmentation ? "Enabled" : "—"}</b></p>
              <p><span>Seed</span><b>{String(config?.seed ?? "—")}</b></p>
              <p><span>Device</span><b>{config?.device ?? "—"}</b></p>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

function ClassMetricRow({ cls, rank }: { cls: ClassMetric; rank: number }) {
  const className = cls.class ?? cls.label ?? "Unknown";
  const support = cls.support ?? cls.samples ?? 0;
  const p = cls.precision;
  const r = cls.recall;
  const f1 = cls.f1 ?? cls.f1_score;
  return (
    <div className="table-row" style={{ gridTemplateColumns: "40px 2fr 1fr 1fr 1fr 80px" }}>
      <b style={{ color: "#5ed36f" }}>{rank}</b>
      <span style={{ fontWeight: 500 }}>{className.replace(/[_-]/g, " ")}</span>
      <span>{pct(p)}</span>
      <span>{pct(r)}</span>
      <span>{pct(f1)}</span>
      <b style={{ color: "#79bd84" }}>{support}</b>
    </div>
  );
}

function ModelInsightsPage({ go }: { go: (page: string) => void }) {
  const [report, setReport] = useState<FullReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<"f1" | "precision" | "recall" | "support">("f1");
  const [filter, setFilter] = useState("");
  const [showWorst, setShowWorst] = useState(false);
  const [graphError, setGraphError] = useState(false);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      setReport(await getAnalyticsFullReport());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load per-class insights.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void load(); }, []);

  const perClass: ClassMetric[] = useMemo(() => {
    if (!report) return [];
    if (Array.isArray(report.per_class) && report.per_class.length > 0) {
      return report.per_class.map((c) => ({ ...c, class: c.class ?? c.label ?? "", support: c.support ?? c.samples, f1: c.f1 ?? c.f1_score }));
    }
    if (report.classification_report) {
      return Object.entries(report.classification_report)
        .filter(([k]) => !["accuracy", "macro avg", "weighted avg"].includes(k))
        .map(([klass, v]) => ({ class: klass, label: klass, ...v, support: v.support, f1: v.f1 ?? v.f1_score }));
    }
    return [];
  }, [report]);

  const processed: ClassMetric[] = useMemo(() => {
    const filtered = filter
      ? perClass.filter((c) => (c.class ?? "").toLowerCase().includes(filter.toLowerCase()))
      : perClass.slice();
    filtered.sort((a, b) => {
      const av = sortBy === "f1" ? (a.f1 ?? a.f1_score ?? 0) : sortBy === "precision" ? (a.precision ?? 0) : sortBy === "recall" ? (a.recall ?? 0) : (a.support ?? a.samples ?? 0);
      const bv = sortBy === "f1" ? (b.f1 ?? b.f1_score ?? 0) : sortBy === "precision" ? (b.precision ?? 0) : sortBy === "recall" ? (b.recall ?? 0) : (b.support ?? b.samples ?? 0);
      return showWorst ? av - bv : bv - av;
    });
    return filtered;
  }, [perClass, sortBy, filter, showWorst]);

  const headCounts = useMemo(() => {
    if (perClass.length === 0) return { above80: 0, above60: 0, below40: 0, avgF1: 0, zeroF1: 0 };
    let above80 = 0;
    let above60 = 0;
    let below40 = 0;
    let zeroF1 = 0;
    let sum = 0;
    for (const c of perClass) {
      const v = (c.f1 ?? c.f1_score ?? 0) * 100;
      if (v <= 0) zeroF1++;
      if (v >= 80) above80++;
      if (v >= 60) above60++;
      if (v < 40) below40++;
      sum += v;
    }
    return { above80, above60, below40, avgF1: sum / perClass.length, zeroF1 };
  }, [perClass]);

  return (
    <div className="page-content inner-page">
      <div className="section-title">
        <div>
          <h1>Model Insights</h1>
          <p>Per-class precision, recall, and F1 — identify which classes your model nails and which need more data.</p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <Button onClick={() => void load()}>⟳ Refresh</Button>
          <Button onClick={() => go("Model Performance")}>◒ Overview</Button>
        </div>
      </div>

      {loading && (
        <Card className="loading-state"><p>Loading per-class model insights…</p></Card>
      )}

      {!loading && error && (
        <Card>
          <p className="error">Failed to load insights: {error}</p>
          <Button onClick={() => void load()}>Retry</Button>
        </Card>
      )}

      {!loading && !error && report && (
        <>
          <div className="metrics" style={{ gridTemplateColumns: "repeat(5,1fr)", marginBottom: 10 }}>
            <Metric title="Classes Evaluated" value={String(perClass.length)} change={perClass.length ? "Per-class report" : "—"} tone="green" icon="⌗" />
            <Metric title="Avg. Per-Class F1" value={`${headCounts.avgF1.toFixed(1)}%`} change="Unweighted mean" tone="blue" icon="✦" />
            <Metric title="Classes ≥ 80% F1" value={String(headCounts.above80)} change={`${perClass.length ? ((headCounts.above80 / perClass.length) * 100).toFixed(0) : 0}% of set`} tone="green" icon="♧" />
            <Metric title="Classes ≥ 60% F1" value={String(headCounts.above60)} change={`${perClass.length ? ((headCounts.above60 / perClass.length) * 100).toFixed(0) : 0}% of set`} tone="gold" icon="▥" />
            <Metric title="Classes &lt; 40% F1" value={String(headCounts.below40)} change={`${headCounts.zeroF1} with 0% F1`} tone="red" icon="✣" />
          </div>

          <div className="dashboard-grid">
            <Card style={{ gridColumn: "1 / -1" }}>
              <div className="card-heading">
                <h3>Confusion Matrix (Full)</h3>
                <small className="subtle">Diagonal elements = correct predictions</small>
              </div>
              {graphError
                ? <p className="error">Confusion matrix image is unavailable.</p>
                : (
                  <img
                    alt="Confusion Matrix Full"
                    src={getGraphUrl("confusion_matrix")}
                    style={{ width: "100%", height: "auto", borderRadius: 6, display: "block", marginTop: 4 }}
                    onError={() => setGraphError(true)}
                  />
                )}
            </Card>

            <Card style={{ gridColumn: "1 / -1" }}>
              <div className="card-heading">
                <h3>Per-Class Performance ({processed.length} of {perClass.length})</h3>
                <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                  <input
                    placeholder="Search class…"
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                    style={{ padding: "6px 10px", borderRadius: 4, border: "1px solid #274c37", background: "#0c1d13", color: "#d6e9db", fontSize: 12 }}
                  />
                  <select
                    className="button"
                    style={{ cursor: "pointer" }}
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
                  >
                    <option value="f1">Sort by F1</option>
                    <option value="precision">Sort by Precision</option>
                    <option value="recall">Sort by Recall</option>
                    <option value="support">Sort by Support</option>
                  </select>
                  <button
                    className={showWorst ? "button primary" : "button"}
                    onClick={() => setShowWorst((v) => !v)}
                  >
                    {showWorst ? "Worst ↑" : "Best ↓"}
                  </button>
                </div>
              </div>
              {processed.length === 0 ? (
                <p className="subtle">No classes match your filter.</p>
              ) : (
                <div className="table">
                  <div className="table-head">
                    <span>#</span>
                    <span>Class</span>
                    <span>Precision</span>
                    <span>Recall</span>
                    <span>F1</span>
                    <span>Support</span>
                  </div>
                  {processed.slice(0, 89).map((cls, rank) => (
                    <ClassMetricRow cls={cls} rank={rank + 1} key={cls.class ?? rank} />
                  ))}
                </div>
              )}
              {processed.length > 89 && (
                <p className="subtle" style={{ marginTop: 8 }}>Showing top 89 of {processed.length} classes.</p>
              )}
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

function GenericPage({ name, go }: { name: string; go: (x: string) => void }) {
  const descriptions: Record<string, string> = { "AI Chat Assistant": "Ask TerraMind about plant health, treatment, and farm care.", "Treatment Advisor": "Evidence-based treatment guidance will be grounded in your diagnosis.", "Fertilizer Guide": "Find nutrient plans tailored to crop and growth stage.", "Care Recommendations": "Practical care recommendations for healthier plants.", "RAG Knowledge Base": "Trusted agricultural documents and retrieval sources.", "Agent Orchestrator": "Coordinate specialized TerraMind agents in one workflow.", "Agent Workflow": "Monitor the active diagnosis workflow from vision to report.", "Model Performance": "Real evaluation metrics from the archived EfficientNet-B0 v3 model.", "Model Insights": "Explore class-level performance and inference insights.", "System Architecture": "TerraMind service health and component architecture.", History: "Your saved real diagnosis history.", Reports: "Generated diagnosis reports and downloadable summaries.", Settings: "Manage API connection and application preferences.", Profile: "Pankaj Yadav · Premium Plan", "Live Detection": "Camera-based real-time detection is ready for a connected stream." };
  return <div className="page-content inner-page"><div className="section-title"><div><h1>{name}</h1><p>{descriptions[name] || "TerraMind AI workspace"}</p></div></div><Card className="empty-state"><span className="empty-icon">◇</span><h2>{name === "History" ? "No recent history in this workspace" : `${name} workspace`}</h2><p>Connect this view to the live TerraMind API as data becomes available. Your existing backend and model remain unchanged.</p></Card></div>;
}

export default function Home() {
  const [page, setPage] = useState("Dashboard"); const [menu, setMenu] = useState(false); const [health, setHealth] = useState("Checking");
  useEffect(() => { fetch(`${API_BASE_URL}/health`).then((r) => r.ok ? setHealth("Operational") : setHealth("Offline")).catch(() => setHealth("Offline")); }, []);
  const content = useMemo(() => page === "Dashboard" ? <Dashboard go={setPage} /> : page === "Upload Image" ? <Upload /> : page === "Batch Detection" ? <Upload batch /> : page === "AI Chat Assistant" ? <ChatAssistant /> : page === "History" ? <HistoryPage go={setPage} /> : page === "Reports" ? <ReportsPage go={setPage} /> : page === "Model Performance" ? <ModelPerformancePage go={setPage} /> : page === "Model Insights" ? <ModelInsightsPage go={setPage} /> : <GenericPage name={page} go={setPage} />, [page]);
  return <main className="app-shell"><aside className={`sidebar ${menu ? "open" : ""}`}><div className="brand"><span>✦</span><b>TerraMind <em>AI</em></b></div><div className="nav">{nav.map((section) => <div key={section.group}><label>{section.group}</label>{section.items.map(([name, icon]) => <button className={page === name ? "active" : ""} key={name} onClick={() => { setPage(name === "Plant Diagnosis" ? "Upload Image" : name); setMenu(false); }}><Icon>{icon}</Icon>{name}{name === "Plant Diagnosis" && <small>⌄</small>}</button>)}</div>)}</div><div className="sidebar-status"><span>◉</span><div><b>System Status</b><small>{health === "Operational" ? "All Systems Operational" : health}</small></div></div><footer>v1.0.0</footer></aside>{menu && <button className="sidebar-backdrop" aria-label="Close navigation" onClick={() => setMenu(false)} />}<div className="workspace"><header className="topbar"><button className="hamburger" aria-label="Toggle navigation" aria-expanded={menu} onClick={() => setMenu(!menu)}>☰</button><label className="search"><span aria-hidden="true">⌕</span><input aria-label="Search" placeholder="Search anything…" /><kbd>Ctrl + K</kbd></label><div className="top-actions"><button aria-label="Notifications">♧<sup>3</sup></button><button aria-label="Toggle theme">☾</button><button aria-label="Enter fullscreen">⛶</button><div className="user-avatar">P</div><div className="user-name"><b>Pankaj Yadav</b><small>Premium Plan⌄</small></div></div></header>{content}<div className="copyright">© 2025 TerraMind AI, all rights reserved.<span>Built with <b>♥</b> for Smarter Agriculture ✦</span></div></div></main>;
}
