"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import { Session } from "@supabase/supabase-js";
import { API_BASE_URL } from "../services/api";
import { getAnalyticsSummary, getGraphUrl, pct } from "../services/analytics";
import { AgentDiagnosis, AgentEvent, listLocalHistory, runAgentDiagnosis } from "../services/agents";
import { getAgentStatus, getRagStats, RagStats, AgentStatus } from "../services/dashboard";
import { retrieveKnowledge, RagRetrieval, RetrievedSource } from "../services/rag";
import { getHealth, getServingModelInfo, getSupabaseStatus, getModelLabels, getModelRegistryStatus, getLlmStatus, HealthResponse, ServingModelInfo, SupabaseStatus, ModelRegistryStatus, LlmStatus } from "../services/system";
import { signIn, signOut, signUp } from "../services/auth";
import { supabase } from "../services/supabase";
import { analyzeMultimodalImage, getMultimodalStatus, MultimodalAnalysis, MultimodalStatus } from "../services/multimodal";
import {
  analyzeFusionImage,
  getFusionStatus,
  FusionAnalysis,
  FusionKind,
  FusionStatus,
} from "../services/fusion";


function Card({ children, className = "", style }: { children: React.ReactNode; className?: string; style?: React.CSSProperties }) {
  return <section className={`card ${className}`} style={style}>{children}</section>;
}

function Button({ children, primary = false, type = "button", onClick, disabled = false }: { children: React.ReactNode; primary?: boolean; type?: "button" | "submit"; onClick?: () => void; disabled?: boolean }) {
  return <button type={type} className={primary ? "button primary" : "button"} onClick={onClick} disabled={disabled}>{children}</button>;
}

function PageTitle({ title, description, children }: { title: string; description: string; children?: React.ReactNode }) {
  return <div className="section-title"><div><h1>{title}</h1><p>{description}</p></div>{children && <div className="page-actions">{children}</div>}</div>;
}

function RequestState({ loading, error, empty, children }: { loading: boolean; error: string | null; empty?: boolean; children?: React.ReactNode }) {
  if (loading) return <Card className="loading-state"><p>Loading live data…</p></Card>;
  if (error) return <Card><p className="error">{error}</p></Card>;
  if (empty) return <Card className="subtle">No matching backend data is available yet.</Card>;
  return <>{children}</>;
}

function SourceList({ sources }: { sources: RetrievedSource[] }) {
  if (!sources.length) return <Card className="empty-state"><span className="empty-icon">▱</span><h2>No sources retrieved</h2><p>Try a crop name, disease, or a more specific question. Recommendations only appear when the RAG API returns supporting sources.</p></Card>;
  return <div className="source-grid">{sources.map((source, index) => {
    const norm = (source.source || "").replace(/\\/g, "/");
    let category = source.category || "Agricultural Reference";
    if (norm.includes("/diseases/")) category = "Pathogen Profile";
    else if (norm.includes("/treatment/")) category = "Treatment Guideline";
    else if (norm.includes("/fertilizer/")) category = "Fertilizer Guide";
    else if (norm.includes("/plant_care/")) category = "Plant Care Guide";
    else if (norm.includes("/agriculture/")) category = "Agronomy Extension";
    return (
      <Card className="source-card" key={`${source.source}-${source.title}-${index}`}>
        <div className="card-heading"><h3>[{index + 1}] {source.title || "Knowledge source"}</h3><span className="live">{category}</span></div>
        {typeof source.score === "number" && <small className="subtle">Relevance score: {source.score.toFixed(3)}</small>}
        <p>{source.text || source.excerpt || "No excerpt was returned by the RAG API."}</p>
        <small className="citation">Reference Type: {category}</small>
      </Card>
    );
  })}</div>;
}

const recommendationConfig: Record<string, { category: string; intro: string; hint: string }> = {
  "Treatment Advisor": { category: "treatment", intro: "Search treatment guidance retrieved from the TerraMind knowledge index.", hint: "Describe the plant, disease, and symptoms you want treatment guidance for…" },
  "Fertilizer Guide": { category: "fertilizer", intro: "Search fertilizer and nutrient guidance returned by RAG.", hint: "Enter a crop, growth stage, soil condition, or nutrient question…" },
  "Care Recommendations": { category: "plant care", intro: "Search plant care recommendations grounded in indexed sources.", hint: "Ask about watering, pruning, light, soil, or seasonal care…" },
};

export function RecommendationPage({ name }: { name: string }) {
  const config = recommendationConfig[name];
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<RagRetrieval | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function search(event: FormEvent) {
    event.preventDefault();
    const question = query.trim();
    if (question.length < 3 || loading) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await retrieveKnowledge(question, config.category, 8));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "RAG retrieval failed.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return <div className="page-content inner-page">
    <PageTitle title={name} description={config.intro} />
    <Card>
      <form className="query-form" onSubmit={search}>
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={config.hint} minLength={3} disabled={loading} />
        <Button primary type="submit" disabled={loading || query.trim().length < 3}>{loading ? "Searching…" : "Find sourced guidance"}</Button>
      </form>
      {error && <p className="error">RAG request failed: {error}</p>}
    </Card>
    {loading && <Card className="loading-state">Retrieving knowledge sources…</Card>}
    {!loading && !error && result && <>
      <p className="subtle">{result.grounded ? `${result.retrieved} grounded sources returned for “${result.query}”.` : "The knowledge API returned no supporting sources for this query."}</p>
      <SourceList sources={result.sources} />
    </>}
    {!loading && !error && !result && <Card className="empty-state"><span className="empty-icon">⌕</span><h2>Search the knowledge base</h2><p>Recommendations appear here only after the backend returns matching source passages.</p></Card>}
  </div>;
}

export function RagKnowledgeBasePage() {
  const [stats, setStats] = useState<RagStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const [result, setResult] = useState<RagRetrieval | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  async function loadStats() {
    setStatsLoading(true);
    setStatsError(null);
    try { setStats(await getRagStats()); }
    catch (cause) { setStatsError(cause instanceof Error ? cause.message : "Could not read RAG stats."); }
    finally { setStatsLoading(false); }
  }
  useEffect(() => { void loadStats(); }, []);

  async function search(event: FormEvent) {
    event.preventDefault();
    if (query.trim().length < 3 || searching) return;
    setSearching(true);
    setSearchError(null);
    try { setResult(await retrieveKnowledge(query.trim(), category || undefined, 10)); }
    catch (cause) { setResult(null); setSearchError(cause instanceof Error ? cause.message : "RAG retrieval failed."); }
    finally { setSearching(false); }
  }

  return <div className="page-content inner-page">
    <PageTitle title="RAG Knowledge Base" description="Live index statistics and source-backed retrieval from the backend knowledge pipeline."><Button onClick={() => void loadStats()}>Refresh stats</Button></PageTitle>
    {statsLoading ? <Card className="loading-state">Loading RAG index statistics…</Card> : statsError ? <Card><p className="error">{statsError}</p><Button onClick={() => void loadStats()}>Retry</Button></Card> : stats && <>
      <div className="metrics" style={{ gridTemplateColumns: "repeat(4,1fr)", marginBottom: 10 }}>
        <div className="card metric"><span>Index</span><strong>{stats.indexed ? "Ready" : "Empty"}</strong><small>{stats.backend || "Backend unavailable"}</small></div>
        <div className="card metric"><span>Indexed chunks</span><strong>{stats.total_points}</strong><small>{stats.collection || "Knowledge collection"}</small></div>
        <div className="card metric"><span>Source documents</span><strong>{stats.sources_indexed}</strong><small>Count reported by vector store</small></div>
        <div className="card metric"><span>TF-IDF model</span><strong>{stats.embeddings?.fitted ? "Fitted" : "Unavailable"}</strong><small>{stats.embeddings?.vocabulary_size ?? 0} vocabulary terms · {stats.tfidf_size_bytes} bytes</small></div>
      </div>
      <Card>
        <div className="card-heading"><h3>Indexed chunks by category</h3></div>
        {Object.keys(stats.by_category).length ? Object.entries(stats.by_category).map(([key, value]) => <p className="status-row" key={key}><span>{key}</span><b>{value}</b></p>) : <p className="subtle">The vector store returned no category counts.</p>}
      </Card>
    </>}
    <Card>
      <div className="card-heading"><h3>Retrieve sources</h3><span className="subtle">POST /api/rag/retrieve</span></div>
      <form className="query-form" onSubmit={search}>
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search indexed agricultural knowledge…" minLength={3} disabled={searching} />
        <select className="button" value={category} onChange={(event) => setCategory(event.target.value)} disabled={searching}>
          <option value="">All categories</option><option value="diseases">Diseases</option><option value="treatment">Treatment</option><option value="fertilizer">Fertilizer</option><option value="plant care">Plant care</option><option value="agriculture">Agriculture</option>
        </select>
        <Button primary type="submit" disabled={searching || query.trim().length < 3}>{searching ? "Searching…" : "Search"}</Button>
      </form>
      {searchError && <p className="error">{searchError}</p>}
    </Card>
    {searching && <Card className="loading-state">Retrieving matching chunks…</Card>}
    {!searching && !searchError && result && <><p className="subtle">{result.retrieved} results · {result.grounded ? "grounded" : "no supporting results"}</p><SourceList sources={result.sources} /></>}
  </div>;
}

function eventListFromHistory(item: Record<string, unknown> | null): AgentEvent[] {
  return Array.isArray(item?.agent_events) ? item.agent_events as AgentEvent[] : [];
}

export function AgentToolsPage({ mode }: { mode: "orchestrator" | "workflow" }) {
  const isWorkflow = mode === "workflow";
  const title = isWorkflow ? "Agent Workflow" : "Agent Orchestrator";
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [query, setQuery] = useState("");
  const [run, setRun] = useState<AgentDiagnosis | null>(null);
  const [previous, setPrevious] = useState<Record<string, unknown> | null>(null);
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    Promise.allSettled([getAgentStatus(), listLocalHistory(100)]).then(([statusResult, historyResult]) => {
      if (!alive) return;
      if (statusResult.status === "fulfilled") setStatus(statusResult.value);
      if (historyResult.status === "fulfilled") {
        const latest = historyResult.value.items.find((item) => Array.isArray((item as Record<string, unknown>).agent_events));
        if (latest) setPrevious(latest as unknown as Record<string, unknown>);
      }
      setLoadingHistory(false);
    });
    return () => { alive = false; };
  }, []);

  async function execute(event: FormEvent) {
    event.preventDefault();
    if (!selectedFile || running) return;
    setRunning(true);
    setError(null);
    try {
      const diagnosis = await runAgentDiagnosis(selectedFile, query.trim());
      setRun(diagnosis);
      setPrevious(diagnosis as unknown as Record<string, unknown>);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Agent diagnosis failed.");
    } finally { setRunning(false); }
  }

  const activeEvents = run?.agent_events ?? eventListFromHistory(previous);
  const agentResults = (run?.agent_results ?? previous?.agent_results ?? {}) as Record<string, Record<string, unknown>>;
  const research = agentResults.research_rag;
  const researchSources = Array.isArray(research?.sources) ? research.sources as RetrievedSource[] : [];

  return <div className="page-content inner-page">
    <PageTitle title={title} description={isWorkflow ? "Review the latest executed agent events or run an image through the supervisor workflow." : "Run image diagnosis through the real supervisor and specialist agents."}>
      <span className={status?.supervisor.ready ? "live" : "low-confidence"}>{loadingHistory ? "Checking…" : status?.supervisor.ready ? "Supervisor ready" : "Unavailable"}</span>
    </PageTitle>
    <Card>
      <form className="query-form agent-form" onSubmit={execute}>
        <label className="agent-file">Plant image<input type="file" accept="image/*" onChange={(event: ChangeEvent<HTMLInputElement>) => setSelectedFile(event.target.files?.[0] ?? null)} /></label>
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Optional question or crop context" disabled={running} />
        <Button primary type="submit" disabled={!selectedFile || running}>{running ? "Running agents…" : "Run agent workflow"}</Button>
      </form>
      {selectedFile && <p className="subtle">Selected: {selectedFile.name}</p>}
      {error && <p className="error">{error}</p>}
      {running && <p className="loading-state">Vision, research, specialist, and report agents are processing the image…</p>}
    </Card>
    {!loadingHistory && !run && !previous && <Card className="empty-state"><span className="empty-icon">⌘</span><h2>No workflow has been recorded yet</h2><p>Choose a plant image above to execute and record the real agent workflow.</p></Card>}
    {activeEvents.length > 0 && <>
      <Card>
        <div className="card-heading"><h3>Executed workflow</h3><small className="subtle">{run?.agent_workflow_id || String(previous?.agent_workflow_id || "Previously persisted diagnosis")}</small></div>
        <div className="agent-event-list">{activeEvents.map((event, index) => <div className="agent-event" key={`${event.agent_name}-${index}`}>
          <span className={event.status === "completed" ? "event-ok" : event.status === "failed" ? "event-failed" : "event-running"}>{event.status === "completed" ? "●" : event.status === "failed" ? "!" : "…"}</span>
          <div><b>{event.agent_name}</b><small>{event.status}{typeof event.duration_ms === "number" ? ` · ${event.duration_ms.toFixed(1)} ms` : ""}</small>{event.error && <small className="error">{event.error}</small>}</div>
        </div>)}</div>
      </Card>
      {(run || previous) && <Card>
        <div className="card-heading"><h3>Workflow result</h3><span className={(agentResults.report?.grounded ?? false) ? "live" : "subtle"}>{agentResults.report?.grounded ? "Grounded report" : "Report status returned"}</span></div>
        <p><b>Diagnosis:</b> {String((run ?? previous as unknown as AgentDiagnosis)?.predicted_class?.label ?? "—")}</p>
        <p><b>Confidence:</b> {typeof (run ?? previous as unknown as AgentDiagnosis)?.predicted_class?.confidence === "number" ? `${(((run ?? previous as unknown as AgentDiagnosis).predicted_class.confidence) * 100).toFixed(2)}%` : "—"}</p>
        {researchSources.length > 0 ? <SourceList sources={researchSources} /> : <p className="subtle">The research agent returned no source list for this run.</p>}
      </Card>}
    </>}
  </div>;
}

type LiveStatus = { health: HealthResponse | null; model: ServingModelInfo | null; rag: RagStats | null; agents: AgentStatus | null; supabase: SupabaseStatus | null; registry: ModelRegistryStatus | null; llm: LlmStatus | null };

export function SystemStatusPage() {
  const [data, setData] = useState<LiveStatus>({ health: null, model: null, rag: null, agents: null, supabase: null, registry: null, llm: null });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [checkedAt, setCheckedAt] = useState<string | null>(null);
  async function load() {
    setLoading(true);
    const requests = await Promise.allSettled([getHealth(), getServingModelInfo(), getRagStats(), getAgentStatus(), getSupabaseStatus(), getModelRegistryStatus(), getLlmStatus()]);
    const keys = ["health", "model", "rag", "agents", "supabase", "registry", "llm"] as const;
    const next = {} as LiveStatus;
    const nextErrors: Record<string, string> = {};
    requests.forEach((result, index) => {
      const key = keys[index];
      if (result.status === "fulfilled") next[key] = result.value as never;
      else { next[key] = null; nextErrors[key] = result.reason instanceof Error ? result.reason.message : "Request failed"; }
    });
    setData(next);
    setErrors(nextErrors);
    setCheckedAt(new Date().toLocaleString());
    setLoading(false);
  }
  useEffect(() => { void load(); }, []);

  const rows = [
    { name: "Backend API", value: data.health ? `${data.health.service} · v${data.health.version}` : "Unavailable", detail: data.health?.status ?? errors.health ?? "Not loaded", ok: data.health?.status === "ok" },
    { name: "Vision inference", value: data.model?.backbone ?? "Unavailable", detail: data.health?.inference_available ? `${data.model?.num_classes} classes · ${data.model?.img_size}px` : errors.model ?? "Inference unavailable", ok: Boolean(data.health?.inference_available && data.model) },
    { name: "RAG index", value: data.rag ? `${data.rag.total_points} chunks · ${data.rag.sources_indexed} sources` : "Unavailable", detail: data.rag ? `${data.rag.backend || "unknown backend"} · ${data.rag.indexed ? "indexed" : "empty"}` : errors.rag ?? "Not loaded", ok: Boolean(data.rag?.indexed) },
    { name: "Agent supervisor", value: data.agents?.supervisor.state ?? "Unavailable", detail: data.agents ? `${data.agents.supervisor.ready ? "Supervisor" : "Supervisor unavailable"} · ${data.agents.supervisor.chat_ready ? "Chat ready" : "Chat unavailable"}` : errors.agents ?? "Not loaded", ok: Boolean(data.agents?.supervisor.ready && data.agents.supervisor.chat_ready) },
    { name: "Supabase", value: data.supabase ? (data.supabase.configured ? "Configured" : "Not configured") : "Status unavailable", detail: data.supabase?.missing.length ? `Missing: ${data.supabase.missing.join(", ")}` : data.supabase ? "Configuration status only; connectivity is checked on requests." : errors.supabase ?? "Not loaded", ok: Boolean(data.supabase?.configured) },
    { name: "LLM provider", value: data.llm ? (data.llm.configured ? `${data.llm.provider} configured` : "Not configured") : "Status unavailable", detail: data.llm ? (data.llm.configured ? `Model: ${data.llm.model}` : "Chat uses cited evidence-template responses") : errors.llm ?? "Not loaded", ok: Boolean(data.llm?.configured) },
  ];

  return <div className="page-content inner-page">
    <PageTitle title="System Status" description="Current health from the TerraMind backend endpoints."><Button onClick={() => void load()} disabled={loading}>{loading ? "Checking…" : "Refresh"}</Button></PageTitle>
    {loading && <Card className="loading-state">Checking backend services…</Card>}
    {!loading && <><Card><div className="card-heading"><h3>Service health</h3><small className="subtle">Checked {checkedAt ?? "—"}</small></div>
      {rows.map((row) => <div className="status-detail" key={row.name}><div><b>{row.name}</b><small>{row.detail}</small></div><span className={row.ok ? "live" : "low-confidence"}>{row.value}</span></div>)}
    </Card><Card><div className="card-heading"><h3>Served model</h3></div>{data.model ? <><p><b>{data.model.backbone}</b></p><p>{data.model.num_classes} classes · {data.model.img_size}px · {data.model.preprocessing || "preprocessing metadata unavailable"}</p><p>Test accuracy {pct(data.model.test_acc)} · Test macro F1 {pct(data.model.test_macro_f1 ?? undefined)}</p></> : <p className="error">{errors.model || "Model information unavailable."}</p>}</Card>
    <Card><div className="card-heading"><h3>Model availability</h3><small className="subtle">Experimental weights remain unloaded</small></div>
      {data.registry ? <>
        <p className="settings-row"><span>{data.registry.production.name}</span><b>{data.registry.production.available_for_inference ? "Loaded · available" : "Unavailable"}</b></p>
        {data.registry.experimental_candidates.map((candidate) => <div key={candidate.kind} className="status-detail"><div><b>{candidate.name}</b><small>{candidate.reason}</small></div><span>{candidate.status}{candidate.metrics?.macro_f1 != null ? ` · duplicate-free test subset (n=${candidate.metrics.n}): macro F1 ${pct(candidate.metrics.macro_f1)}` : ""}</span></div>)}
        {data.registry.fusion_models.map((fusion) => <div key={fusion.kind} className="status-detail"><div><b>{fusion.name}</b><small>{fusion.reason || (fusion.requires_text ? "Requires valid paired text" : "")}</small></div><span>{fusion.status}{fusion.validation_macro_f1 != null ? ` · val macro F1 ${pct(fusion.validation_macro_f1)}` : ""}</span></div>)}
        {data.registry.text_only_baseline && <div className="status-detail"><div><b>{data.registry.text_only_baseline.name}</b><small>{data.registry.text_only_baseline.requires_text ? "Requires valid paired text" : ""}</small></div><span>{data.registry.text_only_baseline.status}{data.registry.text_only_baseline.validation_macro_f1 != null ? ` · val macro F1 ${pct(data.registry.text_only_baseline.validation_macro_f1)}` : ""}</span></div>}
      </> : <p className="error">{errors.registry || "Model registry status unavailable."}</p>}
    </Card></>}
  </div>;
}

export function SettingsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [supabaseStatus, setSupabaseStatus] = useState<SupabaseStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState("0.35");
  const [saved, setSaved] = useState(false);
  async function refresh() {
    setLoading(true); setError(null);
    const [healthResult, dbResult] = await Promise.allSettled([getHealth(), getSupabaseStatus()]);
    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    else setError(healthResult.reason instanceof Error ? healthResult.reason.message : "Backend status unavailable.");
    if (dbResult.status === "fulfilled") setSupabaseStatus(dbResult.value);
    setLoading(false);
  }
  useEffect(() => {
    if (typeof window !== "undefined") {
      const stored = window.localStorage.getItem("terramind.lowConfidenceThreshold");
      if (stored) setThreshold(stored);
    }
    void refresh();
  }, []);
  function save(event: FormEvent) {
    event.preventDefault();
    const parsed = Number(threshold);
    if (!Number.isFinite(parsed) || parsed < 0.05 || parsed > 0.95) { setError("Choose a confidence threshold between 0.05 and 0.95."); return; }
    window.localStorage.setItem("terramind.lowConfidenceThreshold", String(parsed));
    setThreshold(String(parsed)); setSaved(true); setError(null);
  }
  return <div className="page-content inner-page">
    <PageTitle title="Settings" description="Live backend configuration and the local diagnosis confidence preference."><Button onClick={() => void refresh()} disabled={loading}>Refresh status</Button></PageTitle>
    <Card><div className="card-heading"><h3>Backend connection</h3><span className={health?.status === "ok" ? "live" : "low-confidence"}>{loading ? "Checking…" : health?.status ?? "Unavailable"}</span></div>
      <p className="settings-row"><span>API base URL</span><b>{API_BASE_URL}</b></p><p className="settings-row"><span>Backend version</span><b>{health?.version ?? "—"}</b></p><p className="settings-row"><span>Vision inference</span><b>{health ? (health.inference_available ? "Available" : "Unavailable") : "—"}</b></p><p className="settings-row"><span>Supabase configuration</span><b>{supabaseStatus ? (supabaseStatus.configured ? "Configured" : `Local mode · ${supabaseStatus.missing.join(", ")}`) : "—"}</b></p>
      {error && <p className="error">{error}</p>}
    </Card>
    <Card><div className="card-heading"><h3>Diagnosis preference</h3><small className="subtle">Saved in this browser and sent with prediction requests</small></div>
      <form className="query-form settings-form" onSubmit={save}><label>Low-confidence threshold<input type="number" min="0.05" max="0.95" step="0.01" value={threshold} onChange={(event) => { setThreshold(event.target.value); setSaved(false); }} /></label><Button primary type="submit">Save preference</Button></form>
      {saved && <p className="live">Saved · predictions below this confidence will be marked low confidence.</p>}
    </Card>
  </div>;
}

export function ProfilePage() {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(Boolean(supabase));
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<"sign-in" | "sign-up">("sign-in");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    if (!supabase) { setLoading(false); return; }
    let active = true;
    supabase.auth.getSession().then(({ data, error: sessionError }) => {
      if (!active) return;
      if (sessionError) setError(sessionError.message);
      setSession(data.session);
      setLoading(false);
    }).catch((cause) => { if (active) { setError(cause instanceof Error ? cause.message : "Could not read account session."); setLoading(false); } });
    const { data } = supabase.auth.onAuthStateChange((_event, nextSession) => setSession(nextSession));
    return () => { active = false; data.subscription.unsubscribe(); };
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError(null); setNotice(null);
    try {
      const response = mode === "sign-in" ? await signIn(email, password) : await signUp(email, password, displayName);
      if (response.error) throw response.error;
      setSession(response.data.session);
      if (!response.data.session && mode === "sign-up") setNotice("Account created. Check your email if confirmation is enabled.");
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Authentication request failed."); }
    finally { setBusy(false); }
  }

  async function logout() {
    setBusy(true); setError(null);
    try { await signOut(); setSession(null); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Sign out failed."); }
    finally { setBusy(false); }
  }

  const user = session?.user;
  const name = String(user?.user_metadata?.full_name || user?.user_metadata?.name || "").trim();
  return <div className="page-content inner-page">
    <PageTitle title="Profile" description="Account details from the active authentication session." />
    {loading ? <Card className="loading-state">Loading account session…</Card> : !supabase ? <Card className="empty-state"><span className="empty-icon">♙</span><h2>Authentication is not configured</h2><p>Set the public Supabase URL and anon key to enable account sign-in. No profile data is being assumed.</p></Card> : user ? <Card>
      <div className="profile-head"><div className="user-avatar">{(name || user.email || "U").slice(0, 1).toUpperCase()}</div><div><h2>{name || user.email}</h2><p className="subtle">{user.email}</p></div></div>
      <p className="settings-row"><span>User ID</span><b>{user.id}</b></p><p className="settings-row"><span>Email verified</span><b>{user.email_confirmed_at ? "Yes" : "Not confirmed"}</b></p><p className="settings-row"><span>Created</span><b>{user.created_at ? new Date(user.created_at).toLocaleString() : "—"}</b></p>
      <Button onClick={() => void logout()} disabled={busy}>{busy ? "Signing out…" : "Sign out"}</Button>
    </Card> : <Card>
      <div className="card-heading"><h3>{mode === "sign-in" ? "Sign in" : "Create account"}</h3><Button onClick={() => { setMode(mode === "sign-in" ? "sign-up" : "sign-in"); setError(null); }}>{mode === "sign-in" ? "Create account" : "Back to sign in"}</Button></div>
      <form className="profile-form" onSubmit={submit}>
        {mode === "sign-up" && <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="Full name" autoComplete="name" />}
        <input type="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="Email address" autoComplete="email" required />
        <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Password" autoComplete={mode === "sign-in" ? "current-password" : "new-password"} required minLength={6} />
        <Button primary type="submit" disabled={busy}>{busy ? "Please wait…" : mode === "sign-in" ? "Sign in" : "Create account"}</Button>
      </form>
    </Card>}
    {error && <Card><p className="error">{error}</p></Card>}{notice && <Card><p className="live">{notice}</p></Card>}
  </div>;
}

export function ServedModelPerformancePage({ go }: { go: (page: string) => void }) {
  const [model, setModel] = useState<ServingModelInfo | null>(null);
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [graphError, setGraphError] = useState<Record<string, boolean>>({});
  async function load() {
    setLoading(true); setError(null);
    const results = await Promise.allSettled([getServingModelInfo(), getAnalyticsSummary()]);
    if (results[0].status === "fulfilled") setModel(results[0].value);
    else setError(results[0].reason instanceof Error ? results[0].reason.message : "Model info unavailable.");
    if (results[1].status === "fulfilled") setSummary(results[1].value);
    setLoading(false);
  }
  useEffect(() => { void load(); }, []);
  const analytics = summary?.metrics as Record<string, unknown> | undefined;
  return <div className="page-content inner-page">
    <PageTitle title="Model Performance" description="Metrics and identity from the model currently served by the backend."><Button onClick={() => void load()} disabled={loading}>Refresh</Button><Button onClick={() => go("Model Insights")}>Class labels</Button></PageTitle>
    {loading && <Card className="loading-state">Loading served model information…</Card>}
    {!loading && error && <Card><p className="error">{error}</p><Button onClick={() => void load()}>Retry</Button></Card>}
    {!loading && model && <>
      <div className="metrics" style={{ gridTemplateColumns: "repeat(3,1fr)", marginBottom: 10 }}>
        <div className="card metric"><span>Test accuracy</span><strong>{pct(model.test_acc)}</strong><small>Served checkpoint metric</small></div>
        <div className="card metric"><span>Test macro F1</span><strong>{pct(model.test_macro_f1 ?? undefined)}</strong><small>Served checkpoint metric</small></div>
        <div className="card metric"><span>Best validation macro F1</span><strong>{pct(model.best_val_macro_f1 ?? undefined)}</strong><small>Best epoch {model.best_epoch ?? "—"}</small></div>
      </div>
      <div className="dashboard-grid">
        <Card><div className="card-heading"><h3>Training curves</h3><small className="subtle">Analytics API graph</small></div>{graphError.training ? <p className="error">Training curves unavailable.</p> : <img alt="Training curves" src={getGraphUrl("training_curves")} className="analytics-graph" onError={() => setGraphError((state) => ({ ...state, training: true }))} />}</Card>
        <Card><div className="card-heading"><h3>Confusion matrix</h3><small className="subtle">{model.num_classes} classes</small></div>{graphError.confusion ? <p className="error">Confusion matrix unavailable.</p> : <img alt="Confusion matrix" src={getGraphUrl("confusion_matrix")} className="analytics-graph" onError={() => setGraphError((state) => ({ ...state, confusion: true }))} />}</Card>
        <Card style={{ gridColumn: "1 / -1" }}><div className="card-heading"><h3>Served checkpoint</h3></div>
          <p className="settings-row"><span>Backbone</span><b>{model.backbone}</b></p><p className="settings-row"><span>Classes</span><b>{model.num_classes}</b></p><p className="settings-row"><span>Input size</span><b>{model.img_size}px</b></p><p className="settings-row"><span>Preprocessing</span><b>{model.preprocessing ?? "—"}</b></p><p className="settings-row"><span>Best epoch</span><b>{model.best_epoch ?? "—"}</b></p><p className="settings-row"><span>Validation accuracy</span><b>{model.best_val_acc == null ? "Not provided by the serving API" : pct(model.best_val_acc)}</b></p>
          <small className="subtle">{typeof analytics?.best_val_macro_f1_pct === "number" ? `Analytics endpoint agrees on best validation macro F1: ${analytics.best_val_macro_f1_pct}%` : "No additional split metrics are available for this served checkpoint."}</small>
        </Card>
      </div>
    </>}
  </div>;
}

export function ServedModelInsightsPage() {
  const [labels, setLabels] = useState<string[]>([]);
  const [model, setModel] = useState<ServingModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  async function load() {
    setLoading(true); setError(null);
    const results = await Promise.allSettled([getModelLabels(), getServingModelInfo()]);
    if (results[0].status === "fulfilled") setLabels(results[0].value);
    else setError(results[0].reason instanceof Error ? results[0].reason.message : "Model class labels unavailable.");
    if (results[1].status === "fulfilled") setModel(results[1].value);
    setLoading(false);
  }
  useEffect(() => { void load(); }, []);
  const filtered = useMemo(() => labels.filter((label) => label.toLowerCase().includes(filter.toLowerCase())), [labels, filter]);
  return <div className="page-content inner-page">
    <PageTitle title="Model Insights" description="The served model’s actual class labels and aggregate evaluation metrics."><Button onClick={() => void load()} disabled={loading}>Refresh</Button></PageTitle>
    {loading && <Card className="loading-state">Loading model labels…</Card>}
    {!loading && error && <Card><p className="error">{error}</p><Button onClick={() => void load()}>Retry</Button></Card>}
    {!loading && model && <>
      <div className="metrics" style={{ gridTemplateColumns: "repeat(3,1fr)", marginBottom: 10 }}>
        <div className="card metric"><span>Served classes</span><strong>{labels.length}</strong><small>Labels returned by /api/model/labels</small></div>
        <div className="card metric"><span>Test accuracy</span><strong>{pct(model.test_acc)}</strong><small>{model.backbone}</small></div>
        <div className="card metric"><span>Test macro F1</span><strong>{pct(model.test_macro_f1 ?? undefined)}</strong><small>Best validation macro F1 {pct(model.best_val_macro_f1 ?? undefined)}</small></div>
      </div>
      <Card><div className="card-heading"><h3>Served class labels ({filtered.length})</h3><input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Filter model classes…" /></div>
        {filtered.length ? <div className="class-label-grid">{filtered.map((label, index) => <div className="knowledge-item" key={`${label}-${index}`}><span>{index + 1}</span><b>{label.replace(/[_-]/g, " ")}</b></div>)}</div> : <p className="subtle">No labels match this filter.</p>}
        <p className="subtle">The current serving API provides class names and aggregate scores; it does not expose per-class precision, recall, or F1.</p>
      </Card>
    </>}
  </div>;
}

export function MultimodalPage() {
  const [status, setStatus] = useState<MultimodalStatus | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [pairedText, setPairedText] = useState("");
  const [analysis, setAnalysis] = useState<MultimodalAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  async function refresh() {
    setLoading(true);
    setStatusError(null);
    try { setStatus(await getMultimodalStatus()); }
    catch (cause) { setStatusError(cause instanceof Error ? cause.message : "Multimodal status is unavailable."); }
    finally { setLoading(false); }
  }
  useEffect(() => { void refresh(); }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file || !pairedText.trim() || running || !status?.trained) return;
    setRunning(true);
    setAnalysisError(null);
    setAnalysis(null);
    try { setAnalysis(await analyzeMultimodalImage(file, pairedText)); }
    catch (cause) { setAnalysisError(cause instanceof Error ? cause.message : "Multimodal analysis failed."); }
    finally { setRunning(false); }
  }

  const architecture = status?.architecture ?? {};
  return <div className="page-content inner-page">
    <PageTitle title="Multimodal AI" description="Experimental shared image and class-metadata representation with cross-attention."><Button onClick={() => void refresh()} disabled={loading}>{loading ? "Checking…" : "Refresh status"}</Button></PageTitle>
    {loading ? <Card className="loading-state"><p>Checking multimodal checkpoint status…</p></Card> : statusError && !status ? <Card><p className="error">Status request failed: {statusError}</p><Button onClick={() => void refresh()}>Retry</Button></Card> : status && <>
      {statusError && <Card><p className="error">Status refresh failed: {statusError}</p><Button onClick={() => void refresh()}>Retry</Button></Card>}
      <Card>
        <div className="card-heading"><h3>{status.module}</h3><span className={status.trained ? "live" : "low-confidence"}>{status.trained ? "Trained checkpoint ready" : "Unavailable · not trained"}</span></div>
        <p>{status.message || "Validated multimodal checkpoint is available."}</p>
        <p className="settings-row"><span>Image encoder</span><b>{String(architecture.image_encoder ?? "EfficientNet-B0 compatible features")}</b></p>
        <p className="settings-row"><span>Text input</span><b>{status.data.text_source}</b></p>
        <p className="settings-row"><span>PlantWild train / validation entries</span><b>{status.data.train_split_entries.toLocaleString()} / {status.data.validation_split_entries.toLocaleString()}</b></p>
        <p className="settings-row"><span>Cross-attention</span><b>{String(architecture.decoder ?? "Bidirectional image-text Transformer decoders")}</b></p>
        <p className="subtle">No paired natural-language descriptions are used. The production EfficientNet-B0 classifier remains responsible for diagnosis.</p>
        <h3>Diagnosis model availability</h3>
        {status.model_availability?.map((model) => <p className="settings-row" key={model.name}><span>{model.name}</span><b className={model.trained ? "live" : "low-confidence"}>{model.trained ? "Available" : "Unavailable"}</b></p>)}
        <p className="subtle">{status.text_modality_gate}</p>
      </Card>
      <Card>
        <div className="card-heading"><h3>Multimodal analysis</h3><small className="subtle">Requires completed and validated multimodal training</small></div>
        {!status.trained && <p className="low-confidence">Analysis is disabled until a real trained checkpoint is installed. No embeddings or performance claims are generated from random weights.</p>}
        <form className="query-form agent-form" onSubmit={submit}>
          <label className="agent-file">Plant image<input type="file" accept="image/*" onChange={(event) => setFile(event.target.files?.[0] ?? null)} disabled={!status.trained || running} /></label>
          <label>Observed paired text<textarea value={pairedText} onChange={(event) => setPairedText(event.target.value)} placeholder="Describe only observations available for this image…" disabled={!status.trained || running} /></label>
          <Button primary type="submit" disabled={!file || !pairedText.trim() || !status.trained || running}>{running ? "Analyzing…" : "Run multimodal analysis"}</Button>
        </form>
        {file && <p className="subtle">Selected: {file.name}</p>}
        {running && <p className="loading-state">Running EfficientNet-B0 diagnosis and multimodal fusion…</p>}
        {analysisError && <p className="error">Analysis failed: {analysisError}</p>}
      </Card>
      {analysis && <>
        <Card><div className="card-heading"><h3>Production diagnosis</h3><span className={analysis.diagnosis.low_confidence ? "low-confidence" : "live"}>{analysis.diagnosis.low_confidence ? "Low confidence" : "Diagnosis returned"}</span></div>
          <h2>{analysis.diagnosis.predicted_class.label}</h2><p>Confidence {(analysis.diagnosis.predicted_class.confidence * 100).toFixed(2)}% · {analysis.production_classifier} · {analysis.diagnosis.model_info.num_classes} classes</p>
          <p className="settings-row"><span>Fused shared representation</span><b>{analysis.fused_representation.dimension} dimensions</b></p>
          <p className="subtle">Paired text used: {analysis.metadata_text}. This representation is returned for downstream consumers; it does not replace classifier output.</p>
        </Card>
        <Card>
          <div className="card-heading"><h3>RAG guidance sources</h3><small className="subtle">{analysis.guidance.retrieved ?? 0} passages retrieved</small></div>
          {analysis.guidance.sources?.length ? analysis.guidance.sources.map((source, index) => {
            const norm = (source.source || "").replace(/\\/g, "/");
            let category = "Reference Profile";
            if (norm.includes("/diseases/")) category = "Disease Profile";
            else if (norm.includes("/treatment/")) category = "Treatment Guideline";
            else if (norm.includes("/fertilizer/")) category = "Fertilizer Guide";
            else if (norm.includes("/plant_care/")) category = "Plant Care";
            return (
              <div className="source-item" key={`${source.source}-${index}`}>
                <b>[{index + 1}] {source.title || "Knowledge source"}</b>
                <span className="source-category-tag">{category}</span>
                <p>{source.text || source.excerpt || "No excerpt was returned."}</p>
              </div>
            );
          }) : <p className="subtle">No grounded guidance sources were returned{analysis.guidance.error ? `: ${analysis.guidance.error}` : "."}</p>}
        </Card>
      </>}
    </>}
  </div>;
}

const FUSION_ARCHITECTURES: {
  kind: FusionKind;
  name: string;
  tagline: string;
  description: string;
  textRequired: boolean;
  textNote: string;
}[] = [
  {
    kind: "early",
    name: "Early Fusion",
    tagline: "Feature-level concatenation",
    description: "Concatenates extracted visual feature vectors with tokenized symptom embeddings before deep joint multilayer classification.",
    textRequired: true,
    textNote: "Requires observed symptom text to produce joint representation.",
  },
  {
    kind: "intermediate",
    name: "Intermediate / Gated Fusion",
    tagline: "Cross-attention gating",
    description: "Uses a learned gating network and cross-attention layers to dynamically balance visual cues against observed text descriptions.",
    textRequired: true,
    textNote: "Requires paired symptom text to compute cross-attention weights.",
  },
  {
    kind: "late",
    name: "Late Fusion",
    tagline: "Decision-level ensemble with fallback",
    description: "Maintains parallel image and text classification branches combined via learned fusion weights. Supports image-only fallback.",
    textRequired: false,
    textNote: "Optional paired text. Runs in image-only fallback mode if text is omitted.",
  },
];

export function FusionAnalysisPage() {
  const [status, setStatus] = useState<FusionStatus | null>(null);
  const [selectedKind, setSelectedKind] = useState<FusionKind>("early");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [pairedText, setPairedText] = useState("");
  const [topK, setTopK] = useState(5);
  const [loadingStatus, setLoadingStatus] = useState(true);
  const [running, setRunning] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [result, setResult] = useState<FusionAnalysis | null>(null);

  async function refresh() {
    setLoadingStatus(true);
    setStatusError(null);
    try {
      setStatus(await getFusionStatus());
    } catch (cause) {
      setStatusError(cause instanceof Error ? cause.message : "Fusion service status is unavailable.");
    } finally {
      setLoadingStatus(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    if (preview) URL.revokeObjectURL(preview);
    setFile(selected);
    setPreview(selected ? URL.createObjectURL(selected) : null);
    setValidationError(null);
  }

  const selectedArch = FUSION_ARCHITECTURES.find((a) => a.kind === selectedKind) || FUSION_ARCHITECTURES[0];
  const isTextMissing = selectedArch.textRequired && !pairedText.trim();

  async function submit(event: FormEvent) {
    event.preventDefault();
    setValidationError(null);
    setAnalysisError(null);

    if (!file) {
      setValidationError("Please select a plant image to analyze.");
      return;
    }
    if (selectedArch.textRequired && !pairedText.trim()) {
      setValidationError(
        `Paired symptom text is required for ${selectedArch.name}. Please enter field observations or switch to Late Fusion.`
      );
      return;
    }

    setRunning(true);
    setResult(null);
    try {
      const res = await analyzeFusionImage(selectedKind, file, pairedText.trim() || undefined, topK);
      setResult(res);
    } catch (cause) {
      setAnalysisError(cause instanceof Error ? cause.message : "Fusion analysis request failed.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="page-content inner-page">
      <PageTitle
        title="Multimodal Fusion Analysis"
        description="Select and evaluate trained multimodal neural fusion architectures (Early, Intermediate/Gated, and Late) on plant disease imagery and symptom observations."
      >
        <Button onClick={() => void refresh()} disabled={loadingStatus}>
          {loadingStatus ? "Checking…" : "Refresh status"}
        </Button>
      </PageTitle>

      {statusError && (
        <Card>
          <p className="error">Fusion status error: {statusError}</p>
          <Button onClick={() => void refresh()}>Retry</Button>
        </Card>
      )}

      {/* Architecture Selection Grid */}
      <Card>
        <div className="card-heading">
          <h3>1. Select Fusion Architecture</h3>
          <small className="subtle">3 trained architectures available</small>
        </div>
        <p className="subtle" style={{ margin: "4px 0 12px" }}>
          Choose an architecture to compare how image features and symptom text descriptions are integrated during classification.
        </p>
        <div className="fusion-arch-grid">
          {FUSION_ARCHITECTURES.map((arch) => {
            const isSelected = selectedKind === arch.kind;
            const modelAvail = status?.fusion_models?.find((m) => m.kind === arch.kind);
            return (
              <button
                type="button"
                key={arch.kind}
                className={`fusion-arch-card ${isSelected ? "selected" : ""}`}
                onClick={() => {
                  setSelectedKind(arch.kind);
                  setValidationError(null);
                }}
              >
                <div className="fusion-arch-head">
                  <h4 className="fusion-arch-title">{arch.name}</h4>
                  <span className={modelAvail?.available ? "live" : "low-confidence"}>
                    {modelAvail?.available ? "Ready" : "Unavailable"}
                  </span>
                </div>
                <small style={{ color: "#16a267", fontWeight: 600 }}>{arch.tagline}</small>
                <p className="fusion-arch-desc">{arch.description}</p>
                <div className="fusion-arch-badges">
                  <span className={`fusion-req-tag ${arch.textRequired ? "required" : "optional"}`}>
                    {arch.textRequired ? "Text Required" : "Text Optional · Fallback Supported"}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </Card>

      {/* Input Form Card */}
      <Card>
        <div className="card-heading">
          <h3>2. Provide Image &amp; Observation Inputs</h3>
          <small className="subtle">Selected architecture: {selectedArch.name}</small>
        </div>

        <form onSubmit={submit}>
          <div className="fusion-form-grid">
            {/* Image upload column */}
            <div>
              <label className="agent-file" style={{ display: "block", marginBottom: 6, fontWeight: 600 }}>
                Plant Leaf Image <span style={{ color: "#db5a57" }}>*</span>
              </label>
              <label className="fusion-upload-box">
                {preview ? (
                  <>
                    <img src={preview} alt="Selected sample" className="fusion-preview-img" />
                    <small className="subtle">Click or drag to change image: {file?.name}</small>
                  </>
                ) : (
                  <>
                    <span style={{ fontSize: 36, color: "#16a267" }}>↥</span>
                    <b>Upload Plant Leaf / Canopy Image</b>
                    <small className="subtle">Supports JPG, PNG, WEBP (validated for 89 agricultural classes)</small>
                  </>
                )}
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  disabled={running}
                  style={{ display: "none" }}
                />
              </label>
            </div>

            {/* Paired text column */}
            <div className="fusion-text-group">
              <label htmlFor="fusion-paired-text">
                Observed Symptom Text {selectedArch.textRequired ? <span style={{ color: "#db5a57" }}>* (Required)</span> : <span className="subtle">(Optional)</span>}
              </label>
              <textarea
                id="fusion-paired-text"
                value={pairedText}
                onChange={(e) => {
                  setPairedText(e.target.value);
                  setValidationError(null);
                }}
                placeholder="Enter field observations, leaf discoloration, lesion patterns, or crop context (e.g., 'circular dark brown lesions with yellow halo on apple leaf margins')..."
                disabled={running}
              />
              {selectedArch.textRequired ? (
                <p className="fusion-hint required">
                  ⚠️ <b>Text Required:</b> {selectedArch.name} cannot evaluate without paired symptom observations.
                </p>
              ) : (
                <p className="fusion-hint optional">
                  ℹ️ <b>Text Optional:</b> If omitted, Late Fusion falls back to unimodal image classification via its vision stream.
                </p>
              )}

              <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 8 }}>
                <label style={{ fontSize: 11, fontWeight: 600, color: "#43596c" }}>
                  Top-K Predictions:
                </label>
                <select
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  disabled={running}
                  style={{
                    padding: "4px 8px",
                    borderRadius: 5,
                    border: "1px solid #d4e0e6",
                    background: "#fff",
                    font: "inherit",
                    fontSize: 11,
                  }}
                >
                  <option value={3}>Top 3</option>
                  <option value={5}>Top 5</option>
                  <option value={10}>Top 10</option>
                </select>
              </div>
            </div>
          </div>

          {validationError && (
            <p className="error" style={{ marginTop: 12, fontWeight: 600 }}>
              ⚠️ {validationError}
            </p>
          )}

          {analysisError && (
            <div className="prediction-alert-banner low-confidence-banner" style={{ marginTop: 12 }}>
              <div className="prediction-alert-content">
                <b>Analysis Error</b>
                <p>{analysisError}</p>
              </div>
            </div>
          )}

          <div style={{ marginTop: 16, display: "flex", gap: 10, alignItems: "center" }}>
            <Button
              primary
              type="submit"
              disabled={running || !file || isTextMissing}
            >
              {running ? `Analyzing with ${selectedArch.name}…` : `Run ${selectedArch.name} Analysis`}
            </Button>
            {running && <span className="loading-state">Running fusion neural network inference…</span>}
          </div>
        </form>
      </Card>

      {/* Results Display */}
      {result && (
        <Card style={{ marginTop: 16 }}>
          <div className="card-heading">
            <h3>3. Fusion Diagnostic Results</h3>
            <span className="live">Inference Completed</span>
          </div>

          {/* Fallback & Modality Alert Banner */}
          {result.image_only_fallback ? (
            <div className="prediction-alert-banner low-confidence-banner">
              <span className="prediction-alert-icon">ℹ</span>
              <div className="prediction-alert-content">
                <b>Late Fusion Image-Only Fallback Active</b>
                <p>
                  No paired symptom text was provided; the model evaluated solely using the visual classification branch.
                </p>
              </div>
            </div>
          ) : (
            <div className="prediction-alert-banner standard-banner">
              <span className="prediction-alert-icon">⚛</span>
              <div className="prediction-alert-content">
                <b>Multimodal Fusion Active</b>
                <p>
                  Diagnostic prediction synthesized across visual features and provided symptom text observations.
                </p>
              </div>
            </div>
          )}

          {/* Primary Result Details */}
          <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 20, marginTop: 12 }}>
            <div>
              <span className="prediction-label-prefix">Predicted Class</span>
              <h2 style={{ fontSize: 22, color: "#16a267", margin: "2px 0 8px", textTransform: "capitalize" }}>
                {result.predicted_class.label.replace(/[_-]/g, " ")}
              </h2>
              <p style={{ margin: "0 0 10px", fontSize: 13, color: "#324456" }}>
                Confidence: <strong style={{ color: "#16a267", fontSize: 16 }}>{(result.confidence * 100).toFixed(2)}%</strong>
              </p>

              <div className="modality-badges">
                <span className="subtle" style={{ fontSize: 11, marginRight: 4 }}>Modalities:</span>
                {result.modalities_used.map((m) => (
                  <span key={m} className={`modality-tag ${m}`}>
                    {m}
                  </span>
                ))}
                {result.image_only_fallback && (
                  <span className="modality-tag fallback">Image-Only Fallback</span>
                )}
              </div>

              <div style={{ marginTop: 14 }}>
                <p className="settings-row">
                  <span>Architecture</span>
                  <b>{result.model.fusion_kind.toUpperCase()} FUSION</b>
                </p>
                <p className="settings-row">
                  <span>Inference Latency</span>
                  <b>{result.inference_ms.toFixed(1)} ms</b>
                </p>
                <p className="settings-row">
                  <span>Classification Space</span>
                  <b>{result.model.class_count} plant disease classes</b>
                </p>
                <p className="settings-row">
                  <span>Text Observation Provided</span>
                  <b>{result.text_provided ? "Yes" : "No"}</b>
                </p>
              </div>
              <p className="subtle" style={{ marginTop: 10, fontStyle: "italic" }}>
                {result.notice}
              </p>
            </div>

            {/* Top-K Confidence Bars */}
            <div>
              <div className="card-heading" style={{ marginBottom: 8 }}>
                <h4 style={{ margin: 0, fontSize: 12, color: "#43596c" }}>
                  Top-{result.top_k.length} Class Probabilities
                </h4>
              </div>
              <div className="top-list">
                {result.top_k.map((item, idx) => {
                  const confPct = item.confidence * 100;
                  return (
                    <div key={item.class_id} style={{ marginBottom: 9 }}>
                      <p style={{ margin: "0 0 3px" }}>
                        <span>
                          {idx + 1}. <strong style={{ textTransform: "capitalize" }}>{item.label.replace(/[_-]/g, " ")}</strong>
                        </span>
                        <b>{confPct.toFixed(2)}%</b>
                        <i>
                          <em style={{ width: `${Math.max(2, confPct)}%` }} />
                        </i>
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}

