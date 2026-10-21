import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity, AlertCircle, ArrowDownRight, ArrowUpRight, BookOpen, Boxes, Check,
  CheckCircle2, ChevronDown, ChevronRight, CircleHelp, Clock3, Code2, Database,
  FileCode2, FileText, Gauge, GitBranch, LayoutDashboard, LoaderCircle, Menu,
  MessageSquareText, Plus, RefreshCw, Rocket, Search, Send, ShieldCheck,
  SlidersHorizontal, Sparkles, X, XCircle, Zap,
} from 'lucide-react'
import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

type Model = {
  id: string
  provider: string
  intended_capability: string
  model_type: string
  availability_status: string
  last_inference_status: string
  discovered: boolean
  verified: boolean
  approved: boolean
  deployed: boolean
}
type Deployment = { id: string | number; name?: string; model?: string; status?: string; [key: string]: unknown }
type Health = { status?: string; [key: string]: unknown }
type ChatResult = { model: string; response: string; latency_seconds: number; input_tokens: number | null; output_tokens: number | null; tokens_per_second: number | null }
type Finding = { source: string; severity: string; title: string; explanation: string; recommendation: string }
type Source = { document: string; chunk: number; text: string; score: number }

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers },
  })
  const raw = await response.text()
  let data: unknown = null
  try { data = raw ? JSON.parse(raw) : null } catch { data = raw }
  if (!response.ok) {
    const message = typeof data === 'object' && data !== null && 'detail' in data
      ? String((data as { detail: unknown }).detail)
      : typeof data === 'string' && data ? data : `Request failed (${response.status})`
    throw new Error(message)
  }
  return data as T
}
const post = (path: string, body: unknown) => api(path, { method: 'POST', body: JSON.stringify(body) })
const pretty = (value: unknown) => JSON.stringify(value, null, 2)
const titleCase = (value: string) => value.replace(/[_-]/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value)
const INSECURE_KUBERNETES_SAMPLE = `apiVersion: apps/v1
kind: Deployment
metadata:
  name: modelops-review-demo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: modelops-review-demo
  template:
    metadata:
      labels:
        app: modelops-review-demo
    spec:
      hostNetwork: true
      containers:
        - name: demo
          image: nginx:latest
          securityContext:
            privileged: true
            allowPrivilegeEscalation: true`

const sections = [
  { heading: 'WORKSPACE', items: [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'registry', label: 'Model Registry', icon: Boxes },
    { id: 'playground', label: 'LLM Playground', icon: MessageSquareText },
    { id: 'benchmark', label: 'Model Benchmark', icon: Gauge },
  ] },
  { heading: 'BUILD & SHIP', items: [
    { id: 'knowledge', label: 'RAG Knowledge Base', icon: BookOpen },
    { id: 'reviewer', label: 'AI Infrastructure Reviewer', icon: FileCode2 },
    { id: 'deployments', label: 'Model Deployment', icon: Rocket },
  ] },
  { heading: 'OPERATE', items: [
    { id: 'observability', label: 'Observability', icon: Activity },
    { id: 'governance', label: 'Governance & Audit', icon: ShieldCheck },
  ] },
]

function App() {
  const [active, setActive] = useState('dashboard')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState('')
  const [models, setModels] = useState<Model[]>([])
  const [deployments, setDeployments] = useState<Deployment[]>([])
  const [summary, setSummary] = useState<unknown>(null)
  const [metricsText, setMetricsText] = useState('')
  const [refreshing, setRefreshing] = useState(true)
  const [dataError, setDataError] = useState('')

  const refreshCore = useCallback(async () => {
    setRefreshing(true)
    const results = await Promise.allSettled([
      api<Health>('/api/health'),
      api<{ models?: Model[] }>('/api/llm/models'),
      api<{ deployments?: Deployment[] } | Deployment[]>('/api/deployments'),
      api<unknown>('/api/metrics/summary'),
      fetch(`${API_BASE}/metrics`).then(async (r) => {
        if (!r.ok) throw new Error(`Request failed (${r.status})`)
        return r.text()
      }),
    ])
    const [h, m, d, s, metrics] = results
    const failures: string[] = []
    if (m.status === 'rejected') failures.push(`Model registry: ${m.reason instanceof Error ? m.reason.message : 'request failed'}`)
    if (d.status === 'rejected') failures.push(`Deployments: ${d.reason instanceof Error ? d.reason.message : 'request failed'}`)
    if (s.status === 'rejected') failures.push(`Metrics summary: ${s.reason instanceof Error ? s.reason.message : 'request failed'}`)
    if (metrics.status === 'rejected') failures.push(`Prometheus metrics: ${metrics.reason instanceof Error ? metrics.reason.message : 'request failed'}`)
    setDataError(failures.join(' · '))
    if (h.status === 'fulfilled') { setHealth(h.value); setHealthError('') }
    else { setHealth(null); setHealthError(h.reason instanceof Error ? h.reason.message : 'Health check failed') }
    if (m.status === 'fulfilled') setModels(Array.isArray(m.value.models) ? m.value.models : [])
    if (d.status === 'fulfilled') {
      const list = Array.isArray(d.value) ? d.value : d.value.deployments
      setDeployments(Array.isArray(list) ? list : [])
    }
    if (s.status === 'fulfilled') setSummary(s.value)
    if (metrics.status === 'fulfilled') setMetricsText(metrics.value)
    setRefreshing(false)
  }, [])

  useEffect(() => { void refreshCore() }, [refreshCore])
  const currentTitle = sections.flatMap((section) => section.items).find((item) => item.id === active)?.label ?? 'Dashboard'
  const navigate = (id: string) => { setActive(id); setMobileOpen(false) }
  const contentProps = { models, deployments, summary, metricsText, onRefresh: refreshCore }

  return (
    <div className="app-shell">
      {mobileOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
      <aside className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <div className="brand-mark"><span className="brand-glyph">M</span></div>
          <div><div className="brand-name">modelops<span>hub</span></div><div className="brand-caption">AI PLATFORM</div></div>
          <button className="icon-button sidebar-close" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X size={17} /></button>
        </div>
        <div className="workspace-picker"><div className="workspace-avatar">N</div><div className="workspace-copy"><b>Northstar Labs</b><span>Enterprise workspace</span></div><ChevronDown size={15} /></div>
        <nav className="nav-groups">
          {sections.map((section) => <div className="nav-group" key={section.heading}>
            <div className="nav-heading">{section.heading}</div>
            {section.items.map(({ id, label, icon: Icon }) => <button key={id} className={`nav-item ${active === id ? 'nav-active' : ''}`} onClick={() => navigate(id)}>
              <Icon size={17} strokeWidth={1.8} /><span>{label}</span>{active === id && <span className="active-indicator" />}
            </button>)}
          </div>)}
        </nav>
        <div className="sidebar-bottom">
          <div className={`api-health ${health && !healthError ? 'health-up' : 'health-down'}`}><span className="health-dot" /><div><b>API {health && !healthError ? 'connected' : 'unavailable'}</b><span>{health && !healthError ? API_BASE.replace(/^https?:\/\//, '') : 'Check backend connection'}</span></div><button onClick={() => void refreshCore()} aria-label="Refresh API status"><RefreshCw size={14} className={refreshing ? 'spin' : ''} /></button></div>
          <div className="user-profile"><div className="user-avatar">SN</div><div className="user-copy"><b>Workspace admin</b><span>Platform team</span></div><ChevronDown size={15} /></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-leading"><button className="icon-button mobile-menu" aria-label="Open navigation" onClick={() => setMobileOpen(true)}><Menu size={19} /></button><div className="breadcrumb">Workspace <ChevronRight size={13} /> <b>{currentTitle}</b></div></div>
          <div className="topbar-actions"><span className="environment-badge"><span /> DEVELOPMENT</span><button className="icon-button" title="Refresh dashboard" onClick={() => void refreshCore()}><RefreshCw size={16} className={refreshing ? 'spin' : ''} /></button><div className="topbar-avatar">SN</div></div>
        </header>
        <div className="content-wrap">
          {healthError && <div className="connection-banner"><AlertCircle size={16} /><span>Backend API is unreachable at <code>{API_BASE}</code>. Confirm the service is running and check <code>VITE_API_BASE_URL</code>.</span><button onClick={() => void refreshCore()}><RefreshCw size={14} /> Retry</button></div>}
          {dataError && !healthError && <div className="data-warning"><AlertCircle size={15} /><span>Some API data could not be refreshed: {dataError}</span><button onClick={() => void refreshCore()}><RefreshCw size={14} /> Retry</button></div>}
          {active === 'dashboard' && <Dashboard {...contentProps} health={health} healthError={healthError} refreshing={refreshing} />}
          {active === 'registry' && <Registry models={models} />}
          {active === 'playground' && <Playground models={models} />}
          {active === 'benchmark' && <Benchmark />}
          {active === 'knowledge' && <KnowledgeBase models={models} />}
          {active === 'reviewer' && <Reviewer />}
          {active === 'deployments' && <Deployments models={models} deployments={deployments} onRefresh={refreshCore} />}
          {active === 'observability' && <Observability summary={summary} metricsText={metricsText} />}
          {active === 'governance' && <Governance models={models} deployments={deployments} />}
          <footer className="page-footer"><span>ModelOps Hub <span className="footer-dot">·</span> Connected to your ModelOps API</span><span>API <code>{API_BASE}</code></span></footer>
        </div>
      </main>
    </div>
  )
}

function PageHeading({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: React.ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action && <div className="heading-action">{action}</div>}</div>
}
function Card({ children, className = '' }: { children: React.ReactNode; className?: string }) { return <section className={`card ${className}`}>{children}</section> }
function CardHeading({ title, subtitle, action }: { title: string; subtitle?: string; action?: React.ReactNode }) {
  return <div className="card-heading"><div><h3>{title}</h3>{subtitle && <p>{subtitle}</p>}</div>{action}</div>
}
function EmptyState({ icon: Icon = Database, title, children }: { icon?: typeof Database; title: string; children: React.ReactNode }) {
  return <div className="empty-state"><div className="empty-icon"><Icon size={19} /></div><b>{title}</b><p>{children}</p></div>
}
function ErrorNotice({ error, onDismiss }: { error: string; onDismiss?: () => void }) {
  if (!error) return null
  return <div className="error-notice"><AlertCircle size={16} /><span>{error}</span>{onDismiss && <button onClick={onDismiss} aria-label="Dismiss error"><X size={15} /></button>}</div>
}
function LoadingButton({ loading, children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { loading?: boolean }) {
  return <button {...props} disabled={props.disabled || loading}>{loading ? <LoaderCircle size={15} className="spin" /> : children}</button>
}
function StatusPill({ value }: { value: string | boolean | undefined }) {
  const label = typeof value === 'boolean' ? (value ? 'Yes' : 'No') : value || 'Unknown'
  const tone = value === false || /error|failed|unavailable|rejected|blocked|not_loaded/i.test(String(value))
    ? 'pill-bad' : value === true || /available|success|healthy|ready|active|approved|deployed|pass/i.test(String(value)) ? 'pill-good' : 'pill-neutral'
  return <span className={`status-pill ${tone}`}><span />{titleCase(label)}</span>
}

function Dashboard({ models, deployments, summary, metricsText, health, healthError, refreshing, onRefresh }: {
  models: Model[]; deployments: Deployment[]; summary: unknown; metricsText: string; health: Health | null; healthError: string; refreshing: boolean; onRefresh: () => void
}) {
  const apiMetricCount = metricsText.split('\n').filter((line) => line && !line.startsWith('#')).length
  const cards = [
    { label: 'Registered models', value: models.length, note: 'From the model registry', icon: Boxes, color: 'teal' },
    { label: 'Deployment records', value: deployments.length, note: 'Manifests returned by the API', icon: Rocket, color: 'blue' },
    { label: 'Approved models', value: models.filter((model) => model.approved).length, note: 'Governance approval status', icon: ShieldCheck, color: 'violet' },
    { label: 'API metrics', value: apiMetricCount, note: 'Live samples from /metrics', icon: Activity, color: 'amber' },
  ]
  const summaryFields = isRecord(summary) ? Object.entries(summary).filter(([, value]) => typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean') : []
  return <>
    <PageHeading eyebrow="OVERVIEW" title="Good evening, team" description="Your AI operations at a glance. Data is sourced directly from the connected ModelOps API."
      action={<button className="button-secondary" onClick={onRefresh}><RefreshCw size={15} className={refreshing ? 'spin' : ''} /> Refresh data</button>} />
    <div className="welcome-strip"><div className="welcome-icon"><Sparkles size={19} /></div><div><b>Your model operations workspace</b><span>Manage the full lifecycle, from discovery and evaluation to deployment and observability.</span></div><div className={`connection-chip ${health && !healthError ? 'connected' : ''}`}><span />{health && !healthError ? 'API connected' : 'API unavailable'}</div></div>
    <div className="stat-grid">{cards.map(({ label, value, note, icon: Icon, color }) => <Card className="stat-card" key={label}><div className={`stat-icon ${color}`}><Icon size={18} /></div><div className="stat-label">{label}</div><div className="stat-value">{refreshing ? '—' : value}</div><div className="stat-note">{note}</div></Card>)}</div>
    <div className="dashboard-grid">
      <Card className="registry-overview"><CardHeading title="Model registry" subtitle="Live inventory from your API" action={<span className="live-label"><i /> LIVE DATA</span>} />
        {models.length ? <div className="table-scroll"><table><thead><tr><th>MODEL</th><th>PROVIDER</th><th>CAPABILITY</th><th>STATUS</th><th>APPROVED</th></tr></thead><tbody>{models.slice(0, 5).map((model) => <tr key={model.id}><td><div className="model-cell"><span className="model-symbol"><Boxes size={14} /></span><b>{model.id}</b></div></td><td>{model.provider}</td><td>{model.intended_capability || '—'}</td><td><StatusPill value={model.availability_status} /></td><td><StatusPill value={model.approved} /></td></tr>)}</tbody></table></div> : <EmptyState icon={Boxes} title="No registry data available">Models appear here after the API returns registry results.</EmptyState>}
        {models.length > 5 && <div className="card-bottom-note">Showing 5 of {models.length} registered models</div>}
      </Card>
      <Card className="health-card"><CardHeading title="Platform status" subtitle="Connectivity and live service health" />
        <div className="health-summary"><div className={`health-orb ${health && !healthError ? 'orb-up' : 'orb-down'}`}><Activity size={21} /></div><div><b>{health && !healthError ? titleCase(String(health.status || 'Connected')) : 'Health check failed'}</b><span>{healthError || 'Response from GET /api/health'}</span></div></div>
        <div className="health-details"><div><span>API base URL</span><b>{API_BASE}</b></div><div><span>Registry models</span><b>{models.length}</b></div><div><span>Deployments returned</span><b>{deployments.length}</b></div></div>
        {health && <details className="raw-details"><summary>Health response <ChevronDown size={14} /></summary><pre>{pretty(health)}</pre></details>}
      </Card>
    </div>
    <div className="dashboard-grid dashboard-grid-bottom">
      <Card><CardHeading title="Metrics summary" subtitle="Values from GET /api/metrics/summary" action={<span className="subtle-tag">LIVE</span>} />
        {summaryFields.length ? <div className="summary-metrics">{summaryFields.slice(0, 8).map(([key, value]) => <div className="summary-metric" key={key}><span>{titleCase(key)}</span><b>{String(value)}</b></div>)}</div>
          : <EmptyState icon={Activity} title="No summary metrics returned">The metrics summary endpoint has not returned displayable values.</EmptyState>}
      </Card>
      <Card><CardHeading title="Recent deployment manifests" subtitle="Latest records from the deployment API" action={<Rocket size={16} className="muted-icon" />} />
        {deployments.length ? <div className="deployment-list">{deployments.slice(0, 4).map((deployment, index) => <div className="deployment-row" key={String(deployment.id ?? index)}><div className="deployment-glyph"><GitBranch size={15} /></div><div className="deployment-info"><b>{deployment.name || deployment.model || `Deployment ${deployment.id}`}</b><span>{deployment.model || 'Model not specified'}</span></div><StatusPill value={String(deployment.status || 'Unknown')} /></div>)}</div>
          : <EmptyState icon={Rocket} title="No deployments returned">Deployment records will appear here when available from the backend.</EmptyState>}
      </Card>
    </div>
    <FraudInference />
    <div className="dashboard-bottom"><span><Zap size={14} /> API metrics endpoint returned {apiMetricCount} samples</span><span>{summary ? 'Summary loaded from backend' : 'Waiting for metrics summary'}</span></div>
  </>
}

function FraudInference() {
  const [amount, setAmount] = useState('')
  const [merchantRisk, setMerchantRisk] = useState('')
  const [transactionHour, setTransactionHour] = useState('')
  const [transactionVelocity, setTransactionVelocity] = useState('')
  const [international, setInternational] = useState(false)
  const [trainingMetrics, setTrainingMetrics] = useState<Record<string, unknown> | null>(null)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    void api<Record<string, unknown>>('/api/inference/fraud/metrics').then(setTrainingMetrics).catch(() => setTrainingMetrics(null))
  }, [])
  async function submit(event: React.FormEvent) {
    event.preventDefault()
    setLoading(true); setError(''); setResult(null)
    try {
      setResult(await post('/api/inference/fraud', {
        amount: Number(amount),
        merchant_risk_score: Number(merchantRisk),
        transaction_hour: Number(transactionHour),
        transaction_velocity: Number(transactionVelocity),
        international,
      }) as Record<string, unknown>)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Inference request failed')
    } finally { setLoading(false) }
  }
  return <Card className="fraud-card">
    <CardHeading title="Fraud inference demo" subtitle="Calls POST /api/inference/fraud with the transaction features you provide." action={<span className="subtle-tag">DEMONSTRATION MODEL</span>} />
    <div className="fraud-disclaimer"><AlertCircle size={15} /><span>This backend model is trained on synthetic data. Its output is for demonstration only and is not suitable for real financial decisions.</span></div>
    <form className="fraud-form" onSubmit={submit}>
      <Field label="Amount"><input type="number" min="0" max="10000000" step="0.01" value={amount} onChange={(event) => setAmount(event.target.value)} placeholder="0.00" required /></Field>
      <Field label="Merchant risk score (0–1)"><input type="number" min="0" max="1" step="0.01" value={merchantRisk} onChange={(event) => setMerchantRisk(event.target.value)} placeholder="0.25" required /></Field>
      <Field label="Transaction hour (0–23)"><input type="number" min="0" max="23" value={transactionHour} onChange={(event) => setTransactionHour(event.target.value)} placeholder="14" required /></Field>
      <Field label="Transaction velocity · 24h"><input type="number" min="0" max="100000" value={transactionVelocity} onChange={(event) => setTransactionVelocity(event.target.value)} placeholder="2" required /></Field>
      <label className="fraud-checkbox"><input type="checkbox" checked={international} onChange={(event) => setInternational(event.target.checked)} /><span>International transaction</span></label>
      <LoadingButton className="button-primary" loading={loading}><Zap size={14} /> Run demo inference</LoadingButton>
    </form>
    <ErrorNotice error={error} />
    {result && <div className="fraud-result"><div><span>Prediction</span><b>{String(result.prediction || '—')}</b></div><div><span>Fraud probability</span><b>{typeof result.fraud_probability === 'number' ? `${(result.fraud_probability * 100).toFixed(2)}%` : '—'}</b></div><span className="fraud-model">{String(result.model_version || '')} · {typeof result.latency_seconds === 'number' ? `${(result.latency_seconds * 1000).toFixed(2)} ms` : 'latency unavailable'}</span></div>}
    {trainingMetrics && <div className="fraud-disclaimer"><span>Offline synthetic metrics: accuracy {String(trainingMetrics.accuracy)} · precision {String(trainingMetrics.precision)} · recall {String(trainingMetrics.recall)} · F1 {String(trainingMetrics.f1)}</span></div>}
  </Card>
}

function Registry({ models }: { models: Model[] }) {
  const [query, setQuery] = useState('')
  const filtered = models.filter((model) => `${model.id} ${model.provider} ${model.intended_capability} ${model.model_type}`.toLowerCase().includes(query.toLowerCase()))
  return <>
    <PageHeading eyebrow="WORKSPACE / MODELS" title="Model Registry" description="A catalog of discovered models and their verification, approval, and deployment status." />
    <div className="registry-kpis"><div><span>Models returned</span><b>{models.length}</b></div><div><span>Verified</span><b>{models.filter((item) => item.verified).length}</b></div><div><span>Approved</span><b>{models.filter((item) => item.approved).length}</b></div><div><span>Deployed</span><b>{models.filter((item) => item.deployed).length}</b></div></div>
    <Card className="registry-table-card"><div className="table-toolbar"><div><h3>Model inventory</h3><p>Data from <code>GET /api/llm/models</code></p></div><label className="search-box"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search models..." /></label></div>
      {filtered.length ? <div className="table-scroll"><table><thead><tr>{['MODEL', 'PROVIDER', 'CAPABILITY', 'TYPE', 'AVAILABILITY', 'LAST INFERENCE', 'DISCOVERED', 'VERIFIED', 'APPROVED', 'DEPLOYED'].map((col) => <th key={col}>{col}</th>)}</tr></thead><tbody>{filtered.map((model) => <tr key={model.id}><td><div className="model-cell"><span className="model-symbol"><Boxes size={14} /></span><b>{model.id}</b></div></td><td>{model.provider}</td><td>{model.intended_capability || '—'}</td><td>{model.model_type || '—'}</td><td><StatusPill value={model.availability_status} /></td><td>{model.last_inference_status ? <StatusPill value={model.last_inference_status} /> : '—'}</td><td><StatusPill value={model.discovered} /></td><td><StatusPill value={model.verified} /></td><td><StatusPill value={model.approved} /></td><td><StatusPill value={model.deployed} /></td></tr>)}</tbody></table></div> : <EmptyState icon={query ? Search : Boxes} title={query ? 'No matching models' : 'No models returned'}>{query ? 'Try another search term.' : 'The model registry endpoint returned no models.'}</EmptyState>}
    </Card>
  </>
}

function ModelSelect({ models, value, onChange, chatOnly = false }: { models: Model[]; value: string; onChange: (value: string) => void; chatOnly?: boolean }) {
  return <select value={value} onChange={(event) => onChange(event.target.value)} required>
    <option value="">Select a model</option>{models.filter((model) => !chatOnly || (model.model_type.toLowerCase() === 'chat' && model.availability_status === 'available')).map((model) => <option key={model.id} value={model.id}>{model.id} · {model.provider}</option>)}
  </select>
}

function Playground({ models }: { models: Model[] }) {
  const [model, setModel] = useState('')
  const [prompt, setPrompt] = useState('')
  const [temperature, setTemperature] = useState('0.7')
  const [maxTokens, setMaxTokens] = useState('512')
  const [result, setResult] = useState<ChatResult | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  async function submit(event: React.FormEvent) {
    event.preventDefault(); setLoading(true); setError(''); setResult(null)
    try { setResult(await post('/api/llm/chat', { model, prompt, temperature: Number(temperature), max_tokens: Number(maxTokens) }) as ChatResult) }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Inference request failed') } finally { setLoading(false) }
  }
  return <>
    <PageHeading eyebrow="WORKSPACE / EXPERIMENT" title="LLM Playground" description="Run a prompt against a registered model and inspect the real inference response and performance." />
    <div className="tool-layout"><Card className="tool-form-card"><CardHeading title="Configure inference" subtitle="Calls POST /api/llm/chat" /><form className="form-stack" onSubmit={submit}>
      <Field label="Model"><ModelSelect models={models} value={model} onChange={setModel} chatOnly /></Field>
      {!models.some((item) => item.model_type.toLowerCase() === 'chat' && item.availability_status === 'available') && <div className="field-hint">No available chat models are returned by the registry API.</div>}
      <Field label="Prompt"><textarea className="prompt-input" value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Write a prompt to send to the selected model..." required rows={8} /></Field>
      <div className="field-row"><Field label="Temperature"><input type="number" min="0" max="2" step="0.1" value={temperature} onChange={(event) => setTemperature(event.target.value)} /></Field><Field label="Max tokens"><input type="number" min="1" value={maxTokens} onChange={(event) => setMaxTokens(event.target.value)} /></Field></div>
      <ErrorNotice error={error} /><LoadingButton className="button-primary full-button" loading={loading} disabled={!model || !prompt.trim()}><Send size={15} /> Run inference</LoadingButton>
    </form></Card>
    <Card className="tool-result-card"><CardHeading title="Response" subtitle={result ? `Generated by ${result.model}` : 'Inference output will appear here'} action={result && <span className="live-label"><i /> REAL INFERENCE</span>} />
      {result ? <><div className="response-content">{result.response}</div><div className="result-stats">{[['Latency · local', `${result.latency_seconds.toFixed(2)}s`], ['Input tokens', result.input_tokens ?? 'Not reported'], ['Output tokens', result.output_tokens ?? 'Not reported'], ['Throughput', result.tokens_per_second === null ? 'Not reported' : `${result.tokens_per_second.toFixed(1)} tok/s`]].map(([label, value]) => <div key={String(label)}><span>{label}</span><b>{value}</b></div>)}</div></> : <EmptyState icon={MessageSquareText} title="Ready when you are">Choose a model and send a prompt to see its response here.</EmptyState>}
    </Card></div>
  </>
}

function Field({ label, children }: { label: string; children: React.ReactNode }) { return <label className="field"><span className="field-label">{label}</span>{children}</label> }

function Benchmark() {
  const [prompt, setPrompt] = useState('Explain how you would secure an AWS EKS cluster running GPU-based AI inference workloads.')
  const [result, setResult] = useState<unknown>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  async function submit(event: React.FormEvent) {
    event.preventDefault(); setLoading(true); setError(''); setResult(null)
    try { setResult(await post('/api/benchmark', { prompt })) }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Benchmark request failed') } finally { setLoading(false) }
  }
  const entries = isRecord(result) ? Object.entries(result) : []
  return <>
    <PageHeading eyebrow="WORKSPACE / EVALUATION" title="Model Benchmark" description="Submit one prompt to both local chat models sequentially and compare their real responses." />
    <div className="tool-layout benchmark-layout"><Card className="tool-form-card"><CardHeading title="Benchmark request" subtitle="Calls POST /api/benchmark" /><form className="form-stack" onSubmit={submit}><Field label="Prompt"><textarea className="prompt-input" rows={9} value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Enter the prompt to benchmark..." required /></Field><div className="info-callout"><CircleHelp size={16} /><span>Runs sequentially to reduce local resource contention. Measurements are local observations, not production performance guarantees.</span></div><ErrorNotice error={error} /><LoadingButton className="button-primary full-button" loading={loading} disabled={!prompt.trim()}><Gauge size={15} /> Run benchmark</LoadingButton></form></Card>
      <Card className="tool-result-card"><CardHeading title="Benchmark results" subtitle="Backend response" action={result !== null && <span className="live-label"><i /> API RESULT</span>} />{result ? entries.length ? <div className="benchmark-result-grid">{entries.map(([key, value]) => <div className="benchmark-result-item" key={key}><span>{titleCase(key)}</span><b>{typeof value === 'object' ? pretty(value) : String(value)}</b></div>)}</div> : <pre className="code-output">{pretty(result)}</pre> : <EmptyState icon={Gauge} title="No benchmark run yet">Submit a prompt to request benchmark results from your ModelOps API.</EmptyState>}</Card>
    </div>
  </>
}

type DocumentRecord = { id?: string | number; name?: string; [key: string]: unknown }
function KnowledgeBase({ models }: { models: Model[] }) {
  const [documents, setDocuments] = useState<DocumentRecord[]>([])
  const [embeddingModel, setEmbeddingModel] = useState('')
  const [indexedChunks, setIndexedChunks] = useState(0)
  const [loadingDocs, setLoadingDocs] = useState(true)
  const [name, setName] = useState('')
  const [content, setContent] = useState('')
  const [question, setQuestion] = useState('')
  const [model, setModel] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState<Source[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [querying, setQuerying] = useState(false)
  const loadDocuments = useCallback(async () => {
    setLoadingDocs(true)
    try {
      const response = await api<{ documents?: DocumentRecord[]; embedding_model?: string; indexed_chunks?: number } | DocumentRecord[]>('/api/rag/documents')
      setDocuments(Array.isArray(response) ? response : response.documents || [])
      setEmbeddingModel(Array.isArray(response) ? '' : response.embedding_model || '')
      setIndexedChunks(Array.isArray(response) ? 0 : response.indexed_chunks || 0)
      setError('')
    }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not load documents') } finally { setLoadingDocs(false) }
  }, [])
  useEffect(() => { void loadDocuments() }, [loadDocuments])
  async function addDocument(event: React.FormEvent) {
    event.preventDefault(); setBusy(true); setError('')
    try { await post('/api/rag/documents', { name, content }); setName(''); setContent(''); await loadDocuments() }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not add document') } finally { setBusy(false) }
  }
  async function seedDocuments() {
    setBusy(true); setError('')
    try { await post('/api/rag/seed', {}); await loadDocuments() }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not load starter documents') } finally { setBusy(false) }
  }
  async function queryKnowledge(event: React.FormEvent) {
    event.preventDefault(); setQuerying(true); setError(''); setAnswer(''); setSources([])
    try { const response = await post('/api/rag/query', { question, model }) as { answer: string; sources: Source[] }; setAnswer(response.answer); setSources(response.sources || []) }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Knowledge query failed') } finally { setQuerying(false) }
  }
  return <>
    <PageHeading eyebrow="BUILD & SHIP / RETRIEVAL" title="RAG Knowledge Base" description="Add source material as text and ask grounded questions against your backend's indexed documents." />
    <ErrorNotice error={error} onDismiss={() => setError('')} />
    <div className="knowledge-grid">
      <Card><CardHeading title="Add a document" subtitle="UTF-8 text/Markdown is embedded locally; PDFs are not supported." />
        <form className="form-stack" onSubmit={addDocument}>
          <Field label="Document name"><input value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. Product handbook" required /></Field>
          <Field label="Document content"><textarea className="prompt-input" rows={6} value={content} onChange={(event) => setContent(event.target.value)} placeholder="Paste the text you want to make available for retrieval..." required /></Field>
          <LoadingButton className="button-primary" loading={busy} disabled={!name.trim() || !content.trim()}><Plus size={15} /> Add to knowledge base</LoadingButton>
          <button className="button-secondary" type="button" disabled={busy} onClick={() => void seedDocuments()}><BookOpen size={15} /> Load synthetic starter documents</button>
        </form>
      </Card>
      <Card><CardHeading title="Indexed documents" subtitle={`GET /api/rag/documents · ${indexedChunks} chunks · ${embeddingModel || 'embedding model unavailable'}`} action={<button className="icon-button" onClick={() => void loadDocuments()} aria-label="Refresh documents"><RefreshCw size={15} className={loadingDocs ? 'spin' : ''} /></button>} />
        {loadingDocs ? <div className="loading-state"><LoaderCircle className="spin" size={18} /> Loading documents…</div> : documents.length ? <div className="document-list">{documents.map((document, index) => <div className="document-row" key={String(document.id ?? document.name ?? index)}><div className="document-icon"><FileText size={16} /></div><div><b>{document.name || `Document ${index + 1}`}</b><span>{document.id ? `ID ${document.id} · ${String(document.chunks || 0)} chunks` : 'Available in knowledge base'}</span></div><CheckCircle2 size={16} className="success-icon" /></div>)}</div> : <EmptyState icon={FileText} title="No documents indexed">Add text or load the synthetic starter documents. Embedding requests require the local embedding model to be loaded.</EmptyState>}
      </Card>
    </div>
    <Card className="query-card"><CardHeading title="Ask your knowledge base" subtitle="Calls POST /api/rag/query" /><form className="rag-query-form" onSubmit={queryKnowledge}><Field label="Question"><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question about your documents..." required /></Field><Field label="Model"><ModelSelect models={models} value={model} onChange={setModel} chatOnly /></Field><LoadingButton className="button-primary" loading={querying} disabled={!question.trim() || !model.trim()}><Send size={15} /> Ask</LoadingButton></form>
      {answer && <div className="rag-answer"><div className="answer-label"><Sparkles size={15} /> ANSWER</div><p>{answer}</p>{sources.length > 0 && <div className="source-list"><b>Retrieved sources</b>{sources.map((source, index) => <div className="source-item" key={`${source.document}-${source.chunk}-${index}`}><div><b>{source.document}</b><span>Chunk {source.chunk} · score {Number(source.score).toFixed(3)}</span></div><p>{source.text}</p></div>)}</div>}</div>}
    </Card>
  </>
}

function Reviewer() {
  const [code, setCode] = useState(INSECURE_KUBERNETES_SAMPLE)
  const [format, setFormat] = useState('kubernetes')
  const [findings, setFindings] = useState<Finding[]>([])
  const [model, setModel] = useState('')
  const [latency, setLatency] = useState<number | null>(null)
  const [aiError, setAiError] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  async function submit(event: React.FormEvent) {
    event.preventDefault(); setLoading(true); setError(''); setFindings([]); setAiError('')
    try { const result = await post('/api/ai/review', { code, format }) as { findings: Finding[]; model: string; latency_seconds: number; ai_error?: string }; setFindings(result.findings || []); setModel(result.model); setLatency(result.latency_seconds); setAiError(result.ai_error || '') }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Review request failed') } finally { setLoading(false) }
  }
  return <>
    <PageHeading eyebrow="BUILD & SHIP / QUALITY" title="AI Infrastructure Reviewer" description="Review infrastructure-as-code through the backend review endpoint. Results are labeled by source." />
    <div className="reviewer-layout"><Card className="review-editor"><CardHeading title="Review configuration" subtitle="The code is sent to POST /api/ai/review" /><form className="form-stack" onSubmit={submit}><Field label="Configuration format"><select value={format} onChange={(event) => setFormat(event.target.value)}><option value="terraform">Terraform</option><option value="kubernetes">Kubernetes</option></select></Field><div className="info-callout"><ShieldCheck size={16} /><span>Starter content is intentionally insecure for demonstration. AI findings are advisory and do not replace policy-as-code, security scanners, or human approval.</span></div><Field label="Infrastructure code"><div className="editor-shell"><div className="editor-topline"><span><i className="editor-dot" /> editor</span><span>{code.split('\n').length} lines</span></div><textarea className="code-editor" spellCheck={false} value={code} onChange={(event) => setCode(event.target.value)} placeholder="# Paste infrastructure code to review&#10;resource &quot;...&quot; &quot;...&quot; {&#10;  # Your configuration&#10;}" required /></div></Field><button className="button-secondary" type="button" onClick={() => { setFormat('kubernetes'); setCode(INSECURE_KUBERNETES_SAMPLE) }}>Load intentionally insecure Kubernetes sample</button><ErrorNotice error={error} /><LoadingButton className="button-primary" loading={loading} disabled={!code.trim()}><Sparkles size={15} /> Review configuration</LoadingButton></form></Card>
      <Card className="findings-card"><CardHeading title="Review findings" subtitle={model ? `Review model: ${model}` : 'Findings will appear here'} action={latency !== null && <span className="latency-tag"><Clock3 size={13} /> {latency.toFixed(2)}s</span>} />
        {aiError && <div className="info-callout"><CircleHelp size={16} /><span>{aiError}</span></div>}
        {findings.length ? <><div className="review-source-banner"><ShieldCheck size={15} /><b>Finding source is identified per result</b><span>Deterministic or AI — as returned by API</span></div><div className="finding-list">{findings.map((finding, index) => <div className="finding-item" key={`${finding.title}-${index}`}><div className={`severity-mark ${finding.severity.toLowerCase()}`}><AlertCircle size={15} /></div><div className="finding-copy"><div className="finding-top"><b>{finding.title}</b><span className={`severity-pill severity-${finding.severity.toLowerCase()}`}>{titleCase(finding.severity)}</span></div><p>{finding.explanation}</p><div className="finding-recommendation"><b>Recommendation</b><span>{finding.recommendation}</span></div><span className="finding-source">{finding.source}</span></div></div>)}</div></> : <EmptyState icon={FileCode2} title="Ready for review">Submit infrastructure code to receive review findings from the backend.</EmptyState>}
      </Card></div>
  </>
}

function Deployments({ models, deployments, onRefresh }: { models: Model[]; deployments: Deployment[]; onRefresh: () => void }) {
  const [name, setName] = useState('')
  const [image, setImage] = useState('modelops/inference:1.0.0')
  const [replicas, setReplicas] = useState('1')
  const [port, setPort] = useState('8000')
  const [namespace, setNamespace] = useState('default')
  const [model, setModel] = useState('')
  const [environment, setEnvironment] = useState('development')
  const [computeProfile, setComputeProfile] = useState('cpu-standard')
  const [cpuRequest, setCpuRequest] = useState('500m')
  const [cpuLimit, setCpuLimit] = useState('1')
  const [memoryRequest, setMemoryRequest] = useState('512Mi')
  const [memoryLimit, setMemoryLimit] = useState('1Gi')
  const [gpuCount, setGpuCount] = useState('0')
  const [productionApproval, setProductionApproval] = useState(false)
  const [validated, setValidated] = useState(false)
  const [validation, setValidation] = useState<unknown>(null)
  const [result, setResult] = useState<unknown>(null)
  const [selected, setSelected] = useState<Deployment | null>(null)
  const [manifest, setManifest] = useState<unknown>(null)
  const [details, setDetails] = useState<unknown>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [loadingManifest, setLoadingManifest] = useState(false)
  const deployableModels = models.filter((entry) => entry.model_type === 'chat' || entry.model_type === 'classifier')
  useEffect(() => {
    if (!deployableModels.some((entry) => entry.id === model) && deployableModels.length) setModel(deployableModels[0].id)
  }, [deployableModels, model])
  const config = useMemo(() => ({
    name, image, replicas: Number(replicas), port: Number(port), namespace, model, environment,
    compute_profile: computeProfile, cpu_request: cpuRequest, cpu_limit: cpuLimit,
    memory_request: memoryRequest, memory_limit: memoryLimit, gpu_count: Number(gpuCount),
    production_approval: productionApproval,
  }), [name, image, replicas, port, namespace, model, environment, computeProfile, cpuRequest, cpuLimit, memoryRequest, memoryLimit, gpuCount, productionApproval])
  async function validate(event: React.FormEvent) {
    event.preventDefault(); setBusy(true); setError(''); setResult(null); setValidated(false); setValidation(null)
    try { const response = await post('/api/deployments/validate', config) as { valid?: boolean }; setValidation(response); setValidated(response.valid === true) }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Deployment validation failed') } finally { setBusy(false) }
  }
  async function createDeployment() {
    setBusy(true); setError(''); setResult(null)
    try { setResult(await post('/api/deployments', config)); onRefresh() }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Deployment request failed') } finally { setBusy(false) }
  }
  async function inspect(deployment: Deployment, wantManifest: boolean) {
    setSelected(deployment); setManifest(null); setDetails(null); setError(''); setLoadingManifest(true)
    try {
      if (wantManifest) setManifest(await api(`/api/deployments/${encodeURIComponent(String(deployment.id))}/manifest`))
      else setDetails(await api(`/api/deployments/${encodeURIComponent(String(deployment.id))}`))
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not load deployment details') } finally { setLoadingManifest(false) }
  }
  return <>
    <PageHeading eyebrow="BUILD & SHIP / RELEASES" title="Model Deployment" description="Validate deployment configuration and generate a manifest through the connected backend API." />
    <ErrorNotice error={error} onDismiss={() => setError('')} />
    <div className="deployment-layout"><Card><CardHeading title="New deployment manifest" subtitle="This API validates configuration and stores a manifest; it does not provision a live service." /><form className="form-stack" onSubmit={validate}>
      <Field label="Model"><select value={model} onChange={(event) => { setModel(event.target.value); setValidated(false) }}>{deployableModels.map((entry) => <option key={entry.id} value={entry.id}>{entry.id}</option>)}</select></Field>
      <div className="field-row"><Field label="Environment"><select value={environment} onChange={(event) => { setEnvironment(event.target.value); setValidated(false) }}><option value="development">Development</option><option value="staging">Staging</option><option value="production">Production</option></select></Field><Field label="Compute profile"><select value={computeProfile} onChange={(event) => { setComputeProfile(event.target.value); setValidated(false) }}><option value="cpu-standard">CPU standard</option><option value="gpu-inference">GPU inference</option></select></Field></div>
      <Field label="Deployment name"><input value={name} onChange={(event) => { setName(event.target.value); setValidated(false) }} placeholder="e.g. risk-scoring-prod" pattern="[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?" required /></Field><Field label="Container image"><input value={image} onChange={(event) => { setImage(event.target.value); setValidated(false) }} required /></Field>
      <div className="field-row"><Field label="Replicas"><input type="number" min="1" max="100" value={replicas} onChange={(event) => { setReplicas(event.target.value); setValidated(false) }} required /></Field><Field label="GPU count"><input type="number" min="0" max="4" value={gpuCount} onChange={(event) => { setGpuCount(event.target.value); setValidated(false) }} required /></Field></div>
      <div className="field-row"><Field label="CPU request"><input value={cpuRequest} onChange={(event) => { setCpuRequest(event.target.value); setValidated(false) }} placeholder="500m" required /></Field><Field label="CPU limit"><input value={cpuLimit} onChange={(event) => { setCpuLimit(event.target.value); setValidated(false) }} placeholder="1" required /></Field></div>
      <div className="field-row"><Field label="Memory request"><input value={memoryRequest} onChange={(event) => { setMemoryRequest(event.target.value); setValidated(false) }} placeholder="512Mi" required /></Field><Field label="Memory limit"><input value={memoryLimit} onChange={(event) => { setMemoryLimit(event.target.value); setValidated(false) }} placeholder="1Gi" required /></Field></div>
      <div className="field-row"><Field label="Container port"><input type="number" min="1" max="65535" value={port} onChange={(event) => { setPort(event.target.value); setValidated(false) }} required /></Field><Field label="Namespace"><input value={namespace} onChange={(event) => { setNamespace(event.target.value); setValidated(false) }} pattern="[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?" required /></Field></div>
      {environment === 'production' && <label className="checkbox-field"><input type="checkbox" checked={productionApproval} onChange={(event) => { setProductionApproval(event.target.checked); setValidated(false) }} /> Production approval recorded</label>}
      <div className="info-callout"><CircleHelp size={16} /><span>Production models require explicit server-side allowlisting and this approval flag. GPU access is environment-policy controlled.</span></div>
      <div className="deployment-actions"><LoadingButton className="button-secondary" loading={busy && validation === null} disabled={!name.trim() || !image.trim() || !namespace.trim() || !model}><CheckCircle2 size={15} /> Validate config</LoadingButton><button className="button-primary" type="button" onClick={() => void createDeployment()} disabled={!validated || busy}><FileText size={15} /> {busy && validated ? 'Generating…' : 'Generate manifest'}</button></div></form>
      {validation !== null && <div className="api-result-block"><b>Validation response</b><pre>{pretty(validation)}</pre></div>}{result !== null && <div className="api-result-block success-result"><b>Deployment API response</b><pre>{pretty(result)}</pre></div>}
    </Card>
    <Card className="deployment-list-card"><CardHeading title="Deployments" subtitle="Live records from GET /api/deployments" action={<button className="icon-button" onClick={onRefresh} aria-label="Refresh deployments"><RefreshCw size={15} /></button>} />{deployments.length ? <div className="deployment-list">{deployments.map((deployment, index) => <div className={`deployment-row interactive ${selected?.id === deployment.id ? 'selected-row' : ''}`} key={String(deployment.id ?? index)}><div className="deployment-glyph"><Rocket size={15} /></div><div className="deployment-info" onClick={() => void inspect(deployment, false)}><b>{deployment.name || deployment.model || `Deployment ${deployment.id}`}</b><span>{deployment.model || `ID ${deployment.id}`}</span></div><StatusPill value={String(deployment.status || 'Unknown')} /><button className="icon-button manifest-action" title="View manifest" onClick={() => void inspect(deployment, true)}><FileText size={15} /></button></div>)}</div> : <EmptyState icon={Rocket} title="No deployments returned">Validated deployments from the API will be listed here.</EmptyState>}</Card>
    </div>
    {selected && <Card className="manifest-card"><CardHeading title={manifest !== null ? 'Deployment manifest' : 'Deployment details'} subtitle={`${selected.name || selected.model || 'Deployment'} · ID ${selected.id}`} action={<button className="icon-button" onClick={() => { setSelected(null); setManifest(null); setDetails(null) }} aria-label="Close details"><X size={16} /></button>} />{loadingManifest ? <div className="loading-state"><LoaderCircle size={17} className="spin" /> Loading details from API…</div> : manifest !== null ? <pre className="code-output">{typeof manifest === 'string' ? manifest : pretty(manifest)}</pre> : details !== null && <pre className="code-output">{pretty(details)}</pre>}</Card>}
  </>
}

function parsePrometheus(text: string) {
  return text.split('\n').flatMap((line) => {
    if (!line || line.startsWith('#')) return []
    const match = line.match(/^([a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|NaN|[+-]?Inf)(?:\s+\d+)?$/)
    if (!match || !Number.isFinite(Number(match[2]))) return []
    return [{ name: match[1], value: Number(match[2]) }]
  })
}
function Observability({ summary, metricsText }: { summary: unknown; metricsText: string }) {
  const samples = parsePrometheus(metricsText)
  const fields = isRecord(summary) ? Object.entries(summary) : []
  const chartData = samples.slice(0, 16).map((sample) => ({ metric: sample.name.length > 20 ? `${sample.name.slice(0, 17)}…` : sample.name, value: sample.value }))
  return <>
    <PageHeading eyebrow="OPERATE / TELEMETRY" title="Observability" description="Inspect the latest values and metric samples returned by the connected API." />
    <div className="observability-cards"><Card className="chart-card"><CardHeading title="Prometheus metric samples" subtitle={`GET /metrics · ${samples.length} numeric samples`} action={<span className="live-label"><i /> LIVE API</span>} />{chartData.length ? <div className="chart-container"><ResponsiveContainer width="100%" height="100%"><AreaChart data={chartData} margin={{ top: 10, right: 12, left: 0, bottom: 32 }}><defs><linearGradient id="metricGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#42d2c3" stopOpacity={0.28} /><stop offset="96%" stopColor="#42d2c3" stopOpacity={0} /></linearGradient></defs><CartesianGrid stroke="#253247" strokeDasharray="3 5" vertical={false} /><XAxis dataKey="metric" angle={-25} textAnchor="end" interval={0} tick={{ fill: '#7e8da4', fontSize: 10 }} axisLine={{ stroke: '#28364b' }} tickLine={false} /><YAxis tick={{ fill: '#7e8da4', fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ background: '#111a28', border: '1px solid #2b3b52', borderRadius: 8, color: '#edf3fb' }} /><Area type="monotone" dataKey="value" stroke="#39c8c3" fill="url(#metricGradient)" strokeWidth={2} /></AreaChart></ResponsiveContainer></div> : <EmptyState icon={Activity} title="No numeric metric samples">The Prometheus endpoint returned no numeric samples to chart.</EmptyState>}</Card>
      <Card><CardHeading title="Metrics summary" subtitle="GET /api/metrics/summary" />{fields.length ? <div className="summary-metrics">{fields.map(([key, value]) => <div className="summary-metric" key={key}><span>{titleCase(key)}</span><b>{typeof value === 'object' ? pretty(value) : String(value)}</b></div>)}</div> : <EmptyState icon={Activity} title="No summary data">No metrics summary values have been returned by the API.</EmptyState>}</Card></div>
    <Card className="raw-metrics"><CardHeading title="Raw Prometheus output" subtitle="Returned verbatim by GET /metrics" />{metricsText ? <pre className="code-output">{metricsText}</pre> : <EmptyState icon={FileText} title="No metrics response">The metrics endpoint has not returned content.</EmptyState>}</Card>
  </>
}

function Governance({ models, deployments }: { models: Model[]; deployments: Deployment[] }) {
  const approvals = models.filter((model) => model.approved)
  const unverified = models.filter((model) => !model.verified)
  const [events, setEvents] = useState<Record<string, unknown>[]>([])
  const [auditError, setAuditError] = useState('')
  useEffect(() => {
    void api<{ events?: Record<string, unknown>[] }>('/api/audit')
      .then((response) => { setEvents(response.events || []); setAuditError('') })
      .catch((caught) => setAuditError(caught instanceof Error ? caught.message : 'Audit events unavailable'))
  }, [])
  return <>
    <PageHeading eyebrow="OPERATE / RESPONSIBLE AI" title="Governance & Audit" description="A read-only view of model lifecycle flags and deployment records returned by the backend." />
    <div className="governance-callout"><ShieldCheck size={19} /><div><b>Governance status from source data</b><span>Model lifecycle flags and metadata-only audit events are read from the local API. No prompt or infrastructure source code is stored in audit records.</span></div></div>
    <div className="registry-kpis"><div><span>Registered models</span><b>{models.length}</b></div><div><span>Approved models</span><b>{approvals.length}</b></div><div><span>Awaiting verification</span><b>{unverified.length}</b></div><div><span>Deployment records</span><b>{deployments.length}</b></div></div>
    <Card><CardHeading title="Model lifecycle review" subtitle="Current registry flags" />{models.length ? <div className="table-scroll"><table><thead><tr><th>MODEL</th><th>DISCOVERED</th><th>VERIFIED</th><th>APPROVED</th><th>DEPLOYED</th><th>LAST INFERENCE</th></tr></thead><tbody>{models.map((model) => <tr key={model.id}><td><div className="model-cell"><span className="model-symbol"><Boxes size={14} /></span><b>{model.id}</b></div></td><td><StatusPill value={model.discovered} /></td><td><StatusPill value={model.verified} /></td><td><StatusPill value={model.approved} /></td><td><StatusPill value={model.deployed} /></td><td>{model.last_inference_status ? <StatusPill value={model.last_inference_status} /> : '—'}</td></tr>)}</tbody></table></div> : <EmptyState icon={ShieldCheck} title="No governance records">Model lifecycle information is not available from the registry response.</EmptyState>}</Card>
    <Card><CardHeading title="Recent audit events" subtitle="Metadata-only events from GET /api/audit" />{auditError ? <ErrorNotice error={auditError} /> : events.length ? <div className="table-scroll"><table><thead><tr><th>TIME</th><th>ACTION</th><th>MODEL</th><th>STATUS</th><th>LATENCY</th></tr></thead><tbody>{events.map((event, index) => <tr key={`${String(event.created_at)}-${index}`}><td>{String(event.created_at || '—')}</td><td>{String(event.action || '—')}</td><td>{String(event.model || '—')}</td><td><StatusPill value={String(event.status || 'unknown')} /></td><td>{typeof event.latency_seconds === 'number' ? `${event.latency_seconds.toFixed(3)} s` : '—'}</td></tr>)}</tbody></table></div> : <EmptyState icon={Activity} title="No audit events">Audit events appear after platform actions; prompts and code are not recorded.</EmptyState>}</Card>
    <Card className="governance-deployments"><CardHeading title="Deployment records" subtitle="Records returned by GET /api/deployments" />{deployments.length ? <pre className="code-output">{pretty(deployments)}</pre> : <EmptyState icon={Rocket} title="No deployment records">Deployment information will appear when returned by the API.</EmptyState>}</Card>
  </>
}

export default App
