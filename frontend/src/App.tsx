import { useCallback, useEffect, useMemo, useState } from 'react';

type User = { id: number; email: string; display_name: string };
type Service = { id: number; name: string; category: string; description: string; relevance_score: number };
type Offer = { id: number; agent_role: string; amount: number; message: string };
type Negotiation = { id: number; service_id: number; status: string; budget_limit: number; selected_offer_id: number | null; offers: Offer[] };
type Execution = { id: number; session_id: number; status: string; result_details: string };
type Verification = { id: number; execution_id: number; verified: boolean; details: string };
type IntentPlan = { intent: string; category: string; constraints: string[]; missing_information: string[]; summary: string };
type RetrievedAPI = { document_id: string; description: string; api_text: string; ranking_score: number };
type RetrievalResponse = { query: string; results: RetrievedAPI[] };

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem('nexustwin_token') || '');
  const [user, setUser] = useState<User | null>(null);
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [services, setServices] = useState<Service[]>([]);
  const [query, setQuery] = useState('');
  const [budget, setBudget] = useState('10000');
  const [category, setCategory] = useState('');
  const [negotiations, setNegotiations] = useState<Negotiation[]>([]);
  const [activeNegotiation, setActiveNegotiation] = useState<Negotiation | null>(null);
  const [execution, setExecution] = useState<Execution | null>(null);
  const [verification, setVerification] = useState<Verification | null>(null);
  const [preferenceKey, setPreferenceKey] = useState('currency');
  const [preferenceValue, setPreferenceValue] = useState('INR');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [intentMessage, setIntentMessage] = useState('Find affordable accommodation near the airport');
  const [intentPlan, setIntentPlan] = useState<IntentPlan | null>(null);
  const [retrievalQuery, setRetrievalQuery] = useState('Find rental cars in a city');
  const [retrievalResults, setRetrievalResults] = useState<RetrievedAPI[]>([]);
  const [retrievalBusy, setRetrievalBusy] = useState(false);
  const [retrievalSearched, setRetrievalSearched] = useState(false);

  const api = useCallback(async <T,>(path: string, init: RequestInit = {}): Promise<T> => {
    const response = await fetch(API + path, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: 'Bearer ' + token } : {}),
        ...init.headers,
      },
    });
    const data = response.status === 204 ? null : await response.json().catch(() => ({}));
    if (!response.ok) throw new Error((data as { detail?: string }).detail || 'Request failed (' + response.status + ')');
    return data as T;
  }, [token]);

  const refreshServices = useCallback(async () => {
    const params = new URLSearchParams();
    if (query.trim()) params.set('query', query.trim());
    if (category) params.set('category', category);
    setServices(await api<Service[]>('/api/v1/services?' + params.toString()));
  }, [api, query, category]);

  useEffect(() => {
    if (!token) return;
    api<User>('/api/v1/users/me').then(setUser).catch(() => {
      localStorage.removeItem('nexustwin_token');
      setToken('');
      setUser(null);
    });
  }, [api, token]);

  useEffect(() => {
    if (!token || !user) return;
    refreshServices().catch(error => setMessage(error.message));
  }, [token, user, refreshServices]);

  const stats = useMemo(() => ({
    services: services.length,
    negotiations: negotiations.length,
    accepted: negotiations.filter(item => item.status === 'ACCEPTED').length,
  }), [services, negotiations]);

  async function authenticate(event: React.FormEvent) {
    event.preventDefault(); setBusy(true); setMessage('');
    try {
      if (mode === 'register') {
        await api<User>('/api/v1/auth/register', { method: 'POST', body: JSON.stringify({ email, display_name: name, password }) });
      }
      const result = await api<{ access_token: string }>('/api/v1/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) });
      localStorage.setItem('nexustwin_token', result.access_token);
      setToken(result.access_token);
      setMessage('Signed in successfully.');
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Authentication failed.'); }
    finally { setBusy(false); }
  }

  function signOut() {
    localStorage.removeItem('nexustwin_token'); setToken(''); setUser(null);
    setServices([]); setNegotiations([]); setActiveNegotiation(null); setExecution(null); setVerification(null);
  }

  async function rankAPIs(event: React.FormEvent) {
    event.preventDefault();
    setRetrievalBusy(true);
    setMessage('');
    try {
      const result = await api<RetrievalResponse>('/api/v1/retrieval/rank', {
        method: 'POST',
        body: JSON.stringify({ query: retrievalQuery.trim(), top_k: 5 }),
      });
      setRetrievalResults(result.results);
      setRetrievalSearched(true);
      setMessage(
        result.results.length > 0
          ? `Ranked ${result.results.length} API candidates.`
          : 'No suitable API found in the current corpus.'
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'API retrieval failed.');
      setRetrievalResults([]);
      setRetrievalSearched(false);
    } finally {
      setRetrievalBusy(false);
    }
  }

  async function discover() {
    setBusy(true);
    try { await refreshServices(); setMessage('Service discovery refreshed.'); }
    catch (error) { setMessage(error instanceof Error ? error.message : 'Could not discover services.'); }
    finally { setBusy(false); }
  }

  async function startNegotiation(service: Service) {
    setBusy(true); setMessage(''); setExecution(null); setVerification(null);
    try {
      const created = await api<Negotiation>('/api/v1/negotiations', { method: 'POST', body: JSON.stringify({ service_id: service.id, budget_limit: Number(budget), max_offers: 3 }) });
      setActiveNegotiation(created); setNegotiations(old => [created, ...old.filter(item => item.id !== created.id)]);
      setMessage('Negotiation #' + created.id + ' started. Review the simulated offers below.');
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Could not start negotiation.'); }
    finally { setBusy(false); }
  }

  async function decide(status: 'ACCEPTED' | 'REJECTED', offerId?: number) {
    if (!activeNegotiation) return;
    setBusy(true);
    try {
      const updated = await api<Negotiation>('/api/v1/negotiations/' + activeNegotiation.id + '/decision', {
        method: 'POST', body: JSON.stringify({ status, selected_offer_id: offerId ?? null }),
      });
      setActiveNegotiation(updated); setNegotiations(old => [updated, ...old.filter(item => item.id !== updated.id)]);
      setMessage(status === 'ACCEPTED' ? 'Offer accepted. You can now run a local simulation.' : 'Offers rejected.');
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Decision failed.'); }
    finally { setBusy(false); }
  }

  async function execute() {
    if (!activeNegotiation) return;
    setBusy(true); setVerification(null);
    try {
      const result = await api<Execution>('/api/v1/negotiations/' + activeNegotiation.id + '/execute', { method: 'POST' });
      setExecution(result); setMessage('Simulation recorded. Verify the saved outcome next.');
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Execution failed.'); }
    finally { setBusy(false); }
  }

  async function verify() {
    if (!execution) return;
    setBusy(true);
    try {
      const result = await api<Verification>('/api/v1/executions/' + execution.id + '/verify', { method: 'POST' });
      setVerification(result); setMessage('Verification finished.');
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Verification failed.'); }
    finally { setBusy(false); }
  }

  async function interpretIntent(event: React.FormEvent) {
    event.preventDefault(); setBusy(true); setIntentPlan(null); setMessage('');
    try {
      const plan = await api<IntentPlan>('/api/v1/assistant/interpret', {
        method: 'POST',
        body: JSON.stringify({ message: intentMessage, budget_limit: Number(budget) > 0 ? Number(budget) : undefined }),
      });
      setIntentPlan(plan);
      setMessage('The local LLM interpreted your request. No action was executed.');
    } catch (error) { setMessage(error instanceof Error ? error.message : 'LLM interpretation failed.'); }
    finally { setBusy(false); }
  }

  async function savePreference(event: React.FormEvent) {
    event.preventDefault();
    if (!user) return;
    setBusy(true);
    try {
      await api('/api/v1/users/' + user.id + '/preferences/' + encodeURIComponent(preferenceKey), {
        method: 'PUT',
        body: JSON.stringify({ preference_key: preferenceKey, preference_value: preferenceValue, is_sensitive: false }),
      });
      setMessage('Preference saved to your digital twin.');
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Preference save failed.'); }
    finally { setBusy(false); }
  }

  if (!token || !user) {
    return <main className="auth-shell">
      <section className="auth-panel">
        <div className="brand-mark">N<span>✶</span></div>
        <p className="eyebrow">YOUR PERSONAL AI SYSTEM</p>
        <h1>Make decisions.<br /><em>With a twin.</em></h1>
        <p className="muted">A preference-aware digital twin to discover services, compare agent offers, and verify every simulated action.</p>
        <div className="auth-tabs">
          <button className={mode === 'login' ? 'tab active' : 'tab'} onClick={() => setMode('login')}>Sign in</button>
          <button className={mode === 'register' ? 'tab active' : 'tab'} onClick={() => setMode('register')}>Create account</button>
        </div>
        <form onSubmit={authenticate} className="stack">
          {mode === 'register' && <label>Display name<input value={name} onChange={e => setName(e.target.value)} required minLength={1} maxLength={100} placeholder="How should we address you?" /></label>}
          <label>Email address<input type="email" value={email} onChange={e => setEmail(e.target.value)} required placeholder="you@example.com" /></label>
          <label>Password<input type="password" value={password} onChange={e => setPassword(e.target.value)} required minLength={mode === 'register' ? 12 : 1} maxLength={128} placeholder={mode === 'register' ? 'At least 12 characters' : 'Your password'} /></label>
          <button className="primary" disabled={busy}>{busy ? 'Please wait…' : mode === 'login' ? 'Sign in to NexusTwin ↗' : 'Create your twin ↗'}</button>
        </form>
        {message && <p className="notice">{message}</p>}
        <p className="fine-print">API: {API} · No real-world transactions are performed.</p>
      </section>
      <aside className="auth-art"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orbit-core">N<span>✶</span></div><div className="art-copy"><span className="pill">PERSONAL DIGITAL TWIN</span><h2>Your intent.<br />Your boundaries.<br /><em>Your decision.</em></h2><p>Multiple agents. One clear outcome.</p></div><span className="art-index">01 / AUTONOMY WITH OVERSIGHT</span></aside>
    </main>;
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-small">N<span>✶</span></div><div><strong>NexusTwin</strong><small>PERSONAL AI</small></div></div>
      <div className="nav-label">WORKSPACE</div>
      <a className="nav-item selected" href="#overview">◈ <span>Overview</span></a>
      <a className="nav-item" href="#ai-intent">✶ <span>AI intent</span></a><a className="nav-item" href="#discover">⌕ <span>Discover services</span></a>
      <a className="nav-item" href="#negotiation">⇄ <span>Negotiation lab</span></a>
      <a className="nav-item" href="#preferences">◎ <span>Twin preferences</span></a>
      <div className="sidebar-bottom"><div className="privacy-card"><span className="status-dot" /> <strong>Privacy guard active</strong><p>Sensitive preferences are excluded from ranking by default.</p></div><div className="user-chip"><div className="avatar">{user.display_name.slice(0, 1).toUpperCase()}</div><div className="user-info"><strong>{user.display_name}</strong><small>{user.email}</small></div><button className="icon-button" title="Sign out" onClick={signOut}>↗</button></div></div>
    </aside>
    <main className="main-content">
      <header className="topbar"><div><span className="breadcrumb">Workspace</span><span className="slash"> / </span><strong>Overview</strong></div><div className="top-right"><span className="live-dot" /> SYSTEM ONLINE <span className="top-divider" /> <span className="user-initial">{user.display_name.slice(0, 1).toUpperCase()}</span></div></header>
      <section id="overview" className="welcome"><div><p className="eyebrow">THURSDAY · YOUR DIGITAL TWIN IS READY</p><h1>Good to have you here, <em>{user.display_name.split(' ')[0]}.</em></h1><p className="muted">Your preferences shape the search. You stay in control of every decision.</p></div><div className="twin-badge"><span className="pulse" /> TWIN ACTIVE <small>Personal context loaded</small></div></section>
      {message && <div className="toast" role="status"><span>✶</span>{message}<button onClick={() => setMessage('')} aria-label="Dismiss">×</button></div>}
      <section className="stat-grid">
        <div className="stat-card"><div className="stat-top"><span>AVAILABLE SERVICES</span><span className="stat-icon">⌕</span></div><strong>{stats.services.toString().padStart(2, '0')}</strong><small>Matching current search</small></div>
        <div className="stat-card"><div className="stat-top"><span>NEGOTIATIONS</span><span className="stat-icon">⇄</span></div><strong>{stats.negotiations.toString().padStart(2, '0')}</strong><small>Started this session</small></div>
        <div className="stat-card"><div className="stat-top"><span>ACCEPTED OFFERS</span><span className="stat-icon">✓</span></div><strong>{stats.accepted.toString().padStart(2, '0')}</strong><small>Awaiting or completed simulation</small></div>
      </section>
      <section id="ai-intent" className="section-card"><div className="section-heading"><div><p className="eyebrow">00 / LOCAL LLM REASONING</p><h2>Talk to your digital twin</h2><p className="muted">Interpret your request with Ollama running locally. This does not execute bookings or payments.</p></div><span className="count-tag">OLLAMA · LOCAL</span></div><form className="intent-form" onSubmit={interpretIntent}><label>Your request<textarea value={intentMessage} onChange={e => setIntentMessage(e.target.value)} required minLength={3} maxLength={4000} rows={3} placeholder="Describe what you need in plain English…" /></label><button className="primary compact" disabled={busy}>Interpret with LLM ↗</button></form>{intentPlan && <div className="intent-result"><div><small>DETECTED INTENT</small><strong>{intentPlan.intent}</strong></div><div><small>CATEGORY</small><strong>{intentPlan.category}</strong></div><div><small>SUMMARY</small><p>{intentPlan.summary}</p></div><div><small>CONSTRAINTS</small><p>{intentPlan.constraints.length ? intentPlan.constraints.join(' · ') : 'None extracted'}</p></div><div><small>MISSING INFORMATION</small><p>{intentPlan.missing_information.length ? intentPlan.missing_information.join(' · ') : 'No essential details flagged'}</p></div></div>}</section>
      <section id="api-discovery" className="section-card">
        <div className="section-heading">
          <div>
            <p className="eyebrow">01 / TRAINED MODEL RETRIEVAL</p>
            <h2>AI API Discovery</h2>
            <p className="muted">Search API documentation using your trained neural ranker and metadata signals.</p>
          </div>
          <span className="count-tag">MODEL POWERED</span>
        </div>

        <form className="intent-form" onSubmit={rankAPIs}>
          <label>
            What capability do you need?
            <textarea
              value={retrievalQuery}
              onChange={e => setRetrievalQuery(e.target.value)}
              required
              minLength={3}
              maxLength={4000}
              rows={2}
              placeholder="Example: Find rental cars available in a city"
            />
          </label>
          <button className="primary compact" disabled={retrievalBusy || !retrievalQuery.trim()}>
            {retrievalBusy ? 'Ranking APIs...' : 'Find matching APIs ↗'}
          </button>
        </form>

        <div className="service-list">
          {retrievalResults.map((item, index) => (
            <article className="service-row" key={item.document_id}>
              <div className="service-symbol">{String(index + 1).padStart(2, '0')}</div>
              <div className="service-copy">
                <div className="service-title">{item.description || 'API documentation'}</div>
                <p>{item.api_text || 'No API details available.'}</p>
                <small>Document ID: {item.document_id}</small>
              </div>
              <div className="relevance">
                <small>RANKING SCORE</small>
                <strong>{item.ranking_score.toFixed(4)}</strong>
              </div>
            </article>
          ))}

          {retrievalResults.length === 0 && (
            <div className="empty">
              <span>⌕</span>
              <strong>
                {retrievalSearched
                  ? 'No suitable API found'
                  : 'No API candidates yet'}
              </strong>
              <p>
                {retrievalSearched
                  ? 'The current API corpus has no verified match for this request. Try another query or add documentation for a compatible API.'
                  : 'Enter a capability above to retrieve API candidates.'}
              </p>
            </div>
          )}
        </div>

        <p className="fine-print">
          Scores are ranking signals, not probabilities. Retrieved documentation is not
          proof that an API is available, compatible, or safe to execute. Execution remains
          a separate, explicitly controlled simulation.
        </p>
      </section>
      <section id="discover" className="section-card"><div className="section-heading"><div><p className="eyebrow">01 / SERVICE INTELLIGENCE</p><h2>Discover services</h2><p className="muted">Explore active services ranked against non-sensitive preferences.</p></div><span className="count-tag">{services.length} RESULTS</span></div>
        <div className="search-row"><div className="search-input"><span>⌕</span><input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && discover()} placeholder="Search services, categories, keywords…" /></div><select value={category} onChange={e => setCategory(e.target.value)}><option value="">All categories</option><option>Accommodation</option><option>Travel</option><option>Shopping</option><option>Transport</option><option>Food</option></select><button className="primary compact" disabled={busy} onClick={discover}>Search ↗</button></div>
        <div className="service-list">{services.map(service => <article className="service-row" key={service.id}><div className="service-symbol">{service.category.toLowerCase().includes('travel') ? '✈' : service.category.toLowerCase().includes('accommodation') ? '⌂' : '◇'}</div><div className="service-copy"><div className="service-title">{service.name}<span className="category-tag">{service.category}</span></div><p>{service.description || 'No description provided.'}</p></div><div className="relevance"><small>RELEVANCE</small><strong>{Math.round(service.relevance_score * 100)}%</strong><div className="meter"><span style={{ width: Math.round(service.relevance_score * 100) + '%' }} /></div></div><button className="outline compact" disabled={busy} onClick={() => startNegotiation(service)}>Compare offers ↗</button></article>)}
          {services.length === 0 && <div className="empty"><span>⌕</span><strong>No services in this view</strong><p>Try a different keyword or clear the category filter.</p></div>}
        </div>
      </section>
      <section id="negotiation" className="section-card"><div className="section-heading"><div><p className="eyebrow">02 / MULTI-AGENT COORDINATION</p><h2>Negotiation lab</h2><p className="muted">Compare simulated agent offers and explicitly choose your next step.</p></div><span className="count-tag">HUMAN IN CONTROL</span></div>
        <div className="budget-row"><label>Maximum budget <div className="budget-input"><span>₹</span><input type="number" min="1" value={budget} onChange={e => setBudget(e.target.value)} /></div></label>{activeNegotiation && <span className={'status-tag ' + activeNegotiation.status.toLowerCase()}>{activeNegotiation.status}</span>}</div>
        {activeNegotiation ? <><div className="negotiation-meta"><span>NEGOTIATION #{activeNegotiation.id}</span><span>Budget cap: ₹{activeNegotiation.budget_limit.toLocaleString('en-IN')}</span></div><div className="offer-grid">{activeNegotiation.offers.map(offer => <article className={'offer-card ' + (activeNegotiation.selected_offer_id === offer.id ? 'chosen' : '')} key={offer.id}><div className="offer-agent"><span className="agent-icon">{offer.agent_role.startsWith('BUDGET') ? '◉' : offer.agent_role.startsWith('POLICY') ? '⌘' : '◇'}</span><span>{offer.agent_role.replace('_', ' ')}</span></div><strong className="offer-amount">₹{offer.amount.toLocaleString('en-IN')}</strong><p>{offer.message}</p><span className="offer-caption">{offer.amount <= activeNegotiation.budget_limit ? 'WITHIN BUDGET' : 'OVER BUDGET'}</span>{activeNegotiation.status === 'NEGOTIATING' && <button className="outline full" disabled={busy || offer.amount > activeNegotiation.budget_limit} onClick={() => decide('ACCEPTED', offer.id)}>Accept this offer</button>}{activeNegotiation.selected_offer_id === offer.id && <span className="chosen-label">✓ SELECTED OFFER</span>}</article>)}</div>
          {activeNegotiation.status === 'NEGOTIATING' && <button className="text-button" disabled={busy} onClick={() => decide('REJECTED')}>Reject all offers</button>}
          {activeNegotiation.status === 'ACCEPTED' && <div className="action-strip"><div><strong>Offer accepted — ready for simulation</strong><p>This only records a local simulated action, not a booking or payment.</p></div><button className="primary" disabled={busy} onClick={execute}>Run simulation ↗</button></div>}
          {execution && <div className="execution-panel"><div className="execution-heading"><div><p className="eyebrow">03 / EXECUTION RECORD</p><h3>Simulation #{execution.id}</h3></div><span className="status-tag accepted">{execution.status}</span></div><pre>{JSON.stringify(JSON.parse(execution.result_details), null, 2)}</pre><button className="primary compact" disabled={busy} onClick={verify}>Verify outcome ↗</button>{verification && <div className={'verification ' + (verification.verified ? 'verified' : 'failed')}><span>{verification.verified ? '✓' : '!'}</span><div><strong>{verification.verified ? 'Verification passed' : 'Verification failed'}</strong><p>{verification.details}</p></div></div>}</div>}
        </> : <div className="empty"><span>⇄</span><strong>No active negotiation selected</strong><p>Choose “Compare offers” on a service above to start.</p></div>}
      </section>
      <section id="preferences" className="section-card"><div className="section-heading"><div><p className="eyebrow">04 / PERSONAL CONTEXT</p><h2>Twin preferences</h2><p className="muted">Store a preference to personalize service ranking.</p></div><span className="count-tag">PRIVATE TO YOU</span></div><form className="preference-form" onSubmit={savePreference}><label>Preference key<input value={preferenceKey} onChange={e => setPreferenceKey(e.target.value)} required maxLength={100} /></label><label>Preference value<input value={preferenceValue} onChange={e => setPreferenceValue(e.target.value)} required maxLength={5000} /></label><button className="outline compact" disabled={busy}>Save preference</button></form><p className="fine-print">Mark sensitive data only through a trusted interface; sensitive preferences are not used by service ranking.</p></section>
      <footer><span>© 2026 NEXUSTWIN AI</span><span>PRIVACY-FIRST · HUMAN-APPROVED · VERIFIED</span></footer>
    </main>
  </div>;
}