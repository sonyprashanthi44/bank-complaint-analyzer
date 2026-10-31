import { FormEvent, useEffect, useMemo, useState } from "react";
import { analyzeComplaint, getHealth } from "./api";
import type { AnalyzeResponse, HealthResponse } from "./types";

const SAMPLE =
  "A merchant charge of $250 appears twice on my credit card ending in 4412. The merchant confirms only one purchase. I need the duplicate charge removed and any related interest reversed.";

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function friendlyBackend(value: string): string {
  if (value.startsWith("postgres")) return "Vector search";
  if (value.includes("local")) return "Case library";
  return value;
}

function friendlyDraftSource(provider: string, isFallback: boolean): string {
  if (isFallback || provider.includes("template")) return "Standard template";
  if (provider.includes("groq") || provider.includes("openai")) return "AI assistant";
  return provider;
}

export default function App() {
  const [narrative, setNarrative] = useState(SAMPLE);
  const [customerName, setCustomerName] = useState("Alex Rivera");
  const [accountLast4, setAccountLast4] = useState("4412");
  const [channel, setChannel] = useState("web");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const topProbs = useMemo(() => {
    if (!result) return [];
    return result.classification.top_probabilities.map((row) => {
      const [label, score] = Object.entries(row)[0];
      return { label, score };
    });
  }, [result]);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload = await analyzeComplaint({
        narrative,
        customer_name: customerName || undefined,
        account_last4: accountLast4 || undefined,
        channel,
        top_k: 5,
      });
      setResult(payload);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Something went wrong while analyzing.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <div className="topbar">
        <div className="brand">
          <div className="brand-mark">BC</div>
          <div className="brand-copy">
            <strong>Bank Complaint Analyzer</strong>
            <span>Customer care workspace</span>
          </div>
        </div>
        <div className={`pill ${health ? "" : "offline"}`}>
          {health ? "Service online" : "Service offline"}
        </div>
      </div>

      <header className="hero">
        <div className="hero-card">
          <p className="eyebrow">Complaint triage</p>
          <h1>Understand the issue. Match past cases. Draft a careful reply.</h1>
          <p className="lede">
            Drop in a customer complaint and get a product category, similar resolved cases, and a
            first-response draft you can review before anything goes out.
          </p>
        </div>

        <div className="status-card">
          <h2>Service snapshot</h2>
          {health ? (
            <ul>
              <li>
                <span>API</span>
                <strong className="ok">{health.status}</strong>
              </li>
              <li>
                <span>Model</span>
                <strong>{health.classifier_loaded ? "Ready" : "Unavailable"}</strong>
              </li>
              <li>
                <span>Case search</span>
                <strong>{friendlyBackend(health.retrieval_backend)}</strong>
              </li>
              <li>
                <span>Cases available</span>
                <strong>{health.case_count}</strong>
              </li>
              <li>
                <span>Reply style</span>
                <strong>{health.llm_ready ? "AI assisted" : "Standard template"}</strong>
              </li>
            </ul>
          ) : (
            <p className="empty">Can’t reach the API right now. Check the backend and try again.</p>
          )}
        </div>
      </header>

      <main className="grid">
        <section className="panel">
          <div className="panel-head">
            <h2>New complaint</h2>
            <p>Enter the details the customer shared. A sample is loaded so you can try it quickly.</p>
          </div>
          <form onSubmit={onSubmit} className="form">
            <label>
              Customer name
              <input
                value={customerName}
                onChange={(e) => setCustomerName(e.target.value)}
                placeholder="Optional"
              />
            </label>
            <div className="row">
              <label>
                Account last 4
                <input
                  value={accountLast4}
                  onChange={(e) => setAccountLast4(e.target.value.slice(0, 4))}
                  placeholder="4412"
                  maxLength={4}
                />
              </label>
              <label>
                Channel
                <select value={channel} onChange={(e) => setChannel(e.target.value)}>
                  <option value="web">Web</option>
                  <option value="phone">Phone</option>
                  <option value="email">Email</option>
                  <option value="branch">Branch</option>
                </select>
              </label>
            </div>
            <label>
              Complaint narrative
              <textarea
                value={narrative}
                onChange={(e) => setNarrative(e.target.value)}
                rows={8}
                required
                minLength={20}
              />
            </label>
            <div className="actions">
              <button type="button" className="ghost" onClick={() => setNarrative(SAMPLE)}>
                Use sample
              </button>
              <button type="submit" disabled={loading || narrative.trim().length < 20}>
                {loading ? "Working…" : "Analyze"}
              </button>
            </div>
          </form>
          {error && <div className="error">{error}</div>}
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Classification</h2>
            <p>Predicted product area and how confident the model is.</p>
          </div>
          {result ? (
            <div className="stack">
              <div className="metric">
                <span>Product</span>
                <strong>{result.classification.product}</strong>
              </div>
              <div className="metric">
                <span>Confidence</span>
                <strong>{pct(result.classification.confidence)}</strong>
              </div>
              <div className="bars">
                {topProbs.map((item) => (
                  <div key={item.label} className="bar-row">
                    <div className="bar-label">
                      <span>{item.label}</span>
                      <span>{pct(item.score)}</span>
                    </div>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: pct(item.score) }} />
                    </div>
                  </div>
                ))}
              </div>
              <p className="muted">Search mode: {friendlyBackend(result.retrieval_backend)}</p>
            </div>
          ) : (
            <p className="empty">Run an analysis to see the predicted category.</p>
          )}
        </section>

        <section className="panel wide">
          <div className="panel-head">
            <h2>Similar past cases</h2>
            <p>Closest historical complaints and how they were handled.</p>
          </div>
          {result ? (
            <div className="cases">
              {result.similar_cases.map((item) => (
                <article key={item.case_id} className="case-card">
                  <header>
                    <strong>{item.case_id}</strong>
                    <span>{pct(item.similarity)} match</span>
                  </header>
                  <p className="meta">
                    {item.product} · {item.issue}
                  </p>
                  <p>{item.narrative}</p>
                  <p className="resolution">
                    <strong>How it was resolved: </strong>
                    {item.resolution_summary}
                  </p>
                  <p className="muted">Outcome tag: {item.company_response}</p>
                </article>
              ))}
            </div>
          ) : (
            <p className="empty">Matching cases will show up here after you analyze a complaint.</p>
          )}
        </section>

        <section className="panel wide">
          <div className="panel-head">
            <h2>Draft reply</h2>
            <p>A first draft based on the complaint and similar cases. Review it before sending.</p>
          </div>
          {result ? (
            <div className="draft">
              <div className="draft-meta">
                <span className="chip">
                  {friendlyDraftSource(result.draft_reply.provider, result.draft_reply.is_fallback)}
                </span>
                <span className="chip">Tone: clear and professional</span>
                {result.draft_reply.grounded_case_ids.length > 0 && (
                  <span className="chip">
                    Based on {result.draft_reply.grounded_case_ids.join(", ")}
                  </span>
                )}
              </div>
              <h3>{result.draft_reply.subject}</h3>
              <pre>{result.draft_reply.body}</pre>
              {result.notes.length > 0 && (
                <ul className="notes">
                  {result.notes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              )}
            </div>
          ) : (
            <p className="empty">Your draft reply will appear here.</p>
          )}
        </section>
      </main>

      <footer className="footer">
        <p>Built for agent review workflows. Always verify the draft before contacting the customer.</p>
      </footer>
    </div>
  );
}
