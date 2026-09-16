import { useState } from "react";
import { post, type ChatResponse } from "../api";
import { hasRole, keycloak } from "../auth";
import { money } from "../format";

type Turn = { question: string; response?: ChatResponse; error?: string };

const EXAMPLES = (store: string) => [
  `Why should we order FOODS_3_090 for ${store}?`,
  `Show me this week's replenishment plan for ${store}`,
  `What if demand rises 15% and supplier lead time is 2 days longer at ${store}?`,
  "Who can approve a plan above the store limit?",
];

export default function AssistantView({ store }: { store: string }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const ask = async (question: string) => {
    if (!question.trim()) return;
    setBusy(true);
    setInput("");
    setTurns((t) => [...t, { question }]);
    try {
      const response = await post<ChatResponse>("/v1/assistant/chat", { message: question, store_id: store });
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, response } : x)));
    } catch (e) {
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, error: (e as Error).message } : x)));
    } finally {
      setBusy(false);
    }
  };

  const decide = async (turn: Turn, approve: boolean) => {
    const r = turn.response!;
    try {
      const response = await post<ChatResponse>("/v1/assistant/resume", { thread_id: r.thread_id, approve });
      setTurns((t) => [...t, { question: approve ? "Approve this plan" : "Reject this plan", response }]);
    } catch (e) {
      setTurns((t) => [...t, { question: "Decision", error: (e as Error).message }]);
    }
  };

  return (
    <section className="assistant">
      <div className="examples">
        {EXAMPLES(store).map((q) => <button key={q} className="chip" onClick={() => ask(q)} disabled={busy}>{q}</button>)}
      </div>
      <div className="conversation">
        {turns.map((t, i) => (
          <div key={i} className="turn">
            <div className="question">{t.question}</div>
            {t.error && <div className="banner error">{t.error}</div>}
            {!t.response && !t.error && <div className="muted">Working…</div>}
            {t.response && (
              <div className="answer card">
                <p>{t.response.answer}</p>
                {t.response.awaiting_approval && (
                  hasRole("approver") && t.response.awaiting_approval.created_by !== keycloak.tokenParsed?.preferred_username ? (
                    <div className="row">
                      <button onClick={() => decide(t, true)}>Approve {money(t.response.awaiting_approval.total_cost)}</button>
                      <button className="secondary" onClick={() => decide(t, false)}>Reject</button>
                    </div>
                  ) : (
                    <p className="muted">Waiting for an approver other than {t.response.awaiting_approval.created_by}.</p>
                  )
                )}
                <div className="meta">
                  {t.response.tools.map((x) => <span key={x.tool} className={x.ok ? "chip small" : "chip small bad"}>{x.tool}</span>)}
                  {t.response.sources.length > 0 && <span className="muted">Sources: {t.response.sources.join("; ")}</span>}
                  <span className="muted">
                    {t.response.guard?.passed === false ? "Numbers checked: rewritten from tool data" : "Numbers checked"} · {t.response.mode} · {t.response.telemetry.latency_ms} ms
                    {t.response.telemetry.llm_calls > 0 && ` · $${t.response.telemetry.cost_usd.toFixed(5)}`}
                  </span>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
      <form className="row composer" onSubmit={(e) => { e.preventDefault(); ask(input); }}>
        <input id="assistant-input" className="grow" placeholder="Ask about a forecast, a plan, a what-if or a policy" value={input} onChange={(e) => setInput(e.target.value)} maxLength={1000} />
        <button type="submit" disabled={busy}>Send</button>
      </form>
    </section>
  );
}
