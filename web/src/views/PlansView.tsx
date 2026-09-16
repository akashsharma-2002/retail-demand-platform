import { useCallback, useEffect, useState } from "react";
import { api, post, type Plan } from "../api";
import { hasRole, keycloak } from "../auth";
import { money, statusLabel } from "../format";

export default function PlansView({ store }: { store: string }) {
  const [plans, setPlans] = useState<Plan[]>([]);
  const [selected, setSelected] = useState<Plan | null>(null);
  const [budget, setBudget] = useState("");
  const [demand, setDemand] = useState("0");
  const [lead, setLead] = useState("0");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const me = keycloak.tokenParsed?.preferred_username;

  const refresh = useCallback(() => {
    api<Plan[]>(`/v1/plans?store_id=${store}`).then(setPlans).catch((e: Error) => setMessage({ kind: "error", text: e.message }));
  }, [store]);
  useEffect(refresh, [refresh]);

  const open = (id: string) => api<Plan>(`/v1/plans/${id}`).then(setSelected);

  const create = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const plan = await post<Plan>("/v1/plans", {
        store_id: store,
        budget: budget ? Number(budget) : null,
        demand_change_pct: Number(demand),
        lead_time_change_days: Number(lead),
      });
      const replaced = plan.superseded_plans ? ` ${plan.superseded_plans} older waiting plan(s) marked superseded.` : "";
      setMessage({ kind: "ok", text: `Plan saved: ${plan.lines_ordered} items, ${money(plan.total_cost)}. It now waits for an approver.${replaced}` });
      refresh();
      open(plan.plan_id);
    } catch (e) {
      setMessage({ kind: "error", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  };

  const decide = async (plan: Plan, approve: boolean) => {
    try {
      const out = await post<Plan>(`/v1/plans/${plan.plan_id}/${approve ? "approve" : "reject"}`, {});
      const replaced = out.superseded_plans ? ` ${out.superseded_plans} other waiting plan(s) marked superseded, because stock on order changed.` : "";
      setMessage({ kind: "ok", text: `Plan ${statusLabel[out.status].toLowerCase()}.${replaced}` });
      refresh();
      open(plan.plan_id);
    } catch (e) {
      setMessage({ kind: "error", text: (e as Error).message });
    }
  };

  return (
    <section className="split">
      <div>
        {hasRole("planner") && (
          <div className="card">
            <h3>New plan for {store}</h3>
            <div className="row wrap">
              <label className="field">Weekly budget ($)<input id="plan-budget" type="number" min="1" placeholder="Default: forecast cost + 15%" value={budget} onChange={(e) => setBudget(e.target.value)} /></label>
              <label className="field">Demand change (%)<input id="plan-demand" type="number" value={demand} onChange={(e) => setDemand(e.target.value)} /></label>
              <label className="field">Lead time change (days)<input id="plan-lead" type="number" value={lead} onChange={(e) => setLead(e.target.value)} /></label>
            </div>
            <button onClick={create} disabled={busy}>{busy ? "Solving…" : "Create plan"}</button>
          </div>
        )}
        {message && <div className={`banner ${message.kind}`}>{message.text}</div>}
        <div className="card">
          <h3>Plans</h3>
          {plans.length === 0 && <p className="muted">No plans for {store} yet.</p>}
          <ul className="plan-list">
            {plans.map((p) => (
              <li key={p.plan_id}>
                <button className={selected?.plan_id === p.plan_id ? "plan-row selected" : "plan-row"} onClick={() => open(p.plan_id)}>
                  <span className={`pill ${p.status}`}>{statusLabel[p.status]}</span>
                  <span className="num">{money(p.total_cost)}</span>
                  <span className="muted">{p.created_by} · {new Date(p.created_at).toLocaleString()}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
      <div>
        {selected && (
          <div className="card">
            <h3>Plan {selected.plan_id.slice(0, 8)}</h3>
            <dl className="facts">
              <dt>Status</dt><dd><span className={`pill ${selected.status}`}>{statusLabel[selected.status]}</span></dd>
              <dt>Cost</dt><dd>{money(selected.total_cost)} of {money(selected.budget)}</dd>
              <dt>Items</dt><dd>{selected.lines_ordered} items, {selected.total_units.toLocaleString()} units</dd>
              <dt>Solver</dt><dd>{selected.solver_status}</dd>
              <dt>Created by</dt><dd>{selected.created_by}</dd>
              {selected.decided_by && (<><dt>Decided by</dt><dd>{selected.decided_by}</dd></>)}
            </dl>
            {selected.status === "pending_approval" && hasRole("approver") && (
              selected.created_by === me
                ? <p className="muted">You created this plan, so someone else has to approve it.</p>
                : <div className="row"><button onClick={() => decide(selected, true)}>Approve</button><button className="secondary" onClick={() => decide(selected, false)}>Reject</button></div>
            )}
            <div className="table-wrap">
              <table>
                <thead><tr><th>Item</th><th className="num">Position</th><th className="num">Need</th><th className="num">Order</th><th className="num">Cost</th></tr></thead>
                <tbody>
                  {(selected.lines ?? []).slice(0, 200).map((l) => (
                    <tr key={l.item_id}><td>{l.item_id}</td><td className="num">{l.position}</td><td className="num">{l.need}</td><td className="num">{l.units}</td><td className="num">{money(l.cost)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
