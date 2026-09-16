import { useEffect, useState } from "react";
import { api } from "./api";
import { hasRole, keycloak, roles } from "./auth";
import AssistantView from "./views/AssistantView";
import ForecastView from "./views/ForecastView";
import PlansView from "./views/PlansView";

type Tab = "forecast" | "plans" | "assistant";

export default function App() {
  const [tab, setTab] = useState<Tab>("forecast");
  const [stores, setStores] = useState<string[]>([]);
  const [store, setStore] = useState("CA_1");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<string[]>("/v1/stores").then(setStores).catch((e: Error) => setError(e.message));
  }, []);

  const tabs: { id: Tab; label: string; show: boolean }[] = [
    { id: "forecast", label: "Forecast & stock", show: true },
    { id: "plans", label: "Order plans", show: true },
    { id: "assistant", label: "Assistant", show: hasRole("planner") || hasRole("approver") },
  ];

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">Replenishment Planner</div>
        <label className="store-picker">
          Store
          <select id="store" value={store} onChange={(e) => setStore(e.target.value)}>
            {stores.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        <div className="user">
          {keycloak.tokenParsed?.preferred_username} · {roles().filter((r) => ["viewer", "planner", "approver", "admin"].includes(r)).join(", ")}
          <button className="link" onClick={() => keycloak.logout()}>Sign out</button>
        </div>
      </header>
      <nav className="tabs">
        {tabs.filter((t) => t.show).map((t) => (
          <button key={t.id} className={tab === t.id ? "tab active" : "tab"} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>
      {error && <div className="banner error">{error}</div>}
      <main>
        {tab === "forecast" && <ForecastView store={store} />}
        {tab === "plans" && <PlansView store={store} />}
        {tab === "assistant" && <AssistantView store={store} />}
      </main>
    </div>
  );
}
