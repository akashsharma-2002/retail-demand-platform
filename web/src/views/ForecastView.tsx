import { useEffect, useState } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Forecast } from "../api";
import { money } from "../format";

export default function ForecastView({ store }: { store: string }) {
  const [query, setQuery] = useState("FOODS_3_090");
  const [options, setOptions] = useState<{ item_id: string }[]>([]);
  const [data, setData] = useState<Forecast | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const t = setTimeout(() => {
      api<{ item_id: string }[]>(`/v1/stores/${store}/items?q=${encodeURIComponent(query)}`).then(setOptions).catch(() => setOptions([]));
    }, 200);
    return () => clearTimeout(t);
  }, [query, store]);

  const load = (item: string) => {
    setError(null);
    api<Forecast>(`/v1/forecasts/${store}/${item}`).then(setData).catch((e: Error) => {
      setData(null);
      setError(e.message);
    });
  };

  useEffect(() => load(query), [store]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section>
      <form className="row" onSubmit={(e) => { e.preventDefault(); load(query); }}>
        <label className="field grow">
          Item
          <input id="item-query" list="item-options" value={query} onChange={(e) => setQuery(e.target.value.toUpperCase())} />
          <datalist id="item-options">
            {options.map((o) => <option key={o.item_id} value={o.item_id} />)}
          </datalist>
        </label>
        <button type="submit">Show forecast</button>
      </form>
      {error && <div className="banner error">{error}</div>}
      {data && (
        <>
          <div className="tiles">
            <Tile label="Next 28 days (P50)" value={`${data.total_p50} units`} />
            <Tile label={`Cover period, ${data.cover_days} days`} value={`${data.cover_p50} – ${data.cover_p90} units`} note="P50 – P90" />
            <Tile label="Position" value={`${data.position.position} units`} note={`${data.position.on_hand} on hand, ${data.position.on_order} on order`} />
            <Tile label="Supply" value={`${data.position.lead_time_days} day lead time`} note={`case ${data.position.case_pack}, MOQ ${data.position.moq}, ${money(data.position.unit_cost)} each`} />
          </div>
          <div className="card chart">
            <h3>Daily demand forecast · {data.item_id} · {data.store_id}</h3>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={data.daily} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                <XAxis dataKey="day" tickFormatter={(d: string) => d.slice(5)} tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v) => [`${v} units`, "P50"]} />
                <Area type="monotone" dataKey="p50" stroke="var(--accent)" fill="var(--accent-soft)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
            <p className="muted">Model {data.model_version}</p>
          </div>
        </>
      )}
    </section>
  );
}

function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {note && <div className="muted">{note}</div>}
    </div>
  );
}
