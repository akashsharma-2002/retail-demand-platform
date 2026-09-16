import { keycloak } from "./auth";

const BASE = import.meta.env.VITE_API_URL ?? "";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  await keycloak.updateToken(30);
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${keycloak.token}`,
      ...(init.headers ?? {}),
    },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`;
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const post = <T,>(path: string, body: unknown) => api<T>(path, { method: "POST", body: JSON.stringify(body) });

export type Forecast = {
  item_id: string;
  store_id: string;
  model_version: string;
  daily: { day: string; p50: number }[];
  total_p50: number;
  cover_days: number;
  cover_p50: number;
  cover_p90: number;
  position: { on_hand: number; on_order: number; position: number; lead_time_days: number; case_pack: number; moq: number; unit_cost: number };
};

export type PlanLine = { item_id: string; units: number; cost: number; need?: number; position?: number };

export type Plan = {
  plan_id: string;
  store_id: string;
  status: "pending_approval" | "approved" | "rejected" | "superseded";
  budget: number;
  total_units: number;
  total_cost: number;
  lines_ordered: number;
  solver_status: string;
  created_by: string;
  created_at: string;
  decided_by: string | null;
  scenario: { demand_change_pct?: number; lead_time_change_days?: number };
  top_lines?: PlanLine[];
  superseded_plans?: number;
  lines?: PlanLine[];
};

export type ChatResponse = {
  thread_id: string;
  answer: string;
  intent: string;
  mode: string;
  tools: { tool: string; ok: boolean }[];
  sources: string[];
  guard: { passed: boolean; unsupported: string[] } | null;
  awaiting_approval: { plan_id: string; total_cost: number; created_by: string } | null;
  approval: { status?: string; error?: string } | null;
  telemetry: { latency_ms: number; llm_calls: number; cost_usd: number };
};
