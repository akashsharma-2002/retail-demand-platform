export const money = (x: number) => x.toLocaleString("en-US", { style: "currency", currency: "USD" });
export const units = (x: number) => `${Math.round(x).toLocaleString("en-US")} units`;
export const statusLabel: Record<string, string> = {
  pending_approval: "Waiting for approval",
  approved: "Approved",
  rejected: "Rejected",
  superseded: "Superseded",
};
