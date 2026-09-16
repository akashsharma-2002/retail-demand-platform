import Keycloak from "keycloak-js";

export const keycloak = new Keycloak({
  url: import.meta.env.VITE_OIDC_URL ?? "http://localhost:8080",
  realm: "retail",
  clientId: "retail-web",
});

export async function initAuth(): Promise<void> {
  await keycloak.init({ onLoad: "login-required", pkceMethod: "S256", checkLoginIframe: false });
}

export function roles(): string[] {
  return keycloak.tokenParsed?.realm_access?.roles ?? [];
}

export function hasRole(role: string): boolean {
  const r = roles();
  return r.includes(role) || r.includes("admin");
}
