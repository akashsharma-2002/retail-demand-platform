import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { initAuth } from "./auth";
import "./styles.css";

initAuth()
  .then(() =>
    createRoot(document.getElementById("root")!).render(
      <StrictMode>
        <App />
      </StrictMode>,
    ),
  )
  .catch(() => {
    document.getElementById("root")!.textContent = "Sign-in service is unavailable. Check that Keycloak is running.";
  });
