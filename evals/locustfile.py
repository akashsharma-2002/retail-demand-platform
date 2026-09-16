"""Load test: forecast reads, plan previews via the assistant, and policy questions.

Tokens come from the local Keycloak realm (retail-cli client, password grant, local only).
"""

import os
import random

import httpx
from locust import HttpUser, between, task

KEYCLOAK = os.getenv("OIDC_PUBLIC_URL", "http://localhost:8080")
ITEMS = ["FOODS_3_090", "FOODS_3_586", "FOODS_3_252", "HOUSEHOLD_1_118", "HOBBIES_1_234", "FOODS_2_019"]
STORES = ["CA_1", "CA_2", "CA_3", "CA_4"]


def token(username: str, password: str) -> str:
    resp = httpx.post(
        f"{KEYCLOAK}/realms/retail/protocol/openid-connect/token",
        data={"grant_type": "password", "client_id": "retail-cli", "username": username, "password": password},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


class Planner(HttpUser):
    wait_time = between(0.5, 2)

    def on_start(self) -> None:
        self.client.headers["Authorization"] = f"Bearer {token('paula', 'local-planner-pass')}"

    @task(6)
    def forecast(self) -> None:
        self.client.get(
            f"/v1/forecasts/{random.choice(STORES)}/{random.choice(ITEMS)}", name="/v1/forecasts/{store}/{item}"
        )

    @task(3)
    def explain(self) -> None:
        self.client.post(
            "/v1/assistant/chat",
            name="/v1/assistant/chat [explain]",
            json={"message": f"Why should we order {random.choice(ITEMS)} for {random.choice(STORES)}?"},
        )

    @task(2)
    def policy(self) -> None:
        question = random.choice(
            [
                "Who can approve a plan above the store limit?",
                "What service level do hobby items target?",
                "Are emergency orders allowed?",
            ]
        )
        self.client.post("/v1/assistant/chat", name="/v1/assistant/chat [policy]", json={"message": question})

    @task(1)
    def what_if(self) -> None:
        self.client.post(
            "/v1/assistant/chat",
            name="/v1/assistant/chat [what-if]",
            json={"message": f"What if demand rises 10% at {random.choice(STORES)}?"},
        )
