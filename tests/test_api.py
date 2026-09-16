from conftest import ISSUER, auth


def test_health_and_auth_required(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/stores").status_code == 401


def test_rejects_wrong_audience_and_expired_token(client, make_token):
    bad_aud = make_token("ana", ["viewer"], aud="another-api")
    assert client.get("/v1/stores", headers=auth(bad_aud)).status_code == 401
    expired = make_token("ana", ["viewer"], exp=1)
    assert client.get("/v1/stores", headers=auth(expired)).status_code == 401
    wrong_issuer = make_token("ana", ["viewer"], iss=ISSUER + "x")
    assert client.get("/v1/stores", headers=auth(wrong_issuer)).status_code == 401


def test_viewer_reads_forecast_but_cannot_plan(client, make_token):
    token = make_token("vic", ["viewer"])
    stores = client.get("/v1/stores", headers=auth(token))
    assert stores.json() == ["CA_1", "CA_2"]
    fc = client.get("/v1/forecasts/CA_1/FOODS_3_003?days=7", headers=auth(token)).json()
    assert len(fc["daily"]) == 7 and fc["cover_p90"] >= fc["cover_p50"]
    assert client.post("/v1/plans", json={"store_id": "CA_1"}, headers=auth(token)).status_code == 403


def test_plan_lifecycle_with_separation_of_duties(client, make_token):
    planner = make_token("paula", ["planner"])
    created = client.post("/v1/plans", json={"store_id": "CA_1", "budget": 400}, headers=auth(planner))
    assert created.status_code == 201, created.text
    plan = created.json()
    assert plan["status"] == "pending_approval" and plan["total_cost"] <= 400

    assert client.post(f"/v1/plans/{plan['plan_id']}/approve", json={}, headers=auth(planner)).status_code == 403
    self_approver = make_token("paula", ["planner", "approver"])
    resp = client.post(f"/v1/plans/{plan['plan_id']}/approve", json={}, headers=auth(self_approver))
    assert resp.status_code == 403 and "created" in resp.json()["detail"]

    approver = make_token("arun", ["approver"])
    ok = client.post(f"/v1/plans/{plan['plan_id']}/approve", json={"note": "ok"}, headers=auth(approver))
    assert ok.status_code == 200 and ok.json()["status"] == "approved"
    again = client.post(f"/v1/plans/{plan['plan_id']}/reject", json={}, headers=auth(approver))
    assert again.status_code == 409


def test_input_validation(client, make_token):
    planner = make_token("paula", ["planner"])
    assert client.post("/v1/plans", json={"store_id": "x; drop table"}, headers=auth(planner)).status_code == 422
    assert client.post("/v1/plans", json={"store_id": "CA_1", "budget": -5}, headers=auth(planner)).status_code == 422
    assert client.get("/v1/forecasts/CA_1/NOPE_1_001", headers=auth(planner)).status_code == 404


def test_audit_log_written(client, make_token, session_factory):
    from retail_platform.storage.models import AuditLog

    planner = make_token("paula", ["planner"])
    plan = client.post("/v1/plans", json={"store_id": "CA_2"}, headers=auth(planner)).json()
    client.post(
        f"/v1/plans/{plan['plan_id']}/reject",
        json={"note": "too early"},
        headers=auth(make_token("arun", ["approver"])),
    )
    with session_factory() as s:
        actions = [(a.actor, a.action) for a in s.query(AuditLog).order_by(AuditLog.id)]
    assert ("paula", "plan.create") in actions and ("arun", "plan.reject") in actions


def test_newer_plan_and_approval_supersede_stale_waiting_plans(client, make_token):
    planner, approver = make_token("paula", ["planner"]), make_token("arun", ["approver"])
    first = client.post("/v1/plans", json={"store_id": "CA_2"}, headers=auth(planner)).json()
    second = client.post("/v1/plans", json={"store_id": "CA_2"}, headers=auth(planner)).json()
    assert second["superseded_plans"] == 1
    assert client.get(f"/v1/plans/{first['plan_id']}", headers=auth(planner)).json()["status"] == "superseded"
    assert client.post(f"/v1/plans/{first['plan_id']}/approve", json={}, headers=auth(approver)).status_code == 409

    other_store = client.post("/v1/plans", json={"store_id": "CA_1"}, headers=auth(planner)).json()
    approved = client.post(f"/v1/plans/{second['plan_id']}/approve", json={}, headers=auth(approver)).json()
    assert approved["status"] == "approved" and approved["superseded_plans"] == 0
    assert (
        client.get(f"/v1/plans/{other_store['plan_id']}", headers=auth(planner)).json()["status"] == "pending_approval"
    )


def test_rate_limit_applies_when_enabled(session_factory, signing_key, make_token):
    from fastapi.testclient import TestClient

    from retail_platform.api.app import create_app
    from retail_platform.api.auth import TokenVerifier, get_verifier
    from retail_platform.storage.db import get_session

    app = create_app()
    app.state.limiter.enabled = True

    def _session():
        with session_factory() as s:
            yield s

    app.dependency_overrides[get_session] = _session
    app.dependency_overrides[get_verifier] = lambda: TokenVerifier(
        ISSUER, "retail-api", lambda _t: signing_key.public_key()
    )
    c = TestClient(app)
    token = make_token("paula", ["planner"])
    codes = [
        c.post("/v1/plans", json={"store_id": "CA_1", "budget": 50}, headers=auth(token)).status_code for _ in range(22)
    ]
    assert codes[:20] == [201] * 20 and codes[-1] == 429
