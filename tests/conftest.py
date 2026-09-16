import time
from datetime import date, timedelta

import jwt
import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from retail_platform.api.auth import TokenVerifier, get_verifier
from retail_platform.storage.db import get_session
from retail_platform.storage.models import Base, ConformalOffset, Forecast, Item

ISSUER, AUDIENCE = "https://idp.test/realms/retail", "retail-api"


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    seed(factory)
    return factory


def seed(factory) -> None:
    rng = np.random.default_rng(0)
    start = date(2016, 5, 23)
    with factory() as s:
        idx = 0
        for store in ["CA_1", "CA_2"]:
            for n in range(1, 21):
                dept = ["FOODS_3", "HOUSEHOLD_1", "HOBBIES_1"][n % 3]
                item = Item(
                    series_idx=idx,
                    item_id=f"{dept}_{n:03d}",
                    store_id=store,
                    dept_id=dept,
                    cat_id=dept.split("_")[0],
                    price=4.0 + n,
                    unit_cost=(4.0 + n) * 0.7,
                    lead_time=[2, 7, 10][n % 3],
                    case_pack=[12, 4, 2][n % 3],
                    moq=[12, 4, 2][n % 3],
                    shelf_capacity=400,
                    holding_cost_day=0.002 * n,
                    shortage_cost=1.5 + n * 0.3,
                    velocity_bucket=n % 3,
                    on_hand=int(rng.integers(0, 20)),
                    on_order=0,
                )
                s.add(item)
                for d in range(28):
                    s.add(
                        Forecast(
                            series_idx=idx,
                            day=start + timedelta(days=d),
                            p50=float(1 + n % 5 + (d % 7 == 5)),
                            model_version="test-v1",
                        )
                    )
                idx += 1
        for b in range(3):
            for w in range(1, 29):
                s.add(ConformalOffset(bucket=b, window=w, level=0.9, offset=0.5 * w))
        s.commit()


@pytest.fixture(scope="session")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def make_token(signing_key):
    def _make(sub: str, roles: list[str], **overrides) -> str:
        now = int(time.time())
        claims = {
            "sub": sub,
            "preferred_username": sub,
            "iss": ISSUER,
            "aud": AUDIENCE,
            "iat": now,
            "exp": now + 300,
            "realm_access": {"roles": roles},
            **overrides,
        }
        return jwt.encode(claims, signing_key, algorithm="RS256")

    return _make


@pytest.fixture
def client(session_factory, signing_key):
    from retail_platform.api.app import create_app

    app = create_app()
    app.state.limiter.enabled = False  # tests exercise many writes per minute

    def _session():
        with session_factory() as s:
            yield s

    app.dependency_overrides[get_session] = _session
    app.dependency_overrides[get_verifier] = lambda: TokenVerifier(
        ISSUER, AUDIENCE, lambda _t: signing_key.public_key()
    )
    return TestClient(app)


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}
