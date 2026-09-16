"""Runtime settings, read from environment variables (prefix RP_)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RP_", env_file=".env", extra="ignore")

    env: str = "local"
    data_dir: Path = ROOT / "data"
    artifacts_dir: Path = ROOT / "artifacts"
    mlflow_uri: str = f"sqlite:///{ROOT / 'mlflow.db'}"
    mlflow_artifacts: Path = ROOT / "mlartifacts"

    state: str = "CA"
    history_start: str = "2013-01-01"
    train_start: str = "2014-01-01"
    horizon: int = 28
    backtest_cutoffs: list[int] = Field(default_factory=lambda: [1857, 1885, 1913])
    last_day: int = 1941
    review_period: int = 7
    service_level: float = 0.9
    simulation_weeks: int = 8

    database_url: str = "postgresql+psycopg://app:app@localhost:5432/retail"
    redis_url: str = "redis://localhost:6379/0"

    oidc_issuer: str = "http://localhost:8080/realms/retail"
    oidc_audience: str = "retail-api"
    oidc_jwks_url: str = "http://localhost:8080/realms/retail/protocol/openid-connect/certs"
    rate_limit_enabled: bool = True  # disable only for local load tests
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    llm_provider: str = "ollama"  # ollama | openai | none
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    llm_model: str = "gemma4:31b-cloud"
    ollama_url: str = "http://localhost:11434"
    ollama_api_key: str | None = Field(default=None, validation_alias="OLLAMA_API_KEY")
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
