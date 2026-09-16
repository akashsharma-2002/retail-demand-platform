"""Nightly pipeline: ingest -> validate -> backtest -> select -> gate -> simulate -> publish -> index policies."""

import json
import os
import time

import mlflow
from prefect import flow, get_run_logger, task

from retail_platform.config import get_settings
from retail_platform.forecast.context import ForecastContext
from retail_platform.forecast.lgbm import DEFAULT_PARAMS, NUM_ROUNDS
from retail_platform.pipeline import steps

CHAMPION_FILE = "champion.json"


@task
def ingest_task(force: bool = False) -> str:
    settings = get_settings()
    steps.ingest(settings, force)
    return steps.data_fingerprint(settings)


@task
def backtest_task(ctx: ForecastContext, include_tft: bool) -> steps.BacktestResult:
    return steps.backtest(ctx, include_tft)


@task
def simulation_task(ctx: ForecastContext, result: steps.BacktestResult, model: str) -> dict:
    return steps.run_simulation(ctx, result, model)


@flow(name="retail-nightly")
def nightly(include_tft: bool = True, publish_to_db: bool = True) -> dict:
    log = get_run_logger()
    settings = get_settings()
    started = time.time()
    fingerprint = ingest_task()
    ctx = ForecastContext(settings)

    result = backtest_task(ctx, include_tft)
    table = result.table()
    model = steps.select_candidate(result)
    coverage = steps.conformal_report(ctx, result, model)
    champion_path = settings.artifacts_dir / CHAMPION_FILE
    champion = json.loads(champion_path.read_text()) if champion_path.exists() else None
    promoted, reason = steps.gate(
        table[model]["wrmsse_mean"], table["seasonal_naive"]["wrmsse_mean"], coverage, champion
    )
    log.info(f"selected {model}; gate: {promoted} ({reason})")

    simulation = simulation_task(ctx, result, model)
    reports = settings.artifacts_dir / "reports"
    comparison = simulation["comparison_at_service_level"]
    summary: dict = {
        "data_fingerprint": fingerprint,
        "selected_model": model,
        "gate": {"promoted": promoted, "reason": reason},
        "backtest": table,
        "conformal_coverage": coverage,
        "simulation": {k: v for k, v in simulation.items() if not k.startswith("_")},
        "improvement_vs_seasonal_naive_pct": round(
            (1 - table[model]["wrmsse_mean"] / table["seasonal_naive"]["wrmsse_mean"]) * 100, 2
        ),
    }
    steps.write_json(reports / "backtest.json", {"folds": result.folds, "table": table})
    steps.write_json(reports / "simulation.json", simulation)
    from retail_platform.pipeline.drift import drift_report
    from retail_platform.pipeline.drift import write as write_drift

    drift = drift_report(
        ctx.features, (ctx.train_start_d, settings.last_day - 28), (settings.last_day - 27, settings.last_day)
    )
    write_drift(drift, reports / "drift.json")
    summary["drift_any_significant"] = drift["any_significant"]
    steps.write_json(reports / "summary.json", summary)

    mlflow.set_tracking_uri(settings.mlflow_uri)
    if mlflow.get_experiment_by_name("retail-demand") is None:
        mlflow.create_experiment("retail-demand", artifact_location=settings.mlflow_artifacts.as_uri())
    mlflow.set_experiment("retail-demand")
    with mlflow.start_run(run_name=f"nightly-{model}") as run:
        mlflow.log_params(
            {
                "selected_model": model,
                "cutoffs": settings.backtest_cutoffs,
                "horizon": settings.horizon,
                "num_rounds": NUM_ROUNDS,
                **{f"lgbm_{k}": v for k, v in DEFAULT_PARAMS.items()},
                "data_fingerprint": fingerprint,
            }
        )
        for name, row in table.items():
            mlflow.log_metrics({f"{name}.{k}": v for k, v in row.items()})
        mlflow.log_metrics({f"simulation.{k}": v for k, v in comparison.items()})
        mlflow.set_tags({"champion": str(promoted).lower(), "gate_reason": reason})
        mlflow.log_artifacts(str(reports), artifact_path="reports")
        run_id = run.info.run_id

    if promoted:
        champion_path.write_text(
            json.dumps({"run_id": run_id, "model": model, "wrmsse_mean": table[model]["wrmsse_mean"]}, indent=2)
        )
        if publish_to_db:
            from retail_platform.pipeline.publish import publish
            from retail_platform.storage.db import get_engine

            forecast, conformal = steps.production_forecast(ctx, model)
            ending = simulation["_ending"]
            dates = steps.calendar_dates(ctx, settings.last_day + 1, settings.horizon)
            buckets = conformal.buckets(ctx.sales)
            publish(
                get_engine(),
                simulation["_supply"],
                buckets,
                ending.ending_on_hand,
                ending.ending_on_order,
                forecast,
                dates,
                conformal,
                f"{model}-{run_id[:8]}",
                summary,
            )
            if os.getenv("RP_INDEX_POLICIES", "1") == "1":
                index_policies()
    log.info(f"pipeline finished in {time.time() - started:.0f}s")
    return summary


def index_policies() -> None:
    from retail_platform.rag.chunking import load_corpus
    from retail_platform.rag.factory import POLICY_DIR
    from retail_platform.rag.pg_store import index_chunks
    from retail_platform.rag.search import SentenceTransformerEmbedder
    from retail_platform.storage.db import get_engine

    settings = get_settings()
    index_chunks(get_engine(), load_corpus(POLICY_DIR), SentenceTransformerEmbedder(settings.embedding_model))


if __name__ == "__main__":
    print(json.dumps(nightly(include_tft=os.getenv("RP_INCLUDE_TFT", "1") == "1"), indent=2, default=str))
