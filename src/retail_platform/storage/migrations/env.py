from alembic import context
from sqlalchemy import create_engine

from retail_platform.config import get_settings
from retail_platform.storage.models import Base

target_metadata = Base.metadata
MANAGED_ELSEWHERE = {"policy_chunks"}  # created by rag/pg_store.py (vector + BM25 index DDL)


def include_object(obj, name, type_, reflected, compare_to):
    """Only manage this application's tables; extensions (PostGIS, ParadeDB) own theirs."""
    if type_ == "table":
        return name in target_metadata.tables and name not in MANAGED_ELSEWHERE
    return True


def run() -> None:
    engine = create_engine(get_settings().database_url)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            include_object=include_object,
            include_schemas=False,
        )
        with context.begin_transaction():
            context.run_migrations()


run()
