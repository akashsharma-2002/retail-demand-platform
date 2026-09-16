from alembic import command
from alembic.config import Config

from retail_platform.config import ROOT


def upgrade_head() -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "src/retail_platform/storage/migrations"))
    command.upgrade(cfg, "head")


if __name__ == "__main__":
    upgrade_head()
