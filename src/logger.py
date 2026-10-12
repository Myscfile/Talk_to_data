import logging
import uuid
from datetime import datetime

from config import Config


def setup_logging() -> None:
    """Log to the terminal and to a new file per run in logs/."""
    Config.LOG_DIR.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    log_file = Config.LOG_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{run_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
