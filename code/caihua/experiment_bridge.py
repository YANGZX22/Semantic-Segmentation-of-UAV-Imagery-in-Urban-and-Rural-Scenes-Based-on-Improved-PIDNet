"""Find the shared runtime relative to this checkout, on any host."""
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parent
CURRENT = next((p for p in (PROJECT, *PROJECT.parents) if (p / "experiment_runtime").is_dir()), None)
if CURRENT is None:
    raise RuntimeError("Missing code/caihua/experiment_runtime; copy the complete project")
if str(CURRENT) not in sys.path:
    sys.path.insert(0, str(CURRENT))
from experiment_runtime.configuration import add_defaults, finalize_config
