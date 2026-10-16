"""Train UDD with the paper experiment runtime."""
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiment_runtime.train import main

if __name__ == "__main__":
    main()
