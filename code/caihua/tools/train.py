"""Canonical train entry for the shared model-comparison protocol."""
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import experiment_bridge
from experiment_runtime.engine import train_main

if __name__ == "__main__":
    train_main(PROJECT)
