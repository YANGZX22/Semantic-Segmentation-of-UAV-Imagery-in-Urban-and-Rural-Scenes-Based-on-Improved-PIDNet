#!/usr/bin/env python3
"""Evaluate a trained checkpoint without importing the training runner."""
from inference_runtime.cli import main

if __name__ == "__main__":
    main("evaluation")
