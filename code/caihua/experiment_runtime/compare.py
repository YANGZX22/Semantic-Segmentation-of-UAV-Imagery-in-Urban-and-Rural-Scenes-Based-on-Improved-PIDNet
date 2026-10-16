"""Compare structured validation records; never scrape 'best' values from logs."""
import argparse
import json
from pathlib import Path
from .provenance import write_json


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            name = prefix + "." + key if prefix else key
            result += differences(left.get(key), right.get(key), name)
        return result
    return [] if left == right else [{"field": prefix, "left": left, "right": right}]


def compare_runs(paths):
    if len(paths) < 2:
        raise ValueError("At least two run directories are required")
    runs = []
    for path in paths:
        root = Path(path)
        run = json.loads((root / "run.json").read_text(encoding="utf-8"))
        result = json.loads((root / "best_validation.json").read_text(encoding="utf-8"))
        completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
        contract = run["experiment_protocol"]
        if (completion.get("status") != "complete" or completion.get("experiment_protocol") != contract
                or completion.get("epochs") != run["comparison_controls"]["training"]["END_EPOCH"]):
            raise ValueError("Run has not completed its declared training budget: " + str(root))
        if result["experiment_protocol"] != contract or result["role"] != "val":
            raise ValueError("Result identity/role differs from the run record: " + str(root))
        if (not (root / "best.pt").is_file()
                or result["best_epoch"] != completion["best_epoch"]
                or result["best_mIoU"] != completion["best_mIoU"]):
            raise ValueError("Best weights, selected epoch, and completion record differ: " + str(root))
        if run["comparison_controls"] != contract["comparison_controls"]:
            raise ValueError("Run comparison conditions have been edited")
        runs.append((root, run, result))
    reference = runs[0][1]
    entries = []
    for root, run, result in runs:
        delta = differences(reference["comparison_controls"], run["comparison_controls"])
        if reference["environment"] != run["environment"]:
            delta += differences(reference["environment"], run["environment"], "environment")
        coverage = run["initialization"].get("loaded_parameter_fraction")
        entries.append({"run": str(root), "model": run["experiment_protocol"]["model"],
                        "seed": run["comparison_controls"]["seed"], "best_epoch": result["best_epoch"],
                        "validation_mIoU": result["validation"]["mIoU"],
                        "class_IoU": result["validation"]["class_IoU"],
                        "pretrained_parameter_fraction": coverage, "condition_differences": delta})
    return {"role": "validation_diagnostic", "matched_controls": not any(e["condition_differences"] for e in entries),
            "runs": entries, "note": "Different recipes, seeds, folds, or environments are disclosed rather than ranked as a controlled ablation. A matched recipe does not establish significance or equal compute."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare_runs(args.runs)
    write_json(args.output, report)
    print(json.dumps({"matched_controls": report["matched_controls"], "report": args.output}))


if __name__ == "__main__":
    main()
