# 把 runs/*/metrics.json 汇总成 summary.json 和 summary.csv

import argparse
import csv
import json
from pathlib import Path

ORDER = ["vgg11", "sep", "sep_all", "se", "cbam", "gelu", "mish", "gn", "final"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("runs/summary.json"))
    args = parser.parse_args()

    records = []
    for folder in sorted(args.runs.iterdir()):
        metrics_path = folder / "metrics.json"
        if not folder.is_dir() or not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        weights = folder / "model.pt"
        data["folder"] = folder.name
        data["weights_kb"] = round(weights.stat().st_size / 1024, 1) if weights.exists() else 0
        data["seconds_per_epoch"] = round(
            sum(row["seconds"] for row in data["history"]) / len(data["history"]), 2)
        data["final_val_accuracy"] = data["history"][-1]["val_accuracy"]
        data["train_accuracy_last"] = data["history"][-1]["train_accuracy"]
        records.append(data)

    def key(record):
        index = ORDER.index(record["name"]) if record["name"] in ORDER else len(ORDER)
        return (index, record.get("tag", ""))

    records.sort(key=key)
    baseline = next((r for r in records if r["name"] == "vgg11"), None)
    for record in records:
        if baseline and baseline is not record:
            record["delta_accuracy"] = record["test"]["accuracy"] - baseline["test"]["accuracy"]
            record["delta_macs"] = record["macs"] / max(baseline["macs"], 1) - 1
            record["delta_params"] = record["parameters"] / max(baseline["parameters"], 1) - 1
        else:
            record["delta_accuracy"] = 0.0
            record["delta_macs"] = 0.0
            record["delta_params"] = 0.0

    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.out.with_suffix(".csv")
    fields = ["name", "tag", "description", "parameters", "macs", "weights_kb",
              "best_epoch", "seconds_per_epoch", "test_accuracy", "macro_f1",
              "delta_accuracy", "delta_params", "delta_macs"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["test_accuracy"] = record["test"]["accuracy"]
            row["macro_f1"] = record["test"]["macro_f1"]
            writer.writerow(row)

    header = f"{'name':10s} {'params':>10s} {'MACs':>12s} {'acc':>7s} {'F1':>7s} {'d_acc':>8s} {'epoch':>6s} {'s/ep':>6s}"
    print(header)
    for record in records:
        print(f"{record['name']:10s} {record['parameters']:10,d} {record['macs']:12,d} "
              f"{record['test']['accuracy']:7.4f} {record['test']['macro_f1']:7.4f} "
              f"{record['delta_accuracy']:+8.4f} {record['best_epoch']:6d} "
              f"{record['seconds_per_epoch']:6.1f}")
    print(f"\n{len(records)} runs -> {args.out}")


if __name__ == "__main__":
    main()
