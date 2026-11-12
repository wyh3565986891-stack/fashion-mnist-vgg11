# 核对 README 里的数字和 runs/summary.json 是否一致

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
README = HERE / "README.md"
SUMMARY = HERE / "runs" / "summary.json"


def main() -> int:
    summary = {record["name"]: record for record in json.loads(SUMMARY.read_text(encoding="utf-8"))
               if record.get("tag") != "bs16"}
    text = README.read_text(encoding="utf-8")
    problems = []
    checked = 0
    pattern = re.compile(
        r"^\| `(?P<name>[a-z0-9_]+)` \| [^|]+ \| (?P<params>[\d,]+) \| (?P<macs>[\d,]+) \| "
        r"(?P<acc>0\.\d{4}) \| (?P<f1>0\.\d{4}) \| (?P<delta>[+−-]\d\.\d{2}) pp \|$",
        re.MULTILINE)
    for match in pattern.finditer(text):
        name = match.group("name")
        if name not in summary:
            problems.append(f"README 提到了未知配置 {name}")
            continue
        record = summary[name]
        checks = {
            "参数量": (int(match.group("params").replace(",", "")), record["parameters"]),
            "MACs": (int(match.group("macs").replace(",", "")), record["macs"]),
            "准确率": (float(match.group("acc")), round(record["test"]["accuracy"], 4)),
            "macro-F1": (float(match.group("f1")), round(record["test"]["macro_f1"], 4)),
            "相对基线": (float(match.group("delta").replace("−", "-")),
                     round(record["delta_accuracy"] * 100, 2)),
        }
        for label, (quoted, actual) in checks.items():
            checked += 1
            if abs(quoted - actual) > 1e-6:
                problems.append(f"{name} 的{label}：README {quoted} ≠ JSON {actual}")
    print(f"核对了 {checked} 个数字（{len(pattern.findall(text))} 行表格）")
    print("覆盖的配置：" + "、".join(sorted({m.group("name") for m in pattern.finditer(text)})))
    if problems:
        for problem in problems:
            print("  ✗", problem)
        return 1
    print("README 与 runs/summary.json 完全一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
