"""Export to ONNX, quantize to int8 and measure single-thread CPU latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from models import CONFIGS, build_model

MODELS = ("vgg11", "final")


def export(model: torch.nn.Module, path: Path) -> None:
    model.eval()
    dummy = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(
        model, dummy, str(path), input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
        dynamo=False,
    )


def latency_torch(model: torch.nn.Module, repeats: int = 200) -> float:
    model.eval()
    torch.set_num_threads(1)
    sample = torch.randn(1, 1, 28, 28)
    with torch.inference_mode():
        for _ in range(20):
            model(sample)
        started = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        return (time.perf_counter() - started) / repeats * 1000


def latency_session(session, repeats: int = 200) -> float:
    import numpy as np
    sample = np.random.randn(1, 1, 28, 28).astype("float32")
    name = session.get_inputs()[0].name
    for _ in range(20):
        session.run(None, {name: sample})
    started = time.perf_counter()
    for _ in range(repeats):
        session.run(None, {name: sample})
    return (time.perf_counter() - started) / repeats * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    out = args.runs / "export"
    out.mkdir(parents=True, exist_ok=True)

    try:
        import onnxruntime as ort
        have_ort = True
    except ImportError:
        have_ort = False

    records = []
    for name in MODELS:
        model = build_model(**CONFIGS[name])
        model.load_state_dict(torch.load(args.runs / name / "model.pt", weights_only=True))
        record = {"name": name,
                  "parameters": sum(p.numel() for p in model.parameters())}

        fp32_path = out / f"{name}.onnx"
        export(model, fp32_path)
        record["torch_ms"] = round(latency_torch(model), 3)
        record["onnx_kb"] = round(fp32_path.stat().st_size / 1024, 1)

        int8_path = out / f"{name}_int8.onnx"
        if have_ort:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
            record["int8_kb"] = round(int8_path.stat().st_size / 1024, 1)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            record["onnx_ms"] = round(latency_session(ort.InferenceSession(
                str(fp32_path), options, providers=["CPUExecutionProvider"])), 3)
            record["int8_ms"] = round(latency_session(ort.InferenceSession(
                str(int8_path), options, providers=["CPUExecutionProvider"])), 3)
        records.append(record)
        print(record, flush=True)

    (args.runs / "export.json").write_text(
        json.dumps({"onnxruntime": have_ort, "models": records}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
