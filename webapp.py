"""Minimal web demo: upload a garment photo, get top-3 categories and a manual-review flag.

    python webapp.py --port 8000      # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from data import MEAN, STD
from models import CLASSES, CONFIGS, build_model

STATE: dict = {}

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>服饰分类演示</title>
<style>
 body{font-family:"Microsoft YaHei",sans-serif;background:#f7f9fc;color:#1a202c;margin:0;padding:32px;}
 .wrap{max-width:860px;margin:0 auto;}
 h1{font-size:22px;margin:0 0 6px;} .sub{color:#64748b;font-size:13px;margin-bottom:22px;}
 .grid{display:grid;grid-template-columns:300px 1fr;gap:22px;}
 .card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:18px;}
 #drop{border:2px dashed #cbd5e1;border-radius:10px;height:220px;display:flex;align-items:center;
       justify-content:center;color:#94a3b8;font-size:13px;cursor:pointer;text-align:center;}
 #drop.on{border-color:#3b82f6;background:#eff6ff;}
 img#preview{max-width:100%;max-height:260px;image-rendering:pixelated;border-radius:8px;}
 button{background:#2563eb;color:#fff;border:0;border-radius:8px;padding:8px 14px;font-size:13px;cursor:pointer;}
 button.ghost{background:#eef2ff;color:#3730a3;}
 .bar{height:22px;background:#e2e8f0;border-radius:6px;overflow:hidden;margin:4px 0 10px;}
 .bar>span{display:block;height:100%;background:#3b82f6;}
 .row{display:flex;justify-content:space-between;font-size:13px;margin-top:6px;}
 .tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;}
 .ok{background:#dcfce7;color:#166534;} .warn{background:#fee2e2;color:#991b1b;}
 .muted{color:#64748b;font-size:12px;}
</style></head><body><div class="wrap">
<h1>服饰图像分类 · 小模型在线推理</h1>
<div class="sub">上传一张衣物/鞋包照片（或点“随机测试图”），模型给出 Top-3 类目；置信度低于 0.60 时转人工复核。</div>
<div class="grid">
 <div class="card">
  <div id="drop">拖拽图片到这里<br>或点击选择文件<input id="file" type="file" accept="image/*" hidden></div>
  <div style="margin-top:14px;text-align:center"><img id="preview" hidden></div>
  <div style="margin-top:14px;display:flex;gap:8px;justify-content:center">
    <button class="ghost" onclick="randomSample()">随机测试图</button>
  </div>
 </div>
 <div class="card">
  <div id="verdict" class="muted">等待输入…</div>
  <div id="bars" style="margin-top:12px"></div>
  <div id="truth" class="muted" style="margin-top:10px"></div>
 </div>
</div>
<script>
const drop=document.getElementById('drop'), file=document.getElementById('file'),
      preview=document.getElementById('preview'), bars=document.getElementById('bars'),
      verdict=document.getElementById('verdict'), truth=document.getElementById('truth');
drop.onclick=()=>file.click();
file.onchange=e=>{ if(e.target.files[0]) readFile(e.target.files[0]); };
['dragenter','dragover'].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.add('on')}));
['dragleave','drop'].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.remove('on')}));
drop.addEventListener('drop',e=>{ const f=e.dataTransfer.files[0]; if(f) readFile(f); });
function readFile(f){ const r=new FileReader(); r.onload=()=>send(r.result, null); r.readAsDataURL(f); }
function send(dataUrl, truthLabel){
  preview.src=dataUrl; preview.hidden=false;
  fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({image:dataUrl})})
   .then(r=>r.json()).then(show);
  truth.textContent = truthLabel ? ('真实标签：'+truthLabel+'（来自测试集）') : '';
}
function show(d){
  verdict.innerHTML = `预测：<b>${d.top[0].class}</b> &nbsp;<span class="tag ${d.review?'warn':'ok'}">`
      + (d.review?'置信度低，转人工':'自动通过') + `</span> <span class="muted">耗时 ${d.ms} ms</span>`;
  bars.innerHTML = d.top.map(t=>`<div class="row"><span>${t.class}</span><span>${(t.prob*100).toFixed(1)}%</span></div>
      <div class="bar"><span style="width:${(t.prob*100).toFixed(1)}%"></span></div>`).join('');
}
function randomSample(){
  fetch('/sample').then(r=>r.json()).then(d=>send(d.image, d.label));
}
// handy for screenshots and demos: http://127.0.0.1:8000/?demo=1 loads one test image
if (new URLSearchParams(location.search).get('demo')) randomSample();
</script></div></body></html>"""


def preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("L")
    width, height = image.size
    scale = 28 / min(width, height)
    image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR)
    left = (image.width - 28) // 2
    top = (image.height - 28) // 2
    image = image.crop((left, top, left + 28, top + 28))
    tensor = TF.to_tensor(image)
    # Fashion-MNIST draws light garments on black; photos are usually dark on white.
    if tensor.mean() > 0.5:
        tensor = 1 - tensor
    return TF.normalize(tensor, (MEAN,), (STD,)).unsqueeze(0)


@torch.inference_mode()
def predict(tensor: torch.Tensor) -> list[dict]:
    logits = STATE["model"](tensor)
    probabilities = logits.softmax(1)[0]
    values, indices = probabilities.topk(3)
    return [{"class": CLASSES[int(i)], "prob": float(v)} for v, i in zip(values, indices)]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/sample":
            index = random.randrange(len(STATE["test"]))
            image, label = STATE["test"][index]
            tensor = image * STD + MEAN
            buffer = io.BytesIO()
            Image.fromarray((tensor[0].numpy() * 255).astype("uint8")).resize(
                (140, 140), Image.NEAREST).save(buffer, format="PNG")
            data = base64.b64encode(buffer.getvalue()).decode()
            self._json({"image": f"data:image/png;base64,{data}",
                        "label": CLASSES[int(label)]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/predict":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        data = payload["image"].split(",")[-1]
        image = Image.open(io.BytesIO(base64.b64decode(data)))
        import time
        started = time.perf_counter()
        top = predict(preprocess(image))
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        self._json({"top": top, "review": top[0]["prob"] < 0.60, "ms": elapsed})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--name", default="final")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()

    from data import build_datasets
    model = build_model(**CONFIGS[args.name])
    model.load_state_dict(torch.load(args.runs / args.name / "model.pt", weights_only=True))
    model.eval()
    _, _, test = build_datasets(args.data)
    STATE["model"] = model
    STATE["test"] = test
    print(f"model={args.name}  http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
