from __future__ import annotations

import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

HOST = os.getenv("SD_SERVER_HOST", "127.0.0.1")
PORT = int(os.getenv("SD_SERVER_PORT", "7860"))
MODEL_PATH = Path(os.getenv(
    "SD_MODEL_PATH",
    r"C:\Users\AI\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
))

if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна. Цей локальний генератор налаштований на GPU.")

print(f"Завантажую Stable Diffusion з: {MODEL_PATH}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

pipe = StableDiffusionPipeline.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.float16,
    local_files_only=True,
    safety_checker=None,
)
pipe.enable_attention_slicing()
if hasattr(pipe.vae, "enable_slicing"):
    pipe.vae.enable_slicing()
if hasattr(pipe.vae, "enable_tiling"):
    pipe.vae.enable_tiling()
pipe = pipe.to("cuda")

print("Stable Diffusion готовий. Генерація буде виконуватися РІВНО у запитаній роздільності.")


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/sdapi/v1/options":
            self._send_json(200, {"sd_model_checkpoint": "stable-diffusion-v1-5-local"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/sdapi/v1/txt2img":
            self._send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = str(request.get("prompt", "")).strip()
            negative_prompt = str(request.get("negative_prompt", "")).strip()
            width = int(request.get("width", 1920))
            height = int(request.get("height", 1080))
            steps = int(request.get("steps", 16))
            cfg = float(request.get("cfg_scale", 7.0))

            if width != 1920 or height != 1080:
                raise ValueError(
                    f"Потрібна генерація 1920x1080. Отримано {width}x{height}."
                )
            if width % 8 or height % 8:
                raise ValueError("Ширина та висота повинні ділитися на 8.")

            print(f"Генерую: {width}x{height}, steps={steps}, cfg={cfg}")
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    num_images_per_prompt=1,
                )

            image = result.images[0]
            if image.size != (1920, 1080):
                raise RuntimeError(f"Генератор повернув неправильний розмір: {image.size}")

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            self._send_json(200, {"images": [encoded], "info": "1920x1080 native generation"})
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            # Важливо: жодного fallback на меншу роздільність.
            # CUDA OOM та інші помилки повертаються клієнту як HTTP 500.
            print(f"ПОМИЛКА ГЕНЕРАЦІЇ: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._send_json(500, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        print("SD API:", format % args)


server = ThreadingHTTPServer((HOST, PORT), Handler)
print(f"Local SD API: http://{HOST}:{PORT}")
server.serve_forever()
