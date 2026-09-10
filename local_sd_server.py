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

FINAL_WIDTH = 1920
FINAL_HEIGHT = 1080
# 4 columns x 2 rows = 8 independent generations.
# Each tile is generated directly at its final pixel size and then assembled.
TILE_COLS = 4
TILE_ROWS = 2
TILE_WIDTH = FINAL_WIDTH // TILE_COLS   # 480
TILE_HEIGHT = FINAL_HEIGHT // TILE_ROWS  # 540

if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна. Цей локальний генератор налаштований на GPU.")

print(f"Завантажую Stable Diffusion з: {MODEL_PATH}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Фінальний розмір: {FINAL_WIDTH}x{FINAL_HEIGHT}")
print(f"Режим: {TILE_COLS}x{TILE_ROWS} = {TILE_COLS * TILE_ROWS} тайлів по {TILE_WIDTH}x{TILE_HEIGHT}")

pipe = StableDiffusionPipeline.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.float16,
    local_files_only=True,
    safety_checker=None,
)

# Максимально економимо VRAM. Розмір фінального кадру при цьому НЕ змінюється.
pipe.enable_attention_slicing()
if hasattr(pipe.vae, "enable_slicing"):
    pipe.vae.enable_slicing()
if hasattr(pipe.vae, "enable_tiling"):
    pipe.vae.enable_tiling()

# CPU offload дозволяє тримати модель частково в RAM, а не всю в 2 GB VRAM.
# Якщо accelerate недоступний, працюємо звичайним способом.
OFFLOAD_MODE = "none"
try:
    import accelerate  # noqa: F401
    pipe.enable_model_cpu_offload()
    OFFLOAD_MODE = "model_cpu_offload"
except Exception as exc:
    print(f"CPU offload недоступний: {exc}")
    pipe = pipe.to("cuda")

print(f"Stable Diffusion готовий. Memory mode: {OFFLOAD_MODE}")
print("Кожна сцена буде складатися з 8 окремо згенерованих тайлів у фінальний 1920x1080.")


def generate_tile(prompt: str, negative_prompt: str, steps: int, cfg: float, tile_index: int) -> Image.Image:
    """Generate exactly one 480x540 tile, then release temporary GPU memory."""
    print(f"  🧩 Тайл {tile_index}/8: {TILE_WIDTH}x{TILE_HEIGHT}")
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=TILE_WIDTH,
            height=TILE_HEIGHT,
            num_inference_steps=steps,
            guidance_scale=cfg,
            num_images_per_prompt=1,
        )
    image = result.images[0].convert("RGB")
    if image.size != (TILE_WIDTH, TILE_HEIGHT):
        raise RuntimeError(f"Тайл {tile_index} має неправильний розмір: {image.size}")
    del result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return image


def build_full_image(prompt: str, negative_prompt: str, steps: int, cfg: float) -> Image.Image:
    """Generate 8 tiles and stitch them into one exact 1920x1080 image."""
    canvas = Image.new("RGB", (FINAL_WIDTH, FINAL_HEIGHT))
    tiles = []
    try:
        for index in range(8):
            col = index % TILE_COLS
            row = index // TILE_COLS
            tile_prompt = (
                f"{prompt}, full scene detail, tile {index + 1} of 8, "
                f"composition area {col + 1} of {TILE_COLS} horizontally and "
                f"{row + 1} of {TILE_ROWS} vertically"
            )
            tile = generate_tile(tile_prompt, negative_prompt, steps, cfg, index + 1)
            tiles.append((col, row, tile))
            canvas.paste(tile, (col * TILE_WIDTH, row * TILE_HEIGHT))
    finally:
        for _, _, tile in tiles:
            tile.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if canvas.size != (FINAL_WIDTH, FINAL_HEIGHT):
        raise RuntimeError(f"Фінальний кадр має неправильний розмір: {canvas.size}")
    return canvas


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
            self._send_json(200, {
                "sd_model_checkpoint": "stable-diffusion-v1-5-local",
                "generation_mode": "8_tiles_1920x1080",
                "tile_size": f"{TILE_WIDTH}x{TILE_HEIGHT}",
            })
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
            width = int(request.get("width", FINAL_WIDTH))
            height = int(request.get("height", FINAL_HEIGHT))
            steps = int(request.get("steps", 16))
            cfg = float(request.get("cfg_scale", 7.0))

            # The API remains strictly 1920x1080 from the client's point of view.
            # Internally it is split into 8 smaller generations to fit low VRAM.
            if width != FINAL_WIDTH or height != FINAL_HEIGHT:
                raise ValueError(
                    f"Потрібен фінальний розмір 1920x1080. Отримано {width}x{height}."
                )

            print(
                f"Генерую фінальну сцену {FINAL_WIDTH}x{FINAL_HEIGHT} "
                f"через 8 тайлів {TILE_WIDTH}x{TILE_HEIGHT}, steps={steps}, cfg={cfg}"
            )
            image = build_full_image(prompt, negative_prompt, steps, cfg)

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image.close()
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            self._send_json(200, {
                "images": [encoded],
                "info": "1920x1080 assembled from 8 native tiles",
            })

        except torch.cuda.OutOfMemoryError as exc:
            print(f"ПОМИЛКА CUDA OOM: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._send_json(500, {
                "error": "Навіть один тайл не помістився у VRAM.",
                "details": str(exc),
                "final_resolution": "1920x1080",
                "tile_resolution": f"{TILE_WIDTH}x{TILE_HEIGHT}",
            })
        except Exception as exc:
            # Важливо: жодного fallback на меншу фінальну роздільність.
            print(f"ПОМИЛКА ГЕНЕРАЦІЇ: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._send_json(500, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        print("SD API:", format % args)


server = ThreadingHTTPServer((HOST, PORT), Handler)
print(f"Local SD API: http://{HOST}:{PORT}")
server.serve_forever()
