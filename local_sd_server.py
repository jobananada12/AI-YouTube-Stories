from __future__ import annotations

import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageEnhance, ImageOps

HOST = os.getenv("SD_SERVER_HOST", "127.0.0.1")
PORT = int(os.getenv("SD_SERVER_PORT", "7860"))
MODEL_PATH = Path(os.getenv(
    "SD_MODEL_PATH",
    r"C:\Users\AI\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
))

FINAL_WIDTH = 1920
FINAL_HEIGHT = 1080

# IMPORTANT:
# We never generate 8 independent SD images anymore. That was the source of
# the 2x4 collage. Stable Diffusion creates ONE coherent composition at a
# memory-safe base resolution, then ordinary image processing enlarges that
# single image to the required 1920x1080 canvas.
#
# Override with SD_BASE_WIDTH / SD_BASE_HEIGHT when more VRAM is available.
# Recommended SD 1.5 sizes:
#   768x432  -> very low VRAM
#   896x504  -> low/medium VRAM
#   1024x576 -> about 6-8GB+ VRAM depending on settings
SD_BASE_WIDTH = int(os.getenv("SD_BASE_WIDTH", "768"))
SD_BASE_HEIGHT = int(os.getenv("SD_BASE_HEIGHT", "432"))

if SD_BASE_WIDTH % 8 or SD_BASE_HEIGHT % 8:
    raise RuntimeError("SD_BASE_WIDTH та SD_BASE_HEIGHT повинні ділитися на 8.")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна. Цей локальний генератор налаштований на GPU.")

print(f"Завантажую Stable Diffusion з: {MODEL_PATH}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Фінальний розмір: {FINAL_WIDTH}x{FINAL_HEIGHT}")
print(f"ЄДИНА генерація: {SD_BASE_WIDTH}x{SD_BASE_HEIGHT} -> upscale -> {FINAL_WIDTH}x{FINAL_HEIGHT}")

pipe = StableDiffusionPipeline.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.float16,
    local_files_only=True,
    safety_checker=None,
)

pipe.enable_attention_slicing("max")
if hasattr(pipe.vae, "enable_slicing"):
    pipe.vae.enable_slicing()
if hasattr(pipe.vae, "enable_tiling"):
    # This is VAE tiling only. It does NOT create eight independent scenes.
    pipe.vae.enable_tiling()

OFFLOAD_MODE = "none"
try:
    import accelerate  # noqa: F401
    pipe.enable_sequential_cpu_offload()
    OFFLOAD_MODE = "sequential_cpu_offload"
except Exception as exc:
    print(f"Sequential CPU offload недоступний: {exc}")
    pipe = pipe.to("cuda")

print(f"Stable Diffusion готовий. Memory mode: {OFFLOAD_MODE}")
print("Режим: 1 coherent scene -> 1 image -> grayscale -> upscale -> 1920x1080")


def generate_one_image(prompt: str, negative_prompt: str, steps: int, cfg: float) -> Image.Image:
    print(f"  🎨 ОДНА SD генерація: {SD_BASE_WIDTH}x{SD_BASE_HEIGHT}")
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=SD_BASE_WIDTH,
            height=SD_BASE_HEIGHT,
            num_inference_steps=steps,
            guidance_scale=cfg,
            num_images_per_prompt=1,
        )

    image = result.images[0].convert("RGB")
    del result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if image.size != (SD_BASE_WIDTH, SD_BASE_HEIGHT):
        actual_size = image.size
        image.close()
        raise RuntimeError(f"SD повернув неправильний розмір: {actual_size}")

    return image


def make_final_image(image: Image.Image) -> Image.Image:
    # Guarantee the visual language requested by the project even if SD adds
    # a small amount of color despite the prompt.
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)

    # Preserve the exact 16:9 composition and enlarge the ONE existing image.
    final = gray.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.Resampling.LANCZOS)

    # A restrained contrast boost makes graphite strokes clearer after upscale.
    final = ImageEnhance.Contrast(final).enhance(1.08)
    return final.convert("L")


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
                "generation_mode": "single_coherent_image_upscale",
                "base_generation_size": f"{SD_BASE_WIDTH}x{SD_BASE_HEIGHT}",
                "final_size": f"{FINAL_WIDTH}x{FINAL_HEIGHT}",
                "memory_mode": OFFLOAD_MODE,
                "tile_generation": False,
                "vae_tiling": True,
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

            if width != FINAL_WIDTH or height != FINAL_HEIGHT:
                raise ValueError(
                    f"Клієнт повинен запитувати фінальний розмір 1920x1080. Отримано {width}x{height}."
                )

            if not prompt:
                raise ValueError("Порожній prompt.")

            print(
                f"Генерую ОДНУ цілісну сцену: base={SD_BASE_WIDTH}x{SD_BASE_HEIGHT}, "
                f"final={FINAL_WIDTH}x{FINAL_HEIGHT}, steps={steps}, cfg={cfg}"
            )
            image = generate_one_image(prompt, negative_prompt, steps, cfg)
            try:
                final_image = make_final_image(image)
                image.close()
                try:
                    buffer = io.BytesIO()
                    final_image.save(buffer, format="PNG", optimize=True)
                    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                finally:
                    final_image.close()
            except Exception:
                image.close()
                raise

            self._send_json(200, {
                "images": [encoded],
                "info": "One coherent grayscale graphite scene, upscaled to 1920x1080",
            })

        except torch.cuda.OutOfMemoryError as exc:
            print(f"ПОМИЛКА CUDA OOM: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._send_json(500, {
                "error": "Навіть базове цілісне зображення не помістилося у VRAM.",
                "details": str(exc),
                "hint": "Зменш SD_BASE_WIDTH/SD_BASE_HEIGHT, наприклад до 640x360 або 512x288.",
            })
        except Exception as exc:
            print(f"ПОМИЛКА ГЕНЕРАЦІЇ: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._send_json(500, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        print("SD API:", format % args)


server = ThreadingHTTPServer((HOST, PORT), Handler)
print(f"Local SD API: http://{HOST}:{PORT}")
server.serve_forever()
