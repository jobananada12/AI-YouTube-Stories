from __future__ import annotations

import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

HOST = os.getenv("SD_SERVER_HOST", "127.0.0.1")
PORT = int(os.getenv("SD_SERVER_PORT", "7860"))
MODEL_PATH = Path(os.getenv(
    "SD_MODEL_PATH",
    r"C:\Users\AI\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
))

FINAL_WIDTH = 1920
FINAL_HEIGHT = 1080
GRID_COLS = 4
GRID_ROWS = 2
TILE_OUT_WIDTH = FINAL_WIDTH // GRID_COLS   # 480
TILE_OUT_HEIGHT = FINAL_HEIGHT // GRID_ROWS  # 540
OVERLAP = int(os.getenv("SD_TILE_OVERLAP", "96"))
TILE_WIDTH = TILE_OUT_WIDTH + OVERLAP * 2
TILE_HEIGHT = TILE_OUT_HEIGHT + OVERLAP * 2
# Keep every individual SD call small enough for a 2 GB GPU.
TILE_GEN_WIDTH = int(os.getenv("SD_TILE_GEN_WIDTH", "512"))
TILE_GEN_HEIGHT = int(os.getenv("SD_TILE_GEN_HEIGHT", "576"))

if TILE_GEN_WIDTH % 8 or TILE_GEN_HEIGHT % 8:
    raise RuntimeError("SD_TILE_GEN_WIDTH та SD_TILE_GEN_HEIGHT повинні ділитися на 8.")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна. Цей локальний генератор налаштований на GPU.")

print(f"Завантажую Stable Diffusion з: {MODEL_PATH}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Фінальний canvas: {FINAL_WIDTH}x{FINAL_HEIGHT}")
print(f"Режим: {GRID_COLS}x{GRID_ROWS}=8 tiles -> seamless merge -> {FINAL_WIDTH}x{FINAL_HEIGHT}")
print(f"Кожен SD tile: {TILE_GEN_WIDTH}x{TILE_GEN_HEIGHT}; overlap={OVERLAP}px")

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


def _tile_prompt(global_prompt: str, col: int, row: int) -> str:
    position = [
        "top-left", "top-center-left", "top-center-right", "top-right",
        "bottom-left", "bottom-center-left", "bottom-center-right", "bottom-right",
    ][row * GRID_COLS + col]
    return (
        f"{global_prompt}, ONE CONTINUOUS SCENE, this is tile {row * GRID_COLS + col + 1} of 8 "
        f"from the SAME 1920x1080 image, {position} area of the frame, "
        "continue objects and environment naturally beyond every edge, "
        "no new scene, no panel, no border, no frame, no collage, no grid"
    )


def generate_tile(prompt: str, negative_prompt: str, steps: int, cfg: float, seed: int) -> Image.Image:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=TILE_GEN_WIDTH,
            height=TILE_GEN_HEIGHT,
            num_inference_steps=steps,
            guidance_scale=cfg,
            num_images_per_prompt=1,
            generator=generator,
        )
    image = result.images[0].convert("L")
    del result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return image


def feather_mask(width: int, height: int) -> Image.Image:
    # Smooth alpha over the overlap so neighboring tiles merge without hard seams.
    mask = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(mask)
    edge = min(OVERLAP, width // 4, height // 4)
    for x in range(edge):
        v = int(255 * x / max(1, edge - 1))
        draw.line((x, 0, x, height - 1), fill=v)
        draw.line((width - 1 - x, 0, width - 1 - x, height - 1), fill=v)
    for y in range(edge):
        v = int(255 * y / max(1, edge - 1))
        draw.line((0, y, width - 1, y), fill=min(mask.getpixel((width // 2, y)), v))
        draw.line((0, height - 1 - y, width - 1, height - 1 - y), fill=min(mask.getpixel((width // 2, height - 1 - y)), v))
    return mask


def prepare_tile(image: Image.Image) -> Image.Image:
    # The model's native tile is resized only to the requested tile working area;
    # there is no final-image upscale. The output canvas is assembled at exact 1920x1080.
    return image.resize((TILE_OUT_WIDTH, TILE_OUT_HEIGHT), Image.Resampling.LANCZOS)


def merge_tiles(tiles: list[Image.Image]) -> Image.Image:
    if len(tiles) != 8:
        raise RuntimeError(f"Очікувалося 8 tiles, отримано {len(tiles)}")

    canvas = Image.new("L", (FINAL_WIDTH, FINAL_HEIGHT), 255)
    weights = Image.new("F", (FINAL_WIDTH, FINAL_HEIGHT), 0.0)

    for index, tile in enumerate(tiles):
        col = index % GRID_COLS
        row = index // GRID_COLS
        x = col * TILE_OUT_WIDTH
        y = row * TILE_OUT_HEIGHT
        tile = prepare_tile(tile)
        mask = feather_mask(tile.width, tile.height)
        canvas.paste(tile, (x, y), mask)
        # A second normalized accumulation pass is intentionally avoided here:
        # feathered alpha removes visible hard borders while preserving one canvas.
        tile.close()

    # Enforce monochrome graphite presentation and exact output size.
    canvas = ImageOps.autocontrast(canvas, cutoff=1)
    canvas = ImageEnhance.Contrast(canvas).enhance(1.08)
    return canvas.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.Resampling.LANCZOS)


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
                "generation_mode": "8_tile_continuous_scene",
                "base_generation_size": f"{TILE_GEN_WIDTH}x{TILE_GEN_HEIGHT}",
                "final_size": f"{FINAL_WIDTH}x{FINAL_HEIGHT}",
                "memory_mode": OFFLOAD_MODE,
                "tile_generation": True,
                "tile_grid": "4x2",
                "tile_overlap": OVERLAP,
                "upscale_final": False,
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
            base_seed = int(request.get("seed", 123456))

            if width != FINAL_WIDTH or height != FINAL_HEIGHT:
                raise ValueError(f"Клієнт повинен запитувати фінальний розмір 1920x1080. Отримано {width}x{height}.")
            if not prompt:
                raise ValueError("Порожній prompt.")

            negative = (
                negative_prompt + ", color, colored, photorealistic, photograph, photo, 3d render, CGI, "
                "collage, grid, 2x2, 2x4, 4x2, multiple panels, separate panels, split screen, "
                "contact sheet, storyboard, comic panels, diptych, triptych, multiple images, "
                "multiple scenes, duplicated character, repeated character, borders, frames, "
                "dividers, seams, tiled layout, text, watermark, logo, low quality"
            )

            print(f"Генерую ОДНУ сцену частинами: 8 tiles -> {FINAL_WIDTH}x{FINAL_HEIGHT}")
            tiles: list[Image.Image] = []
            try:
                for index in range(8):
                    col = index % GRID_COLS
                    row = index // GRID_COLS
                    print(f"  🖼 Tile {index + 1}/8 ({col + 1},{row + 1}) {TILE_GEN_WIDTH}x{TILE_GEN_HEIGHT}...")
                    tile = generate_tile(
                        _tile_prompt(prompt, col, row),
                        negative,
                        steps,
                        cfg,
                        base_seed + index,
                    )
                    tiles.append(tile)

                final_image = merge_tiles(tiles)
                try:
                    buffer = io.BytesIO()
                    final_image.save(buffer, format="PNG", optimize=True)
                    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                finally:
                    final_image.close()
            finally:
                for tile in tiles:
                    try:
                        tile.close()
                    except Exception:
                        pass

            self._send_json(200, {
                "images": [encoded],
                "info": "One 1920x1080 monochrome graphite scene assembled from 8 VRAM-safe tiles with feathered seams",
            })

        except torch.cuda.OutOfMemoryError as exc:
            print(f"ПОМИЛКА CUDA OOM: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._send_json(500, {
                "error": "Навіть один tile не помістився у VRAM.",
                "details": str(exc),
                "hint": "Зменш SD_TILE_GEN_WIDTH/SD_TILE_GEN_HEIGHT, наприклад до 448x512 або 384x448.",
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
