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
# One SD call creates one tile. Tiles overlap in the final canvas and are
# feather-blended, so the eight calls form ONE continuous scene.
TILE_GEN_WIDTH = int(os.getenv("SD_TILE_GEN_WIDTH", "512"))
TILE_GEN_HEIGHT = int(os.getenv("SD_TILE_GEN_HEIGHT", "576"))
STEP_X = FINAL_WIDTH // GRID_COLS  # 480
STEP_Y = FINAL_HEIGHT // GRID_ROWS  # 540
TOTAL_TILE_WIDTH = STEP_X * (GRID_COLS - 1) + TILE_GEN_WIDTH
TOTAL_TILE_HEIGHT = STEP_Y * (GRID_ROWS - 1) + TILE_GEN_HEIGHT
CROP_X = max(0, TOTAL_TILE_WIDTH - FINAL_WIDTH) // 2
CROP_Y = max(0, TOTAL_TILE_HEIGHT - FINAL_HEIGHT) // 2

if TILE_GEN_WIDTH % 8 or TILE_GEN_HEIGHT % 8:
    raise RuntimeError("SD_TILE_GEN_WIDTH та SD_TILE_GEN_HEIGHT повинні ділитися на 8.")
if TILE_GEN_WIDTH < STEP_X or TILE_GEN_HEIGHT < STEP_Y:
    raise RuntimeError("Розмір tile повинен бути не меншим за крок сітки для перекриття.")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна. Цей локальний генератор налаштований на GPU.")

print(f"Завантажую Stable Diffusion з: {MODEL_PATH}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Фінальний canvas: {FINAL_WIDTH}x{FINAL_HEIGHT}")
print(f"Режим: 4x2=8 tiles -> overlap blend -> exact {FINAL_WIDTH}x{FINAL_HEIGHT}")
print(f"Tile generation: {TILE_GEN_WIDTH}x{TILE_GEN_HEIGHT}; steps: {STEP_X}x{STEP_Y}")
print(f"Загальне перекриття: X={TILE_GEN_WIDTH - STEP_X}px, Y={TILE_GEN_HEIGHT - STEP_Y}px")

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
        f"{global_prompt}, ONE SINGLE CONTINUOUS SCENE, this is tile {row * GRID_COLS + col + 1} of 8 "
        f"from the SAME 1920x1080 image, {position} area of the same frame, "
        "continue every object, character, architecture and background naturally across all edges, "
        "the scene continues outside this tile, same perspective and same lighting, "
        "single coherent composition, no new scene"
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
    if image.size != (TILE_GEN_WIDTH, TILE_GEN_HEIGHT):
        image.close()
        raise RuntimeError("Stable Diffusion повернув tile неправильного розміру")
    return image


def feather_mask(width: int, height: int) -> Image.Image:
    # The mask fades only inside the actual overlap area. Interior pixels stay opaque.
    overlap_x = max(1, width - STEP_X)
    overlap_y = max(1, height - STEP_Y)
    mask = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(mask)
    for x in range(overlap_x):
        alpha = int(255 * (x + 1) / (overlap_x + 1))
        draw.line((x, 0, x, height - 1), fill=alpha)
        draw.line((width - 1 - x, 0, width - 1 - x, height - 1), fill=alpha)
    for y in range(overlap_y):
        alpha = int(255 * (y + 1) / (overlap_y + 1))
        draw.line((0, y, width - 1, y), fill=min(mask.getpixel((width // 2, y)), alpha))
        draw.line((0, height - 1 - y, width - 1, height - 1 - y), fill=min(mask.getpixel((width // 2, height - 1 - y)), alpha))
    return mask


def merge_tiles(tiles: list[Image.Image]) -> Image.Image:
    if len(tiles) != 8:
        raise RuntimeError(f"Очікувалося 8 tiles, отримано {len(tiles)}")

    # Work on an oversized canvas because neighboring tiles intentionally overlap.
    canvas = Image.new("L", (TOTAL_TILE_WIDTH, TOTAL_TILE_HEIGHT), 255)
    coverage = Image.new("L", (TOTAL_TILE_WIDTH, TOTAL_TILE_HEIGHT), 0)
    mask = feather_mask(TILE_GEN_WIDTH, TILE_GEN_HEIGHT)

    for index, tile in enumerate(tiles):
        col = index % GRID_COLS
        row = index // GRID_COLS
        x = col * STEP_X
        y = row * STEP_Y
        canvas.paste(tile, (x, y), mask)
        # Keep track of the covered area so final edge crop is deterministic.
        coverage.paste(255, (x, y), mask)

    # Center-crop only the overlap margin. This is not an upscale: the final
    # pixels are directly assembled from the generated tiles at native size.
    final = canvas.crop((CROP_X, CROP_Y, CROP_X + FINAL_WIDTH, CROP_Y + FINAL_HEIGHT))
    final = ImageOps.autocontrast(final, cutoff=1)
    final = ImageEnhance.Contrast(final).enhance(1.08)
    return final


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
                "tile_overlap_x": TILE_GEN_WIDTH - STEP_X,
                "tile_overlap_y": TILE_GEN_HEIGHT - STEP_Y,
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
                "dividers, seams, hard seam, tiled layout, text, watermark, logo, low quality"
            )

            print(f"Генерую ОДНУ сцену: 8 tiles -> seamless overlap blend -> {FINAL_WIDTH}x{FINAL_HEIGHT}")
            tiles: list[Image.Image] = []
            try:
                for index in range(8):
                    col = index % GRID_COLS
                    row = index // GRID_COLS
                    print(f"  🖼 Tile {index + 1}/8 ({col + 1},{row + 1}) {TILE_GEN_WIDTH}x{TILE_GEN_HEIGHT}...")
                    tiles.append(generate_tile(
                        _tile_prompt(prompt, col, row),
                        negative,
                        steps,
                        cfg,
                        base_seed + index,
                    ))

                final_image = merge_tiles(tiles)
                try:
                    buffer = io.BytesIO()
                    final_image.save(buffer, format="PNG", optimize=True)
                    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                finally:
                    final_image.close()
            finally:
                for tile in tiles:
                    tile.close()

            self._send_json(200, {
                "images": [encoded],
                "info": "One exact 1920x1080 monochrome graphite scene assembled from 8 overlapping VRAM-safe tiles; no final upscale",
            })

        except torch.cuda.OutOfMemoryError as exc:
            print(f"ПОМИЛКА CUDA OOM: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._send_json(500, {
                "error": "Tile не помістився у 2 GB VRAM.",
                "details": str(exc),
                "hint": "Зменш SD_TILE_GEN_WIDTH/SD_TILE_GEN_HEIGHT до 448x512 або 384x448.",
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
