from __future__ import annotations

import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageEnhance, ImageOps

HOST = os.getenv("SD_SERVER_HOST", "127.0.0.1")
PORT = int(os.getenv("SD_SERVER_PORT", "7860"))
MODEL_PATH = Path(os.getenv("SD_MODEL_PATH", r"C:\Users\AI\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14"))

FINAL_WIDTH, FINAL_HEIGHT = 1920, 1080
COLS, ROWS = 4, 2
# A tile is a physical part of the final canvas. It is NOT an independent scene.
TILE_W = int(os.getenv("SD_TILE_WIDTH", "480"))
TILE_H = int(os.getenv("SD_TILE_HEIGHT", "544"))
OVERLAP = int(os.getenv("SD_TILE_OVERLAP", "64"))
STEPS = int(os.getenv("SD_TILE_STEPS", "16"))
DENOISE = float(os.getenv("SD_TILE_DENOISE", "0.42"))

if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна.")
if TILE_W % 8 or TILE_H % 8:
    raise RuntimeError("Tile dimensions must be divisible by 8.")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"FINAL CANVAS: {FINAL_WIDTH}x{FINAL_HEIGHT}")
print(f"ONE SCENE: {COLS}x{ROWS} sequential context tiles")
print(f"Tile: {TILE_W}x{TILE_H}, overlap: {OVERLAP}, steps: {STEPS}, denoise: {DENOISE}")

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    str(MODEL_PATH), torch_dtype=torch.float16, local_files_only=True, safety_checker=None
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
    print(f"CPU offload unavailable: {exc}")
    pipe = pipe.to("cuda")

STYLE = (
    "black and white graphite pencil drawing, monochrome hand-drawn pencil sketch, "
    "visible graphite strokes, realistic pencil shading, white paper texture, "
    "detailed linework, cinematic composition, natural perspective, coherent environment"
)
NEGATIVE = (
    "color, colored, photograph, photorealistic, photo, 3d render, CGI, digital painting, "
    "collage, grid, 2x2, 2x4, 4x2, multiple panels, split screen, contact sheet, storyboard, "
    "comic panels, multiple images, multiple scenes, duplicated character, repeated character, "
    "borders, frames, dividers, seams, hard seam, tiled layout, text, watermark, logo"
)


def target_box(col: int, row: int):
    x0 = col * FINAL_WIDTH // COLS
    x1 = (col + 1) * FINAL_WIDTH // COLS
    y0 = row * FINAL_HEIGHT // ROWS
    y1 = (row + 1) * FINAL_HEIGHT // ROWS
    return x0, y0, x1, y1


def context_box(col: int, row: int):
    x0, y0, x1, y1 = target_box(col, row)
    return max(0, x0 - OVERLAP), max(0, y0 - OVERLAP), min(FINAL_WIDTH, x1 + OVERLAP), min(FINAL_HEIGHT, y1 + OVERLAP)


def prompt_for(scene: str, col: int, row: int, has_context: bool) -> str:
    return (
        f"{scene}. {STYLE}. ONE SINGLE CONTINUOUS IMAGE, ONE COHERENT SCENE, SINGLE FRAME. "
        f"This is physical tile {row * COLS + col + 1} of 8 from one 1920x1080 artwork. "
        + ("Use the supplied neighboring artwork as visual continuity. Preserve every existing object, character, architecture, perspective, lighting and style and EXTEND them naturally across the boundary. " if has_context else "Establish the master scene composition and leave natural continuation toward the other tiles. ")
        + "Never make this tile a separate picture."
    )


def generate_tile(init: Image.Image, prompt: str, negative: str, seed: int, strength: float) -> Image.Image:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=init.convert("RGB"),
            strength=strength,
            num_inference_steps=STEPS,
            guidance_scale=7.0,
            generator=generator,
        )
    out = result.images[0].convert("L")
    del result
    torch.cuda.empty_cache()
    return out


def blend(canvas: Image.Image, generated: Image.Image, cb, tb, first: bool):
    cx0, cy0, cx1, cy1 = cb
    tx0, ty0, tx1, ty1 = tb
    generated = generated.resize((cx1 - cx0, cy1 - cy0), Image.Resampling.LANCZOS)
    if first:
        canvas.paste(generated, (cx0, cy0))
        return
    # New tile owns its target rectangle. In the overlap, blend gradually from
    # the existing neighboring pixels to the new pixels; no visible tile frame.
    mask = Image.new("L", generated.size, 255)
    p = mask.load()
    for y in range(generated.height):
        wy = cy0 + y
        for x in range(generated.width):
            wx = cx0 + x
            a = 255
            if wx < tx0:
                a = min(a, int(255 * (wx - cx0 + 1) / max(1, tx0 - cx0)))
            if wy < ty0:
                a = min(a, int(255 * (wy - cy0 + 1) / max(1, ty0 - cy0)))
            p[x, y] = a
    old = canvas.crop(cb)
    merged = Image.composite(generated, old, mask)
    canvas.paste(merged, (cx0, cy0))
    generated.close(); old.close(); merged.close(); mask.close()


def generate_scene(scene: str, negative: str, seed: int) -> Image.Image:
    # ONE master canvas lives in CPU RAM. GPU receives only one tile/context at a time.
    canvas = Image.new("L", (FINAL_WIDTH, FINAL_HEIGHT), 235)
    done = False
    for i in range(8):
        col, row = i % COLS, i // COLS
        tb = target_box(col, row)
        cb = context_box(col, row)
        has_context = done
        if has_context:
            init = canvas.crop(cb)
        else:
            init = Image.new("RGB", (cb[2] - cb[0], cb[3] - cb[1]), (235, 235, 235))
        try:
            print(f"  TILE {i + 1}/8  target={tb} context={cb}")
            tile = generate_tile(init, prompt_for(scene, col, row, has_context), negative, seed + i, 0.97 if not has_context else DENOISE)
        finally:
            init.close()
        try:
            blend(canvas, tile, cb, tb, i == 0)
        finally:
            tile.close()
        done = True
    return ImageEnhance.Contrast(ImageOps.autocontrast(canvas, cutoff=1)).enhance(1.08)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: dict):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/sdapi/v1/options":
            self.send_json(200, {
                "generation_mode": "ONE_SCENE_8_CONTEXT_TILES",
                "tile_grid": "4x2",
                "tile_size": f"{TILE_W}x{TILE_H}",
                "final_size": "1920x1080",
                "overlap": OVERLAP,
                "upscale_final": False,
                "memory_mode": OFFLOAD_MODE,
            })
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/sdapi/v1/txt2img":
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = str(req.get("prompt", "")).strip()
            negative = f"{req.get('negative_prompt', '')}, {NEGATIVE}"
            width = int(req.get("width", FINAL_WIDTH))
            height = int(req.get("height", FINAL_HEIGHT))
            seed = int(req.get("seed", 123456))
            if not prompt:
                raise ValueError("Порожній prompt.")
            if (width, height) != (FINAL_WIDTH, FINAL_HEIGHT):
                raise ValueError("Фінальний розмір повинен бути 1920x1080.")
            print("=== ONE CONTINUOUS SCENE: 8 tiles, shared visual context ===")
            image = generate_scene(prompt, negative, seed)
            try:
                buf = io.BytesIO(); image.save(buf, format="PNG", optimize=True)
                encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            finally:
                image.close()
            self.send_json(200, {"images": [encoded], "info": "One continuous 1920x1080 graphite scene from 8 context-aware tiles; no upscale."})
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            self.send_json(500, {"error": "CUDA OOM", "details": str(exc), "hint": "Try SD_TILE_WIDTH=384 SD_TILE_HEIGHT=448"})
        except Exception as exc:
            torch.cuda.empty_cache()
            print(f"GENERATION ERROR: {exc}")
            self.send_json(500, {"error": str(exc)})

    def log_message(self, format: str, *args):
        print("SD API:", format % args)


server = ThreadingHTTPServer((HOST, PORT), Handler)
print(f"Local SD API: http://{HOST}:{PORT}")
server.serve_forever()
