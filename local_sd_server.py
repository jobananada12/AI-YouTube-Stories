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
MODEL_PATH = Path(os.getenv("SD_MODEL_PATH", r"C:\Users\AI\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14"))

# Generate one complete 16:9 image at a VRAM-friendly size, then upscale it.
GEN_WIDTH, GEN_HEIGHT = 768, 432
FINAL_WIDTH, FINAL_HEIGHT = 1920, 1080
STEPS = int(os.getenv("SD_STEPS", "20"))
CFG = float(os.getenv("SD_CFG", "7.0"))

if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна.")
if GEN_WIDTH % 8 or GEN_HEIGHT % 8:
    raise RuntimeError("Generation dimensions must be divisible by 8.")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GENERATE: {GEN_WIDTH}x{GEN_HEIGHT} (16:9)")
print(f"FINAL: {FINAL_WIDTH}x{FINAL_HEIGHT} (16:9)")
print(f"ONE COMPLETE SCENE, no tiles, no collage, steps: {STEPS}, cfg: {CFG}")

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
    print(f"CPU offload unavailable: {exc}")
    pipe = pipe.to("cuda")

STYLE = (
    "black and white graphite pencil drawing, monochrome hand-drawn pencil sketch, "
    "visible graphite strokes, realistic pencil shading, white paper texture, "
    "detailed linework, cinematic composition, natural perspective, coherent environment"
)
NEGATIVE = (
    "color, colored, photograph, photorealistic, photo, 3d render, CGI, digital painting, "
    "collage, grid, multiple panels, split screen, contact sheet, storyboard, comic panels, "
    "multiple images, multiple scenes, duplicated character, repeated character, borders, "
    "frames, dividers, seams, tiled layout, text, watermark, logo, low quality, blurry, "
    "deformed, cropped subject"
)


def generate_scene(prompt: str, negative: str, seed: int) -> Image.Image:
    """Generate ONE complete 16:9 scene, then upscale the finished image."""
    full_prompt = (
        f"{prompt}. {STYLE}. ONE SINGLE COMPLETE IMAGE. ONE SINGLE CONTINUOUS SCENE. "
        "Wide cinematic 16:9 composition. Show the entire environment as one coherent "
        "composition with natural perspective, consistent lighting and complete objects. "
        "Do not split the image into panels or separate pictures."
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            prompt=full_prompt,
            negative_prompt=negative,
            width=GEN_WIDTH,
            height=GEN_HEIGHT,
            num_inference_steps=STEPS,
            guidance_scale=CFG,
            generator=generator,
            batch_size=1,
        )
    image = result.images[0].convert("L")
    del result
    torch.cuda.empty_cache()

    # Upscale the already-complete composition. No tiles and no cropping.
    image = image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.Resampling.LANCZOS)
    image = ImageEnhance.Contrast(ImageOps.autocontrast(image, cutoff=1)).enhance(1.08)
    return image


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
                "generation_mode": "ONE_COMPLETE_SCENE_UPSCALE",
                "generation_size": f"{GEN_WIDTH}x{GEN_HEIGHT}",
                "final_size": f"{FINAL_WIDTH}x{FINAL_HEIGHT}",
                "upscale_final": True,
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
            # Client requests the final canvas, but the server generates the complete
            # scene at 768x432 and upscales it to 1920x1080 without cropping.
            if (width, height) != (FINAL_WIDTH, FINAL_HEIGHT):
                raise ValueError("Фінальний розмір повинен бути 1920x1080.")
            print("=== ONE COMPLETE 16:9 SCENE: 768x432 -> 1920x1080 ===")
            image = generate_scene(prompt, negative, seed)
            try:
                buf = io.BytesIO()
                image.save(buf, format="PNG", optimize=True)
                encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            finally:
                image.close()
            self.send_json(200, {
                "images": [encoded],
                "info": "One complete 768x432 graphite scene upscaled to 1920x1080; no tiles and no crop.",
            })
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            self.send_json(500, {
                "error": "CUDA OOM",
                "details": str(exc),
                "hint": "Set SD_STEPS lower or reduce generation size if needed.",
            })
        except Exception as exc:
            torch.cuda.empty_cache()
            print(f"GENERATION ERROR: {exc}")
            self.send_json(500, {"error": str(exc)})

    def log_message(self, format: str, *args):
        print("SD API:", format % args)


server = ThreadingHTTPServer((HOST, PORT), Handler)
print(f"Local SD API: http://{HOST}:{PORT}")
server.serve_forever()
