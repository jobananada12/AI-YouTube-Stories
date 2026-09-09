import os
import json
import time
import gc
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


# ============================================================
# AI YouTube Stories — Stable Diffusion Scene Generator
# Optimized for NVIDIA GT 1030
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SCENES_FILE = BASE_DIR / "scenes.json"
OUTPUT_DIR = BASE_DIR / "generated_scenes"

MODEL_ID = "runwayml/stable-diffusion-v1-5"

# ------------------------------------------------------------
# GT 1030 optimization
# ------------------------------------------------------------

WIDTH = 384
HEIGHT = 384

STEPS = 10

GUIDANCE_SCALE = 6.5

SEED_BASE = 1000

# ------------------------------------------------------------
# Style
# ------------------------------------------------------------

STYLE_SUFFIX = (
    ", cinematic movie still, realistic digital illustration, "
    "dramatic lighting, atmospheric, detailed environment, "
    "professional cinematography, high detail"
)

NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed, bad anatomy, "
    "extra fingers, extra limbs, duplicate person, "
    "text, watermark, logo, subtitles, letters, "
    "cartoon, anime, 3d render, oversaturated"
)


# ============================================================
# Helpers
# ============================================================

def clear_memory():
    """Free Python and CUDA memory."""

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def format_time(seconds):
    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def progress_bar(current, total, width=30):
    if total <= 0:
        return "[" + "?" * width + "]"

    ratio = current / total
    filled = int(width * ratio)

    return "[" + "█" * filled + "░" * (width - filled) + "]"


def load_scenes():
    if not SCENES_FILE.exists():
        raise FileNotFoundError(
            f"\nНе знайдено файл:\n{SCENES_FILE}\n\n"
            "Спочатку запусти scene_planner.py"
        )

    with open(SCENES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # scenes.json може бути або списком,
    # або об'єктом {"scenes": [...]}
    if isinstance(data, list):
        scenes = data

    elif isinstance(data, dict):
        scenes = data.get("scenes", [])

    else:
        scenes = []

    if not scenes:
        raise ValueError("У scenes.json немає сцен.")

    return scenes


def get_scene_id(scene, index):
    value = scene.get("id", index + 1)

    try:
        return int(value)
    except Exception:
        return index + 1


def get_prompt(scene):
    """
    Підтримує кілька можливих назв поля prompt.
    """

    for key in [
        "prompt",
        "image_prompt",
        "stable_diffusion_prompt",
        "sd_prompt",
    ]:
        value = scene.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def build_prompt(prompt):
    prompt = prompt.strip()

    # Якщо planner уже додав стиль,
    # все одно можна безпечно додати наш cinematic suffix.
    return prompt + STYLE_SUFFIX


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 64)
    print("       AI YouTube Stories — Image Generator")
    print("=" * 64)
    print()

    print(f"Scenes file : {SCENES_FILE}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"Model       : {MODEL_ID}")
    print(f"Resolution  : {WIDTH}x{HEIGHT}")
    print(f"Steps       : {STEPS}")
    print(f"GPU         : NVIDIA GT 1030 optimization")
    print()

    # --------------------------------------------------------
    # Load scenes
    # --------------------------------------------------------

    scenes = load_scenes()

    total = len(scenes)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Знайдено сцен: {total}")
    print()

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = "cuda"

        gpu_name = torch.cuda.get_device_name(0)

        print(f"GPU: {gpu_name}")

        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            total_memory_mb = total_memory / 1024 / 1024

            print(f"VRAM: {total_memory_mb:.0f} MB")

        except Exception:
            pass

    else:

        device = "cpu"

        print("УВАГА: CUDA не знайдено.")
        print("Генерація буде виконуватися на CPU.")
        print()

    print()

    # --------------------------------------------------------
    # Load Stable Diffusion
    # --------------------------------------------------------

    print("Завантаження Stable Diffusion...")
    print()

    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # --------------------------------------------------------
    # Memory optimizations
    # --------------------------------------------------------

    if device == "cuda":

        pipe = pipe.to("cuda")

        # Attention slicing greatly reduces VRAM usage.
        try:
            pipe.enable_attention_slicing()
            print("✓ Attention slicing")
        except Exception:
            pass

        # VAE slicing
        try:
            pipe.enable_vae_slicing()
            print("✓ VAE slicing")
        except Exception:
            pass

        # xformers if installed and compatible
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("✓ xFormers memory efficient attention")
        except Exception:
            print("• xFormers не використовується")

    else:

        pipe = pipe.to("cpu")

    print()
    print("Stable Diffusion готовий.")
    print()

    clear_memory()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    generated = 0
    skipped = 0
    failed = 0

    started_at = time.time()

    # ========================================================
    # Generate scenes
    # ========================================================

    for index, scene in enumerate(scenes):

        scene_id = get_scene_id(scene, index)

        output_file = OUTPUT_DIR / f"scene_{scene_id:03d}.png"

        # ----------------------------------------------------
        # Skip existing
        # ----------------------------------------------------

        if output_file.exists():

            skipped += 1

            elapsed = time.time() - started_at

            completed = generated + skipped

            print()
            print(
                f"[{completed}/{total}] "
                f"{progress_bar(completed, total)} "
                f"{completed / total * 100:5.1f}%"
            )

            print(
                f"Scene {scene_id:03d}: "
                f"вже існує → пропускаємо"
            )

            continue

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = get_prompt(scene)

        if not prompt:

            failed += 1

            print()
            print(
                f"Scene {scene_id:03d}: "
                f"ПОМИЛКА — prompt відсутній"
            )

            continue

        final_prompt = build_prompt(prompt)

        # ----------------------------------------------------
        # Scene header
        # ----------------------------------------------------

        completed = generated + skipped

        print()
        print("=" * 64)
        print(
            f"SCENE {scene_id:03d} "
            f"({completed + 1}/{total})"
        )
        print("=" * 64)

        print()
        print("Prompt:")
        print(final_prompt)
        print()

        scene_started = time.time()

        # ----------------------------------------------------
        # Deterministic seed
        # ----------------------------------------------------

        seed = SEED_BASE + scene_id

        if device == "cuda":

            generator = torch.Generator(
                device="cuda"
            ).manual_seed(seed)

        else:

            generator = torch.Generator(
                device="cpu"
            ).manual_seed(seed)

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        try:

            print("Генерація...")

            result = pipe(
                prompt=final_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                width=WIDTH,
                height=HEIGHT,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            )

            image = result.images[0]

            image.save(output_file)

            generated += 1

            scene_time = time.time() - scene_started
            total_time = time.time() - started_at

            completed = generated + skipped

            # ------------------------------------------------
            # ETA
            # ------------------------------------------------

            if completed > 0:

                average = total_time / completed

                remaining = total - completed

                eta = average * remaining

            else:

                eta = 0

            print()
            print(
                f"✓ Готово: {output_file.name}"
            )

            print(
                f"Час сцени: {format_time(scene_time)}"
            )

            print(
                f"Загальний час: {format_time(total_time)}"
            )

            print(
                f"Орієнтовно залишилось: "
                f"{format_time(eta)}"
            )

            print()

            print(
                f"{progress_bar(completed, total)} "
                f"{completed}/{total} "
                f"({completed / total * 100:.1f}%)"
            )

        except torch.cuda.OutOfMemoryError:

            failed += 1

            print()
            print(
                "!!! НЕДОСТАТНЬО VRAM !!!"
            )

            print(
                "Очищаю CUDA memory..."
            )

            clear_memory()

            print(
                "Scene пропущено."
            )

        except Exception as e:

            failed += 1

            print()
            print(
                f"!!! ПОМИЛКА SCENE {scene_id:03d} !!!"
            )

            print(str(e))

            print()
            print(
                "Продовжуємо з наступної сцени..."
            )

            clear_memory()

        finally:

            clear_memory()

    # ========================================================
    # Final report
    # ========================================================

    total_time = time.time() - started_at

    print()
    print("=" * 64)
    print("                         ГОТОВО")
    print("=" * 64)
    print()

    print(f"Всього сцен:       {total}")
    print(f"Створено:          {generated}")
    print(f"Пропущено готових: {skipped}")
    print(f"Помилок:           {failed}")
    print()

    print(
        f"Загальний час: {format_time(total_time)}"
    )

    print()

    print(
        f"Зображення знаходяться тут:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print("=" * 64)

    if failed > 0:

        print(
            "УВАГА: деякі сцени не були створені."
        )

        print(
            "Запусти програму ще раз — готові "
            "зображення будуть пропущені."
        )

    else:

        print(
            "Усі сцени успішно створені."
        )

    print("=" * 64)
    print()


if __name__ == "__main__":
    main()