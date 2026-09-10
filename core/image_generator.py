from __future__ import annotations

import base64
from pathlib import Path

import requests

from app.config import settings
from core.character_bible_schema import CharacterBible
from core.scene_schema import ScenePlan


class ImageGenerationError(RuntimeError):
    pass


class ImageGenerator:
    """Generate one complete 16:9 scene through the local SD server."""

    def __init__(self, api_url: str | None = None, model: str | None = None):
        self.api_url = (api_url or settings.image_api_url).rstrip("/")
        self.model = model or settings.image_model

    def _prompt(self, scene, bible: CharacterBible) -> str:
        anchors = []
        for name in scene.characters:
            match = next((c for c in bible.characters if c.name == name), None)
            if match and match.image_prompt_anchor:
                anchors.append(match.image_prompt_anchor)

        parts = [
            "ONE SINGLE COMPLETE IMAGE",
            "ONE SINGLE CONTINUOUS SCENE",
            "ONE COHERENT CINEMATIC COMPOSITION",
            "the entire frame is one connected environment, not separate pictures",
            "wide 16:9 composition",
            "black and white graphite pencil drawing",
            "hand-drawn graphite pencil sketch",
            "pure monochrome grayscale",
            "visible graphite pencil strokes",
            "realistic pencil shading",
            "cross-hatching",
            "detailed paper texture",
            "illustrated drawing, not a photograph",
            "consistent character appearance, perspective, lighting and architecture",
            scene.visual_prompt.strip(),
        ]
        if anchors:
            parts.extend(anchors)
        return ", ".join(p for p in parts if p)

    def _check_server(self) -> None:
        try:
            response = requests.get(f"{self.api_url}/sdapi/v1/options", timeout=10)
            response.raise_for_status()
            options = response.json()
            mode = options.get("generation_mode", "unknown")
            generation = options.get("generation_size", "unknown")
            final = options.get("final_size", "unknown")
            upscale = options.get("upscale_final", False)
            memory = options.get("memory_mode", "unknown")
            print(
                f"  🧠 SD server: {mode}, generate={generation}, "
                f"final={final}, upscale={upscale}, memory={memory}"
            )
            if final != "1920x1080":
                raise ImageGenerationError("SD server не налаштований на фінальний розмір 1920x1080.")
            if generation != "768x432":
                raise ImageGenerationError("SD server не налаштований на генерацію 768x432 (16:9).")
            if mode != "ONE_COMPLETE_SCENE_UPSCALE" or not upscale:
                raise ImageGenerationError(
                    "SD server не працює у режимі одна повна сцена 16:9 -> upscale."
                )
        except ImageGenerationError:
            raise
        except requests.RequestException as exc:
            raise ImageGenerationError(
                f"Не можу підключитися до локального генератора: {self.api_url}. "
                "Запусти local_sd_server.py та перевір IMAGE_API_URL у .env."
            ) from exc

    def generate(self, scene_plan: ScenePlan, bible: CharacterBible, output_dir: str | Path) -> list[Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if not self.api_url:
            raise ImageGenerationError("IMAGE_API_URL не задано.")

        self._check_server()
        result: list[Path] = []
        total = len(scene_plan.scenes)

        for index, scene in enumerate(scene_plan.scenes, 1):
            payload = {
                "prompt": self._prompt(scene, bible),
                "negative_prompt": (
                    "color, colored, photorealistic, photograph, photo, 3d render, CGI, "
                    "collage, grid, multiple panels, separate pictures, split screen, "
                    "contact sheet, storyboard, comic panels, diptych, triptych, "
                    "multiple images, multiple scenes, duplicated character, repeated character, "
                    "borders, frames, dividers, seams, tiled layout, text, watermark, logo, "
                    "low quality, blurry, deformed, cropped subject"
                ),
                # Client asks for the final canvas. The local server generates one
                # complete 768x432 composition and upscales that same image to 1920x1080.
                "width": settings.image_width,
                "height": settings.image_height,
                "steps": settings.image_steps,
                "cfg_scale": settings.image_cfg_scale,
                "batch_size": 1,
                "n_iter": 1,
                "seed": 100000 + int(scene.number),
            }

            print(
                f"  🖼 Сцена {index}/{total}: ОДНА ПОВНА КАРТИНА "
                "768x432 -> 1920x1080, без тайлів і без обрізання..."
            )
            try:
                response = requests.post(
                    f"{self.api_url}/sdapi/v1/txt2img",
                    json=payload,
                    timeout=settings.image_timeout,
                )
                if not response.ok:
                    detail = response.text.strip()
                    try:
                        body = response.json()
                        detail = body.get("error") or body.get("details") or detail
                    except ValueError:
                        pass
                    raise ImageGenerationError(
                        f"SD API HTTP {response.status_code}: {detail or 'невідома помилка сервера'}"
                    )
                data = response.json()
            except ImageGenerationError:
                raise
            except requests.RequestException as exc:
                raise ImageGenerationError(f"Помилка генерації сцени {scene.number}: {exc}") from exc

            images = data.get("images") or []
            if not images:
                raise ImageGenerationError(f"Генератор не повернув зображення для сцени {scene.number}")

            target = output / f"scene_{scene.number:03d}.png"
            try:
                raw = base64.b64decode(images[0])
                target.write_bytes(raw)
            except Exception as exc:
                raise ImageGenerationError(
                    f"Не вдалося зберегти зображення сцени {scene.number}: {exc}"
                ) from exc
            result.append(target)

        return result
