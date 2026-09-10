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
    """Generate one 1920x1080 coherent scene through the local 8-tile SD API."""

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
            "ONE SINGLE CONTINUOUS IMAGE",
            "ONE SINGLE SCENE, ONE COHERENT CINEMATIC COMPOSITION",
            "the entire final frame is one connected environment, not separate pictures",
            "black and white graphite pencil drawing",
            "hand-drawn graphite pencil sketch",
            "pure monochrome grayscale",
            "visible graphite pencil strokes",
            "realistic pencil shading",
            "cross-hatching",
            "detailed paper texture",
            "illustrated drawing, not a photograph",
            "consistent character appearance, perspective, lighting and architecture",
            "objects and characters continue naturally across image boundaries",
            scene.visual_prompt.strip(),
        ]
        if anchors:
            parts.extend(anchors)
        parts.append("16:9 composition")
        return ", ".join(p for p in parts if p)

    def _check_server(self) -> None:
        try:
            response = requests.get(f"{self.api_url}/sdapi/v1/options", timeout=10)
            response.raise_for_status()
            options = response.json()
            mode = options.get("generation_mode", "unknown")
            memory = options.get("memory_mode", "unknown")
            tile_size = options.get("tile_size", "unknown")
            final = options.get("final_size", "unknown")
            tiles = options.get("tile_grid", "unknown")
            print(
                f"  🧠 SD server: {mode}, tiles={tiles}, tile={tile_size}, "
                f"final={final}, memory={memory}"
            )
            if final != "1920x1080":
                raise ImageGenerationError("SD server не налаштований на фінальний canvas 1920x1080.")
            if mode != "ONE_SCENE_8_CONTEXT_TILES" or tiles != "4x2":
                raise ImageGenerationError(
                    "SD server не працює у потрібному режимі ONE_SCENE_8_CONTEXT_TILES (4x2)."
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
                    "collage, grid, 2x2, 2x4, 4x2, multiple panels, separate panels, "
                    "split screen, contact sheet, storyboard, comic panels, diptych, triptych, "
                    "multiple images, multiple scenes, duplicated character, repeated character, "
                    "borders, frames, dividers, hard seams, visible seams, tiled layout, "
                    "text, watermark, logo, low quality"
                ),
                # The server always returns exactly one final 1920x1080 canvas.
                # Internally it creates 8 overlapping VRAM-safe tiles and merges them.
                "width": settings.image_width,
                "height": settings.image_height,
                "steps": settings.image_steps,
                "cfg_scale": settings.image_cfg_scale,
                "batch_size": 1,
                "n_iter": 1,
                "seed": 100000 + int(scene.number),
            }
            if self.model:
                payload["override_settings"] = {"sd_model_checkpoint": self.model}

            print(
                f"  🖼 Сцена {index}/{total}: 8 частин ОДНІЄЇ сцени -> "
                f"цілісний {settings.image_width}x{settings.image_height} без фінального upscale..."
            )
            try:
                response = requests.post(
                    f"{self.api_url}/sdapi/v1/txt2img",
                    json=payload,
                    timeout=settings.image_timeout * 8,
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
