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
    """Generate scene PNGs through a local Automatic1111-compatible API."""

    def __init__(self, api_url: str | None = None, model: str | None = None):
        self.api_url = (api_url or settings.image_api_url).rstrip("/")
        self.model = model or settings.image_model

    def _prompt(self, scene, bible: CharacterBible) -> str:
        # scene.visual_prompt is validated by ScenePlanner and must be English-only.
        # Character anchors are appearance canon and are appended only for characters
        # explicitly present in this scene.
        anchors = []
        for name in scene.characters:
            match = next((c for c in bible.characters if c.name == name), None)
            if match and match.image_prompt_anchor:
                anchors.append(match.image_prompt_anchor)

        parts = [scene.visual_prompt.strip()]
        if anchors:
            parts.extend(anchors)
        parts.append("cinematic still, realistic lighting, detailed environment, 16:9 composition")
        return ", ".join(p for p in parts if p)

    def _check_server(self) -> None:
        try:
            response = requests.get(f"{self.api_url}/sdapi/v1/options", timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ImageGenerationError(
                f"Не можу підключитися до локального генератора: {self.api_url}. "
                "Запусти Automatic1111/сумісний API з --api та перевір IMAGE_API_URL у .env."
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
                    "text, watermark, logo, celebrity, copyrighted character, franchise, "
                    "deformed hands, extra fingers, duplicate person, blurry, low quality"
                ),
                "width": settings.image_width,
                "height": settings.image_height,
                "steps": settings.image_steps,
                "cfg_scale": settings.image_cfg_scale,
                "batch_size": 1,
                "n_iter": 1,
            }
            if self.model:
                payload["override_settings"] = {"sd_model_checkpoint": self.model}

            print(f"  🖼 Сцена {index}/{total}: генерую {settings.image_width}x{settings.image_height}...")
            try:
                response = requests.post(
                    f"{self.api_url}/sdapi/v1/txt2img",
                    json=payload,
                    timeout=settings.image_timeout,
                )
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as exc:
                raise ImageGenerationError(f"Помилка генерації сцени {scene.number}: {exc}") from exc

            images = data.get("images") or []
            if not images:
                raise ImageGenerationError(f"Генератор не повернув зображення для сцени {scene.number}")

            target = output / f"scene_{scene.number:03d}.png"
            try:
                target.write_bytes(base64.b64decode(images[0]))
            except Exception as exc:
                raise ImageGenerationError(
                    f"Не вдалося зберегти зображення сцени {scene.number}: {exc}"
                ) from exc
            result.append(target)

        return result
