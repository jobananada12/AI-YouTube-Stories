from __future__ import annotations

import base64
import json
from pathlib import Path

import requests

from app.config import settings
from core.character_bible_schema import CharacterBible
from core.scene_schema import ScenePlan


class ImageGenerationError(RuntimeError):
    pass


class ImageGenerator:
    """Generate scene PNGs through an optional local Automatic1111-compatible API."""

    def __init__(self, api_url: str | None = None, model: str | None = None):
        self.api_url = (api_url or settings.image_api_url).rstrip("/")
        self.model = model or settings.image_model

    def _prompt(self, scene, bible: CharacterBible) -> str:
        anchors = []
        for name in scene.characters:
            match = next((c for c in bible.characters if c.name == name), None)
            if match:
                anchors.append(match.image_prompt_anchor)
        return (
            "Оригінальна вигадана сцена для YouTube-історії. "
            "Кінематографічна композиція, природне освітлення, деталізоване середовище, "
            "виразні емоції, 16:9. Не використовуй реальних людей, відомих персонажів, "
            "франшизи або імена художників. " + scene.visual_prompt + " " + " ".join(anchors)
        )

    def generate(self, scene_plan: ScenePlan, bible: CharacterBible, output_dir: str | Path) -> list[Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if not self.api_url:
            raise ImageGenerationError(
                "IMAGE_API_URL не задано. Для локальної генерації запусти Automatic1111/сумісний API "
                "і вкажи IMAGE_API_URL у .env."
            )
        result = []
        for scene in scene_plan.scenes:
            payload = {
                "prompt": self._prompt(scene, bible),
                "negative_prompt": "text, watermark, logo, celebrity, copyrighted character, deformed hands, extra fingers",
                "width": 1280,
                "height": 720,
                "steps": settings.image_steps,
                "cfg_scale": settings.image_cfg_scale,
            }
            if self.model:
                payload["override_settings"] = {"sd_model_checkpoint": self.model}
            response = requests.post(f"{self.api_url}/sdapi/v1/txt2img", json=payload, timeout=900)
            response.raise_for_status()
            data = response.json()
            images = data.get("images") or []
            if not images:
                raise ImageGenerationError(f"Генератор не повернув зображення для сцени {scene.number}")
            target = output / f"scene_{scene.number:03d}.png"
            target.write_bytes(base64.b64decode(images[0]))
            result.append(target)
        return result
