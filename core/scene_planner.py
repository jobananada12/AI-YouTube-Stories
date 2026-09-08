import json
import re
from typing import Iterable

from core.llm import OllamaClient
from core.scene_schema import ScenePlan, SceneSpec
from core.story_schema import StorySpec


SCENE_PROMPT = """
Ти — режисер і сторібордист YouTube-історій.

Отримуєш оригінальну історію та її готовий текст оповіді українською.
Твоє завдання — перетворити текст на режисерський план сцен для подальшого
створення AI-зображень, озвучки та монтажу.

ВАЖЛИВІ ПРАВИЛА:
- Історія повністю оригінальна. Не копіюй і не імітуй відомі фільми, книги,
  ігри, мультфільми, авторів або YouTube-канали.
- Не вигадуй нових подій, яких немає в наданій оповіді.
- Поле narration має містити ДОСЛІВНО відповідний фрагмент наданого тексту,
  без перефразування, скорочення чи додавання.
- Зберігай порядок усіх фрагментів.
- Кожен фрагмент має належати лише одній сцені.
- visual_prompt — детальний опис кадру для генератора зображень: персонажі,
  зовнішність, дія, локація, освітлення, атмосфера, композиція. Не називай
  реальних акторів, відомі франшизи чи конкретні художні стилі живих авторів.
- Підтримуй візуальну послідовність персонажів і локацій.
- Орієнтуйся приблизно на 25–50 секунд оповіді на сцену, але не ламай речення.
- Відповідь — ТІЛЬКИ валідний JSON без markdown і без пояснень.

Формат:
{
  "scenes": [
    {
      "number": 1,
      "title": "...",
      "purpose": "...",
      "narration": "дослівний фрагмент",
      "estimated_duration_seconds": 35,
      "characters": ["..."],
      "location": "...",
      "time_of_day": "...",
      "action": "...",
      "visual_prompt": "...",
      "mood": "...",
      "continuity_notes": "...",
      "transition": "cut"
    }
  ]
}

ІСТОРІЯ:
{story_json}

ТЕКСТ ОПОВІДІ:
{script}
"""


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_script(script: str, target_words: int = 85) -> list[str]:
    """Split narration into natural chunks without rewriting the source text."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", script) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    words = 0

    for paragraph in paragraphs:
        p_words = len(paragraph.split())
        if current and words + p_words > target_words:
            chunks.append("\n\n".join(current))
            current = []
            words = 0
        current.append(paragraph)
        words += p_words

        if words >= target_words:
            chunks.append("\n\n".join(current))
            current = []
            words = 0

    if current:
        chunks.append("\n\n".join(current))
    return chunks


class ScenePlanner:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def plan(self, story: StorySpec, script: str, minutes: int = 30) -> ScenePlan:
        # First create natural chunks. The LLM then enriches them with visual metadata.
        chunks = split_script(script)
        chunked_script = "\n\n--- СЦЕНА-КАНДИДАТ ---\n\n".join(chunks)

        prompt = SCENE_PROMPT.format(
            story_json=json.dumps(story.model_dump(), ensure_ascii=False, indent=2),
            script=chunked_script,
        )
        raw = self.client.generate(prompt)
        data = self._parse_json(raw)

        scenes = [SceneSpec.model_validate(item) for item in data.get("scenes", [])]
        if not scenes:
            raise ValueError("Модель не повернула жодної сцени")

        # Keep source narration authoritative. This prevents the LLM from silently
        # changing the script that will later be sent to TTS.
        if len(scenes) != len(chunks):
            raise ValueError(
                f"Кількість сцен ({len(scenes)}) не збігається з кількістю фрагментів ({len(chunks)})."
            )

        for index, (scene, source) in enumerate(zip(scenes, chunks), start=1):
            scene.number = index
            scene.narration = source
            scene.estimated_duration_seconds = max(1, round(len(source.split()) / 2.3))

        total = sum(scene.estimated_duration_seconds for scene in scenes)
        return ScenePlan(
            target_duration_seconds=minutes * 60,
            total_duration_seconds=total,
            scenes=scenes,
        )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise ValueError("Ollama повернув невалідний JSON для режисерського плану") from exc
