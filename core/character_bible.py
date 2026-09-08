import json
import re

from core.llm import OllamaClient
from core.story_schema import StorySpec


CHARACTER_BIBLE_PROMPT = """
Ти — character designer для оригінального YouTube-серіалу.
Створи CHARACTER BIBLE на основі вже затверджених персонажів історії.

Мета: кожен персонаж повинен мати стабільний вигляд у всіх майбутніх AI-зображеннях.
Не змінюй імена та роль персонажів. Не додавай нових персонажів.
Не копіюй зовнішність реальних людей або персонажів відомих франшиз.
Не використовуй імена акторів, художників або конкретні існуючі образи як референс.
Усі значення пиши українською.

Для кожного персонажа створи:
- name
- identity: коротка ідентичність і роль
- age
- gender_presentation
- height_build
- face: форма обличчя, очі, брови, ніс, губи, особливі риси
- hair: колір, довжина, зачіска
- skin
- clothing: базовий стабільний одяг
- accessories
- palette: 3–5 описових кольорів одягу/аксесуарів
- personality_visual_cues: як характер проявляється в позі та міміці
- signature_details: 2–4 деталі, які не повинні змінюватися
- image_prompt_anchor: один короткий стабільний опис, який можна додавати до кожного visual_prompt

Відповідь ТІЛЬКИ валідний JSON:
{
  "characters": [
    {
      "name": "...",
      "identity": "...",
      "age": "...",
      "gender_presentation": "...",
      "height_build": "...",
      "face": "...",
      "hair": "...",
      "skin": "...",
      "clothing": "...",
      "accessories": "...",
      "palette": ["..."],
      "personality_visual_cues": "...",
      "signature_details": ["..."],
      "image_prompt_anchor": "..."
    }
  ]
}

ПЕРСОНАЖІ ІСТОРІЇ:
{characters_json}
"""


class CharacterBible:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def generate(self, story: StorySpec) -> dict:
        source = [character.model_dump() for character in story.characters]
        raw = self.client.generate(
            CHARACTER_BIBLE_PROMPT.format(
                characters_json=json.dumps(source, ensure_ascii=False, indent=2)
            )
        )
        data = self._parse_json(raw)
        characters = data.get("characters", [])

        expected = {item.name for item in story.characters}
        actual = {item.get("name") for item in characters}
        if expected != actual:
            missing = expected - actual
            extra = actual - expected
            raise ValueError(
                f"Character Bible не збігається з історією. Відсутні: {sorted(missing)}; зайві: {sorted(extra)}"
            )
        return data

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
            raise ValueError("Ollama повернув невалідний JSON для Character Bible") from exc
