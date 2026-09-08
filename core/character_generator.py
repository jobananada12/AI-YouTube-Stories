import json
import re

from core.character_bible_schema import CharacterBible
from core.llm import OllamaClient
from core.story_schema import StorySpec


CHARACTER_BIBLE_PROMPT = '''
Ти — художник персонажів для оригінального YouTube-проєкту.
Створи Character Bible для ВСІХ персонажів наданої історії.

Мета: один і той самий персонаж повинен мати максимально стабільну зовнішність
на всіх майбутніх AI-зображеннях.

ПРАВИЛА:
- Не копіюй зовнішність реальних людей або відомих персонажів.
- Не посилайся на акторів, фільми, ігри, книги, франшизи чи живих художників.
- Не змінюй ім'я, роль або базову зовнішність персонажа.
- Конкретно зафіксуй волосся, очі, обличчя, статуру, вік та одяг.
- Вкажи унікальні прикмети та постійний одяг.
- image_prompt_anchor — детальний стабільний опис для кожного майбутнього кадру.
- Усі текстові значення українською.
- Відповідь — ТІЛЬКИ валідний JSON.

Формат:
{"characters":[{"name":"...","role":"...","age":"...","gender_presentation":"...","height":"...","build":"...","skin":"...","face":"...","eyes":"...","eyebrows":"...","nose":"...","lips":"...","hair":"...","facial_hair":"...","signature_clothing":"...","footwear":"...","accessories":"...","distinctive_features":"...","typical_expression":"...","color_palette":["..."],"image_prompt_anchor":"..."}]}

ІСТОРІЯ:
{story_json}
'''


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find('{'), text.rfind('}')
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError('Ollama повернув невалідний JSON для Character Bible') from exc


class CharacterBibleGenerator:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def generate(self, story: StorySpec) -> CharacterBible:
        prompt = CHARACTER_BIBLE_PROMPT.format(
            story_json=json.dumps(story.model_dump(), ensure_ascii=False, indent=2)
        )
        bible = CharacterBible.model_validate(_parse_json(self.client.generate(prompt)))
        expected = {c.name for c in story.characters}
        actual = {c.name for c in bible.characters}
        missing = expected - actual
        if missing:
            raise ValueError('Character Bible не містить персонажів: ' + ', '.join(sorted(missing)))
        return bible
'''
