import json
import re

from core.character_bible_schema import CharacterBible
from core.llm import OllamaClient
from core.story_schema import StorySpec


CHARACTER_BIBLE_PROMPT = '''
You are the character continuity designer for an original YouTube story.
Create a Character Bible for ALL characters in the supplied story.

The goal is stable image generation: the same character must keep the same face,
body, hair, age, clothing and distinctive features in every future Stable Diffusion shot.

RULES:
- Do not copy real people or famous fictional characters.
- Do not mention actors, films, games, books, franchises or living artists.
- Never invent a character that is not present in the story.
- Keep each supplied character name and role unchanged.
- Fix concrete visual traits: age, presentation, height, build, skin, face, eyes,
  eyebrows, nose, lips, hair, facial hair, clothing, footwear and accessories.
- image_prompt_anchor MUST be written in clean English and be directly usable in a
  Stable Diffusion prompt. It must describe appearance only, not events or actions.
- All other text values may be Ukrainian; image_prompt_anchor must be English.
- Return ONLY valid JSON.

FORMAT:
{"characters":[{"name":"...","role":"...","age":"...","gender_presentation":"...","height":"...","build":"...","skin":"...","face":"...","eyes":"...","eyebrows":"...","nose":"...","lips":"...","hair":"...","facial_hair":"...","signature_clothing":"...","footwear":"...","accessories":"...","distinctive_features":"...","typical_expression":"...","color_palette":["..."],"image_prompt_anchor":"..."}]}

STORY:
{story_json}
'''


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\\s*', '', text)
        text = re.sub(r'\\s*```$', '', text)
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
