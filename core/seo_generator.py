from __future__ import annotations

import json
import re
from pathlib import Path

from core.llm import OllamaClient
from core.seo_schema import YouTubeMetadata
from core.story_schema import StorySpec


PROMPT = '''Створи SEO-метадані для оригінальної української YouTube-історії.
Не вигадуй факти, яких немає в історії. Не використовуй назви відомих творів,
персонажів, каналів або брендів як ключові слова. Заголовок має бути природним,
цікавим і без клікбейту, до 100 символів. Опис — 2-4 абзаци з коротким вступом,
сюжетом без спойлерів і релевантними пошуковими фразами. Keywords і tags мають
бути релевантними темі. Hashtags починай символом #.
Відповідь ТІЛЬКИ JSON.

ІСТОРІЯ:
{story}
'''


def _json(raw: str) -> dict:
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find('{'), text.rfind('}')
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError('Ollama повернув невалідний SEO JSON')


class YouTubeMetadataGenerator:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def generate(self, story: StorySpec, output_dir: str | Path) -> YouTubeMetadata:
        metadata = YouTubeMetadata.model_validate(_json(self.client.generate(
            PROMPT.format(story=json.dumps(story.model_dump(), ensure_ascii=False, indent=2))
        )))
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'youtube.json').write_text(
            json.dumps(metadata.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8'
        )
        (output / 'youtube.md').write_text(
            f'# {metadata.title}\n\n{metadata.description}\n\n'
            f'**Keywords:** {", ".join(metadata.keywords)}\n\n'
            f'**Tags:** {", ".join(metadata.tags)}\n\n'
            f'{" ".join(metadata.hashtags)}\n', encoding='utf-8'
        )
        return metadata
