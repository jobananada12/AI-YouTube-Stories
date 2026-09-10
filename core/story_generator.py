import json

from core.llm import OllamaClient
from core.prompts import STORY_SYSTEM_PROMPT
from core.story_schema import StorySpec


GENERATOR_PROMPT = """
Create a completely original fictional story concept for a Ukrainian YouTube
narration channel.

SEED:
{topic}

TARGET LENGTH: approximately {minutes} minutes.

Return ONLY valid JSON matching this exact structure:
{{
  "title": "...",
  "genre": "...",
  "tone": "...",
  "logline": "...",
  "premise": "...",
  "theme": "...",
  "setting": "...",
  "protagonist_goal": "...",
  "central_conflict": "...",
  "ending_type": "...",
  "characters": [
    {{"name":"...","role":"...","age":"...","appearance":"...","personality":"...","motivation":"..."}}
  ],
  "outline": ["...", "..."]
}}

Rules:
- Everything must be newly invented.
- Do not use existing copyrighted characters, plots, dialogue or settings as substitutes for invention.
- The outline must contain enough distinct beats for a long-form story.
- Characters must have different motivations and personalities.
- Avoid generic filler and repetitive plot beats.
- Write all values in Ukrainian.
""".strip()


class StoryGenerator:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def generate(self, topic: str, minutes: int = 30) -> StorySpec:
        raw = self.client.generate(
            GENERATOR_PROMPT.format(topic=topic, minutes=minutes),
            system=STORY_SYSTEM_PROMPT,
        )
        data = json.loads(raw)
        return StorySpec.model_validate(data)
