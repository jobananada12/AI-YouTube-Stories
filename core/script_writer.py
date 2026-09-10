"""Long-form Ukrainian narration writer.

A 30-minute script is too large for one Ollama response, so generation is
split into sequential chunks. The story map keeps the chunks on one causal
track and the previous tail gives continuity between chunks.
"""

from __future__ import annotations

from core.llm import OllamaClient
from core.prompts import STORY_SYSTEM_PROMPT
from core.story_schema import StorySpec


SCRIPT_MAP_PROMPT = """
Create a detailed causal story map in Ukrainian for the story below.
This is planning material, NOT the final narration.
Create approximately {blocks} sequential blocks. For every block state:
1) what happens, 2) who is present, 3) important place/object, 4) what changes afterward.
Do not invent unnecessary characters. Keep the supplied story facts consistent.

TITLE: {title}
GENRE: {genre}
TONE: {tone}
PREMISE: {premise}
SETTING: {setting}
PROTAGONIST GOAL: {goal}
CENTRAL CONFLICT: {conflict}
CHARACTERS:
{characters}
OUTLINE:
{outline}
""".strip()


SCRIPT_CHUNK_PROMPT = """
Write part {part} of {total_parts} of one original Ukrainian narration.
Target about {words} words for this part. The complete narration should be about
{total_words} words (approximately {minutes} minutes at normal narration speed).

Use ONLY the story map and supplied story facts. Do not create an alternative plot,
new characters, new relationships, unexplained objects, or events outside the map.
Continue directly from the previous part. Keep cause-and-effect and character details
consistent. Do not summarize. Do not use headings, lists, scene labels, camera notes,
TTS instructions, or meta-comments. Write only natural Ukrainian narration.

STORY MAP:
{story_map}

PREVIOUS PART TAIL:
{previous_tail}

Write part {part} now.
""".strip()


class ScriptWriter:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    @staticmethod
    def _target_words(minutes: int) -> int:
        # Comfortable Ukrainian narration pace; timing is ultimately measured by TTS.
        return max(1000, int(minutes * 130))

    def write(self, story: StorySpec, minutes: int = 30) -> str:
        characters = "\n".join(
            f"- {c.name}: {c.role}; {c.age}; {c.appearance}; {c.personality}; мотивація: {c.motivation}"
            for c in story.characters
        )
        outline = "\n".join(f"{i + 1}. {beat}" for i, beat in enumerate(story.outline))
        total_words = self._target_words(minutes)
        total_parts = max(6, (total_words + 1799) // 1800)
        words_per_part = max(900, total_words // total_parts)

        map_prompt = SCRIPT_MAP_PROMPT.format(
            blocks=max(18, minutes // 2),
            title=story.title,
            genre=story.genre,
            tone=story.tone,
            premise=story.premise,
            setting=story.setting,
            goal=story.protagonist_goal,
            conflict=story.central_conflict,
            characters=characters,
            outline=outline,
        )
        story_map = self.client.generate(map_prompt, system=STORY_SYSTEM_PROMPT).strip()

        parts: list[str] = []
        for part_number in range(1, total_parts + 1):
            previous_tail = parts[-1][-1600:] if parts else "(Початок історії.)"
            prompt = SCRIPT_CHUNK_PROMPT.format(
                part=part_number,
                total_parts=total_parts,
                words=words_per_part,
                total_words=total_words,
                minutes=minutes,
                story_map=story_map,
                previous_tail=previous_tail,
            )
            text = self.client.generate(prompt, system=STORY_SYSTEM_PROMPT).strip()
            if text:
                parts.append(text)

        return "\n\n".join(parts).strip()
