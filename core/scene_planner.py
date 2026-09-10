import json
import re

from core.character_bible_schema import CharacterBible
from core.llm import OllamaClient
from core.scene_schema import ScenePlan, SceneSpec
from core.story_schema import StorySpec


SCENE_PROMPT = """
You are a strict storyboard planner for an original YouTube narration.
Convert the supplied narration chunks into visual scene metadata.

NON-NEGOTIABLE RULES:
- The narration supplied for each candidate is authoritative. Never rewrite it.
- Do not invent events, objects, characters, relationships, actions, locations or
  emotions that are not supported by that narration and the story facts.
- Use only characters explicitly present in the candidate narration. An absent
  character MUST NOT be added just because they exist in the Character Bible.
- visual_prompt MUST be clean English only. Never use Cyrillic, Ukrainian or Russian.
- visual_prompt must describe only what is actually visible in this narration chunk.
- Do not add cinematic events, props or background people merely to make the image
  more interesting.
- For a character that is present, copy the exact English image_prompt_anchor from
  the Character Bible and then add only the visible action, setting, lighting and
  composition supported by the narration.
- Do not mention real people, actors, franchises, copyrighted characters or living artists.
- Preserve the candidate order exactly. Return exactly one scene for every candidate.
- Keep titles, purpose, mood and continuity notes concise.
- estimated_duration_seconds is only an estimate; later TTS timing is authoritative.
- Return ONLY valid JSON. No markdown and no explanation.

JSON FORMAT:
{"scenes":[{"number":1,"title":"...","purpose":"...","narration":"exact candidate text","estimated_duration_seconds":35,"characters":["..."],"location":"...","time_of_day":"...","action":"...","visual_prompt":"clean English prompt","mood":"...","continuity_notes":"...","transition":"cut"}]}

STORY FACTS:
{story_json}

CHARACTER BIBLE:
{bible_json}

NARRATION CANDIDATES:
{script}
"""


def split_script(script: str, target_words: int = 85) -> list[str]:
    """Split narration into natural chunks without rewriting source text."""
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

    def plan(
        self,
        story: StorySpec,
        script: str,
        minutes: int = 30,
        character_bible: CharacterBible | None = None,
    ) -> ScenePlan:
        chunks = split_script(script)
        chunked_script = "\n\n--- SCENE CANDIDATE ---\n\n".join(chunks)

        if character_bible is not None:
            bible_json = json.dumps(character_bible.model_dump(), ensure_ascii=False, indent=2)
        else:
            bible_json = '{"characters":[]}'

        prompt = SCENE_PROMPT.format(
            story_json=json.dumps(story.model_dump(), ensure_ascii=False, indent=2),
            bible_json=bible_json,
            script=chunked_script,
        )
        raw = self.client.generate(prompt)
        data = self._parse_json(raw)
        scenes = [SceneSpec.model_validate(item) for item in data.get("scenes", [])]
        if not scenes:
            raise ValueError("Модель не повернула жодної сцени")
        if len(scenes) != len(chunks):
            raise ValueError(
                f"Кількість сцен ({len(scenes)}) не збігається з кількістю фрагментів ({len(chunks)})."
            )

        for index, (scene, source) in enumerate(zip(scenes, chunks), start=1):
            scene.number = index
            scene.narration = source
            scene.estimated_duration_seconds = max(1, round(len(source.split()) / 2.3))
            # A final defensive check: generated SD prompts must never contain Cyrillic.
            if re.search(r"[\u0400-\u04FF]", scene.visual_prompt or ""):
                raise ValueError(f"Сцена {index}: visual_prompt містить кирилицю")

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
