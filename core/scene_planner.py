import json
import re

from core.character_bible_schema import CharacterBible
from core.llm import OllamaClient
from core.scene_schema import ScenePlan, SceneSpec
from core.story_schema import StorySpec


SCENE_PROMPT = """
You are a strict storyboard planner for an original YouTube narration.
Create EXACTLY 100 visual scenes for the complete supplied narration.

NON-NEGOTIABLE RULES:
- Exactly 100 scenes. Never return fewer or more.
- Cover the entire narration from beginning to end, in order.
- Split the narration into 100 contiguous source excerpts. Do not rewrite them.
- The narration is authoritative. Never invent events, objects, characters, relationships,
  actions, locations or emotions that are not supported by the narration and story facts.
- Use only characters explicitly present in the candidate narration. An absent character
  MUST NOT be added just because they exist in the Character Bible.
- Do not add the cat unless the candidate narration explicitly contains the cat.
- Normalize Irene/Irena to Irina when applicable; never invent a family relation.
- visual_prompt MUST be clean English only. Never use Cyrillic, Ukrainian or Russian.
- visual_prompt must describe only what is actually visible in that narration chunk.
- Do not add cinematic events, props or background people merely to make the image interesting.
- For every present character, copy the exact English image_prompt_anchor from the Character
  Bible and then add only the visible action, setting, lighting and composition supported by
  the narration.
- No captions, subtitles, logos, watermarks or written words unless explicitly required by
  the narration.
- Preserve candidate order exactly.
- estimated_duration_seconds is only an estimate; later TTS timing is authoritative.
- Return ONLY valid JSON. No markdown and no explanation.

JSON FORMAT:
{"scenes":[{"number":1,"title":"...","purpose":"...","narration":"exact source excerpt","estimated_duration_seconds":18,"characters":["..."],"location":"...","time_of_day":"...","action":"...","visual_prompt":"clean English prompt","mood":"...","continuity_notes":"...","transition":"cut"}]}

STORY FACTS:
{story_json}

CHARACTER BIBLE:
{bible_json}

FULL NARRATION:
{script}
"""


def split_script(script: str, target_scenes: int = 100) -> list[str]:
    """Split the source narration into exactly 100 natural, contiguous chunks."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", script.strip()) if s.strip()]
    if len(sentences) < target_scenes:
        words = script.split()
        if not words:
            return []
        size = max(1, len(words) // target_scenes)
        chunks = []
        for i in range(target_scenes):
            start = round(i * len(words) / target_scenes)
            end = round((i + 1) * len(words) / target_scenes)
            if end <= start:
                end = start + 1
            chunks.append(" ".join(words[start:end]))
        return chunks

    chunks: list[str] = []
    for i in range(target_scenes):
        start = round(i * len(sentences) / target_scenes)
        end = round((i + 1) * len(sentences) / target_scenes)
        if end <= start:
            end = start + 1
        chunks.append(" ".join(sentences[start:end]))
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
        chunks = split_script(script, 100)
        if len(chunks) != 100:
            raise ValueError(f"Не вдалося підготувати рівно 100 сцен: отримано {len(chunks)}")

        chunked_script = "\n\n--- SCENE CANDIDATE ---\n\n".join(chunks)
        bible_json = (
            json.dumps(character_bible.model_dump(), ensure_ascii=False, indent=2)
            if character_bible is not None
            else '{"characters":[]}'
        )
        prompt = SCENE_PROMPT.format(
            story_json=json.dumps(story.model_dump(), ensure_ascii=False, indent=2),
            bible_json=bible_json,
            script=chunked_script,
        )
        raw = self.client.generate(prompt)
        data = self._parse_json(raw)
        scenes = [SceneSpec.model_validate(item) for item in data.get("scenes", [])]
        if len(scenes) != 100:
            raise ValueError(f"Модель повернула {len(scenes)} сцен замість 100")

        for index, (scene, source) in enumerate(zip(scenes, chunks), start=1):
            scene.number = index
            scene.narration = source
            scene.estimated_duration_seconds = max(1, round(len(source.split()) / 2.3))
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
