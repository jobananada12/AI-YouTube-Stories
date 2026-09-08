from core.llm import OllamaClient
from core.prompts import STORY_SYSTEM_PROMPT
from core.story_schema import StorySpec


SCRIPT_PROMPT = """
Write a complete original Ukrainian narration script based on the story plan below.

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

TARGET DURATION: about {minutes} minutes.

Requirements:
- Narration only: no screenplay camera directions and no dialogue labels.
- Natural Ukrainian suitable for a professional narrator.
- Strong opening hook within the first minute.
- Expand every outline beat into meaningful narrative.
- Keep cause-and-effect clear and characters consistent.
- Build tension progressively toward a real climax and an earned ending.
- Avoid padding, repeated sentences, empty descriptions and copied phrases.
- Do not imitate any existing author, film, game, book or channel.
- The script must be newly written from this plan.
""".strip()


class ScriptWriter:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def write(self, story: StorySpec, minutes: int = 30) -> str:
        characters = "\n".join(
            f"- {c.name}: {c.role}; {c.age}; {c.appearance}; {c.personality}; мотивація: {c.motivation}"
            for c in story.characters
        )
        outline = "\n".join(f"{i + 1}. {beat}" for i, beat in enumerate(story.outline))
        prompt = SCRIPT_PROMPT.format(
            title=story.title,
            genre=story.genre,
            tone=story.tone,
            premise=story.premise,
            setting=story.setting,
            goal=story.protagonist_goal,
            conflict=story.central_conflict,
            characters=characters,
            outline=outline,
            minutes=minutes,
        )
        return self.client.generate(prompt, system=STORY_SYSTEM_PROMPT)
