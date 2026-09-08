import json
from pathlib import Path

from core.scene_schema import ScenePlan
from core.story_schema import StorySpec


class StoryProject:
    def __init__(self, root: str | Path = "projects"):
        self.root = Path(root)

    def create(
        self,
        story: StorySpec,
        script: str,
        project_id: str,
        scene_plan: ScenePlan | None = None,
        character_bible: dict | None = None,
    ) -> Path:
        project = self.root / project_id
        for folder in ("audio", "images", "music", "sfx", "thumbnail", "youtube", "final"):
            (project / folder).mkdir(parents=True, exist_ok=True)

        (project / "story.json").write_text(
            json.dumps(story.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (project / "script.md").write_text(script, encoding="utf-8")

        if character_bible is not None:
            (project / "character_bible.json").write_text(
                json.dumps(character_bible, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if scene_plan is not None:
            (project / "scenes.json").write_text(
                json.dumps(scene_plan.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return project
