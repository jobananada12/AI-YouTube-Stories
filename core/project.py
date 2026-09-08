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
    ) -> Path:
        project = self.root / project_id
        (project / "audio").mkdir(parents=True, exist_ok=True)
        (project / "images").mkdir(parents=True, exist_ok=True)
        (project / "music").mkdir(parents=True, exist_ok=True)
        (project / "sfx").mkdir(parents=True, exist_ok=True)
        (project / "thumbnail").mkdir(parents=True, exist_ok=True)
        (project / "youtube").mkdir(parents=True, exist_ok=True)
        (project / "final").mkdir(parents=True, exist_ok=True)

        (project / "story.json").write_text(
            json.dumps(story.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (project / "script.md").write_text(script, encoding="utf-8")

        if scene_plan is not None:
            (project / "scenes.json").write_text(
                json.dumps(scene_plan.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return project
