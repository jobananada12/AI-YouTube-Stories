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

    def load(self, project_id: str) -> Path:
        project = self.root / project_id
        required = ("story.json", "script.md", "scenes.json", "character_bible.json")
        missing = [name for name in required if not (project / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Проєкт {project_id} неповний. Відсутні: {', '.join(missing)}"
            )
        return project

    def load_story(self, project_id: str) -> StorySpec:
        project = self.load(project_id)
        data = json.loads((project / "story.json").read_text(encoding="utf-8"))
        return StorySpec.model_validate(data)

    def load_scene_plan(self, project_id: str) -> ScenePlan:
        project = self.load(project_id)
        data = json.loads((project / "scenes.json").read_text(encoding="utf-8"))
        plan = ScenePlan.model_validate(data)
        if len(plan.scenes) != 100:
            raise ValueError(f"У проєкті має бути рівно 100 сцен, знайдено {len(plan.scenes)}")
        return plan
