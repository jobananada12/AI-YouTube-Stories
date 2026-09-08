from pathlib import Path

from core.scene_schema import ScenePlan


def validate_scene_assets(scene_plan: ScenePlan, project_dir: str | Path) -> None:
    project_dir = Path(project_dir)
    missing = []
    for scene in scene_plan.scenes:
        image = project_dir / "images" / f"scene_{scene.number:03d}.png"
        audio = project_dir / "audio" / f"narration_{scene.number:03d}.wav"
        if not image.exists():
            missing.append(str(image))
        if not audio.exists():
            missing.append(str(audio))
    if missing:
        raise FileNotFoundError("Відсутні медіафайли:\n" + "\n".join(missing))
