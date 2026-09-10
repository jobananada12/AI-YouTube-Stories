from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.config import settings
from core.character_bible import CharacterBibleGenerator
from core.final_package import FinalProjectPackager
from core.project import StoryProject
from core.scene_planner import ScenePlanner
from core.script_writer import ScriptWriter
from core.story_generator import StoryGenerator
from core.tts_generator import NarrationGenerator
from core.video_renderer import VideoRenderer


class StoryPipeline:
    """Generate story/script/100 prompts/audio and optionally render with external images.

    Local Stable Diffusion is intentionally not used here. Images are supplied separately
    by Rich Gen Image Tool and imported by app.main --images-dir.
    """

    def run(
        self,
        topic: str,
        minutes: int = 30,
        project_id: str | None = None,
        render: bool = True,
        images_dir: str | Path | None = None,
    ) -> Path:
        if not 10 <= minutes <= 120:
            raise ValueError("minutes must be between 10 and 120")

        print("1/6 Створюю сюжет...")
        story = StoryGenerator().generate(topic, minutes)

        print("2/6 Фіксую Character Bible...")
        character_bible = CharacterBibleGenerator().generate(story)

        print("3/6 Пишу повний сценарій...")
        script = ScriptWriter().write(story, minutes)

        print("4/6 Створюю рівно 100 сцен та English image prompts для Rich Gen...")
        scene_plan = ScenePlanner().plan(story, script, minutes, character_bible)
        if len(scene_plan.scenes) != 100:
            raise RuntimeError(f"Потрібно рівно 100 сцен, отримано {len(scene_plan.scenes)}")

        project_id = project_id or datetime.now().strftime("story_%Y%m%d_%H%M%S")
        project = StoryProject().create(story, script, project_id, scene_plan, character_bible.model_dump())

        print("5/6 Генерую українську озвучку...")
        narration = NarrationGenerator().generate(
            scene_plan=scene_plan,
            output_dir=project / "audio",
            voice_profile=settings.filmdubua_voice_profile,
            rate=settings.tts_rate,
            volume=settings.tts_volume,
        )
        (project / "narration.json").write_text(
            json.dumps(narration.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if images_dir:
            from app.main import import_external_images
            print("6/6 Імпортую 100 готових зображень Rich Gen Image Tool...")
            import_external_images(images_dir, project, expected_count=100)
        else:
            print("6/6 Зображення не генерую: їх потрібно зробити в Rich Gen Image Tool")

        if render:
            if not images_dir:
                print("Рендер пропущено: немає 100 зовнішніх зображень.")
            else:
                print("Збираю фінальне відео...")
                VideoRenderer(
                    ffmpeg_bin=settings.ffmpeg_bin,
                    fps=settings.output_fps,
                    width=settings.video_width,
                    height=settings.video_height,
                ).render(scene_plan, project)

        report = FinalProjectPackager().validate(project, require_video=render and bool(images_dir))
        FinalProjectPackager().write_manifest(project, report)
        return project
