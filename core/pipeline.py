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
from core.thumbnail_generator import ThumbnailGenerator
from core.tts_generator import NarrationGenerator
from core.video_renderer import VideoRenderer


class StoryPipeline:
    def run(self, topic: str, minutes: int = 30, project_id: str | None = None, render: bool = True) -> Path:
        if not 10 <= minutes <= 120:
            raise ValueError("minutes must be between 10 and 120")

        story = StoryGenerator().generate(topic, minutes)
        character_bible = CharacterBibleGenerator().generate(story)
        script = ScriptWriter().write(story, minutes)
        scene_plan = ScenePlanner().plan(story, script, minutes)

        project_id = project_id or datetime.now().strftime("story_%Y%m%d_%H%M%S")
        project = StoryProject().create(story, script, project_id, scene_plan, character_bible.model_dump())

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

        # Use the first generated scene as the thumbnail source when available.
        first_image = project / "images" / "scene_001.png"
        if first_image.exists():
            ThumbnailGenerator().generate(story.title, first_image, project / "thumbnail")

        if render:
            music_files = sorted((project / "music").glob("*.wav")) if (project / "music").exists() else []
            VideoRenderer(
                ffmpeg_bin=settings.ffmpeg_bin,
                fps=settings.output_fps,
                width=settings.video_width,
                height=settings.video_height,
            ).render(scene_plan, project, music_file=music_files[0] if music_files else None)

        report = FinalProjectPackager().validate(project, require_video=render)
        FinalProjectPackager().write_manifest(project, report)
        return project
