import argparse
import json
from datetime import datetime

from app.config import settings
from core.character_bible import CharacterBibleGenerator
from core.project import StoryProject
from core.scene_planner import ScenePlanner
from core.script_writer import ScriptWriter
from core.story_generator import StoryGenerator
from core.tts_generator import NarrationGenerator
from core.video_renderer import VideoRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="AI YouTube Stories")
    parser.add_argument("topic", help="Seed or idea for the original story")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--project", default=None, help="Project ID; generated automatically when omitted")
    parser.add_argument("--no-tts", action="store_true", help="Skip narration generation")
    parser.add_argument("--no-render", action="store_true", help="Skip final MP4 rendering")
    args = parser.parse_args()

    if args.minutes < 10 or args.minutes > 120:
        parser.error("--minutes must be between 10 and 120")

    story = StoryGenerator().generate(args.topic, args.minutes)
    print(f"\nІСТОРІЯ: {story.title}")
    print(f"ЖАНР: {story.genre}")
    print(f"ЛОГЛАЙН: {story.logline}\n")

    print("Створюю Character Bible...")
    character_bible = CharacterBibleGenerator().generate(story)
    print(f"Персонажів: {len(character_bible.characters)}")

    script = ScriptWriter().write(story, args.minutes)
    print("Створюю режисерський план сцен...")
    scene_plan = ScenePlanner().plan(story, script, args.minutes)
    print(f"Сцен: {len(scene_plan.scenes)}")
    print(f"Орієнтовна тривалість: {scene_plan.total_duration_seconds // 60} хв")

    project_id = args.project or datetime.now().strftime("story_%Y%m%d_%H%M%S")
    project = StoryProject().create(
        story,
        script,
        project_id,
        scene_plan,
        character_bible.model_dump(),
    )

    if not args.no_tts:
        print("\n🎙 Створюю українську озвучку через FilmDubUA/Piper...")
        manifest = NarrationGenerator().generate(
            scene_plan=scene_plan,
            output_dir=project / "audio",
            voice_profile=settings.filmdubua_voice_profile,
            rate=settings.tts_rate,
            volume=settings.tts_volume,
        )
        (project / "narration.json").write_text(
            json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Озвучено сцен: {len(manifest.segments)}")
        print(f"Тривалість озвучки: {manifest.total_duration_seconds / 60:.1f} хв")
    else:
        print("\n⏭ Озвучку пропущено (--no-tts)")

    if not args.no_render:
        if args.no_tts:
            print("\n⛔ Рендер пропущено: для фінального відео потрібна озвучка.")
        else:
            print("\n🎬 Збираю фінальне MP4 через FFmpeg...")
            output = VideoRenderer(
                ffmpeg_bin=settings.ffmpeg_bin,
                fps=settings.output_fps,
                width=settings.video_width,
                height=settings.video_height,
            ).render(
                scene_plan=scene_plan,
                project_dir=project,
                music_file=next(project.joinpath("music").glob("*.wav"), None),
            )
            print(f"Фінальне відео: {output}")
    else:
        print("\n⏭ Рендер пропущено (--no-render)")

    print(f"\nГотово. Проєкт збережено: {project}")
    print(f"Сценарій: {project / 'script.md'}")
    print(f"Character Bible: {project / 'character_bible.json'}")
    print(f"Режисерський план: {project / 'scenes.json'}")
    if not args.no_tts:
        print(f"Озвучка: {project / 'audio'}")
        print(f"Маніфест озвучки: {project / 'narration.json'}")
    if not args.no_render and not args.no_tts:
        print(f"Фінальне відео: {project / 'final' / 'story.mp4'}")


if __name__ == "__main__":
    main()
