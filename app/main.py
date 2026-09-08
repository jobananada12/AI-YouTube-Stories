import argparse
import json
from datetime import datetime

from app.config import settings
from core.character_bible import CharacterBibleGenerator
from core.final_package import FinalPackageError, FinalProjectPackager
from core.project import StoryProject
from core.scene_planner import ScenePlanner
from core.script_writer import ScriptWriter
from core.story_generator import StoryGenerator
from core.thumbnail_generator import ThumbnailGenerator
from core.tts_generator import NarrationGenerator
from core.video_renderer import VideoRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="AI YouTube Stories")
    parser.add_argument("topic", help="Seed or idea for the original story")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--project", default=None, help="Project ID; generated automatically when omitted")
    parser.add_argument("--no-tts", action="store_true", help="Skip narration generation")
    parser.add_argument("--no-render", action="store_true", help="Skip final MP4 rendering")
    parser.add_argument("--package", action="store_true", help="Create a ZIP package when the project is complete")
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
    project = StoryProject().create(story, script, project_id, scene_plan, character_bible.model_dump())

    if not args.no_tts:
        print("\n🎙 Створюю українську озвучку через FilmDubUA/Piper...")
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
        print(f"Озвучено сцен: {len(narration.segments)}")
        print(f"Тривалість озвучки: {narration.total_duration_seconds / 60:.1f} хв")
    else:
        print("\n⏭ Озвучку пропущено (--no-tts)")

    first_image = project / "images" / "scene_001.png"
    if first_image.exists():
        print("\n🖼 Створюю thumbnail...")
        thumbnail = ThumbnailGenerator().generate(story.title, first_image, project / "thumbnail")
        print(f"Thumbnail: {thumbnail}")
    else:
        print("\n⏭ Thumbnail пропущено: ще немає scene_001.png")

    if not args.no_render:
        if args.no_tts:
            print("\n⛔ Рендер пропущено: для фінального відео потрібна озвучка.")
        else:
            print("\n🎬 Збираю фінальне MP4 через FFmpeg...")
            music_files = sorted(project.joinpath("music").glob("*.wav"))
            output = VideoRenderer(
                ffmpeg_bin=settings.ffmpeg_bin,
                fps=settings.output_fps,
                width=settings.video_width,
                height=settings.video_height,
            ).render(
                scene_plan=scene_plan,
                project_dir=project,
                music_file=music_files[0] if music_files else None,
            )
            print(f"Фінальне відео: {output}")
    else:
        print("\n⏭ Рендер пропущено (--no-render)")

    packager = FinalProjectPackager()
    require_video = not args.no_render and not args.no_tts
    report = packager.validate(project, require_video=require_video)
    manifest_path = packager.write_manifest(project, report)
    print(f"\n📦 Маніфест проєкту: {manifest_path}")

    if args.package:
        try:
            archive = packager.package(project, require_video=require_video)
        except FinalPackageError as exc:
            print(f"\n⛔ ZIP не створено: {exc}")
        else:
            print(f"ZIP-пакет: {archive}")
    elif report["valid"]:
        print("Проєкт пройшов перевірку. Для ZIP додай --package.")
    else:
        print("Проєкт ще не повністю готовий:")
        for item in report["missing"]:
            print(f"  - {item}")

    print(f"\nГотово. Проєкт збережено: {project}")


if __name__ == "__main__":
    main()
