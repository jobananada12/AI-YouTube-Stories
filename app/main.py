import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from app.config import settings
from core.final_package import FinalPackageError, FinalProjectPackager
from core.music_generator import ProceduralMusicGenerator
from core.project import StoryProject
from core.scene_planner import ScenePlanner
from core.script_writer import ScriptWriter
from core.seo_generator import YouTubeMetadataGenerator
from core.sfx_generator import ProceduralSFXGenerator
from core.story_generator import StoryGenerator
from core.thumbnail_generator import ThumbnailGenerator
from core.tts_generator import NarrationGenerator
from core.video_renderer import VideoRenderer


def import_external_images(source_dir: str | Path, project_dir: Path, expected_count: int) -> None:
    """Import Rich Gen Image Tool output as scene_001.* ... scene_NNN.*."""
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Папку із зображеннями не знайдено: {source}")

    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for old in images_dir.glob("scene_*.*"):
        old.unlink()

    candidates = sorted(
        [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}],
        key=lambda p: p.name.lower(),
    )
    if len(candidates) != expected_count:
        raise ValueError(
            f"Очікується рівно {expected_count} зображень, але знайдено {len(candidates)} у {source}"
        )

    for index, image in enumerate(candidates, start=1):
        destination = images_dir / f"scene_{index:03d}{image.suffix.lower()}"
        shutil.copy2(image, destination)

    manifest = {
        "provider": "Rich Gen Image Tool / external",
        "source": str(source),
        "count": len(candidates),
        "files": [f"scene_{i:03d}{candidates[i - 1].suffix.lower()}" for i in range(1, expected_count + 1)],
    }
    (project_dir / "external_images.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Імпортовано зовнішніх зображень: {len(candidates)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI YouTube Stories — story + 100 prompts + Piper + Rich Gen + FFmpeg"
    )
    parser.add_argument("topic", help="Seed or idea for the original story")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--project", default=None)
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Folder containing exactly 100 images made by Rich Gen Image Tool",
    )
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--no-music", action="store_true")
    parser.add_argument("--no-sfx", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()

    if not 10 <= args.minutes <= 120:
        parser.error("--minutes must be between 10 and 120")

    story = StoryGenerator().generate(args.topic, args.minutes)
    print(f"\nІСТОРІЯ: {story.title}\nЖАНР: {story.genre}\nЛОГЛАЙН: {story.logline}\n")

    print("Створюю Character Bible...")
    character_bible = CharacterBibleGenerator().generate(story)
    script = ScriptWriter().write(story, args.minutes)

    print("Створюю 100 сцен та English prompts для Rich Gen Image Tool...")
    scene_plan = ScenePlanner().plan(story, script, args.minutes, character_bible)
    if len(scene_plan.scenes) != 100:
        raise RuntimeError(f"Pipeline очікує рівно 100 сцен, але отримано {len(scene_plan.scenes)}")

    project_id = args.project or datetime.now().strftime("story_%Y%m%d_%H%M%S")
    project = StoryProject().create(story, script, project_id, scene_plan, character_bible.model_dump())

    if not args.no_tts:
        print("\n🎙 Створюю українську озвучку через FilmDubUA/Piper...")
        narration = NarrationGenerator().generate(
            scene_plan, project / "audio", settings.filmdubua_voice_profile,
            settings.tts_rate, settings.tts_volume
        )
        (project / "narration.json").write_text(
            json.dumps(narration.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Озвучено: {len(narration.segments)} сцен, {narration.total_duration_seconds / 60:.1f} хв")
    else:
        print("\n⏭ Озвучку пропущено")

    if args.images_dir:
        print("\n🖼 Імпортую 100 готових зображень Rich Gen Image Tool...")
        import_external_images(args.images_dir, project, expected_count=100)
    else:
        print("\n🖼 Зображення НЕ генерую старим способом.")
        print("   Зараз треба зробити 100 картинок у Rich Gen за prompts із scenes.json.")

    if not args.no_music:
        print("\n🎵 Створюю оригінальну процедурну музику...")
        music = ProceduralMusicGenerator().generate(
            max(1, scene_plan.total_duration_seconds), project / "music"
        )
        (project / "music" / "music_manifest.json").write_text(
            json.dumps({"provider": "procedural_local", "file": music.name,
                        "license": "originally generated by this project"},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if not args.no_sfx:
        sfx = ProceduralSFXGenerator().generate_hit(project / "sfx")
        (project / "sfx" / "sfx_manifest.json").write_text(
            json.dumps({"provider": "procedural_local", "file": sfx.name,
                        "license": "originally generated by this project"},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )

    first_image = next(iter(sorted((project / "images").glob("scene_001.*"))), None)
    if first_image and first_image.exists():
        ThumbnailGenerator().generate(story.title, first_image, project / "thumbnail")

    print("\n🔎 Створюю YouTube SEO...")
    YouTubeMetadataGenerator().generate(story, project / "youtube")

    if not args.no_render and not args.no_tts and args.images_dir:
        print("\n🎬 Збираю фінальне MP4 через FFmpeg...")
        music_files = sorted(project.joinpath("music").glob("*.wav"))
        output = VideoRenderer(
            settings.ffmpeg_bin, settings.output_fps,
            settings.video_width, settings.video_height
        ).render(
            scene_plan, project, music_file=music_files[0] if music_files else None
        )
        print(f"Фінальне відео: {output}")
    else:
        print("\n⏭ Рендер пропущено: спочатку потрібні 100 Rich Gen зображень.")

    require_video = not args.no_render and not args.no_tts and bool(args.images_dir)
    packager = FinalProjectPackager()
    report = packager.validate(project, require_video=require_video)
    manifest = packager.write_manifest(project, report)
    print(f"\n📦 Маніфест: {manifest}")

    if args.package:
        try:
            archive = packager.package(project, require_video=require_video)
            print(f"ZIP-пакет: {archive}")
        except FinalPackageError as exc:
            print(f"\n⛔ ZIP не створено: {exc}")
    elif report["valid"]:
        print("Проєкт пройшов фінальну перевірку.")
    else:
        print("Проєкт ще не готовий:")
        for item in report["missing"]:
            print(f"  - {item}")

    print(f"\nГотово. Проєкт: {project}")


if __name__ == "__main__":
    main()
