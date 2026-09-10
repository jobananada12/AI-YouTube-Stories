import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from app.config import settings
from core.character_bible import CharacterBibleGenerator
from core.final_package import FinalProjectPackager
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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def import_external_images(source_dir: str | Path, project_dir: Path, expected_count: int = 100) -> None:
    """Import Rich Gen images and rename them deterministically to scene_001...scene_100."""
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Папку із зображеннями не знайдено: {source}")

    candidates = sorted(
        [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )
    if len(candidates) != expected_count:
        raise ValueError(
            f"Rich Gen: потрібно рівно {expected_count} зображень, а знайдено {len(candidates)} у {source}"
        )

    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for old in images_dir.iterdir():
        if old.is_file() and old.suffix.lower() in IMAGE_EXTENSIONS:
            old.unlink()

    for index, image in enumerate(candidates, start=1):
        shutil.copy2(image, images_dir / f"scene_{index:03d}{image.suffix.lower()}")

    manifest = {
        "provider": "Rich Gen Image Tool / external",
        "source": str(source),
        "count": expected_count,
        "files": [
            f"scene_{i:03d}{candidates[i - 1].suffix.lower()}"
            for i in range(1, expected_count + 1)
        ],
    }
    (project_dir / "external_images.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Імпортовано зображень Rich Gen: {expected_count}")


def prepare_project(topic: str, minutes: int, project_id: str | None) -> Path:
    print("1/6 Створюю сюжет...")
    story = StoryGenerator().generate(topic, minutes)
    print(f"\nІСТОРІЯ: {story.title}\nЖАНР: {story.genre}\nЛОГЛАЙН: {story.logline}\n")

    print("2/6 Фіксую Character Bible...")
    character_bible = CharacterBibleGenerator().generate(story)

    print("3/6 Пишу повний сценарій...")
    script = ScriptWriter().write(story, minutes)

    print("4/6 Створюю рівно 100 сцен та English prompts для Rich Gen...")
    scene_plan = ScenePlanner().plan(story, script, minutes, character_bible)
    if len(scene_plan.scenes) != 100:
        raise RuntimeError(f"Потрібно рівно 100 сцен, отримано {len(scene_plan.scenes)}")

    project_id = project_id or datetime.now().strftime("story_%Y%m%d_%H%M%S")
    project = StoryProject().create(
        story, script, project_id, scene_plan, character_bible.model_dump()
    )

    print("5/6 Створюю українську озвучку через FilmDubUA/Piper...")
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

    print("6/6 Готую допоміжні матеріали...")
    music = ProceduralMusicGenerator().generate(
        max(1, narration.total_duration_seconds), project / "music"
    )
    (project / "music" / "music_manifest.json").write_text(
        json.dumps(
            {"provider": "procedural_local", "file": music.name},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    sfx = ProceduralSFXGenerator().generate_hit(project / "sfx")
    (project / "sfx" / "sfx_manifest.json").write_text(
        json.dumps(
            {"provider": "procedural_local", "file": sfx.name},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    YouTubeMetadataGenerator().generate(story, project / "youtube")

    print("\n========================================")
    print("ПІДГОТОВКА ЗАВЕРШЕНА")
    print(f"Проєкт: {project}")
    print("Створено: story + script + Character Bible + 100 scenes + 100 WAV")
    print("ЗОБРАЖЕННЯ НЕ ГЕНЕРУЮТЬСЯ.")
    print("Зробіть 100 картинок у Rich Gen Image Tool за prompts із scenes.json.")
    print("========================================\n")
    return project


def render_existing_project(project_id: str, images_dir: str | Path) -> Path:
    project_store = StoryProject()
    project = project_store.load(project_id)
    scene_plan = project_store.load_scene_plan(project_id)

    narration_file = project / "narration.json"
    if not narration_file.is_file():
        raise FileNotFoundError(f"Не знайдено озвучку: {narration_file}")

    print("1/3 Імпортую рівно 100 зображень Rich Gen...")
    import_external_images(images_dir, project, 100)

    print("2/3 Перевіряю проєкт та озвучку...")
    audio_files = sorted((project / "audio").glob("scene_*.wav"))
    if len(audio_files) != 100:
        raise ValueError(f"Потрібно 100 WAV, знайдено {len(audio_files)} у {project / 'audio'}")

    print("3/3 Рендерю 100 сцен та фінальне MP4...")
    music_files = sorted((project / "music").glob("*.wav"))
    output = VideoRenderer(
        settings.ffmpeg_bin,
        settings.output_fps,
        settings.video_width,
        settings.video_height,
    ).render(scene_plan, project, music_file=music_files[0] if music_files else None)

    report = FinalProjectPackager().validate(project, require_video=True)
    FinalProjectPackager().write_manifest(project, report)
    if not report["valid"]:
        raise RuntimeError("Фінальна перевірка не пройдена: " + "; ".join(report["missing"]))

    print(f"\nГОТОВО: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI YouTube Stories — 30 min story + 100 Rich Gen prompts + Piper + FFmpeg"
    )
    parser.add_argument("topic", nargs="?", help="Тема нової історії")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--project", default=None, help="ID проєкту")
    parser.add_argument("--images-dir", default=None, help="Папка з рівно 100 готовими Rich Gen images")
    parser.add_argument(
        "--render-project",
        default=None,
        help="Рендерити вже підготовлений проєкт без повторної генерації історії/сценарію/TTS",
    )
    args = parser.parse_args()

    if args.render_project:
        if not args.images_dir:
            parser.error("Для --render-project обов'язково вкажіть --images-dir")
        render_existing_project(args.render_project, args.images_dir)
        return

    if not args.topic:
        parser.error("Для нової історії потрібно вказати тему")
    if not 10 <= args.minutes <= 120:
        parser.error("--minutes must be between 10 and 120")
    if args.images_dir:
        parser.error(
            "Спочатку створіть проєкт без --images-dir. Після генерації 100 Rich Gen images "
            "запустіть --render-project PROJECT_ID --images-dir PATH"
        )

    prepare_project(args.topic, args.minutes, args.project)


if __name__ == "__main__":
    main()
