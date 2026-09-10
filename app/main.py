import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from app.config import settings
from core.character_bible_schema import CharacterBible
from core.final_package import FinalProjectPackager
from core.music_generator import ProceduralMusicGenerator
from core.project import StoryProject
from core.scene_schema import ScenePlan, SceneSpec
from core.seo_generator import YouTubeMetadataGenerator
from core.sfx_generator import ProceduralSFXGenerator
from core.story_schema import CharacterSpec, StorySpec
from core.tts_generator import NarrationGenerator
from core.video_renderer import VideoRenderer

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def import_external_images(source_dir: str | Path, project_dir: Path, expected_count: int = 100) -> None:
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Папку із зображеннями не знайдено: {source}")
    candidates = sorted(
        [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )
    if len(candidates) != expected_count:
        raise ValueError(f"Rich Gen: потрібно рівно {expected_count} зображень, а знайдено {len(candidates)} у {source}")
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
        "files": [f"scene_{i:03d}{candidates[i - 1].suffix.lower()}" for i in range(1, expected_count + 1)],
    }
    (project_dir / "external_images.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Імпортовано зображень Rich Gen: {expected_count}")


def _fixed_character_bible(data: dict) -> CharacterBible:
    entries = []
    for c in data["characters"]:
        name = c["name"]
        if name == "Alexander":
            anchor = c["appearance"]
            role = "protagonist"
            age = c["age"]
            appearance = c["appearance"]
            gender = "male"
            height = "average"
            build = "thin but muscular"
            skin = "sun-tanned weathered skin"
            face = "thoughtful weathered face"
            eyes = "thoughtful eyes"
            eyebrows = "natural dark eyebrows"
            nose = "straight nose"
            lips = "natural lips"
            hair = "short tousled dark hair"
            facial_hair = "light stubble"
            clothing = "worn brown-gray jacket, blue sweater with a subtle tree pattern, dark trousers"
            footwear = "worn boots"
            accessories = "none"
            distinctive = "weathered appearance"
            expression = "thoughtful and cautious"
            palette = ["brown-gray", "blue", "dark gray"]
        elif name == "Irina":
            anchor = c["appearance"]
            role = "daughter of the former scientist"
            age = c["age"]
            appearance = c["appearance"]
            gender = "female"
            height = "average"
            build = "slim"
            skin = "natural light skin"
            face = "intelligent thoughtful face"
            eyes = "attentive eyes"
            eyebrows = "natural eyebrows"
            nose = "straight nose"
            lips = "natural lips"
            hair = "shoulder-length dark hair"
            facial_hair = "none"
            clothing = "simple dark coat, practical trousers"
            footwear = "practical dark shoes"
            accessories = "none"
            distinctive = "calm attentive appearance"
            expression = "thoughtful and restrained"
            palette = ["dark gray", "black"]
        else:
            anchor = c["appearance"]
            role = "house cat"
            age = "adult"
            gender = "animal"
            height = "small"
            build = "compact"
            skin = "gray-and-white short fur"
            face = "small cat face"
            eyes = "alert eyes"
            eyebrows = "none"
            nose = "small cat nose"
            lips = "none"
            hair = "short gray-and-white fur"
            facial_hair = "none"
            clothing = "none"
            footwear = "none"
            accessories = "none"
            distinctive = "gray-and-white coat"
            expression = "cautious and curious"
            palette = ["gray", "white"]
        entries.append({
            "name": name, "role": role, "age": age, "gender_presentation": gender,
            "height": height, "build": build, "skin": skin, "face": face, "eyes": eyes,
            "eyebrows": eyebrows, "nose": nose, "lips": lips, "hair": hair,
            "facial_hair": facial_hair, "signature_clothing": clothing, "footwear": footwear,
            "accessories": accessories, "distinctive_features": distinctive,
            "typical_expression": expression, "color_palette": palette,
            "image_prompt_anchor": anchor,
        })
    return CharacterBible.model_validate({"characters": entries})


def load_fixed_content(content_path: str | Path):
    path = Path(content_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Файл готового сюжету не знайдено: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if len(data.get("scenes", [])) != 100:
        raise ValueError("Готовий сюжет повинен містити рівно 100 сцен")
    characters = [
        CharacterSpec(
            name=c["name"], role=c["role"], age=c["age"], appearance=c["appearance"],
            personality=c["personality"], motivation=c["motivation"],
        ) for c in data["characters"]
    ]
    story = StorySpec(
        title=data["title"], genre=data["genre"], tone=data["tone"], logline=data["logline"],
        premise=data["premise"], theme=data["theme"], setting=data["setting"],
        protagonist_goal="зрозуміти походження прихованої кімнати і зберегти правду",
        central_conflict="суперечність між фізичною реальністю будинку та людською пам'яттю",
        ending_type="розкриття та спокійний фінал",
        characters=characters,
        outline=[s["title"] for s in data["scenes"][::10]],
    )
    scene_specs = []
    for s in data["scenes"]:
        present = []
        text = s["narration"] + " " + s["visual_prompt"]
        for name in ("Alexander", "Irina", "Cat"):
            if name in text:
                present.append(name)
        if not present:
            present = ["Alexander"]
        scene_specs.append(SceneSpec(
            number=s["number"], title=s["title"], purpose=s["narration"], narration=s["narration"],
            estimated_duration_seconds=max(1, int(len(s["narration"].split()) / 2.1)),
            characters=present, location=data["setting"], time_of_day="day or evening according to story context",
            action=s["narration"], visual_prompt=s["visual_prompt"], mood=data["tone"],
            continuity_notes="Use only the characters and objects explicitly present in this scene.", transition="cut",
        ))
    plan = ScenePlan(
        target_duration_seconds=30 * 60,
        total_duration_seconds=sum(s.estimated_duration_seconds for s in scene_specs),
        scenes=scene_specs,
    )
    return story, _fixed_character_bible(data), plan


def prepare_fixed_content(content_path: str | Path, project_id: str | None) -> Path:
    print("1/4 Завантажую мій готовий сюжет і 100 готових prompts...")
    story, character_bible, scene_plan = load_fixed_content(content_path)
    print(f"\nІСТОРІЯ: {story.title}\nСЦЕН: {len(scene_plan.scenes)}\n")
    project_id = project_id or f"story_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project = StoryProject().create(story, "\n\n".join(s.narration for s in scene_plan.scenes), project_id, scene_plan, character_bible.model_dump())
    print("2/4 Створюю українську озвучку з мого готового сценарію...")
    narration = NarrationGenerator().generate(
        scene_plan=scene_plan, output_dir=project / "audio", voice_profile=settings.filmdubua_voice_profile,
        rate=settings.tts_rate, volume=settings.tts_volume,
    )
    (project / "narration.json").write_text(json.dumps(narration.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    print("3/4 Готую музику, SFX та YouTube-метадані...")
    music = ProceduralMusicGenerator().generate(max(1, narration.total_duration_seconds), project / "music")
    (project / "music" / "music_manifest.json").write_text(json.dumps({"provider":"procedural_local","file":music.name}, ensure_ascii=False, indent=2), encoding="utf-8")
    sfx = ProceduralSFXGenerator().generate_hit(project / "sfx")
    (project / "sfx" / "sfx_manifest.json").write_text(json.dumps({"provider":"procedural_local","file":sfx.name}, ensure_ascii=False, indent=2), encoding="utf-8")
    YouTubeMetadataGenerator().generate(story, project / "youtube")
    print("4/4 ГОТОВО: сюжет, Character Bible, 100 сцен, 100 prompts і WAV підготовлені.")
    print(f"Проєкт: {project}")
    print("Тепер зробіть 100 картинок у Rich Gen за visual_prompt з scenes.json.")
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
    output = VideoRenderer(settings.ffmpeg_bin, settings.output_fps, settings.video_width, settings.video_height).render(
        scene_plan, project, music_file=music_files[0] if music_files else None
    )
    report = FinalProjectPackager().validate(project, require_video=True)
    FinalProjectPackager().write_manifest(project, report)
    if not report["valid"]:
        raise RuntimeError("Фінальна перевірка не пройдена: " + "; ".join(report["missing"]))
    print(f"\nГОТОВО: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="AI YouTube Stories — fixed story + 100 Rich Gen prompts + Piper + FFmpeg")
    parser.add_argument("topic", nargs="?", help="Тема нової історії")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--project", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--render-project", default=None)
    parser.add_argument("--content", default=None, help="Готовий сюжет JSON із 100 сцен; без Ollama story/script generation")
    args = parser.parse_args()

    if args.render_project:
        if not args.images_dir:
            parser.error("Для --render-project обов'язково вкажіть --images-dir")
        render_existing_project(args.render_project, args.images_dir)
        return

    if args.content:
        if args.images_dir:
            parser.error("Для --content спочатку створіть WAV-проєкт, потім використовуйте --render-project")
        prepare_fixed_content(args.content, args.project)
        return

    if not args.topic:
        parser.error("Для нової історії потрібно вказати тему або --content")
    parser.error("Для цього проєкту використовуйте готовий контент через --content")


if __name__ == "__main__":
    main()
