import argparse
import json
from datetime import datetime
from pathlib import Path

from app.config import settings
from core.character_bible_schema import CharacterBible
from core.final_package import FinalProjectPackager
from core.image_generator import ImageGenerator
from core.music_generator import ProceduralMusicGenerator
from core.project import StoryProject
from core.scene_schema import ScenePlan, SceneSpec
from core.seo_generator import YouTubeMetadataGenerator
from core.sfx_generator import ProceduralSFXGenerator
from core.story_schema import StorySpec
from core.tts_generator import NarrationGenerator
from core.video_renderer import VideoRenderer


def load_authored_content(content_path: str | Path) -> tuple[StorySpec, CharacterBible, ScenePlan]:
    """Load a hand-authored story and its ready-made image prompts.

    Ollama is NOT used here. The story, scenes and visual prompts are already
    authored and stored in the repository. The program only performs production:
    TTS -> local Stable Diffusion -> FFmpeg video.
    """
    path = Path(content_path)
    if not path.is_file():
        raise FileNotFoundError(f"Контент не знайдено: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    story = StorySpec.model_validate({k: v for k, v in data.items() if k in StorySpec.model_fields})

    characters = []
    for character in data.get("characters", []):
        appearance = character.get("appearance", "")
        characters.append({
            "name": character.get("name", ""),
            "role": character.get("role", ""),
            "age": character.get("age", ""),
            "gender_presentation": "",
            "height": "",
            "build": "",
            "skin": "",
            "face": "",
            "eyes": "",
            "eyebrows": "",
            "nose": "",
            "lips": "",
            "hair": "",
            "facial_hair": "",
            "signature_clothing": appearance,
            "footwear": "",
            "accessories": "",
            "distinctive_features": "",
            "typical_expression": character.get("personality", ""),
            "color_palette": ["monochrome grayscale"],
            "image_prompt_anchor": appearance,
        })
    bible = CharacterBible.model_validate({"characters": characters})

    scenes = []
    for item in data.get("scenes", []):
        scenes.append(SceneSpec.model_validate({
            "number": item["number"],
            "title": item.get("title", f"Scene {item['number']}"),
            "purpose": item.get("purpose", "Advance the authored story."),
            "narration": item["narration"],
            "estimated_duration_seconds": max(1, int(item.get("estimated_duration_seconds", 1))),
            "characters": item.get("characters", []),
            "location": item.get("location", "old house"),
            "time_of_day": item.get("time_of_day", "evening"),
            "action": item.get("action", "").strip() or item["narration"],
            "visual_prompt": item["visual_prompt"].strip(),
            "mood": item.get("mood", "mysterious"),
            "continuity_notes": item.get("continuity_notes", ""),
            "transition": item.get("transition", "cut"),
        }))

    if len(scenes) != 100:
        raise ValueError(f"Авторський контент має містити рівно 100 сцен, знайдено {len(scenes)}")

    return story, bible, ScenePlan(
        target_duration_seconds=30 * 60,
        total_duration_seconds=sum(s.estimated_duration_seconds for s in scenes),
        scenes=scenes,
    )


def build_authored_project(content_path: str | Path, project_id: str | None, minutes: int) -> Path:
    story, bible, scene_plan = load_authored_content(content_path)
    project_id = project_id or f"story_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project = StoryProject().create(story, "\n\n".join(s.narration for s in scene_plan.scenes), project_id, scene_plan, bible.model_dump())

    print(f"СЮЖЕТ: {story.title}")
    print("Контент авторський: сюжет + 100 сцен + 100 готових prompts")
    print("Ollama для сюжету/prompts НЕ запускається.")

    print("1/3 Генерую українську озвучку через FilmDubUA/Piper...")
    narration = NarrationGenerator().generate(
        scene_plan=scene_plan,
        output_dir=project / "audio",
        voice_profile=settings.filmdubua_voice_profile,
        rate=settings.tts_rate,
        volume=settings.tts_volume,
    )
    (project / "narration.json").write_text(json.dumps(narration.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    print("2/3 Генерую 100 картинок локально через Stable Diffusion...")
    ImageGenerator().generate(scene_plan, bible, project / "images")

    print("3/3 Збираю фінальне відео...")
    music = ProceduralMusicGenerator().generate(max(1, narration.total_duration_seconds), project / "music")
    (project / "music" / "music_manifest.json").write_text(json.dumps({"provider": "procedural_local", "file": music.name}, ensure_ascii=False, indent=2), encoding="utf-8")
    sfx = ProceduralSFXGenerator().generate_hit(project / "sfx")
    (project / "sfx" / "sfx_manifest.json").write_text(json.dumps({"provider": "procedural_local", "file": sfx.name}, ensure_ascii=False, indent=2), encoding="utf-8")
    YouTubeMetadataGenerator().generate(story, project / "youtube")

    output = VideoRenderer(
        settings.ffmpeg_bin, settings.output_fps, settings.video_width, settings.video_height
    ).render(scene_plan, project, music_file=music)

    report = FinalProjectPackager().validate(project, require_video=True)
    FinalProjectPackager().write_manifest(project, report)
    if not report["valid"]:
        raise RuntimeError("Фінальна перевірка не пройдена: " + "; ".join(report["missing"]))
    print(f"\nГОТОВО: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="AI YouTube Stories — authored story + local Stable Diffusion + Piper + FFmpeg")
    parser.add_argument("--content", default="content/room_without_plan.json", help="Готовий авторський JSON із сюжетом і 100 prompts")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()
    if not 10 <= args.minutes <= 120:
        parser.error("--minutes must be between 10 and 120")
    build_authored_project(args.content, args.project, args.minutes)


if __name__ == "__main__":
    main()
