from __future__ import annotations

import json
import zipfile
from pathlib import Path


class FinalPackageError(RuntimeError):
    pass


class FinalProjectPackager:
    """Validate and package one completed YouTube story project."""

    REQUIRED_FILES = ("story.json", "script.md", "character_bible.json", "scenes.json")
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

    @classmethod
    def _scene_images(cls, project: Path) -> list[Path]:
        images_dir = project / "images"
        if not images_dir.exists():
            return []
        return sorted(
            [p for p in images_dir.iterdir() if p.is_file() and p.name.startswith("scene_") and p.suffix.lower() in cls.IMAGE_EXTENSIONS],
            key=lambda p: p.name.lower(),
        )

    def validate(self, project_dir: str | Path, require_video: bool = True) -> dict:
        project = Path(project_dir)
        if not project.exists():
            raise FinalPackageError(f"Проєкт не знайдено: {project}")
        missing: list[str] = [name for name in self.REQUIRED_FILES if not (project / name).is_file()]
        audio_files = sorted((project / "audio").glob("narration_*.wav")) if (project / "audio").exists() else []
        image_files = self._scene_images(project)
        if not audio_files:
            missing.append("audio/narration_*.wav")
        if not image_files:
            missing.append("images/scene_*.{png,jpg,jpeg,webp}")
        if audio_files and image_files and len(audio_files) != len(image_files):
            missing.append(f"однакова кількість audio/images (audio={len(audio_files)}, images={len(image_files)})")
        if audio_files and len(audio_files) != 100:
            missing.append(f"рівно 100 WAV-сегментів (знайдено={len(audio_files)})")
        if image_files and len(image_files) != 100:
            missing.append(f"рівно 100 зображень сцен (знайдено={len(image_files)})")
        thumbnail = project / "thumbnail" / "thumbnail.jpg"
        youtube_json = project / "youtube" / "youtube.json"
        youtube_md = project / "youtube" / "youtube.md"
        video = project / "final" / "story.mp4"
        if not thumbnail.is_file():
            missing.append("thumbnail/thumbnail.jpg")
        if not (youtube_json.is_file() or youtube_md.is_file()):
            missing.append("youtube/youtube.json")
        if require_video and not video.is_file():
            missing.append("final/story.mp4")
        report = {
            "project": str(project), "valid": not missing, "missing": missing,
            "audio_segments": len(audio_files), "image_segments": len(image_files),
            "expected_scene_count": 100,
            "has_thumbnail": thumbnail.is_file(),
            "has_youtube_metadata": youtube_json.is_file() or youtube_md.is_file(),
            "has_music": any((project / "music").glob("*.wav")) if (project / "music").exists() else False,
            "has_sfx": any((project / "sfx").glob("*.wav")) if (project / "sfx").exists() else False,
            "has_video": video.is_file(),
        }
        return report

    def write_manifest(self, project_dir: str | Path, report: dict) -> Path:
        project = Path(project_dir)
        output = project / "project_manifest.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    def package(self, project_dir: str | Path, output_dir: str | Path | None = None, require_video: bool = True) -> Path:
        project = Path(project_dir)
        report = self.validate(project, require_video=require_video)
        self.write_manifest(project, report)
        if not report["valid"]:
            raise FinalPackageError("Проєкт не готовий до фінальної упаковки: " + ", ".join(report["missing"]))
        destination = Path(output_dir) if output_dir else project.parent / "packages"
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / f"{project.name}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in project.rglob("*"):
                if path.is_file() and path != archive:
                    zf.write(path, path.relative_to(project.parent))
        return archive
