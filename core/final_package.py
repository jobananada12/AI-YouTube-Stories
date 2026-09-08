from __future__ import annotations

import json
import zipfile
from pathlib import Path


class FinalPackageError(RuntimeError):
    pass


class FinalProjectPackager:
    """Validate and package one completed YouTube story project."""

    REQUIRED_FILES = (
        "story.json",
        "script.md",
        "character_bible.json",
        "scenes.json",
    )

    def validate(self, project_dir: str | Path, require_video: bool = True) -> dict:
        project = Path(project_dir)
        if not project.exists():
            raise FinalPackageError(f"Проєкт не знайдено: {project}")

        missing: list[str] = []
        for name in self.REQUIRED_FILES:
            if not (project / name).is_file():
                missing.append(name)

        audio_files = sorted((project / "audio").glob("narration_*.wav")) if (project / "audio").exists() else []
        image_files = sorted((project / "images").glob("scene_*.png")) if (project / "images").exists() else []
        if not audio_files:
            missing.append("audio/narration_*.wav")
        if not image_files:
            missing.append("images/scene_*.png")

        thumbnail = project / "thumbnail" / "thumbnail.jpg"
        youtube_json = project / "youtube" / "youtube.json"
        youtube_md = project / "youtube" / "youtube.md"
        video = project / "final" / "story.mp4"

        if require_video and not video.is_file():
            missing.append("final/story.mp4")

        report = {
            "project": str(project),
            "valid": not missing,
            "missing": missing,
            "audio_segments": len(audio_files),
            "image_segments": len(image_files),
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

    def package(
        self,
        project_dir: str | Path,
        output_dir: str | Path | None = None,
        require_video: bool = True,
    ) -> Path:
        project = Path(project_dir)
        report = self.validate(project, require_video=require_video)
        self.write_manifest(project, report)
        if not report["valid"]:
            raise FinalPackageError(
                "Проєкт не готовий до фінальної упаковки: " + ", ".join(report["missing"])
            )

        destination = Path(output_dir) if output_dir else project.parent / "packages"
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / f"{project.name}.zip"

        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in project.rglob("*"):
                if path.is_file() and path != archive:
                    zf.write(path, path.relative_to(project.parent))
        return archive
