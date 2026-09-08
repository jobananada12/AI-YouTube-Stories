from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path


class FinalPackageError(RuntimeError):
    pass


class FinalProjectPackager:
    """Validate and package all generated assets for one YouTube story."""

    REQUIRED = ("story.json", "script.md", "character_bible.json", "scenes.json")

    def validate(self, project_dir: str | Path, require_video: bool = True) -> dict:
        project = Path(project_dir)
        if not project.exists():
            raise FinalPackageError(f"Проєкт не знайдено: {project}")

        missing = [name for name in self.REQUIRED if not (project / name).exists()]
        if not (project / "audio").exists():
            missing.append("audio/")
        if not (project / "images").exists():
            missing.append("images/")
        if require_video and not (project / "final" / "story.mp4").exists():
            missing.append("final/story.mp4")

        youtube = project / "youtube"
        thumbnail = project / "thumbnail" / "thumbnail.jpg"
        report = {
            "project": str(project),
            "valid": not missing,
            "missing": missing,
            "has_thumbnail": thumbnail.exists(),
            "has_youtube_metadata": (youtube / "youtube.json").exists() or (youtube / "youtube.md").exists(),
            "has_music": any((project / "music").glob("*.wav")) if (project / "music").exists() else False,
            "has_sfx": any((project / "sfx").glob("*.wav")) if (project / "sfx").exists() else False,
            "has_video": (project / "final" / "story.mp4").exists(),
        }
        return report

    def write_manifest(self, project_dir: str | Path, report: dict) -> Path:
        project = Path(project_dir)
        output = project / "project_manifest.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    def package(self, project_dir: str | Path, output_dir: str | Path | None = None) -> Path:
        project = Path(project_dir)
        report = self.validate(project, require_video=True)
        self.write_manifest(project, report)
        if not report["valid"]:
            raise FinalPackageError("Проєкт не готовий до фінальної упаковки: " + ", ".join(report["missing"]))

        destination = Path(output_dir) if output_dir else project.parent / "packages"
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / f"{project.name}.zip"

        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in project.rglob("*"):
                if path.is_file() and path.name != archive.name:
                    zf.write(path, path.relative_to(project.parent))
        return archive
