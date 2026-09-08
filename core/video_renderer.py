import json
import shutil
import subprocess
from pathlib import Path

from core.scene_schema import ScenePlan


class VideoRenderError(RuntimeError):
    pass


class VideoRenderer:
    """Assemble scene images, narration and background music into an MP4 with FFmpeg."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg", fps: int = 30, width: int = 1920, height: int = 1080):
        self.ffmpeg_bin = ffmpeg_bin
        self.fps = fps
        self.width = width
        self.height = height

    def _check_ffmpeg(self) -> None:
        if shutil.which(self.ffmpeg_bin) is None and not Path(self.ffmpeg_bin).exists():
            raise VideoRenderError(f"FFmpeg не знайдено: {self.ffmpeg_bin}")

    def _run(self, args: list[str]) -> None:
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise VideoRenderError(result.stderr[-4000:] or "FFmpeg завершився з помилкою")

    def render(
        self,
        scene_plan: ScenePlan,
        project_dir: str | Path,
        output_name: str = "story.mp4",
        music_file: str | Path | None = None,
    ) -> Path:
        self._check_ffmpeg()
        project_dir = Path(project_dir)
        final_dir = project_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        output = final_dir / output_name
        work_dir = final_dir / "render_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        concat_file = work_dir / "scenes.txt"
        normalized = []
        for scene in scene_plan.scenes:
            image = project_dir / "images" / f"scene_{scene.number:03d}.png"
            audio = project_dir / "audio" / f"narration_{scene.number:03d}.wav"
            if not image.exists():
                raise VideoRenderError(f"Відсутнє зображення сцени {scene.number}: {image}")
            if not audio.exists():
                raise VideoRenderError(f"Відсутня озвучка сцени {scene.number}: {audio}")

            duration = max(1.0, float(scene.estimated_duration_seconds))
            normalized_image = work_dir / f"image_{scene.number:03d}.png"
            self._run([
                self.ffmpeg_bin, "-y", "-i", str(image),
                "-vf", f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
                "-frames:v", "1", str(normalized_image),
            ])
            normalized.append((normalized_image, duration))

        with concat_file.open("w", encoding="utf-8", newline="\n") as f:
            for image, duration in normalized:
                safe_path = image.resolve().as_posix().replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")
                f.write(f"duration {duration:.3f}\n")
            if normalized:
                safe_path = normalized[-1][0].resolve().as_posix().replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        silent_video = work_dir / "visual.mp4"
        self._run([
            self.ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-vf", f"fps={self.fps},format=yuv420p", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            str(silent_video),
        ])

        audio_list = work_dir / "audio.txt"
        with audio_list.open("w", encoding="utf-8", newline="\n") as f:
            for scene in scene_plan.scenes:
                audio = (project_dir / "audio" / f"narration_{scene.number:03d}.wav").resolve().as_posix().replace("'", "'\\''")
                f.write(f"file '{audio}'\n")

        narration_audio = work_dir / "narration.wav"
        self._run([
            self.ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(audio_list),
            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(narration_audio),
        ])

        if music_file and Path(music_file).exists():
            mixed_audio = work_dir / "mixed.m4a"
            self._run([
                self.ffmpeg_bin, "-y", "-i", str(narration_audio), "-stream_loop", "-1", "-i", str(music_file),
                "-filter_complex", "[0:a]volume=1.0[n];[1:a]volume=0.12,aloop=loop=-1:size=2e+09[m];[n][m]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[a]",
                "-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-shortest", str(mixed_audio),
            ])
            audio_input = mixed_audio
        else:
            audio_input = narration_audio

        self._run([
            self.ffmpeg_bin, "-y", "-i", str(silent_video), "-i", str(audio_input),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", "-shortest", str(output),
        ])

        manifest = {
            "output": str(output),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "scenes": len(scene_plan.scenes),
            "music": str(music_file) if music_file else None,
            "renderer": "ffmpeg",
        }
        (final_dir / "render.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return output
