from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

from core.scene_schema import ScenePlan


class VideoRenderError(RuntimeError):
    pass


class VideoRenderer:
    """Render each scene from the real WAV duration, then concatenate scenes."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg", fps: int = 30, width: int = 1920, height: int = 1080):
        self.ffmpeg_bin = ffmpeg_bin
        self.fps = fps
        self.width = width
        self.height = height
        self.tail_seconds = 0.25

    def _check_ffmpeg(self) -> None:
        if shutil.which(self.ffmpeg_bin) is None and not Path(self.ffmpeg_bin).exists():
            raise VideoRenderError(f"FFmpeg не знайдено: {self.ffmpeg_bin}")

    def _run(self, args: list[str]) -> None:
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise VideoRenderError(result.stderr[-5000:] or "FFmpeg завершився з помилкою")

    @staticmethod
    def _wav_duration(path: Path) -> float:
        try:
            with wave.open(str(path), "rb") as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                if rate <= 0:
                    raise ValueError("invalid WAV sample rate")
                return frames / float(rate)
        except (wave.Error, OSError, ValueError) as exc:
            raise VideoRenderError(f"Не вдалося визначити тривалість WAV: {path}: {exc}") from exc

    def _normalize_image(self, image: Path, output: Path) -> None:
        self._run([
            self.ffmpeg_bin, "-y", "-i", str(image),
            "-vf", f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-frames:v", "1", str(output),
        ])

    def _render_scene(self, image: Path, audio: Path, output: Path, duration: float) -> None:
        # The image is held for the exact narration duration plus a tiny silent visual tail.
        self._run([
            self.ffmpeg_bin, "-y",
            "-loop", "1", "-i", str(image),
            "-i", str(audio),
            "-t", f"{duration + self.tail_seconds:.3f}",
            "-vf", f"fps={self.fps},format=yuv420p",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-af", "apad",
            "-shortest", "-movflags", "+faststart",
            str(output),
        ])

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
        scenes_dir = final_dir / "scenes"
        scenes_dir.mkdir(parents=True, exist_ok=True)
        work_dir = final_dir / "render_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        output = final_dir / output_name

        scene_files: list[Path] = []
        timing: list[dict] = []

        for scene in scene_plan.scenes:
            image = project_dir / "images" / f"scene_{scene.number:03d}.png"
            audio = project_dir / "audio" / f"narration_{scene.number:03d}.wav"
            if not image.exists():
                raise VideoRenderError(f"Відсутнє зображення сцени {scene.number}: {image}")
            if not audio.exists():
                raise VideoRenderError(f"Відсутня озвучка сцени {scene.number}: {audio}")

            duration = self._wav_duration(audio)
            if duration <= 0:
                raise VideoRenderError(f"Порожня озвучка сцени {scene.number}: {audio}")

            normalized_image = work_dir / f"image_{scene.number:03d}.png"
            self._normalize_image(image, normalized_image)

            scene_video = scenes_dir / f"scene_{scene.number:03d}.mp4"
            self._render_scene(normalized_image, audio, scene_video, duration)
            scene_files.append(scene_video)
            timing.append({
                "scene": scene.number,
                "audio_seconds": round(duration, 3),
                "video_seconds": round(duration + self.tail_seconds, 3),
                "tail_seconds": self.tail_seconds,
            })

        if not scene_files:
            raise VideoRenderError("Немає сцен для рендерингу")

        concat_file = work_dir / "scenes.txt"
        with concat_file.open("w", encoding="utf-8", newline="\n") as f:
            for scene_file in scene_files:
                safe_path = scene_file.resolve().as_posix().replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        narration_audio = work_dir / "narration.wav"
        audio_list = work_dir / "audio.txt"
        with audio_list.open("w", encoding="utf-8", newline="\n") as f:
            for scene in scene_plan.scenes:
                audio = (project_dir / "audio" / f"narration_{scene.number:03d}.wav").resolve().as_posix().replace("'", "'\\''")
                f.write(f"file '{audio}'\n")

        self._run([
            self.ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(audio_list),
            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(narration_audio),
        ])

        if music_file and Path(music_file).exists():
            mixed_audio = work_dir / "mixed.m4a"
            self._run([
                self.ffmpeg_bin, "-y", "-i", str(narration_audio), "-stream_loop", "-1", "-i", str(music_file),
                "-filter_complex", "[0:a]volume=1.0[n];[1:a]volume=0.12[m];[n][m]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[a]",
                "-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-shortest", str(mixed_audio),
            ])
            audio_input = mixed_audio
        else:
            audio_input = narration_audio

        silent_concat = work_dir / "visual_concat.mp4"
        self._run([
            self.ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", str(silent_concat),
        ])

        self._run([
            self.ffmpeg_bin, "-y", "-i", str(silent_concat), "-i", str(audio_input),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-movflags", "+faststart", str(output),
        ])

        manifest = {
            "output": str(output),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "scenes": len(scene_files),
            "tail_seconds": self.tail_seconds,
            "timing_source": "actual WAV duration",
            "music": str(music_file) if music_file else None,
            "renderer": "ffmpeg-per-scene",
            "timing": timing,
        }
        (final_dir / "render.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return output
