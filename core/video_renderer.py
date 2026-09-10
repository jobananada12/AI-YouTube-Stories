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

    @staticmethod
    def _progress(current: int, total: int, label: str) -> None:
        width = 32
        ratio = current / max(total, 1)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r{label}: [{bar}] {current}/{total} ({ratio * 100:5.1f}%)", end="", flush=True)
        if current >= total:
            print()

    def _normalize_image(self, image: Path, output: Path) -> None:
        # Never stretch the source image: preserve its aspect ratio and pad to exact 1920x1080.
        self._run([
            self.ffmpeg_bin, "-y", "-i", str(image),
            "-vf", f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-frames:v", "1", str(output),
        ])

    def _render_scene(self, image: Path, audio: Path, output: Path, duration: float) -> None:
        # The image remains visible for the exact narration duration plus a silent 0.25s tail.
        # apad makes that tail real silence in the scene audio, so scene boundaries stay aligned.
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
        total_scenes = len(scene_plan.scenes)

        print(f"Рендер сцен: {total_scenes} шт. | 1920x1080 | 30 FPS")
        for index, scene in enumerate(scene_plan.scenes, start=1):
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
            self._progress(index, total_scenes, "Сцени")

        if not scene_files:
            raise VideoRenderError("Немає сцен для рендерингу")

        print("Збираю відео та аудіо без чорних кадрів...")
        concat_file = work_dir / "scenes.txt"
        with concat_file.open("w", encoding="utf-8", newline="\n") as f:
            for scene_file in scene_files:
                safe_path = scene_file.resolve().as_posix().replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        visual_concat = work_dir / "visual_concat.mp4"
        self._run([
            self.ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", str(visual_concat),
        ])

        # Use the audio embedded in the concatenated scene videos. Each scene already
        # contains its exact WAV duration plus the required 0.25s silent tail.
        if music_file and Path(music_file).exists():
            mixed_audio = work_dir / "mixed.m4a"
            self._run([
                self.ffmpeg_bin, "-y", "-i", str(visual_concat), "-stream_loop", "-1", "-i", str(music_file),
                "-filter_complex", "[0:a]volume=1.0[n];[1:a]volume=0.12[m];[n][m]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[a]",
                "-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-shortest", str(mixed_audio),
            ])
            audio_input = mixed_audio
        else:
            audio_input = visual_concat

        print("Нормалізую фінальний звук: -16 LUFS / -1.5 dBTP...")
        self._run([
            self.ffmpeg_bin, "-y", "-i", str(visual_concat), "-i", str(audio_input),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-movflags", "+faststart", str(output),
        ])
        print(f"Готово: {output}")

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
