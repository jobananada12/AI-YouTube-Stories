from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
FPS = 30
WIDTH = 1920
HEIGHT = 1080
PAUSE = 0.25


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(f'\"{x}\"' if " " in x else x for x in cmd))
    subprocess.run(cmd, check=True)


def natural_key(path: Path):
    m = re.search(r"(\d+)", path.stem)
    return (int(m.group(1)) if m else 10**9, path.stem.lower())


def find_audio(project: Path, explicit: Path | None) -> Path:
    if explicit:
        if not explicit.exists():
            raise FileNotFoundError(f"Аудіопапка не знайдена: {explicit}")
        return explicit

    candidates = [
        project / "audio",
        project / "narration",
        project / "voice",
        project / "voices",
        project / "wav",
        Path("audio"),
        Path("narration"),
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(p.suffix.lower() in AUDIO_EXTS for p in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(
        "Не знайшов папку з аудіо. Використай --audio <папка>. "
        "Очікуються WAV/MP3/M4A/FLAC/OGG."
    )


def scene_number(path: Path) -> int | None:
    m = re.search(r"(?:scene|сцена)?[_ -]?(\d+)", path.stem, re.I)
    return int(m.group(1)) if m else None


def match_image(images: list[Path], number: int, index: int) -> Path:
    for image in images:
        if scene_number(image) == number:
            return image
    if index < len(images):
        return images[index]
    raise FileNotFoundError(f"Не знайдена картинка для сцени {number:03d}")


def audio_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def make_scene(image: Path, audio: Path, out: Path) -> None:
    # The audio is normalized while the scene is encoded, so scene-to-scene
    # loudness is consistent before the final concatenation.
    run([
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(image),
        "-i", str(audio),
        "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-r", str(FPS), "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(out),
    ])


def make_pause(image: Path, out: Path) -> None:
    # Keep the last scene image on screen for exactly 0.25 s with silence.
    run([
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(image),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-t", str(PAUSE),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-r", str(FPS), "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(out),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scene MP4 files from narration audio + generated images, then make final.mp4")
    parser.add_argument("--project", default="room_without_plan_01")
    parser.add_argument("--audio", type=Path, default=None, help="Папка з аудіо; якщо не задана, скрипт шукає її автоматично")
    parser.add_argument("--keep-scenes", action="store_true", help="Не видаляти проміжні MP4 сцени")
    args = parser.parse_args()

    project = Path("projects") / args.project
    image_dir = project / "images"
    work_dir = project / "video_scenes"
    final = project / "final.mp4"

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("FFmpeg/ffprobe не знайдені в PATH.")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Немає папки із зображеннями: {image_dir}")

    audio_dir = find_audio(project, args.audio)
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS], key=natural_key)
    audio = sorted([p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS], key=natural_key)

    if not images:
        raise FileNotFoundError(f"У {image_dir} немає зображень.")
    if not audio:
        raise FileNotFoundError(f"У {audio_dir} немає аудіофайлів.")

    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Проєкт: {project}")
    print(f"Картинок: {len(images)}")
    print(f"Аудіофайлів: {len(audio)}")
    print(f"Аудіо: {audio_dir}")
    print(f"Пауза між сценами: {PAUSE:.2f} с, останній кадр + тиша")

    if len(images) < len(audio):
        raise RuntimeError(f"Аудіофайлів більше, ніж картинок: {len(audio)} > {len(images)}")

    concat_list = work_dir / "concat.txt"
    lines: list[str] = []

    for index, audio_file in enumerate(audio):
        number = scene_number(audio_file) or (index + 1)
        image = match_image(images, number, index)
        scene_out = work_dir / f"scene_{number:03d}.mp4"
        pause_out = work_dir / f"pause_{number:03d}.mp4"
        print(f"\n=== СЦЕНА {number:03d}: {audio_file.name} + {image.name} ===")
        print(f"Тривалість аудіо: {audio_duration(audio_file):.2f} с")
        make_scene(image, audio_file, scene_out)
        make_pause(image, pause_out)
        lines.append(f"file '{scene_out.resolve().as_posix()}'")
        lines.append(f"file '{pause_out.resolve().as_posix()}'")

    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n=== ЗБИРАЮ ФІНАЛЬНИЙ MP4 ===")
    run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy",
        "-movflags", "+faststart",
        str(final),
    ])

    print(f"\nГОТОВО: {final}")
    if not args.keep_scenes:
        print("Проміжні MP4 залишені у", work_dir, "для перевірки; видаляти їх автоматично не будемо.")


if __name__ == "__main__":
    main()
