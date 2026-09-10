import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
WORK_DIR = OUTPUT_DIR / "_render"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


def run(cmd, capture=False):
    print(">", " ".join(str(x) for x in cmd))
    return subprocess.run(
        [str(x) for x in cmd],
        check=True,
        text=True,
        capture_output=capture
    )


def duration(path):
    result = run([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ], capture=True)

    return float(result.stdout.strip())


def scene_number(path):
    name = path.stem.lower()

    patterns = [
        r"scene[_\-\s]*(\d+)",
        r"sc[_\-\s]*(\d+)",
        r"shot[_\-\s]*(\d+)",
        r"(\d+)"
    ]

    for pattern in patterns:
        m = re.search(pattern, name)
        if m:
            return int(m.group(1))

    return None


def natural_key(path):
    number = scene_number(path)
    return (number if number is not None else 999999, path.name.lower())


def find_images():
    result = []

    for p in ROOT.rglob("*"):
        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
            and OUTPUT_DIR not in p.parents
        ):
            result.append(p)

    return sorted(result, key=natural_key)


def find_audio():
    result = []

    for p in ROOT.rglob("*"):
        if (
            p.is_file()
            and p.suffix.lower() in AUDIO_EXTS
            and OUTPUT_DIR not in p.parents
        ):
            result.append(p)

    return sorted(result, key=natural_key)


def parse_time(value):
    value = str(value).strip().replace(",", ".")

    try:
        parts = value.split(":")

        if len(parts) == 3:
            h = float(parts[0])
            m = float(parts[1])
            s = float(parts[2])
            return h * 3600 + m * 60 + s

        if len(parts) == 2:
            m = float(parts[0])
            s = float(parts[1])
            return m * 60 + s

        return float(value)

    except Exception:
        return None


def load_timeline():

    candidates = [
        ROOT / "scenes.json",
        ROOT / "timeline.json",
        ROOT / "timecodes.json",
        ROOT / "timestamps.json",
    ]

    for file in candidates:

        if not file.exists():
            continue

        try:
            data = json.loads(
                file.read_text(
                    encoding="utf-8-sig"
                )
            )
        except Exception:
            continue

        if isinstance(data, dict):

            for key in [
                "scenes",
                "timeline",
                "segments",
                "subtitles"
            ]:
                if isinstance(data.get(key), list):
                    data = data[key]
                    break

        if not isinstance(data, list):
            continue

        result = []

        for index, item in enumerate(data, 1):

            if not isinstance(item, dict):
                continue

            start = None
            end = None

            for key in [
                "start",
                "start_time",
                "startTime",
                "start_seconds",
                "start_sec"
            ]:
                if key in item:
                    start = parse_time(item[key])
                    break

            for key in [
                "end",
                "end_time",
                "endTime",
                "end_seconds",
                "end_sec"
            ]:
                if key in item:
                    end = parse_time(item[key])
                    break

            if (
                start is not None
                and end is not None
                and end > start
            ):
                result.append({
                    "id": index,
                    "start": start,
                    "end": end
                })

        if result:
            print(f"Timeline found: {file}")
            return result

    return []


def match_by_scene(files):

    result = {}

    for file in files:

        number = scene_number(file)

        if number is not None:
            result.setdefault(number, file)

    return result


def create_scene_video(
    image,
    audio,
    output,
    scene_duration,
    resolution,
    fps
):

    width, height = resolution.split("x")

    filter_video = (
        f"scale={width}:{height}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"zoompan="
        f"z='min(zoom+0.0005,1.08)':"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':"
        f"d=1:"
        f"s={width}x{height}:"
        f"fps={fps}"
    )

    run([
        "ffmpeg",
        "-y",

        "-loop",
        "1",

        "-i",
        image,

        "-i",
        audio,

        "-t",
        f"{scene_duration:.3f}",

        "-vf",
        filter_video,

        "-r",
        str(fps),

        "-c:v",
        "libx264",

        "-preset",
        "medium",

        "-crf",
        "20",

        "-pix_fmt",
        "yuv420p",

        "-c:a",
        "aac",

        "-b:a",
        "192k",

        "-ar",
        "48000",

        "-shortest",

        output
    ])


def cut_audio(
    source,
    start,
    end,
    output
):

    length = end - start

    run([
        "ffmpeg",
        "-y",

        "-ss",
        f"{start:.3f}",

        "-i",
        source,

        "-t",
        f"{length:.3f}",

        "-vn",

        "-c:a",
        "pcm_s16le",

        output
    ])


def concat_videos(clips, output):

    concat_file = WORK_DIR / "concat.txt"

    with concat_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        for clip in clips:

            path = (
                str(clip.resolve())
                .replace("\\", "/")
                .replace("'", "'\\''")
            )

            f.write(
                f"file '{path}'\n"
            )

    run([
        "ffmpeg",
        "-y",

        "-f",
        "concat",

        "-safe",
        "0",

        "-i",
        concat_file,

        "-c:v",
        "libx264",

        "-preset",
        "medium",

        "-crf",
        "20",

        "-pix_fmt",
        "yuv420p",

        "-c:a",
        "aac",

        "-b:a",
        "192k",

        "-ar",
        "48000",

        "-movflags",
        "+faststart",

        output
    ])


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resolution",
        default="1920x1080"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=24
    )

    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print(
            "ERROR: ffmpeg not found in PATH."
        )
        sys.exit(1)

    if shutil.which("ffprobe") is None:
        print(
            "ERROR: ffprobe not found in PATH."
        )
        sys.exit(1)

    print()
    print("=" * 60)
    print("AI YOUTUBE STORIES")
    print("VIDEO ASSEMBLER")
    print("=" * 60)

    images = find_images()
    audios = find_audio()

    print()
    print(
        f"Images found : {len(images)}"
    )

    print(
        f"Audio found  : {len(audios)}"
    )

    if not images:
        print(
            "\nERROR: images not found."
        )
        sys.exit(1)

    if not audios:
        print(
            "\nERROR: audio not found."
        )
        sys.exit(1)

    timeline = load_timeline()

    image_by_scene = match_by_scene(images)
    audio_by_scene = match_by_scene(audios)

    full_audio = None

    if len(audios) == 1:
        full_audio = audios[0]

    scenes = []

    # ---------------------------------------------------------
    # MODE 1
    # One full narration + scene timestamps
    # ---------------------------------------------------------

    if full_audio and timeline:

        print()
        print(
            "Mode: ONE FULL AUDIO + TIMECODES"
        )

        for index, item in enumerate(
            timeline,
            1
        ):

            image = image_by_scene.get(
                index
            )

            if image is None and index <= len(images):
                image = images[index - 1]

            if image is None:
                print(
                    f"WARNING: image for scene {index} missing"
                )
                continue

            scenes.append({
                "id": index,
                "image": image,
                "start": item["start"],
                "end": item["end"]
            })

    # ---------------------------------------------------------
    # MODE 2
    # One audio per scene
    # ---------------------------------------------------------

    elif not full_audio:

        print()
        print(
            "Mode: ONE AUDIO FILE PER SCENE"
        )

        scene_numbers = sorted(
            set(image_by_scene)
            & set(audio_by_scene)
        )

        for number in scene_numbers:

            audio = audio_by_scene[number]

            scenes.append({
                "id": number,
                "image": image_by_scene[number],
                "audio": audio,
                "duration": duration(audio)
            })

    # ---------------------------------------------------------
    # MODE 3
    # One audio but no timeline
    # ---------------------------------------------------------

    elif full_audio:

        print()
        print(
            "WARNING: no timeline found."
        )

        print(
            "Splitting narration equally between images."
        )

        total = duration(
            full_audio
        )

        for index, image in enumerate(
            images,
            1
        ):

            start = (
                total
                * (index - 1)
                / len(images)
            )

            end = (
                total
                * index
                / len(images)
            )

            scenes.append({
                "id": index,
                "image": image,
                "start": start,
                "end": end
            })

    if not scenes:

        print(
            "\nERROR: no scenes could be assembled."
        )

        sys.exit(1)

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    WORK_DIR.mkdir(
        exist_ok=True
    )

    # Remove previous temporary render.
    for file in WORK_DIR.iterdir():

        if file.is_file():
            file.unlink()

    clips = []

    print()
    print(
        f"Scenes to render: {len(scenes)}"
    )

    print()

    for index, scene in enumerate(
        scenes,
        1
    ):

        scene_id = scene["id"]

        image = scene["image"]

        video_file = (
            WORK_DIR
            / f"scene_{scene_id:03d}.mp4"
        )

        if "audio" in scene:

            audio = scene["audio"]

            scene_duration = scene[
                "duration"
            ]

            create_scene_video(
                image,
                audio,
                video_file,
                scene_duration,
                args.resolution,
                args.fps
            )

        else:

            start = scene["start"]
            end = scene["end"]

            scene_duration = (
                end - start
            )

            audio_file = (
                WORK_DIR
                / f"audio_{scene_id:03d}.wav"
            )

            cut_audio(
                full_audio,
                start,
                end,
                audio_file
            )

            create_scene_video(
                image,
                audio_file,
                video_file,
                scene_duration,
                args.resolution,
                args.fps
            )

        clips.append(
            video_file
        )

        print(
            f"[{index}/{len(scenes)}] "
            f"Scene {scene_id} "
            f"{scene_duration:.2f}s"
        )

    output_file = (
        OUTPUT_DIR
        / "youtube_story.mp4"
    )

    print()
    print(
        "Joining all scenes..."
    )

    concat_videos(
        clips,
        output_file
    )

    final_duration = duration(
        output_file
    )

    print()
    print("=" * 60)
    print("VIDEO READY")
    print("=" * 60)

    print(
        f"File     : {output_file}"
    )

    print(
        f"Duration : {final_duration / 60:.2f} min"
    )

    print(
        f"Scenes   : {len(scenes)}"
    )

    print(
        f"Video    : {args.resolution}"
    )

    print(
        f"FPS      : {args.fps}"
    )

    print("=" * 60)

    shutil.rmtree(
        WORK_DIR,
        ignore_errors=True
    )


if __name__ == "__main__":
    main()