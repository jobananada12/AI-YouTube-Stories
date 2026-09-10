from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "tts" / "narrator_config.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def split_text(text: str, max_chars: int = 650) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?…])\s+", paragraph)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
    return chunks


def synthesize_chunk(text: str, output: Path, voice: str, model_dir: Path, speed: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "piper",
        "-m",
        voice,
        "--data-dir",
        str(model_dir),
        "--length-scale",
        str(1.0 / speed),
        "--sentence-silence",
        "0.28",
        "-f",
        str(output),
        "--",
        text,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def concat_wavs(files: list[Path], output: Path, paragraph_silence: float) -> None:
    if not files:
        raise RuntimeError("No WAV chunks were generated")

    with wave.open(str(files[0]), "rb") as first:
        params = first.getparams()
        nchannels = first.getnchannels()
        sampwidth = first.getsampwidth()
        framerate = first.getframerate()

    with wave.open(str(output), "wb") as out:
        out.setnchannels(nchannels)
        out.setsampwidth(sampwidth)
        out.setframerate(framerate)

        silence = b"\x00" * int(framerate * paragraph_silence) * nchannels * sampwidth
        for index, path in enumerate(files):
            with wave.open(str(path), "rb") as src:
                out.writeframes(src.readframes(src.getnframes()))
            if index < len(files) - 1:
                out.writeframes(silence)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Ukrainian narrator using Piper")
    parser.add_argument("--input", default="narration_ua_30min.txt")
    parser.add_argument("--output", default="projects/_tts_preview/narration_ua.wav")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--max-chars", type=int, default=650)
    args = parser.parse_args()

    config = load_config()
    voice = args.voice or config["voice"]
    model_dir = Path(args.model_dir or config["model_dir"])
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir

    input_path = ROOT / args.input
    output_path = ROOT / args.output
    text = input_path.read_text(encoding="utf-8")
    chunks = split_text(text, args.max_chars)

    work_dir = output_path.parent / "chunks"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Voice: {voice}")
    print(f"Chunks: {len(chunks)}")
    print(f"Output: {output_path}")

    wavs: list[Path] = []
    for index, chunk in enumerate(chunks, 1):
        wav = work_dir / f"chunk_{index:04d}.wav"
        if not wav.exists():
            print(f"[{index}/{len(chunks)}] synthesizing...")
            synthesize_chunk(chunk, wav, voice, model_dir, float(config["speed"]))
        else:
            print(f"[{index}/{len(chunks)}] cached")
        wavs.append(wav)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_wavs(wavs, output_path, float(config["paragraph_silence"]))
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
