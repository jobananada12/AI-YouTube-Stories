import wave as wavfile
from pathlib import Path
import re
import subprocess
import shutil

from tts_uk.inference import synthesis


# ============================================================
# CONFIG
# ============================================================

INPUT = Path("narration_ua_30min.txt")
WORK = Path("tts/narration_parts")
OUTPUT_WAV = Path("tts/narration_30min.wav")
OUTPUT_MP3 = Path("tts/narration_30min.mp3")

VOICE = "mykyta"
SAMPLE_RATE = 44100

# Цільовий розмір одного фрагмента.
# Не робимо занадто довгі запити до моделі.
MAX_CHARS = 650

# Пауза між абзацами/частинами.
PAUSE_MS = 650


# ============================================================
# TEXT SPLITTING
# ============================================================

def clean_text(text):
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Прибираємо зайві пробіли
    text = re.sub(r"[ \t]+", " ", text)

    # Не більше двох переносів поспіль
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_sentences(text):
    """
    Розбиваємо український текст на речення,
    але залишаємо крапку/знак питання/оклику.
    """

    sentences = re.split(
        r"(?<=[.!?…])\s+",
        text.strip()
    )

    return [s.strip() for s in sentences if s.strip()]


def make_chunks(text):
    """
    Групуємо речення приблизно по 650 символів.
    Речення не розриваємо посередині.
    """

    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        sentences = split_sentences(paragraph)

        current = ""

        for sentence in sentences:

            if not current:
                current = sentence
                continue

            candidate = current + " " + sentence

            if len(candidate) <= MAX_CHARS:
                current = candidate
            else:
                chunks.append(current.strip())
                current = sentence

        if current:
            chunks.append(current.strip())

    return chunks


# ============================================================
# WAV HELPERS
# ============================================================

def read_wav(path):
    with wavfile.open(str(path), "rb") as f:
        params = f.getparams()
        frames = f.readframes(f.getnframes())

    return params, frames


def make_silence(ms):
    samples = int(SAMPLE_RATE * ms / 1000)
    return b"\x00\x00" * samples


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 60)
print("TTS-UK / MYKYTA — 30 MIN NARRATION")
print("=" * 60)
print()

if not INPUT.exists():
    raise FileNotFoundError(
        f"Не знайдено файл: {INPUT}"
    )

text = INPUT.read_text(
    encoding="utf-8"
)

text = clean_text(text)

chunks = make_chunks(text)

print(f"Input: {INPUT}")
print(f"Characters: {len(text):,}")
print(f"Chunks: {len(chunks)}")
print(f"Voice: {VOICE}")
print(f"Sample rate: {SAMPLE_RATE} Hz")
print()

# Очистити старі частини
if WORK.exists():
    shutil.rmtree(WORK)

WORK.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# SYNTHESIS
# ============================================================

generated = []

for i, chunk in enumerate(chunks, 1):

    output = WORK / f"part_{i:04d}.wav"

    print(
        f"[{i:03d}/{len(chunks):03d}] "
        f"{len(chunk):4d} chars..."
    )

    mels, wave, stats = synthesis(
        text=chunk,
        voice=VOICE,
        n_takes=1,
        use_latest_take=False,

        # Ті самі параметри, що у хорошому test_mykyta.wav
        token_dur_scaling=1,

        f0_mean=0,
        f0_std=0,

        energy_mean=0,
        energy_std=0,

        sigma_decoder=0.8,
        sigma_token_duration=0.666,
        sigma_f0=1,
        sigma_energy=1,
    )

    audio = (
        wave
        .detach()
        .cpu()
        .squeeze()
        .numpy()
    )

    audio = audio.clip(-1.0, 1.0)

    pcm16 = (
        audio * 32767
    ).astype("<i2")

    with wavfile.open(
        str(output),
        "wb"
    ) as f:

        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)

        f.writeframes(
            pcm16.tobytes()
        )

    generated.append(output)

    duration = len(audio) / SAMPLE_RATE

    print(
        f"       {duration:.2f} sec"
    )


# ============================================================
# CONCATENATE
# ============================================================

print()
print("Збираємо всі частини...")

params, first_frames = read_wav(
    generated[0]
)

all_frames = []

for i, path in enumerate(generated):

    p, frames = read_wav(path)

    if p.nchannels != 1:
        raise RuntimeError(
            f"Неправильна кількість каналів: {path}"
        )

    if p.sampwidth != 2:
        raise RuntimeError(
            f"Неправильна bit depth: {path}"
        )

    if p.framerate != SAMPLE_RATE:
        raise RuntimeError(
            f"Неправильна sample rate: {path}"
        )

    all_frames.append(frames)

    # Пауза між частинами
    if i < len(generated) - 1:
        all_frames.append(
            make_silence(PAUSE_MS)
        )


OUTPUT_WAV.parent.mkdir(
    parents=True,
    exist_ok=True
)

with wavfile.open(
    str(OUTPUT_WAV),
    "wb"
) as out:

    out.setnchannels(1)
    out.setsampwidth(2)
    out.setframerate(SAMPLE_RATE)

    for frames in all_frames:
        out.writeframes(frames)


# ============================================================
# DURATION
# ============================================================

with wavfile.open(
    str(OUTPUT_WAV),
    "rb"
) as f:

    total_frames = f.getnframes()
    total_rate = f.getframerate()

duration = total_frames / total_rate

hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = int(duration % 60)

print()
print("=" * 60)
print("ГОТОВО")
print("=" * 60)
print()
print(f"WAV: {OUTPUT_WAV}")
print(
    f"Duration: "
    f"{hours:02d}:{minutes:02d}:{seconds:02d}"
)
print(
    f"Size: "
    f"{OUTPUT_WAV.stat().st_size / 1024 / 1024:.1f} MB"
)
print()


# ============================================================
# MP3
# ============================================================

ffmpeg = shutil.which("ffmpeg")

if ffmpeg:

    print("Створюємо MP3 для зручного прослуховування...")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(OUTPUT_WAV),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "128k",
        str(OUTPUT_MP3),
    ]

    subprocess.run(
        cmd,
        check=True
    )

    print()
    print(f"MP3: {OUTPUT_MP3}")
    print(
        f"Size: "
        f"{OUTPUT_MP3.stat().st_size / 1024 / 1024:.1f} MB"
    )

else:

    print(
        "FFmpeg не знайдено — MP3 не створено."
    )


print()
print("Готово.")
