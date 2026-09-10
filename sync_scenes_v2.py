import json
import re
from pathlib import Path
import whisper

BASE = Path(r"C:\AI-YouTube-Stories")
AUDIO = BASE / "tts" / "narration_30min.mp3"
SCRIPT = BASE / "narration_ua_30min.txt"
OUT = BASE / "scene_timing.json"
WHISPER_JSON = BASE / "whisper_transcription.json"

print("=== ТОЧНА СИНХРОНІЗАЦІЯ 50 СЦЕН ===")
print()

if not AUDIO.exists():
    raise FileNotFoundError(AUDIO)

if not SCRIPT.exists():
    raise FileNotFoundError(SCRIPT)

def norm(text):
    text = text.lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[^а-яіїєґa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

print("Завантажую Whisper...")
model = whisper.load_model("small")

print("Розпізнаю аудіо з таймкодами...")
result = model.transcribe(
    str(AUDIO),
    language="uk",
    task="transcribe",
    fp16=False,
    verbose=True,
    word_timestamps=True,
)

# Зберігаємо повну транскрипцію, щоб не запускати Whisper повторно
with open(WHISPER_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------
# Отримуємо слова Whisper із таймкодами
# ------------------------------------------------------------

words = []

for segment in result.get("segments", []):
    for word in segment.get("words", []):
        text = norm(word.get("word", ""))
        if not text:
            continue

        words.append({
            "text": text,
            "start": float(word["start"]),
            "end": float(word["end"]),
        })

print()
print(f"Whisper слів: {len(words)}")

# ------------------------------------------------------------
# Читаємо сценарій
# ------------------------------------------------------------

text = SCRIPT.read_text(encoding="utf-8")

# Нормалізуємо переноси
text = text.replace("\r\n", "\n").replace("\r", "\n")

# Абзаци
paragraphs = [
    p.strip()
    for p in re.split(r"\n\s*\n+", text)
    if p.strip()
]

print(f"Абзаців у narration_ua_30min.txt: {len(paragraphs)}")

# ------------------------------------------------------------
# Якщо файл має рівно 50 абзаців — це ідеальний варіант.
# Якщо більше — пробуємо об'єднати дуже короткі абзаци.
# ------------------------------------------------------------

if len(paragraphs) != 50:

    print()
    print("Увага: у narration_ua_30min.txt не рівно 50 абзаців.")
    print("Будую 50 послідовних текстових блоків за кількістю символів.")

    total = sum(len(norm(p)) for p in paragraphs)

    blocks = []
    current = []
    current_len = 0

    target = total / 50

    for p in paragraphs:
        p_len = len(norm(p))

        # Не розриваємо абзац.
        if (
            current
            and current_len + p_len > target
            and len(blocks) < 49
        ):
            blocks.append(" ".join(current))
            current = []
            current_len = 0

        current.append(p)
        current_len += p_len

    if current:
        blocks.append(" ".join(current))

    # Якщо через абзаци отримали не 50 блоків,
    # робимо останнє балансування.
    if len(blocks) != 50:
        print(f"Отримано {len(blocks)} блоків.")
        print("Використовую 50 послідовних ділянок тексту.")
        normalized_full = norm(text)
        positions = [
            round(len(normalized_full) * i / 50)
            for i in range(50)
        ]

        blocks = []
        for i in range(50):
            a = positions[i]
            b = positions[i + 1] if i < 49 else len(normalized_full)
            blocks.append(normalized_full[a:b])

else:
    blocks = paragraphs

print(f"Текстових блоків для сцен: {len(blocks)}")

# ------------------------------------------------------------
# Будуємо один суцільний текст Whisper.
# Одночасно зберігаємо позицію кожного слова.
# ------------------------------------------------------------

whisper_text = ""
word_positions = []

for i, w in enumerate(words):
    if whisper_text:
        whisper_text += " "

    start_pos = len(whisper_text)
    whisper_text += w["text"]
    end_pos = len(whisper_text)

    word_positions.append({
        "start_pos": start_pos,
        "end_pos": end_pos,
        "start": w["start"],
        "end": w["end"],
    })

# ------------------------------------------------------------
# Пошук кожного блоку сценарію у Whisper.
# Використовуємо перші та останні слова блоку.
# Це значно надійніше за пошук повного абзацу,
# бо Whisper може помилятися в окремих словах.
# ------------------------------------------------------------

def find_phrase_position(phrase, cursor, from_end=False):
    phrase = norm(phrase)

    if not phrase:
        return None

    # Спочатку пробуємо весь текст.
    p = whisper_text.find(phrase, cursor)

    if p >= 0:
        return p

    words_phrase = phrase.split()

    # Пробуємо 12, 10, 8, 6, 5 слів.
    for count in (12, 10, 8, 6, 5, 4):
        if len(words_phrase) < count:
            continue

        if from_end:
            part = " ".join(words_phrase[-count:])
        else:
            part = " ".join(words_phrase[:count])

        p = whisper_text.find(part, cursor)

        if p >= 0:
            return p

    return None

def position_to_time(position):
    if not word_positions:
        return 0.0

    for item in word_positions:
        if item["start_pos"] <= position <= item["end_pos"]:
            return item["start"]

    if position <= word_positions[0]["start_pos"]:
        return word_positions[0]["start"]

    return word_positions[-1]["end"]

# ------------------------------------------------------------
# Знаходимо початки сцен
# ------------------------------------------------------------

scene_starts = []
cursor = 0

for i, block in enumerate(blocks, start=1):

    block_norm = norm(block)
    block_words = block_norm.split()

    first_phrase = " ".join(block_words[:12])
    last_phrase = " ".join(block_words[-12:])

    start_pos = find_phrase_position(
        first_phrase,
        cursor,
        from_end=False
    )

    if start_pos is None:
        # Резерв: шукаємо перші 6 слів
        start_pos = find_phrase_position(
            " ".join(block_words[:6]),
            cursor,
            from_end=False
        )

    if start_pos is None:
        print(f"Сцена {i:02d}: початок не знайдено")
        scene_starts.append(None)
        continue

    scene_starts.append(start_pos)

    # Наступний пошук починаємо після знайденого місця
    cursor = start_pos + max(1, len(first_phrase))

    print(
        f"Сцена {i:02d}: "
        f"{position_to_time(start_pos):.2f}s"
    )

# ------------------------------------------------------------
# Виправляємо пропуски
# ------------------------------------------------------------

for i in range(len(scene_starts)):

    if scene_starts[i] is not None:
        continue

    previous = None
    next_pos = None

    for j in range(i - 1, -1, -1):
        if scene_starts[j] is not None:
            previous = scene_starts[j]
            break

    for j in range(i + 1, len(scene_starts)):
        if scene_starts[j] is not None:
            next_pos = scene_starts[j]
            break

    if previous is not None and next_pos is not None:
        scene_starts[i] = (previous + next_pos) // 2
    elif previous is not None:
        scene_starts[i] = previous
    else:
        scene_starts[i] = 0

# ------------------------------------------------------------
# Гарантуємо послідовність
# ------------------------------------------------------------

for i in range(1, len(scene_starts)):
    if scene_starts[i] < scene_starts[i - 1]:
        scene_starts[i] = scene_starts[i - 1]

# ------------------------------------------------------------
# Створюємо результат
# ------------------------------------------------------------

audio_duration = (
    float(result["segments"][-1]["end"])
    if result.get("segments")
    else 0.0
)

output = []

for i in range(50):

    start = position_to_time(scene_starts[i])

    if i < 49:
        end = position_to_time(scene_starts[i + 1])
    else:
        end = audio_duration

    output.append({
        "scene": i + 1,
        "image": f"scene_{i + 1:03d}.png",
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(max(0, end - start), 3),
        "text_preview": blocks[i][:160].replace("\n", " ")
    })

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print()
print("=" * 60)
print("ГОТОВО!")
print("=" * 60)
print()
print(f"Файл: {OUT}")
print(f"Тривалість аудіо: {audio_duration:.2f} сек")
print()

for item in output:
    print(
        f"{item['scene']:02d} | "
        f"{item['start']:8.2f} -> "
        f"{item['end']:8.2f} | "
        f"{item['duration']:7.2f} сек | "
        f"{item['image']}"
    )

print()
print("Відео ще НЕ створюємо.")
