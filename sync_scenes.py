import json
from pathlib import Path
import re
import whisper

BASE = Path(r"C:\AI-YouTube-Stories")
AUDIO = BASE / "tts" / "narration_30min.mp3"
SCRIPT = BASE / "narration_ua_30min.txt"
SCENES = BASE / "scenes.json"
OUT = BASE / "scene_timing.json"

print("=== SYNCHRONIZATION 50 SCENES ===")
print(f"Audio:  {AUDIO}")
print(f"Script: {SCRIPT}")
print()

if not AUDIO.exists():
    raise FileNotFoundError(f"Не знайдено аудіо: {AUDIO}")

if not SCRIPT.exists():
    raise FileNotFoundError(f"Не знайдено текст: {SCRIPT}")

if not SCENES.exists():
    raise FileNotFoundError(f"Не знайдено scenes.json: {SCENES}")

print("Завантажую Whisper...")
model = whisper.load_model("small")

print("Розпізнаю аудіо. Це може зайняти деякий час...")
result = model.transcribe(
    str(AUDIO),
    language="uk",
    task="transcribe",
    fp16=False,
    verbose=True,
)

segments = result["segments"]

print()
print(f"Whisper отримав {len(segments)} сегментів.")

with open(SCRIPT, "r", encoding="utf-8") as f:
    narration = f.read()

with open(SCENES, "r", encoding="utf-8") as f:
    scenes = json.load(f)

if len(scenes) != 50:
    raise RuntimeError(f"Очікувалося 50 сцен, знайдено {len(scenes)}")

# Нормалізація тексту для пошуку
def normalize(text):
    text = text.lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[^а-яіїєґa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

full_text = normalize(narration)

# Whisper-текст
whisper_text = " ".join(
    normalize(s["text"]) for s in segments
)

# Пошук сцен.
#
# У scenes.json можуть бути назви/описи, які не є дослівним текстом
# озвучки, тому спочатку використовуємо текстову послідовність
# самого narration_ua_30min.txt.
#
# Скрипт шукає назву сцени в narration, якщо вона присутня.
# Якщо назва відсутня, використовуємо рівномірний поділ тексту
# як резервний варіант і позначаємо його для перевірки.

results = []

# Беремо текстові частини сцен, якщо вони присутні в scenes.json.
def scene_text(scene):
    parts = []

    for key in ("title", "description", "text", "narration", "story"):
        value = scene.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)

    return normalize(" ".join(parts))

# Знаходимо приблизну позицію сцени в загальному тексті.
# Використовуємо назву, а якщо її немає — ключові слова.
positions = []

cursor = 0

for i, scene in enumerate(scenes, start=1):
    title = scene.get("title", f"Сцена {i}")
    st = scene_text(scene)

    pos = -1

    if st:
        # Спочатку шукаємо повний фрагмент
        pos = full_text.find(st, cursor)

        # Потім шукаємо перші значущі слова
        if pos < 0:
            words = st.split()
            candidates = []

            for n in (12, 10, 8, 6, 5):
                if len(words) >= n:
                    phrase = " ".join(words[:n])
                    p = full_text.find(phrase, cursor)
                    if p >= 0:
                        candidates.append(p)

            if candidates:
                pos = min(candidates)

    positions.append(pos)

# Якщо scenes.json містить лише описи, а не текст озвучки,
# робимо пропорційні текстові межі як резерв.
found = [p for p in positions if p >= 0]

if len(found) < 40:
    print()
    print("Увага: сцени не містять достатньо дослівного тексту озвучки.")
    print("Використовую послідовні межі narration_ua_30min.txt.")
    print()

    # Розподіл за кількістю символів тексту.
    # Це НЕ фінальна синхронізація відео — лише підготовка.
    starts = []

    for i in range(50):
        starts.append(round(len(full_text) * i / 50))

    positions = starts

# Перетворення текстової позиції у приблизний аудіотаймкод
# через відповідність позиції тексту Whisper-сегментам.

whisper_items = []
offset = 0

for seg in segments:
    txt = normalize(seg["text"])
    if not txt:
        continue

    start_pos = offset
    end_pos = offset + len(txt)

    whisper_items.append({
        "start_pos": start_pos,
        "end_pos": end_pos,
        "start": float(seg["start"]),
        "end": float(seg["end"]),
    })

    offset = end_pos + 1

def textpos_to_time(pos):
    if not whisper_items:
        return 0.0

    pos = max(0, min(pos, whisper_items[-1]["end_pos"]))

    for item in whisper_items:
        if item["start_pos"] <= pos <= item["end_pos"]:
            length = max(1, item["end_pos"] - item["start_pos"])
            ratio = (pos - item["start_pos"]) / length
            return item["start"] + (item["end"] - item["start"]) * ratio

    if pos < whisper_items[0]["start_pos"]:
        return whisper_items[0]["start"]

    return whisper_items[-1]["end"]

# Сортуємо межі та створюємо 50 сцен.
times = [textpos_to_time(p) for p in positions]

# Гарантуємо послідовність
for i in range(1, len(times)):
    if times[i] < times[i - 1]:
        times[i] = times[i - 1]

audio_duration = float(segments[-1]["end"]) if segments else 0.0

output = []

for i in range(50):
    start = times[i]
    end = times[i + 1] if i < 49 else audio_duration

    output.append({
        "scene": i + 1,
        "image": f"scene_{i + 1:03d}.png",
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(max(0, end - start), 3),
        "title": scenes[i].get("title", f"Сцена {i + 1}")
    })

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print()
print("ГОТОВО!")
print(f"Створено: {OUT}")
print()
print("Перші сцени:")
for item in output[:5]:
    print(
        f"{item['image']} | "
        f"{item['start']:.2f}s -> {item['end']:.2f}s | "
        f"{item['duration']:.2f}s | "
        f"{item['title']}"
    )

print()
print("Поки що відео НЕ створюємо.")
print("Спочатку перевіримо scene_timing.json.")
