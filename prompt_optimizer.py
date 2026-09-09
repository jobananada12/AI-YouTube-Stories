import json
import re
import time
from pathlib import Path

# ============================================================
# AI YouTube Stories — Stable Diffusion Prompt Optimizer
# ============================================================

BASE_DIR = Path(r"C:\AI-YouTube-Stories")

INPUT_FILE = BASE_DIR / "scenes.json"
OUTPUT_FILE = BASE_DIR / "scenes_optimized.json"

# SD 1.5 CLIP has a practical limit of 77 tokens.
# We deliberately keep prompts shorter than that.
MAX_WORDS = 58

# Repeated phrases that waste CLIP space.
REMOVE_PHRASES = [
    "cinematic movie still",
    "realistic digital illustration",
    "professional cinematography",
    "high detail",
    "highly detailed",
    "detailed environment",
    "dramatic lighting",
    "atmospheric",
    "cinematic",
    "movie still",
    "beautiful composition",
    "masterpiece",
    "ultra detailed",
    "8k",
    "4k",
    "award winning",
]

# Obvious language contamination.
REPLACEMENTS = {
    "задумчивий": "thoughtful",
    "задумчивый": "thoughtful",
    "задумчиво": "thoughtfully",
    "таємничий": "mysterious",
    "таємнича": "mysterious",
    "темний": "dark",
    "темна": "dark",
    "старий": "old",
    "стара": "old",
    "чоловік": "man",
    "жінка": "woman",
    "кімната": "room",
    "будинок": "house",
    "двері": "door",
    "підвал": "basement",
    "коридор": "corridor",
}


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.strip()

    # Replace known Ukrainian/Russian contamination.
    for old, new in REPLACEMENTS.items():
        text = re.sub(
            rf"\b{re.escape(old)}\b",
            new,
            text,
            flags=re.IGNORECASE
        )

    # Remove phrases that waste CLIP tokens.
    for phrase in REMOVE_PHRASES:
        text = re.sub(
            re.escape(phrase),
            "",
            text,
            flags=re.IGNORECASE
        )

    # Remove duplicate commas.
    text = re.sub(r",\s*,+", ", ", text)

    # Remove duplicate spaces.
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing punctuation.
    text = text.strip(" ,.;:-")

    return text


def split_prompt(prompt):
    """
    Split prompt into meaningful comma-separated pieces.
    """
    parts = [
        p.strip()
        for p in prompt.split(",")
        if p.strip()
    ]

    return parts


def is_redundant(part, selected):
    """
    Detect repeated / nearly identical prompt components.
    """
    normalized = re.sub(r"[^a-z0-9 ]", "", part.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        return True

    for existing in selected:
        existing_norm = re.sub(
            r"[^a-z0-9 ]",
            "",
            existing.lower()
        )
        existing_norm = re.sub(
            r"\s+",
            " ",
            existing_norm
        ).strip()

        if normalized == existing_norm:
            return True

        # Very rough duplicate detection.
        words_a = set(normalized.split())
        words_b = set(existing_norm.split())

        if len(words_a) >= 4 and len(words_b) >= 4:
            overlap = len(words_a & words_b) / max(
                1,
                len(words_a | words_b)
            )

            if overlap > 0.75:
                return True

    return False


def optimize_prompt(prompt):
    prompt = clean_text(prompt)

    parts = split_prompt(prompt)

    selected = []
    word_count = 0

    # Keep the most important visual information first.
    priority_keywords = [
        "man",
        "woman",
        "alexander",
        "irina",
        "irene",
        "room",
        "house",
        "basement",
        "corridor",
        "door",
        "window",
        "journal",
        "camera",
        "flashlight",
        "night",
        "morning",
        "dark",
        "light",
        "standing",
        "walking",
        "looking",
        "holding",
        "opening",
        "sitting",
    ]

    def priority(part):
        low = part.lower()

        score = 0

        for keyword in priority_keywords:
            if keyword in low:
                score += 2

        # Short descriptions are generally better.
        words = len(part.split())

        if words <= 8:
            score += 1

        return score

    parts.sort(
        key=priority,
        reverse=True
    )

    for part in parts:
        if is_redundant(part, selected):
            continue

        part_words = len(part.split())

        if word_count + part_words > MAX_WORDS:
            continue

        selected.append(part)
        word_count += part_words

    result = ", ".join(selected)

    # Final cleanup.
    result = clean_text(result)

    # Guarantee no accidental double punctuation.
    result = re.sub(r",\s*,+", ", ", result)
    result = re.sub(r"\s+", " ", result)

    return result.strip(" ,.")


def find_prompt_key(scene):
    """
    Try to find the field containing the Stable Diffusion prompt.
    """
    possible_keys = [
        "prompt",
        "image_prompt",
        "sd_prompt",
        "stable_diffusion_prompt",
        "visual_prompt",
        "imagePrompt",
    ]

    for key in possible_keys:
        if key in scene and isinstance(scene[key], str):
            return key

    return None


def main():

    print("=" * 70)
    print("AI YouTube Stories — Prompt Optimizer")
    print("=" * 70)
    print()

    if not INPUT_FILE.exists():
        print("ERROR:")
        print(f"Не знайдено файл:")
        print(INPUT_FILE)
        print()
        print("Перевір, що файл scenes.json знаходиться в:")
        print(BASE_DIR)
        return

    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if isinstance(data, dict):
        scenes = data.get("scenes")

        if scenes is None:
            print("ERROR: У JSON немає поля 'scenes'.")
            return

        root_is_dict = True

    elif isinstance(data, list):
        scenes = data
        root_is_dict = False

    else:
        print("ERROR: Непідтримуваний формат scenes.json")
        return

    print(f"Знайдено сцен: {len(scenes)}")
    print()

    optimized_count = 0
    unchanged_count = 0

    for index, scene in enumerate(scenes, start=1):

        if not isinstance(scene, dict):
            print(f"[{index:03d}] SKIP — scene is not object")
            continue

        key = find_prompt_key(scene)

        if key is None:
            print(f"[{index:03d}] SKIP — prompt field not found")
            continue

        old_prompt = scene[key]
        new_prompt = optimize_prompt(old_prompt)

        old_words = len(old_prompt.split())
        new_words = len(new_prompt.split())

        scene[key] = new_prompt

        if old_prompt.strip() == new_prompt.strip():
            unchanged_count += 1
            status = "UNCHANGED"
        else:
            optimized_count += 1
            status = "OPTIMIZED"

        print(
            f"[{index:03d}] {status:10} "
            f"{old_words:3d} -> {new_words:3d} words"
        )

    # Save optimized copy.
    if root_is_dict:
        output_data = data
        output_data["scenes"] = scenes
    else:
        output_data = scenes

    # Backup output if it already exists.
    if OUTPUT_FILE.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        backup_file = (
            BASE_DIR /
            f"scenes_optimized_backup_{timestamp}.json"
        )

        OUTPUT_FILE.rename(backup_file)

        print()
        print("Попередній scenes_optimized.json переміщено в:")
        print(backup_file)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)
    print()
    print(f"Всього сцен:       {len(scenes)}")
    print(f"Оптимізовано:      {optimized_count}")
    print(f"Без змін:          {unchanged_count}")
    print()
    print("Новий файл:")
    print(OUTPUT_FILE)
    print()
    print("ВАЖЛИВО:")
    print("Оригінальний scenes.json НЕ змінено.")
    print("Наявні PNG також НЕ змінено.")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()