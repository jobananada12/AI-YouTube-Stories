import json
import re
import sys
from pathlib import Path


# ============================================================
# AI YOUTUBE STORIES
# Prompt Optimizer V2
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT = BASE_DIR / "scenes.json"
DEFAULT_OUTPUT = BASE_DIR / "scenes_optimized_v2.json"
CHARACTER_FILE = BASE_DIR / "character.json"


# ============================================================
# CHARACTER BIBLE
# ============================================================

CHARACTERS = {
    "alexander": (
        "Alexander, 35 years old, a thin but muscular man with "
        "sun-tanned weathered skin, short tousled dark hair, "
        "thoughtful eyes, light stubble, wearing a worn brown-gray "
        "jacket, a blue sweater with a subtle tree pattern, "
        "dark trousers and worn boots"
    ),

    "irina": (
        "Irina, the daughter of the former scientist, "
        "a woman in her early thirties with shoulder-length dark hair, "
        "an intelligent thoughtful expression, wearing a simple dark coat "
        "and practical trousers"
    ),

    "cat": (
        "a domestic cat with short gray-and-white fur, "
        "alert eyes and a cautious but curious expression"
    )
}


# ============================================================
# CYRILLIC DETECTION
# ============================================================

CYRILLIC_RE = re.compile(
    r"[\u0400-\u04FF\u0500-\u052F]"
)


def contains_cyrillic(text: str) -> bool:
    return bool(CYRILLIC_RE.search(text or ""))


# ============================================================
# COMMON UKRAINIAN/RUSSIAN FRAGMENTS
# ============================================================

TRANSLATIONS = {
    "в кімнаті": "inside the room",
    "у кімнаті": "inside the room",
    "в кімнату": "into the room",
    "до кімнати": "toward the room",
    "старий будинок": "old house",
    "старому будинку": "old house",
    "старого будинку": "old house",
    "старий науковець": "former scientist",
    "старого науковця": "former scientist",
    "дочка старого науковця": "the former scientist's daughter",
    "дочь старого науковця": "the former scientist's daughter",
    "дочь": "daughter",
    "документи": "documents",
    "книги": "books",
    "книга": "book",
    "двері": "door",
    "двері кімнати": "room door",
    "коридор": "corridor",
    "будинок": "house",
    "кімната": "room",
    "парк": "park",
    "ліс": "forest",
    "дерево": "tree",
    "вікно": "window",
    "стіл": "table",
    "стілець": "chair",
    "сходи": "stairs",
    "підвал": "basement",
    "горище": "attic",
    "ніч": "night",
    "вечір": "evening",
    "ранок": "morning",
    "день": "day",
    "темна кімната": "dark room",
    "старі документи": "old documents",
    "старі книги": "old books",
    "читає": "reads",
    "читав": "reads",
    "читає документи": "reads documents",
    "розглядає": "examines",
    "дивиться": "looks at",
    "дивиться на": "looks at",
    "відкриває": "opens",
    "закриває": "closes",
    "заходить": "enters",
    "виходить": "leaves",
    "іде": "walks",
    "йде": "walks",
    "стоїть": "stands",
    "сидить": "sits",
    "тримає": "holds",
    "торкається": "touches",
    "шукає": "searches for",
    "слухає": "listens",
    "спостерігає": "observes",
    "помічає": "notices",
    "знаходить": "finds",
    "відчуває": "feels",
    "боїться": "is afraid",
    "наляканий": "frightened",
    "здивований": "surprised",
    "обережно": "carefully",
    "повільно": "slowly",
    "раптово": "suddenly",
    "тихо": "quietly",
    "кот": "cat",
    "кіт": "cat",
    "котик": "cat",
    "Олександр": "Alexander",
    "Олександра": "Alexander",
    "Ірина": "Irina",
    "Ірини": "Irina",
}


# ============================================================
# SAFE TEXT CLEANING
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Replace known fragments, longest first
    for src in sorted(TRANSLATIONS, key=len, reverse=True):
        text = text.replace(src, TRANSLATIONS[src])

    return text


# ============================================================
# CHARACTER DETECTION
# ============================================================

def detect_characters(scene):
    blob = " ".join(
        str(scene.get(key, ""))
        for key in (
            "title",
            "description",
            "location",
            "time",
            "action",
            "camera",
            "mood",
            "prompt",
            "original_prompt",
        )
    ).lower()

    result = []

    if any(x in blob for x in [
        "alexander",
        "олександр",
        "чоловік",
        "man",
        "his",
    ]):
        result.append("alexander")

    if any(x in blob for x in [
        "irina",
        "ірина",
        "daughter",
        "дочка",
        "дочь",
    ]):
        result.append("irina")

    if any(x in blob for x in [
        "cat",
        "кіт",
        "кот",
        "котик",
    ]):
        result.append("cat")

    # Alexander is the protagonist of this project.
    # If the scene is clearly about him, keep him.
    if not result and scene.get("original_prompt"):
        original = str(scene["original_prompt"]).lower()

        if any(x in original for x in [
            "man",
            "worker",
            "gardener",
            "park worker",
            "alexander",
        ]):
            result.append("alexander")

    return result


# ============================================================
# ACTION NORMALIZATION
# ============================================================

def normalize_action(scene):
    action = clean_text(scene.get("action", ""))

    if not action:
        original = clean_text(scene.get("original_prompt", ""))
        description = clean_text(scene.get("description", ""))

        source = original or description

        # Try to extract a useful action from the source
        action_patterns = [
            (r"\breads\b", "reads documents"),
            (r"\bexamines\b", "examines the object carefully"),
            (r"\blooks at\b", "looks carefully at the scene"),
            (r"\bopens\b", "opens the door carefully"),
            (r"\benters\b", "enters the room cautiously"),
            (r"\bwalks\b", "walks slowly through the scene"),
            (r"\bstands\b", "stands quietly and observes"),
            (r"\bsits\b", "sits quietly and studies the surroundings"),
            (r"\bsearches\b", "searches the room carefully"),
            (r"\bobserves\b", "observes the surroundings carefully"),
        ]

        for pattern, replacement in action_patterns:
            if re.search(pattern, source, re.I):
                action = replacement
                break

    if not action:
        action = "stands quietly and observes the surroundings"

    return action


# ============================================================
# CAMERA NORMALIZATION
# ============================================================

def normalize_camera(scene):
    camera = clean_text(scene.get("camera", ""))

    if not camera:
        return "cinematic medium shot"

    low = camera.lower()

    if any(x in low for x in [
        "close-up",
        "close up",
        "круп",
    ]):
        return "cinematic close-up"

    if any(x in low for x in [
        "wide",
        "широк",
        "далек",
    ]):
        return "cinematic wide shot"

    if any(x in low for x in [
        "medium",
        "серед",
    ]):
        return "cinematic medium shot"

    if any(x in low for x in [
        "low angle",
        "низьк",
    ]):
        return "cinematic low-angle shot"

    if any(x in low for x in [
        "high angle",
        "висок",
    ]):
        return "cinematic high-angle shot"

    if "pov" in low:
        return "cinematic POV shot"

    return "cinematic medium shot"


# ============================================================
# MOOD
# ============================================================

def normalize_mood(scene):
    mood = clean_text(scene.get("mood", ""))

    if not mood:
        return "quiet mysterious atmosphere"

    mood = mood.strip(" .,")

    return mood


# ============================================================
# LOCATION
# ============================================================

def normalize_location(scene):
    location = clean_text(scene.get("location", ""))

    if not location:
        location = "inside the old house"

    return location


# ============================================================
# TIME
# ============================================================

def normalize_time(scene):
    time = clean_text(scene.get("time", ""))

    if not time:
        time = "dim natural light"

    return time


# ============================================================
# REMOVE BROKEN LANGUAGE
# ============================================================

def remove_cyrillic_fragments(text):
    # First replace known terms
    text = clean_text(text)

    # Remove remaining Cyrillic words instead of allowing
    # broken Ukrainian/Russian fragments into SD prompts.
    text = CYRILLIC_RE.sub("", text)

    # Clean double spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# PROMPT SOURCE SELECTION
# ============================================================

def get_base_prompt(scene):
    """
    original_prompt is the most important source.

    The previously generated 'prompt' may already be corrupted,
    so we deliberately prefer original_prompt.
    """

    original = scene.get("original_prompt")

    if original and str(original).strip():
        return clean_text(original)

    prompt = scene.get("prompt")

    if prompt and str(prompt).strip():
        return clean_text(prompt)

    description = scene.get("description")

    if description and str(description).strip():
        return clean_text(description)

    return ""


# ============================================================
# REMOVE CHARACTER DESCRIPTION DUPLICATION
# ============================================================

def remove_duplicate_character_phrases(text):
    phrases = [
        "Alexander, 35 years old",
        "a thin but muscular man",
        "sun-tanned weathered skin",
        "short tousled dark hair",
        "thoughtful eyes",
        "light stubble",
        "worn brown-gray jacket",
        "blue sweater with a subtle tree pattern",
        "dark trousers",
        "worn boots",
    ]

    for phrase in phrases:
        text = re.sub(
            re.escape(phrase),
            "",
            text,
            flags=re.IGNORECASE
        )

    text = re.sub(r"\s+", " ", text)

    return text.strip(" ,.")


# ============================================================
# FIX COMMON GRAMMAR PROBLEMS
# ============================================================

def grammar_cleanup(text):
    replacements = {
        "Alexander reads documents aloud": "Alexander reads the documents",
        "Alexander читає": "Alexander reads",
        "Alexander просимо документи": "Alexander examines the documents",
        "Alexander вчителіть документи": "Alexander examines the documents",
        "cat впивається": "the cat moves toward",
        "in room": "inside the room",
        "в room": "inside the room",
        "at room": "inside the room",
        "his daughter Irina": "Irina, daughter of the former scientist",
        "Alexander's daughter Irina": "Irina, daughter of the former scientist",
        "daughter of Alexander": "daughter of the former scientist",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    return text


# ============================================================
# BUILD FINAL PROMPT
# ============================================================

def build_prompt(scene):
    base = get_base_prompt(scene)

    base = remove_cyrillic_fragments(base)
    base = grammar_cleanup(base)

    action = normalize_action(scene)
    camera = normalize_camera(scene)
    location = normalize_location(scene)
    time = normalize_time(scene)
    mood = normalize_mood(scene)

    characters = detect_characters(scene)

    # Remove an existing broken character description
    base = remove_duplicate_character_phrases(base)

    parts = []

    if base:
        parts.append(base)

    # Add stable character descriptions
    for character in characters:
        description = CHARACTERS.get(character)

        if description:
            parts.append(description)

    # Add scene action
    if action:
        parts.append(f"Action: {action}")

    # Add environment only when useful
    if location:
        parts.append(f"Location: {location}")

    if time:
        parts.append(f"Lighting and time: {time}")

    if mood:
        parts.append(f"Atmosphere: {mood}")

    parts.append(camera)

    parts.append(
        "cinematic composition, realistic detailed environment, "
        "natural anatomy, realistic lighting, subtle film grain, "
        "high visual detail, coherent perspective, dramatic storytelling"
    )

    prompt = ". ".join(
        part.strip(" .")
        for part in parts
        if part and part.strip()
    )

    prompt = remove_cyrillic_fragments(prompt)
    prompt = grammar_cleanup(prompt)

    # Final cleanup
    prompt = re.sub(r"\.{2,}", ".", prompt)
    prompt = re.sub(r"\s+", " ", prompt)
    prompt = prompt.strip()

    return prompt


# ============================================================
# VALIDATION
# ============================================================

def validate_prompt(prompt, scene):
    warnings = []

    if contains_cyrillic(prompt):
        warnings.append("CYRILLIC_REMAINS")

    if len(prompt.split()) < 25:
        warnings.append("LOW_DETAIL")

    action_words = [
        "reads",
        "examines",
        "looks",
        "opens",
        "closes",
        "enters",
        "leaves",
        "walks",
        "stands",
        "sits",
        "holds",
        "touches",
        "searches",
        "listens",
        "observes",
        "notices",
        "finds",
        "moves",
        "studies",
        "looks",
    ]

    has_action = any(
        re.search(r"\b" + re.escape(word) + r"\b", prompt, re.I)
        for word in action_words
    )

    if not has_action:
        warnings.append("NO_CLEAR_ACTION_VERB")

    camera_words = [
        "shot",
        "close-up",
        "wide shot",
        "medium shot",
        "low-angle",
        "high-angle",
        "POV",
    ]

    if not any(word.lower() in prompt.lower() for word in camera_words):
        warnings.append("NO_CAMERA")

    if not scene.get("location"):
        warnings.append("MISSING_LOCATION_FIELD")

    return warnings


# ============================================================
# PROCESS ONE SCENE
# ============================================================

def process_scene(scene):
    new_prompt = build_prompt(scene)

    warnings = validate_prompt(new_prompt, scene)

    result = dict(scene)

    result["prompt"] = new_prompt
    result["prompt_word_count"] = len(new_prompt.split())
    result["prompt_warnings"] = warnings

    # Keep original prompt untouched
    if "original_prompt" not in result:
        result["original_prompt"] = scene.get("prompt", "")

    if "original_prompt_word_count" not in result:
        result["original_prompt_word_count"] = len(
            str(result.get("original_prompt", "")).split()
        )

    return result


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


# ============================================================
# SAVE JSON
# ============================================================

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# MAIN
# ============================================================

def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else DEFAULT_OUTPUT
    )

    print("=" * 70)
    print("AI YOUTUBE STORIES - PROMPT OPTIMIZER V2")
    print("=" * 70)
    print()

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print()

    if not input_path.exists():
        print("ERROR: Input file not found.")
        print()
        print("Expected:")
        print(input_path)
        print()
        return 1

    try:
        data = load_json(input_path)
    except Exception as e:
        print("ERROR: Cannot read JSON.")
        print(e)
        return 1

    # Accept either:
    # [
    #   {...},
    #   {...}
    # ]
    #
    # or:
    # {
    #   "scenes": [...]
    # }

    if isinstance(data, list):
        scenes = data
        container_type = "list"

    elif isinstance(data, dict) and isinstance(data.get("scenes"), list):
        scenes = data["scenes"]
        container_type = "dict"

    else:
        print("ERROR: Unsupported JSON structure.")
        print("Expected a list of scenes or an object containing 'scenes'.")
        return 1

    print(f"Scenes found: {len(scenes)}")
    print()

    optimized = []

    total_warnings = 0
    cyrillic_count = 0

    for index, scene in enumerate(scenes, start=1):
        result = process_scene(scene)
        optimized.append(result)

        warnings = result.get("prompt_warnings", [])

        total_warnings += len(warnings)

        if "CYRILLIC_REMAINS" in warnings:
            cyrillic_count += 1

        scene_id = scene.get("id", index)

        print(
            f"[{index:03d}/{len(scenes):03d}] "
            f"Scene {scene_id}: "
            f"{result['prompt_word_count']} words"
        )

        if warnings:
            print("       warnings:", ", ".join(warnings))
        else:
            print("       OK")

    if container_type == "list":
        output_data = optimized
    else:
        output_data = dict(data)
        output_data["scenes"] = optimized

    save_json(output_path, output_data)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Saved: {output_path}")
    print(f"Scenes: {len(optimized)}")
    print(f"Warnings: {total_warnings}")
    print(f"Cyrillic prompts: {cyrillic_count}")
    print()

    if cyrillic_count == 0:
        print("SUCCESS: No Cyrillic remains in final prompts.")
    else:
        print("WARNING: Some prompts still contain Cyrillic.")

    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())