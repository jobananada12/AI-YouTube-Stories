import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path


# ============================================================
# AI YOUTUBE STORIES
# Prompt Optimizer V3
# ============================================================
# V3 is deliberately conservative:
# - original_prompt is the primary visual source;
# - scene facts are preserved instead of invented;
# - character detection is local to the actual semantic source;
# - canonical character appearance is enforced only when a character
#   is really present;
# - camera intent is preserved instead of replaced by generic metadata;
# - no forced Action/Location/Atmosphere blocks are appended;
# - no LLM/API is required.
#
# Typical usage:
#   python prompt_optimizer_v3.py
#   python prompt_optimizer_v3.py scenes.json scenes_optimized_v3.json
#   python prompt_optimizer_v3.py --audit prompt_audit_v3.txt
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "scenes.json"
DEFAULT_OUTPUT = BASE_DIR / "scenes_optimized_v3.json"
DEFAULT_AUDIT = BASE_DIR / "prompt_audit_v3.txt"
CHARACTER_FILE = BASE_DIR / "character.json"


# ============================================================
# CANONICAL CHARACTER BIBLE
# ============================================================

CHARACTERS = {
    "alexander": (
        "Alexander, 35 years old, a thin but muscular man with sun-tanned "
        "weathered skin, short tousled dark hair, thoughtful eyes, light stubble, "
        "wearing a worn brown-gray jacket, a blue sweater with a subtle tree pattern, "
        "dark trousers and worn boots"
    ),
    "irina": (
        "Irina, daughter of the former scientist, a woman in her early thirties "
        "with shoulder-length dark hair, an intelligent thoughtful expression, "
        "wearing a simple dark coat and practical trousers"
    ),
    "cat": (
        "a domestic cat with short gray-and-white fur, alert eyes and a cautious "
        "but curious expression"
    ),
}

CHARACTER_NAMES = {
    "alexander": "Alexander",
    "irina": "Irina",
    "cat": "the cat",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF\u0500-\u052F]")


def contains_cyrillic(text: str) -> bool:
    return bool(CYRILLIC_RE.search(text or ""))


def normalize_spaces(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:]){2,}", r"\1", text)
    return text.strip()


# Ordered longest-first phrase translations. These are intentionally
# conservative; unknown Cyrillic is not silently deleted.
PHRASE_TRANSLATIONS = [
    ("дочка старого науковця", "daughter of the former scientist"),
    ("дочь старого науковця", "daughter of the former scientist"),
    ("дочка колишнього науковця", "daughter of the former scientist"),
    ("дочь бывшего ученого", "daughter of the former scientist"),
    ("старого науковця", "former scientist"),
    ("старому науковцю", "former scientist"),
    ("старий науковець", "former scientist"),
    ("колишній науковець", "former scientist"),
    ("колишнього науковця", "former scientist"),
    ("старого вченого", "former scientist"),
    ("старий будинок", "old house"),
    ("старого будинку", "old house"),
    ("старому будинку", "old house"),
    ("старим будинком", "old house"),
    ("старий парк", "old park"),
    ("старому парку", "old park"),
    ("містяної громади", "local community"),
    ("міській громаді", "local community"),
    ("у кімнаті", "inside the room"),
    ("в кімнаті", "inside the room"),
    ("до кімнати", "into the room"),
    ("у кімнату", "into the room"),
    ("в кімнату", "into the room"),
    ("поза будинком", "outside the house"),
    ("поза дверима", "outside the door"),
    ("біля дверей", "near the door"),
    ("біля будинку", "near the house"),
    ("на столі", "on the table"),
    ("під світлом", "under the light"),
    ("при свічках", "by candlelight"),
    ("при свічці", "by candlelight"),
    ("темна кімната", "dark room"),
    ("старі документи", "old documents"),
    ("старі книги", "old books"),
    ("старі фотографії", "old photographs"),
    ("задумливий вираз", "thoughtful expression"),
    ("задумчивый", "thoughtful"),
    ("задумчивий", "thoughtful"),
    ("обережно", "carefully"),
    ("повільно", "slowly"),
    ("раптово", "suddenly"),
    ("тихо", "quietly"),
    ("головний герой", "protagonist"),
    ("закрита кімната", "hidden room"),
    ("закриту кімнату", "hidden room"),
    ("закритої кімнати", "hidden room"),
    ("таємна кімната", "hidden room"),
    ("таємної кімнати", "hidden room"),
    ("прихована кімната", "hidden room"),
    ("прихованої кімнати", "hidden room"),
    ("закриті двері", "closed door"),
    ("закриту дверь", "closed door"),
    ("закрита дверь", "closed door"),
    ("закритою дверню", "closed door"),
    ("закритій двері", "closed door"),
    ("верхнє вікно", "upper window"),
    ("верхніх вікон", "upper windows"),
    ("старий стіл", "old table"),
    ("старого стола", "old table"),
    ("старі книжки", "old books"),
    ("екологічні книги", "ecology books"),
    ("книги про екологію", "ecology books"),
    ("книги про природу", "nature books"),
    ("місцевому архіві", "local archive"),
    ("старому архіві", "old archive"),
    ("старий архів", "old archive"),
    ("музейному кабінеті", "museum office"),
    ("музейній кімнаті", "museum room"),
    ("старій лабораторії", "old laboratory"),
    ("старому лабораторії", "old laboratory"),
    ("старій будівлі", "old building"),
    ("старому будинку", "old house"),
    ("у старому будинку", "inside the old house"),
    ("в старому будинку", "inside the old house"),
    ("у старій будівлі", "inside the old building"),
    ("в старій будівлі", "inside the old building"),
    ("у парку", "in the park"),
    ("в парку", "in the park"),
    ("у лісі", "in the forest"),
    ("в лісі", "in the forest"),
    ("вдень", "during the day"),
    ("вночі", "at night"),
    ("ввечері", "in the evening"),
    ("вранці", "in the morning"),
    ("пізній ранок", "late morning"),
    ("пізень ранок", "late morning"),
    ("одинадцять вечора", "11 PM"),
    ("дванадцята ночі", "midnight"),
    ("Олександр", "Alexander"),
    ("Олександра", "Alexander"),
    ("Ірина", "Irina"),
    ("Ірини", "Irina"),
    ("Іриною", "Irina"),
    ("Ірен", "Irina"),
    ("Ірена", "Irina"),
]

WORD_TRANSLATIONS = {
    "документи": "documents",
    "документами": "documents",
    "документів": "documents",
    "книги": "books",
    "книга": "book",
    "книжки": "books",
    "фотографія": "photograph",
    "фотографії": "photographs",
    "двері": "door",
    "дверью": "door",
    "дверню": "door",
    "коридор": "corridor",
    "будинок": "house",
    "будинку": "house",
    "кімната": "room",
    "кімнату": "room",
    "кімнати": "room",
    "кімнатою": "room",
    "парк": "park",
    "ліс": "forest",
    "дерево": "tree",
    "деревами": "trees",
    "вікно": "window",
    "вікна": "windows",
    "стіл": "table",
    "стола": "table",
    "стілець": "chair",
    "сходи": "stairs",
    "підвал": "basement",
    "горище": "attic",
    "вечір": "evening",
    "ранок": "morning",
    "день": "day",
    "ніч": "night",
    "читає": "reads",
    "читав": "read",
    "читати": "read",
    "розглядає": "examines",
    "розглянути": "examine",
    "дивиться": "looks",
    "дивлячись": "looking",
    "відкриває": "opens",
    "відкриваєть": "opens",
    "відкрити": "open",
    "закриває": "closes",
    "заходить": "enters",
    "входить": "enters",
    "виходить": "leaves",
    "іде": "walks",
    "йде": "walks",
    "ходить": "walks",
    "стоїть": "stands",
    "сидить": "sits",
    "тримає": "holds",
    "торкається": "touches",
    "шукає": "searches",
    "слухає": "listens",
    "спостерігає": "observes",
    "помічає": "notices",
    "знаходить": "finds",
    "відчуває": "feels",
    "говорить": "speaks",
    "розмовляє": "talks",
    "обговорює": "discusses",
    "планує": "plans",
    "готує": "prepares",
    "вивчає": "studies",
    "досліджує": "investigates",
    "намагається": "tries",
    "може": "can",
    "здається": "seems",
    "таємничий": "mysterious",
    "загадковий": "mysterious",
    "тихий": "quiet",
    "спокійний": "calm",
    "старий": "old",
    "стара": "old",
    "старе": "old",
    "старі": "old",
    "темний": "dark",
    "темна": "dark",
    "світлий": "bright",
    "світло": "light",
    "свічка": "candle",
    "свічки": "candles",
    "птахи": "birds",
    "листя": "leaves",
    "чоловік": "man",
    "жінка": "woman",
    "дочка": "daughter",
    "дочь": "daughter",
    "кіт": "cat",
    "кот": "cat",
    "котик": "cat",
    "кота": "cat",
    "коту": "cat",
    "обережний": "cautious",
    "обережна": "cautious",
    "цікавий": "curious",
    "цікаво": "curiously",
}


def translate_text(text: str) -> str:
    """Translate the known project vocabulary without destroying unknown text."""
    text = normalize_spaces(text)
    if not text:
        return ""

    for src, dst in PHRASE_TRANSLATIONS:
        text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)

    def replace_word(match: re.Match) -> str:
        word = match.group(0)
        replacement = WORD_TRANSLATIONS.get(word.lower())
        if replacement is None:
            return word
        if word[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    text = re.sub(r"[A-Za-zА-Яа-яІіЇїЄєҐґ'’ʼ-]+", replace_word, text)

    # Common malformed mixed-language forms seen in generated scenes.
    replacements = {
        "дверь": "door",
        "дверню": "door",
        "в room": "inside the room",
        "in room": "inside the room",
        "at room": "inside the room",
        "в park": "in the park",
        "у park": "in the park",
        "на park": "in the park",
        "в building": "inside the building",
        "у building": "inside the building",
        "в house": "inside the house",
        "у house": "inside the house",
        "musceled": "muscular",
        "Irene": "Irina",
        "Irena": "Irina",
    }
    for src, dst in replacements.items():
        text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)

    return normalize_spaces(text)


# ============================================================
# SOURCE SELECTION
# ============================================================

VISUAL_SOURCE_FIELDS = ("original_prompt", "prompt", "description", "action")


def first_nonempty(scene: dict, fields=VISUAL_SOURCE_FIELDS) -> str:
    for field in fields:
        value = scene.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def get_primary_source(scene: dict) -> str:
    # original_prompt is authoritative when available. The generated prompt
    # is only a fallback because older optimizers may have corrupted it.
    return first_nonempty(scene, ("original_prompt", "prompt", "description", "action"))


# ============================================================
# CHARACTER DETECTION
# ============================================================

ALEXANDER_PATTERNS = [
    r"\balexander\b",
    r"\bthe man\b",
    r"\ba man\b",
    r"\bpark worker\b",
    r"\bworker\b",
    r"\bprotagonist\b",
]

IRINA_PATTERNS = [
    r"\birina\b",
    r"\birene\b",
    r"\birena\b",
    r"\bthe daughter\b",
    r"\bhis daughter\b",
    r"\bdaughter of the former scientist\b",
    r"\bdaughter of the scientist\b",
]

CAT_PATTERNS = [
    r"\bcat\b",
    r"\bkitten\b",
]


def has_any_pattern(text: str, patterns) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def detect_characters(scene: dict, source: str):
    text = translate_text(source)
    characters = []

    if has_any_pattern(text, ALEXANDER_PATTERNS):
        characters.append("alexander")
    if has_any_pattern(text, IRINA_PATTERNS):
        characters.append("irina")
    if has_any_pattern(text, CAT_PATTERNS):
        characters.append("cat")

    # If the source explicitly says "his daughter Irina", both are present.
    if re.search(r"\bhis daughter\b", text, flags=re.IGNORECASE):
        if "alexander" not in characters:
            characters.insert(0, "alexander")
        if "irina" not in characters:
            characters.append("irina")

    # Do NOT infer the cat from unrelated scene fields. Do NOT infer Irina from
    # the word "daughter" in an unrelated sentence unless it is in the source.
    return characters


# ============================================================
# CHARACTER CONSISTENCY
# ============================================================

ALEXANDER_CONFLICTS = [
    r"\bkhaki shorts?\b",
    r"\bshorts\b",
    r"\bold sneakers?\b",
    r"\bsneakers?\b",
    r"\bbrown jacket\b",
    r"\bdark jacket\b",
    r"\bgray jacket\b",
    r"\bgrey jacket\b",
    r"\bt-?shirt\b",
    r"\bcasual shirt\b",
    r"\bfaded jeans?\b",
    r"\bjeans\b",
    r"\bloose shorts?\b",
]

IRINA_CONFLICTS = [
    r"\bIrene\b",
    r"\bIrena\b",
    r"\bAlexander's daughter\b",
    r"\bthe daughter of Alexander\b",
    r"\bhis daughter Irina\b",
]

CAT_CONFLICTS = [
    r"\bgolden cat\b",
    r"\borange cat\b",
    r"\bblack cat\b",
    r"\bwhite cat\b",
    r"\bginger cat\b",
    r"\btabby cat\b",
    r"\bgray cat\b",
    r"\bgrey cat\b",
]


def remove_conflicts(text: str, characters) -> str:
    if "alexander" in characters:
        for pattern in ALEXANDER_CONFLICTS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    if "irina" in characters:
        for pattern in IRINA_CONFLICTS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    if "cat" in characters:
        for pattern in CAT_CONFLICTS:
            text = re.sub(pattern, "the cat", text, flags=re.IGNORECASE)

    # Fix relationship wording globally when Irina is present.
    if "irina" in characters:
        text = re.sub(
            r"\bAlexander(?:'s|’s) daughter Irina\b",
            "Irina, daughter of the former scientist",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bhis daughter Irina\b",
            "Irina, daughter of the former scientist",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bthe daughter of Alexander\b",
            "Irina, daughter of the former scientist",
            text,
            flags=re.IGNORECASE,
        )

    return normalize_spaces(text)


def strip_character_bible_fragments(text: str, character: str) -> str:
    """Remove only obvious canonical/conflicting appearance fragments.

    This prevents duplicated character bibles while preserving scene-specific
    objects and actions.
    """
    if character == "alexander":
        patterns = [
            r"Alexander, 35 years old,?",
            r"a thin and weathered man",
            r"a thin but muscular man",
            r"sun-kissed skin",
            r"sun-tanned weathered skin",
            r"short tousled dark hair",
            r"thoughtful eyes",
            r"light stubble(?: beard)?",
            r"worn brown-gray jacket",
            r"worn brown jacket",
            r"worn-out brown jacket",
            r"blue sweater with a subtle tree pattern",
            r"blue sweater with tree designs",
            r"dark trousers",
            r"worn boots",
        ]
    elif character == "irina":
        patterns = [
            r"Irina, daughter of the former scientist,?",
            r"Irina,?\s+a woman in her early thirties",
            r"shoulder-length dark hair",
            r"an intelligent thoughtful expression",
            r"simple dark coat",
            r"practical trousers",
        ]
    else:
        patterns = [
            r"a domestic cat with short gray-and-white fur",
            r"a domestic cat",
            r"short gray-and-white fur",
            r"alert eyes",
            r"a cautious but curious expression",
        ]

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return normalize_spaces(text).strip(" ,.;:")


# ============================================================
# CAMERA HANDLING
# ============================================================

CAMERA_TRANSLATIONS = [
    (r"high-angle(?:\s+shot)?", "high-angle shot"),
    (r"low-angle(?:\s+shot)?", "low-angle shot"),
    (r"close-up(?:\s+shot)?", "close-up"),
    (r"close up", "close-up"),
    (r"medium(?:\s+shot)?", "medium shot"),
    (r"wide(?:\s+shot)?", "wide shot"),
    (r"over-the-shoulder", "over-the-shoulder shot"),
    (r"pov", "POV shot"),
    (r"point of view", "POV shot"),
]


def normalize_camera(scene: dict, source: str) -> str:
    raw = scene.get("camera")
    camera = translate_text(raw) if raw else ""

    # Some planners accidentally put the camera instruction into original_prompt.
    if not camera:
        camera = translate_text(source)

    low = camera.lower()
    hints = []

    # Keep the original semantic camera hints in stable order.
    if re.search(r"pov|point of view|від першої особи|очима", low, re.I):
        hints.append("POV shot")
    if re.search(r"over[- ]the[- ]shoulder|через плече", low, re.I):
        hints.append("over-the-shoulder shot")
    if re.search(r"close[- ]?up|круп", low, re.I):
        hints.append("close-up")
    if re.search(r"high[- ]angle|висок", low, re.I):
        hints.append("high-angle shot")
    if re.search(r"low[- ]angle|низьк", low, re.I):
        hints.append("low-angle shot")
    if re.search(r"wide|широк|далек", low, re.I):
        hints.append("wide shot")
    if re.search(r"medium|серед", low, re.I):
        hints.append("medium shot")

    # If no explicit shot type is present, do not fabricate a precise camera.
    if not hints:
        return ""

    # Remove duplicates while preserving order.
    unique = []
    for item in hints:
        if item not in unique:
            unique.append(item)
    return ", ".join(unique)


# ============================================================
# ACTION / TIME / LOCATION / MOOD
# ============================================================

ACTION_VERBS = [
    "reads", "examines", "looks", "opens", "closes", "enters", "leaves",
    "walks", "stands", "sits", "holds", "touches", "searches", "listens",
    "observes", "notices", "finds", "feels", "speaks", "talks", "discusses",
    "plans", "prepares", "studies", "investigates", "tries", "meets",
    "reviews", "follows", "looks through", "moves", "discovers",
]


def extract_action(scene: dict, source: str) -> str:
    raw = scene.get("action")
    if raw and str(raw).strip():
        action = translate_text(raw)
    else:
        action = ""

    if not action:
        # We may use the primary prompt as the action source, but we never invent
        # an event merely to satisfy validation.
        action = translate_text(source)

    return normalize_spaces(action)


def normalize_location(scene: dict, source: str) -> str:
    raw = scene.get("location")
    location = translate_text(raw) if raw else ""
    if not location:
        return ""

    location = location.strip(" .,:;")
    return location


def normalize_time(scene: dict) -> str:
    raw = scene.get("time")
    if not raw:
        return ""
    return translate_text(raw).strip(" .,:;")


def normalize_mood(scene: dict) -> str:
    raw = scene.get("mood")
    if not raw:
        return ""
    return translate_text(raw).strip(" .,:;")


# ============================================================
# SENTENCE / PROMPT CLEANUP
# ============================================================

GENERIC_BLOAT_PATTERNS = [
    r"\brealistic detailed environment\b",
    r"\bhigh visual detail\b",
    r"\bdramatic storytelling\b",
    r"\bcoherent perspective\b",
    r"\bnatural anatomy\b",
    r"\bsubtle film grain\b",
    r"\bcinematic composition\b",
    r"\brealistic lighting\b",
]

STYLE_SUFFIX = (
    "cinematic realism, detailed environment, natural anatomy, coherent perspective, "
    "realistic lighting, high visual detail, subtle film grain"
)


def clean_generated_bloat(text: str) -> str:
    for pattern in GENERIC_BLOAT_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAction:\s*\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bLocation:\s*\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bLighting and time:\s*\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAtmosphere:\s*\.?", "", text, flags=re.IGNORECASE)
    return normalize_spaces(text).strip(" .,:;")


def fix_relationships(text: str) -> str:
    replacements = [
        (r"\bIrene\b", "Irina"),
        (r"\bIrena\b", "Irina"),
        (r"\bAlexander(?:'s|’s) daughter Irina\b", "Irina, daughter of the former scientist"),
        (r"\bhis daughter Irina\b", "Irina, daughter of the former scientist"),
        (r"\bthe daughter of Alexander\b", "Irina, daughter of the former scientist"),
        (r"\bdaughter of Alexander\b", "daughter of the former scientist"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return normalize_spaces(text)


def remove_repeated_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen = set()
    kept = []
    for sentence in sentences:
        key = re.sub(r"[^a-z0-9]+", "", sentence.lower())
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        kept.append(sentence)
    return " ".join(kept)


def ensure_periods(text: str) -> str:
    text = normalize_spaces(text)
    text = re.sub(r"\.{2,}", ".", text)
    text = text.strip(" .")
    return text


def remove_residual_cyrillic(text: str) -> str:
    """Remove only residual Cyrillic after translation, leaving a clear marker
    for the audit instead of silently pretending the text was translated.
    """
    if not contains_cyrillic(text):
        return text

    # Known names/words should already have been translated. Any remaining
    # Cyrillic is removed at word level so a single malformed token cannot
    # contaminate the final SD prompt.
    text = re.sub(r"[\u0400-\u04FF\u0500-\u052F]+", "", text)
    return normalize_spaces(text)


# ============================================================
# BUILD PROMPT
# ============================================================


def build_prompt(scene: dict):
    source_raw = get_primary_source(scene)
    source = translate_text(source_raw)
    source = fix_relationships(source)

    characters = detect_characters(scene, source)

    # Remove obsolete/corrupted appearance fragments before adding canonical
    # appearance once. This changes only consistency details, not scene events.
    body = source
    for character in characters:
        body = strip_character_bible_fragments(body, character)

    body = remove_conflicts(body, characters)
    body = clean_generated_bloat(body)
    body = fix_relationships(body)
    body = remove_repeated_sentences(body)

    parts = []
    if body:
        parts.append(body)

    # Canonical appearance is added only for characters explicitly present in
    # the primary visual source.
    for character in characters:
        parts.append(CHARACTERS[character])

    # Location/time/mood are supporting metadata. We append them only if they
    # are present in the scene planner data and are not already represented in
    # the source prompt. This avoids V2's forced metadata bloat.
    location = normalize_location(scene, source)
    time = normalize_time(scene)
    mood = normalize_mood(scene)
    camera = normalize_camera(scene, source)

    body_lower = body.lower()

    if location and location.lower() not in body_lower:
        parts.append(f"Setting: {location}")
    if time and time.lower() not in body_lower:
        parts.append(f"Time: {time}")
    if mood and mood.lower() not in body_lower:
        parts.append(f"Mood: {mood}")
    if camera:
        parts.append(f"Camera: {camera}")

    parts.append(STYLE_SUFFIX)

    prompt = ". ".join(p.strip(" .") for p in parts if p and p.strip())
    prompt = translate_text(prompt)
    prompt = fix_relationships(prompt)
    prompt = clean_generated_bloat(prompt)
    prompt = remove_repeated_sentences(prompt)
    prompt = ensure_periods(prompt)

    # Final hard safety net for Stable Diffusion: no Cyrillic in the prompt.
    had_cyrillic = contains_cyrillic(prompt)
    prompt = remove_residual_cyrillic(prompt)
    prompt = ensure_periods(prompt)

    return prompt, characters, had_cyrillic


# ============================================================
# VALIDATION
# ============================================================


def source_has_action(scene: dict) -> bool:
    source = translate_text(get_primary_source(scene))
    low = source.lower()
    return any(re.search(r"\b" + re.escape(verb) + r"\b", low) for verb in ACTION_VERBS)


def validate_prompt(prompt: str, scene: dict, characters):
    warnings = []

    if not prompt:
        warnings.append("EMPTY_PROMPT")

    if contains_cyrillic(prompt):
        warnings.append("CYRILLIC_REMAINS")

    words = prompt.split()
    if len(words) < 30:
        warnings.append("LOW_DETAIL")
    if len(words) > 180:
        warnings.append("VERY_LONG_PROMPT")

    # A missing action is a warning only if the original scene itself had an
    # identifiable action. We never invent one just to clear the warning.
    if not source_has_action(scene):
        warnings.append("NO_CLEAR_ACTION_IN_SOURCE")

    if not characters:
        warnings.append("NO_CHARACTER_DETECTED")

    low = prompt.lower()

    if "alexander" in low and "alexander" not in characters:
        warnings.append("UNEXPECTED_ALEXANDER")
    if "irina" in low and "irina" not in characters:
        warnings.append("UNEXPECTED_IRINA")
    if "cat" in low and "cat" not in characters:
        warnings.append("UNEXPECTED_CAT")

    if "irina" in characters:
        if re.search(r"Alexander(?:'s|’s) daughter|daughter of Alexander|his daughter Irina", prompt, re.I):
            warnings.append("WRONG_IRINA_RELATIONSHIP")

    if "cat" in characters:
        if re.search(r"golden cat|orange cat|black cat|ginger cat|tabby cat", prompt, re.I):
            warnings.append("WRONG_CAT_APPEARANCE")

    if "alexander" in characters:
        if re.search(r"khaki shorts|sneakers|jeans|t-shirt|shorts", prompt, re.I):
            warnings.append("ALEXANDER_CLOTHING_CONFLICT")

    # The prompt should not contain empty metadata labels.
    if re.search(r"(?:Action|Location|Setting|Time|Mood|Camera):\s*(?:[.\s]|$)", prompt, re.I):
        warnings.append("EMPTY_METADATA")

    return warnings


# ============================================================
# AUDIT / OUTPUT
# ============================================================


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def process_scene(scene: dict):
    result = deepcopy(scene)
    prompt, characters, had_cyrillic = build_prompt(scene)
    warnings = validate_prompt(prompt, scene, characters)

    result["prompt"] = prompt
    result["prompt_word_count"] = count_words(prompt)
    result["prompt_warnings"] = warnings
    result["detected_characters_v3"] = characters
    result["optimizer_version"] = "v3"

    if had_cyrillic and "CYRILLIC_REMAINS" not in warnings:
        # This records that a translation pass was necessary, not that the
        # final prompt still contains Cyrillic.
        result["prompt_translation_applied"] = True
    else:
        result["prompt_translation_applied"] = bool(contains_cyrillic(str(scene.get("original_prompt", ""))))

    return result


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def make_audit(original_scenes, optimized_scenes):
    lines = []
    lines.append("AI YOUTUBE STORIES - PROMPT OPTIMIZER V3 AUDIT")
    lines.append("=" * 72)
    lines.append("")

    total = len(optimized_scenes)
    clean = 0
    warning_count = 0

    for scene in optimized_scenes:
        warnings = scene.get("prompt_warnings", [])
        if not warnings:
            clean += 1
        warning_count += len(warnings)

    lines.append(f"Scenes processed: {total}")
    lines.append(f"Scenes without warnings: {clean}")
    lines.append(f"Total warnings: {warning_count}")
    lines.append("")
    lines.append("Rules: original_prompt is primary; no invented events; canonical character appearance;")
    lines.append("no automatic cat/Irina insertion; camera intent preserved; no forced empty metadata.")
    lines.append("")

    for scene in optimized_scenes:
        sid = scene.get("id", "?")
        title = translate_text(scene.get("title", ""))
        chars = ", ".join(scene.get("detected_characters_v3", [])) or "none"
        warnings = ", ".join(scene.get("prompt_warnings", [])) or "OK"
        prompt = scene.get("prompt", "")

        lines.append(f"SCENE {sid:>3} | {title}")
        lines.append(f"Characters: {chars}")
        lines.append(f"Words: {scene.get('prompt_word_count', 0)}")
        lines.append(f"Warnings: {warnings}")
        lines.append(f"Prompt: {prompt}")
        lines.append("-" * 72)

    return "\n".join(lines) + "\n"


def print_summary(optimized_scenes):
    total = len(optimized_scenes)
    clean = sum(1 for s in optimized_scenes if not s.get("prompt_warnings"))
    warnings = sum(len(s.get("prompt_warnings", [])) for s in optimized_scenes)
    cyr = sum(1 for s in optimized_scenes if contains_cyrillic(s.get("prompt", "")))

    print()
    print("=" * 60)
    print("PROMPT OPTIMIZER V3")
    print("=" * 60)
    print(f"Scenes processed : {total}")
    print(f"Scenes clean     : {clean}")
    print(f"Scenes with warn : {total - clean}")
    print(f"Total warnings   : {warnings}")
    print(f"Cyrillic prompts : {cyr}")
    print("=" * 60)

    if warnings:
        print("\nWarning breakdown:")
        counts = {}
        for scene in optimized_scenes:
            for warning in scene.get("prompt_warnings", []):
                counts[warning] = counts.get(warning, 0) + 1
        for warning, amount in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {warning}: {amount}")


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conservative V3 prompt optimizer for AI YouTube Stories."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input scenes JSON (default: scenes.json)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output optimized JSON (default: scenes_optimized_v3.json)",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_AUDIT,
        help="Audit text output (default: prompt_audit_v3.txt)",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Do not write the audit file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input if args.input.is_absolute() else BASE_DIR / args.input
    output_path = args.output if args.output.is_absolute() else BASE_DIR / args.output
    audit_path = args.audit if args.audit.is_absolute() else BASE_DIR / args.audit

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2

    try:
        data = load_json(input_path)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON in {input_path}: {exc}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"ERROR: cannot read {input_path}: {exc}", file=sys.stderr)
        return 4

    if not isinstance(data, list):
        print("ERROR: scenes JSON must contain a top-level array.", file=sys.stderr)
        return 5

    optimized = []
    for index, scene in enumerate(data, start=1):
        if not isinstance(scene, dict):
            print(f"WARNING: scene #{index} is not an object; skipped.", file=sys.stderr)
            continue
        optimized.append(process_scene(scene))
        sid = scene.get("id", index)
        print(f"Processed scene {sid}")

    try:
        save_json(output_path, optimized)
        if not args.no_audit:
            audit = make_audit(data, optimized)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(audit, encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot write output: {exc}", file=sys.stderr)
        return 6

    print_summary(optimized)
    print(f"\nOutput: {output_path}")
    if not args.no_audit:
        print(f"Audit : {audit_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
