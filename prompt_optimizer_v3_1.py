import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

# ============================================================
# AI YOUTUBE STORIES - PROMPT OPTIMIZER V3.1
# ============================================================
# V3.1 fixes the main V3 failure mode: trusting a corrupted
# original_prompt over the structured scene facts.
#
# Priority:
#   1. action + description  -> story facts
#   2. camera                -> shot intent
#   3. location/time/mood     -> supporting context
#   4. original_prompt        -> fallback/style hints only
#
# The optimizer is deliberately deterministic and does NOT call an LLM.
# It never invents a room, object, animal, person, event or relationship.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "scenes.json"
DEFAULT_OUTPUT = BASE_DIR / "scenes_optimized_v3_1.json"
DEFAULT_AUDIT = BASE_DIR / "prompt_audit_v3_1.txt"

CHARACTERS = {
    "alexander": (
        "Alexander, 35 years old, a thin but muscular man with sun-tanned weathered skin, "
        "short tousled dark hair, thoughtful eyes, light stubble, wearing a worn brown-gray jacket, "
        "a blue sweater with a subtle tree pattern, dark trousers and worn boots"
    ),
    "irina": (
        "Irina, daughter of the former scientist, a woman in her early thirties with shoulder-length "
        "dark hair, an intelligent thoughtful expression, wearing a simple dark coat and practical trousers"
    ),
    "cat": (
        "a domestic cat with short gray-and-white fur, alert eyes and a cautious but curious expression"
    ),
}

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF\u0500-\u052F]")

# Longest phrases first. These cover the vocabulary actually appearing in the project.
PHRASES = [
    ("дочка колишнього науковця", "daughter of the former scientist"),
    ("дочка старого науковця", "daughter of the former scientist"),
    ("дочь бывшего ученого", "daughter of the former scientist"),
    ("дочь старого ученого", "daughter of the former scientist"),
    ("колишнього науковця", "former scientist"),
    ("старого науковця", "former scientist"),
    ("старого вченого", "former scientist"),
    ("старий науковець", "former scientist"),
    ("старий будинок", "old house"),
    ("старого будинку", "old house"),
    ("старому будинку", "old house"),
    ("у старому будинку", "inside the old house"),
    ("в старому будинку", "inside the old house"),
    ("старій будівлі", "old building"),
    ("у старій будівлі", "inside the old building"),
    ("в старій будівлі", "inside the old building"),
    ("старий парк", "old park"),
    ("старому парку", "old park"),
    ("у старому парку", "in the old park"),
    ("в старому парку", "in the old park"),
    ("містяної громади", "local community"),
    ("міській громаді", "local community"),
    ("місцевому архіві", "local archive"),
    ("старому архіві", "old archive"),
    ("музейному кабінеті", "museum office"),
    ("музейній кімнаті", "museum room"),
    ("старій лабораторії", "old laboratory"),
    ("у старій лабораторії", "inside the old laboratory"),
    ("в старій лабораторії", "inside the old laboratory"),
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
    ("верхнє вікно", "upper window"),
    ("верхніх вікон", "upper windows"),
    ("старі документи", "old documents"),
    ("старі книги", "old books"),
    ("старі книжки", "old books"),
    ("книги про екологію", "ecology books"),
    ("книги про природу", "nature books"),
    ("екологічні книги", "ecology books"),
    ("старий стіл", "old table"),
    ("старого стола", "old table"),
    ("під світлом", "under the light"),
    ("при свічках", "by candlelight"),
    ("при свічці", "by candlelight"),
    ("задумливий вираз", "thoughtful expression"),
    ("задумчивий", "thoughtful"),
    ("задумливий", "thoughtful"),
    ("пізень ранок", "late morning"),
    ("пізній ранок", "late morning"),
    ("одинадцять вечора", "11 PM"),
    ("дванадцята ночі", "midnight"),
    ("вранці", "in the morning"),
    ("ввечері", "in the evening"),
    ("вночі", "at night"),
    ("вдень", "during the day"),
    ("біля дверей", "near the door"),
    ("біля будинку", "near the house"),
    ("поза будинком", "outside the house"),
    ("поза дверима", "outside the door"),
    ("на столі", "on the table"),
    ("у кімнаті", "inside the room"),
    ("в кімнаті", "inside the room"),
    ("до кімнати", "into the room"),
    ("у кімнату", "into the room"),
    ("в кімнату", "into the room"),
    ("у парку", "in the park"),
    ("в парку", "in the park"),
    ("у лісі", "in the forest"),
    ("в лісі", "in the forest"),
    ("Олександр", "Alexander"),
    ("Олександра", "Alexander"),
    ("Ірина", "Irina"),
    ("Ірини", "Irina"),
    ("Іриною", "Irina"),
    ("Ірен", "Irina"),
    ("Ірена", "Irina"),
]

WORDS = {
    "документи": "documents", "документами": "documents", "документів": "documents",
    "книги": "books", "книга": "book", "книжки": "books",
    "фотографія": "photograph", "фотографії": "photographs", "фото": "photograph",
    "двері": "door", "дверю": "door", "дверью": "door", "дверню": "door", "дверь": "door",
    "коридор": "corridor", "будинок": "house", "будинку": "house", "кімната": "room",
    "кімнату": "room", "кімнати": "room", "кімнатою": "room", "парк": "park", "ліс": "forest",
    "дерево": "tree", "деревами": "trees", "вікно": "window", "вікна": "windows",
    "стіл": "table", "стола": "table", "стілець": "chair", "сходи": "stairs",
    "підвал": "basement", "горище": "attic", "вечір": "evening", "ранок": "morning",
    "день": "day", "ніч": "night", "світло": "light", "свічка": "candle", "свічки": "candles",
    "птахи": "birds", "листя": "leaves", "чоловік": "man", "жінка": "woman",
    "дочка": "daughter", "дочь": "daughter", "кіт": "cat", "кот": "cat", "котик": "cat",
    "кота": "cat", "коту": "cat", "обережний": "cautious", "обережна": "cautious",
    "цікавий": "curious", "цікаво": "curiously", "старий": "old", "стара": "old", "старе": "old",
    "старі": "old", "темний": "dark", "темна": "dark", "світлий": "bright",
    "читає": "reads", "читав": "read", "читати": "read", "розглядає": "examines",
    "розглянути": "examine", "дивиться": "looks", "дивлячись": "looking", "відкриває": "opens",
    "відкрити": "open", "закриває": "closes", "заходить": "enters", "входить": "enters",
    "виходить": "leaves", "іде": "walks", "йде": "walks", "ходить": "walks", "стоїть": "stands",
    "сидить": "sits", "тримає": "holds", "торкається": "touches", "шукає": "searches",
    "слухає": "listens", "спостерігає": "observes", "помічає": "notices", "знаходить": "finds",
    "відчуває": "feels", "говорить": "speaks", "розмовляє": "talks", "обговорює": "discusses",
    "планує": "plans", "готує": "prepares", "вивчає": "studies", "досліджує": "investigates",
    "намагається": "tries", "зустрічає": "meets", "знайшов": "found", "знайдена": "found",
    "показує": "shows", "бачить": "sees", "починає": "begins", "продовжує": "continues",
    "таємничий": "mysterious", "таємнича": "mysterious", "загадковий": "mysterious",
    "загадкова": "mysterious", "тихий": "quiet", "тиха": "quiet", "спокійний": "calm",
    "спокійна": "calm", "напружена": "tense", "напружений": "tense", "уважно": "carefully",
    "обережно": "carefully", "повільно": "slowly", "раптово": "suddenly", "головний": "main",
    "герой": "character", "музей": "museum", "архів": "archive", "лабораторія": "laboratory",
    "кабінет": "office", "будівля": "building", "кімнатою": "room", "фонарик": "flashlight",
    "ключ": "key", "креслам": "drawings", "стіні": "wall", "стіна": "wall", "ручку": "handle",
    "ручка": "handle", "саквояж": "satchel", "тетрадь": "notebook", "записи": "notes",
    "рисунки": "drawings", "шепот": "whisper", "шепіт": "whisper", "шуму": "noise",
    "звук": "sound", "джерело": "source", "портретна": "portrait", "портрету": "portrait",
}

CAMERA_PHRASES = [
    (r"pov|point of view|очима|від першої особи", "POV shot"),
    (r"over[- ]the[- ]shoulder|через плече", "over-the-shoulder shot"),
    (r"close[- ]?up|крупн", "close-up"),
    (r"high[- ]angle|high angle|висок", "high-angle shot"),
    (r"low[- ]angle|low angle|низьк", "low-angle shot"),
    (r"wide|широк|далек", "wide shot"),
    (r"medium|серед", "medium shot"),
    (r"tracking|слідку|просліду", "tracking shot"),
    (r"detail|детал", "detail shot"),
]

BAD_ORIGINAL_PATTERNS = [
    r"\bkhaki shorts?\b", r"\bshorts\b", r"\bold sneakers?\b", r"\bsneakers?\b",
    r"\bjeans\b", r"\bt-?shirt\b", r"\bwearing a,?\s*(?:and)?\b",
    r"\bfaded photographs?\b", r"\bholding an old photograph of a room\b",
    r"\bimpossible dark corridor\b", r"\bhis cat\b", r"\bgolden cat\b", r"\borange cat\b",
]

STYLE_SUFFIX = "cinematic realism, natural anatomy, coherent perspective, realistic lighting, detailed environment, subtle film grain"


def clean(text):
    text = str(text or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:]){2,}", r"\1", text)
    return text.strip()


def translate(text):
    text = clean(text)
    if not text:
        return ""
    for src, dst in PHRASES:
        text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)

    def repl(m):
        w = m.group(0)
        r = WORDS.get(w.lower())
        if r is None:
            return w
        return r.capitalize() if w[:1].isupper() else r

    text = re.sub(r"[A-Za-zА-Яа-яІіЇїЄєҐґ'’ʼ-]+", repl, text)
    replacements = {
        "в park": "in the park", "у park": "in the park", "на park": "in the park",
        "в building": "inside the building", "у building": "inside the building",
        "в house": "inside the house", "у house": "inside the house",
        "in room": "inside the room", "в room": "inside the room", "at room": "inside the room",
        "Irene": "Irina", "Irena": "Irina", "musceled": "muscular",
    }
    for src, dst in replacements.items():
        text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)
    return clean(text)


def normalize_relationships(text):
    text = re.sub(r"\bIrene\b|\bIrena\b", "Irina", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAlexander(?:'s|’s) daughter Irina\b", "Irina, daughter of the former scientist", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhis daughter Irina\b", "Irina, daughter of the former scientist", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe daughter of Alexander\b", "Irina, daughter of the former scientist", text, flags=re.IGNORECASE)
    return clean(text)


def present(scene):
    # Character presence is detected ONLY from semantic scene fields, never from
    # the old optimizer output and never from generic words such as 'his'.
    source = " ".join(translate(scene.get(k, "")) for k in ("description", "action", "camera"))
    low = source.lower()
    chars = []
    if re.search(r"\balexander\b|\bthe man\b|\bprotagonist\b|\bworker\b", low):
        chars.append("alexander")
    if re.search(r"\birina\b|\birene\b|\birena\b|\bhis daughter\b|\bthe daughter\b|\bdaughter of the former scientist\b", low):
        chars.append("irina")
    if re.search(r"\bcat\b|\bkitten\b", low):
        chars.append("cat")
    return chars


def camera(scene):
    raw = translate(scene.get("camera", ""))
    found = []
    for pattern, value in CAMERA_PHRASES:
        if re.search(pattern, raw, re.IGNORECASE) and value not in found:
            found.append(value)
    return ", ".join(found)


def strip_bible(text, character):
    patterns = {
        "alexander": [
            r"Alexander,?\s*35 years old,?", r"thin and weathered man", r"thin but muscular man",
            r"sun-kissed skin", r"sun-tanned weathered skin", r"short tousled dark hair", r"thoughtful eyes",
            r"light stubble(?: beard)?", r"worn brown-gray jacket", r"worn brown jacket", r"worn-out brown jacket",
            r"blue sweater with a subtle tree pattern", r"blue sweater with tree designs", r"dark trousers", r"worn boots",
        ],
        "irina": [
            r"Irina,?\s*(?:daughter of the former scientist,?)?\s*a woman in her early thirties",
            r"shoulder-length dark hair", r"an intelligent thoughtful expression", r"simple dark coat", r"practical trousers",
        ],
        "cat": [
            r"a domestic cat with short gray-and-white fur", r"a domestic cat", r"short gray-and-white fur",
            r"alert eyes", r"a cautious but curious expression",
        ],
    }
    for p in patterns.get(character, []):
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return clean(text).strip(" ,.;:")


def split_sentences(text):
    text = clean(text)
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def dedupe(text):
    seen = set()
    out = []
    for s in split_sentences(text):
        key = re.sub(r"[^a-z0-9]+", "", s.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return " ".join(out)


def remove_bad_original(text):
    for p in BAD_ORIGINAL_PATTERNS:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return clean(text)


def build_prompt(scene):
    chars = present(scene)

    # The structured scene is authoritative. Description and action are joined
    # because either one may contain facts missing from the other.
    description = normalize_relationships(translate(scene.get("description", "")))
    action = normalize_relationships(translate(scene.get("action", "")))
    structured = dedupe(" ".join(x for x in (description, action) if x))

    # original_prompt is allowed to contribute ONLY when it contains information
    # that is not contradicted by structured fields. We deliberately do not use it
    # as the scene body because V3 proved it can be semantically corrupted.
    original = normalize_relationships(translate(scene.get("original_prompt", "")))
    original = remove_bad_original(original)

    # Add safe original details only if they mention an already-established object
    # or event. This keeps the optimizer conservative rather than hallucinating.
    safe_original_bits = []
    if original:
        low_struct = structured.lower()
        object_words = [
            "door", "key", "book", "books", "documents", "photograph", "photographs", "notebook",
            "flashlight", "handle", "table", "window", "corridor", "laboratory", "archive", "museum",
            "park", "house", "room", "cat", "candle", "candles", "satchel",
        ]
        for sentence in split_sentences(original):
            low = sentence.lower()
            if any(w in low_struct and w in low for w in object_words):
                safe_original_bits.append(sentence)
    if safe_original_bits:
        structured = dedupe(structured + " " + " ".join(safe_original_bits))

    # Remove appearance fragments before adding exactly one canonical bible.
    body = structured
    for ch in chars:
        body = strip_bible(body, ch)
    body = normalize_relationships(body)

    parts = []
    if body:
        parts.append(body)
    for ch in chars:
        parts.append(CHARACTERS[ch])

    location = translate(scene.get("location", "")).strip(" .,:;")
    time = translate(scene.get("time", "")).strip(" .,:;")
    mood = translate(scene.get("mood", "")).strip(" .,:;")
    shot = camera(scene)

    body_low = body.lower()
    if location and location.lower() not in body_low:
        parts.append(f"Setting: {location}")
    if time and time.lower() not in body_low:
        parts.append(f"Time: {time}")
    if mood and mood.lower() not in body_low:
        parts.append(f"Mood: {mood}")
    if shot:
        parts.append(f"Camera: {shot}")
    parts.append(STYLE_SUFFIX)

    prompt = ". ".join(p.strip(" .") for p in parts if p and p.strip())
    prompt = normalize_relationships(clean(prompt))
    prompt = re.sub(r"\b(?:Action|Location|Mood|Time|Camera):\s*(?:\.|$)", "", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\s+", " ", prompt).strip(" .")
    had_cyr = bool(CYRILLIC_RE.search(prompt))
    prompt = CYRILLIC_RE.sub("", prompt)
    prompt = clean(prompt).strip(" .")
    return prompt, chars, had_cyr


def validate(prompt, scene, chars):
    warnings = []
    if not prompt:
        warnings.append("EMPTY_PROMPT")
    if CYRILLIC_RE.search(prompt):
        warnings.append("CYRILLIC_REMAINS")
    words = prompt.split()
    if len(words) < 35:
        warnings.append("LOW_DETAIL")
    if len(words) > 180:
        warnings.append("VERY_LONG_PROMPT")
    if not scene.get("description") and not scene.get("action"):
        warnings.append("NO_SCENE_FACTS")
    low = prompt.lower()
    if "alexander" in low and "alexander" not in chars:
        warnings.append("UNEXPECTED_ALEXANDER")
    if "irina" in low and "irina" not in chars:
        warnings.append("UNEXPECTED_IRINA")
    if "cat" in low and "cat" not in chars:
        warnings.append("UNEXPECTED_CAT")
    if "irina" in chars and re.search(r"Alexander(?:'s|’s) daughter|daughter of Alexander|his daughter Irina", prompt, re.I):
        warnings.append("WRONG_IRINA_RELATIONSHIP")
    if "cat" in chars and re.search(r"golden cat|orange cat|black cat|ginger cat|tabby cat|white cat", prompt, re.I):
        warnings.append("WRONG_CAT_APPEARANCE")
    if "alexander" in chars and re.search(r"khaki shorts|sneakers|jeans|t-shirt|shorts", prompt, re.I):
        warnings.append("ALEXANDER_CLOTHING_CONFLICT")
    if re.search(r"wearing a\s*,|\band\s*,|Mood:\s*\.?|Setting:\s*\.?|Time:\s*\.?|Camera:\s*\.?", prompt, re.I):
        warnings.append("MALFORMED_METADATA")
    return warnings


def count_words(text):
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def process(scene):
    result = deepcopy(scene)
    prompt, chars, had_cyr = build_prompt(scene)
    warnings = validate(prompt, scene, chars)
    result["prompt"] = prompt
    result["prompt_word_count"] = count_words(prompt)
    result["prompt_warnings"] = warnings
    result["detected_characters_v3_1"] = chars
    result["optimizer_version"] = "v3.1"
    result["prompt_translation_applied"] = had_cyr or bool(CYRILLIC_RE.search(str(scene.get("description", "")) + str(scene.get("action", ""))))
    return result


def load(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def audit(data):
    total = len(data)
    clean_count = sum(not s.get("prompt_warnings") for s in data)
    warning_count = sum(len(s.get("prompt_warnings", [])) for s in data)
    lines = [
        "AI YOUTUBE STORIES - PROMPT OPTIMIZER V3.1 AUDIT",
        "=" * 72,
        "",
        f"Scenes processed: {total}",
        f"Scenes without warnings: {clean_count}",
        f"Total warnings: {warning_count}",
        "",
        "V3.1 policy:",
        "- structured description + action are authoritative scene facts",
        "- camera is read from the dedicated camera field",
        "- original_prompt is fallback/style input only and cannot override scene facts",
        "- no invented people, animals, rooms, objects, events or relationships",
        "- canonical Alexander / Irina / cat appearance is applied only when present",
        "- Irina is never Alexander's daughter",
        "- final Stable Diffusion prompt contains no Cyrillic",
        "",
    ]
    for s in data:
        sid = s.get("id", "?")
        title = translate(s.get("title", ""))
        chars = ", ".join(s.get("detected_characters_v3_1", [])) or "none"
        warns = ", ".join(s.get("prompt_warnings", [])) or "OK"
        lines += [
            f"SCENE {sid:>3} | {title}",
            f"Characters: {chars}",
            f"Words: {s.get('prompt_word_count', 0)}",
            f"Warnings: {warns}",
            f"Prompt: {s.get('prompt', '')}",
            "-" * 72,
        ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Conservative V3.1 prompt optimizer")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--no-audit", action="store_true")
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else BASE_DIR / args.input
    output_path = args.output if args.output.is_absolute() else BASE_DIR / args.output
    audit_path = args.audit if args.audit.is_absolute() else BASE_DIR / args.audit

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2
    try:
        scenes = load(input_path)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON: {exc}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"ERROR: cannot read input: {exc}", file=sys.stderr)
        return 4

    if not isinstance(scenes, list):
        print("ERROR: scenes JSON must contain a top-level array.", file=sys.stderr)
        return 5

    optimized = []
    for i, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            print(f"WARNING: scene #{i} is not an object; skipped.", file=sys.stderr)
            continue
        optimized.append(process(scene))
        print(f"Processed scene {scene.get('id', i)}")

    try:
        save(output_path, optimized)
        if not args.no_audit:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(audit(optimized), encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot write output: {exc}", file=sys.stderr)
        return 6

    total = len(optimized)
    clean_count = sum(not s.get("prompt_warnings") for s in optimized)
    warning_count = sum(len(s.get("prompt_warnings", [])) for s in optimized)
    cyr_count = sum(bool(CYRILLIC_RE.search(s.get("prompt", ""))) for s in optimized)
    print()
    print("=" * 60)
    print("PROMPT OPTIMIZER V3.1")
    print("=" * 60)
    print(f"Scenes processed : {total}")
    print(f"Scenes clean     : {clean_count}")
    print(f"Scenes with warn : {total - clean_count}")
    print(f"Total warnings   : {warning_count}")
    print(f"Cyrillic prompts : {cyr_count}")
    print("=" * 60)
    if warning_count:
        counts = {}
        for s in optimized:
            for w in s.get("prompt_warnings", []):
                counts[w] = counts.get(w, 0) + 1
        print("\nWarning breakdown:")
        for w, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {w}: {n}")
    print(f"\nOutput: {output_path}")
    if not args.no_audit:
        print(f"Audit : {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
