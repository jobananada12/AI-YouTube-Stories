import argparse
import json
import re
from pathlib import Path

from core.llm import OllamaClient

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "scenes.json"
DEFAULT_OUTPUT = BASE_DIR / "scenes_optimized_v3_4_2.json"
DEFAULT_AUDIT = BASE_DIR / "prompt_audit_v3_4_2.txt"
MAX_WORDS = 58
MIN_WORDS = 25
CYRILLIC_RE = re.compile(r"[\u0400-\u052F]")

# Conservative normalization of known machine-translation errors.
NORMALIZATION = {
    "серовиртній": "сіруватий",
    "серовиртна": "сірувата",
    "серовиртний": "сіруватий",
    "серовиртні": "сіруваті",
    "дверь": "двері",
    "способы вхіду": "способи входу",
    "способы входу": "способи входу",
    "верхніх вікон": "верхнє вікно",
    "одна з верхніх вікон": "одне з верхніх вікон",
    "обокладують до кімнати": "залазить до кімнати",
    "обокладують": "залазить",
    "розкривають це вікно": "відкриває це вікно",
    "вони бачать": "він бачить",
    "вони розкривають": "він відкриває",
    "здатний спостерігати": "може спостерігати",
    "старий стол": "старий стіл",
}

REQUIREMENTS_SYSTEM = """You extract a STRICT visual requirements specification from a structured Ukrainian scene.
The structured scene is the ONLY source of truth. NEVER use legacy scene.prompt.

First normalize obvious machine-translation errors by meaning. Example: серовиртній means сіруватий/grayish; дверь means двері; do not preserve nonsense words.

Extract ALL concrete visual facts from description, location, time, action, camera and mood. Every distinct concrete action must be a separate mandatory action in chronological order. Do not collapse actions into generic observation.

Preserve temporal qualifiers such as for several days, at 23:00, at noon, before/after, etc. Preserve exact clothing categories and important properties. Preserve object properties and explicit relations such as books on a table or climbing through a window.

If the source says a fact is specific, NEVER turn it into an OR choice. If the source says blocked upper window, do not output upper OR blocked window.

Return JSON only in this exact shape:
{
  "character": ["..."],
  "time": "...",
  "location": "...",
  "clothing": ["..."],
  "actions": [{"id":"a1","text":"...","mandatory":true}],
  "objects": [{"id":"o1","text":"...","mandatory":true}],
  "relations": [{"id":"r1","text":"...","mandatory":true}],
  "atmosphere": ["..."]
}

Use concise natural English requirement text. Character Олександр -> Alexander. Do not invent facts."""

TRANSLATOR_SYSTEM = """You are a professional cinematic Stable Diffusion prompt writer.
Create ONE natural English image prompt from the authoritative structured scene and extracted requirements.

NON-NEGOTIABLE:
- Every mandatory action must be explicit and in exactly the same chronological order.
- Every temporal qualifier that materially describes the scene must remain.
- Every mandatory object, property and relationship must be explicit.
- Character, exact time and clothing category must be preserved.
- A specific action cannot be replaced by a generic verb. 'observes' does not cover 'waits', 'opens', 'climbs through', or 'enters'.
- Never replace a specific fact with an OR choice.
- Do not invent people, objects, clothing, locations, events or motivations.
- Abstract thoughts may be omitted only when they are not visually necessary.
- Use fluent native English, concrete visual language, no metadata labels, English only.
- Natural name: Alexander.
- Return JSON only: {"prompt_en":"..."}.
- Target 30-55 words, never exceed 58 words."""

VALIDATOR_SYSTEM = """You are an EXTREMELY STRICT semantic coverage validator.
Validate ONLY against the extracted requirements. Do not weaken, reinterpret, merge or infer missing requirements.

Check EVERY mandatory action individually, EVERY mandatory object individually, EVERY mandatory relation individually, plus character, time, clothing and material location/atmosphere facts.

A generic action does not cover a specific action. Examples:
- 'observes the door' does NOT cover 'waits for it to open'.
- 'examines the window' does NOT cover 'opens the window'.
- 'is in the room' does NOT cover 'enters the room'.
- 'moves through the window' is acceptable for 'climbs through the window' only if the climbing meaning is explicit.
- 'books nearby' does NOT cover 'an old table holding books about ecology and nature'.
- 'window' alone does NOT cover 'blocked upper window'.

Temporal qualifiers are mandatory when extracted as mandatory facts. Action order is mandatory. If the prompt uses 'or' to weaken a specific requirement, report a contradiction/missing specificity.

PASS only when every mandatory requirement is explicitly covered, in order, with no material inventions, contradictions or weakened specificity.
Return JSON only:
{"ok":true,"missing":[],"invented":[],"contradictions":[],"covered":[]}"""


def normalize_source_text(text: str) -> str:
    text = text or ""
    for bad, good in NORMALIZATION.items():
        text = text.replace(bad, good)
    return text


def normalized_scene(scene: dict) -> dict:
    out = dict(scene)
    for key in ("title", "description", "location", "time", "action", "camera", "mood"):
        out[key] = normalize_source_text(str(out.get(key, "")))
    # Legacy prompt is deliberately excluded from every LLM request.
    out.pop("prompt", None)
    return out


def clean_json_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_first_json_object(text: str):
    text = clean_json_text(text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_json_object(text: str) -> dict:
    candidates = [clean_json_text(text)]
    extracted = extract_first_json_object(text)
    if extracted and extracted not in candidates:
        candidates.append(extracted)
    last_error = None
    for candidate in candidates:
        for variant in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                value = json.loads(variant)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError as exc:
                last_error = exc
    raise ValueError(f"LLM did not return valid JSON: {last_error}")


def words(text: str):
    return re.findall(r"\b[\w'-]+\b", text or "", flags=re.UNICODE)


def normalize_prompt(text: str) -> str:
    text = re.sub(r"[\r\n]+", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(' \"“”')
    text = re.sub(r"^(?:prompt|description)\s*:\s*", "", text, flags=re.I)
    return text


def cap_words(text: str) -> str:
    text = normalize_prompt(text)
    if len(words(text)) <= MAX_WORDS:
        return text.rstrip(" .") + "."
    # Prefer whole sentences, then a hard word cap as last resort.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept, count = [], 0
    for sentence in sentences:
        n = len(words(sentence))
        if count + n > MAX_WORDS:
            break
        kept.append(sentence.strip())
        count += n
    result = " ".join(x for x in kept if x)
    if len(words(result)) >= MIN_WORDS:
        return result.rstrip(" .") + "."
    return " ".join(words(text)[:MAX_WORDS]).rstrip(" .") + "."


def local_audit(prompt: str):
    errors = []
    if not prompt:
        errors.append("EMPTY_PROMPT")
    if CYRILLIC_RE.search(prompt):
        errors.append("CYRILLIC_IN_FINAL_PROMPT")
    n = len(words(prompt))
    if n < MIN_WORDS:
        errors.append(f"WORD_LIMIT_TOO_SHORT:{n}")
    if n > MAX_WORDS:
        errors.append(f"WORD_LIMIT_EXCEEDED:{n}")
    return errors


def scene_payload(scene: dict) -> str:
    source = {k: scene.get(k, "") for k in ("id", "title", "description", "location", "time", "action", "camera", "mood")}
    return json.dumps(source, ensure_ascii=False, indent=2)


def requirements_text(req: dict) -> str:
    return json.dumps(req, ensure_ascii=False, indent=2)


def normalize_requirements(req: dict) -> dict:
    req = dict(req or {})
    for key in ("character", "clothing", "atmosphere"):
        value = req.get(key, [])
        req[key] = value if isinstance(value, list) else ([str(value)] if value else [])
    for key in ("actions", "objects", "relations"):
        value = req.get(key, [])
        req[key] = value if isinstance(value, list) else []
        fixed = []
        for i, item in enumerate(req[key], 1):
            if isinstance(item, dict):
                fixed.append({"id": str(item.get("id") or f"{key[0]}{i}"), "text": str(item.get("text", "")), "mandatory": bool(item.get("mandatory", True))})
            elif item:
                fixed.append({"id": f"{key[0]}{i}", "text": str(item), "mandatory": True})
        req[key] = fixed
    req.setdefault("time", "")
    req.setdefault("location", "")
    return req


def extract_requirements(client: OllamaClient, scene: dict, attempts: int = 4) -> dict:
    scene = normalized_scene(scene)
    request = f"""Extract ALL strict visual requirements from this authoritative scene.

{scene_payload(scene)}

Normalize obvious source errors by meaning. Keep every concrete action as a separate mandatory item and preserve its order. Preserve temporal qualifiers, object properties and relations. Return JSON only."""
    last_error = None
    for _ in range(attempts):
        try:
            return normalize_requirements(parse_json_object(client.generate(request, system=REQUIREMENTS_SYSTEM)))
        except Exception as exc:
            last_error = exc
    raise ValueError(f"requirements extraction failed: {last_error}")


def translate_scene(client: OllamaClient, scene: dict, req: dict, correction: str = "") -> str:
    scene = normalized_scene(scene)
    correction_block = f"\nPREVIOUS DRAFT PROBLEMS — FIX EVERY ONE:\n{correction}\n" if correction else ""
    request = f"""Create one cinematic English image prompt.{correction_block}

MANDATORY REQUIREMENTS:
{requirements_text(req)}

AUTHORITATIVE STRUCTURED SCENE:
{scene_payload(scene)}

Write a single coherent image prompt. Preserve all mandatory actions in chronological order, all mandatory objects/relations, time and clothing. Never use legacy scene.prompt. English only. 30-55 words, maximum 58."""
    data = parse_json_object(client.generate(request, system=TRANSLATOR_SYSTEM))
    prompt = str(data.get("prompt_en", ""))
    if not prompt:
        raise ValueError("Missing prompt_en")
    return cap_words(prompt)


def validate_scene(client: OllamaClient, req: dict, prompt: str) -> dict:
    request = f"""Validate this prompt against the requirements with zero tolerance for omitted mandatory facts.

EXTRACTED REQUIREMENTS:
{requirements_text(req)}

PROPOSED PROMPT:
{prompt}

Check every action individually and in chronological order, every object/property, every relation, character, time, clothing and temporal qualifier. Return JSON only."""
    result = parse_json_object(client.generate(request, system=VALIDATOR_SYSTEM))
    result.setdefault("ok", False)
    result.setdefault("missing", [])
    result.setdefault("invented", [])
    result.setdefault("contradictions", [])
    result.setdefault("covered", [])
    return result


def semantic_tokens_for_action(text: str):
    t = text.lower()
    groups = []
    if any(x in t for x in ("several days", "multiple days", "over days")):
        groups.append(("duration_several_days", ["several days", "multiple days", "over several days", "over multiple days"]))
    if any(x in t for x in ("wait", "waiting", "waits")):
        groups.append(("wait", ["wait", "waiting", "waits"]))
    if any(x in t for x in ("search", "look for", "seek")):
        groups.append(("search", ["search", "searches", "looking for", "look for", "seeks"]))
    if "open" in t:
        groups.append(("open", ["opens", "open", "opening"]))
    if "climb" in t or "climbs through" in t:
        groups.append(("climb_through", ["climbs through", "climb through", "climbing through"]))
    if "enter" in t or "entering" in t:
        groups.append(("enter", ["enters", "entering", "enter the", "into the room"]))
    if any(x in t for x in ("observe", "observes", "examines", "examining", "study", "studies")):
        groups.append(("observe", ["observes", "observe", "examines", "examining", "studies", "study"]))
    return groups


def deterministic_audit(req: dict, prompt: str):
    p = prompt.lower()
    errors = []

    def has_any(anchors):
        return any(a in p for a in anchors)

    # Basic global requirements.
    for c in req.get("character", []):
        if str(c).lower() == "alexander" and "alexander" not in p:
            errors.append("MISSING_CHARACTER:Alexander")

    time_text = str(req.get("time", "")).lower()
    if "23:00" in time_text or "11:00 pm" in time_text or "11 pm" in time_text:
        if not has_any(["11 pm", "11:00 pm", "23:00"]):
            errors.append("MISSING_TIME:23:00")
    if "12:00" in time_text or "12 pm" in time_text or "noon" in time_text:
        if not has_any(["noon", "12 pm", "12:00 pm", "12:00"]):
            errors.append("MISSING_TIME:12:00")

    for c in req.get("clothing", []):
        t = str(c).lower()
        if "jacket" in t and "jacket" not in p:
            errors.append("MISSING_CLOTHING:jacket")
        if "parka" in t and "parka" not in p:
            errors.append("MISSING_CLOTHING:parka")
        if "sweater" in t and "sweater" not in p:
            errors.append("MISSING_CLOTHING:sweater")

    # Object/property/relationship anchors.
    for obj in req.get("objects", []):
        if not obj.get("mandatory", True):
            continue
        t = str(obj.get("text", "")).lower()
        oid = str(obj.get("id", "o"))
        if "window" in t and "window" not in p:
            errors.append(f"MISSING_OBJECT:{oid}:window")
        if "upper" in t and "window" in t and "upper" not in p:
            errors.append(f"MISSING_PROPERTY:{oid}:upper")
        if "blocked" in t and "blocked" not in p:
            errors.append(f"MISSING_PROPERTY:{oid}:blocked")
        if "broken" in t and "broken" not in p:
            errors.append(f"MISSING_PROPERTY:{oid}:broken")
        if "door" in t and "door" not in p:
            errors.append(f"MISSING_OBJECT:{oid}:door")
        if "table" in t and "table" not in p:
            errors.append(f"MISSING_OBJECT:{oid}:table")
        if "book" in t and "book" not in p:
            errors.append(f"MISSING_OBJECT:{oid}:books")
        if "room" in t and "room" not in p:
            errors.append(f"MISSING_OBJECT:{oid}:room")

    for rel in req.get("relations", []):
        if not rel.get("mandatory", True):
            continue
        t = str(rel.get("text", "")).lower()
        rid = str(rel.get("id", "r"))
        if "on the table" in t or ("table" in t and "book" in t):
            if not ("table" in p and has_any(["on the table", "on an old table", "on a table"])):
                errors.append(f"MISSING_RELATION:{rid}:books_on_table")
        if "ecology" in t and "ecology" not in p:
            errors.append(f"MISSING_RELATION:{rid}:ecology")
        if "nature" in t and "nature" not in p:
            errors.append(f"MISSING_RELATION:{rid}:nature")
        if "through" in t and "window" in t and not has_any(["through the window", "through a window", "through the upper window"]):
            errors.append(f"MISSING_RELATION:{rid}:through_window")

    # Action-specific anchors and temporal qualifiers.
    action_positions = []
    for action in req.get("actions", []):
        if not action.get("mandatory", True):
            continue
        text = str(action.get("text", ""))
        aid = str(action.get("id", "a"))
        for label, anchors in semantic_tokens_for_action(text):
            if not has_any(anchors):
                errors.append(f"MISSING_ACTION:{aid}:{label}")
            else:
                pos = min((p.find(a) for a in anchors if p.find(a) >= 0), default=-1)
                action_positions.append((aid, label, pos))
        # Explicit specificity: a mandatory specific fact must not be weakened with OR.
        if "window" in text.lower() and "upper" in text.lower() and "or" in p:
            # Only flag if the prompt actually contains an OR near window, not every 'or'.
            for m in re.finditer(r"[^.]{0,50}\b(?:upper|blocked|broken)[^.]{0,50}\bwindow\b[^.]{0,30}\bor\b[^.]{0,50}", p):
                errors.append(f"WEAKENED_SPECIFICITY:{aid}:window")
                break

    # Verify chronological order of actions when all relevant positions are known.
    ordered = [x for x in action_positions if x[2] >= 0]
    if len(ordered) >= 2:
        positions = [x[2] for x in ordered]
        if positions != sorted(positions):
            errors.append("ACTION_ORDER_VIOLATION")

    # The final prompt must never contain the known bad source word.
    if "серовирт" in p or "дверь" in p:
        errors.append("BAD_SOURCE_WORD_IN_PROMPT")
    return sorted(set(errors))


def correction_text(validation: dict, errors):
    parts = []
    for key in ("missing", "invented", "contradictions"):
        for item in validation.get(key, []) or []:
            parts.append(f"{key}: {item}")
    parts.extend(errors)
    return "; ".join(str(x) for x in parts) or "missing mandatory requirement"


def process_scene(client: OllamaClient, scene: dict):
    req = extract_requirements(client, scene)
    last_prompt = ""
    last_validation = {"ok": False, "missing": [], "invented": [], "contradictions": [], "covered": []}
    last_errors = []
    correction = ""

    for attempt in range(1, 6):
        prompt = translate_scene(client, scene, req, correction)
        validation = validate_scene(client, req, prompt)
        errors = local_audit(prompt) + deterministic_audit(req, prompt)
        ok = bool(validation.get("ok")) and not errors
        last_prompt, last_validation, last_errors = prompt, validation, errors
        if ok:
            return req, prompt, validation, errors, attempt
        correction = correction_text(validation, errors)

    return req, last_prompt, last_validation, last_errors, 5


def main():
    parser = argparse.ArgumentParser(description="AI YouTube Stories V3.4.2 strict semantic prompt pipeline")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_path = Path(args.audit)

    scenes = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(scenes, dict) and "scenes" in scenes:
        scenes = scenes["scenes"]
    if not isinstance(scenes, list):
        raise ValueError("Input JSON must be a list of scenes or an object containing 'scenes'.")
    if args.limit > 0:
        scenes = scenes[:args.limit]

    client = OllamaClient()
    output = []
    audit_lines = ["AI YouTube Stories — Prompt Audit V3.4.2", "=" * 55, ""]
    passed = failed = 0

    for scene in scenes:
        sid = scene.get("id", "?")
        print(f"[V3.4.2] scene {sid}: normalize + extract + generate + strict validate...")
        try:
            req, prompt, validation, errors, attempts = process_scene(client, scene)
            ok = bool(validation.get("ok")) and not errors
            if ok:
                passed += 1
            else:
                failed += 1

            item = dict(scene)
            item["prompt_requirements_v3_4_2"] = req
            item["prompt_v3_4_2"] = prompt
            item["prompt"] = prompt
            item["prompt_source"] = "structured_scene_v3_4_2_requirements"
            item["prompt_validation"] = validation
            item["prompt_errors"] = errors
            item["prompt_attempts"] = attempts
            output.append(item)

            audit_lines.append(f"SCENE {sid}: {'PASS' if ok else 'FAIL'}")
            audit_lines.append(f"Attempts: {attempts}")
            audit_lines.append(f"Words: {len(words(prompt))}")
            audit_lines.append(f"Prompt: {prompt}")
            audit_lines.append("Requirements:")
            audit_lines.append(json.dumps(req, ensure_ascii=False, indent=2))
            audit_lines.append("Validation:")
            audit_lines.append(json.dumps(validation, ensure_ascii=False, indent=2))
            audit_lines.append(f"Deterministic errors: {errors}")
            audit_lines.append("")
        except Exception as exc:
            failed += 1
            audit_lines.append(f"SCENE {sid}: FAIL")
            audit_lines.append(f"PIPELINE ERROR: {type(exc).__name__}: {exc}")
            audit_lines.append("")
            print(f"  ERROR: {exc}")

    audit_lines.append(f"SUMMARY: PASS={passed} FAIL={failed}")
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8")

    print(f"\nDONE: {output_path}")
    print(f"AUDIT: {audit_path}")
    print(f"PASS={passed} FAIL={failed}")


if __name__ == "__main__":
    main()
