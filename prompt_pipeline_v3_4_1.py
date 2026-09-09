import argparse
import json
import re
import sys
from pathlib import Path

from core.llm import OllamaClient

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "scenes.json"
DEFAULT_OUTPUT = BASE_DIR / "scenes_optimized_v3_4_1.json"
DEFAULT_AUDIT = BASE_DIR / "prompt_audit_v3_4_1.txt"
MAX_WORDS = 58
MIN_WORDS = 25
CYRILLIC_RE = re.compile(r"[\u0400-\u052F]")

# Common machine-translation errors observed in scenes.json.
NORMALIZATION = {
    "серовиртній": "сіруватий",
    "серовиртна": "сірувата",
    "серовиртний": "сіруватий",
    "дверь": "двері",
    "способы вхіду": "способи входу",
    "способы входу": "способи входу",
    "верхніх вікон": "верхнє вікно",
    "одна з верхніх вікон": "одне з верхніх вікон",
    "обокладують до кімнати": "залазить до кімнати",
    "обокладують": "залазить",
    "розкривають це вікно": "відкриває це вікно",
    "вони бачать": "він бачить",
    "вони бачать старий": "він бачить старий",
    "вони розкривають": "він відкриває",
    "здатний спостерігати": "може спостерігати",
    "виглядає диким і пустим": "виглядає моторошним і порожнім",
    "старий стол": "старий стіл",
}

REQUIREMENTS_SYSTEM = """You extract a strict visual requirements specification from a structured Ukrainian scene.
The structured scene is the only source of truth. NEVER use legacy scene.prompt.

FIRST normalize obvious machine-translation errors by meaning. Preserve facts; do not invent.
Important examples: 'серовиртній/серовиртна' means 'сіруватий/сірувата'; 'дверь' means 'двері'; 'способы вхіду' means 'способи входу'; 'обокладують до кімнати' means entering/climbing into the room.

Extract ALL concrete visual facts from BOTH description and action, especially ordered actions. Keep action sequence as separate mandatory items. Keep objects separate from relations. Do not collapse a specific action into a generic observation.

For a scene involving a window and entry, distinguish these actions when present: searches for another entrance -> notices upper/blocked/broken window -> opens the window -> climbs through it -> enters room -> examines room.
For a scene involving a table and books, preserve the relation that the old table holds books, and preserve what the books are about.

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

Use concise English requirement text so the downstream validator never receives transliterated Ukrainian. Character name Олександр -> Alexander. Do not add facts that are not in the scene."""

TRANSLATOR_SYSTEM = """You are a professional cinematic Stable Diffusion prompt writer.
Create ONE natural English image prompt from the authoritative structured scene and the extracted requirements.

NON-NEGOTIABLE:
- Every mandatory action must appear clearly, in the same order.
- Every mandatory object and relationship must appear.
- Preserve exact time and clothing category.
- Preserve important location and atmosphere without inventing details.
- A specific action cannot be replaced by a generic verb: 'opens the window' is required if opening is required; 'enters the room' is required if entering is required.
- If the sequence is too long for one frame, describe the final informative moment while explicitly retaining the causal action sequence in compact natural wording.
- Do not invent clothing types. Jacket remains jacket; parka remains parka; sweater remains sweater.
- Do not invent people, objects, events, motivations, or possessions.
- Abstract thoughts may be omitted unless visually necessary.
- Use fluent native English, concrete visual language, no metadata labels, English only.
- Natural name: Alexander.
- Return JSON only: {"prompt_en":"..."}.
- Target 30-55 words and never exceed 58 words."""

VALIDATOR_SYSTEM = """You are a strict semantic coverage validator.
Validate ONLY against the extracted requirements. Do not reinterpret or weaken them.

For every mandatory action, object, and relation, check explicit semantic coverage. Synonyms are allowed, but generic wording is not enough.
Examples:
- 'examines the window' does NOT cover 'opens the window'.
- 'is in the room' does NOT cover 'enters the room'.
- 'books nearby' does NOT cover 'an old table holding books'.
- 'window' alone does NOT cover 'blocked upper window' when those properties are mandatory.

Action order matters. Time and clothing category are strict. Flag material invented facts.
Return JSON only:
{"ok":true,"missing":[],"invented":[],"contradictions":[],"covered":[]}
PASS only when every mandatory requirement is covered and there are no material contradictions or inventions."""


def normalize_source_text(text: str) -> str:
    text = text or ""
    for bad, good in NORMALIZATION.items():
        text = text.replace(bad, good)
    return text


def normalized_scene(scene: dict) -> dict:
    out = dict(scene)
    for key in ("title", "description", "location", "time", "action", "camera", "mood"):
        out[key] = normalize_source_text(str(out.get(key, "")))
    return out


def clean_json_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def repair_json(text: str) -> str:
    text = clean_json_text(text)
    # Remove common trailing commas.
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def extract_first_json_object(text: str) -> str | None:
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
        for variant in (candidate, repair_json(candidate)):
            try:
                value = json.loads(variant)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError as exc:
                last_error = exc
    raise ValueError(f"LLM did not return valid JSON: {last_error}")


def words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)


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


def local_audit(prompt: str) -> list[str]:
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


def correction_text(validation: dict, errors: list[str]) -> str:
    parts = []
    for key in ("missing", "invented", "contradictions"):
        for item in validation.get(key, []) or []:
            parts.append(f"{key}: {item}")
    parts.extend(errors)
    return "; ".join(parts) or "missing mandatory requirement"


def normalize_requirements(req: dict) -> dict:
    # Convert accidental scalar fields to safe lists.
    for key in ("character", "clothing", "atmosphere"):
        if not isinstance(req.get(key), list):
            req[key] = [str(req[key])] if req.get(key) else []
    for key in ("actions", "objects", "relations"):
        if not isinstance(req.get(key), list):
            req[key] = []
    return req


def extract_requirements(client: OllamaClient, scene: dict, max_attempts: int = 4) -> dict:
    scene = normalized_scene(scene)
    request = f"""Extract ALL visual requirements from this authoritative scene.

{scene_payload(scene)}

Return JSON only. Normalize source-language errors by meaning. Keep every concrete action from description and action, in order. Keep important objects and their relationships."""
    last_error = None
    for _ in range(max_attempts):
        try:
            req = normalize_requirements(parse_json_object(client.generate(request, system=REQUIREMENTS_SYSTEM)))
            req.setdefault("character", [])
            req.setdefault("time", "")
            req.setdefault("location", "")
            req.setdefault("clothing", [])
            req.setdefault("actions", [])
            req.setdefault("objects", [])
            req.setdefault("relations", [])
            req.setdefault("atmosphere", [])
            return req
        except Exception as exc:
            last_error = exc
    raise ValueError(f"requirements extraction failed: {last_error}")


def translate_scene(client: OllamaClient, scene: dict, req: dict, correction: str = "") -> str:
    scene = normalized_scene(scene)
    correction_block = f"\nPREVIOUS DRAFT PROBLEMS — FIX ALL:\n{correction}\n" if correction else ""
    request = f"""Create one cinematic English image prompt.{correction_block}

MANDATORY REQUIREMENTS:
{requirements_text(req)}

AUTHORITATIVE STRUCTURED SCENE:
{scene_payload(scene)}

Before returning JSON, mentally check every mandatory action, object, relationship, character, time, and clothing item. Preserve action order. Never use legacy scene.prompt. English only, 30-55 words, maximum 58 words."""
    data = parse_json_object(client.generate(request, system=TRANSLATOR_SYSTEM))
    prompt = str(data.get("prompt_en", ""))
    if not prompt:
        raise ValueError("Missing prompt_en")
    return cap_words(prompt)


def validate_scene(client: OllamaClient, req: dict, prompt: str) -> dict:
    request = f"""Validate this prompt strictly.

EXTRACTED REQUIREMENTS:
{requirements_text(req)}

PROPOSED PROMPT:
{prompt}

Check every mandatory action/object/relation individually and check their order. Return JSON only."""
    result = parse_json_object(client.generate(request, system=VALIDATOR_SYSTEM))
    result.setdefault("ok", False)
    result.setdefault("missing", [])
    result.setdefault("invented", [])
    result.setdefault("contradictions", [])
    result.setdefault("covered", [])
    return result


def deterministic_audit(req: dict, prompt: str) -> list[str]:
    p = prompt.lower()
    errors = []

    def has_any(anchors):
        return any(a in p for a in anchors)

    time_text = str(req.get("time", "")).lower()
    if "23:00" in time_text or "11 pm" in time_text:
        if not has_any(["11 pm", "11:00 pm", "23:00"]):
            errors.append("MISSING_ANCHOR:time_11pm")
    if "12:00" in time_text or "12 pm" in time_text or "noon" in time_text:
        if not has_any(["noon", "12 pm", "12:00 pm", "12:00"]):
            errors.append("MISSING_ANCHOR:time_noon")

    for c in req.get("character", []):
        if str(c).lower() == "alexander" and "alexander" not in p:
            errors.append("MISSING_ANCHOR:character_alexander")

    for c in req.get("clothing", []):
        t = str(c).lower()
        if "jacket" in t and not has_any(["jacket"]):
            errors.append("MISSING_ANCHOR:jacket")
        if "parka" in t and not has_any(["parka"]):
            errors.append("MISSING_ANCHOR:parka")
        if "sweater" in t and "sweater" not in p:
            errors.append("MISSING_ANCHOR:sweater")

    for action in req.get("actions", []):
        if not action.get("mandatory", True):
            continue
        t = str(action.get("text", "")).lower()
        aid = str(action.get("id", "")).lower()
        if any(x in t or x in aid for x in ["open", "opening"]):
            if not has_any(["opens", "open", "opening"]):
                errors.append(f"MISSING_ANCHOR:{aid or 'open'}")
        if any(x in t or x in aid for x in ["climb", "through"]):
            if not has_any(["climbs", "climb", "climbing"]):
                errors.append(f"MISSING_ANCHOR:{aid or 'climb'}")
        if any(x in t or x in aid for x in ["enter", "into the room"]):
            if not has_any(["enters", "enter", "entering", "into the room"]):
                errors.append(f"MISSING_ANCHOR:{aid or 'enter'}")

    for obj in req.get("objects", []):
        if not obj.get("mandatory", True):
            continue
        t = str(obj.get("text", "")).lower()
        oid = str(obj.get("id", "")).lower()
        for noun, anchors in [
            ("window", ["window"]), ("table", ["table"]), ("book", ["book", "books"]),
            ("door", ["door"]), ("room", ["room"])
        ]:
            if noun in t or noun in oid:
                if not has_any(anchors):
                    errors.append(f"MISSING_ANCHOR:{oid or noun}")

    for rel in req.get("relations", []):
        if not rel.get("mandatory", True):
            continue
        t = str(rel.get("text", "")).lower()
        if "table" in t and "book" in t and not ("table" in p and "book" in p):
            errors.append("MISSING_RELATION:books_on_table")
        if "window" in t and "through" in t and "climb" in t:
            if not ("window" in p and "through" in p and "climb" in p):
                errors.append("MISSING_RELATION:climb_through_window")
        if "ecology" in t or "nature" in t:
            if not has_any(["ecology", "nature"]):
                errors.append("MISSING_RELATION:books_ecology_nature")

    return sorted(set(errors))


def process_scene(client: OllamaClient, scene: dict, max_attempts: int = 6):
    req = extract_requirements(client, scene)
    correction = ""
    last_validation = {"ok": False, "missing": [], "invented": [], "contradictions": [], "covered": []}
    last_prompt = ""
    last_errors = []

    for _ in range(max_attempts):
        try:
            prompt = translate_scene(client, scene, req, correction)
            last_prompt = prompt
            validation = validate_scene(client, req, prompt)
            det = deterministic_audit(req, prompt)
            local = local_audit(prompt)
            errors = local + det
            last_validation = validation
            last_errors = errors
            ok = bool(validation.get("ok")) and not validation.get("missing") and not validation.get("contradictions") and not validation.get("invented") and not errors
            if ok:
                return prompt, req, [], validation
            correction = correction_text(validation, errors)
        except Exception as exc:
            last_errors = [f"PIPELINE_ERROR:{type(exc).__name__}:{exc}"]
            correction = last_errors[0]

    semantic = []
    for key in ("missing", "invented", "contradictions"):
        for item in last_validation.get(key, []) or []:
            semantic.append(f"{key}={item}")
    semantic.extend(last_errors)
    return last_prompt, req, sorted(set(semantic)), last_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_path = Path(args.audit)

    scenes = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(scenes, list):
        raise ValueError("Input scenes.json must contain a list")

    selected = scenes[:args.limit] if args.limit > 0 else scenes
    client = OllamaClient()
    output = []
    audit_lines = ["AI YouTube Stories — V3.4.1 semantic requirements pipeline", ""]
    passed = failed = 0

    selected_ids = {scene.get("id") for scene in selected}
    for scene in scenes:
        if scene.get("id") not in selected_ids:
            output.append(scene)
            continue

        print(f"[V3.4.1] scene {scene.get('id')}: normalize + extract + generate + strict validate...", flush=True)
        try:
            prompt, req, errors, validation = process_scene(client, scene)
        except Exception as exc:
            prompt, req, errors, validation = "", {}, [f"PIPELINE_ERROR:{type(exc).__name__}:{exc}"], {"ok": False}

        item = dict(scene)
        item["prompt_requirements_v3_4_1"] = req
        item["prompt_v3_4_1"] = prompt
        item["prompt"] = prompt or scene.get("prompt", "")
        item["prompt_source"] = "structured_scene_v3_4_1_requirements"
        item["prompt_validation"] = validation
        item["prompt_errors"] = errors
        output.append(item)

        if not errors and validation.get("ok"):
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        audit_lines.append(f"SCENE {scene.get('id')}: {status} | words={len(words(prompt)) if prompt else 0}")
        audit_lines.append(f"TITLE: {scene.get('title', '')}")
        audit_lines.append(f"REQUIREMENTS: {json.dumps(req, ensure_ascii=False)}")
        audit_lines.append(f"PROMPT: {prompt}")
        audit_lines.append(f"VALIDATION: {json.dumps(validation, ensure_ascii=False)}")
        audit_lines.append(f"ERRORS: {errors}")
        audit_lines.append("")

    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_lines.append(f"SUMMARY: PASS={passed} FAIL={failed}")
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8")

    print(f"\nDONE: {output_path}")
    print(f"AUDIT: {audit_path}")
    print(f"PASS={passed} FAIL={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
