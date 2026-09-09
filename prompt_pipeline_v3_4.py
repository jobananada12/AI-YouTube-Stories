import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

from core.llm import OllamaClient

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "scenes.json"
DEFAULT_OUTPUT = BASE_DIR / "scenes_optimized_v3_4.json"
DEFAULT_AUDIT = BASE_DIR / "prompt_audit_v3_4.txt"
MAX_WORDS = 58
MIN_WORDS = 25
CYRILLIC_RE = re.compile(r"[\u0400-\u052F]")
META_RE = re.compile(r"(?:^|\s)(?:mood|location|time|camera|description|action)\s*:\s*", re.I)
AWKWARD_PHRASE_RE = re.compile(
    r"\b(?:internally pondering|reflecting on his own reflections|reflecting on reflections|"
    r"pondering internally|contemplates his own reflections|gaze reflecting on personal reflections)\b",
    re.I,
)

REQUIREMENTS_SYSTEM = """You extract a strict visual requirements specification from a structured Ukrainian scene.
The structured scene fields are the only source of truth. Ignore any legacy prompt.
Extract only facts that can be represented or explicitly present in one cinematic image.

RULES:
- Character identity, exact time, location, important clothing, concrete actions, concrete objects, and meaningful relationships are requirements.
- Preserve ordered concrete action sequences. Do not replace opening/climbing/entering with a generic observation.
- Normalize obvious machine-translation errors using context (for example, a misspelled Ukrainian adjective meaning gray should become gray/grayish). Never invent a new fact.
- Abstract thoughts may be omitted unless they have a clear visual consequence.
- Mark concrete scene-defining facts as mandatory.
- Atmosphere/mood can be optional unless it changes the scene identity.
- Do not use the old scene.prompt.
- Return JSON only in this exact shape:
{
  "character": ["..."],
  "time": "...",
  "location": "...",
  "clothing": ["..."],
  "actions": [{"id":"...","text":"...","mandatory":true}],
  "objects": [{"id":"...","text":"...","mandatory":true}],
  "relations": [{"id":"...","text":"...","mandatory":true}],
  "atmosphere": ["..."]
}"""

TRANSLATOR_SYSTEM = """You are a professional cinematic Stable Diffusion prompt writer.
Create ONE natural English prompt from the AUTHORITATIVE STRUCTURED SCENE and its EXTRACTED REQUIREMENTS.
The requirements are a checklist, not a license to invent.

NON-NEGOTIABLE:
1. Include every mandatory requirement.
2. Preserve every mandatory concrete action and its order/direction when possible.
3. Preserve mandatory objects and relationships.
4. Preserve exact time and clothing category.
5. Use natural English names: Олександр -> Alexander unless another Latin name is explicitly supplied.
6. Do not invent people, objects, clothing, locations, events, or actions.
7. Abstract thoughts may be omitted.
8. Use native fluent cinematic English, concrete visual language, no metadata labels, English only.
9. One visual frame: combine the most informative supported moment and resulting objects naturally.
10. Return JSON only: {\"prompt_en\": \"...\"}.
11. Target 30-55 words and never exceed 58 words."""

VALIDATOR_SYSTEM = """You are a strict deterministic-style semantic coverage validator.
Validate the proposed English image prompt ONLY against the EXTRACTED REQUIREMENTS.
Do not reinterpret the original scene and do not excuse missing mandatory facts.

For EVERY mandatory action/object/relation, decide whether the prompt clearly expresses it.
Synonyms and natural paraphrases are valid, but generic verbs do not satisfy specific actions.
Examples: 'examines the window' does not satisfy 'opens the window'; 'is in the room' does not satisfy 'enters the room'; 'books' does not satisfy a required 'old table holding books' relationship unless the table is present.
Order and direction of mandatory sequences matter.
Time and clothing category are strict.
A fact not present in requirements must not be required, but invented facts should be flagged.

Return JSON only:
{
  "ok": true,
  "missing": [],
  "invented": [],
  "contradictions": [],
  "covered": []
}
PASS only when every mandatory requirement is covered and there are no material contradictions or inventions."""


def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_object(text: str) -> dict:
    text = clean_json_text(text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        value = json.loads(match.group(0))
        if isinstance(value, dict):
            return value
    raise ValueError(f"LLM did not return valid JSON: {text[:500]}")


def words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)


def normalize_prompt(text: str) -> str:
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(' \"“”')
    text = re.sub(r"^(?:prompt|description)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text


def cap_words(text: str, limit: int = MAX_WORDS) -> str:
    text = normalize_prompt(text)
    if len(words(text)) <= limit:
        return text.rstrip(" .") + "."
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept, count = [], 0
    for sentence in sentences:
        n = len(words(sentence))
        if count + n > limit:
            break
        kept.append(sentence.strip())
        count += n
    result = " ".join(x for x in kept if x)
    if len(words(result)) >= MIN_WORDS:
        return result.rstrip(" .") + "."
    return " ".join(words(text)[:limit]).rstrip(" .") + "."


def local_audit(prompt: str) -> list[str]:
    errors = []
    if not prompt:
        errors.append("EMPTY_PROMPT")
    if CYRILLIC_RE.search(prompt):
        errors.append("CYRILLIC_IN_FINAL_PROMPT")
    if META_RE.search(prompt):
        errors.append("METADATA_LABEL_IN_PROMPT")
    if AWKWARD_PHRASE_RE.search(prompt):
        errors.append("AWKWARD_LITERAL_PHRASE")
    n = len(words(prompt))
    if n < MIN_WORDS:
        errors.append(f"WORD_LIMIT_TOO_SHORT:{n}")
    if n > MAX_WORDS:
        errors.append(f"WORD_LIMIT_EXCEEDED:{n}")
    if "  " in prompt:
        errors.append("DOUBLE_SPACE")
    return errors


def scene_payload(scene: dict) -> str:
    source = {
        "id": scene.get("id"),
        "title": scene.get("title", ""),
        "description": scene.get("description", ""),
        "location": scene.get("location", ""),
        "time": scene.get("time", ""),
        "action": scene.get("action", ""),
        "camera": scene.get("camera", ""),
        "mood": scene.get("mood", ""),
    }
    return json.dumps(source, ensure_ascii=False, indent=2)


def requirements_text(req: dict) -> str:
    return json.dumps(req, ensure_ascii=False, indent=2)


def correction_text(validation: dict, local_errors: list[str]) -> str:
    parts = []
    for key in ("missing", "invented", "contradictions"):
        for item in validation.get(key, []) or []:
            parts.append(f"{key}: {item}")
    parts.extend(local_errors)
    return "; ".join(parts) or "missing mandatory requirement"


def extract_requirements(client: OllamaClient, scene: dict, max_attempts: int = 3) -> dict:
    request = f"""Extract the visual requirements for this scene.

STRUCTURED SCENE — AUTHORITATIVE:
{scene_payload(scene)}

Important: include all concrete actions from both DESCRIPTION and ACTION, especially ordered sequences. Include important objects and their relationships. Correct obvious source-language typos by meaning, but do not add facts.
Return JSON only."""
    last_error = None
    for _ in range(max_attempts):
        try:
            req = parse_json_object(client.generate(request, system=REQUIREMENTS_SYSTEM))
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
    correction_block = ""
    if correction:
        correction_block = f"""
PREVIOUS DRAFT FAILED COVERAGE VALIDATION.
Fix these problems without dropping already-correct requirements:
{correction}
"""
    request = f"""Create one cinematic English image prompt.
{correction_block}
EXTRACTED REQUIREMENTS — MANDATORY CHECKLIST:
{requirements_text(req)}

STRUCTURED SCENE — CONTEXT ONLY:
{scene_payload(scene)}

Before writing, ensure every mandatory action, object, relationship, character, time, and clothing fact appears clearly in the final prompt. Preserve action order. Do not use legacy scene.prompt. English only, 30-55 words, maximum 58 words, JSON only."""
    data = parse_json_object(client.generate(request, system=TRANSLATOR_SYSTEM))
    result = str(data.get("prompt_en", ""))
    if not result:
        raise ValueError("Missing prompt_en")
    return cap_words(result)


def validate_scene(client: OllamaClient, req: dict, prompt: str) -> dict:
    request = f"""Validate this proposed prompt against the requirements.

EXTRACTED REQUIREMENTS:
{requirements_text(req)}

PROPOSED ENGLISH PROMPT:
{prompt}

Check every mandatory item individually, especially action sequence order and action-object relationships. Return JSON only."""
    result = parse_json_object(client.generate(request, system=VALIDATOR_SYSTEM))
    result.setdefault("ok", False)
    result.setdefault("missing", [])
    result.setdefault("invented", [])
    result.setdefault("contradictions", [])
    result.setdefault("covered", [])
    return result


def deterministic_requirement_audit(req: dict, prompt: str) -> list[str]:
    """Extra guard against an LLM validator declaring PASS while a central noun/action is absent.
    This is intentionally conservative: it checks obvious English anchors for mandatory requirements.
    """
    p = prompt.lower()
    errors = []

    def require_anchor(text: str, anchors: list[str], label: str):
        if not any(a in p for a in anchors):
            errors.append(f"MISSING_ANCHOR:{label}")

    time_text = str(req.get("time", "")).lower()
    if "23:00" in time_text or "11 pm" in time_text or "11:00 pm" in time_text:
        require_anchor(time_text, ["11 pm", "11:00 pm", "23:00"], "time_11pm")
    elif "12:00" in time_text or "12 pm" in time_text or "noon" in time_text:
        require_anchor(time_text, ["noon", "12 pm", "12:00 pm", "12:00"], "time_noon")

    for item in req.get("character", []):
        if str(item).lower() == "alexander":
            require_anchor(str(item), ["alexander"], "character_alexander")

    for item in req.get("clothing", []):
        t = str(item).lower()
        if "parka" in t or "jacket" in t:
            require_anchor(t, ["parka", "jacket"], "jacket_or_parka")
        if "sweater" in t:
            require_anchor(t, ["sweater"], "sweater")

    for action in req.get("actions", []):
        if not action.get("mandatory", True):
            continue
        text = str(action.get("text", "")).lower()
        aid = str(action.get("id", "")).lower()
        anchors = []
        if "open" in text or "open" in aid:
            anchors += ["opens", "open", "opening"]
        if "climb" in text or "climb" in aid:
            anchors += ["climb", "climbs", "climbing"]
        if "enter" in text or "enter" in aid:
            anchors += ["enter", "enters", "entering", "into the room"]
        if "examin" in text or "observ" in text:
            anchors += ["examines", "examining", "studies", "observes", "watching", "watches"]
        if anchors:
            require_anchor(text, anchors, aid or text[:40])

    for obj in req.get("objects", []):
        if not obj.get("mandatory", True):
            continue
        text = str(obj.get("text", "")).lower()
        oid = str(obj.get("id", "")).lower()
        if "window" in text or "window" in oid:
            require_anchor(text, ["window"], oid or "window")
        if "table" in text or "table" in oid:
            require_anchor(text, ["table"], oid or "table")
        if "book" in text or "book" in oid:
            require_anchor(text, ["book", "books"], oid or "books")
        if "door" in text or "door" in oid:
            require_anchor(text, ["door"], oid or "door")
        if "room" in text or "room" in oid:
            require_anchor(text, ["room"], oid or "room")

    for rel in req.get("relations", []):
        if not rel.get("mandatory", True):
            continue
        text = str(rel.get("text", "")).lower()
        if "table" in text and "book" in text:
            if "table" not in p or "book" not in p:
                errors.append("MISSING_RELATION:books_on_table")
        if "through" in text and "window" in text and "climb" in text:
            if "climb" not in p or "through" not in p or "window" not in p:
                errors.append("MISSING_RELATION:climb_through_window")

    return sorted(set(errors))


def process_scene(client: OllamaClient, scene: dict, max_attempts: int = 5) -> tuple[str, dict, list[str], dict]:
    req = extract_requirements(client, scene)
    correction = ""
    last_validation = {"ok": False, "missing": [], "invented": [], "contradictions": [], "covered": []}
    last_prompt = ""
    last_det = []

    for attempt in range(1, max_attempts + 1):
        prompt = translate_scene(client, scene, req, correction)
        last_prompt = prompt
        validation = validate_scene(client, req, prompt)
        deterministic_errors = deterministic_requirement_audit(req, prompt)
        local_errors = local_audit(prompt)
        all_errors = local_errors + deterministic_errors
        ok = bool(validation.get("ok", False)) and not all_errors and not validation.get("missing") and not validation.get("contradictions")
        if ok:
            return prompt, req, [], validation
        last_validation = validation
        last_det = deterministic_errors
        correction = correction_text(validation, all_errors)

    errors = ["SEMANTIC_COVERAGE_FAILED"]
    errors.extend(last_det)
    errors.extend(local_audit(last_prompt))
    return last_prompt, req, errors, last_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="AI YouTube Stories Prompt Pipeline V3.4 - requirements extraction")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=1)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    scenes = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(scenes, list):
        print("ERROR: scenes.json must contain a JSON array", file=sys.stderr)
        return 2

    client = OllamaClient(model=args.model) if args.model else OllamaClient()
    selected = [s for s in scenes if int(s.get("id", 0)) >= args.start]
    if args.limit > 0:
        selected = selected[:args.limit]
    selected_ids = {s.get("id") for s in selected}

    output = []
    audit_lines = [
        "AI YOUTUBE STORIES - PROMPT AUDIT V3.4",
        "Architecture: structured scene -> requirements extraction -> prompt -> LLM coverage validation -> deterministic guard.",
        "Legacy scene.prompt is ignored.",
        "Mandatory concrete actions, objects, relationships, time, character, and clothing are required.",
        "English-only final prompts; 25-58 words.",
        "",
    ]
    passed = failed = 0

    for scene in scenes:
        if scene.get("id") not in selected_ids:
            output.append(deepcopy(scene))
            continue

        sid = scene.get("id")
        print(f"[V3.4] scene {sid}: extracting requirements + generating + validating...", flush=True)
        try:
            prompt, req, errors, validation = process_scene(client, scene)
            item = deepcopy(scene)
            item["prompt_requirements_v3_4"] = req
            item["prompt_v3_4"] = prompt
            item["prompt"] = prompt
            item["prompt_source"] = "structured_scene_v3_4_requirements"
            item["prompt_validation"] = validation
            item["prompt_errors"] = errors
            output.append(item)

            status = "PASS" if not errors else "FAIL"
            if errors:
                failed += 1
            else:
                passed += 1
            audit_lines.append(
                f"SCENE {sid}: {status} | words={len(words(prompt))} | "
                f"missing={validation.get('missing', [])} | invented={validation.get('invented', [])} | "
                f"contradictions={validation.get('contradictions', [])} | errors={errors}"
            )
            audit_lines.append(f"  REQUIREMENTS: {json.dumps(req, ensure_ascii=False)}")
            audit_lines.append(f"  PROMPT: {prompt}")
        except Exception as exc:
            failed += 1
            item = deepcopy(scene)
            item["prompt_requirements_v3_4"] = {}
            item["prompt_v3_4"] = ""
            item["prompt"] = ""
            item["prompt_source"] = "structured_scene_v3_4_requirements"
            item["prompt_validation"] = {"ok": False, "missing": [], "invented": [], "contradictions": [], "covered": []}
            item["prompt_errors"] = [f"PIPELINE_ERROR:{type(exc).__name__}:{exc}"]
            output.append(item)
            audit_lines.append(f"SCENE {sid}: FAIL | {item['prompt_errors'][0]}")

    audit_lines.extend(["", f"SUMMARY: processed={passed + failed}, passed={passed}, failed={failed}"])
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    args.audit.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    print(f"\nDONE: {args.output}")
    print(f"AUDIT: {args.audit}")
    print(f"PASS={passed} FAIL={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
