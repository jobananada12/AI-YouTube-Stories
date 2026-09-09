import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

from core.llm import OllamaClient

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "scenes.json"
DEFAULT_OUTPUT = BASE_DIR / "scenes_optimized_v3_3.json"
DEFAULT_AUDIT = BASE_DIR / "prompt_audit_v3_3.txt"
MAX_WORDS = 58
MIN_WORDS = 25
CYRILLIC_RE = re.compile(r"[\u0400-\u052F]")
META_RE = re.compile(r"(?:^|\s)(?:mood|location|time|camera|description|action)\s*:\s*", re.I)
AWKWARD_PHRASE_RE = re.compile(
    r"\b(?:internally pondering|reflecting on his own reflections|reflecting on reflections|"
    r"pondering internally|contemplates his own reflections|gaze reflecting on personal reflections)\b",
    re.I,
)

TRANSLATOR_SYSTEM = """You are a professional cinematic screenplay-to-image-prompt writer.
The Ukrainian STRUCTURED SCENE is the only source of truth. Ignore any legacy prompt.
Create one natural English Stable Diffusion prompt for ONE visual frame.

NON-NEGOTIABLE SEMANTIC RULES:
1. ACTION is mandatory and is the core of the image. Preserve every concrete, visually representable action that matters.
2. DESCRIPTION and ACTION are complementary. If they describe an ordered sequence, preserve the key visible sequence naturally.
3. Never weaken a concrete action into merely 'observes', 'stands', 'looks', or 'contemplates'.
4. Preserve important concrete objects. If the source explicitly contains a window, table, books, door, vehicle, tool, etc., do not silently remove it.
5. Preserve concrete action + object relationships: opens a window, climbs through a window, enters a room, examines books on a table, etc.
6. Preserve exact time. 23:00 = 11 PM; 12:00 = noon/12 PM. Never change time of day.
7. Preserve clothing category. Jacket/parka is not trench coat; sweater is not shirt.
8. Use natural English names. Олександр -> Alexander unless another Latin name is explicitly supplied.
9. Do not invent people, objects, clothing, locations, events, or actions.
10. Abstract thoughts may be omitted. Never invent awkward literal introspection to represent a thought.
11. Location may be naturally shortened, but never changed into a different place.
12. For a still frame, represent the most informative moment of the supported sequence. When possible, show the action and the important resulting objects in the same frame.

QUALITY:
- Native, fluent cinematic English.
- Concrete visual language: subject + setting + action + key objects + time/lighting + mood/composition.
- No metadata labels, no explanations, no Ukrainian/Russian text.
- Return JSON only: {\"prompt_en\": \"...\"}.
- 25-55 words."""

VALIDATOR_SYSTEM = """You are a strict semantic continuity validator for cinematic image prompts.
The STRUCTURED SCENE is authoritative. Compare MEANING, not literal wording.
Return JSON only: {\"ok\": true/false, \"missing\": [], \"invented\": [], \"contradictions\": []}.

STRICT RULES:
- Every important concrete visual action from ACTION/DESCRIPTION must be represented, especially action sequences.
- A generic verb such as 'examines' or 'observes' does NOT satisfy a source action such as opens, climbs through, enters, takes, reads, writes, walks into, or removes.
- Every important concrete object needed for the scene identity must remain visible or explicitly present in the prompt.
- Concrete relationships matter: if the source says an old table with ecology/nature books, 'ecology books' alone is insufficient if the table is a meaningful scene object.
- If the source says he opens a window and enters through it, a prompt that only says he examines a room is FAIL.
- If the source supports an action sequence, the prompt may compress it naturally, but must retain the key actions and their direction/order.
- TIME is strict. 12:00 cannot become early morning; 23:00 cannot become daytime.
- Clothing category is strict.
- Character identity is strict, but Олександр -> Alexander is valid natural English.
- Do not mark a fact invented if it is supported anywhere in DESCRIPTION, ACTION, LOCATION, TIME, CAMERA, or MOOD.
- Do not demand minor atmospheric wording or contextual duration such as 'over several days' unless it changes the visible scene.
- Natural location shortening is valid.
- Abstract thoughts may be omitted; awkward introspective wording is a quality failure.
- Flag contradictions, missing central actions/objects, invented facts, and materially wrong time/clothing/location.
- A PASS requires the prompt to be semantically faithful AND visually useful."""


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
    if len(words(text)) <= limit:
        return text.rstrip(" .") + "."
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    count = 0
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


def correction_text(validation: dict, local_errors: list[str]) -> str:
    parts = []
    for key in ("missing", "invented", "contradictions"):
        for item in validation.get(key, []) or []:
            parts.append(f"{key}: {item}")
    parts.extend(local_errors)
    return "; ".join(parts) or "semantic mismatch"


def translate_scene(client: OllamaClient, scene: dict, correction: str = "") -> str:
    correction_block = ""
    if correction:
        correction_block = f"""
PREVIOUS DRAFT FAILED VALIDATION.
Fix ONLY these problems while preserving all supported source facts:
{correction}
Do not replace a missing concrete action with a generic observation. Do not invent anything while correcting it.
"""

    request = f"""Create one cinematic English image prompt from this structured scene.
{correction_block}
STRUCTURED SCENE — AUTHORITATIVE:
{scene_payload(scene)}

Before writing, silently extract:
- main person/people
- every important concrete object
- every concrete visual action and action sequence
- exact time
- clothing category
- location
- useful camera/mood guidance

Then write ONE fluent visual prompt.

MANDATORY:
- Keep concrete actions such as opening, climbing through, entering, reading, taking, writing, walking, etc.
- Keep the important objects involved in those actions.
- Preserve action order when supported.
- Preserve exact time and clothing type.
- Do not use the legacy scene.prompt.
- Do not add unsupported objects or events.
- Do not turn thoughts into invented visual objects.
- Avoid phrases like 'contemplates his own reflections'.
- English only, 25-55 words, JSON only with prompt_en."""

    data = parse_json_object(client.generate(request, system=TRANSLATOR_SYSTEM))
    result = normalize_prompt(str(data.get("prompt_en", "")))
    if not result:
        raise ValueError("Missing prompt_en")
    return cap_words(result)


def validate_scene(client: OllamaClient, scene: dict, prompt: str) -> dict:
    request = f"""Validate this proposed image prompt.

STRUCTURED SCENE — AUTHORITATIVE:
{scene_payload(scene)}

PROPOSED ENGLISH PROMPT:
{prompt}

Pay special attention to concrete action sequences and their objects. A prompt that only describes the final room but omits opening/climbing/entering fails when those actions are central to ACTION/DESCRIPTION.
Return JSON only."""
    return parse_json_object(client.generate(request, system=VALIDATOR_SYSTEM))


def process_scene(client: OllamaClient, scene: dict, max_attempts: int = 4) -> tuple[str, list[str], dict]:
    correction = ""
    last_validation = {"ok": False, "missing": [], "invented": [], "contradictions": []}
    last_prompt = ""

    for attempt in range(1, max_attempts + 1):
        prompt = translate_scene(client, scene, correction)
        last_prompt = prompt
        validation = validate_scene(client, scene, prompt)
        last_validation = validation
        local_errors = local_audit(prompt)
        ok = bool(validation.get("ok", False)) and not local_errors
        if ok:
            return prompt, [], validation
        correction = correction_text(validation, local_errors)

    errors = ["SEMANTIC_VALIDATION_FAILED"]
    errors.extend(local_audit(last_prompt))
    return last_prompt, errors, last_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="AI YouTube Stories Prompt Pipeline V3.3")
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
        "AI YOUTUBE STORIES - PROMPT AUDIT V3.3",
        "Source of truth: structured Ukrainian scene fields.",
        "Legacy scene.prompt is ignored.",
        "Concrete actions and important objects are mandatory.",
        "Ordered action sequences must remain semantically visible.",
        "TIME and clothing category are strict.",
        "English-only final prompts; 25-58 words.",
        "",
    ]

    passed = 0
    failed = 0

    for scene in scenes:
        if scene.get("id") not in selected_ids:
            output.append(deepcopy(scene))
            continue

        sid = scene.get("id")
        print(f"[V3.3] scene {sid}: translating + strict validating...", flush=True)
        try:
            prompt, errors, validation = process_scene(client, scene)
            item = deepcopy(scene)
            item["prompt_v3_3"] = prompt
            item["prompt"] = prompt
            item["prompt_source"] = "structured_scene_v3_3_strict"
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
                f"missing={validation.get('missing', [])} | "
                f"invented={validation.get('invented', [])} | "
                f"contradictions={validation.get('contradictions', [])} | errors={errors}"
            )
            audit_lines.append(f"  PROMPT: {prompt}")
        except Exception as exc:
            failed += 1
            item = deepcopy(scene)
            item["prompt_v3_3"] = ""
            item["prompt"] = ""
            item["prompt_source"] = "structured_scene_v3_3_strict"
            item["prompt_validation"] = {"ok": False, "missing": [], "invented": [], "contradictions": []}
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
