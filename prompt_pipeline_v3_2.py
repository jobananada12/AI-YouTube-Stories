import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

from core.llm import OllamaClient

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "scenes.json"
DEFAULT_OUTPUT = BASE_DIR / "scenes_optimized_v3_2.json"
DEFAULT_AUDIT = BASE_DIR / "prompt_audit_v3_2.txt"
MAX_WORDS = 58
CYRILLIC_RE = re.compile(r"[\u0400-\u052F]")
META_RE = re.compile(r"(?:^|\s)(?:mood|location|time|camera|description|action)\s*:\s*", re.I)
DUPLICATE_OBJECT_RE = re.compile(r"\b(house|room|door|cat|dog|journal|basement)\b.*\b\1\b", re.I)
AWKWARD_PHRASE_RE = re.compile(r"\b(?:gaze reflecting on personal reflections|internally pondering|reflecting on reflections)\b", re.I)

TRANSLATOR_SYSTEM = """You are a professional screenplay-to-image-prompt translator.
The Ukrainian structured scene is the ONLY source of truth.
Translate the meaning of the WHOLE scene into natural, cinematic English; never translate word-by-word and never concatenate dictionary fragments.
Write one coherent visual description, not a literal translation of every grammatical mistake in the source.
Do not invent, remove, swap, or move people, objects, actions, locations, time, or events.
Preserve the intended visual facts even when the Ukrainian wording is awkward.
Abstract thoughts may be omitted or expressed only through visible facial expression, posture, or atmosphere when that is clearly supported by the scene; never invent a new event or object.
Use character names consistently and preserve the source name exactly when one is given.
Prefer concrete visual details: subject, setting, action, key object, time/lighting, and mood/camera when useful.
Avoid awkward literal phrases such as 'gaze reflecting on personal reflections' or 'internally pondering'.
Do not use the old prompt as a source of story facts.
Return valid JSON only with exactly: {\"prompt_en\": \"...\"}.
The prompt must be a single natural English visual description for Stable Diffusion, 25-55 words, with no headings, metadata labels, explanations, Ukrainian/Russian text, or quotation marks around the prompt."""

VALIDATOR_SYSTEM = """You are a strict semantic continuity and natural-language editor.
Compare the Ukrainian structured scene with the proposed English Stable Diffusion prompt.
The Ukrainian scene is authoritative.
Return valid JSON only: {\"ok\": true/false, \"missing\": [\"...\"], \"invented\": [\"...\"]}.
Mark ok=false if an important person, object, action, setting, time cue, or event is missing or invented.
Also mark ok=false if the English is clearly unnatural, fragmented, contradictory, or contains awkward literal thought phrases that do not form a usable cinematic prompt.
Minor grammatical differences and harmless visual phrasing are acceptable.
Never require details that are absent from the Ukrainian scene."""


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
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
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
    ws = words(text)
    if len(ws) <= limit:
        return text.rstrip(" .") + "."
    pieces = re.split(r"(?<=[.!?])\s+|,\s+", text)
    out = []
    count = 0
    for piece in pieces:
        n = len(words(piece))
        if count + n > limit:
            break
        out.append(piece.strip())
        count += n
    result = ", ".join(p for p in out if p)
    if len(words(result)) >= 12:
        return result.rstrip(" ,.;") + "."
    return " ".join(ws[:limit]).rstrip(" ,.;") + "."


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
    if n > MAX_WORDS:
        errors.append(f"WORD_LIMIT_EXCEEDED:{n}")
    if DUPLICATE_OBJECT_RE.search(prompt):
        errors.append("REPEATED_OBJECT_PATTERN")
    if "  " in prompt:
        errors.append("DOUBLE_SPACE")
    return errors


def scene_payload(scene: dict) -> str:
    # Legacy scene.prompt is intentionally excluded because it contains known semantic drift.
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


def translate_scene(client: OllamaClient, scene: dict, correction: str = "") -> str:
    extra = ""
    if correction:
        extra = f"\nA previous draft failed semantic or quality validation. Correct only these problems: {correction}\n"
    prompt = f"""Translate this complete structured scene into one natural cinematic English visual prompt.
{extra}
STRUCTURED SCENE (authoritative):
{scene_payload(scene)}

Requirements:
- Preserve every important event and action that is visually representable.
- Preserve named people exactly as names; do not shorten or rename them.
- Preserve the stated location and time when visually useful and do not invent a different time of day.
- Use camera information only as shot/composition guidance.
- Convert awkward abstract wording into concise visual language when possible, without inventing facts.
- Do not copy or repair the legacy prompt.
- Do not add cats, photographs, corridors, documents, furniture, clothing, rooms, doors, or other details unless the structured scene supports them.
- Do not write labels such as Mood:, Location:, Time:, Camera:, Action:.
- Avoid phrases such as "internally pondering", "gaze reflecting on personal reflections", or other literal descriptions of thoughts.
- The result must read like a polished cinematic image prompt written by a native English speaker.
- Return JSON only."""
    data = parse_json_object(client.generate(prompt, system=TRANSLATOR_SYSTEM))
    result = normalize_prompt(str(data.get("prompt_en", "")))
    if not result:
        raise ValueError("Missing prompt_en")
    return cap_words(result)


def validate_scene(client: OllamaClient, scene: dict, prompt: str) -> dict:
    request = f"""Check this proposed prompt against the authoritative structured scene.

STRUCTURED SCENE:
{scene_payload(scene)}

PROPOSED ENGLISH PROMPT:
{prompt}

Return JSON only."""
    return parse_json_object(client.generate(request, system=VALIDATOR_SYSTEM))


def process_scene(client: OllamaClient, scene: dict, max_attempts: int = 2) -> tuple[str, list[str], dict]:
    correction = ""
    last_validation = {}
    for attempt in range(1, max_attempts + 1):
        prompt = translate_scene(client, scene, correction)
        validation = validate_scene(client, scene, prompt)
        last_validation = validation
        local_errors = local_audit(prompt)
        ok = bool(validation.get("ok", False)) and not local_errors
        if ok:
            return prompt, [], validation
        missing = validation.get("missing", [])
        invented = validation.get("invented", [])
        correction = "; ".join([
            *(f"missing: {x}" for x in missing),
            *(f"invented: {x}" for x in invented),
            *local_errors,
        ]) or "semantic mismatch"
        if attempt == max_attempts:
            return prompt, ["SEMANTIC_VALIDATION_FAILED", *local_errors], validation
    return "", ["NO_RESULT"], last_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="AI YouTube Stories Prompt Pipeline V3.2")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N selected scenes; 0 = all")
    parser.add_argument("--start", type=int, default=1, help="First scene id to process")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    scenes = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(scenes, list):
        print("ERROR: scenes.json must contain a JSON array", file=sys.stderr)
        return 2

    client = OllamaClient(model=args.model) if args.model else OllamaClient()
    output = []
    audit_lines = [
        "AI YOUTUBE STORIES - PROMPT AUDIT V3.2",
        "Source of truth: structured Ukrainian scene fields.",
        "Legacy scene.prompt is intentionally ignored by the translator.",
        "Final Stable Diffusion prompts must be English-only and <= 58 words.",
        "",
    ]

    selected = [s for s in scenes if int(s.get("id", 0)) >= args.start]
    if args.limit > 0:
        selected = selected[:args.limit]
    selected_ids = {s.get("id") for s in selected}

    for scene in scenes:
        if scene.get("id") not in selected_ids:
            output.append(deepcopy(scene))
            continue

        sid = scene.get("id")
        print(f"[V3.2] scene {sid}: translating + validating...", flush=True)
        try:
            prompt, errors, validation = process_scene(client, scene)
            item = deepcopy(scene)
            item["prompt_v3_2"] = prompt
            item["prompt"] = prompt
            item["prompt_source"] = "structured_scene_v3_2"
            item["prompt_validation"] = validation
            item["prompt_errors"] = errors
            output.append(item)

            status = "PASS" if not errors else "FAIL"
            audit_lines.append(
                f"SCENE {sid}: {status} | words={len(words(prompt))} | "
                f"missing={validation.get('missing', [])} | invented={validation.get('invented', [])} | errors={errors}"
            )
            audit_lines.append(f"  PROMPT: {prompt}")
        except Exception as exc:
            item = deepcopy(scene)
            item["prompt_v3_2"] = ""
            item["prompt"] = ""
            item["prompt_source"] = "structured_scene_v3_2"
            item["prompt_validation"] = {"ok": False, "missing": [], "invented": []}
            item["prompt_errors"] = [f"PIPELINE_ERROR:{type(exc).__name__}:{exc}"]
            output.append(item)
            audit_lines.append(f"SCENE {sid}: FAIL | PIPELINE_ERROR:{type(exc).__name__}:{exc}")

    processed = sum(1 for s in output if s.get("prompt_source") == "structured_scene_v3_2")
    passed = sum(1 for s in output if s.get("prompt_source") == "structured_scene_v3_2" and not s.get("prompt_errors"))
    failed = processed - passed
    audit_lines += ["", f"SUMMARY: processed={processed}, passed={passed}, failed={failed}"]

    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    args.audit.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    print(f"\nDONE: {args.output}")
    print(f"AUDIT: {args.audit}")
    print(f"PASS={passed} FAIL={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
