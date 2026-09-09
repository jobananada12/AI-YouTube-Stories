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
AWKWARD_PHRASE_RE = re.compile(r"\b(?:gaze reflecting on personal reflections|internally pondering|reflecting on reflections|pondering internally|reflecting on his own reflections)\b", re.I)

TRANSLATOR_SYSTEM = """You are a professional screenplay-to-image-prompt translator and cinematic continuity editor.
The Ukrainian structured scene is the ONLY source of truth.
Translate the meaning of the WHOLE scene into natural cinematic English. Never translate word-by-word and never concatenate dictionary fragments.
Write one coherent visual description suitable for Stable Diffusion.

HARD CONTINUITY RULES:
- The ACTION field is mandatory and is the core of the prompt. Preserve every important, visually representable action.
- DESCRIPTION and ACTION are complementary. Use both together to understand the intended scene moment. If they describe an ordered sequence, preserve the meaningful sequence naturally.
- Preserve concrete action sequences such as opens -> climbs through -> enters -> looks around when the structured scene supports that sequence.
- Never replace a concrete action with a weaker generic verb such as observes, stands, or looks when the source specifies a stronger action.
- Preserve important people, objects, clothing, location, exact time, and important events. Do not invent, remove, swap, or contradict them.
- The TIME field is authoritative. If it says 12:00, write noon, 12 PM, or another exact equivalent. Never turn 12:00 into early morning or late evening.
- Preserve the clothing category and material/type when supported. A jacket/parka is NOT a trench coat. A sweater is NOT a shirt.
- Use natural English character names. For the Ukrainian name Олександр, use Alexander, not Oleksandr, unless the structured scene explicitly uses a different Latin name.
- Preserve specific clothing words when supported by the source; do not invent a more specific garment.
- Location can be expressed in natural English. Do not invent a different place, but do not force every administrative descriptor into awkward English.
- Context such as 'over several days' is not mandatory in every still-image prompt unless it is visually important to the depicted moment.
- If DESCRIPTION and ACTION refer to different moments of the same short sequence, choose a coherent visual moment that preserves the important concrete action and key objects without inventing a new event.
- Camera information is composition guidance, not a reason to add story facts.
- Abstract thoughts may be omitted or rendered only through visible facial expression, posture, or atmosphere when clearly supported. Never invent a new object or event to visualize a thought.

QUALITY RULES:
- Use concrete cinematic English written by a native English speaker.
- Prefer subject + setting + action + key objects + exact time/lighting + useful mood/composition.
- Avoid literal thought phrases such as 'internally pondering', 'reflecting on his own reflections', or other awkward introspective wording.
- Do not use headings, metadata labels, explanations, Ukrainian/Russian text, or quotation marks around the prompt.
- Do not use the old prompt as a story source.
- Return valid JSON only with exactly {\"prompt_en\": \"...\"}.
- The prompt must be 25-55 words."""

VALIDATOR_SYSTEM = """You are a strict but intelligent semantic continuity and natural-language editor.
Compare the Ukrainian structured scene with the proposed English Stable Diffusion prompt. The Ukrainian structured scene is authoritative.
Return valid JSON only: {\"ok\": true/false, \"missing\": [\"...\"], \"invented\": [\"...\"]}.

VALIDATION PRINCIPLES:
- Compare MEANING, not literal wording. Natural English paraphrases are allowed.
- DESCRIPTION and ACTION are complementary. A prompt is valid when it faithfully represents the same scene moment or a logically supported moment in the described action sequence.
- Do NOT mark a fact as invented when it is explicitly supported anywhere in DESCRIPTION, ACTION, LOCATION, TIME, or the other structured fields.
- Do NOT require every contextual phrase. Duration such as 'over several days' may be omitted from a single still-image prompt unless it is essential to the visible moment.
- Location may be naturally shortened. For example, 'old park in an urban community' may become 'an old park' without being a semantic failure.
- The TIME field is strict and authoritative. A materially different time of day is a failure. Example: 12:00 cannot become 'early morning'.
- Clothing type is strict. 'gray jacket' or 'gray parka' cannot become 'charcoal trench coat'.
- Character identity must remain consistent. Олександр may naturally become Alexander in English; do not require Ukrainian transliteration such as Oleksandr.
- Do not flag valid action sequences as invented. If the source says he opens an upper window and gets into the room, 'opens the upper window and climbs through it' is valid even if the wording is not identical.
- Concrete actions such as opening, entering, climbing, taking, reading, writing, or walking must not be replaced by merely observing or looking when the action is important.
- Flag genuinely missing central objects when their omission materially changes the scene, but do not demand every minor background detail.
- Flag genuinely invented people, objects, clothing, places, times, actions, or events.
- Flag contradictions, wrong time, wrong clothing category, wrong character identity, or materially different action.
- Also fail if the English is clearly unnatural, fragmented, contradictory, or contains awkward literal thought phrases.
- Minor grammatical differences and harmless cinematic phrasing are acceptable.

The goal is a useful professional prompt, not a literal translation test."""


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


def translate_scene(client: OllamaClient, scene: dict, correction: str = "") -> str:
    extra = ""
    if correction:
        extra = f"\nA previous draft failed validation. Correct these exact problems and preserve all other facts. Do not introduce new facts while correcting it: {correction}\n"
    prompt = f"""Create one natural cinematic English image prompt from this complete structured scene.
{extra}
STRUCTURED SCENE (AUTHORITATIVE):
{scene_payload(scene)}

MANDATORY CHECK BEFORE ANSWERING:
1. Extract the main person or people.
2. Extract every important concrete object.
3. Extract the exact action or ordered action sequence from ACTION and DESCRIPTION.
4. Extract the authoritative exact time from TIME.
5. Extract supported clothing details and keep their garment type.
6. Extract the location without inventing a different place.
7. Convert the above into one fluent visual sentence or short paragraph.

REQUIREMENTS:
- ACTION is mandatory. Do not weaken concrete actions into generic observing/looking.
- Preserve action order when multiple actions are explicitly supported.
- Preserve exact time. 12:00 means noon/12 PM, not early morning.
- Preserve exact character identity. Олександр should become Alexander in natural English.
- Preserve important objects, clothing, location, and key events.
- Do not invent cats, photographs, corridors, documents, furniture, rooms, doors, clothing, people, or other details unless supported by the structured scene.
- Do not invent a more specific garment: jacket/parka must not become trench coat.
- Contextual duration may be omitted when it is not visually important to the single frame.
- Do not copy or repair the legacy prompt.
- Do not write labels such as Mood:, Location:, Time:, Camera:, Action:.
- Avoid literal descriptions of thoughts and awkward introspective phrases.
- The result must sound like polished cinematic English written by a native speaker.
- Return JSON only with prompt_en.
- 25-55 words."""
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

Evaluate semantic meaning, not literal word matching. DESCRIPTION and ACTION are complementary. TIME is strict. Clothing category is strict. Do not flag natural paraphrases, valid action sequences, or details explicitly supported elsewhere in the structured scene.
Return JSON only."""
    return parse_json_object(client.generate(request, system=VALIDATOR_SYSTEM))


def process_scene(client: OllamaClient, scene: dict, max_attempts: int = 3) -> tuple[str, list[str], dict]:
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
    parser = argparse.ArgumentParser(description="AI YouTube Stories Prompt Pipeline V3.3")
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
        "AI YOUTUBE STORIES - PROMPT AUDIT V3.3",
        "Source of truth: structured Ukrainian scene fields.",
        "Legacy scene.prompt is intentionally ignored by the translator.",
        "Final Stable Diffusion prompts must be English-only and <= 58 words.",
        "Action preservation is mandatory, including ordered multi-action sequences.",
        "TIME is strict; clothing type is strict; natural semantic paraphrases are allowed.",
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
        print(f"[V3.3] scene {sid}: translating + validating...", flush=True)
        try:
            prompt, errors, validation = process_scene(client, scene)
            item = deepcopy(scene)
            item["prompt_v3_2"] = prompt
            item["prompt"] = prompt
            item["prompt_source"] = "structured_scene_v3_3"
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
            item["prompt_source"] = "structured_scene_v3_3"
            item["prompt_validation"] = {"ok": False, "missing": [], "invented": []}
            item["prompt_errors"] = [f"PIPELINE_ERROR:{type(exc).__name__}:{exc}"]
            output.append(item)
            audit_lines.append(f"SCENE {sid}: FAIL | PIPELINE_ERROR:{type(exc).__name__}:{exc}")

    processed = sum(1 for s in output if s.get("prompt_source") == "structured_scene_v3_3")
    passed = sum(1 for s in output if s.get("prompt_source") == "structured_scene_v3_3" and not s.get("prompt_errors"))
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
