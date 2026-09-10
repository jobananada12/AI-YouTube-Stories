import argparse
import json
import re
from pathlib import Path

from core.llm import OllamaClient

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "scenes.json"
DEFAULT_OUTPUT = BASE_DIR / "scenes_optimized_v4.json"
DEFAULT_AUDIT = BASE_DIR / "prompt_audit_v4.txt"
MAX_WORDS = 58
MIN_WORDS = 20
CYRILLIC_RE = re.compile(r"[\u0400-\u052F]")

# V4 IS STORY-AGNOSTIC.
# There are intentionally NO character names, places, objects, translation fixes,
# timestamps, or action rules from any particular story in this file.

LEDGER_SYSTEM = r'''You are a zero-hallucination source-fact extractor for an image-prompt pipeline.

The structured scene fields are the ONLY source of truth:
description, location, time, action, camera, mood.
The legacy field scene.prompt is FORBIDDEN and must be ignored completely.

Your job is to build a COMPLETE FACT LEDGER, not to write a prompt.

Universal rules:
1. Read every authoritative field separately.
2. Preserve every concrete action as its own chronological item. Never collapse distinct actions into a vague verb such as "observes".
3. Preserve who does each action, explicit duration, time, location, clothing, object properties, spatial relations, and concrete visual details.
4. Preserve source uncertainty. If the source is uncertain, keep the uncertainty. If it states a fact, do not turn it into a possibility.
5. Do not invent people, objects, events, motives, causes, outcomes, emotions, clothing, or relationships.
6. Pronouns normally refer to an already established character; do not create a second person unless the source explicitly introduces one.
7. Treat obvious machine-translation or grammatical corruption as wording noise ONLY when the intended meaning is unambiguous from the surrounding source. Repair wording without adding facts. If meaning is ambiguous, preserve the ambiguity and mark it.
8. Camera and mood become mandatory visual facts only when they describe something that must visibly appear. Do not turn a camera instruction into a story event.
9. Abstract thoughts are not mandatory visual events unless the source explicitly makes them visible or actionable.
10. Do not use the legacy prompt even if it contains cleaner English.

Return JSON only:
{
  "characters": [{"id":"c1","text":"...","mandatory":true}],
  "time": "...",
  "location": "...",
  "clothing": [{"id":"cl1","text":"...","mandatory":true}],
  "actions": [{"id":"a1","order":1,"text":"...","mandatory":true}],
  "objects": [{"id":"o1","text":"...","mandatory":true}],
  "properties": [{"id":"p1","text":"...","mandatory":true}],
  "relations": [{"id":"r1","text":"...","mandatory":true}],
  "visual_details": [{"id":"v1","text":"...","mandatory":true}],
  "atmosphere": ["..."],
  "uncertainties": ["..."],
  "source_facts": ["..."],
  "forbidden_inventions": ["..."],
  "field_coverage": {
    "description":"...","location":"...","time":"...","action":"...","camera":"...","mood":"..."
  }
}

Every concrete fact that affects what is shown in the image should be represented. Keep the ledger concise but complete.'''

LEDGER_AUDIT_SYSTEM = r'''You audit a proposed source-fact ledger against the authoritative structured scene.

The structured fields description, location, time, action, camera, mood are authoritative. scene.prompt is forbidden.

This is a completeness audit, not a rewriting task.

Find only:
- concrete facts omitted from the ledger;
- concrete facts distorted or changed in meaning;
- actions merged so that a distinct action disappeared;
- lost chronology, duration, object properties, or spatial relations.

Do NOT report normal paraphrases, grammar cleanup, pronoun resolution, or harmless wording differences.
Do NOT invent missing facts yourself.

Return JSON only:
{"ok":true,"missing":[],"distorted":[],"notes":[]}
Set ok=false if missing or distorted contains a real concrete omission/distortion.'''

GENERATOR_SYSTEM = r'''You write ONE coherent cinematic Stable Diffusion image prompt from a source-fact ledger.

SOURCE FIDELITY is the highest priority. The image prompt must depict the source, not a generic reinterpretation.

Rules:
1. Include EVERY mandatory action individually and in chronological order when the scene contains multiple actions.
2. Include EVERY mandatory character, object, property, relation, location, time, clothing detail, duration, and concrete visual detail.
3. Preserve uncertainty exactly when it is part of the source. Never invent a resolution of an uncertain fact.
4. Never add a person, companion, object, prop, clothing item, event, cause, motive, emotion, success, failure, or outcome that is not supported by the ledger.
5. Do not replace a specific action with a weaker generic action. For example, observing is not the same as opening, waiting, searching, climbing, entering, leaving, touching, speaking, or picking something up.
6. Use natural native English. Translate the meaning, not broken source wording literally.
7. Keep concrete object identity and relations exact: do not substitute a similar object.
8. Use visually concrete language suitable for Stable Diffusion, but never use style words to hide missing source facts.
9. English only. No labels, no metadata, no commentary.
10. Prefer 28-58 words. Never add invented content just to reach a word count.

Return JSON only: {"prompt_en":"..."}.'''

VALIDATOR_SYSTEM = r'''You are a strict semantic validator of a generated Stable Diffusion prompt.

Validate ONLY against the supplied SOURCE FACT LEDGER. Do not use outside knowledge and do not use the legacy scene.prompt.

PASS only when:
- every mandatory action is explicitly represented and in the correct order;
- every mandatory character, object, property, relation, location, time, clothing detail, duration, and visual detail is represented;
- source uncertainty is not incorrectly resolved;
- no unsupported person, object, event, clothing, motive, cause, emotion, success/failure, or outcome was added;
- no concrete fact is materially weakened, contradicted, or replaced by a different object/action.

Judge semantic meaning, not word identity. Natural paraphrases are valid.
Pronouns referring to an already named character are not new people.
A generic observation verb cannot stand in for a distinct concrete action.

Return JSON only:
{"ok":true,"missing":[],"invented":[],"contradictions":[],"order_errors":[],"covered":[]}
If any real issue exists in any error array, ok MUST be false.'''


def clean_json_text(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text):
    text = clean_json_text(text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    quoted = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
            continue
        if ch == '"':
            quoted = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_json(text):
    candidates = [clean_json_text(text)]
    extracted = extract_json_object(text)
    if extracted and extracted not in candidates:
        candidates.append(extracted)
    last = None
    for candidate in candidates:
        variants = [candidate, re.sub(r",\s*([}\]])", r"\1", candidate)]
        for variant in variants:
            try:
                value = json.loads(variant)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError as exc:
                last = exc
    raise ValueError(f"Invalid JSON from Ollama: {last}")


def as_list(value):
    if isinstance(value, list):
        return value
    return [value] if value not in (None, "") else []


def normalize_items(value, prefix):
    result = []
    for i, item in enumerate(as_list(value), 1):
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            result.append({
                "id": str(item.get("id") or f"{prefix}{i}"),
                "order": item.get("order", i),
                "text": text,
                "mandatory": bool(item.get("mandatory", True)),
            })
        elif str(item).strip():
            result.append({"id": f"{prefix}{i}", "order": i, "text": str(item).strip(), "mandatory": True})
    return result


def normalize_ledger(data):
    data = dict(data or {})
    data["characters"] = normalize_items(data.get("characters"), "c")
    data["clothing"] = normalize_items(data.get("clothing"), "cl")
    data["actions"] = normalize_items(data.get("actions"), "a")
    data["objects"] = normalize_items(data.get("objects"), "o")
    data["properties"] = normalize_items(data.get("properties"), "p")
    data["relations"] = normalize_items(data.get("relations"), "r")
    data["visual_details"] = normalize_items(data.get("visual_details"), "v")
    for key in ("atmosphere", "uncertainties", "source_facts", "forbidden_inventions"):
        data[key] = [str(x).strip() for x in as_list(data.get(key)) if str(x).strip()]
    data["time"] = str(data.get("time", "")).strip()
    data["location"] = str(data.get("location", "")).strip()
    data["field_coverage"] = dict(data.get("field_coverage") or {})
    return data


def source_payload(scene):
    # Deliberately excludes scene.prompt.
    fields = ("description", "location", "time", "action", "camera", "mood")
    return "\n\n".join(f"[{key}]\n{str(scene.get(key, '')).strip()}" for key in fields)


def ledger_payload(ledger):
    return json.dumps(ledger, ensure_ascii=False, indent=2)


def extract_ledger(client, scene, attempts):
    payload = source_payload(scene)
    request = f"Extract a complete source-fact ledger. Inspect every field independently before answering.\n\n{payload}"
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            ledger = normalize_ledger(parse_json(client.generate(request, system=LEDGER_SYSTEM)))
            audit_request = f"AUTHORITATIVE STRUCTURED SCENE:\n{payload}\n\nPROPOSED LEDGER:\n{ledger_payload(ledger)}\n\nAudit completeness."
            audit = parse_json(client.generate(audit_request, system=LEDGER_AUDIT_SYSTEM))
            missing = as_list(audit.get("missing"))
            distorted = as_list(audit.get("distorted"))
            if bool(audit.get("ok")) and not missing and not distorted:
                return ledger
            request = (
                f"Rebuild the ledger completely. Previous audit found real omissions/distortions: "
                f"missing={missing}; distorted={distorted}. Preserve every concrete fact from every field.\n\n{payload}"
            )
            last_error = ValueError(f"ledger audit failed: missing={missing}; distorted={distorted}")
        except Exception as exc:
            last_error = exc
    raise ValueError(f"ledger extraction failed after {attempts} attempts: {last_error}")


def generate_prompt(client, scene, ledger, correction=""):
    correction_text = f"\nPrevious validation errors to fix:\n{correction}\n" if correction else ""
    request = f"""Create one cinematic image prompt from this ledger.{correction_text}

SOURCE FACT LEDGER:
{ledger_payload(ledger)}

AUTHORITATIVE STRUCTURED SOURCE:
{source_payload(scene)}

Remember: scene.prompt is forbidden. Return JSON only."""
    data = parse_json(client.generate(request, system=GENERATOR_SYSTEM))
    prompt = str(data.get("prompt_en", "")).strip()
    prompt = re.sub(r"[\r\n]+", " ", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip().strip('"“”')
    if not prompt:
        raise ValueError("prompt_en is empty")
    return prompt


def deterministic_audit(ledger, prompt):
    errors = []
    if not prompt.strip():
        errors.append("EMPTY_PROMPT")
        return errors
    if CYRILLIC_RE.search(prompt):
        errors.append("PROMPT_NOT_ENGLISH_ONLY")
    word_count = len(re.findall(r"\b[\w'-]+\b", prompt))
    if word_count > MAX_WORDS:
        errors.append(f"TOO_MANY_WORDS:{word_count}>{MAX_WORDS}")
    if word_count < MIN_WORDS:
        errors.append(f"TOO_FEW_WORDS:{word_count}<{MIN_WORDS}")
    return errors


def format_validation_errors(result):
    parts = []
    for key in ("missing", "invented", "contradictions", "order_errors"):
        values = as_list(result.get(key))
        if values:
            parts.append(f"{key}: " + "; ".join(map(str, values)))
    return "\n".join(parts)


def validate_llm(client, ledger, prompt):
    request = f"SOURCE FACT LEDGER:\n{ledger_payload(ledger)}\n\nGENERATED PROMPT:\n{prompt}\n\nValidate semantic coverage and invention."
    return parse_json(client.generate(request, system=VALIDATOR_SYSTEM))


def process_scene(client, scene, attempts):
    ledger = extract_ledger(client, scene, attempts)
    correction = ""
    last_det = []
    last_llm = {}
    prompt = ""
    for attempt in range(1, attempts + 1):
        prompt = generate_prompt(client, scene, ledger, correction)
        det = deterministic_audit(ledger, prompt)
        try:
            llm = validate_llm(client, ledger, prompt)
        except Exception as exc:
            llm = {"ok": False, "missing": [f"validator_error:{exc}"], "invented": [], "contradictions": [], "order_errors": [], "covered": []}
        last_det, last_llm = det, llm
        if not det and bool(llm.get("ok")) and not any(as_list(llm.get(k)) for k in ("missing", "invented", "contradictions", "order_errors")):
            return {"status":"PASS","attempts":attempt,"prompt":prompt,"requirements":ledger,"deterministic_errors":[],"llm_validation":llm}
        correction_parts = []
        if det:
            correction_parts.append("DETERMINISTIC ERRORS:\n" + "\n".join(det))
        validation_text = format_validation_errors(llm)
        if validation_text:
            correction_parts.append("SEMANTIC VALIDATION ERRORS:\n" + validation_text)
        correction = "\n\n".join(correction_parts)
    return {"status":"FAIL","attempts":attempts,"prompt":prompt,"requirements":ledger,"deterministic_errors":last_det,"llm_validation":last_llm}


def main():
    parser = argparse.ArgumentParser(description="Universal AI YouTube Stories Prompt Pipeline V4")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args()

    client = OllamaClient()
    scenes = json.loads(args.input.read_text(encoding="utf-8"))
    if isinstance(scenes, dict):
        scenes = scenes.get("scenes", [])
    if args.limit > 0:
        scenes = scenes[:args.limit]

    results = []
    audit_lines = [
        "AI YouTube Stories — Universal Prompt Pipeline V4",
        "Source of truth: structured scene fields only.",
        "Legacy scene.prompt: NEVER USED.",
        "Story-specific rules: NONE.",
        "",
    ]
    passed = failed = 0
    for scene in scenes:
        sid = scene.get("id")
        print(f"[V4] scene {sid}: ledger -> audit -> generate -> semantic validate...")
        try:
            result = process_scene(client, scene, args.attempts)
        except Exception as exc:
            result = {"status":"FAIL","attempts":0,"prompt":"","requirements":{},"deterministic_errors":[f"PIPELINE_ERROR:{exc}"],"llm_validation":{}}
        results.append({"id":sid, **result})
        if result["status"] == "PASS":
            passed += 1
        else:
            failed += 1
        audit_lines.append(f"SCENE {sid}: {result['status']} attempts={result['attempts']}")
        audit_lines.append(f"PROMPT: {result.get('prompt','')}")
        audit_lines.append("DETERMINISTIC: " + ("OK" if not result.get("deterministic_errors") else " | ".join(result["deterministic_errors"])))
        llm = result.get("llm_validation") or {}
        audit_lines.append("LLM: " + ("OK" if llm.get("ok") else format_validation_errors(llm) or "FAIL/NO DETAILS"))
        audit_lines.append("LEDGER: " + json.dumps(result.get("requirements",{}), ensure_ascii=False))
        audit_lines.append("")

    output = {
        "version":"4.0",
        "pipeline":"universal_story_agnostic",
        "source_of_truth":"structured_scene_fields",
        "legacy_prompt_used":False,
        "story_specific_rules":False,
        "summary":{"processed":len(results),"pass":passed,"fail":failed},
        "scenes":results,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    args.audit.write_text("\n".join(audit_lines), encoding="utf-8")
    print(f"\nDONE: {args.output}")
    print(f"AUDIT: {args.audit}")
    print(f"PASS={passed} FAIL={failed}")


if __name__ == "__main__":
    main()
