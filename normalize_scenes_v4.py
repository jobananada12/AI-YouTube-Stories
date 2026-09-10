import argparse
import json
import re
from pathlib import Path

from core.llm import OllamaClient

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "scenes.json"
DEFAULT_OUTPUT = BASE_DIR / "scenes_normalized_v4.json"
DEFAULT_AUDIT = BASE_DIR / "scene_normalization_audit_v4.txt"

FIELDS = ("title", "description", "location", "time", "action", "camera", "mood")
CYRILLIC_RE = re.compile(r"[\u0400-\u052F]")

NORMALIZER_SYSTEM = r'''You are a conservative Ukrainian story-scene editor.

The supplied structured scene fields are the ONLY source of truth:
title, description, location, time, action, camera, mood.
The legacy field scene.prompt is forbidden and must be ignored completely.

Your task is NOT to rewrite the story. Clean obvious machine-translation damage and grammar so the scene reads as natural Ukrainian while preserving the exact story facts.

Rules:
1. Preserve every concrete fact, action, object, person, property, relationship, location, time and uncertainty.
2. Preserve chronology and do not merge distinct actions.
3. Fix obvious machine-translation artifacts, malformed words, Russianisms and broken grammar ONLY when the intended meaning is unambiguous from the supplied scene.
4. Never invent a person, object, event, motive, cause, result, emotion, clothing item or relationship.
5. Never use scene.prompt as a correction source, even if it sounds cleaner.
6. If a phrase is genuinely ambiguous, keep its meaning conservative and record the ambiguity in uncertainties rather than guessing.
7. Pronouns such as "вони" may refer to the already established protagonist; do not create another character unless the source explicitly introduces one.
8. Do not modernize, embellish, dramatize or stylistically rewrite the scene.
9. Keep titles concise and faithful to the source.
10. Output natural Ukrainian.

Return JSON only:
{
  "title":"...",
  "description":"...",
  "location":"...",
  "time":"...",
  "action":"...",
  "camera":"...",
  "mood":"...",
  "uncertainties":["..."]
}'''

AUDITOR_SYSTEM = r'''You are a strict fact-preservation auditor.

Compare the ORIGINAL structured scene with the NORMALIZED scene. The legacy scene.prompt is forbidden.

Report only real semantic changes:
- omitted concrete facts;
- added unsupported facts;
- changed action, chronology, object, person, location, time or relation;
- incorrectly resolved genuine ambiguity.

Grammar cleanup, spelling fixes, natural Ukrainian phrasing and harmless paraphrases are NOT errors.

Return JSON only:
{"ok":true,"omitted":[],"added":[],"changed":[],"ambiguity_errors":[]}
Set ok=false if any real issue exists.'''


def clean_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start >= 0:
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
            else:
                if ch == '"':
                    quoted = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        text = text[start:i + 1]
                        break
    return text.strip()


def parse_json(text):
    candidate = clean_json(text)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        value = json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
    if not isinstance(value, dict):
        raise ValueError("LLM response is not a JSON object")
    return value


def scene_payload(scene):
    return "\n\n".join(f"[{field}]\n{str(scene.get(field, '')).strip()}" for field in FIELDS)


def normalize_result(original, data):
    result = dict(original)
    for field in FIELDS:
        value = str(data.get(field, original.get(field, ""))).strip()
        if not value:
            value = str(original.get(field, "")).strip()
        result[field] = value
    result.pop("prompt", None)
    uncertainties = data.get("uncertainties", [])
    if isinstance(uncertainties, str):
        uncertainties = [uncertainties] if uncertainties.strip() else []
    result["normalization_uncertainties"] = [str(x).strip() for x in uncertainties if str(x).strip()]
    return result


def process_scene(client, scene, attempts):
    original = {field: scene.get(field, "") for field in FIELDS}
    request = f"Normalize this scene conservatively. Preserve facts exactly.\n\nORIGINAL STRUCTURED SCENE:\n{scene_payload(scene)}"
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            normalized = normalize_result(scene, parse_json(client.generate(request, system=NORMALIZER_SYSTEM)))
            audit_request = (
                f"ORIGINAL:\n{scene_payload(original)}\n\nNORMALIZED:\n"
                f"{scene_payload(normalized)}\n\nAudit semantic preservation."
            )
            audit = parse_json(client.generate(audit_request, system=AUDITOR_SYSTEM))
            omitted = audit.get("omitted", []) or []
            added = audit.get("added", []) or []
            changed = audit.get("changed", []) or []
            ambiguity_errors = audit.get("ambiguity_errors", []) or []
            if bool(audit.get("ok")) and not any((omitted, added, changed, ambiguity_errors)):
                return normalized, {"status": "PASS", "attempts": attempt, "issues": []}
            issues = {
                "omitted": omitted,
                "added": added,
                "changed": changed,
                "ambiguity_errors": ambiguity_errors,
            }
            request = (
                "Re-normalize conservatively. Fix ONLY the audited semantic errors and preserve all original facts. "
                f"AUDIT ERRORS: {json.dumps(issues, ensure_ascii=False)}\n\nORIGINAL STRUCTURED SCENE:\n{scene_payload(original)}"
            )
            last_error = issues
        except Exception as exc:
            last_error = str(exc)
    raise ValueError(f"normalization failed after {attempts} attempts: {last_error}")


def main():
    parser = argparse.ArgumentParser(description="Universal AI YouTube Stories Scene Normalizer V4")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args()

    scenes = json.loads(args.input.read_text(encoding="utf-8"))
    if isinstance(scenes, dict):
        scenes = scenes.get("scenes", [])
    if args.limit > 0:
        scenes = scenes[:args.limit]

    client = OllamaClient()
    output = []
    audit_lines = [
        "AI YouTube Stories — Scene Normalization V4",
        "Source: structured fields only; legacy scene.prompt ignored.",
        "Universal/story-agnostic conservative normalization.",
        "",
    ]

    passed = failed = 0
    for index, scene in enumerate(scenes, 1):
        scene_id = scene.get("id", index)
        print(f"[V4 NORMALIZE] scene {scene_id}: normalize -> semantic audit...")
        try:
            normalized, audit = process_scene(client, scene, args.attempts)
            output.append(normalized)
            passed += 1
            audit_lines.append(f"SCENE {scene_id}: PASS (attempts={audit['attempts']})")
        except Exception as exc:
            output.append(dict(scene))
            failed += 1
            audit_lines.append(f"SCENE {scene_id}: FAIL — {exc}")
            print(f"  FAIL: {exc}")

    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_lines.extend(["", f"TOTAL={len(output)} PASS={passed} FAIL={failed}"])
    args.audit.write_text("\n".join(audit_lines), encoding="utf-8")
    print(f"\nDONE: {args.output}")
    print(f"AUDIT: {args.audit}")
    print(f"PASS={passed} FAIL={failed}")


if __name__ == "__main__":
    main()
