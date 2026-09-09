import argparse
import json
import re
from pathlib import Path

from core.llm import OllamaClient

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "scenes.json"
DEFAULT_OUTPUT = BASE_DIR / "scenes_optimized_v3_4_4.json"
DEFAULT_AUDIT = BASE_DIR / "prompt_audit_v3_4_4.txt"
MAX_WORDS = 58
CYRILLIC_RE = re.compile(r"[\u0400-\u052F]")

# Repair only obvious machine-translation corruption. Do not enrich meaning.
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
    "розкривають це вікно": "відкриває це вікно",
    "вони бачать": "він бачить",
    "вони розкривають": "він відкриває",
    "старий стол": "старий стіл",
}

LEDGER_SYSTEM = """You are a ZERO-HALLUCINATION source-fact extractor.

The structured Ukrainian fields are the ONLY source of truth. The legacy field scene.prompt is forbidden and must be ignored completely.

Normalize only obvious machine-translation corruption. Example: серовиртній -> сіруватий. Never invent or improve facts.

Read EVERY structured field separately: description, location, time, action, camera, mood.
Build a COMPLETE SOURCE FACT LEDGER. Do not summarize away concrete facts.

Rules:
1. Every concrete action must be a separate chronological action item. Never collapse open/climb/enter/search/wait into generic observation.
2. Preserve durations such as "for several days".
3. Preserve explicit object properties and relations, e.g. books ON a table, upper window, closed door, room behind door.
4. Preserve uncertainty only when the source itself is uncertain. If source says one concrete fact, do not turn it into OR/maybe.
5. Never create a companion, second person, extra object, failed attempt, success/failure, cause, motivation, or outcome unless explicitly present.
6. Pronouns such as he/his refer to the already named character; they are NOT new people.
7. Camera/mood may be marked atmospheric only if they are visually relevant. Do not turn camera wording into an event.
8. Abstract thoughts may be recorded but are not mandatory visual events unless the source makes them concrete.

Return JSON only:
{
  "character": ["Alexander"],
  "time": "...",
  "location": "...",
  "clothing": ["..."],
  "actions": [{"id":"a1","text":"...","mandatory":true}],
  "objects": [{"id":"o1","text":"...","mandatory":true}],
  "relations": [{"id":"r1","text":"...","mandatory":true}],
  "atmosphere": ["..."],
  "source_facts": ["..."],
  "forbidden_inventions": ["..."],
  "coverage_notes": ["description: ...", "location: ...", "time: ...", "action: ...", "camera: ...", "mood: ..."]
}

Every concrete action/object/relation is mandatory=true. Keep text concise but complete."""

LEDGER_AUDIT_SYSTEM = """You audit whether a proposed source-fact ledger completely represents the structured source.

The structured Ukrainian scene is authoritative. scene.prompt is forbidden.

Find ONLY omissions or distortions of concrete source facts. Do not call paraphrases omissions. Do not treat pronouns as new people. Do not invent facts yourself.

Especially check:
- every action in description/action, individually and in order;
- duration/time;
- every concrete object;
- object properties and relations;
- entrance/window/door/open/climb/enter events;
- table/books and what the books are about;
- camera and mood only when visually concrete.

Return JSON only:
{"ok":true,"missing_from_ledger":[],"distorted_in_ledger":[]}
Set ok=false if either array is non-empty."""

GENERATOR_SYSTEM = """You write ONE cinematic Stable Diffusion prompt from a strict source ledger.

SOURCE FIDELITY is more important than style.

NON-NEGOTIABLE:
- Preserve EVERY mandatory action individually and in chronological order.
- Preserve EVERY mandatory object, property, and relation.
- Preserve character, time, location, clothing, and mandatory temporal qualifiers.
- Do NOT add any person, companion, object, clothing, event, failed attempt, success/failure, cause, motivation, or outcome absent from the source.
- Do NOT weaken a concrete fact into an alternative. Never write "or", "maybe", or "may be" for a concrete source fact.
- "Observes" cannot replace "opens", "waits", "climbs through", or "enters".
- A pronoun refers to the already named character; never invent another person.
- Use natural native English and concrete visual language.
- English only. No labels. No metadata.
- 28-58 words when possible. Do not add invented padding just to reach a word count.
- Return JSON only: {"prompt_en":"..."}."""

VALIDATOR_SYSTEM = """You are a strict semantic validator of a generated image prompt.

Validate ONLY against the SOURCE FACT LEDGER. Do not use outside knowledge.

PASS requires:
- every mandatory concrete action is explicitly present and in chronological order;
- every mandatory object/property/relation is present;
- character, time, location, clothing and mandatory duration are preserved;
- no unsupported concrete person, companion, object, clothing, event, failure, success, cause, motivation, or outcome;
- no contradiction or material weakening of specificity.

Important semantic distinctions:
- observe/study/examine are not open/wait/search/climb/enter;
- noticing a window is not opening it;
- being in a room is not necessarily climbing through a window or entering it;
- books and table are separate objects;
- "books about ecology and nature" is stronger than only "books about ecology";
- "books on a table" must preserve the table relation.

Do NOT flag normal pronouns such as he/his/him as invented people. Do NOT flag harmless paraphrases as inventions.

Return JSON only:
{"ok":true,"missing":[],"invented":[],"contradictions":[],"order_errors":[],"covered":[]}
If any missing/invented/contradictions/order_errors exist, ok MUST be false."""


def normalize_text(text):
    text = text or ""
    for bad, good in NORMALIZATION.items():
        text = text.replace(bad, good)
    return text


def normalized_scene(scene):
    # Intentionally excludes legacy scene.prompt.
    return {
        key: normalize_text(str(scene.get(key, "")))
        for key in ("id", "title", "description", "location", "time", "action", "camera", "mood")
    }


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
        for variant in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                value = json.loads(variant)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError as exc:
                last = exc
    raise ValueError(f"Invalid JSON from Ollama: {last}")


def words(text):
    return re.findall(r"\b[\w'-]+\b", text or "", flags=re.UNICODE)


def clean_prompt(text):
    text = re.sub(r"[\r\n]+", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(' \"“”')
    text = re.sub(r"^(?:prompt|description)\s*:\s*", "", text, flags=re.I)
    return text.rstrip(" .") + "." if text else ""


def ensure_list(value):
    if isinstance(value, list):
        return value
    return [value] if value else []


def normalize_ledger(data):
    data = dict(data or {})
    for key in ("character", "clothing", "atmosphere", "source_facts", "forbidden_inventions", "coverage_notes"):
        data[key] = [str(x) for x in ensure_list(data.get(key, [])) if str(x).strip()]
    for key, prefix in (("actions", "a"), ("objects", "o"), ("relations", "r")):
        fixed = []
        for i, item in enumerate(ensure_list(data.get(key, [])), 1):
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                fixed.append({"id": str(item.get("id") or f"{prefix}{i}"), "text": text, "mandatory": bool(item.get("mandatory", True))})
            elif str(item).strip():
                fixed.append({"id": f"{prefix}{i}", "text": str(item).strip(), "mandatory": True})
        data[key] = fixed
    data["time"] = str(data.get("time", "")).strip()
    data["location"] = str(data.get("location", "")).strip()
    return data


def source_payload(scene):
    s = normalized_scene(scene)
    # Explicitly print each authoritative field so the model cannot silently skip action details.
    return "\n".join([
        f"[description]\n{s['description']}",
        f"[location]\n{s['location']}",
        f"[time]\n{s['time']}",
        f"[action]\n{s['action']}",
        f"[camera]\n{s['camera']}",
        f"[mood]\n{s['mood']}",
    ])


def ledger_payload(ledger):
    return json.dumps(ledger, ensure_ascii=False, indent=2)


def extract_ledger(client, scene, attempts=4):
    payload = source_payload(scene)
    request = f"""Extract the COMPLETE source-fact ledger from the following structured scene.

{payload}

Do not use or request scene.prompt. Check each field one by one before returning JSON."""
    last = None
    for _ in range(attempts):
        try:
            ledger = normalize_ledger(parse_json(client.generate(request, system=LEDGER_SYSTEM)))
            audit_request = f"""STRUCTURED SOURCE:\n{payload}\n\nPROPOSED LEDGER:\n{ledger_payload(ledger)}\n\nAudit completeness only."""
            audit = parse_json(client.generate(audit_request, system=LEDGER_AUDIT_SYSTEM))
            missing = ensure_list(audit.get("missing_from_ledger", []))
            distorted = ensure_list(audit.get("distorted_in_ledger", []))
            if bool(audit.get("ok", False)) and not missing and not distorted:
                return ledger
            last = ValueError(f"ledger incomplete: missing={missing}; distorted={distorted}")
        except Exception as exc:
            last = exc
    raise ValueError(f"source ledger extraction failed: {last}")


def generate_prompt(client, scene, ledger, correction=""):
    correction_block = f"\nPREVIOUS FAILURE — fix ONLY these issues:\n{correction}\n" if correction else ""
    request = f"""Create ONE coherent cinematic Stable Diffusion prompt.{correction_block}

SOURCE FACT LEDGER:
{ledger_payload(ledger)}

AUTHORITATIVE STRUCTURED SOURCE:
{source_payload(scene)}

The legacy scene.prompt is forbidden. Do not invent anything. Return JSON only."""
    data = parse_json(client.generate(request, system=GENERATOR_SYSTEM))
    prompt = clean_prompt(str(data.get("prompt_en", "")))
    if not prompt:
        raise ValueError("prompt_en is empty")
    return prompt


def contains_cyrillic(text):
    return bool(CYRILLIC_RE.search(text or ""))


def deterministic_audit(ledger, prompt):
    p = prompt.lower()
    errors = []

    # Character.
    for c in ledger.get("character", []):
        cl = c.lower()
        if cl in {"alexander", "oleksandr"} and not re.search(r"\b(alexander|oleksandr)\b", p):
            errors.append("MISSING_CHARACTER:Alexander")

    # Time / duration.
    t = ledger.get("time", "").lower()
    if any(x in t for x in ("23:00", "11:00 pm", "11 pm")) and not re.search(r"\b11(?::00)?\s*pm\b|\b23:00\b", p):
        errors.append("MISSING_TIME:23:00")
    if any(x in t for x in ("12:00", "12 pm", "noon")) and not re.search(r"\b12(?::00)?\s*pm\b|\bnoon\b|\b12:00\b", p):
        errors.append("MISSING_TIME:12:00")

    all_actions = ledger.get("actions", [])
    for item in all_actions:
        if not item.get("mandatory", True):
            continue
        text = item["text"].lower()
        aid = item["id"]
        checks = []
        if re.search(r"several days|multiple days|over .*days|for days", text):
            checks.append(("duration", r"several days|multiple days|over (several|multiple) days|for days"))
        if re.search(r"\bwait|unable to settle|cannot calm", text):
            checks.append(("wait", r"wait|waiting|waits|unable to settle|cannot calm|can't calm"))
        if re.search(r"search|look for|seek|find another", text):
            checks.append(("search", r"search|searches|searching|look for|looking for|seeks|seek"))
        if re.search(r"\bopen", text):
            checks.append(("open", r"opens|open|opening"))
        if re.search(r"climb.*through|through.*window", text):
            checks.append(("climb_through", r"climbs? through|climbing through"))
        if re.search(r"\benter|gets into|goes into", text):
            checks.append(("enter", r"enters?|entering|gets? into|goes? into"))
        if re.search(r"observe|examine|study|inspect|look at", text):
            checks.append(("observe", r"observes?|examines?|studies?|inspects?|looks? at"))
        for label, pattern in checks:
            if not re.search(pattern, p):
                errors.append(f"MISSING_ACTION:{aid}:{label}")

    # Objects and high-value properties.
    for item in ledger.get("objects", []):
        if not item.get("mandatory", True):
            continue
        text = item["text"].lower()
        oid = item["id"]
        if "door" in text and "door" not in p:
            errors.append(f"MISSING_OBJECT:{oid}:door")
        if "window" in text and "window" not in p:
            errors.append(f"MISSING_OBJECT:{oid}:window")
        if "table" in text and not re.search(r"\btable\b", p):
            errors.append(f"MISSING_OBJECT:{oid}:table")
        if "book" in text and not re.search(r"\bbooks?\b", p):
            errors.append(f"MISSING_OBJECT:{oid}:books")
        if "ecology" in text and "ecology" not in p:
            errors.append(f"MISSING_PROPERTY:{oid}:ecology")
        if "nature" in text and not re.search(r"\bnature\b", p):
            errors.append(f"MISSING_PROPERTY:{oid}:nature")
        if "closed" in text and "closed" not in p:
            errors.append(f"MISSING_PROPERTY:{oid}:closed")
        if "old" in text and "old" not in p:
            errors.append(f"MISSING_PROPERTY:{oid}:old")
        if "upper" in text and "upper" not in p:
            errors.append(f"MISSING_PROPERTY:{oid}:upper")
        if "broken" in text and not re.search(r"broken|damaged|weathered", p):
            errors.append(f"MISSING_PROPERTY:{oid}:broken")
        if "dim" in text and not re.search(r"dim|dark|poorly lit", p):
            errors.append(f"MISSING_PROPERTY:{oid}:dim")
        if "empty" in text and not re.search(r"empty|bare|vacant", p):
            errors.append(f"MISSING_PROPERTY:{oid}:empty")

    # Relations are checked conservatively with key lexical anchors.
    for item in ledger.get("relations", []):
        if not item.get("mandatory", True):
            continue
        text = item["text"].lower()
        rid = item["id"]
        if "book" in text and "table" in text and not re.search(r"books?.{0,50}(on|atop|on top of)\s+(an?\s+)?old?\s*table|table.{0,50}books?", p):
            errors.append(f"MISSING_RELATION:{rid}:books_on_table")
        if "room" in text and "behind" in text and "plan" in text and "plan" not in p:
            errors.append(f"MISSING_RELATION:{rid}:room_not_on_plan")

    if contains_cyrillic(prompt):
        errors.append("NON_ENGLISH_CYRILLIC")

    n = len(words(prompt))
    if n > MAX_WORDS:
        errors.append(f"WORD_COUNT_TOO_HIGH:{n}>{MAX_WORDS}")
    return errors


def validate_llm(client, ledger, prompt):
    request = f"""Validate this proposed prompt.

SOURCE FACT LEDGER:
{ledger_payload(ledger)}

PROPOSED PROMPT:
{prompt}

Return JSON only."""
    result = parse_json(client.generate(request, system=VALIDATOR_SYSTEM))
    for key in ("missing", "invented", "contradictions", "order_errors", "covered"):
        if not isinstance(result.get(key), list):
            result[key] = []
    result["ok"] = bool(result.get("ok", False)) and not any(result[k] for k in ("missing", "invented", "contradictions", "order_errors"))
    return result


def format_llm_errors(result):
    chunks = []
    for key in ("missing", "invented", "contradictions", "order_errors"):
        vals = result.get(key, [])
        if vals:
            chunks.append(key + ": " + "; ".join(str(x) for x in vals))
    return "\n".join(chunks)


def process_scene(client, scene, max_attempts=4):
    ledger = extract_ledger(client, scene)
    correction = ""
    last_det = []
    last_llm = {}
    prompt = ""
    for attempt in range(1, max_attempts + 1):
        prompt = generate_prompt(client, scene, ledger, correction)
        det = deterministic_audit(ledger, prompt)
        try:
            llm = validate_llm(client, ledger, prompt)
        except Exception as exc:
            llm = {"ok": False, "missing": [f"validator_error:{exc}"], "invented": [], "contradictions": [], "order_errors": [], "covered": []}
        last_det, last_llm = det, llm
        if not det and llm.get("ok", False):
            return {
                "status": "PASS",
                "attempts": attempt,
                "prompt": prompt,
                "requirements": ledger,
                "deterministic_errors": [],
                "llm_validation": llm,
            }
        correction_parts = []
        if det:
            correction_parts.append("DETERMINISTIC ERRORS:\n" + "\n".join(det))
        llm_text = format_llm_errors(llm)
        if llm_text:
            correction_parts.append("LLM VALIDATOR ERRORS:\n" + llm_text)
        correction = "\n\n".join(correction_parts)

    return {
        "status": "FAIL",
        "attempts": max_attempts,
        "prompt": prompt,
        "requirements": ledger,
        "deterministic_errors": last_det,
        "llm_validation": last_llm,
    }


def load_scenes(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "scenes" in data:
        data = data["scenes"]
    if not isinstance(data, list):
        raise ValueError("scenes.json must contain a list of scenes")
    return data


def main():
    parser = argparse.ArgumentParser(description="AI YouTube Stories prompt pipeline V3.4.4")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args()

    client = OllamaClient()
    scenes = load_scenes(args.input)
    if args.limit > 0:
        scenes = scenes[:args.limit]

    results = []
    audit_lines = [
        "AI YouTube Stories — Prompt Pipeline V3.4.4",
        "Source: structured scenes.json fields only; legacy scene.prompt is forbidden.",
        "",
    ]
    passed = failed = 0

    for scene in scenes:
        sid = scene.get("id")
        print(f"[V3.4.4] scene {sid}: normalize -> complete ledger -> generate -> strict validate...")
        try:
            result = process_scene(client, scene, max_attempts=args.attempts)
        except Exception as exc:
            result = {
                "status": "FAIL",
                "attempts": 0,
                "prompt": "",
                "requirements": {},
                "deterministic_errors": [f"PIPELINE_ERROR:{exc}"],
                "llm_validation": {},
            }
        results.append({"id": sid, **result})
        if result["status"] == "PASS":
            passed += 1
        else:
            failed += 1
        audit_lines.append(f"SCENE {sid}: {result['status']} attempts={result['attempts']}")
        audit_lines.append(f"PROMPT: {result.get('prompt','')}")
        audit_lines.append("DETERMINISTIC: " + ("OK" if not result.get("deterministic_errors") else " | ".join(result["deterministic_errors"])))
        llm = result.get("llm_validation") or {}
        audit_lines.append("LLM: " + ("OK" if llm.get("ok") else format_llm_errors(llm) or "FAIL/NO DETAILS"))
        audit_lines.append("LEDGER: " + json.dumps(result.get("requirements", {}), ensure_ascii=False))
        audit_lines.append("")

    output = {
        "version": "3.4.4",
        "source_of_truth": "structured_scene_fields",
        "legacy_prompt_used": False,
        "summary": {"processed": len(results), "pass": passed, "fail": failed},
        "scenes": results,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    args.audit.write_text("\n".join(audit_lines), encoding="utf-8")

    print(f"\nDONE: {args.output}")
    print(f"AUDIT: {args.audit}")
    print(f"PASS={passed} FAIL={failed}")


if __name__ == "__main__":
    main()
