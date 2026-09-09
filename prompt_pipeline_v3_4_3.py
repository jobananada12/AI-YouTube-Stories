import argparse
import json
import re
from pathlib import Path

from core.llm import OllamaClient

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "scenes.json"
DEFAULT_OUTPUT = BASE_DIR / "scenes_optimized_v3_4_3.json"
DEFAULT_AUDIT = BASE_DIR / "prompt_audit_v3_4_3.txt"
MAX_WORDS = 58
MIN_WORDS = 25
CYRILLIC_RE = re.compile(r"[\u0400-\u052F]")

# Only obvious machine-translation corruption. Never invent or enrich the scene.
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

REQUIREMENTS_SYSTEM = """You are a strict source-fact extractor for cinematic image prompts.
The structured Ukrainian scene is the ONLY source of truth. Ignore legacy scene.prompt completely.

First normalize only obvious machine-translation corruption by meaning. For example, 'серовиртній' means 'сіруватий' (grayish), and 'дверь' means 'двері'. Never add facts.

Build a SOURCE FACT LEDGER. Extract every concrete fact that a generated image prompt must preserve:
- character identity
- exact time or temporal qualifier
- location
- clothing and important properties
- every concrete action as a separate chronological step
- every concrete object and its important properties
- every explicit relation between objects/actions
- visually relevant atmosphere

IMPORTANT:
1. Never collapse a specific action into generic observation.
2. Preserve durations such as 'for several days'.
3. Preserve action order: notice -> open -> climb through -> enter, etc.
4. If the source gives one specific fact, do NOT turn it into an OR choice.
5. Do not invent companions, people, failed attempts, causes, outcomes, objects, clothing, or events.
6. Abstract thoughts may be recorded as abstract and need not be visually literal, but concrete visual facts are mandatory.

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
  "forbidden_inventions": ["..."],
  "source_facts": ["...", "..."]
}

Every concrete action/object/relation must be mandatory=true. Keep requirement text concise, natural English. Do not invent facts."""

TRANSLATOR_SYSTEM = """You write one cinematic Stable Diffusion image prompt from a strict source-fact ledger.

SOURCE FIDELITY IS MORE IMPORTANT THAN STYLE.

NON-NEGOTIABLE:
- Preserve EVERY mandatory concrete action, individually, in chronological order.
- Preserve every mandatory object, object property, and relation.
- Preserve character, exact time, location, clothing, and important temporal qualifiers.
- Do NOT add any person, companion, object, clothing item, event, failed attempt, success/failure, cause, motivation, or outcome absent from the ledger.
- Do NOT transform one fact into an OR choice. Never write alternatives such as 'cut or blocked' when the ledger gives one fact.
- A generic verb cannot replace a specific action. 'observes' does not mean 'waits', 'opens', 'climbs through', or 'enters'.
- If an abstract thought cannot be shown literally, express only the visible consequence or omit it; never invent an event.
- Use natural native English and concrete visual language.
- English only. No labels, no metadata.
- Return JSON only: {"prompt_en":"..."}.
- Target 30-55 words; maximum 58 words."""

VALIDATOR_SYSTEM = """You are the final zero-tolerance semantic validator.
Validate the proposed prompt ONLY against the SOURCE FACT LEDGER.

PASS requires:
- every mandatory concrete action is explicitly present and in chronological order;
- every mandatory object and important property is present;
- every mandatory relation is explicit;
- character, time, location, clothing and mandatory temporal qualifiers are preserved;
- no invented person, companion, object, clothing, event, failed attempt, success/failure, cause or outcome;
- no material contradiction or weakened specificity.

Semantic equivalents are allowed when unmistakable. For example, 'studies' can cover 'observes', but 'observes' cannot cover 'opens'. 'Climbs through the window' covers climbing through a window. 'Enters the room' is not covered merely by being in the room.

If the source says one specific state, a phrase using 'or', 'maybe', 'may be', or an alternative that weakens it is a failure.

Return JSON only:
{"ok":true,"missing":[],"invented":[],"contradictions":[],"order_errors":[],"covered":[]}"""


def normalize_source_text(text):
    text = text or ""
    for bad, good in NORMALIZATION.items():
        text = text.replace(bad, good)
    return text


def normalized_scene(scene):
    out = {}
    for key in ("id", "title", "description", "location", "time", "action", "camera", "mood"):
        out[key] = normalize_source_text(str(scene.get(key, "")))
    return out


def clean_json_text(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_first_json_object(text):
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


def parse_json_object(text):
    candidates = [clean_json_text(text)]
    extracted = extract_first_json_object(text)
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


def words(text):
    return re.findall(r"\b[\w'-]+\b", text or "", flags=re.UNICODE)


def clean_prompt(text):
    text = re.sub(r"[\r\n]+", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(' \"“”')
    text = re.sub(r"^(?:prompt|description)\s*:\s*", "", text, flags=re.I)
    return text.rstrip(" .") + "." if text else ""


def word_count_ok(text):
    n = len(words(text))
    return n >= MIN_WORDS and n <= MAX_WORDS


def scene_payload(scene):
    return json.dumps(scene, ensure_ascii=False, indent=2)


def req_payload(req):
    return json.dumps(req, ensure_ascii=False, indent=2)


def normalize_requirements(req):
    req = dict(req or {})
    for key in ("character", "clothing", "atmosphere", "forbidden_inventions", "source_facts"):
        value = req.get(key, [])
        req[key] = value if isinstance(value, list) else ([str(value)] if value else [])
    for key, prefix in (("actions", "a"), ("objects", "o"), ("relations", "r")):
        value = req.get(key, [])
        fixed = []
        if isinstance(value, list):
            for i, item in enumerate(value, 1):
                if isinstance(item, dict):
                    fixed.append({
                        "id": str(item.get("id") or f"{prefix}{i}"),
                        "text": str(item.get("text", "")),
                        "mandatory": bool(item.get("mandatory", True)),
                    })
                elif item:
                    fixed.append({"id": f"{prefix}{i}", "text": str(item), "mandatory": True})
        req[key] = fixed
    req.setdefault("time", "")
    req.setdefault("location", "")
    return req


def extract_requirements(client, scene, attempts=4):
    s = normalized_scene(scene)
    request = f"""Extract the complete source-fact ledger from this authoritative structured scene.

{s and scene_payload(s)}

Return JSON only. Do not use the legacy prompt field. Do not invent anything."""
    last = None
    for _ in range(attempts):
        try:
            return normalize_requirements(parse_json_object(client.generate(request, system=REQUIREMENTS_SYSTEM)))
        except Exception as exc:
            last = exc
    raise ValueError(f"requirements extraction failed: {last}")


def generate_prompt(client, scene, req, correction=""):
    s = normalized_scene(scene)
    correction_block = f"\nPREVIOUS FAILURE — FIX EXACTLY THESE PROBLEMS:\n{correction}\n" if correction else ""
    request = f"""Create ONE coherent cinematic image prompt.{correction_block}

SOURCE FACT LEDGER:
{req_payload(req)}

AUTHORITATIVE STRUCTURED SCENE:
{scene_payload(s)}

The ledger and structured scene are the only sources of truth. Never use scene.prompt. Return JSON only."""
    data = parse_json_object(client.generate(request, system=TRANSLATOR_SYSTEM))
    prompt = clean_prompt(str(data.get("prompt_en", "")))
    if not prompt:
        raise ValueError("prompt_en is empty")
    return prompt


def validate_llm(client, req, prompt):
    request = f"""Validate with zero tolerance.

SOURCE FACT LEDGER:
{req_payload(req)}

PROPOSED PROMPT:
{prompt}

Return JSON only. Do not infer facts that are not explicitly present."""
    result = parse_json_object(client.generate(request, system=VALIDATOR_SYSTEM))
    result.setdefault("ok", False)
    for key in ("missing", "invented", "contradictions", "order_errors", "covered"):
        if not isinstance(result.get(key), list):
            result[key] = []
    return result


def semantic_requirements(req):
    """Create deterministic checks for high-risk facts without requiring literal wording."""
    checks = []
    for item in req.get("actions", []):
        if not item.get("mandatory", True):
            continue
        t = item["text"].lower()
        groups = []
        if re.search(r"several days|multiple days|over days|for days", t):
            groups.append(("duration_several_days", ["several days", "multiple days", "over several days", "over multiple days", "for days"]))
        if re.search(r"\bwait", t):
            groups.append(("wait", ["wait", "waiting", "waits"]))
        if re.search(r"search|look for|seek|find another", t):
            groups.append(("search", ["search", "searches", "looking for", "look for", "seeks"]))
        if re.search(r"\bopen", t):
            groups.append(("open", ["opens", "open", "opening"]))
        if re.search(r"climb.*through|through.*window", t):
            groups.append(("climb_through", ["climbs through", "climb through", "climbing through"]))
        if re.search(r"\benter", t):
            groups.append(("enter", ["enters", "entering", "enter the", "into the room"]))
        if re.search(r"observe|examine|study|look at|inspect", t):
            groups.append(("observe", ["observes", "observe", "examines", "examining", "studies", "study", "inspects", "inspect"]))
        checks.append((item["id"], groups))
    return checks


def deterministic_audit(req, prompt):
    p = prompt.lower()
    errors = []

    # Character.
    for c in req.get("character", []):
        if str(c).lower() == "alexander" and "alexander" not in p:
            errors.append("MISSING_CHARACTER:Alexander")

    # Time: recognize common exact forms.
    time = str(req.get("time", "")).lower()
    if "23:00" in time or "11:00 pm" in time or "11 pm" in time:
        if not re.search(r"\b11(?::00)?\s*pm\b|\b23:00\b", p):
            errors.append("MISSING_TIME:23:00")
    if "12:00" in time or "12 pm" in time or "noon" in time:
        if not re.search(r"\b12(?::00)?\s*pm\b|\bnoon\b|\b12:00\b", p):
            errors.append("MISSING_TIME:12:00")

    # Clothing: semantic category pairs, not brittle exact sentences.
    clothing = " ".join(str(x).lower() for x in req.get("clothing", []))
    if "gray jacket" in clothing or "grey jacket" in clothing:
        if not re.search(r"\b(gray|grey)\s+jacket\b", p):
            errors.append("MISSING_CLOTHING:gray_jacket")
    if "blue sweater" in clothing:
        if not re.search(r"\bblue\s+sweater\b", p):
            errors.append("MISSING_CLOTHING:blue_sweater")

    # Concrete objects and their properties. The LLM validator handles broader synonyms.
    for item in req.get("objects", []):
        if not item.get("mandatory", True):
            continue
        t = item["text"].lower()
        if "table" in t and not re.search(r"\btable\b", p):
            errors.append(f"MISSING_OBJECT:{item['id']}:table")
        if "door" in t and not re.search(r"\bdoor\b", p):
            errors.append(f"MISSING_OBJECT:{item['id']}:door")
        if "window" in t and not re.search(r"\bwindow\b", p):
            errors.append(f"MISSING_OBJECT:{item['id']}:window")
        if "book" in t and not re.search(r"\bbooks?\b", p):
            errors.append(f"MISSING_OBJECT:{item['id']}:books")
        if "room" in t and not re.search(r"\broom\b", p):
            errors.append(f"MISSING_OBJECT:{item['id']}:room")

    # Relations with distinctive visual wording.
    for item in req.get("relations", []):
        if not item.get("mandatory", True):
            continue
        t = item["text"].lower()
        if "book" in t and "table" in t:
            if not re.search(r"table[^.]{0,80}(hold|with|contain|book)|book[^.]{0,80}(on|atop|upon)\s+(an?\s+)?table", p):
                errors.append(f"MISSING_RELATION:{item['id']}:books_on_table")
        if "climb" in t and "window" in t:
            if not re.search(r"climb(?:s|ing)?[^.]{0,50}through[^.]{0,30}window", p):
                errors.append(f"MISSING_RELATION:{item['id']}:climb_through_window")

    # High-risk specificity weakening.
    if re.search(r"\b(?:cut|blocked)\s+or\s+(?:blocked|cut)\b|\bmay be\b", p):
        for item in req.get("objects", []) + req.get("relations", []):
            t = item.get("text", "").lower()
            if "blocked" in t and ("may be" in p or " or " in p):
                errors.append("WEAKENED_SPECIFICITY:blocked_fact")
                break

    # Word count and Cyrillic.
    n = len(words(prompt))
    if n < MIN_WORDS:
        errors.append(f"WORD_LIMIT_TOO_SHORT:{n}")
    if n > MAX_WORDS:
        errors.append(f"WORD_LIMIT_EXCEEDED:{n}")
    if CYRILLIC_RE.search(prompt):
        errors.append("CYRILLIC_IN_FINAL_PROMPT")

    return errors


def order_audit(req, prompt):
    """Deterministic ordering for common action verbs. Only flags clear inversions."""
    p = prompt.lower()
    positions = []
    for item in req.get("actions", []):
        t = item.get("text", "").lower()
        anchor = None
        if "open" in t:
            anchor = re.search(r"\bopens?\b|\bopening\b", p)
        elif "climb" in t and "through" in t:
            anchor = re.search(r"\bclimbs? through\b|\bclimbing through\b", p)
        elif "enter" in t:
            anchor = re.search(r"\benters?\b|\bentering\b|\binto the room\b", p)
        elif "search" in t or "look for" in t:
            anchor = re.search(r"\bsearch(?:es|ing)?\b|\blook(?:s|ing)? for\b", p)
        if anchor:
            positions.append((item["id"], anchor.start()))
    errors = []
    for (a, pa), (b, pb) in zip(positions, positions[1:]):
        if pa > pb:
            errors.append(f"ORDER_ERROR:{a}_before_{b}")
    return errors


def audit_summary(scene_id, prompt, req, llm_result, deterministic_errors, attempt):
    status = "PASS" if llm_result.get("ok") and not deterministic_errors else "FAIL"
    return {
        "scene": scene_id,
        "status": status,
        "attempt": attempt,
        "words": len(words(prompt)),
        "prompt": prompt,
        "requirements": req,
        "llm_validation": llm_result,
        "deterministic_errors": deterministic_errors,
    }


def main():
    parser = argparse.ArgumentParser(description="AI-YouTube-Stories strict prompt pipeline V3.4.3")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--limit", type=int, default=0, help="0 = all scenes")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_path = Path(args.audit)

    scenes = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(scenes, dict):
        scenes = scenes.get("scenes", [])
    if args.limit:
        scenes = scenes[:args.limit]

    client = OllamaClient()
    results = []
    audit_lines = [
        "AI-YouTube-Stories Prompt Audit V3.4.3",
        "Strict source-fact ledger + anti-invention validation",
        "",
    ]
    pass_count = 0
    fail_count = 0

    for scene in scenes:
        sid = scene.get("id", "?")
        print(f"[V3.4.3] scene {sid}: source ledger + generate + strict validate...")
        req = extract_requirements(client, scene)
        prompt = ""
        final_validation = {"ok": False, "missing": [], "invented": [], "contradictions": [], "order_errors": [], "covered": []}
        deterministic_errors = ["NOT_GENERATED"]
        last_problems = ""

        for attempt in range(1, 7):
            try:
                prompt = generate_prompt(client, scene, req, last_problems)
                final_validation = validate_llm(client, req, prompt)
                deterministic_errors = deterministic_audit(req, prompt) + order_audit(req, prompt)
                problems = []
                problems.extend(f"LLM_MISSING:{x}" for x in final_validation.get("missing", []))
                problems.extend(f"LLM_INVENTED:{x}" for x in final_validation.get("invented", []))
                problems.extend(f"LLM_CONTRADICTION:{x}" for x in final_validation.get("contradictions", []))
                problems.extend(f"LLM_ORDER:{x}" for x in final_validation.get("order_errors", []))
                problems.extend(deterministic_errors)
                if final_validation.get("ok") and not deterministic_errors and word_count_ok(prompt):
                    break
                last_problems = "\n".join(problems) or "VALIDATION_FAILED: make every mandatory fact explicit and remove all inventions."
            except Exception as exc:
                deterministic_errors = [f"GENERATION_ERROR:{exc}"]
                last_problems = deterministic_errors[0]

        result = audit_summary(sid, prompt, req, final_validation, deterministic_errors, attempt)
        results.append(result)
        if result["status"] == "PASS":
            pass_count += 1
        else:
            fail_count += 1

        audit_lines.append(f"SCENE {sid}: {result['status']} attempt={attempt} words={result['words']}")
        audit_lines.append(f"PROMPT: {prompt}")
        audit_lines.append(f"DETERMINISTIC: {json.dumps(deterministic_errors, ensure_ascii=False)}")
        audit_lines.append(f"LLM: {json.dumps(final_validation, ensure_ascii=False)}")
        audit_lines.append("")

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_lines.append(f"SUMMARY: PASS={pass_count} FAIL={fail_count}")
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8")

    print(f"\nDONE: {output_path}")
    print(f"AUDIT: {audit_path}")
    print(f"PASS={pass_count} FAIL={fail_count}")


if __name__ == "__main__":
    main()
