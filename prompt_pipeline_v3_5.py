import argparse
import json
from pathlib import Path

import prompt_pipeline_v3_4_4 as base

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "scenes.json"
DEFAULT_OUTPUT = BASE_DIR / "scenes_optimized_v3_5.json"
DEFAULT_AUDIT = BASE_DIR / "prompt_audit_v3_5.txt"

# V3.5: repair only known machine-translation corruption.
# IMPORTANT: these repairs restore the intended meaning; they do not add story facts.
base.NORMALIZATION.update({
    "серовиртній": "сіруватий",
    "серовиртна": "сірувата",
    "серовиртний": "сіруватий",
    "серовиртні": "сіруваті",
    "дверь": "двері",
    "дверю": "двері",
    "дверню": "двері",
    "способы вхіду": "способи входу",
    "способы входу": "способи входу",
    "одна з верхніх вікон": "одне з верхніх вікон",
    "розкривають це вікно": "відкриває це вікно",
    "Вони бачать": "Олександр бачить",
    "Вони розкривають": "Олександр відкриває",
    "старий стол": "старий стіл",
    "обокладують до кімнати": "пробирається до кімнати через це вікно",
    "обокладують": "пробирається",
    "очевидно не вдаючись впливати на двері": "очевидно, не намагаючись впливати на двері",
})

# The V3.4.4 validator was too literal. V3.5 explicitly accepts faithful
# semantic translations such as "пробирається через вікно" -> "climbs through the window".
base.VALIDATOR_SYSTEM = r'''You are a strict semantic validator of a generated image prompt.

Validate ONLY against the SOURCE FACT LEDGER. The legacy scene.prompt is forbidden.

A faithful natural-English paraphrase is NOT an invention. Judge meaning, not word identity.

PASS requires:
- every mandatory concrete action is explicitly present and in chronological order;
- every mandatory object, property and relation is present;
- character, time, location, clothing and mandatory duration are preserved;
- no unsupported person, companion, object, clothing, event, failure, success, cause, motivation or outcome;
- no contradiction or material weakening of specificity.

IMPORTANT EQUIVALENCES:
- "пробирається/потрапляє до кімнати через вікно" may be translated as "climbs through the window into the room";
- "відкриває вікно" = "opens the window";
- "шукає інші способи входу" = "searches for another way/entry";
- "спостерігає та розглядає" = "observes/examines/studies";
- "старий стіл" = "old table". Do not change it into an unrelated object such as desk unless the source itself supports that;
- "книги про екологію та природу" requires both ecology and nature;
- books and table are separate objects, and "books on a table" preserves their relation;
- pronouns he/his/him refer to the already named Alexander and are never invented people.

Do NOT flag harmless paraphrases. Do NOT flag a semantic translation merely because the English wording differs from the ledger wording.
Do flag a genuinely new event or unsupported detail.

Return JSON only:
{"ok":true,"missing":[],"invented":[],"contradictions":[],"order_errors":[],"covered":[]}
If any missing/invented/contradictions/order_errors exist, ok MUST be false.'''

base.GENERATOR_SYSTEM = r'''You write ONE coherent cinematic Stable Diffusion prompt from a strict source ledger.

SOURCE FIDELITY is more important than style.

NON-NEGOTIABLE:
- Preserve EVERY mandatory action individually and in chronological order.
- Preserve EVERY mandatory object, property and relation.
- Preserve character, time, location, clothing and mandatory duration.
- Translate the SOURCE MEANING into natural native English. Do not translate broken Ukrainian literally.
- Do NOT add any person, companion, object, clothing, event, failed attempt, success/failure, cause, motivation or outcome absent from the source.
- Do NOT weaken a concrete fact into an alternative. Preserve source uncertainty only when the source is uncertain.
- "observes" cannot replace "opens", "waits", "searches", "climbs through" or "enters".
- If the source says a person gets into a room through a window, explicitly show the transition through the window.
- Use "table" when the source says table; do not invent "desk".
- Use "books about ecology and nature" when both topics are stated.
- A pronoun refers to Alexander; never invent a second person.
- English only. No labels. No metadata.
- 28-58 words when possible. Never add invented padding just to reach a word count.
- Return JSON only: {"prompt_en":"..."}.'''


def audit_ledger_directly(scene, ledger):
    """Cheap structural guard against the V3.4.4 failure mode where the LLM ledger
    silently drops an action. It does not attempt to understand the whole story;
    it checks high-value action/object anchors present in the authoritative fields."""
    text = " ".join(str(scene.get(k, "")) for k in ("description", "action", "camera")).lower()
    ledger_text = json.dumps(ledger, ensure_ascii=False).lower()
    errors = []

    checks = [
        (("шукати", "шукає", "способи входу", "спосіб входу"), "search/other entry"),
        (("вікн",), "window"),
        (("відкрива", "відкривають", "розкрива"), "open window"),
        (("обокладують", "пробирається", "потрапляє", "входить"), "enter/get through"),
        (("стіл",), "table"),
        (("книг",), "books"),
        (("еколог",), "ecology"),
        (("природ",), "nature"),
    ]
    for source_terms, label in checks:
        if any(term in text for term in source_terms) and not any(term in ledger_text for term in source_terms):
            # A translated/corrected ledger may use a different Ukrainian form.
            if label == "enter/get through" and any(x in ledger_text for x in ("кімнати", "кімната")):
                continue
            errors.append(f"LEDGER_DROPPED:{label}")
    return errors


def process_scene_v35(client, scene, attempts=4):
    ledger = base.extract_ledger(client, scene, attempts=attempts)
    direct_errors = audit_ledger_directly(scene, ledger)
    if direct_errors:
        # One controlled re-extraction with an explicit completeness warning.
        warning = (
            "The previous ledger dropped authoritative facts. Rebuild it completely. "
            + "; ".join(direct_errors)
            + ". Every concrete action/object from description and action must remain."
        )
        payload = base.source_payload(scene)
        request = f"{warning}\n\nSTRUCTURED SOURCE:\n{payload}"
        ledger = base.normalize_ledger(base.parse_json(client.generate(request, system=base.LEDGER_SYSTEM)))

    correction = ""
    last_det = []
    last_llm = {}
    prompt = ""
    for attempt in range(1, attempts + 1):
        prompt = base.generate_prompt(client, scene, ledger, correction)
        det = base.deterministic_audit(ledger, prompt)
        try:
            llm = base.validate_llm(client, ledger, prompt)
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
        llm_text = base.format_llm_errors(llm)
        if llm_text:
            correction_parts.append("LLM VALIDATOR ERRORS:\n" + llm_text)
        correction = "\n\n".join(correction_parts)

    return {
        "status": "FAIL",
        "attempts": attempts,
        "prompt": prompt,
        "requirements": ledger,
        "deterministic_errors": last_det,
        "llm_validation": last_llm,
    }


def main():
    parser = argparse.ArgumentParser(description="AI YouTube Stories prompt pipeline V3.5")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args()

    client = base.OllamaClient()
    scenes = base.load_scenes(args.input)
    if args.limit > 0:
        scenes = scenes[:args.limit]

    results = []
    audit_lines = [
        "AI YouTube Stories — Prompt Pipeline V3.5",
        "Source: structured scenes.json fields only; legacy scene.prompt is forbidden.",
        "V3.5: semantic normalization + ledger completeness guard + semantic validator.",
        "",
    ]
    passed = failed = 0

    for scene in scenes:
        sid = scene.get("id")
        print(f"[V3.5] scene {sid}: normalize -> complete ledger -> generate -> strict validate...")
        try:
            result = process_scene_v35(client, scene, attempts=args.attempts)
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
        audit_lines.append(f"PROMPT: {result.get('prompt', '')}")
        audit_lines.append("DETERMINISTIC: " + ("OK" if not result.get("deterministic_errors") else " | ".join(result["deterministic_errors"])))
        llm = result.get("llm_validation") or {}
        audit_lines.append("LLM: " + ("OK" if llm.get("ok") else base.format_llm_errors(llm) or "FAIL/NO DETAILS"))
        audit_lines.append("LEDGER: " + json.dumps(result.get("requirements", {}), ensure_ascii=False))
        audit_lines.append("")

    output = {
        "version": "3.5",
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
