#!/usr/bin/env python3
"""Differential test: the Rust port against the Python package's own pure logic.

Extracts the pure functions verbatim from ../laya/ (no torch needed), generates one
shared corpus of cases, runs both implementations over it and compares the results.

    python3 scripts/diff_against_python.py [--corpus-only] [--verbose]

Exit code 0 means every case matched.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
REPO = os.path.dirname(CRATE)
sys.path.insert(0, os.path.join(CRATE, "target", "examples"))


# --------------------------------------------------------------------- reference
def extract_reference(tmpdir):
    """Copy the pure-logic modules out of the Python package, verbatim."""
    import pathlib

    src = pathlib.Path(REPO, "laya")
    out = pathlib.Path(tmpdir)

    router = (src / "router.py").read_text().replace("from .lang import analyse", "from lang import analyse")
    (out / "router.py").write_text(router)
    for name in ("lang.py", "presets.py", "email.py"):
        (out / name).write_text((src / name).read_text())

    common = (src / "common.py").read_text()
    wanted = ["QTYPES", "QTYPE_NAMES", "render_criterion", "render_options",
              "confidence_from_probs", "temp_bucket", "ece_score"]
    lines = common.split("\n")
    blocks, i = [], 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^(def |QTYPES|QTYPE_NAMES)", line) and any(
            line.startswith("def " + n + "(") or line.startswith(n) for n in wanted
        ):
            block = [line]
            i += 1
            while i < len(lines) and (lines[i].startswith((" ", "\t")) or lines[i].strip() == ""):
                block.append(lines[i])
                i += 1
            while block and block[-1].strip() == "":
                block.pop()
            blocks.append("\n".join(block))
        else:
            i += 1
    header = ('"""Pure functions extracted verbatim from laya/common.py (no torch)."""\n'
              "import json\nimport math\nfrom typing import Dict, List\n\nimport numpy as np\n\n")
    (out / "common_pure.py").write_text(header + "\n\n".join(blocks) + "\n")
    return str(out)


# --------------------------------------------------------------------- corpus
TEXTS = [
    "The customer was charged twice and wants a refund.",
    "Հայերեն", "ՀԱՅԵՐԵՆ", "։֊",
    "Le client a été facturé deux fois et demande un remboursement.",
    "ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।",
    "お客様は二重に請求されたため返金を希望しています。",
    "客户被重复扣款要求退款",
    "고객이 두 번 청구되어 환불을 원합니다",
    "تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال",
    "வாடிக்கையாளரிடம் இருமுறை கட்டணம் வசூலிக்கப்பட்டது",
    "С клиента дважды сняли деньги и он хочет возврат",
    "ลูกค้าถูกเรียกเก็บเงินสองครั้งและต้องการเงินคืน",
    "Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων",
    "הלקוח חויב פעמיים ורוצה החזר כספי",
    "", "12345 6789", "   ", "\n\t ",
    "mixed Հայերեն and english text here",
    "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die Rechnung",
    "Ich wurde zweimal fuer Rechnung 4411 belastet, bitte erstatten Sie den Betrag.",
    "Me cobraron dos veces la factura 4411, por favor devuelvanme el dinero.",
    "Le client a ete facture deux fois et il demande un remboursement pour la facture",
    "O cliente foi cobrado duas vezes e quer o dinheiro de volta por favor",
    "Il cliente è stato addebitato due volte e vuole il rimborso della fattura per favore",
    "De klant is twee keer belastet en wil het geld terug voor de factuur alstublieft",
    "refund me", "refund", "ok",
    "Please refund the duplicate charge on invoice 4411 today because we have been waiting",
    "मुझसे दो बार शुल्क लिया गया", "二重に請求されました", "두 번 청구되었습니다",
    "emoji 🎉 and english", "tab\tseparated\twords here now",
    "Ünïcödé àccênts în Énglish-ish text with enough words to score",
]

STATES = [
    None, "", "plain english text about a refund request", 12345, True,
    {}, {"body": "I was charged twice, please refund."},
    {"subject": "नमस्ते", "body": "ग्राहक से दो बार शुल्क लिया गया"},
    {"a": {"b": ["deep", {"c": "deeper"}]}},
    ["x", {"y": "z"}, ["nested", "list"]],
    {"n": 3, "body": "charged twice", "flag": False, "nil": None},
    {"deep": {"1": {"2": {"3": {"4": {"5": {"6": {"7": "too deep"}}}}}}}},
    {"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung"},
    {"body": "Հայերեն"}, {"body": "お客様は二重に請求されました"},
    {"empty": "", "body": "hello there friend"},
    [None, "", "english words in a list"],
]

EMAIL_BODIES = [
    None, "", "simple body",
    "Hi, we were billed twice for invoice 4411 in March.\nCould you refund the duplicate? Thanks.",
    "line one\r\nline two\r\n",
    "literal\\nnewline here",
    "quoted below\n\nOn Mon, Mar 3 2026 at 10:00, billing@acme.com wrote:\n> old message\n> more old",
    "--\nJane Doe\nAccount Manager",
    "Best regards,\nBob",
    "sent from my iPhone",
    "CONFIDENTIALITY NOTICE: This email is intended solely for the addressee.\n\nreal content here",
    "If you have received this email in error please delete it.",
    "________\nFrom: someone@example.com\nstuff",
    "---Original Message---\nstuff",
    "a" * 5000,
    "multiple\n\n\n\nblank   lines\t\there",
    "  leading and trailing  ",
    "> only quoted\n> lines",
    "short\n--\nsig",
    "x" * 50 + "\n" + "y" * 50 + "\nregards,",
]

QUESTIONS = [
    {"t": "choice", "ins": "Which team?", "crit": {"billing": None, "tech": None}},
    {"t": "choice", "ins": "x", "crit": {"billing": {"desc": "payments"}, "tech": None, "sales": ""}},
    {"t": "choice", "ins": "x", "crit": {"zero": 0, "no": False, "f": 1.5}},
    {"t": "choice", "ins": "x", "crit": {"a": "first", "b": None}},
    {"t": "choice", "ins": "x", "crit": {"u": "münchen", "n": [1, 2], "d": {"k": "v"}}},
    {"t": "score", "ins": "How urgent?", "crit": ["low", "high"]},
    {"t": "score", "ins": "x", "crit": [{"d": "low"}, "high", 2, None, True]},
    {"t": "noul", "ins": "Is this phishing?"},
    {"t": "noul", "ins": "x", "crit": None},
    {"t": "noul", "ins": "x", "crit": {"true": "yes it is", "false": "no"}},
    {"t": "noul", "ins": "x", "crit": {"true": {"desc": "phishing"}, "false": {"desc": "legit"}}},
    {"t": "noul", "ins": "x", "crit": {"true": "", "false": None}},
    {"t": "noul", "ins": "x", "crit": {"false": "only false"}},
    {"t": "choice", "ins": "unicode ünïcödé", "crit": {"ä": "ö", "ü": None}},
]

PUBLIC_QUESTIONS = [
    {"dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "tech": None}}},
    {"dept": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "tech"]}},
    {"dept": {"type": "choice", "instructions": "Which team?", "criteria": ["a", 3, True, None]}},
    {"u": {"type": "score", "instructions": "How urgent?", "criteria": ["no", "soon", "now"]}},
    {"c": {"type": "noul", "instructions": "Does the user threaten to cancel?"}},
    {"c": {"type": "noul", "instructions": {"structured": ["instructions", 1]}}},
    {"p": {"type": "noul", "instructions": "phishing?",
           "criteria": {"true": "phishing, scam, or fraud", "false": "a legitimate email"}}},
    {"a": {"type": "choice", "instructions": "x", "criteria": {"k": {"nested": {"deep": [1, 2]}}}}},
    {"z": {"type": "choice", "instructions": "x"}},
    {"z": {"type": "score", "instructions": "x"}},
    {"z": {"type": "noul"}},
    {"z": {"type": "bogus", "instructions": "x"}},
    {"z": {"instructions": "x"}},
    {"z": "not an object"},
    {},
]

TD_IDS = {
    "agent_trace_observability": ["action", "needs_review", "outcome", "risk", "urgency"],
    "customer_service": ["action", "category", "churn_risk", "needs_human", "urgency"],
    "invoice_processing": ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"],
    "security_incidents": ["credential_compromise", "disposition", "severity", "true_positive", "urgency"],
}

ALIASES = ["en", "laya", "default", "multi", "ml", "laya-multilingual", "typed", "typed_decisions",
           "laya-typed-decisions", "decisions", "english", "multilingual", "typed-decisions",
           "English", "  EN  ", "ML", "nope", "", "laya ", "TYPED"]

PROBS = [
    ([0.5, 0.5], 2), ([1.0, 0.0], 2), ([0.7, 0.3], 2), ([0.9, 0.05, 0.05], 3),
    ([0.25, 0.25, 0.25, 0.25], 4), ([0.99, 0.01], 2), ([0.6, 0.4], 2),
    ([0.34, 0.33, 0.33], 3), ([1.0], 1), ([0.0, 0.0, 1.0], 3),
    ([0.2, 0.2, 0.2, 0.2, 0.2], 5), ([0.5, 0.5], 1), ([0.8, 0.1, 0.1], 2),
]

ECE_CASES = [
    ([0.9, 0.8], [1.0, 0.0], 15), ([0.9, 0.85], [1.0, 1.0], 15),
    ([0.9, 0.85], [0.9, 0.85], 15), ([], [], 15),
    ([0.1, 0.5, 0.9], [0.0, 1.0, 1.0], 15), ([0.5], [1.0], 10),
    ([0.05, 0.15, 0.25, 0.95], [0.0, 0.0, 1.0, 1.0], 20),
    ([0.3, 0.7], [1.0, 0.0], 2), ([0.99, 0.01, 0.5], [1.0, 0.0, 1.0], 5),
    ([0.0, 1.0], [0.0, 1.0], 15),
]


def build_corpus():
    cases = []
    for t in TEXTS:
        cases.append({"op": "detect_script", "text": t})
        cases.append({"op": "script_profile", "text": t})
        cases.append({"op": "guess_latin_language", "text": t})
    for s in STATES:
        cases.append({"op": "state_text", "state": s})
        cases.append({"op": "is_english", "state": s})
        cases.append({"op": "analyse", "state": s})
    for b in EMAIL_BODIES:
        cases.append({"op": "clean_email_body", "body": b, "max_chars": 3000})
        cases.append({"op": "clean_email_body", "body": b, "max_chars": 7})
        cases.append({"op": "email_state", "subject": "  Subject  ", "body": b,
                      "sender": "ap@acme.com", "clean": True, "extra": {"tier": "gold", "skip": None}})
        cases.append({"op": "email_state", "subject": None, "body": b, "sender": None,
                      "clean": False, "extra": {}})
    for q in QUESTIONS:
        cases.append({"op": "render_options", "question": q})
    for v in [None, True, False, 0, 1, 3.5, -2, "", "text", [], ["a", "b"], {}, {"d": "x"},
              {"a": [1, {"b": None}]}, "münchen", {"u": "ü"}, [None, True, 1.5]]:
        cases.append({"op": "render_criterion", "value": v})
    for p, k in PROBS:
        cases.append({"op": "confidence_from_probs", "p": p, "k": k})
    for qt in (0, 1, 2):
        for k in (1, 2, 3, 5, 6, 10, 11, 20):
            cases.append({"op": "temp_bucket", "qtype": qt, "k": k})
    for conf, correct, bins in ECE_CASES:
        cases.append({"op": "ece_score", "conf": conf, "correct": correct, "bins": bins})
    for a in ALIASES:
        cases.append({"op": "normalise_name", "name": a})
    for wf, ids in TD_IDS.items():
        cases.append({"op": "match_workflow", "ids": ids})
    cases.append({"op": "match_workflow", "ids": ["urgency", "category"]})
    cases.append({"op": "match_workflow", "ids": TD_IDS["customer_service"] + ["extra"]})
    cases.append({"op": "match_workflow", "ids": []})
    cases.append({"op": "match_workflow", "ids": ["action", "category", "churn_risk", "needs_human"]})

    # routing: every precedence branch, with and without options
    td_customer_service = {i: {"type": "noul", "instructions": "x"} for i in TD_IDS["customer_service"]}
    route_states = [
        None, "", "12345", {"body": "I was charged twice, please refund."},
        {"body": "मुझसे दो बार शुल्क लिया गया"}, {"body": "Հայերեն"},
        {"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung"},
        {"body": "お客様は二重に請求されました"}, {"body": "두 번 청구되었습니다"},
        {"body": "تم خصم المبلغ مرتين"}, {"body": "hello there"},
    ]
    route_questions = PUBLIC_QUESTIONS[:8] + [td_customer_service]
    route_opts = [
        {}, {"default": "multilingual"}, {"auto_task_detection": True},
        {"standalone_repos": True}, {"max_loaded": 2},
        {"models": {"english": "/tmp/en", "multilingual": "/tmp/ml"}},
        {"models": {"english": ["some/repo", "sub"]}},
    ]
    for st in route_states:
        for qs in route_questions:
            for opts in route_opts:
                cases.append({"op": "route", "state": st, "questions": qs, "opts": opts})
    for kw in [{"model": "english"}, {"model": "multilingual"}, {"model": "typed"},
               {"task": "typed_decisions"}, {"task": "typed-decisions"}, {"task": "english"},
               {"lang": "en"}, {"lang": "eng"}, {"lang": "english"}, {"lang": "de"}, {"lang": "pt-BR"},
               {"lang": "EN"}, {"model": "nope"}, {"task": "nope"}, {"lang": ""}]:
        cases.append({"op": "route", "state": {"body": "मुझसे दो बार"},
                      "questions": PUBLIC_QUESTIONS[0], "opts": {}, **kw})
        cases.append({"op": "route", "state": {"body": "hello"},
                      "questions": dict({i: {"type": "noul", "instructions": "x"}
                                         for i in TD_IDS["customer_service"]}),
                      "opts": {"auto_task_detection": True}, **kw})

    for name in ("triage", "email", "guard", "moderation", "router"):
        cases.append({"op": "preset", "name": name})
    cases.append({"op": "preset", "name": "email",
                  "categories": {"a": "first", "b": "second"}})
    cases.append({"op": "preset", "name": "email", "categories": {}})
    return cases


# --------------------------------------------------------------------- python side
def py_run_case(case, ref):
    op = case["op"]
    try:
        if op == "detect_script":
            return {"ok": ref.lang.detect_script(case.get("text") or "")}
        if op == "script_profile":
            return {"ok": ref.lang.script_profile(case.get("text") or "")}
        if op == "guess_latin_language":
            return {"ok": ref.lang.guess_latin_language(case.get("text") or "")}
        if op == "state_text":
            return {"ok": ref.lang.state_text(case.get("state"))}
        if op == "is_english":
            return {"ok": ref.lang.is_english(case.get("state"))}
        if op == "analyse":
            return {"ok": ref.lang.analyse(case.get("state"))}
        if op == "normalise_name":
            return {"ok": ref.router.normalise_name(case.get("name") or "")}
        if op == "match_workflow":
            ids = case.get("ids") or []
            return {"ok": ref.router.match_typed_decisions_workflow({i: {} for i in ids})}
        if op == "render_criterion":
            return {"ok": ref.common_pure.render_criterion(case.get("value"))}
        if op == "render_options":
            return {"ok": ref.common_pure.render_options(case["question"])}
        if op == "confidence_from_probs":
            import numpy as np
            return {"ok": ref.common_pure.confidence_from_probs(np.array(case["p"]), case["k"])}
        if op == "temp_bucket":
            return {"ok": ref.common_pure.temp_bucket(case["qtype"], case["k"])}
        if op == "ece_score":
            import numpy as np
            return {"ok": ref.common_pure.ece_score(np.array(case["conf"]),
                                                    np.array(case["correct"]), case["bins"])}
        if op == "clean_email_body":
            return {"ok": ref.email.clean_email_body(case.get("body"), case.get("max_chars", 3000))}
        if op == "email_state":
            extra = case.get("extra") or {}
            return {"ok": ref.email.email_state(case.get("subject"), case.get("body"),
                                                case.get("sender"), case.get("clean", True), **extra)}
        if op == "route":
            opts = case.get("opts") or {}
            models = opts.get("models")
            r = ref.router.Router(models=models, device=opts.get("device"), token=opts.get("token"),
                                  max_loaded=opts.get("max_loaded", 1), default=opts.get("default", "english"),
                                  auto_task_detection=opts.get("auto_task_detection", False),
                                  standalone_repos=opts.get("standalone_repos", False))
            d = r.route(case.get("state"), case.get("questions"), model=case.get("model"),
                        task=case.get("task"), lang=case.get("lang"))
            return {"ok": dict(d)}
        if op == "preset":
            name = case["name"]
            fn = {"triage": ref.presets.triage_questions, "email": ref.presets.email_questions,
                  "guard": ref.presets.guard_questions, "moderation": ref.presets.moderation_questions,
                  "router": ref.presets.router_questions}[name]
            return {"ok": fn(case["categories"]) if name == "email" and "categories" in case else fn()}
        return {"err": "unknown op %s" % op}
    except Exception as e:  # noqa: BLE001 - the harness compares error-vs-error
        return {"err": "%s: %s" % (type(e).__name__, e)}


# --------------------------------------------------------------------- compare
def jsonify(value):
    """Normalise a Python result the way a JSON round-trip would (tuples become lists)."""
    if isinstance(value, dict):
        return {k: jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    return value


def close(a, b, path, problems):
    # serde_json cannot represent NaN, so the Rust example emits null for it; the value
    # itself is compared by the Rust unit tests.
    if a is None and isinstance(b, float) and math.isnan(b):
        return
    if b is None and isinstance(a, float) and math.isnan(a):
        return
    if isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            problems.append("%s: %r != %r" % (path, a, b))
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return
        if math.isnan(a) and math.isnan(b):
            return
        if abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)):
            return
        problems.append("%s: %r != %r" % (path, a, b))
        return
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        problems.append("%s: type %s != %s (%r vs %r)" % (path, type(a).__name__, type(b).__name__, a, b))
        return
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                problems.append("%s: key %r only on one side (%r vs %r)" % (path, k, a.get(k), b.get(k)))
            else:
                close(a[k], b[k], "%s.%s" % (path, k), problems)
        return
    if isinstance(a, list):
        if len(a) != len(b):
            problems.append("%s: length %d != %d" % (path, len(a), len(b)))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, "%s[%d]" % (path, i), problems)
        return
    if a != b:
        problems.append("%s: %r != %r" % (path, a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--corpus-only", action="store_true")
    args = ap.parse_args()

    cases = build_corpus()
    with tempfile.TemporaryDirectory() as tmp:
        refdir = extract_reference(tmp)
        sys.path.insert(0, refdir)
        import common_pure
        import email as email_ref
        import lang as lang_ref
        import presets as presets_ref
        import router as router_ref

        class ref:
            pass

        ref.common_pure, ref.email, ref.lang, ref.presets, ref.router = (
            common_pure, email_ref, lang_ref, presets_ref, router_ref)

        corpus_path = os.path.join(tmp, "cases.json")
        with open(corpus_path, "w") as f:
            json.dump(cases, f)

        if args.corpus_only:
            print("corpus: %d cases -> %s" % (len(cases), corpus_path))
            return

        rust_out = os.path.join(tmp, "rust.json")
        subprocess.run(["cargo", "run", "--quiet", "--example", "diff", "--",
                        corpus_path, rust_out], cwd=CRATE, check=True)
        with open(rust_out) as f:
            rust_results = json.load(f)

        py_results = [jsonify(py_run_case(c, ref)) for c in cases]

    problems = []
    for i, (case, r, p) in enumerate(zip(cases, rust_results, py_results)):
        if ("err" in r) != ("err" in p):
            problems.append("case %d (%s): error mismatch rust=%r py=%r" % (i, case["op"], r, p))
            continue
        if "err" in r:
            # both errored; the normalise_name message is replicated exactly, the rest is not
            if case["op"] == "normalise_name":
                py_msg = p["err"]
                if py_msg.startswith("ValueError: "):
                    py_msg = py_msg[len("ValueError: "):]
                close(r["err"], py_msg, "case %d.err" % i, problems)
            continue
        close(r["ok"], p["ok"], "case %d (%s)" % (i, case["op"]), problems)

    print("%d cases, %d mismatches" % (len(cases), len(problems)))
    for prob in problems[:40]:
        print("  MISMATCH", prob)
    if len(problems) > 40:
        print("  ... and %d more" % (len(problems) - 40))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
