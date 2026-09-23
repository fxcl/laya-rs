#!/usr/bin/env python3
"""Sequence-building parity: `build_sequence` in Rust against `laya.common`."""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
REPO = os.path.dirname(CRATE)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, REPO)

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/laya_models/laya")

CASES = [
    {"state": {"message": "I was charged twice for invoice 4411, please refund."},
     "questions": {"dept": {"type": "choice", "instructions": "Which team?",
                            "criteria": {"billing": "invoices, payments, refunds",
                                         "tech": "bugs, outages"}}}},
    {"state": {"message": "मुझसे दो बार शुल्क लिया गया"},
     "questions": {"u": {"type": "noul", "instructions": "Does the customer ask for money back?"}}},
    {"state": "plain string state",
     "questions": {"s": {"type": "score", "instructions": "How urgent?",
                         "criteria": ["no", "soon", "now"]}}},
    {"state": {"a": {"b": ["nested", {"c": "deep"}]}},
     "questions": {"q": {"type": "choice", "instructions": "x", "criteria": ["a", "b", "c"]}}},
]


def main():
    import laya
    from laya.common import build_sequence
    from laya.agent import Agent

    agent = Agent(MODEL_DIR, device="cpu")
    tok = agent.tok
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)

    py = []
    for case in CASES:
        per = []
        for qid, qdef in case["questions"].items():
            q = agent._to_internal(qdef)
            ids, markers = build_sequence(tok, case["state"], q, max_len, head_max_len)
            per.append({"qid": qid, "ids": ids, "markers": markers})
        py.append(per)

    with tempfile.TemporaryDirectory() as tmp:
        cases_path = os.path.join(tmp, "cases.json")
        out_path = os.path.join(tmp, "rust.json")
        with open(cases_path, "w") as f:
            json.dump(CASES, f)
        subprocess.run(["cargo", "run", "--quiet", "--example", "dumpseq", "--",
                        cases_path, out_path, MODEL_DIR], cwd=CRATE, check=True)
        with open(out_path) as f:
            rust = json.load(f)

    problems = []
    for i, (p_case, r_case) in enumerate(zip(py, rust)):
        for p, r in zip(p_case, r_case):
            if p["ids"] != r["ids"]:
                problems.append("case %d/%s: ids differ\n     py   %s\n     rust %s"
                                % (i, p["qid"], p["ids"][:30], r["ids"][:30]))
            if p["markers"] != r["markers"]:
                problems.append("case %d/%s: markers %s vs %s"
                                % (i, p["qid"], p["markers"], r["markers"]))
    print("%d cases, %d mismatches" % (len(CASES), len(problems)))
    for p in problems:
        print("  MISMATCH", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
