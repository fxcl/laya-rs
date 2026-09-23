#!/usr/bin/env python3
"""Tight parity check: Rust (f32) against the Python package forced to f32.

The shipped Python path keeps the checkpoint's fp16 weights on CPU, so comparing against
it directly shows ~5e-2 probability deltas on flat distributions. Casting the Python model
to float32 removes that variable: if Rust matches *this* to ~1e-4, the port is correct and
the remaining gap is purely weight precision.

    python3 scripts/check_inference_parity_f32.py [model_dir]
"""
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

sys.path.insert(0, os.path.join(CRATE, "scripts"))
from check_inference_parity import CASES, PRESETS, build_cases  # noqa: E402


def main():
    import laya
    import torch

    cases = build_cases()
    agent = laya.load(MODEL_DIR, device="cpu")
    # force the reference to f32 so only the implementation differs, not the precision
    agent.model = agent.model.float()
    agent.dtype = torch.float32
    print("   python: loaded %s on %s (cast to float32)" % (MODEL_DIR, agent.device), flush=True)

    py = []
    for case in cases:
        py.append(agent.system_one(case["state"], case["questions"]))
        print("   python: %-18s done" % case["label"], flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        cases_path = os.path.join(tmp, "cases.json")
        out_path = os.path.join(tmp, "rust.json")
        with open(cases_path, "w") as f:
            json.dump([{"state": c["state"], "questions": c["questions"]} for c in cases], f)
        subprocess.run(
            ["cargo", "run", "--release", "--quiet", "--example", "infer", "--",
             cases_path, out_path, MODEL_DIR],
            cwd=CRATE, check=True,
        )
        with open(out_path) as f:
            rs = json.load(f)

    problems = []
    worst_prob = 0.0
    worst_logit_proxy = 0.0
    checks = 0
    for case, p, r in zip(cases, py, rs):
        label = case["label"]
        for qid, pa in p["answers"].items():
            ra = r["answers"][qid]
            if pa["type"] == "choice":
                if pa["choice"] != ra["choice"]:
                    problems.append("%s/%s: choice %r vs %r" % (label, qid, pa["choice"], ra["choice"]))
                for k in pa["probabilities"]:
                    d = abs(pa["probabilities"][k] - ra["probabilities"][k])
                    worst_prob = max(worst_prob, d)
                    checks += 1
                    if d > 1e-3:
                        problems.append("%s/%s: p[%s] %.6f vs %.6f"
                                        % (label, qid, k, pa["probabilities"][k], ra["probabilities"][k]))
            elif pa["type"] == "score":
                d = abs(pa["score"] - ra["score"])
                worst_logit_proxy = max(worst_logit_proxy, d)
                if d > 1e-3:
                    problems.append("%s/%s: score %.6f vs %.6f" % (label, qid, pa["score"], ra["score"]))
            else:
                d = abs(pa["noul"] - ra["noul"])
                worst_prob = max(worst_prob, d)
                checks += 1
                if d > 1e-3:
                    problems.append("%s/%s: noul %.6f vs %.6f" % (label, qid, pa["noul"], ra["noul"]))
            if abs(pa["confidence"] - ra["confidence"]) > 1e-3:
                problems.append("%s/%s: confidence %.6f vs %.6f"
                                % (label, qid, pa["confidence"], ra["confidence"]))
            if abs(pa["action"]["act_probability"] - ra["action"]["act_probability"]) > 1e-3:
                problems.append("%s/%s: act_probability %.6f vs %.6f"
                                % (label, qid, pa["action"]["act_probability"],
                                   ra["action"]["act_probability"]))

    print("\n%d cases, %d probability comparisons" % (len(cases), checks))
    print("worst probability delta %.6f, worst score delta %.6f" % (worst_prob, worst_logit_proxy))
    print("%d problems" % len(problems))
    for prob in problems[:30]:
        print("  MISMATCH", prob)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
