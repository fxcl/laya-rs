#!/usr/bin/env python3
"""Model-forward parity on a REAL batch.

Builds the batch with the Python package's own `build_sequence` + `collate_items`, then runs
both models on it. Any difference here is in the encoder or the head -- sequence building,
tokenization and post-processing are all bypassed.

    python3 scripts/check_forward_parity.py [model_dir]
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

STATE = {"message": "I was charged twice for invoice 4411 and nobody has answered for three "
                    "days. Refund the duplicate today or we are cancelling.",
         "account_tier": "enterprise"}
QUESTIONS = {
    "intent": {"type": "choice", "instructions": "What does the customer want in `message`?",
               "criteria": {"refund": "money returned or a duplicate charge reversed",
                            "technical_help": "a bug, outage or integration problem",
                            "billing_question": "a question about an invoice, plan or payment method",
                            "information": "general information, pricing or how-to",
                            "cancellation": "wants to cancel or downgrade",
                            "other": "none of the other options fits"}},
    "is_urgent": {"type": "noul", "instructions": "Does `message` communicate time pressure?"},
    "frustration": {"type": "score", "instructions": "How frustrated does the customer sound?",
                    "criteria": ["calm", "concerned", "annoyed", "angry"]},
}


def main():
    import torch
    import laya
    from laya.common import QTYPES, build_sequence, collate_items

    agent = laya.load(MODEL_DIR, device="cpu")
    model = agent.model.float()
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)

    items = []
    for qid, qdef in QUESTIONS.items():
        q = agent._to_internal(qdef)
        seq, markers = build_sequence(agent.tok, STATE, q, max_len, head_max_len)
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    b = collate_items([items], agent.tok.pad_token_id)
    print("   python: batch %s, kmax=%d" % (tuple(b["input_ids"].shape), b["marker_mask"].shape[1]),
          flush=True)

    with torch.no_grad():
        logits, act = model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                            b["marker_mask"], b["qtype"])
        hidden = model.encoder(input_ids=b["input_ids"],
                               attention_mask=b["attention_mask"]).last_hidden_state
    py_logits = logits.float().tolist()
    py_act = act.float().tolist()
    py_hidden = hidden.float().tolist()

    batch = {
        "input_ids": b["input_ids"].tolist(),
        "attention_mask": b["attention_mask"].tolist(),
        "marker_pos": b["marker_pos"].tolist(),
        "marker_mask": b["marker_mask"].tolist(),
        "qtype": b["qtype"].tolist(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        batch_path = os.path.join(tmp, "batch.json")
        out_path = os.path.join(tmp, "rust.json")
        with open(batch_path, "w") as f:
            json.dump(batch, f)
        subprocess.run(["cargo", "run", "--release", "--quiet", "--example", "dumplogits", "--",
                        batch_path, out_path, MODEL_DIR], cwd=CRATE, check=True)
        with open(out_path) as f:
            rust = json.load(f)

    def flatten(xs):
        out = []
        stack = [xs]
        while stack:
            x = stack.pop()
            if isinstance(x, list):
                stack.extend(reversed(x))
            else:
                out.append(x)
        return out

    worst = 0.0
    worst_at = None
    for name, py, rs in (("logits", py_logits, rust["logits"]),
                         ("act_logits", py_act, rust["act_logits"]),
                         ("hidden", py_hidden, rust["hidden"])):
        pf, rf = flatten(py), flatten(rs)
        for i, (a, bb) in enumerate(zip(pf, rf)):
            d = abs(a - bb)
            if d > worst:
                worst, worst_at = d, "%s[%d] py=%.6f rust=%.6f" % (name, i, a, bb)
    print("worst absolute delta %.6f at %s" % (worst, worst_at))
    for i in range(len(py_logits)):
        print("  row %d py  %s" % (i, ["%.4f" % v for v in py_logits[i]]))
        print("  row %d rust %s" % (i, ["%.4f" % v for v in rust["logits"][i]]))
    sys.exit(0 if worst < 1e-3 else 1)


if __name__ == "__main__":
    main()
