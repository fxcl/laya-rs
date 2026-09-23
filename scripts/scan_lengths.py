#!/usr/bin/env python3
"""Length scan: find the sequence length at which the Rust encoder starts to diverge.

The sliding window is 65, so lengths <= 65 never exercise it. If parity holds up to 65 and
breaks after, the window mask is the culprit; if it breaks earlier, something else is.
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

MODEL_DIR = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/laya_models/laya")
LENGTHS = [8, 32, 60, 64, 66, 70, 100, 125]


def main():
    import torch
    import laya

    agent = laya.load(MODEL_DIR, device="cpu")
    model = agent.model.float()
    tok = agent.tok

    # one long text, truncated to each length; no padding at all
    text = ("The customer was charged twice for invoice 4411 and nobody has answered for "
            "three days. Refund the duplicate today or we are cancelling. ") * 12
    full = tok(text, add_special_tokens=False)["input_ids"]

    results = []
    for n in LENGTHS:
        ids = [tok.cls_token_id] + full[: n - 2] + [tok.sep_token_id]
        ids = ids[:n]
        L = len(ids)
        input_ids = torch.tensor([ids])
        attention_mask = torch.ones((1, L), dtype=torch.long)
        marker_pos = torch.tensor([[3, L - 2]])
        marker_mask = torch.tensor([[True, True]])
        qtype = torch.tensor([0])

        with torch.no_grad():
            hidden = model.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask).last_hidden_state
            logits, _ = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        py_hidden = hidden.float().tolist()
        py_logits = logits.float().tolist()

        batch = {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "marker_pos": marker_pos.tolist(),
            "marker_mask": marker_mask.tolist(),
            "qtype": qtype.tolist(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            bp = os.path.join(tmp, "b.json")
            op = os.path.join(tmp, "r.json")
            with open(bp, "w") as f:
                json.dump(batch, f)
            subprocess.run(["cargo", "run", "--release", "--quiet", "--example", "dumplogits",
                            "--", bp, op, MODEL_DIR], cwd=CRATE, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(op) as f:
                rust = json.load(f)

        def flat(xs):
            out = []
            stack = [xs]
            while stack:
                x = stack.pop()
                if isinstance(x, list):
                    stack.extend(reversed(x))
                else:
                    out.append(x)
            return out

        ph, rh = flat(py_hidden), flat(rust["hidden"])
        worst_h = max(abs(a - b) for a, b in zip(ph, rh))
        pl, rl = flat(py_logits), flat(rust["logits"])
        worst_l = max(abs(a - b) for a, b in zip(pl, rl))
        results.append((L, worst_h, worst_l))
        print("L=%-4d worst hidden delta %10.6f   worst logit delta %10.6f" % (L, worst_h, worst_l))

    bad = [r for r in results if r[1] > 1e-3]
    print("\n%d of %d lengths diverge" % (len(bad), len(results)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
