#!/usr/bin/env python3
"""Verify that the Rust port is usable end to end.

Runs every check that does not need a GPU or a network, then optionally the ones that do.
Each check prints PASS/FAIL/SKIP with the evidence, and the script exits non-zero if any
required check fails.

    python3 scripts/verify_usability.py              # offline checks only
    python3 scripts/verify_usability.py --full       # also weights, hub, gpu
    python3 scripts/verify_usability.py --model DIR  # point at a local checkpoint

What "usable" means here, concretely:
  1. it builds (lib, CLI, extension module)
  2. the CLI answers the four things a user asks of it
  3. the Python module imports and exposes the same API as `laya`
  4. inference produces the same numbers as the Python package
  5. it does not fall over on bad input
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
REPO = os.path.dirname(CRATE)
CLI = os.path.join(CRATE, "target", "release", "laya")

FULL = "--full" in sys.argv
MODEL = None
if "--model" in sys.argv:
    MODEL = sys.argv[sys.argv.index("--model") + 1]
elif os.environ.get("LAYA_TEST_MODEL"):
    MODEL = os.environ["LAYA_TEST_MODEL"]
else:
    for cand in [
        os.path.expanduser("~/laya_models/laya"),
        os.path.expanduser("~/laya_models/laya-multilingual"),
    ]:
        if os.path.exists(os.path.join(cand, "model.safetensors")):
            MODEL = cand
            break

results = []


def check(name, ok, detail="", required=True):
    results.append((name, ok, detail, required))
    tag = "PASS" if ok else ("FAIL" if required else "WARN")
    print("[{}] {}{}".format(tag, name, " -- " + detail if detail else ""))
    return ok


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def section(title):
    print("\n=== {} ===".format(title))


# ---------------------------------------------------------------- 1. builds
section("builds")
r = run(["cargo", "build", "--release", "--bin", "laya"], cwd=CRATE)
check("CLI builds", r.returncode == 0, r.stderr.strip().splitlines()[-1] if r.returncode else "")

r = run(["cargo", "build", "--release", "--lib"], cwd=CRATE)
check("library builds", r.returncode == 0)

r = run(["cargo", "clippy", "--all-targets"], cwd=CRATE)
warnings = [l for l in r.stderr.splitlines() if l.startswith("warning: ") and "generated" not in l]
check("clippy is clean", not warnings, "{} warnings".format(len(warnings)))

r = run(["cargo", "fmt", "--all", "--", "--check"], cwd=CRATE)
check("rustfmt is clean", r.returncode == 0)

# ---------------------------------------------------------------- 2. unit tests
section("unit tests (no weights needed)")
r = run(["cargo", "test", "--release", "--lib"], cwd=CRATE)
line = [l for l in r.stdout.splitlines() if l.startswith("test result")]
check("unit tests pass", bool(line) and "0 failed" in line[0], line[0] if line else "no result")

# ---------------------------------------------------------------- 3. CLI
section("CLI")
if not os.path.exists(CLI):
    check("CLI binary present", False, "run cargo build --release --bin laya first")
else:
    r = run([CLI, "--version"])
    check("--version", r.returncode == 0 and "0.3.4" in r.stdout, r.stdout.strip())

    r = run([CLI, "--help"])
    check("--help lists commands", all(c in r.stdout for c in ("route", "ask", "info", "presets")))

    r = run([CLI, "route", "--state", "I was charged twice, please refund"])
    check(
        "route picks english for English text",
        r.returncode == 0 and "english" in r.stdout,
        r.stdout.strip().splitlines()[0] if r.stdout else "",
    )

    r = run([CLI, "route", "--state", "मुझे दो बार शुल्क लिया गया"])
    check(
        "route picks multilingual for Devanagari",
        "multilingual" in r.stdout,
        r.stdout.strip().splitlines()[0] if r.stdout else "",
    )

    r = run([CLI, "route", "--state", "hello", "--lang", "de"])
    check("explicit --lang overrides detection", "multilingual" in r.stdout)

    r = run([CLI, "route", "--state", "hello", "--json"])
    try:
        d = json.loads(r.stdout)
        ok = d["model"] == "english" and d["detection"]["script"] == "latin"
    except Exception as e:
        ok = False
        d = {"err": str(e)}
    check("--json emits a machine-readable decision", ok, json.dumps(d)[:80])

    r = run([CLI, "presets"])
    check(
        "presets lists all five question sets",
        all(p in r.stdout for p in ("triage", "email", "guard", "moderation", "router")),
    )

    # bad input must fail loudly, not silently
    r = run([CLI, "route", "--preset", "nope", "--state", "x"])
    check("unknown preset exits non-zero", r.returncode != 0, r.stderr.strip()[:60])
    r = run([CLI, "frobnicate"])
    check("unknown command exits non-zero", r.returncode != 0)
    r = run([CLI, "ask", "--state", "x", "--model", "/definitely/not/here"])
    check("missing checkpoint exits non-zero", r.returncode != 0, r.stderr.strip()[:60])

# ---------------------------------------------------------------- 4. python module
section("python module")
py = shutil.which("python3")
try:
    import laya_rs  # noqa: F401

    have = True
except ImportError:
    have = False
if have:
    import laya_rs

    check("module imports", True, "v" + laya_rs.__version__)
    expected = {
        "detect_language", "detect_script", "is_english", "clean_email_body", "email_state",
        "email_questions", "guard_questions", "moderation_questions", "router_questions",
        "triage_questions", "Router", "Agent", "render_options", "confidence_from_probs",
        "QTYPES", "QTYPE_NAMES", "DEFAULT_MODELS", "RLAgent",
    }
    missing = expected - set(laya_rs.__all__)
    check("exports the laya API surface", not missing, "missing: {}".format(sorted(missing)) if missing else "")

    d = laya_rs.detect_language({"message": "I was charged twice"})
    check("detect_language works", d["script"] == "latin" and d["is_english"] is True)
    check("normalise_name resolves aliases", laya_rs.normalise_name("EN") == "english")
    check(
        "render_criterion matches json.dumps",
        laya_rs.render_criterion({"desc": "phishing"}) == '{"desc": "phishing"}',
    )
    r = laya_rs.Router()
    dec = r.route({"message": "hello"}, laya_rs.triage_questions())
    check("Router.route works", dec["model"] == "english" and dec.model == "english")
else:
    check(
        "module imports",
        False,
        "not installed; run: maturin build --release --features python && pip install <wheel>",
        required=False,
    )

# ---------------------------------------------------------------- 5. inference
section("inference")
if MODEL is None:
    check("local checkpoint available", False, "set LAYA_TEST_MODEL or pass --model", required=False)
else:
    check("local checkpoint available", True, MODEL)
    if os.path.exists(CLI):
        r = run([CLI, "info", MODEL])
        ok = r.returncode == 0 and "layers" in r.stdout and "0 layers" not in r.stdout
        check("info reports real encoder geometry", ok, r.stdout.strip().splitlines()[-1] if r.stdout else "")

        r = run([CLI, "ask", "--preset", "triage", "--state",
                 "I was charged twice, please refund", "--model", MODEL])
        ok = r.returncode == 0 and "intent -> refund" in r.stdout
        check("ask runs a preset end to end", ok, r.stdout.strip().splitlines()[0] if r.stdout else "")

        r = run([CLI, "ask", "--preset", "email", "--state",
                 "URGENT: verify your password at http://secure-login-verify.xyz now",
                 "--model", MODEL])
        check("ask flags phishing", "is_phishing -> true" in r.stdout)

    # parity against the Python package, if it is importable
    try:
        sys.path.insert(0, REPO)
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_TORCH", "1")
        import laya  # noqa: F401

        have_py = True
    except Exception as e:
        have_py = False
        print("      (python laya not importable: {})".format(str(e)[:60]))
    if have_py:
        r = run([sys.executable, os.path.join(HERE, "check_inference_parity.py"), MODEL])
        tail = [l for l in r.stdout.splitlines() if "cases," in l or "problems" in l]
        check(
            "inference matches the python package",
            r.returncode == 0 and "0 total problems" in r.stdout,
            tail[-1] if tail else r.stderr.strip()[:80],
        )

# ---------------------------------------------------------------- 6. optional
if FULL:
    section("optional (network / gpu)")
    env = dict(os.environ)
    env.pop("all_proxy", None)
    env.pop("HF_ENDPOINT", None)
    r = run([CLI, "info", "convaiinnovations/laya"], env=env)
    check("downloads a checkpoint from the hub", r.returncode == 0, r.stderr.strip()[:80])

    r = run(["cargo", "build", "--release", "--features", "metal", "--bin", "laya"], cwd=CRATE)
    check("builds with the metal backend", r.returncode == 0)

# ---------------------------------------------------------------- summary
print("\n=== summary ===")
failed = [n for n, ok, _, req in results if not ok and req]
warned = [n for n, ok, _, req in results if not ok and not req]
print("{} checks, {} passed, {} failed, {} skipped".format(
    len(results), sum(1 for _, ok, _, _ in results if ok), len(failed), len(warned)))
if warned:
    print("skipped (not required): {}".format(", ".join(warned)))
if failed:
    print("FAILED: {}".format(", ".join(failed)))
sys.exit(1 if failed else 0)
