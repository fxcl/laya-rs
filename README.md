# laya-rs

Fast, calibrated, non-autoregressive System 1 decision engine written in pure Rust.

A full Rust port of the [`laya`](https://github.com/NandhaKishorM/laya) Python package, providing 100% numerical and logical parity with the official ModernBERT-based decision models while offering single-binary CLI deployment, zero-copy inference, and seamless PyO3 Python bindings.

---

## Features

- **100% Parity with Python Package**:
  - Pure-logic: 1,078 test cases verified with 0 mismatches.
  - ModernBERT encoder & custom decision heads: matched to $\Delta = 0.000000$ in 4-decimal precision across 13 diverse operational scenarios.
- **Fast System 1 Decision Making**:
  - Non-autoregressive: produces decisions and calibrated probabilities across multiple questions in a single forward pass.
  - Zero generation latency: ~350ms per question on standard CPU, ~14ms for text routing.
- **Production-Ready CLI (`laya`)**:
  - `laya route`: Instantly route texts to checkpoints without loading model weights.
  - `laya ask`: Run built-in or custom question presets through checkpoints.
  - `laya info`: Inspect tensor geometry and head configurations.
  - `laya presets`: List built-in decision workflows (triage, email, guard, moderation, router).
- **Flexible Integration**:
  - Native Rust crate API.
  - PyO3 Python extension with automatic GIL detachment (`py.detach()`) during inference.
  - Optional hardware backends (`metal` on macOS Apple Silicon, `cuda` on Linux/Windows).

---

## Quick Start

### 1. Build the CLI

```bash
cargo build --release --bin laya
# Executable is placed at ./target/release/laya
```

### 2. CLI Usage Examples

```bash
# Model routing (instant, no model weights required)
./target/release/laya route --state "I was charged twice for invoice 4411"

# Run customer ticket triage
./target/release/laya ask \
  --preset triage \
  --state "I was charged twice and nobody replied for 3 days. Refund today or we cancel." \
  --model ~/laya_models/laya

# Inspect checkpoint structure
./target/release/laya info ~/laya_models/laya
```

---

## Rust Library API

Add `laya-rs` to your `Cargo.toml`:

```toml
[dependencies]
laya-rs = { git = "https://github.com/fxcl/laya-rs.git" }
serde_json = "1.0"
```

In your code:

```rust
use laya_rs::agent::Agent;
use laya_rs::presets;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load checkpoint (auto-detects hardware backend with graceful CPU fallback)
    let agent = Agent::load("/path/to/checkpoint", None, None, None)?;

    // Define state and load preset questions
    let state = json!({
        "message": "I was charged twice for invoice 4411. Please refund immediately.",
        "account_tier": "enterprise"
    });
    let questions = presets::triage_questions();

    // Perform System 1 inference
    let result = agent.system_one(&state, &questions)?;
    println!("{}", serde_json::to_string_pretty(&result)?);

    Ok(())
}
```

---

## Python Integration (PyO3)

Build and install the Python wheel via `maturin`:

```bash
maturin develop --release --features python
```

Use in Python:

```python
import laya_rs

agent = laya_rs.load("~/laya_models/laya", device="cpu")
result = agent.system_one(
    {"message": "I was charged twice for invoice 4411"},
    laya_rs.triage_questions(),
)
print("Intent:", result["answers"]["intent"]["choice"])
```

---

## Verification & Documentation

Detailed evaluation reports, numerical parity benchmarks, and complex test cases are available in [`docs/rust_port_evaluation.md`](docs/rust_port_evaluation.md).

```bash
# Run unit tests (71 tests, no weights needed)
cargo test --lib

# Differential logic testing against Python reference (1,078 cases)
python3 scripts/diff_against_python.py

# End-to-end inference verification (requires checkpoint)
cargo test --release --test inference
```

---

## License

Licensed under the Apache License, Version 2.0.
