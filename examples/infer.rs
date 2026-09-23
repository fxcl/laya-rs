//! Numerical parity harness for the inference runtime.
//!
//! Reads a JSON list of `{state, questions}` cases, runs `Agent::system_one` on each and
//! writes the payloads back out, so a Python driver can compare them against the real
//! `laya` package.
//!
//!     cargo run --release --example infer -- cases.json out.json [model_dir]

use std::env;

use laya_rs::agent::Agent;
use laya_rs::criteria::Questions;

fn main() {
    let args: Vec<String> = env::args().collect();
    let cases_path = &args[1];
    let out_path = &args[2];
    let model_dir = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| format!("{}/laya_models/laya", env::var("HOME").unwrap()));
    // LAYA_DEVICE selects the backend; unset means auto-detect (cuda > metal > cpu).
    let device = env::var("LAYA_DEVICE").ok();

    let raw = std::fs::read_to_string(cases_path).expect("read cases");
    let cases: Vec<serde_json::Value> = serde_json::from_str(&raw).expect("parse cases");

    let agent = Agent::load(&model_dir, device.as_deref(), None, None).expect("load checkpoint");
    eprintln!("loaded {} on {}", model_dir, agent.device());

    let mut out = Vec::new();
    for case in &cases {
        let state = case
            .get("state")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let questions =
            Questions::from_public(case.get("questions").unwrap()).expect("parse questions");
        let result = agent.system_one(&state, &questions).expect("system_one");
        out.push(result);
    }
    std::fs::write(out_path, serde_json::to_string_pretty(&out).unwrap()).expect("write out");
    eprintln!("wrote {} results to {}", out.len(), out_path);
}
