//! Sequence-building parity: dump the token ids and marker positions the Rust
//! `build_sequence` produces, so a Python driver can compare against `laya.common`.
//!     cargo run --example dumpseq -- cases.json out.json [model_dir]

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

    let raw = std::fs::read_to_string(cases_path).expect("read cases");
    let cases: Vec<serde_json::Value> = serde_json::from_str(&raw).expect("parse cases");

    let agent = Agent::load(&model_dir, Some("cpu"), None, None).expect("load checkpoint");
    let mut out = Vec::new();
    for case in &cases {
        let state = case
            .get("state")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let questions = Questions::from_public(case.get("questions").unwrap()).unwrap();
        let mut per_question = Vec::new();
        for (qid, q) in &questions.0 {
            let seq = laya_rs::sequence::build_sequence(
                agent.tokenizer(),
                &state,
                q,
                agent.max_len(),
                agent.head_max_len(),
            )
            .expect("build_sequence");
            per_question.push(serde_json::json!({
                "qid": qid,
                "ids": seq.ids,
                "markers": seq.markers,
            }));
        }
        out.push(per_question);
    }
    std::fs::write(out_path, serde_json::to_string_pretty(&out).unwrap()).expect("write");
    eprintln!("wrote {} cases", out.len());
}
