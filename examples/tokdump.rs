//! Tokenizer parity check: dump the ids the Rust tokenizer produces for a set of texts.
//!     cargo run --example tokdump -- texts.json out.json [tokenizer_dir]

use std::env;

use laya_rs::sequence::Tokenizer;

fn main() {
    let args: Vec<String> = env::args().collect();
    let texts_path = &args[1];
    let out_path = &args[2];
    let dir = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| format!("{}/laya_models/laya/tokenizer", env::var("HOME").unwrap()));

    let raw = std::fs::read_to_string(texts_path).expect("read texts");
    let texts: Vec<String> = serde_json::from_str(&raw).expect("parse texts");

    let tok = Tokenizer::load(&dir).expect("load tokenizer");
    eprintln!(
        "cls={} mask={} sep={} pad={}",
        tok.cls_id, tok.mask_id, tok.sep_id, tok.pad_id
    );

    let mut out = Vec::new();
    for text in &texts {
        out.push(serde_json::json!({
            "text": text,
            "ids": tok.encode(text).unwrap(),
        }));
    }
    std::fs::write(out_path, serde_json::to_string_pretty(&out).unwrap()).expect("write");
    eprintln!("wrote {} encodings", out.len());
}
