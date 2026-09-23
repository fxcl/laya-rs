//! Model-forward parity check: run one fixed batch through the Rust model and dump the
//! raw logits, so a Python driver can compare them against the same batch in torch.
//!     cargo run --release --example dumplogits -- batch.json out.json [model_dir]

use std::env;

use candle_core::Tensor;
use laya_rs::agent::Agent;
use laya_rs::model::DecisionModel;

fn main() {
    let args: Vec<String> = env::args().collect();
    let batch_path = &args[1];
    let out_path = &args[2];
    let model_dir = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| format!("{}/laya_models/laya", env::var("HOME").unwrap()));

    let raw = std::fs::read_to_string(batch_path).expect("read batch");
    let batch: serde_json::Value = serde_json::from_str(&raw).expect("parse batch");

    let agent = Agent::load(&model_dir, Some("cpu"), None, None).expect("load checkpoint");
    let model: &DecisionModel = agent.model();

    let ids: Vec<i64> = batch["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_i64().unwrap()))
        .collect();
    let att: Vec<f32> = batch["attention_mask"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
        })
        .collect();
    let mpos: Vec<i64> = batch["marker_pos"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_i64().unwrap()))
        .collect();
    // the Python side emits a bool tensor, so accept both booleans and numbers
    let mmask: Vec<f32> = batch["marker_mask"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| {
            r.as_array().unwrap().iter().map(|v| {
                v.as_f64()
                    .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                    .unwrap() as f32
            })
        })
        .collect();
    let qtype: Vec<i64> = batch["qtype"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();

    let n = batch["input_ids"].as_array().unwrap().len();
    let l = batch["input_ids"].as_array().unwrap()[0]
        .as_array()
        .unwrap()
        .len();
    let kmax = batch["marker_pos"].as_array().unwrap()[0]
        .as_array()
        .unwrap()
        .len();

    let device = candle_core::Device::Cpu;
    let input_ids = Tensor::from_slice(&ids, (n, l), &device).unwrap();
    let attention_mask = Tensor::from_slice(&att, (n, l), &device).unwrap();
    let marker_pos = Tensor::from_slice(&mpos, (n, kmax), &device).unwrap();
    let marker_mask = Tensor::from_slice(&mmask, (n, kmax), &device).unwrap();
    let qtype = Tensor::from_slice(&qtype, (n,), &device).unwrap();

    let out = model
        .forward(
            &input_ids,
            &attention_mask,
            &marker_pos,
            &marker_mask,
            &qtype,
        )
        .expect("forward");

    let logits = out.logits.to_vec2::<f32>().unwrap();
    let act = out.act_logits.to_vec2::<f32>().unwrap();
    let hidden = model
        .encoder_forward(&input_ids, &attention_mask)
        .expect("encoder");
    let hidden = hidden.to_vec3::<f32>().unwrap();
    std::fs::write(
        out_path,
        serde_json::to_string_pretty(&serde_json::json!({
            "logits": logits,
            "act_logits": act,
            "hidden": hidden,
        }))
        .unwrap(),
    )
    .expect("write");
    eprintln!(
        "wrote logits {}x{}, hidden {}x{}x{}",
        logits.len(),
        logits[0].len(),
        hidden.len(),
        hidden[0].len(),
        hidden[0][0].len()
    );
}
