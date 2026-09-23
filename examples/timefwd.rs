//! Time the encoder and the head separately.
use laya_rs::agent::Agent;
use laya_rs::criteria::Questions;
use laya_rs::model::DecisionModel;
use std::time::Instant;

fn main() {
    let dir = std::env::var("HOME").unwrap() + "/laya_models/laya";
    let t = Instant::now();
    let agent = Agent::load(&dir, Some("cpu"), None, None).unwrap();
    println!("load: {:?}", t.elapsed());

    let state = serde_json::json!({"message": "I was charged twice for invoice 4411 and nobody has answered for three days. Refund the duplicate today or we are cancelling."});
    let questions = Questions::from_public(&serde_json::json!({
        "intent": {"type":"choice","instructions":"What does the customer want in `message`?",
                   "criteria":{"refund":"money returned","technical_help":"a bug","billing_question":"a question","information":"general info","cancellation":"wants to cancel","other":"none"}},
        "is_urgent": {"type":"noul","instructions":"Does `message` communicate time pressure?"},
        "frustration": {"type":"score","instructions":"How frustrated?","criteria":["calm","concerned","annoyed","angry"]},
        "refund_requested": {"type":"noul","instructions":"Does the customer ask for money back?"},
        "churn_risk": {"type":"noul","instructions":"May the customer leave?"}
    })).unwrap();

    // build the batch once, then time the forward repeatedly
    let mut seqs = Vec::new();
    let mut qtypes = Vec::new();
    for (_, q) in &questions.0 {
        let s = laya_rs::sequence::build_sequence(
            agent.tokenizer(),
            &state,
            q,
            agent.max_len(),
            agent.head_max_len(),
        )
        .unwrap();
        seqs.push(s);
        qtypes.push(q.t.index());
    }
    let batch = laya_rs::sequence::collate(
        &seqs,
        &qtypes,
        agent.tokenizer().pad_id,
        agent.candle_device(),
    )
    .unwrap();
    println!(
        "batch: {:?}, seq len {}",
        batch.input_ids.dims(),
        seqs[0].ids.len()
    );

    let model: &DecisionModel = agent.model();
    for i in 0..2 {
        let t = Instant::now();
        let _ = model
            .encoder_forward(&batch.input_ids, &batch.attention_mask)
            .unwrap();
        println!("run {}: encoder_forward {:?}", i, t.elapsed());
    }
    for i in 0..2 {
        let t = Instant::now();
        let _ = model
            .forward(
                &batch.input_ids,
                &batch.attention_mask,
                &batch.marker_pos,
                &batch.marker_mask,
                &batch.qtype,
            )
            .unwrap();
        println!("run {}: full forward    {:?}", i, t.elapsed());
    }
    // the exact same single-question workload the Python-side diagnostic uses
    let q1 = Questions::from_public(&serde_json::json!(
        {"q": {"type": "noul", "instructions": "Does the customer ask for money back?"}}
    ))
    .unwrap();
    let st1 =
        serde_json::json!({"message": "I was charged twice for invoice 4411, please refund."});
    let s1 = laya_rs::sequence::build_sequence(
        agent.tokenizer(),
        &st1,
        &q1.0[0].1,
        agent.max_len(),
        agent.head_max_len(),
    )
    .unwrap();
    let b1 = laya_rs::sequence::collate(
        &[s1],
        &[q1.0[0].1.t.index()],
        agent.tokenizer().pad_id,
        agent.candle_device(),
    )
    .unwrap();
    println!("single-question batch: {:?}", b1.input_ids.dims());
    for i in 0..2 {
        let t = Instant::now();
        let _ = model
            .forward(
                &b1.input_ids,
                &b1.attention_mask,
                &b1.marker_pos,
                &b1.marker_mask,
                &b1.qtype,
            )
            .unwrap();
        println!("run {}: single forward {:?}", i, t.elapsed());
    }
    for i in 0..2 {
        let t = Instant::now();
        let _ = agent.system_one(&state, &questions).unwrap();
        println!("run {}: system_one     {:?}", i, t.elapsed());
    }
}
