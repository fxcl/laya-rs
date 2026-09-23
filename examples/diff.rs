//! Differential-test harness: run a JSON list of cases through the Rust port and
//! emit one JSON result per case, so the Python reference can be compared against it.
//!
//!   cargo run --example diff -- cases.json results.json

use std::fs;

use indexmap::IndexMap;
use laya_rs::criteria::{
    confidence_from_probs, ece_score, render_criterion, render_options, temp_bucket, QType,
    Question, Questions,
};
use laya_rs::email::{clean_email_body, email_state};
use laya_rs::lang::{
    analyse, detect_script, guess_latin_language, is_english, script_profile, state_text,
};
use laya_rs::presets;
use laya_rs::router::{
    match_typed_decisions_workflow, normalise_name, ModelSpec, RouteDecision, Router, RouterOptions,
};
use laya_rs::LayaError;
use serde_json::{json, Value};

fn opt_str<'a>(case: &'a Value, key: &str) -> Option<&'a str> {
    case.get(key).and_then(|v| v.as_str())
}

fn run_case(case: &Value) -> Value {
    let op = case.get("op").and_then(|v| v.as_str()).unwrap_or("");
    let result: Result<Value, LayaError> = (|| match op {
        "detect_script" => Ok(Value::String(detect_script(
            case.get("text").and_then(|v| v.as_str()).unwrap_or(""),
        ))),
        "script_profile" => {
            let prof = script_profile(case.get("text").and_then(|v| v.as_str()).unwrap_or(""));
            let mut m = serde_json::Map::new();
            for (k, v) in prof {
                m.insert(k, json!(v));
            }
            Ok(Value::Object(m))
        }
        "guess_latin_language" => Ok(
            match guess_latin_language(case.get("text").and_then(|v| v.as_str()).unwrap_or("")) {
                Some(l) => Value::String(l),
                None => Value::Null,
            },
        ),
        "state_text" => Ok(Value::String(state_text(
            case.get("state").unwrap_or(&Value::Null),
        ))),
        "is_english" => Ok(Value::Bool(is_english(
            case.get("state").unwrap_or(&Value::Null),
        ))),
        "analyse" => Ok(analyse(case.get("state").unwrap_or(&Value::Null)).to_value()),
        "normalise_name" => Ok(Value::String(normalise_name(
            case.get("name").and_then(|v| v.as_str()).unwrap_or(""),
        )?)),
        "match_workflow" => {
            let ids: Vec<String> = case
                .get("ids")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let questions = Questions(
                ids.into_iter()
                    .map(|id| {
                        (
                            id,
                            Question {
                                t: QType::Noul,
                                ins: String::new(),
                                crit: laya_rs::Criteria::None,
                            },
                        )
                    })
                    .collect(),
            );
            Ok(match match_typed_decisions_workflow(&questions) {
                Some(w) => Value::String(w),
                None => Value::Null,
            })
        }
        "render_criterion" => Ok(Value::String(render_criterion(
            case.get("value").unwrap_or(&Value::Null),
        ))),
        "render_options" => {
            let q = Question::from_internal(case.get("question").unwrap_or(&Value::Null))?;
            Ok(json!(render_options(&q)?))
        }
        "confidence_from_probs" => {
            let p: Vec<f64> = case
                .get("p")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                .unwrap_or_default();
            let k = case.get("k").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            Ok(json!(confidence_from_probs(&p, k)))
        }
        "temp_bucket" => {
            let qt = case.get("qtype").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let k = case.get("k").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            Ok(Value::String(temp_bucket(
                QType::from_index(qt).ok_or(LayaError::QuestionUnknownType(qt.to_string()))?,
                k,
            )))
        }
        "ece_score" => {
            let conf: Vec<f64> = case
                .get("conf")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                .unwrap_or_default();
            let correct: Vec<f64> = case
                .get("correct")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                .unwrap_or_default();
            let bins = case.get("bins").and_then(|v| v.as_u64()).unwrap_or(15) as usize;
            Ok(json!(ece_score(&conf, &correct, bins)))
        }
        "clean_email_body" => {
            let body = case.get("body").and_then(|v| v.as_str());
            let max_chars = case
                .get("max_chars")
                .and_then(|v| v.as_u64())
                .unwrap_or(3000) as usize;
            Ok(Value::String(clean_email_body(body, max_chars)))
        }
        "email_state" => {
            let extra: Vec<(String, Option<String>)> = case
                .get("extra")
                .and_then(|v| v.as_object())
                .map(|m| {
                    m.iter()
                        .map(|(k, v)| (k.clone(), v.as_str().map(String::from)))
                        .collect()
                })
                .unwrap_or_default();
            let pairs = email_state(
                opt_str(case, "subject"),
                opt_str(case, "body"),
                opt_str(case, "sender"),
                case.get("clean").and_then(|v| v.as_bool()).unwrap_or(true),
                &extra,
            );
            let mut m = serde_json::Map::new();
            for (k, v) in pairs {
                m.insert(k, Value::String(v));
            }
            Ok(Value::Object(m))
        }
        "route" => {
            let state = case.get("state").unwrap_or(&Value::Null);
            let questions = Questions::from_public(case.get("questions").unwrap_or(&Value::Null))?;
            let opts_v = case.get("opts").cloned().unwrap_or(json!({}));
            let models: Option<IndexMap<String, ModelSpec>> =
                opts_v.get("models").and_then(|v| v.as_object()).map(|m| {
                    m.iter()
                        .map(|(k, v)| (k.clone(), ModelSpec::from_value(v)))
                        .collect()
                });
            let opts = RouterOptions {
                models,
                device: opts_v
                    .get("device")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                token: opts_v
                    .get("token")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                max_loaded: opts_v
                    .get("max_loaded")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize),
                default: opts_v
                    .get("default")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                auto_task_detection: opts_v.get("auto_task_detection").and_then(|v| v.as_bool()),
                standalone_repos: opts_v.get("standalone_repos").and_then(|v| v.as_bool()),
            };
            let r = Router::new(opts)?;
            let d: RouteDecision = r.route(
                state,
                &questions,
                opt_str(case, "model"),
                opt_str(case, "task"),
                opt_str(case, "lang"),
            )?;
            Ok(d.to_value())
        }
        "preset" => {
            let name = case.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let q = match name {
                "triage" => presets::triage_questions(),
                "email" => {
                    let cats: Option<Vec<(String, String)>> =
                        case.get("categories").and_then(|v| v.as_object()).map(|m| {
                            m.iter()
                                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                                .collect()
                        });
                    presets::email_questions(cats.as_deref())
                }
                "guard" => presets::guard_questions(),
                "moderation" => presets::moderation_questions(),
                "router" => presets::router_questions(),
                other => return Err(LayaError::QuestionUnknownType(other.to_string())),
            };
            Ok(q.to_public())
        }
        other => Err(LayaError::QuestionUnknownType(other.to_string())),
    })();
    match result {
        Ok(v) => json!({"ok": v}),
        Err(e) => json!({"err": e.to_string()}),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cases: Value = serde_json::from_str(&fs::read_to_string(&args[1]).unwrap()).unwrap();
    let out: Vec<Value> = cases.as_array().unwrap().iter().map(run_case).collect();
    fs::write(&args[2], serde_json::to_string_pretty(&out).unwrap()).unwrap();
}
