//! Question types, criteria rendering and calibration helpers.
//!
//! Rust port of the pure-Python parts of `laya/common.py` (`render_criterion`,
//! `render_options`, `confidence_from_probs`, `temp_bucket`, `ece_score`) plus the
//! public/internal question-definition shapes from `laya/agent.py`.

use serde_json::{Map, Value};

use crate::pyjson::dumps;
use crate::LayaError;

/// Question type ids, matching Python's `QTYPES = {"choice": 0, "score": 1, "noul": 2}`.
pub const QTYPE_NAMES: [&str; 3] = ["choice", "score", "noul"];

/// The same mapping as ordered `(name, index)` pairs, for callers that want to iterate it
/// the way Python's `QTYPES.items()` does.
pub const QTYPES: [(&str, usize); 3] = [("choice", 0), ("score", 1), ("noul", 2)];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QType {
    Choice,
    Score,
    Noul,
}

impl QType {
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }

    /// Index into the temperature vector, as in Python's `QTYPES`.
    pub fn index(self) -> usize {
        match self {
            QType::Choice => 0,
            QType::Score => 1,
            QType::Noul => 2,
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "choice" => Some(QType::Choice),
            "score" => Some(QType::Score),
            "noul" => Some(QType::Noul),
            _ => None,
        }
    }

    /// Inverse of [`QType::index`].
    pub fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(QType::Choice),
            1 => Some(QType::Score),
            2 => Some(QType::Noul),
            _ => None,
        }
    }
}

/// Criteria payload, mirroring the three shapes the Python internal form uses.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Criteria {
    /// No criteria at all (`None`).
    #[default]
    None,
    /// Ordered key/value pairs (`choice` options, `noul` false/true descriptions).
    Map(Vec<(String, Value)>),
    /// Ordered levels (`score` rubrics).
    List(Vec<Value>),
}

/// A question in the internal form: `{"t": ..., "ins": ..., "crit": ...}`.
#[derive(Clone, Debug, PartialEq)]
pub struct Question {
    pub t: QType,
    pub ins: String,
    pub crit: Criteria,
}

/// An ordered set of questions, keyed by question id.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Questions(pub Vec<(String, Question)>);

impl Questions {
    /// Question ids in insertion order.
    pub fn ids(&self) -> Vec<&str> {
        self.0.iter().map(|(id, _)| id.as_str()).collect()
    }

    pub fn get(&self, id: &str) -> Option<&Question> {
        self.0.iter().find(|(k, _)| k == id).map(|(_, q)| q)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Parse the public form: `{qid: {"type": ..., "instructions": ..., "criteria": ...}}`.
    ///
    /// Mirrors `Agent._to_internal`, including the `choice` list-to-dict promotion.
    pub fn from_public(value: &Value) -> Result<Self, LayaError> {
        let obj = value.as_object().ok_or(LayaError::QuestionNotObject)?;
        let mut out = Vec::with_capacity(obj.len());
        for (qid, def) in obj {
            out.push((qid.clone(), Question::from_public(def)?));
        }
        Ok(Questions(out))
    }

    /// Build from a public questions object keeping only the ids.
    ///
    /// Routing reads question ids and nothing else, so the definitions are not validated
    /// here -- `Router.route` must accept the same inputs the Python one does, including
    /// `{qid: {}}`.
    pub fn ids_only(value: &Value) -> Result<Self, LayaError> {
        let obj = value.as_object().ok_or(LayaError::QuestionNotObject)?;
        Ok(Questions(
            obj.keys()
                .map(|qid| {
                    (
                        qid.clone(),
                        Question {
                            t: QType::Noul,
                            ins: String::new(),
                            crit: Criteria::None,
                        },
                    )
                })
                .collect(),
        ))
    }

    /// Rebuild the public form, e.g. to hand back to Python.
    pub fn to_public(&self) -> Value {
        let mut obj = Map::new();
        for (qid, q) in &self.0 {
            obj.insert(qid.clone(), q.to_public());
        }
        Value::Object(obj)
    }
}

impl Question {
    /// Parse one public question definition (`Agent._to_internal`).
    pub fn from_public(def: &Value) -> Result<Self, LayaError> {
        let obj = def.as_object().ok_or(LayaError::QuestionNotObject)?;
        let t = obj
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or(LayaError::QuestionMissingType)?;
        let t = QType::from_name(t).ok_or_else(|| LayaError::QuestionUnknownType(t.to_string()))?;
        let mut crit = obj.get("criteria").cloned().unwrap_or(Value::Null);
        if t == QType::Choice {
            if let Value::Array(items) = &crit {
                let mut m = Map::new();
                for c in items {
                    m.insert(py_str(c), Value::Null);
                }
                crit = Value::Object(m);
            }
        }
        let ins = match obj.get("instructions") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => dumps(other),
            None => return Err(LayaError::QuestionMissingInstructions),
        };
        let crit = criteria_from_value(crit)?;
        Ok(Question { t, ins, crit })
    }

    /// Parse one internal question definition (`{"t", "ins", "crit"}`).
    pub fn from_internal(def: &Value) -> Result<Self, LayaError> {
        let obj = def.as_object().ok_or(LayaError::QuestionNotObject)?;
        let t = obj
            .get("t")
            .and_then(|v| v.as_str())
            .ok_or(LayaError::QuestionMissingType)?;
        let t = QType::from_name(t).ok_or_else(|| LayaError::QuestionUnknownType(t.to_string()))?;
        let ins = obj
            .get("ins")
            .and_then(|v| v.as_str())
            .ok_or(LayaError::QuestionMissingInstructions)?
            .to_string();
        let crit = criteria_from_value(obj.get("crit").cloned().unwrap_or(Value::Null))?;
        Ok(Question { t, ins, crit })
    }

    /// Rebuild the public form of this question.
    ///
    /// `criteria` is omitted entirely when there is none, matching the preset dicts the
    /// Python package ships (a `noul` question simply has no `criteria` key).
    pub fn to_public(&self) -> Value {
        let mut obj = Map::new();
        obj.insert("type".to_string(), Value::String(self.t.name().to_string()));
        obj.insert("instructions".to_string(), Value::String(self.ins.clone()));
        match &self.crit {
            Criteria::None => {}
            Criteria::Map(pairs) => {
                let mut m = Map::new();
                for (k, v) in pairs {
                    m.insert(k.clone(), v.clone());
                }
                obj.insert("criteria".to_string(), Value::Object(m));
            }
            Criteria::List(items) => {
                obj.insert("criteria".to_string(), Value::Array(items.clone()));
            }
        }
        Value::Object(obj)
    }
}

/// Python's `str()` of a scalar, used when a `choice` criteria list holds non-strings.
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Number(n) => n.to_string(),
        other => dumps(other),
    }
}

/// Shape a raw criteria value into [`Criteria`], mirroring what each question type accepts.
fn criteria_from_value(crit: Value) -> Result<Criteria, LayaError> {
    Ok(match crit {
        Value::Null => Criteria::None,
        Value::Object(m) => Criteria::Map(m.into_iter().collect()),
        Value::Array(a) => Criteria::List(a),
        // Python's `enumerate(crit)` walks a string character by character.
        Value::String(s) => {
            Criteria::List(s.chars().map(|c| Value::String(c.to_string())).collect())
        }
        // Python raises TypeError on `enumerate(3)`.
        _ => {
            return Err(LayaError::CriteriaMismatch {
                qtype: "score",
                expected: "a list of levels",
            })
        }
    })
}

/// Render one criterion value as text.
///
/// Strings pass through; anything structured (dict, list, number) becomes compact JSON, so a
/// rubric reads as JSON rather than a Python repr.
pub fn render_criterion(value: &Value) -> String {
    if let Value::String(s) = value {
        return s.clone();
    }
    dumps(value)
}

/// Render option texts in label-index order. Noul is always [false, true].
pub fn render_options(q: &Question) -> Result<Vec<String>, LayaError> {
    match (&q.t, &q.crit) {
        (QType::Choice, Criteria::Map(pairs)) => Ok(pairs
            .iter()
            .map(|(k, v)| {
                // only None/"" mean "no description"; 0 and False are legitimate criterion values
                if v.is_null() || v.as_str() == Some("") {
                    k.clone()
                } else {
                    format!("{}: {}", k, render_criterion(v))
                }
            })
            .collect()),
        (QType::Score, Criteria::List(items)) => Ok(items
            .iter()
            .enumerate()
            .map(|(i, c)| format!("level {}: {}", i, render_criterion(c)))
            .collect()),
        (QType::Noul, Criteria::Map(pairs)) => {
            let get = |key: &str| pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone());
            let false_crit = get("false");
            let true_crit = get("true");
            let render = |v: Option<Value>, default: &str| -> String {
                match v {
                    Some(v) if !v.is_null() && v.as_str() != Some("") => render_criterion(&v),
                    _ => default.to_string(),
                }
            };
            Ok(vec![
                format!(
                    "false: {}",
                    render(false_crit, "no, the statement does not hold")
                ),
                format!("true: {}", render(true_crit, "yes, the statement holds")),
            ])
        }
        (QType::Noul, Criteria::None) => Ok(vec![
            "false: no, the statement does not hold".to_string(),
            "true: yes, the statement holds".to_string(),
        ]),
        (t, _) => Err(LayaError::CriteriaMismatch {
            qtype: t.name(),
            expected: match t {
                QType::Choice => "a map of option -> description",
                QType::Score => "a list of levels",
                QType::Noul => "a map or nothing",
            },
        }),
    }
}

/// Normalized Shannon entropy confidence: 1 - H(p) / log(k).
pub fn confidence_from_probs(p: &[f64], k: usize) -> f64 {
    if k < 2 {
        return 1.0;
    }
    let p = &p[..k.min(p.len())];
    let ent: f64 = p
        .iter()
        .map(|&x| {
            let c = x.clamp(1e-12, 1.0);
            -c * c.ln()
        })
        .sum();
    (1.0 - ent / (k as f64).ln()).clamp(0.0, 1.0)
}

/// Temperature bucket key: `"{qtype}:{option-count bucket}"`.
pub fn temp_bucket(qtype: QType, k: usize) -> String {
    let size = if k <= 2 {
        "2"
    } else if k <= 5 {
        "3-5"
    } else if k <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{}", qtype.name(), size)
}

/// Expected Calibration Error across confidence bins.
pub fn ece_score(conf: &[f64], correct: &[f64], bins: usize) -> f64 {
    if conf.is_empty() {
        return f64::NAN;
    }
    let n = conf.len() as f64;
    // numpy's linspace(0, 1, bins+1) computes i * (1/bins), not i/bins.
    let step = 1.0 / bins as f64;
    let mut e = 0.0;
    for i in 0..bins {
        let lo = i as f64 * step;
        let hi = if i + 1 == bins {
            1.0
        } else {
            (i + 1) as f64 * step
        };
        let sel: Vec<usize> = (0..conf.len())
            .filter(|&j| conf[j] > lo && conf[j] <= hi)
            .collect();
        if sel.is_empty() {
            continue;
        }
        let frac = sel.len() as f64 / n;
        let cm: f64 = sel.iter().map(|&j| conf[j]).sum::<f64>() / sel.len() as f64;
        let am: f64 = sel.iter().map(|&j| correct[j]).sum::<f64>() / sel.len() as f64;
        e += frac * (cm - am).abs();
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn q(t: &str, ins: &str, crit: Value) -> Question {
        Question::from_internal(&json!({"t": t, "ins": ins, "crit": crit})).unwrap()
    }

    // --------------------------------------------------------------- render_criterion
    #[test]
    fn render_criterion_table() {
        assert_eq!(
            render_criterion(&json!("phishing or scam")),
            "phishing or scam"
        );
        assert_eq!(
            render_criterion(&json!({"desc": "phishing"})),
            "{\"desc\": \"phishing\"}"
        );
        assert_eq!(render_criterion(&json!(["a", "b"])), "[\"a\", \"b\"]");
        assert_eq!(render_criterion(&json!(3)), "3");
        assert_eq!(render_criterion(&json!(false)), "false");
        assert_eq!(
            render_criterion(&json!({"d": "münchen"})),
            "{\"d\": \"münchen\"}"
        );
    }

    // --------------------------------------------------------------- the reported crash
    #[test]
    fn noul_dict_criteria_does_not_crash() {
        let out = render_options(&q(
            "noul",
            "Is this phishing?",
            json!({"true": {"desc": "phishing, scam or fraud"}, "false": {"desc": "legitimate"}}),
        ))
        .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], "false: {\"desc\": \"legitimate\"}");
        assert_eq!(out[1], "true: {\"desc\": \"phishing, scam or fraud\"}");
        assert!(!out.join("").contains('\''));
    }

    // --------------------------------------------------------------- choice and score
    #[test]
    fn choice_and_score_rendering() {
        let out = render_options(&q(
            "choice",
            "x",
            json!({"billing": {"desc": "payments"}, "tech": null, "sales": ""}),
        ))
        .unwrap();
        assert_eq!(out[0], "billing: {\"desc\": \"payments\"}");
        assert_eq!(out[1], "tech");
        assert_eq!(out[2], "sales");
        assert!(!out.join("").contains("{'"));

        // 0 and False are real criterion values, not "missing"
        let out = render_options(&q("choice", "x", json!({"zero": 0, "no": false}))).unwrap();
        assert_eq!(out[0], "zero: 0");
        assert_eq!(out[1], "no: false");

        let out = render_options(&q("score", "x", json!([{"d": "low"}, "high", 2]))).unwrap();
        assert_eq!(out[0], "level 0: {\"d\": \"low\"}");
        assert_eq!(out[1], "level 1: high");
        assert_eq!(out[2], "level 2: 2");
    }

    // --------------------------------------------------------------- unchanged behaviour
    #[test]
    fn default_noul_text() {
        let out = render_options(&q("noul", "x", Value::Null)).unwrap();
        assert_eq!(out[0], "false: no, the statement does not hold");
        assert_eq!(out[1], "true: yes, the statement holds");
    }

    #[test]
    fn string_criteria_still_work() {
        assert_eq!(
            render_options(&q("noul", "x", json!({"true": "yes it is", "false": "no"}))).unwrap(),
            vec!["false: no".to_string(), "true: yes it is".to_string()]
        );
        assert_eq!(
            render_options(&q("choice", "x", json!({"a": "first", "b": null}))).unwrap(),
            vec!["a: first".to_string(), "b".to_string()]
        );
        assert_eq!(
            render_options(&q("score", "x", json!(["low", "high"]))).unwrap(),
            vec!["level 0: low".to_string(), "level 1: high".to_string()]
        );
    }

    #[test]
    fn every_rendered_option_is_a_string() {
        for def in [
            json!({"t": "choice", "ins": "x", "crit": {"a": {"n": 1}, "b": [1, 2], "c": 3.5}}),
            json!({"t": "score", "ins": "x", "crit": [{"a": 1}, [2], null]}),
            json!({"t": "noul", "ins": "x", "crit": {"true": [1], "false": {"z": 0}}}),
        ] {
            let question = Question::from_internal(&def).unwrap();
            let out = render_options(&question).unwrap();
            assert!(out.iter().all(|o| !o.is_empty()), "{:?}", def);
        }
    }

    #[test]
    fn emitted_json_round_trips() {
        let out = render_options(&q(
            "noul",
            "x",
            json!({"true": {"a": 1}, "false": {"b": 2}}),
        ))
        .unwrap();
        let payload = out[1].strip_prefix("true: ").unwrap();
        let parsed: Value = serde_json::from_str(payload).unwrap();
        assert_eq!(parsed, json!({"a": 1}));
    }

    // --------------------------------------------------------------- public form
    #[test]
    fn public_form_promotes_choice_lists() {
        let questions = Questions::from_public(&json!({
            "dept": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "tech"]}
        }))
        .unwrap();
        let q = questions.get("dept").unwrap();
        assert_eq!(q.t, QType::Choice);
        assert_eq!(
            render_options(q).unwrap(),
            vec!["billing".to_string(), "tech".to_string()]
        );
    }

    #[test]
    fn public_form_json_dumps_non_string_instructions() {
        let questions = Questions::from_public(&json!({
            "q": {"type": "noul", "instructions": {"a": 1}}
        }))
        .unwrap();
        assert_eq!(questions.get("q").unwrap().ins, "{\"a\": 1}");
    }

    #[test]
    fn ids_only_keeps_ids_without_validating_definitions() {
        // routing reads ids only, so `{qid: {}}` must be accepted exactly as Python accepts it
        let q = Questions::ids_only(&json!({"action": {}, "category": {}, "churn_risk": {},
                                            "needs_human": {}, "urgency": {}}))
        .unwrap();
        assert_eq!(
            q.ids(),
            vec!["action", "category", "churn_risk", "needs_human", "urgency"]
        );
        assert_eq!(
            crate::router::match_typed_decisions_workflow(&q).as_deref(),
            Some("customer_service")
        );
        assert!(Questions::ids_only(&json!("not an object")).is_err());
    }

    #[test]
    fn public_form_rejects_bad_definitions() {
        assert_eq!(
            Questions::from_public(&json!({"q": {"instructions": "x"}})).unwrap_err(),
            LayaError::QuestionMissingType
        );
        assert_eq!(
            Questions::from_public(&json!({"q": {"type": "bogus", "instructions": "x"}}))
                .unwrap_err(),
            LayaError::QuestionUnknownType("bogus".to_string())
        );
        assert_eq!(
            Questions::from_public(&json!({"q": {"type": "noul"}})).unwrap_err(),
            LayaError::QuestionMissingInstructions
        );
    }

    // --------------------------------------------------------------- confidence
    #[test]
    fn confidence_from_probs_values() {
        assert_eq!(confidence_from_probs(&[0.5, 0.5], 2), 0.0);
        // a zero probability is clipped to 1e-12, so it still contributes ~2.8e-11 of entropy
        assert!((confidence_from_probs(&[1.0, 0.0], 2) - 1.0).abs() < 1e-9);
        assert_eq!(confidence_from_probs(&[0.7, 0.3], 2), 0.1187091007693073);
        assert_eq!(confidence_from_probs(&[1.0], 1), 1.0);
    }

    #[test]
    fn temp_bucket_keys() {
        assert_eq!(temp_bucket(QType::Choice, 2), "choice:2");
        assert_eq!(temp_bucket(QType::Score, 4), "score:3-5");
        assert_eq!(temp_bucket(QType::Noul, 8), "noul:6-10");
        assert_eq!(temp_bucket(QType::Choice, 20), "choice:11+");
    }

    #[test]
    fn ece_of_perfect_and_worst() {
        // 0.8 sits exactly on the 12/15 bin edge and the bin test is `conf > lo`,
        // so the two points land in different bins: 0.5*|0.9-1.0| + 0.5*|0.8-0.0|
        assert_eq!(ece_score(&[0.9, 0.8], &[1.0, 0.0], 15), 0.45);
        // both points inside one bin
        assert_eq!(ece_score(&[0.9, 0.85], &[1.0, 1.0], 15), 0.125);
        // perfectly calibrated
        assert_eq!(ece_score(&[0.9, 0.85], &[0.9, 0.85], 15), 0.0);
        assert!(ece_score(&[], &[], 15).is_nan());
    }
}
