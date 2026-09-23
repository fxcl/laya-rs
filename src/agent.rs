//! The inference runtime: load a checkpoint and answer questions in one forward pass.
//!
//! Rust port of `Agent` in `laya/agent.py`. The public surface (`system_one` / `predict`)
//! and the returned payload shape match the Python one exactly.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use candle_core::DType;
use serde_json::{Map, Value};

use crate::criteria::{
    confidence_from_probs, render_options, temp_bucket, QType, Questions, QTYPE_NAMES,
};
use crate::model::DecisionModel;
use crate::modernbert::candle_err;
use crate::pyjson::dumps;
use crate::router::{Loader, ModelSpec, Predictor};
use crate::sequence::{build_sequence, collate, Tokenizer};
use crate::weights::Weights;
use crate::LayaError;

/// Alias matching the Python package's `RLAgent = Agent`.
pub type RLAgent = Agent;

/// A loaded Laya checkpoint.
pub struct Agent {
    model: DecisionModel,
    tok: Tokenizer,
    cfg: Value,
    encoder_cfg: Value,
    max_len: usize,
    head_max_len: usize,
    temperature: Vec<f64>,
    temperature_by_options: HashMap<String, f64>,
    device: candle_core::Device,
    device_name: String,
}

impl Agent {
    /// Load a checkpoint from a local directory or a Hub repo.
    ///
    /// `subfolder` selects one checkpoint from a repo that bundles several, e.g.
    /// `Agent::load("convaiinnovations/laya", Some("multilingual"))`.
    pub fn load(
        model_id_or_path: &str,
        device: Option<&str>,
        token: Option<&str>,
        subfolder: Option<&str>,
    ) -> Result<Self, LayaError> {
        // cuda > metal > cpu when unset; an explicit but unavailable GPU falls back to CPU
        // with a warning, exactly like the Python runtime.
        let request = match device {
            Some(d) => crate::device::DeviceRequest::Explicit(d.to_string()),
            None => crate::device::DeviceRequest::Auto,
        };
        let resolved = crate::device::resolve(&request)?;
        if let Some(from) = &resolved.fell_back_from {
            let why = resolved.probe_error.clone().unwrap_or_else(|| {
                "the requested device is not available in this build".to_string()
            });
            eprint!("{}", crate::device::fallback_warning(from, &why));
        }
        let device = resolved.device;
        let device_name = match device.location() {
            candle_core::DeviceLocation::Cpu => "cpu".to_string(),
            candle_core::DeviceLocation::Cuda { gpu_id } => format!("cuda:{}", gpu_id),
            candle_core::DeviceLocation::Metal { gpu_id } => format!("metal:{}", gpu_id),
        };

        let model_dir = resolve_model_dir(model_id_or_path, subfolder, token)?;

        let cfg_path = format!("{}/rl_agent_config.json", model_dir);
        if !Path::new(&cfg_path).exists() {
            return Err(LayaError::NotFound(format!(
                "Incompatible model: {} does not contain 'rl_agent_config.json'. Make sure \
                 you are loading a compatible RL Agent model (e.g. 'convaiinnovations/rl-agent').",
                crate::router::py_repr(model_id_or_path)
            )));
        }
        let cfg = crate::weights::read_json(&cfg_path)?;

        let weights_path = format!("{}/model.safetensors", model_dir);
        if !Path::new(&weights_path).exists() {
            return Err(LayaError::NotFound(format!(
                "Incompatible model: 'model.safetensors' not found in {}.",
                crate::router::py_repr(model_id_or_path)
            )));
        }

        let tok_dir = format!("{}/tokenizer", model_dir);
        let tok = if Path::new(&format!("{}/tokenizer.json", tok_dir)).exists() {
            Tokenizer::load(&tok_dir)?
        } else {
            // fall back to the encoder repo named in the config
            let encoder = cfg
                .get("encoder")
                .and_then(|v| v.as_str())
                .unwrap_or("answerdotai/ModernBERT-large")
                .to_string();
            let dir = resolve_model_dir(&encoder, None, token)?;
            Tokenizer::load(&format!("{}/tokenizer", dir))?
        };

        let encoder_cfg = crate::weights::read_json(&format!("{}/encoder/config.json", model_dir))?;

        // Fail fast on a mismatched checkpoint, with the same wording as the Python runtime.
        crate::weights::verify_config_keys(&cfg, &["encoder", "head_layers"], model_id_or_path)?;
        let plan = DecisionModel::weight_plan(&encoder_cfg, &cfg)?;
        let mut weights = Weights::load(&weights_path, &device)?;
        weights.verify_compatibility(&plan, model_id_or_path)?;
        let model = DecisionModel::load(&encoder_cfg, &cfg, &mut weights)?;
        // strict=True: every tensor in the checkpoint must have been consumed
        weights.finish()?;

        let max_len = cfg.get("max_len").and_then(|v| v.as_u64()).unwrap_or(512) as usize;
        let head_max_len = cfg
            .get("head_max_len")
            .and_then(|v| v.as_u64())
            .unwrap_or(192) as usize;
        let temperature: Vec<f64> = cfg
            .get("temperature")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
            .unwrap_or_else(|| vec![1.0, 1.0, 1.0]);
        let mut temperature_by_options = HashMap::new();
        if let Some(m) = cfg
            .get("temperature_by_options")
            .and_then(|v| v.as_object())
        {
            for (k, v) in m {
                if let Some(f) = v.as_f64() {
                    temperature_by_options.insert(k.clone(), f);
                }
            }
        }

        Ok(Agent {
            model,
            tok,
            cfg,
            encoder_cfg,
            max_len,
            head_max_len,
            temperature,
            temperature_by_options,
            device,
            device_name,
        })
    }

    pub fn device(&self) -> &str {
        &self.device_name
    }

    /// The candle device the model and batches live on.
    pub fn candle_device(&self) -> &candle_core::Device {
        &self.device
    }

    /// The underlying model, for diagnostics and parity harnesses.
    pub fn model(&self) -> &DecisionModel {
        &self.model
    }

    /// The tokenizer, for diagnostics and parity harnesses.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    pub fn head_max_len(&self) -> usize {
        self.head_max_len
    }

    /// The raw config, as loaded from `rl_agent_config.json`.
    /// The ModernBERT encoder config (`encoder/config.json`), which is a different file
    /// from [`Agent::config`] -- the latter's `encoder` key is only a repo name.
    pub fn encoder_config(&self) -> &Value {
        &self.encoder_cfg
    }

    pub fn config(&self) -> &Value {
        &self.cfg
    }

    /// Evaluate typed questions across state in a single, parallel forward pass.
    pub fn system_one(&self, state: &Value, questions: &Questions) -> Result<Value, LayaError> {
        let ids: Vec<String> = questions.0.iter().map(|(id, _)| id.clone()).collect();
        let mut seqs = Vec::with_capacity(ids.len());
        let mut qtypes = Vec::with_capacity(ids.len());
        let mut option_counts = Vec::with_capacity(ids.len());

        for (qid, q) in &questions.0 {
            let seq = build_sequence(&self.tok, state, q, self.max_len, self.head_max_len)?;
            let n_options = render_options(q)?.len();
            if seq.markers.len() != n_options {
                return Err(LayaError::OptionsExceedHeadMaxLen {
                    qid: qid.clone(),
                    head_max_len: self.head_max_len,
                });
            }
            seqs.push(seq);
            qtypes.push(q.t.index());
            option_counts.push(n_options);
        }

        let batch = collate(&seqs, &qtypes, self.tok.pad_id, &self.device)?;
        let out = self.model.forward(
            &batch.input_ids,
            &batch.attention_mask,
            &batch.marker_pos,
            &batch.marker_mask,
            &batch.qtype,
        )?;

        let logits = out.logits.to_dtype(DType::F32).map_err(candle_err)?;
        let logits = logits.to_vec2::<f32>().map_err(candle_err)?;
        let act = candle_nn::ops::softmax(
            &out.act_logits.to_dtype(DType::F32).map_err(candle_err)?,
            candle_core::D::Minus1,
        )
        .map_err(candle_err)?
        .to_vec2::<f32>()
        .map_err(candle_err)?;

        let n_tokens: f32 = batch
            .attention_mask
            .sum_all()
            .map_err(candle_err)?
            .to_scalar()
            .map_err(candle_err)?;

        let mut answers = Map::new();
        for (r, (qid, q)) in questions.0.iter().enumerate() {
            let k = option_counts[r];
            let bucket = temp_bucket(q.t, k);
            let t_scale = self
                .temperature_by_options
                .get(&bucket)
                .copied()
                .unwrap_or(self.temperature[q.t.index()]);
            let scale = t_scale.max(1e-3);

            let row = &logits[r];
            let z: Vec<f64> = row[..k].iter().map(|&v| v as f64 / scale).collect();
            let zmax = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = z.iter().map(|&v| (v - zmax).exp()).collect();
            let sum: f64 = exp.iter().sum();
            let p: Vec<f64> = exp.iter().map(|&v| v / sum).collect();

            let conf = round4(confidence_from_probs(&p, k));
            let act_probability = round4(act[r][0] as f64);
            let action = {
                let mut m = Map::new();
                m.insert("act_probability".to_string(), json_num(act_probability));
                Value::Object(m)
            };

            let entry = match q.t {
                QType::Choice => {
                    let keys: Vec<String> = match &q.crit {
                        crate::criteria::Criteria::Map(pairs) => {
                            pairs.iter().map(|(k, _)| k.clone()).collect()
                        }
                        _ => Vec::new(),
                    };
                    let mut probs = Map::new();
                    for (i, key) in keys.iter().enumerate() {
                        probs.insert(key.clone(), json_num(round4(p[i])));
                    }
                    let best = p
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    let mut m = Map::new();
                    m.insert("type".to_string(), Value::String("choice".to_string()));
                    m.insert(
                        "choice".to_string(),
                        Value::String(keys[best.min(keys.len().saturating_sub(1))].clone()),
                    );
                    m.insert("probabilities".to_string(), Value::Object(probs));
                    m.insert("confidence".to_string(), json_num(conf));
                    m.insert("action".to_string(), action);
                    Value::Object(m)
                }
                QType::Score => {
                    let levels: Vec<String> = match &q.crit {
                        crate::criteria::Criteria::List(items) => items
                            .iter()
                            .map(|c| match c {
                                Value::String(s) => s.clone(),
                                other => dumps(other),
                            })
                            .collect(),
                        _ => Vec::new(),
                    };
                    let exp_score: f64 = p.iter().enumerate().map(|(i, &v)| i as f64 * v).sum();
                    let mut probs = Map::new();
                    for (i, v) in p.iter().enumerate() {
                        probs.insert(i.to_string(), json_num(round4(*v)));
                    }
                    let mut legend = Map::new();
                    for (i, c) in levels.iter().enumerate() {
                        legend.insert(i.to_string(), Value::String(c.clone()));
                    }
                    let mut m = Map::new();
                    m.insert("type".to_string(), Value::String("score".to_string()));
                    m.insert("score".to_string(), json_num(round4(exp_score)));
                    m.insert("legend".to_string(), Value::Object(legend));
                    m.insert("probabilities".to_string(), Value::Object(probs));
                    m.insert("confidence".to_string(), json_num(conf));
                    m.insert("action".to_string(), action);
                    Value::Object(m)
                }
                QType::Noul => {
                    let noul = p.get(1).copied().unwrap_or(0.0);
                    let mut m = Map::new();
                    m.insert("type".to_string(), Value::String("noul".to_string()));
                    m.insert("noul".to_string(), json_num(round4(noul)));
                    m.insert(
                        "confidence".to_string(),
                        json_num(round4(noul.max(1.0 - noul))),
                    );
                    m.insert("action".to_string(), action);
                    Value::Object(m)
                }
            };
            answers.insert(qid.clone(), entry);
        }

        let mut usage = Map::new();
        // Python emits plain ints here (`int(...)` and a literal 0), so keep them integral
        // rather than letting them serialise as 0.0.
        usage.insert("input_tokens".to_string(), json_int(n_tokens as i64));
        usage.insert("output_tokens".to_string(), json_int(0));

        let mut result = Map::new();
        result.insert(
            "model".to_string(),
            Value::String("laya-rl-agent".to_string()),
        );
        result.insert("answers".to_string(), Value::Object(answers));
        result.insert("usage".to_string(), Value::Object(usage));
        Ok(Value::Object(result))
    }
}

impl Predictor for Agent {
    fn system_one(&self, state: &Value, questions: &Questions) -> Result<Value, LayaError> {
        Agent::system_one(self, state, questions)
    }
}

/// The loader the Router installs by default: build an [`Agent`] for a model spec.
pub fn default_loader() -> Loader {
    default_loader_with(None, None)
}

/// Like [`default_loader`], but honouring a device and Hub token.
pub fn default_loader_with(device: Option<String>, token: Option<String>) -> Loader {
    Arc::new(move |spec: &ModelSpec| {
        let agent = Agent::load(
            &spec.repo,
            device.as_deref(),
            token.as_deref(),
            spec.subfolder.as_deref(),
        )?;
        Ok(Arc::new(agent) as Arc<dyn Predictor>)
    })
}

/// Resolve a repo id or local path to a directory holding the checkpoint files.
fn resolve_model_dir(
    model_id_or_path: &str,
    subfolder: Option<&str>,
    token: Option<&str>,
) -> Result<String, LayaError> {
    let base = if Path::new(model_id_or_path).exists() {
        model_id_or_path.to_string()
    } else if model_id_or_path.starts_with('/')
        || model_id_or_path.starts_with("./")
        || model_id_or_path.starts_with("../")
        || Path::new(model_id_or_path).is_absolute()
    {
        return Err(LayaError::NotFound(format!(
            "Local model path not found: {}. Check that the directory exists and that \
             training saved the model successfully.",
            crate::router::py_repr(model_id_or_path)
        )));
    } else {
        download_checkpoint(model_id_or_path, subfolder, token)?
    };

    match subfolder {
        Some(sub) => {
            let dir = format!("{}/{}", base, sub);
            if !Path::new(&dir).is_dir() {
                return Err(LayaError::NotFound(format!(
                    "Subfolder {} not found in {}.",
                    crate::router::py_repr(sub),
                    crate::router::py_repr(model_id_or_path)
                )));
            }
            Ok(dir)
        }
        None => Ok(base),
    }
}

/// Download just the files a checkpoint needs, mirroring `allow_patterns=[subfolder/*]`.
fn download_checkpoint(
    repo_id: &str,
    subfolder: Option<&str>,
    token: Option<&str>,
) -> Result<String, LayaError> {
    use hf_hub::api::sync::{Api, ApiBuilder};

    let prefix = subfolder.map(|s| format!("{}/", s)).unwrap_or_default();
    let files = [
        format!("{}model.safetensors", prefix),
        format!("{}rl_agent_config.json", prefix),
        format!("{}encoder/config.json", prefix),
        format!("{}tokenizer/tokenizer.json", prefix),
        format!("{}tokenizer/tokenizer_config.json", prefix),
    ];

    let api = match token {
        Some(t) => ApiBuilder::new()
            .with_token(Some(t.to_string()))
            .build()
            .map_err(|e| LayaError::ConfigUnreadable(repo_id.to_string(), e.to_string()))?,
        None => Api::new()
            .map_err(|e| LayaError::ConfigUnreadable(repo_id.to_string(), e.to_string()))?,
    };
    let repo = api.model(repo_id.to_string());

    let mut first: Option<String> = None;
    for file in &files {
        let path = repo
            .get(file)
            .map_err(|e| LayaError::ConfigUnreadable(repo_id.to_string(), e.to_string()))?;
        let parent = path
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        if first.is_none() {
            first = Some(parent);
        }
    }
    first.ok_or_else(|| {
        LayaError::ConfigUnreadable(repo_id.to_string(), "nothing downloaded".to_string())
    })
}

/// Round to 4 decimals the way Python's `round(x, 4)` does.
fn round4(x: f64) -> f64 {
    format!("{:.4}", x).parse().unwrap_or(x)
}

/// Build a JSON number, keeping integers integral.
fn json_num(v: f64) -> Value {
    serde_json::Number::from_f64(v)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// Build a JSON integer.
fn json_int(v: i64) -> Value {
    Value::Number(v.into())
}

/// Question-type names, re-exported for callers building payloads.
pub const QUESTION_TYPES: [&str; 3] = QTYPE_NAMES;
