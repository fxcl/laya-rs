//! Laya: fast, non-autoregressive System 1 decision engine (Rust port).
//!
//! Phase 1 ports the pure-logic surface of the Python `laya` package:
//! language/script detection, checkpoint routing, question presets, email
//! cleaning and criteria rendering. The inference runtime (ModernBERT encoder
//! forward pass, tokenizer, safetensors, Hub download) lands in phase 2 behind
//! the same API -- see [`router::Predictor`].

pub mod agent;
pub mod criteria;
pub mod device;
pub mod email;
pub mod lang;
pub mod model;
pub mod modernbert;
pub mod presets;
pub mod pyjson;
pub mod router;
pub mod sequence;
pub mod weights;

pub use agent::{Agent, RLAgent};
pub use criteria::{
    confidence_from_probs, ece_score, render_criterion, render_options, temp_bucket, Criteria,
    QType, Question, Questions, QTYPES, QTYPE_NAMES,
};
pub use lang::{
    analyse, detect_script, guess_latin_language, is_english, script_profile, state_text, Analysis,
};
pub use router::{
    default_models, match_typed_decisions_workflow, normalise_name, standalone_models, ModelSpec,
    Repo, RouteDecision, Router, RouterOptions, BUNDLE_REPO,
};

#[cfg(feature = "python")]
pub mod python;

/// Errors mirroring the exceptions the Python package raises.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayaError {
    /// `ValueError: unknown model %r` from `normalise_name`.
    UnknownModel(String),
    /// A question definition was not a JSON object.
    QuestionNotObject,
    /// A question definition had no `type` / `t` key.
    QuestionMissingType,
    /// `type` / `t` was not one of choice/score/noul.
    QuestionUnknownType(String),
    /// A question definition had no `instructions` / `ins` key.
    QuestionMissingInstructions,
    /// `render_options` got a criteria shape the question type cannot use
    /// (Python raises AttributeError/TypeError here).
    CriteriaMismatch {
        qtype: &'static str,
        expected: &'static str,
    },
    /// `ValueError: question %r options exceed head_max_len=%d`.
    OptionsExceedHeadMaxLen { qid: String, head_max_len: usize },
    /// Phase 1 has no inference runtime: no checkpoint can be built or run.
    InferenceUnavailable,
    /// A config or weights file could not be read or parsed.
    ConfigUnreadable(String, String),
    /// `model.safetensors` could not be read or parsed.
    WeightsUnreadable(String, String),
    /// A tensor the architecture needs is missing from the checkpoint.
    WeightsIncomplete(String, String),
    /// The checkpoint carries tensors the architecture does not consume.
    WeightsUnexpected(String, Vec<String>),
    /// The requested device is not available in this build.
    DeviceUnavailable(String),
    /// A tensor operation failed.
    Tensor(String),
    /// The checkpoint does not match the configured architecture (`_verify_compatibility`).
    Incompatible(String),
    /// A file the runtime needs is absent. Maps to Python's `FileNotFoundError`.
    NotFound(String),
}

impl std::fmt::Display for LayaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayaError::UnknownModel(msg) => write!(f, "{}", msg),
            LayaError::QuestionNotObject => write!(f, "question definition must be an object"),
            LayaError::QuestionMissingType => write!(f, "question definition is missing 'type'"),
            LayaError::QuestionUnknownType(t) => {
                write!(
                    f,
                    "unknown question type {:?}; expected one of choice, score, noul",
                    t
                )
            }
            LayaError::QuestionMissingInstructions => {
                write!(f, "question definition is missing 'instructions'")
            }
            LayaError::CriteriaMismatch { qtype, expected } => {
                write!(f, "{} questions need {} criteria", qtype, expected)
            }
            LayaError::OptionsExceedHeadMaxLen { qid, head_max_len } => write!(
                f,
                "question {:?} options exceed head_max_len={}",
                qid, head_max_len
            ),
            LayaError::InferenceUnavailable => write!(
                f,
                "the Rust inference runtime is not built yet (phase 2); only routing is available"
            ),
            LayaError::ConfigUnreadable(path, e) => write!(f, "cannot read config {}: {}", path, e),
            LayaError::WeightsUnreadable(path, e) => {
                write!(f, "cannot read weights {}: {}", path, e)
            }
            LayaError::WeightsIncomplete(path, key) => {
                write!(f, "checkpoint {} is missing tensor {}", path, key)
            }
            LayaError::WeightsUnexpected(path, keys) => write!(
                f,
                "checkpoint {} has {} tensor(s) this architecture does not use: {}",
                path,
                keys.len(),
                keys.join(", ")
            ),
            LayaError::DeviceUnavailable(d) => {
                write!(
                    f,
                    "device {:?} is not available in this build (cpu only)",
                    d
                )
            }
            LayaError::Tensor(e) => write!(f, "tensor error: {}", e),
            LayaError::Incompatible(msg) => write!(f, "{}", msg),
            LayaError::NotFound(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for LayaError {}
