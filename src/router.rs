//! Route a request to the Laya checkpoint best suited to it.
//!
//! Rust port of `laya/router.py`. Three checkpoints live behind one bundle repo:
//!
//!   english          convaiinnovations/laya                421M  ModernBERT-large, 512 tokens
//!   multilingual     convaiinnovations/laya-multilingual   322M  mmBERT-base, 1024 tokens, 100+ langs
//!   typed-decisions  convaiinnovations/laya-typed-decisions 421M  ModernBERT-large, 1024 tokens
//!
//! `typed-decisions` is never selected automatically unless you opt in with
//! `auto_task_detection` or pass `task="typed_decisions"`.

use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::criteria::Questions;
use crate::lang::{analyse, Analysis};
use crate::LayaError;

/// The hub repo bundles all three checkpoints; only the requested subfolder is downloaded.
pub const BUNDLE_REPO: &str = "convaiinnovations/laya";

/// A checkpoint location: a repo (or local path) plus an optional subfolder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelSpec {
    pub repo: String,
    pub subfolder: Option<String>,
}

impl ModelSpec {
    /// Human-readable id: 'repo' or 'repo/subfolder'.
    pub fn repo_str(&self) -> String {
        match &self.subfolder {
            Some(sub) => format!("{}/{}", self.repo, sub),
            None => self.repo.clone(),
        }
    }

    /// Normalise a model spec the way Python's `_split` does: only tuples/lists carry a subfolder.
    pub fn from_value(v: &Value) -> Self {
        match v {
            Value::Array(items) => {
                let repo = items
                    .first()
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let sub = items.get(1).and_then(|x| x.as_str()).map(|s| s.to_string());
                ModelSpec {
                    repo,
                    subfolder: sub,
                }
            }
            Value::String(s) => ModelSpec {
                repo: s.clone(),
                subfolder: None,
            },
            other => ModelSpec {
                repo: other.to_string(),
                subfolder: None,
            },
        }
    }
}

/// The default routing table: all three checkpoints inside one bundle repo.
pub fn default_models() -> IndexMap<String, ModelSpec> {
    let mut m = IndexMap::new();
    m.insert(
        "english".to_string(),
        ModelSpec {
            repo: BUNDLE_REPO.to_string(),
            subfolder: None,
        },
    );
    m.insert(
        "multilingual".to_string(),
        ModelSpec {
            repo: BUNDLE_REPO.to_string(),
            subfolder: Some("multilingual".to_string()),
        },
    );
    m.insert(
        "typed-decisions".to_string(),
        ModelSpec {
            repo: BUNDLE_REPO.to_string(),
            subfolder: Some("typed-decisions".to_string()),
        },
    );
    m
}

/// The same checkpoints in their own repos, for anyone who prefers them.
pub fn standalone_models() -> IndexMap<String, ModelSpec> {
    let mut m = IndexMap::new();
    for (k, repo) in [
        ("english", "convaiinnovations/laya"),
        ("multilingual", "convaiinnovations/laya-multilingual"),
        ("typed-decisions", "convaiinnovations/laya-typed-decisions"),
    ] {
        m.insert(
            k.to_string(),
            ModelSpec {
                repo: repo.to_string(),
                subfolder: None,
            },
        );
    }
    m
}

/// Aliases people are likely to type.
const ALIASES: &[(&str, &str)] = &[
    ("en", "english"),
    ("laya", "english"),
    ("default", "english"),
    ("multi", "multilingual"),
    ("ml", "multilingual"),
    ("laya-multilingual", "multilingual"),
    ("typed", "typed-decisions"),
    ("typed_decisions", "typed-decisions"),
    ("laya-typed-decisions", "typed-decisions"),
    ("decisions", "typed-decisions"),
];

/// Question-id signatures of the four typed-decisions workflows, used only when
/// auto_task_detection is enabled.
const TYPED_DECISION_WORKFLOWS: &[(&str, &[&str])] = &[
    (
        "agent_trace_observability",
        &["action", "needs_review", "outcome", "risk", "urgency"],
    ),
    (
        "customer_service",
        &["action", "category", "churn_risk", "needs_human", "urgency"],
    ),
    (
        "invoice_processing",
        &[
            "discrepancy_severity",
            "disposition",
            "duplicate",
            "matches_order",
            "urgency",
        ],
    ),
    (
        "security_incidents",
        &[
            "credential_compromise",
            "disposition",
            "severity",
            "true_positive",
            "urgency",
        ],
    ),
];

/// Python's `repr()` of a string, for the `%r`-formatted reason strings.
pub fn py_repr(s: &str) -> String {
    let has_single = s.contains('\'');
    let has_double = s.contains('"');
    let quote = if has_single && !has_double { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python's `repr()` of a list of strings.
fn py_list_repr(items: &[String]) -> String {
    format!(
        "[{}]",
        items
            .iter()
            .map(|s| py_repr(s))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Resolve a model name or alias to a canonical key.
pub fn normalise_name(name: &str) -> Result<String, LayaError> {
    let key = name.trim().to_lowercase();
    let key = ALIASES
        .iter()
        .find(|(a, _)| *a == key)
        .map(|(_, canonical)| canonical.to_string())
        .unwrap_or(key);
    if !default_models().contains_key(&key) {
        let mut models: Vec<String> = default_models().keys().cloned().collect();
        models.sort();
        let mut aliases: Vec<String> = ALIASES.iter().map(|(a, _)| a.to_string()).collect();
        aliases.sort();
        return Err(LayaError::UnknownModel(format!(
            "unknown model {}; choose one of {} (or an alias: {})",
            py_repr(name),
            py_list_repr(&models),
            py_list_repr(&aliases)
        )));
    }
    Ok(key)
}

/// Name of the typed-decisions workflow whose question ids these are, else None.
///
/// Requires an exact id-set match, so an unrelated schema that happens to contain 'urgency'
/// is never captured.
pub fn match_typed_decisions_workflow(questions: &Questions) -> Option<String> {
    let ids = questions.ids();
    for (wf, sig) in TYPED_DECISION_WORKFLOWS {
        if ids.len() == sig.len() && sig.iter().all(|s| ids.contains(s)) {
            return Some(wf.to_string());
        }
    }
    None
}

/// The `repo` field of a routing decision.
///
/// Every branch emits the human-readable `"repo"` / `"repo/subfolder"` string -- except the
/// auto-detected typed-decisions workflow branch, which emits the raw `(repo, subfolder)`
/// pair. That is an inconsistency in `laya/router.py` (every sibling branch calls
/// `_repo_str`, this one does not); it is reproduced here deliberately so the two
/// implementations stay comparable, and [`Repo::id`] gives the string form either way.
#[derive(Clone, Debug, PartialEq)]
pub enum Repo {
    /// `"repo"` or `"repo/subfolder"`.
    Id(String),
    /// The raw `(repo, subfolder)` pair.
    Spec(ModelSpec),
}

impl Repo {
    /// The human-readable id, whichever form this is.
    pub fn id(&self) -> String {
        match self {
            Repo::Id(s) => s.clone(),
            Repo::Spec(spec) => spec.repo_str(),
        }
    }

    /// The value Python's `RouteDecision` dict carries.
    pub fn to_value(&self) -> Value {
        match self {
            Repo::Id(s) => Value::String(s.clone()),
            Repo::Spec(spec) => Value::Array(vec![
                Value::String(spec.repo.clone()),
                match &spec.subfolder {
                    Some(s) => Value::String(s.clone()),
                    None => Value::Null,
                },
            ]),
        }
    }
}

/// The routing outcome: which model, why, and what was detected.
#[derive(Clone, Debug, PartialEq)]
pub struct RouteDecision {
    pub model: String,
    pub repo: Repo,
    pub reason: String,
    pub detection: Option<Analysis>,
    pub workflow: Option<String>,
}

impl RouteDecision {
    /// The dict form Python's `RouteDecision` (a dict subclass) serialises to.
    pub fn to_value(&self) -> Value {
        let mut m = Map::new();
        m.insert("model".to_string(), Value::String(self.model.clone()));
        m.insert("repo".to_string(), self.repo.to_value());
        m.insert("reason".to_string(), Value::String(self.reason.clone()));
        m.insert(
            "detection".to_string(),
            match &self.detection {
                Some(a) => a.to_value(),
                None => Value::Null,
            },
        );
        m.insert(
            "workflow".to_string(),
            match &self.workflow {
                Some(w) => Value::String(w.clone()),
                None => Value::Null,
            },
        );
        Value::Object(m)
    }
}

/// A runnable checkpoint. Phase 2 implements this over the ModernBERT encoder.
pub trait Predictor: Send + Sync {
    fn system_one(&self, state: &Value, questions: &Questions) -> Result<Value, LayaError>;
}

/// Builds a [`Predictor`] for a model spec (Hub download + safetensors + forward pass).
pub type Loader = std::sync::Arc<
    dyn Fn(&ModelSpec) -> Result<std::sync::Arc<dyn Predictor>, LayaError> + Send + Sync,
>;

/// Construction options for [`Router`], mirroring the Python keyword arguments.
#[derive(Clone, Debug, Default)]
pub struct RouterOptions {
    /// Override or extend the routing table (`{"english": "/path/or/repo"}`).
    pub models: Option<IndexMap<String, ModelSpec>>,
    pub device: Option<String>,
    pub token: Option<String>,
    pub max_loaded: Option<usize>,
    pub default: Option<String>,
    pub auto_task_detection: Option<bool>,
    /// Use the standalone per-checkpoint repos instead of the bundle.
    pub standalone_repos: Option<bool>,
}

/// Lazily loads Laya checkpoints and sends each request to the right one.
///
/// `max_loaded` caps how many stay resident (least-recently-used is evicted), because all
/// three together are ~1.16B parameters. For a server or a demo, preload instead: a cold
/// load costs seconds, while detection costs microseconds.
pub struct Router {
    models: IndexMap<String, ModelSpec>,
    device: Option<String>,
    token: Option<String>,
    max_loaded: usize,
    default: String,
    auto_task_detection: bool,
    agents: IndexMap<String, std::sync::Arc<dyn Predictor>>,
    order: Vec<String>,
    loader: Option<Loader>,
}

impl Router {
    pub fn new(opts: RouterOptions) -> Result<Self, LayaError> {
        let mut models = if opts.standalone_repos.unwrap_or(false) {
            standalone_models()
        } else {
            default_models()
        };
        if let Some(overrides) = opts.models {
            for (k, v) in overrides {
                models.insert(normalise_name(&k)?, v);
            }
        }
        let default = match &opts.default {
            Some(d) => normalise_name(d)?,
            None => "english".to_string(),
        };
        Ok(Router {
            models,
            device: opts.device.clone(),
            token: opts.token.clone(),
            max_loaded: opts.max_loaded.unwrap_or(1).max(1),
            default,
            auto_task_detection: opts.auto_task_detection.unwrap_or(false),
            agents: IndexMap::new(),
            order: Vec::new(),
            // the real checkpoint builder, honouring the requested device and token
            loader: Some(crate::agent::default_loader_with(
                opts.device.clone(),
                opts.token.clone(),
            )),
        })
    }

    /// Install a different checkpoint builder (tests, or a custom runtime).
    pub fn with_loader(mut self, loader: Loader) -> Self {
        self.loader = Some(loader);
        self
    }

    /// Drop the checkpoint builder, so `load` reports `InferenceUnavailable`.
    pub fn without_loader(mut self) -> Self {
        self.loader = None;
        self
    }

    pub fn device(&self) -> Option<&str> {
        self.device.as_deref()
    }

    pub fn token(&self) -> Option<&str> {
        self.token.as_deref()
    }

    pub fn max_loaded(&self) -> usize {
        self.max_loaded
    }

    pub fn set_max_loaded(&mut self, n: usize) {
        self.max_loaded = n.max(1);
    }

    pub fn default(&self) -> &str {
        &self.default
    }

    pub fn auto_task_detection(&self) -> bool {
        self.auto_task_detection
    }

    /// The routing table, in canonical order.
    pub fn models(&self) -> &IndexMap<String, ModelSpec> {
        &self.models
    }

    // ------------------------------------------------------------------ loading
    /// Return the predictor for `name`, downloading and building it on first use.
    pub fn load(&mut self, name: &str) -> Result<std::sync::Arc<dyn Predictor>, LayaError> {
        let key = normalise_name(name)?;
        if let Some(agent) = self.agents.get(&key) {
            let agent = agent.clone();
            self.touch(&key);
            return Ok(agent);
        }
        let spec =
            self.models.get(&key).cloned().ok_or_else(|| {
                LayaError::UnknownModel(format!("unknown model {}", py_repr(name)))
            })?;
        let loader = self.loader.clone().ok_or(LayaError::InferenceUnavailable)?;
        let agent = (loader)(&spec)?;
        self.agents.insert(key.clone(), agent.clone());
        self.order.push(key);
        self.evict();
        Ok(agent)
    }

    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push(key.to_string());
    }

    fn evict(&mut self) {
        while self.order.len() > self.max_loaded {
            let victim = self.order.remove(0);
            self.agents.shift_remove(&victim);
        }
        // keep the two views consistent
        let stale: Vec<String> = self
            .agents
            .keys()
            .filter(|k| !self.order.contains(k))
            .cloned()
            .collect();
        for k in stale {
            self.agents.shift_remove(&k);
        }
    }

    /// Register an already-built predictor under `name` instead of loading a second copy.
    pub fn attach(
        &mut self,
        name: &str,
        agent: std::sync::Arc<dyn Predictor>,
    ) -> Result<std::sync::Arc<dyn Predictor>, LayaError> {
        let key = normalise_name(name)?;
        self.agents.insert(key.clone(), agent.clone());
        self.touch(&key);
        self.max_loaded = self.max_loaded.max(self.agents.len());
        Ok(agent)
    }

    /// Download and build checkpoints up front so no request ever pays a model load.
    pub fn preload(&mut self, names: Option<&[String]>) -> Result<(), LayaError> {
        let names: Vec<String> = match names {
            Some(n) => n
                .iter()
                .map(|s| normalise_name(s))
                .collect::<Result<_, _>>()?,
            None => self.models.keys().cloned().collect(),
        };
        self.max_loaded = self.max_loaded.max(names.len()).max(self.agents.len());
        for n in names {
            if !self.agents.contains_key(&n) {
                self.load(&n)?;
            }
        }
        Ok(())
    }

    /// Free one model, or all of them.
    pub fn unload(&mut self, name: Option<&str>) -> Result<(), LayaError> {
        match name {
            None => {
                self.agents.clear();
                self.order.clear();
            }
            Some(n) => {
                let key = normalise_name(n)?;
                self.agents.shift_remove(&key);
                if let Some(pos) = self.order.iter().position(|k| k == &key) {
                    self.order.remove(pos);
                }
            }
        }
        Ok(())
    }

    /// Resident checkpoints, least-recently-used first.
    pub fn loaded(&self) -> Vec<String> {
        self.order.clone()
    }

    // ------------------------------------------------------------------ routing
    /// Decide which checkpoint to use, without loading or running anything.
    ///
    /// Precedence: explicit `model` > explicit `task` > detected workflow (opt-in) >
    /// explicit `lang` > detected script/language > default.
    pub fn route(
        &self,
        state: &Value,
        questions: &Questions,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
    ) -> Result<RouteDecision, LayaError> {
        if let Some(model) = model {
            let key = normalise_name(model)?;
            return Ok(RouteDecision {
                model: key.clone(),
                repo: Repo::Id(self.repo_str(&key)),
                reason: format!("explicit model={}", py_repr(model)),
                detection: None,
                workflow: None,
            });
        }

        if let Some(task) = task {
            let normalised = task.to_lowercase().replace('-', "_");
            let name = if normalised == "typed_decisions" {
                "typed-decisions"
            } else {
                task
            };
            let key = normalise_name(name)?;
            return Ok(RouteDecision {
                model: key.clone(),
                repo: Repo::Id(self.repo_str(&key)),
                reason: format!("explicit task={}", py_repr(task)),
                detection: None,
                workflow: None,
            });
        }

        let workflow = match_typed_decisions_workflow(questions);
        if let Some(wf) = &workflow {
            if self.auto_task_detection {
                return Ok(RouteDecision {
                    model: "typed-decisions".to_string(),
                    // Python emits the raw (repo, subfolder) pair here, not _repo_str().
                    repo: Repo::Spec(self.models["typed-decisions"].clone()),
                    reason: format!(
                        "question ids match the {} typed-decisions workflow",
                        py_repr(wf)
                    ),
                    detection: None,
                    workflow: Some(wf.clone()),
                });
            }
        }

        if let Some(lang) = lang {
            let lowered = lang.to_lowercase();
            let first = lowered.split('-').next().unwrap_or("");
            let key = if matches!(first, "en" | "eng" | "english") {
                "english"
            } else {
                "multilingual"
            };
            return Ok(RouteDecision {
                model: key.to_string(),
                repo: Repo::Id(self.repo_str(key)),
                reason: format!("explicit lang={}", py_repr(lang)),
                detection: None,
                workflow,
            });
        }

        let det = analyse(state);
        let (key, reason) = if det.script == "unknown" {
            (
                self.default.clone(),
                format!(
                    "no letters detected in state; using default ({})",
                    self.default
                ),
            )
        } else if det.script != "latin" {
            (
                "multilingual".to_string(),
                format!(
                    "non-Latin script ({}, {:.0}% of letters); the English checkpoint cannot read it",
                    det.script,
                    100.0 * det.non_latin_fraction
                ),
            )
        } else if !det.is_english {
            (
                "multilingual".to_string(),
                format!(
                    "Latin script but language looks like {}, not English",
                    match det.language.as_deref() {
                        Some(l) => py_repr(l),
                        None => "None".to_string(),
                    }
                ),
            )
        } else {
            ("english".to_string(), "English Latin text".to_string())
        };
        Ok(RouteDecision {
            model: key.clone(),
            repo: Repo::Id(self.repo_str(&key)),
            reason,
            detection: Some(det),
            workflow,
        })
    }

    // ------------------------------------------------------------------ running
    /// Route, then answer every question in one forward pass on the chosen checkpoint.
    ///
    /// The result is the usual `system_one` payload plus a `routing` key recording the decision.
    pub fn predict(
        &mut self,
        state: &Value,
        questions: &Questions,
        model: Option<&str>,
        task: Option<&str>,
        lang: Option<&str>,
    ) -> Result<Value, LayaError> {
        let decision = self.route(state, questions, model, task, lang)?;
        let agent = self.load(&decision.model)?;
        let mut result = agent.system_one(state, questions)?;
        if let Value::Object(ref mut m) = result {
            m.insert("routing".to_string(), decision.to_value());
        }
        Ok(result)
    }

    fn repo_str(&self, key: &str) -> String {
        self.models
            .get(key)
            .map(|s| s.repo_str())
            .unwrap_or_else(|| key.to_string())
    }
}

impl std::fmt::Debug for Router {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Router(loaded={:?}, max_loaded={}, default={:?})",
            self.loaded(),
            self.max_loaded,
            self.default
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::criteria::Question;
    use serde_json::json;
    use std::sync::Arc;

    struct Stub(String);

    impl Predictor for Stub {
        fn system_one(&self, _state: &Value, _questions: &Questions) -> Result<Value, LayaError> {
            Ok(json!({"model": self.0, "answers": {}, "usage": {}}))
        }
    }

    fn generic_questions() -> Questions {
        Questions(vec![(
            "dept".to_string(),
            Question::from_public(&json!({
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": null, "tech": null}
            }))
            .unwrap(),
        )])
    }

    fn td_questions() -> Questions {
        let ids = ["action", "category", "churn_risk", "needs_human", "urgency"];
        Questions(
            ids.iter()
                .map(|id| {
                    (
                        id.to_string(),
                        Question::from_public(&json!({"type": "noul", "instructions": "x"}))
                            .unwrap(),
                    )
                })
                .collect(),
        )
    }

    fn router() -> Router {
        Router::new(RouterOptions::default()).unwrap()
    }

    // ------------------------------------------------------------------ name normalisation
    #[test]
    fn alias_table() {
        for (alias, want) in [
            ("en", "english"),
            ("laya", "english"),
            ("multi", "multilingual"),
            ("ML", "multilingual"),
            ("typed", "typed-decisions"),
            ("typed_decisions", "typed-decisions"),
            ("English", "english"),
            ("laya", "english"),
        ] {
            assert_eq!(normalise_name(alias).unwrap(), want, "alias/{}", alias);
        }
        assert!(normalise_name("nope").is_err());
    }

    // ------------------------------------------------------------------ workflow signatures
    #[test]
    fn workflow_signatures() {
        let td = [
            (
                "agent_trace_observability",
                ["action", "needs_review", "outcome", "risk", "urgency"],
            ),
            (
                "customer_service",
                ["action", "category", "churn_risk", "needs_human", "urgency"],
            ),
            (
                "invoice_processing",
                [
                    "discrepancy_severity",
                    "disposition",
                    "duplicate",
                    "matches_order",
                    "urgency",
                ],
            ),
            (
                "security_incidents",
                [
                    "credential_compromise",
                    "disposition",
                    "severity",
                    "true_positive",
                    "urgency",
                ],
            ),
        ];
        for (wf, ids) in td {
            let q = Questions(
                ids.iter()
                    .map(|id| {
                        (
                            id.to_string(),
                            Question::from_public(&json!({"type": "noul", "instructions": "x"}))
                                .unwrap(),
                        )
                    })
                    .collect(),
            );
            assert_eq!(
                match_typed_decisions_workflow(&q).as_deref(),
                Some(wf),
                "workflow/{}",
                wf
            );
        }
        // partial overlap, superset and empty never match
        let partial = Questions(vec![(
            "urgency".to_string(),
            Question::from_public(&json!({"type": "noul", "instructions": "x"})).unwrap(),
        )]);
        assert_eq!(match_typed_decisions_workflow(&partial), None);
        let mut superset = td_questions();
        superset.0.push((
            "extra".to_string(),
            Question::from_public(&json!({"type": "noul", "instructions": "x"})).unwrap(),
        ));
        assert_eq!(match_typed_decisions_workflow(&superset), None);
        assert_eq!(match_typed_decisions_workflow(&Questions::default()), None);
    }

    // ------------------------------------------------------------------ routing decisions
    struct RouteCase {
        label: &'static str,
        state: Value,
        questions: Questions,
        kwargs: Vec<(&'static str, &'static str)>,
        want: &'static str,
    }

    #[test]
    fn routing_table() {
        let r = router();
        let q = generic_questions();
        let td = td_questions();
        let cases = [
            RouteCase {
                label: "english text",
                state: json!({"body": "I was charged twice, please refund."}),
                questions: q.clone(),
                kwargs: vec![],
                want: "english",
            },
            RouteCase {
                label: "armenian text",
                state: json!({"body": "Հայերեն"}),
                questions: q.clone(),
                kwargs: vec![],
                want: "multilingual",
            },
            RouteCase {
                label: "hindi text",
                state: json!({"body": "मुझसे दो बार शुल्क लिया गया"}),
                questions: q.clone(),
                kwargs: vec![],
                want: "multilingual",
            },
            RouteCase {
                label: "japanese text",
                state: json!({"body": "二重に請求されました"}),
                questions: q.clone(),
                kwargs: vec![],
                want: "multilingual",
            },
            RouteCase {
                label: "korean text",
                state: json!({"body": "두 번 청구되었습니다"}),
                questions: q.clone(),
                kwargs: vec![],
                want: "multilingual",
            },
            RouteCase {
                label: "arabic text",
                state: json!({"body": "تم خصم المبلغ مرتين"}),
                questions: q.clone(),
                kwargs: vec![],
                want: "multilingual",
            },
            RouteCase {
                label: "german text",
                state: json!({"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung die nicht korrekt ist"}),
                questions: q.clone(),
                kwargs: vec![],
                want: "multilingual",
            },
            RouteCase {
                label: "explicit model",
                state: json!({"body": "anything"}),
                questions: q.clone(),
                kwargs: vec![("model", "multilingual")],
                want: "multilingual",
            },
            RouteCase {
                label: "explicit model overrides script",
                state: json!({"body": "मुझसे दो बार"}),
                questions: q.clone(),
                kwargs: vec![("model", "english")],
                want: "english",
            },
            RouteCase {
                label: "explicit task",
                state: json!({"body": "x"}),
                questions: q.clone(),
                kwargs: vec![("task", "typed_decisions")],
                want: "typed-decisions",
            },
            RouteCase {
                label: "explicit lang en",
                state: json!({"body": "मुझसे दो बार"}),
                questions: q.clone(),
                kwargs: vec![("lang", "en")],
                want: "english",
            },
            RouteCase {
                label: "explicit lang de",
                state: json!({"body": "hello there"}),
                questions: q.clone(),
                kwargs: vec![("lang", "de")],
                want: "multilingual",
            },
            RouteCase {
                label: "td workflow, auto OFF",
                state: json!({"body": "I was charged twice"}),
                questions: td,
                kwargs: vec![],
                want: "english",
            },
            RouteCase {
                label: "empty state",
                state: json!({}),
                questions: q.clone(),
                kwargs: vec![],
                want: "english",
            },
            RouteCase {
                label: "none state",
                state: Value::Null,
                questions: q.clone(),
                kwargs: vec![],
                want: "english",
            },
            RouteCase {
                label: "digits only",
                state: json!("12345"),
                questions: q,
                kwargs: vec![],
                want: "english",
            },
        ];
        for case in cases {
            let kw = |key: &str| case.kwargs.iter().find(|(k, _)| *k == key).map(|(_, v)| *v);
            let got = r
                .route(
                    &case.state,
                    &case.questions,
                    kw("model"),
                    kw("task"),
                    kw("lang"),
                )
                .unwrap();
            assert_eq!(got.model, case.want, "route/{}", case.label);
        }
    }

    #[test]
    fn auto_task_detection_is_opt_in() {
        let q = generic_questions();
        let td = td_questions();
        let r_auto = Router::new(RouterOptions {
            auto_task_detection: Some(true),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            r_auto
                .route(
                    &json!({"body": "I was charged twice"}),
                    &td,
                    None,
                    None,
                    None
                )
                .unwrap()
                .model,
            "typed-decisions"
        );
        assert_eq!(
            r_auto
                .route(
                    &json!({"body": "I was charged twice"}),
                    &q,
                    None,
                    None,
                    None
                )
                .unwrap()
                .model,
            "english"
        );
        // explicit model still beats auto-detected workflow
        assert_eq!(
            r_auto
                .route(&json!({"body": "x"}), &td, Some("multilingual"), None, None)
                .unwrap()
                .model,
            "multilingual"
        );
    }

    #[test]
    fn decision_payload_shape() {
        let r = router();
        let d = r
            .route(
                &json!({"body": "मुझसे दो बार शुल्क लिया गया"}),
                &generic_questions(),
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(d.repo.id(), "convaiinnovations/laya/multilingual");
        assert!(!d.reason.is_empty());
        assert_eq!(d.detection.as_ref().unwrap().script, "devanagari");
        let v = d.to_value();
        assert_eq!(v["model"], json!("multilingual"));
        assert_eq!(v["detection"]["script"], json!("devanagari"));
        assert_eq!(v["workflow"], Value::Null);
    }

    #[test]
    fn workflow_branch_keeps_the_raw_repo_pair() {
        // laya/router.py emits `self.models["typed-decisions"]` (the raw pair) in this one
        // branch while every other branch emits `_repo_str(...)`. The port reproduces it;
        // this test pins the behaviour so a future "cleanup" cannot drift silently.
        let r_auto = Router::new(RouterOptions {
            auto_task_detection: Some(true),
            ..Default::default()
        })
        .unwrap();
        let d = r_auto
            .route(
                &json!({"body": "I was charged twice"}),
                &td_questions(),
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(
            d.repo,
            Repo::Spec(ModelSpec {
                repo: "convaiinnovations/laya".to_string(),
                subfolder: Some("typed-decisions".to_string()),
            })
        );
        // ...and the string form is still available for humans
        assert_eq!(d.repo.id(), "convaiinnovations/laya/typed-decisions");
        assert_eq!(
            d.to_value()["repo"],
            json!(["convaiinnovations/laya", "typed-decisions"])
        );
    }

    #[test]
    fn custom_default() {
        let r = Router::new(RouterOptions {
            default: Some("multilingual".to_string()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            r.route(&json!("12345"), &generic_questions(), None, None, None)
                .unwrap()
                .model,
            "multilingual"
        );
    }

    // ------------------------------------------------------------------ LRU bookkeeping
    fn stubbed_router(max_loaded: usize) -> Router {
        let mut r = Router::new(RouterOptions {
            max_loaded: Some(max_loaded),
            ..Default::default()
        })
        .unwrap();
        let loader: Loader = Arc::new(|spec: &ModelSpec| {
            let key = spec
                .subfolder
                .clone()
                .unwrap_or_else(|| "english".to_string());
            Ok(Arc::new(Stub(key)) as Arc<dyn Predictor>)
        });
        r = r.with_loader(loader);
        r
    }

    #[test]
    fn lru_cap_and_eviction() {
        let mut rr = stubbed_router(1);
        rr.load("english").unwrap();
        rr.load("multilingual").unwrap();
        assert_eq!(rr.loaded(), vec!["multilingual".to_string()]);

        let mut rr = stubbed_router(2);
        rr.load("english").unwrap();
        rr.load("multilingual").unwrap();
        rr.load("typed-decisions").unwrap();
        assert_eq!(
            rr.loaded(),
            vec!["multilingual".to_string(), "typed-decisions".to_string()]
        );
    }

    #[test]
    fn lru_touch_protects() {
        let mut rr = stubbed_router(2);
        rr.load("english").unwrap();
        rr.load("multilingual").unwrap();
        rr.load("english").unwrap(); // touch english
        rr.load("typed-decisions").unwrap();
        let mut loaded = rr.loaded();
        loaded.sort();
        assert_eq!(
            loaded,
            vec!["english".to_string(), "typed-decisions".to_string()]
        );
    }

    #[test]
    fn unload_one_and_all() {
        let mut rr = stubbed_router(2);
        rr.load("english").unwrap();
        rr.load("multilingual").unwrap();
        rr.unload(Some("english")).unwrap();
        assert!(!rr.loaded().contains(&"english".to_string()));
        rr.unload(None).unwrap();
        assert!(rr.loaded().is_empty());
    }

    // ------------------------------------------------------------------ preload
    #[test]
    fn preload_keeps_everything_resident() {
        let mut rp = stubbed_router(1);
        rp.preload(None).unwrap();
        let mut loaded = rp.loaded();
        loaded.sort();
        assert_eq!(
            loaded,
            vec![
                "english".to_string(),
                "multilingual".to_string(),
                "typed-decisions".to_string()
            ]
        );
        assert!(rp.max_loaded() >= 3);

        let mut rp2 = stubbed_router(1);
        rp2.preload(Some(&["english".to_string(), "multilingual".to_string()]))
            .unwrap();
        let mut loaded = rp2.loaded();
        loaded.sort();
        assert_eq!(
            loaded,
            vec!["english".to_string(), "multilingual".to_string()]
        );
        // routing to an already-resident checkpoint must not evict anything
        rp2.load("english").unwrap();
        let mut loaded = rp2.loaded();
        loaded.sort();
        assert_eq!(
            loaded,
            vec!["english".to_string(), "multilingual".to_string()]
        );
    }

    // ------------------------------------------------------------------ attach
    #[test]
    fn attach_registers_and_survives() {
        let mut ra = stubbed_router(1);
        let sentinel: Arc<dyn Predictor> = Arc::new(Stub("already-built".to_string()));
        ra.attach("english", sentinel.clone()).unwrap();
        assert!(ra.loaded().contains(&"english".to_string()));
        assert!(ra.max_loaded() >= 1);
        ra.set_max_loaded(ra.max_loaded().max(2));
        ra.load("multilingual").unwrap();
        let mut loaded = ra.loaded();
        loaded.sort();
        assert_eq!(
            loaded,
            vec!["english".to_string(), "multilingual".to_string()]
        );
        // attaching accepts aliases
        assert!(stubbed_router(1).attach("en", sentinel).is_ok());
    }

    // ------------------------------------------------------------------ bundle vs standalone
    #[test]
    fn bundle_versus_standalone() {
        let q = generic_questions();
        let r_bundle = router();
        let r_alone = Router::new(RouterOptions {
            standalone_repos: Some(true),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            r_bundle
                .route(&json!({"m": "मुझसे दो बार"}), &q, None, None, None)
                .unwrap()
                .repo
                .id(),
            "convaiinnovations/laya/multilingual"
        );
        assert_eq!(
            r_alone
                .route(&json!({"m": "मुझसे दो बार"}), &q, None, None, None)
                .unwrap()
                .repo
                .id(),
            "convaiinnovations/laya-multilingual"
        );
        assert_eq!(
            r_alone
                .route(&json!({"m": "I was charged twice"}), &q, None, None, None)
                .unwrap()
                .repo
                .id(),
            "convaiinnovations/laya"
        );
        // a local-path override must still work
        let mut overrides = IndexMap::new();
        overrides.insert(
            "english".to_string(),
            ModelSpec {
                repo: "/tmp/en".to_string(),
                subfolder: None,
            },
        );
        overrides.insert(
            "multilingual".to_string(),
            ModelSpec {
                repo: "/tmp/ml".to_string(),
                subfolder: None,
            },
        );
        let r_local = Router::new(RouterOptions {
            models: Some(overrides),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            r_local
                .route(&json!({"m": "मुझसे दो बार"}), &q, None, None, None)
                .unwrap()
                .repo
                .id(),
            "/tmp/ml"
        );
    }

    // ------------------------------------------------------------------ predict
    #[test]
    fn predict_adds_the_routing_payload() {
        let mut r = stubbed_router(1);
        let res = r
            .predict(
                &json!({"message": "I was charged twice, please refund."}),
                &generic_questions(),
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(res["routing"]["model"], json!("english"));
        assert_eq!(res["model"], json!("english"));
        assert!(res["answers"].is_object());
    }

    #[test]
    fn predict_without_a_loader_is_a_clear_error() {
        let mut r = router().without_loader();
        let err = r
            .predict(
                &json!({"message": "hi"}),
                &generic_questions(),
                None,
                None,
                None,
            )
            .unwrap_err();
        assert_eq!(err, LayaError::InferenceUnavailable);
    }
}
