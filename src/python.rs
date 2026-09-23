//! Python bindings: the `laya_rs` extension module.
//!
//! Built with maturin (`maturin develop --features python`). The surface mirrors the
//! pure-logic half of the `laya` package; the inference runtime arrives in phase 2.

#![allow(clippy::too_many_arguments)]

use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyString, PyTuple};
use serde_json::{Map, Value};

use crate::agent::Agent;
use crate::criteria::{self, QType, Question, Questions, QTYPES, QTYPE_NAMES};
use crate::email;
use crate::lang;
use crate::presets;
use crate::router::{self, default_models, ModelSpec, RouteDecision, Router, RouterOptions};
use crate::LayaError;

impl From<LayaError> for PyErr {
    fn from(e: LayaError) -> PyErr {
        // Python raises FileNotFoundError for absent files and ValueError for everything
        // else; keep that split so `except FileNotFoundError` keeps working.
        match e {
            LayaError::NotFound(msg) => PyFileNotFoundError::new_err(msg),
            other => PyValueError::new_err(other.to_string()),
        }
    }
}

// --------------------------------------------------------------------- conversion
/// Convert a Python object to JSON. Dict keys must be strings, as in the laya schemas.
pub fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::from(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::from(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        let mut m = Map::new();
        for (k, v) in d.iter() {
            m.insert(k.extract::<String>()?, py_to_json(&v)?);
        }
        return Ok(Value::Object(m));
    }
    if let Ok(l) = obj.cast::<PyList>() {
        let mut out = Vec::with_capacity(l.len());
        for item in l.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(Value::Array(out));
    }
    if let Ok(t) = obj.cast::<PyTuple>() {
        let mut out = Vec::with_capacity(t.len());
        for item in t.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(Value::Array(out));
    }
    Err(PyValueError::new_err(
        "unsupported Python type; laya states and questions are str/dict/list/None",
    ))
}

/// Convert JSON back to a Python object.
pub fn json_to_py(py: Python<'_>, v: &Value) -> Py<PyAny> {
    match v {
        Value::Null => py.None(),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any().unbind(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py).unwrap().into_any().unbind()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py).unwrap().into_any().unbind()
            } else {
                n.as_f64()
                    .unwrap()
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind()
            }
        }
        Value::String(s) => PyString::new(py, s).into_any().unbind(),
        Value::Array(a) => {
            let list = PyList::empty(py);
            for item in a {
                list.append(json_to_py(py, item))
                    .expect("append to a fresh list");
            }
            list.into_any().unbind()
        }
        Value::Object(m) => {
            let dict = PyDict::new(py);
            for (k, item) in m {
                dict.set_item(k, json_to_py(py, item))
                    .expect("set on a fresh dict");
            }
            dict.into_any().unbind()
        }
    }
}

fn state_arg(state: Option<&Bound<'_, PyAny>>) -> PyResult<Value> {
    match state {
        Some(s) => py_to_json(s),
        None => Ok(Value::Null),
    }
}

/// Routing only reads question ids, so definitions are not validated here.
fn question_ids_arg(questions: &Bound<'_, PyAny>) -> PyResult<Questions> {
    Questions::ids_only(&py_to_json(questions)?).map_err(PyErr::from)
}

/// Inference needs the full definitions, so they are validated here.
fn questions_arg(questions: &Bound<'_, PyAny>) -> PyResult<Questions> {
    Questions::from_public(&py_to_json(questions)?).map_err(PyErr::from)
}

// --------------------------------------------------------------------- module
/// `RouteDecision` behaves as a dict (like the Python subclass) and exposes `.model` / `.reason`.
#[pyclass(name = "RouteDecision", module = "laya_rs", mapping)]
struct PyRouteDecision {
    inner: RouteDecision,
}

#[pymethods]
impl PyRouteDecision {
    #[getter]
    fn model(&self) -> String {
        self.inner.model.clone()
    }

    #[getter]
    fn reason(&self) -> String {
        self.inner.reason.clone()
    }

    #[getter]
    fn repo(&self, py: Python<'_>) -> Py<PyAny> {
        json_to_py(py, &self.inner.repo.to_value())
    }

    #[getter]
    fn detection(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.inner.detection {
            Some(a) => json_to_py(py, &a.to_value()),
            None => py.None(),
        }
    }

    #[getter]
    fn workflow(&self) -> Option<String> {
        self.inner.workflow.clone()
    }

    fn keys(&self) -> Vec<String> {
        vec![
            "model".to_string(),
            "repo".to_string(),
            "reason".to_string(),
            "detection".to_string(),
            "workflow".to_string(),
        ]
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> Py<PyAny> {
        let v = self.inner.to_value();
        match v.get(key) {
            Some(v) => json_to_py(py, v),
            None => py.None(),
        }
    }

    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> Py<PyAny> {
        let v = self.inner.to_value();
        match v.get(key) {
            Some(v) => json_to_py(py, v),
            None => default.unwrap_or_else(|| py.None()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RouteDecision(model={:?}, reason={:?})",
            self.inner.model, self.inner.reason
        )
    }
}

/// Lazily loads Laya checkpoints and sends each request to the right one.
#[pyclass(name = "Router", module = "laya_rs")]
struct PyRouter {
    inner: Router,
}

#[pymethods]
impl PyRouter {
    #[new]
    #[pyo3(signature = (*, models=None, device=None, token=None, max_loaded=None, default=None,
                        auto_task_detection=None, standalone_repos=None, preload=false))]
    fn new(
        models: Option<&Bound<'_, PyDict>>,
        device: Option<String>,
        token: Option<String>,
        max_loaded: Option<usize>,
        default: Option<String>,
        auto_task_detection: Option<bool>,
        standalone_repos: Option<bool>,
        preload: bool,
    ) -> PyResult<Self> {
        let models = match models {
            Some(m) => {
                let mut out = indexmap::IndexMap::new();
                for (k, v) in m.iter() {
                    out.insert(
                        k.extract::<String>()?,
                        ModelSpec::from_value(&py_to_json(&v)?),
                    );
                }
                Some(out)
            }
            None => None,
        };
        let mut inner = Router::new(RouterOptions {
            models,
            device,
            token,
            max_loaded,
            default,
            auto_task_detection,
            standalone_repos,
        })?;
        // A cold load costs seconds, so a server or demo asks for the checkpoints up front.
        if preload {
            inner.preload(None)?;
        }
        Ok(PyRouter { inner })
    }

    /// Decide which checkpoint to use, without loading or running anything.
    #[pyo3(signature = (state, questions, model=None, task=None, lang=None))]
    fn route(
        &self,
        state: Option<&Bound<'_, PyAny>>,
        questions: &Bound<'_, PyAny>,
        model: Option<String>,
        task: Option<String>,
        lang: Option<String>,
    ) -> PyResult<PyRouteDecision> {
        let state = state_arg(state)?;
        let questions = question_ids_arg(questions)?;
        let decision = self.inner.route(
            &state,
            &questions,
            model.as_deref(),
            task.as_deref(),
            lang.as_deref(),
        )?;
        Ok(PyRouteDecision { inner: decision })
    }

    #[getter]
    fn loaded(&self) -> Vec<String> {
        self.inner.loaded()
    }

    #[getter]
    fn max_loaded(&self) -> usize {
        self.inner.max_loaded()
    }

    #[setter]
    fn set_max_loaded(&mut self, n: usize) {
        self.inner.set_max_loaded(n);
    }

    #[getter]
    fn default(&self) -> String {
        self.inner.default().to_string()
    }

    #[getter]
    fn auto_task_detection(&self) -> bool {
        self.inner.auto_task_detection()
    }

    /// Route, then answer every question in one forward pass on the chosen checkpoint.
    ///
    /// The result is the usual `system_one` payload plus a `routing` key recording the
    /// decision. Loads the checkpoint on first use, so the first call may download weights.
    #[pyo3(signature = (state, questions, model=None, task=None, lang=None))]
    fn predict(
        &mut self,
        py: Python<'_>,
        state: Option<&Bound<'_, PyAny>>,
        questions: &Bound<'_, PyAny>,
        model: Option<String>,
        task: Option<String>,
        lang: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let state = state_arg(state)?;
        let questions = questions_arg(questions)?;
        let result = self.inner.predict(
            &state,
            &questions,
            model.as_deref(),
            task.as_deref(),
            lang.as_deref(),
        )?;
        Ok(json_to_py(py, &result))
    }

    /// Download and build a checkpoint up front, so no request ever pays a model load.
    ///
    /// Note: the Python version returns the built `Agent`; that object is not exposed
    /// here, so this returns `None` once the load has succeeded.
    fn load(&mut self, name: &str) -> PyResult<()> {
        self.inner.load(name).map(|_| ()).map_err(PyErr::from)
    }

    /// Download and build checkpoints up front (all of them when `names` is omitted).
    #[pyo3(signature = (names=None))]
    fn preload(&mut self, names: Option<Vec<String>>) -> PyResult<()> {
        self.inner.preload(names.as_deref()).map_err(PyErr::from)
    }

    /// Register an already-built Agent under `name` instead of loading a second copy.
    fn attach(&mut self, name: &str, agent: &PyAgent) -> PyResult<()> {
        self.inner
            .attach(name, agent.inner.clone())
            .map(|_| ())
            .map_err(PyErr::from)
    }

    /// Alias of `predict`, matching the Python package's `Router.system_one = predict`.
    #[pyo3(signature = (state, questions, model=None, task=None, lang=None))]
    fn system_one(
        &mut self,
        py: Python<'_>,
        state: Option<&Bound<'_, PyAny>>,
        questions: &Bound<'_, PyAny>,
        model: Option<String>,
        task: Option<String>,
        lang: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        self.predict(py, state, questions, model, task, lang)
    }

    /// Free one model, or all of them.
    #[pyo3(signature = (name=None))]
    fn unload(&mut self, name: Option<String>) -> PyResult<()> {
        self.inner.unload(name.as_deref()).map_err(PyErr::from)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// A loaded Laya checkpoint: the inference runtime.
///
/// Holds an `Arc` so the same checkpoint can be handed to `Router.attach` without
/// duplicating it in memory.
#[pyclass(name = "Agent", module = "laya_rs")]
struct PyAgent {
    inner: std::sync::Arc<Agent>,
}

#[pymethods]
impl PyAgent {
    /// Load a checkpoint from a local directory or a Hub repo.
    ///
    /// `subfolder` selects one checkpoint from a repo that bundles several, e.g.
    /// `Agent("convaiinnovations/laya", subfolder="multilingual")`.
    #[new]
    #[pyo3(signature = (model_id_or_path="convaiinnovations/laya", device=None, token=None, subfolder=None))]
    fn new(
        model_id_or_path: &str,
        device: Option<String>,
        token: Option<String>,
        subfolder: Option<String>,
    ) -> PyResult<Self> {
        let inner = std::sync::Arc::new(Agent::load(
            model_id_or_path,
            device.as_deref(),
            token.as_deref(),
            subfolder.as_deref(),
        )?);
        Ok(PyAgent { inner })
    }

    /// Evaluate typed questions across state in a single, parallel forward pass.
    #[pyo3(signature = (state, questions))]
    fn system_one(
        &self,
        py: Python<'_>,
        state: Option<&Bound<'_, PyAny>>,
        questions: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let state = state_arg(state)?;
        let questions = questions_arg(questions)?;
        // Release the GIL for the forward pass: it is pure Rust and can take seconds, and
        // holding it would serialise every other Python thread in the process.
        let result = py.detach(|| self.inner.system_one(&state, &questions))?;
        Ok(json_to_py(py, &result))
    }

    /// Alias of `system_one`, matching the Python package.
    #[pyo3(signature = (state, questions))]
    fn predict(
        &self,
        py: Python<'_>,
        state: Option<&Bound<'_, PyAny>>,
        questions: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.system_one(py, state, questions)
    }

    #[getter]
    fn device(&self) -> String {
        self.inner.device().to_string()
    }

    #[getter]
    fn max_len(&self) -> usize {
        self.inner.max_len()
    }

    #[getter]
    fn head_max_len(&self) -> usize {
        self.inner.head_max_len()
    }

    #[getter]
    fn config(&self, py: Python<'_>) -> Py<PyAny> {
        json_to_py(py, self.inner.config())
    }

    fn __repr__(&self) -> String {
        format!(
            "Agent(device={:?}, max_len={})",
            self.inner.device(),
            self.inner.max_len()
        )
    }
}

#[pymodule]
fn laya_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    // language / script detection
    m.add_function(wrap_pyfunction!(detect_script, m)?)?;
    m.add_function(wrap_pyfunction!(script_profile, m)?)?;
    m.add_function(wrap_pyfunction!(guess_latin_language, m)?)?;
    m.add_function(wrap_pyfunction!(state_text, m)?)?;
    m.add_function(wrap_pyfunction!(is_english, m)?)?;
    m.add_function(wrap_pyfunction!(analyse, m)?)?;
    // routing
    m.add_function(wrap_pyfunction!(normalise_name, m)?)?;
    m.add_function(wrap_pyfunction!(match_typed_decisions_workflow, m)?)?;
    m.add_class::<PyRouter>()?;
    m.add_class::<PyRouteDecision>()?;
    m.add_class::<PyAgent>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(detect_language, m)?)?;

    // Module-level constants, matching the Python package's exports.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("QTYPES", qtypes_dict(py)?)?;
    m.add("QTYPE_NAMES", qtype_names_dict(py)?)?;
    m.add("DEFAULT_MODELS", default_models_dict(py)?)?;
    // RLAgent is an alias of Agent in the Python package, so bind the same class object.
    m.add("RLAgent", py.get_type::<PyAgent>())?;
    // criteria
    m.add_function(wrap_pyfunction!(render_criterion, m)?)?;
    m.add_function(wrap_pyfunction!(render_options, m)?)?;
    m.add_function(wrap_pyfunction!(confidence_from_probs, m)?)?;
    m.add_function(wrap_pyfunction!(temp_bucket, m)?)?;
    m.add_function(wrap_pyfunction!(ece_score, m)?)?;
    // presets
    m.add_function(wrap_pyfunction!(triage_questions, m)?)?;
    m.add_function(wrap_pyfunction!(email_questions, m)?)?;
    m.add_function(wrap_pyfunction!(guard_questions, m)?)?;
    m.add_function(wrap_pyfunction!(moderation_questions, m)?)?;
    m.add_function(wrap_pyfunction!(router_questions, m)?)?;
    // email
    m.add_function(wrap_pyfunction!(clean_email_body, m)?)?;
    m.add_function(wrap_pyfunction!(email_state, m)?)?;
    Ok(())
}

// --------------------------------------------------------------------- functions
#[pyfunction]
fn detect_script(text: &str) -> String {
    lang::detect_script(text)
}

#[pyfunction]
fn script_profile(py: Python<'_>, text: &str) -> Py<PyAny> {
    let prof = lang::script_profile(text);
    let dict = PyDict::new(py);
    for (k, v) in prof {
        dict.set_item(k, v).expect("set on a fresh dict");
    }
    dict.into_any().unbind()
}

#[pyfunction]
fn guess_latin_language(text: &str) -> Option<String> {
    lang::guess_latin_language(text)
}

#[pyfunction]
#[pyo3(signature = (state=None))]
fn state_text(state: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
    Ok(lang::state_text(&state_arg(state)?))
}

#[pyfunction]
#[pyo3(signature = (state=None))]
fn is_english(state: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
    Ok(lang::is_english(&state_arg(state)?))
}

#[pyfunction]
#[pyo3(signature = (state=None))]
fn analyse(py: Python<'_>, state: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    Ok(json_to_py(
        py,
        &lang::analyse(&state_arg(state)?).to_value(),
    ))
}

/// `{0: "choice", 1: "score", 2: "noul"}` -- Python derives `QTYPE_NAMES` by inverting
/// `QTYPES`, so it is a dict keyed by index, not a list.
fn qtype_names_dict(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    for (index, name) in QTYPE_NAMES.iter().enumerate() {
        dict.set_item(index, name)?;
    }
    Ok(dict.into_any().unbind())
}

/// `{"choice": 0, "score": 1, "noul": 2}`, the shape Python's `QTYPES` has.
fn qtypes_dict(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    for (name, index) in QTYPES {
        dict.set_item(name, index)?;
    }
    Ok(dict.into_any().unbind())
}

/// `{"english": (repo, None), "multilingual": (repo, "multilingual"), ...}`, matching
/// Python's `DEFAULT_MODELS` (values are `(repo, subfolder)` tuples).
fn default_models_dict(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    let tuple_type = py.get_type::<PyTuple>();
    for (name, spec) in default_models() {
        let sub = match &spec.subfolder {
            Some(s) => PyString::new(py, s).into_any().unbind(),
            None => py.None(),
        };
        // NOTE: `PyTuple::new` is broken in pyo3 0.29.2 -- it drops the first element and
        // leaves the last slot uninitialised (reproduced with arrays, Vecs and iterators,
        // while `PyList::new` is fine). Build a list and convert with `tuple(...)` instead.
        let items = PyList::new(py, [PyString::new(py, &spec.repo).into_any().unbind(), sub])?;
        let tuple = tuple_type.call1((items,))?;
        dict.set_item(name, tuple)?;
    }
    Ok(dict.into_any().unbind())
}

/// Alias of `analyse`, matching the Python package's `detect_language` export.
#[pyfunction]
#[pyo3(signature = (state=None))]
fn detect_language(py: Python<'_>, state: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    analyse(py, state)
}

#[pyfunction]
fn normalise_name(name: &str) -> PyResult<String> {
    router::normalise_name(name).map_err(PyErr::from)
}

#[pyfunction]
fn match_typed_decisions_workflow(questions: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    Ok(router::match_typed_decisions_workflow(&question_ids_arg(
        questions,
    )?))
}

#[pyfunction]
fn render_criterion(value: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(criteria::render_criterion(&py_to_json(value)?))
}

#[pyfunction]
fn render_options(question: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let q = Question::from_internal(&py_to_json(question)?)?;
    criteria::render_options(&q).map_err(PyErr::from)
}

#[pyfunction]
fn confidence_from_probs(p: Vec<f64>, k: usize) -> f64 {
    criteria::confidence_from_probs(&p, k)
}

#[pyfunction]
#[pyo3(signature = (qtype, k))]
fn temp_bucket(qtype: usize, k: usize) -> PyResult<String> {
    let qt = QType::from_index(qtype)
        .ok_or_else(|| LayaError::QuestionUnknownType(format!("{} (expected 0, 1 or 2)", qtype)))?;
    Ok(criteria::temp_bucket(qt, k))
}

#[pyfunction]
#[pyo3(signature = (conf, correct, bins=15))]
fn ece_score(conf: Vec<f64>, correct: Vec<f64>, bins: usize) -> f64 {
    criteria::ece_score(&conf, &correct, bins)
}

/// Load a Laya checkpoint. `subfolder` picks one out of a repo that bundles several.
#[pyfunction]
#[pyo3(signature = (model_id_or_path="convaiinnovations/laya", device=None, token=None, subfolder=None))]
fn load(
    model_id_or_path: &str,
    device: Option<String>,
    token: Option<String>,
    subfolder: Option<String>,
) -> PyResult<PyAgent> {
    PyAgent::new(model_id_or_path, device, token, subfolder)
}

#[pyfunction]
fn triage_questions(py: Python<'_>) -> Py<PyAny> {
    json_to_py(py, &presets::triage_questions().to_public())
}

#[pyfunction]
#[pyo3(signature = (categories=None))]
fn email_questions(py: Python<'_>, categories: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    let cats: Option<Vec<(String, String)>> = match categories {
        Some(m) => {
            let mut out = Vec::with_capacity(m.len());
            for (k, v) in m.iter() {
                out.push((k.extract::<String>()?, v.extract::<String>()?));
            }
            Some(out)
        }
        None => None,
    };
    Ok(json_to_py(
        py,
        &presets::email_questions(cats.as_deref()).to_public(),
    ))
}

#[pyfunction]
fn guard_questions(py: Python<'_>) -> Py<PyAny> {
    json_to_py(py, &presets::guard_questions().to_public())
}

#[pyfunction]
fn moderation_questions(py: Python<'_>) -> Py<PyAny> {
    json_to_py(py, &presets::moderation_questions().to_public())
}

#[pyfunction]
fn router_questions(py: Python<'_>) -> Py<PyAny> {
    json_to_py(py, &presets::router_questions().to_public())
}

#[pyfunction]
#[pyo3(signature = (body, max_chars=3000))]
fn clean_email_body(body: Option<String>, max_chars: usize) -> String {
    email::clean_email_body(body.as_deref(), max_chars)
}

#[pyfunction]
#[pyo3(signature = (subject, body, sender=None, clean=true, **kwargs))]
fn email_state(
    py: Python<'_>,
    subject: Option<String>,
    body: Option<String>,
    sender: Option<String>,
    clean: bool,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let extra: Vec<(String, Option<String>)> = match kwargs {
        Some(m) => {
            let mut out = Vec::with_capacity(m.len());
            for (k, v) in m.iter() {
                let value = if v.is_none() {
                    None
                } else {
                    Some(v.extract::<String>()?)
                };
                out.push((k.extract::<String>()?, value));
            }
            out
        }
        None => Vec::new(),
    };
    let pairs = email::email_state(
        subject.as_deref(),
        body.as_deref(),
        sender.as_deref(),
        clean,
        &extra,
    );
    let dict = PyDict::new(py);
    for (k, v) in pairs {
        dict.set_item(k, v).expect("set on a fresh dict");
    }
    Ok(dict.into_any().unbind())
}
