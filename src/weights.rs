//! Strict safetensors weight loading.
//!
//! The Python runtime calls `load_state_dict(weights, strict=True)`, so every tensor in the
//! checkpoint must be consumed and every expected tensor must be present. This loader keeps
//! the same guarantee: it tracks which keys were read and fails if the two sets differ.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use serde_json::Value;

use crate::LayaError;

/// Checkpoint tensors, upcast to f32, with strict consumption tracking.
pub struct Weights {
    tensors: HashMap<String, Tensor>,
    consumed: Vec<String>,
    path: String,
}

impl Weights {
    /// Load `model.safetensors` from `path` onto `device`, upcasting every tensor to f32.
    pub fn load(path: &str, device: &Device) -> Result<Self, LayaError> {
        let data = std::fs::read(path)
            .map_err(|e| LayaError::WeightsUnreadable(path.to_string(), e.to_string()))?;
        let tensors = candle_core::safetensors::load_buffer(&data, device)
            .map_err(|e| LayaError::WeightsUnreadable(path.to_string(), e.to_string()))?;
        let mut map = HashMap::new();
        for (name, tensor) in tensors {
            let tensor = tensor
                .to_dtype(DType::F32)
                .map_err(|e| LayaError::WeightsUnreadable(path.to_string(), e.to_string()))?;
            map.insert(name, tensor);
        }
        Ok(Weights {
            tensors: map,
            consumed: Vec::new(),
            path: path.to_string(),
        })
    }

    /// Take a tensor by key, recording that it was used.
    pub fn take(&mut self, key: &str) -> Result<Tensor, LayaError> {
        let tensor = self
            .tensors
            .get(key)
            .ok_or_else(|| LayaError::WeightsIncomplete(self.path.clone(), key.to_string()))?
            .clone();
        self.consumed.push(key.to_string());
        Ok(tensor)
    }

    /// Take a tensor if present (for genuinely optional weights).
    pub fn take_opt(&mut self, key: &str) -> Option<Tensor> {
        let tensor = self.tensors.get(key)?.clone();
        self.consumed.push(key.to_string());
        Some(tensor)
    }

    /// Check a checkpoint against an expected plan before anything is loaded.
    ///
    /// Mirrors `Agent._verify_compatibility`: required config keys, required weight
    /// prefixes, then presence and shape of every planned tensor. Errors carry the same
    /// wording as the Python ones so both runtimes fail the same way.
    pub fn verify_compatibility(
        &self,
        plan: &[(String, Vec<usize>)],
        model_id: &str,
    ) -> Result<(), LayaError> {
        // 1. required component prefixes
        for prefix in ["encoder.", "type_emb.", "scorer.", "act_head."] {
            if !self.tensors.keys().any(|k| k.starts_with(prefix)) {
                return Err(LayaError::Incompatible(format!(
                    "Incompatible model weights for {}: checkpoint is missing '{}' parameters. \
                     Expected an RL Agent decision model with encoder and decision heads.",
                    crate::router::py_repr(model_id),
                    prefix
                )));
            }
        }

        // 2. presence and shape of every planned tensor
        let mut shape_mismatches: Vec<String> = Vec::new();
        let mut missing: Vec<String> = Vec::new();
        for (key, want) in plan {
            match self.tensors.get(key) {
                None => missing.push(key.clone()),
                Some(t) => {
                    let got = t.dims();
                    if got != want.as_slice() {
                        shape_mismatches.push(format!(
                            "  - {}: expected {}, found {}",
                            key,
                            py_tuple(want),
                            py_tuple(got)
                        ));
                    }
                }
            }
        }
        if !shape_mismatches.is_empty() {
            let mut details = shape_mismatches[..shape_mismatches.len().min(5)].join("\n");
            if shape_mismatches.len() > 5 {
                details += &format!(
                    "\n  ... and {} more mismatched layers.",
                    shape_mismatches.len() - 5
                );
            }
            return Err(LayaError::Incompatible(format!(
                "Model architecture mismatch for {}:\n{}\nThe checkpoint weights do not match \
                 the configured model architecture.",
                crate::router::py_repr(model_id),
                details
            )));
        }
        if !missing.is_empty() {
            return Err(LayaError::Incompatible(format!(
                "Model weights incomplete for {}: missing {} parameter tensors (e.g. {}).",
                crate::router::py_repr(model_id),
                missing.len(),
                missing[..missing.len().min(3)].join(", ")
            )));
        }
        Ok(())
    }

    /// Fail if any checkpoint tensor was never read -- the `strict=True` guarantee.
    pub fn finish(&self) -> Result<(), LayaError> {
        let mut unused: Vec<&String> = self
            .tensors
            .keys()
            .filter(|k| !self.consumed.contains(k))
            .collect();
        unused.sort();
        if !unused.is_empty() {
            return Err(LayaError::WeightsUnexpected(
                self.path.clone(),
                unused.iter().map(|s| s.to_string()).collect(),
            ));
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Read a JSON config file.
pub fn read_json(path: &str) -> Result<Value, LayaError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| LayaError::ConfigUnreadable(path.to_string(), e.to_string()))?;
    serde_json::from_str(&text)
        .map_err(|e| LayaError::ConfigUnreadable(path.to_string(), e.to_string()))
}

/// Format a shape the way Python's `tuple` repr does: `(3, 4)` / `(3,)`.
fn py_tuple(shape: &[usize]) -> String {
    let inner = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    if shape.len() == 1 {
        format!("({},)", inner)
    } else {
        format!("({})", inner)
    }
}

/// Check that a config carries the keys the runtime requires.
///
/// Mirrors the first half of `_verify_compatibility`.
pub fn verify_config_keys(cfg: &Value, required: &[&str], model_id: &str) -> Result<(), LayaError> {
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|k| cfg.get(k).is_none())
        .collect();
    if !missing.is_empty() {
        return Err(LayaError::Incompatible(format!(
            "Incompatible model config for {}: missing configuration keys {:?}. \
             Ensure this is a valid RL Agent decision model.",
            crate::router::py_repr(model_id),
            missing
        )));
    }
    Ok(())
}

/// Fetch a usize field from a config object.
pub fn cfg_usize(cfg: &Value, key: &str, default: usize) -> usize {
    cfg.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(default)
}

/// Fetch an f64 field from a config object.
pub fn cfg_f64(cfg: &Value, key: &str, default: f64) -> f64 {
    cfg.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

/// Fetch a string field from a config object.
pub fn cfg_string(cfg: &Value, key: &str, default: &str) -> String {
    cfg.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or(default)
        .to_string()
}

#[cfg(test)]
mod verify_tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn tensor(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let data = vec![0f32; n.max(1)];
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    fn plan(pairs: &[(&str, &[usize])]) -> Vec<(String, Vec<usize>)> {
        pairs
            .iter()
            .map(|(k, s)| (k.to_string(), s.to_vec()))
            .collect()
    }

    /// Build a Weights without touching the filesystem.
    fn weights_of(pairs: &[(&str, &[usize])]) -> Weights {
        let mut map = std::collections::HashMap::new();
        for (k, s) in pairs {
            map.insert(k.to_string(), tensor(s));
        }
        Weights {
            tensors: map,
            consumed: Vec::new(),
            path: "test".to_string(),
        }
    }

    #[test]
    fn accepts_a_matching_checkpoint() {
        let p = plan(&[
            ("encoder.layers.0.attn.Wqkv.weight", &[192, 64]),
            ("type_emb.weight", &[3, 64]),
            ("scorer.0.weight", &[64]),
            ("act_head.0.weight", &[256, 68]),
        ]);
        let w = weights_of(&[
            ("encoder.layers.0.attn.Wqkv.weight", &[192, 64]),
            ("type_emb.weight", &[3, 64]),
            ("scorer.0.weight", &[64]),
            ("act_head.0.weight", &[256, 68]),
        ]);
        w.verify_compatibility(&p, "some/repo").unwrap();
    }

    #[test]
    fn reports_a_shape_mismatch_like_python() {
        let p = plan(&[
            ("encoder.layers.0.attn.Wqkv.weight", &[192, 64]),
            ("encoder.layers.1.attn.Wqkv.weight", &[192, 64]),
            ("type_emb.weight", &[3, 64]),
            ("scorer.0.weight", &[64]),
            ("act_head.0.weight", &[256, 68]),
        ]);
        let w = weights_of(&[
            ("encoder.layers.0.attn.Wqkv.weight", &[192, 64]),
            ("encoder.layers.1.attn.Wqkv.weight", &[128, 64]), // wrong
            ("type_emb.weight", &[3, 64]),
            ("scorer.0.weight", &[64]),
            ("act_head.0.weight", &[256, 68]),
        ]);
        let err = w
            .verify_compatibility(&p, "some/repo")
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with("Model architecture mismatch for 'some/repo':"),
            "{}",
            err
        );
        assert!(err.contains("encoder.layers.1.attn.Wqkv.weight"), "{}", err);
        assert!(
            err.contains("expected (192, 64), found (128, 64)"),
            "{}",
            err
        );
        assert!(
            err.contains("do not match the configured model architecture"),
            "{}",
            err
        );
    }

    #[test]
    fn reports_a_missing_tensor_like_python() {
        let p = plan(&[
            ("encoder.layers.0.attn.Wqkv.weight", &[192, 64]),
            ("type_emb.weight", &[3, 64]),
            ("scorer.0.weight", &[64]),
            ("act_head.0.weight", &[256, 68]),
            ("scorer.1.weight", &[64, 64]),
        ]);
        let w = weights_of(&[
            ("encoder.layers.0.attn.Wqkv.weight", &[192, 64]),
            ("type_emb.weight", &[3, 64]),
            ("scorer.0.weight", &[64]),
            ("act_head.0.weight", &[256, 68]),
        ]);
        let err = w
            .verify_compatibility(&p, "some/repo")
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with("Model weights incomplete for 'some/repo':"),
            "{}",
            err
        );
        assert!(err.contains("missing 1 parameter tensors"), "{}", err);
        assert!(err.contains("scorer.1.weight"), "{}", err);
    }

    #[test]
    fn reports_a_missing_component_prefix() {
        let p = plan(&[("type_emb.weight", &[3, 64])]);
        // no encoder.* key at all
        let w = weights_of(&[("type_emb.weight", &[3, 64])]);
        let err = w
            .verify_compatibility(&p, "some/repo")
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with("Incompatible model weights for 'some/repo':"),
            "{}",
            err
        );
        assert!(err.contains("missing 'encoder.' parameters"), "{}", err);
    }

    #[test]
    fn config_keys_are_required() {
        let cfg = serde_json::json!({"encoder": "some/repo"});
        let err = verify_config_keys(&cfg, &["encoder", "head_layers"], "some/repo")
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with("Incompatible model config for 'some/repo':"),
            "{}",
            err
        );
        assert!(err.contains("missing configuration keys"), "{}", err);
        assert!(err.contains("head_layers"), "{}", err);

        let ok = serde_json::json!({"encoder": "r", "head_layers": 2});
        verify_config_keys(&ok, &["encoder", "head_layers"], "some/repo").unwrap();
    }

    #[test]
    fn finish_rejects_unused_tensors() {
        let mut w = weights_of(&[("type_emb.weight", &[3, 64]), ("mystery.weight", &[2])]);
        w.take("type_emb.weight").unwrap();
        let err = w.finish().unwrap_err().to_string();
        assert!(err.contains("mystery.weight"), "{}", err);
    }

    #[test]
    fn dtype_is_upcast_to_f32() {
        // the checkpoint stores F16; the loader must hand back F32
        let data = vec![0f32; 4];
        let t = Tensor::from_vec(data, (2, 2), &Device::Cpu).unwrap();
        assert_eq!(t.dtype(), DType::F32);
    }
}
