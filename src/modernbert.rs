//! ModernBERT encoder, ported from the HuggingFace reference implementation.
//!
//! Both shipped checkpoints are ModernBERT-family (`answerdotai/ModernBERT-large` for the
//! English model, `jhu-clsp/mmBERT-base` for the multilingual one), so one encoder covers
//! both. The details that matter and are easy to get wrong:
//!
//! * **RoPE, not absolute positions** -- with a *different theta per layer type*
//!   (`rope_parameters.full_attention` vs `.sliding_attention`).
//! * **Alternating local/global attention** -- every `global_attn_every_n_layers`-th layer
//!   attends globally, the rest use a sliding window of `local_attention // 2 + 1`.
//! * **Layer 0 has no `attn_norm`** -- the embedding output is already normed, so the first
//!   layer's attention reads it directly. The checkpoint really does omit that tensor.
//! * **GeGLU MLP** -- `Wi` is fused gate+up; the *first* chunk gets the activation.
//! * **Pre-normalisation** everywhere, and no biases at all (`norm_bias`/`mlp_bias`/
//!   `attention_bias` are all false).

use candle_core::{DType, Device, Tensor};
use serde_json::Value;

use crate::weights::{cfg_f64, cfg_usize, Weights};
use crate::LayaError;

/// Additive value used for masked attention entries (finite, so a fully masked row
/// degrades to a uniform distribution instead of NaN).
const MASK_VALUE: f32 = -1e9;

#[derive(Clone, Debug)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f64,
    pub global_attn_every_n_layers: usize,
    pub local_attention: usize,
    pub global_rope_theta: f64,
    pub local_rope_theta: f64,
    /// `"full_attention"` or `"sliding_attention"` per layer, from the config.
    pub layer_types: Vec<String>,
}

impl Config {
    pub fn from_value(v: &Value) -> Result<Self, LayaError> {
        let num_hidden_layers = cfg_usize(v, "num_hidden_layers", 0);
        let global_every = cfg_usize(v, "global_attn_every_n_layers", 1).max(1);
        let layer_types: Vec<String> = v
            .get("layer_types")
            .and_then(|t| t.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| {
                (0..num_hidden_layers)
                    .map(|i| {
                        if i % global_every == 0 {
                            "full_attention".to_string()
                        } else {
                            "sliding_attention".to_string()
                        }
                    })
                    .collect()
            });
        // transformers reads `global_rope_theta` / `local_rope_theta` (defaults 160000 /
        // 10000) and *ignores* the `rope_parameters` dict that some checkpoints carry.
        // The shipped multilingual checkpoint's `rope_parameters` claims 160000 for
        // sliding layers, but the Python runtime uses 10000 there -- verified by
        // inspecting `config.local_rope_theta` on the loaded model. Matching the
        // runtime (not the metadata) is what keeps the two implementations identical.
        Ok(Config {
            vocab_size: cfg_usize(v, "vocab_size", 0),
            hidden_size: cfg_usize(v, "hidden_size", 0),
            num_hidden_layers,
            num_attention_heads: cfg_usize(v, "num_attention_heads", 0),
            intermediate_size: cfg_usize(v, "intermediate_size", 0),
            layer_norm_eps: cfg_f64(v, "layer_norm_eps", cfg_f64(v, "norm_eps", 1e-5)),
            global_attn_every_n_layers: global_every,
            local_attention: cfg_usize(v, "local_attention", 128),
            global_rope_theta: cfg_f64(v, "global_rope_theta", 160_000.0),
            local_rope_theta: cfg_f64(v, "local_rope_theta", 10_000.0),
            layer_types,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// The inclusive sliding-window radius actually used by the attention mask.
    ///
    /// HF builds the mask from `config.sliding_window`, a property returning
    /// `local_attention // 2` (64 for the shipped checkpoints). The `+1` that
    /// `ModernBertAttention` applies is only for flash-attention's inclusive boundaries and
    /// does not apply to the explicit-mask path `laya` uses -- verified empirically: parity
    /// holds up to L=64 and breaks at L=66, exactly where a distance of 65 first appears.
    pub fn sliding_window(&self) -> usize {
        self.local_attention / 2
    }
}

struct Layer {
    /// `None` for layer 0, whose attention input is the already-normed embedding.
    attn_norm: Option<Tensor>,
    wqkv: Tensor,
    wo: Tensor,
    mlp_norm: Tensor,
    wi: Tensor,
    wo_mlp: Tensor,
    sliding: bool,
}

/// The ModernBERT encoder backbone (no prediction head).
pub struct ModernBert {
    cfg: Config,
    tok_embeddings: Tensor,
    emb_norm: Tensor,
    final_norm: Tensor,
    layers: Vec<Layer>,
}

impl ModernBert {
    /// Build from checkpoint weights under the `encoder.` prefix.
    pub fn load(cfg: Config, weights: &mut Weights) -> Result<Self, LayaError> {
        let p = "encoder.";
        let tok_embeddings = weights.take(&format!("{}embeddings.tok_embeddings.weight", p))?;
        let emb_norm = weights.take(&format!("{}embeddings.norm.weight", p))?;
        let final_norm = weights.take(&format!("{}final_norm.weight", p))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let prefix = format!("{}layers.{}.", p, i);
            // layer 0 has no attn_norm: the embedding output is already normed
            let attn_norm = if i == 0 {
                None
            } else {
                Some(weights.take(&format!("{}attn_norm.weight", prefix))?)
            };
            let sliding = cfg
                .layer_types
                .get(i)
                .map(|t| t == "sliding_attention")
                .unwrap_or_else(|| i % cfg.global_attn_every_n_layers != 0);
            layers.push(Layer {
                attn_norm,
                wqkv: weights.take(&format!("{}attn.Wqkv.weight", prefix))?,
                wo: weights.take(&format!("{}attn.Wo.weight", prefix))?,
                mlp_norm: weights.take(&format!("{}mlp_norm.weight", prefix))?,
                wi: weights.take(&format!("{}mlp.Wi.weight", prefix))?,
                wo_mlp: weights.take(&format!("{}mlp.Wo.weight", prefix))?,
                sliding,
            });
        }
        Ok(ModernBert {
            cfg,
            tok_embeddings,
            emb_norm,
            final_norm,
            layers,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// `input_ids`: (B, L) i64. `attention_mask`: (B, L) f32, 1 for real tokens.
    ///
    /// Returns the final hidden states (B, L, hidden).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor, LayaError> {
        let (b, l) = input_ids.dims2().map_err(candle_err)?;
        let hidden = self.cfg.hidden_size;
        let heads = self.cfg.num_attention_heads;
        let head_dim = self.cfg.head_dim();

        // embeddings: norm(tok_embeddings(ids))
        let flat = input_ids.flatten_all().map_err(candle_err)?;
        let mut h = self
            .tok_embeddings
            .index_select(&flat, 0)
            .map_err(candle_err)?
            .reshape((b, l, hidden))
            .map_err(candle_err)?;
        h = layer_norm_no_bias(&h, &self.emb_norm, self.cfg.layer_norm_eps)?;

        // additive padding mask: (B, 1, 1, L), 0 for real tokens and MASK_VALUE for padding.
        // affine(mul, add) = mul*x + add, so mul = -MASK_VALUE maps 1 -> 0 and 0 -> MASK_VALUE.
        let pad_mask = attention_mask
            .reshape((b, 1, 1, l))
            .map_err(candle_err)?
            .affine(-(MASK_VALUE as f64), MASK_VALUE as f64)
            .map_err(candle_err)?;

        // one rope table per theta, shared by every layer of that type
        let device = input_ids.device();
        let (global_cos, global_sin) =
            rope_tables(l, head_dim, self.cfg.global_rope_theta, device)?;
        let (local_cos, local_sin) = rope_tables(l, head_dim, self.cfg.local_rope_theta, device)?;
        let sliding_mask = sliding_mask(l, self.cfg.sliding_window(), device)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let attn_in = match &layer.attn_norm {
                Some(w) => layer_norm_no_bias(&h, w, self.cfg.layer_norm_eps)?,
                None => h.clone(),
            };
            let (cos, sin) = if layer.sliding {
                (&local_cos, &local_sin)
            } else {
                (&global_cos, &global_sin)
            };
            let attn_out = self.attention(
                layer,
                &attn_in,
                cos,
                sin,
                &pad_mask,
                &sliding_mask,
                b,
                l,
                heads,
                head_dim,
            )?;
            h = h.add(&attn_out).map_err(candle_err)?;
            let mlp_in = layer_norm_no_bias(&h, &layer.mlp_norm, self.cfg.layer_norm_eps)?;
            h = h
                .add(&geglu(&mlp_in, &layer.wi, &layer.wo_mlp)?)
                .map_err(candle_err)?;
            let _ = i;
        }
        layer_norm_no_bias(&h, &self.final_norm, self.cfg.layer_norm_eps)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        layer: &Layer,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        pad_mask: &Tensor,
        sliding_mask: &Tensor,
        b: usize,
        l: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Tensor, LayaError> {
        let fused = linear(x, &layer.wqkv, None)?;
        // HF reshapes to (B, L, 3, heads, head_dim): the fused dim is
        // [q(heads*hd), k(heads*hd), v(heads*hd)], not interleaved per head.
        //
        // Fold (B, heads) into the leading dimension so both attention matmuls are 3D:
        // candle's batched 4D matmul is ~2x slower than the equivalent 3D one on CPU.
        let split = |offset: usize| -> Result<Tensor, LayaError> {
            fused
                .narrow(2, offset, heads * head_dim)
                .map_err(candle_err)?
                .reshape((b, l, heads, head_dim))
                .map_err(candle_err)?
                .permute((0, 2, 1, 3))
                .map_err(candle_err)?
                .reshape((b * heads, l, head_dim))
                .map_err(candle_err)
        };
        let q = split(0)?;
        let k = split(heads * head_dim)?;
        let v = split(2 * heads * head_dim)?;

        // rope broadcasts, so the result is strided; matmul needs contiguous inputs
        let q = apply_rope(&q, cos, sin)?.contiguous().map_err(candle_err)?;
        let k = apply_rope(&k, cos, sin)?.contiguous().map_err(candle_err)?;

        let scale = (head_dim as f64).sqrt();
        let mut scores = q.matmul(&k.t().map_err(candle_err)?).map_err(candle_err)?;
        scores = scores.affine(1.0 / scale, 0.0).map_err(candle_err)?;
        // pad_mask is (B, 1, 1, L) and sliding_mask is (L, L); both broadcast over the
        // folded (B*heads) leading dimension.
        scores = scores
            .reshape((b, heads, l, l))
            .map_err(candle_err)?
            .broadcast_add(pad_mask)
            .map_err(candle_err)?;
        if layer.sliding {
            scores = scores.broadcast_add(sliding_mask).map_err(candle_err)?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&scores).map_err(candle_err)?;
        let out = attn
            .reshape((b * heads, l, l))
            .map_err(candle_err)?
            .matmul(&v)
            .map_err(candle_err)?;
        let out = out
            .reshape((b, heads, l, head_dim))
            .map_err(candle_err)?
            .permute((0, 2, 1, 3))
            .map_err(candle_err)?
            .reshape((b, l, heads * head_dim))
            .map_err(candle_err)?;
        linear(&out, &layer.wo, None)
    }
}

/// `x @ w.T (+ b)` for any leading shape.
///
/// candle's `matmul` requires both operands to have the same rank, so a 3D activation
/// cannot be multiplied by a 2D weight directly; flatten the leading dimensions instead.
pub fn linear(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Tensor, LayaError> {
    let rank = x.rank();
    let in_features = x.dim(rank - 1).map_err(candle_err)?;
    let out_features = w.dim(0).map_err(candle_err)?;
    let lead: Vec<usize> = x.dims()[..rank - 1].to_vec();
    let rows: usize = lead.iter().product();
    let flat = x.reshape((rows, in_features)).map_err(candle_err)?;
    let mut y = flat
        .matmul(&w.t().map_err(candle_err)?)
        .map_err(candle_err)?;

    if let Some(b) = bias {
        y = y.broadcast_add(b).map_err(candle_err)?;
    }
    let mut shape = lead;
    shape.push(out_features);
    y.reshape(shape.as_slice()).map_err(candle_err)
}

/// LayerNorm without bias, as `norm_bias: false` implies.
pub fn layer_norm_no_bias(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor, LayaError> {
    // candle has no fused CPU layer-norm kernel, so this is a chain of elementwise ops.
    // The one thing worth skipping is the dtype copy when the input is already f32 (the
    // common case -- every weight is upcast once at load).
    let x = if x.dtype() == DType::F32 {
        x.clone()
    } else {
        x.to_dtype(DType::F32).map_err(candle_err)?
    };
    let mean = x.mean_keepdim(candle_core::D::Minus1).map_err(candle_err)?;
    let centered = x.broadcast_sub(&mean).map_err(candle_err)?;
    let var = centered
        .sqr()
        .map_err(candle_err)?
        .mean_keepdim(candle_core::D::Minus1)
        .map_err(candle_err)?;
    let denom = var
        .affine(1.0, eps)
        .map_err(candle_err)?
        .sqrt()
        .map_err(candle_err)?;
    let normed = centered.broadcast_div(&denom).map_err(candle_err)?;
    normed.broadcast_mul(weight).map_err(candle_err)
}

/// GELU with the erf formulation, matching `ACT2FN["gelu"]`.
pub fn gelu_erf(x: &Tensor) -> Result<Tensor, LayaError> {
    let inner = x
        .affine(1.0 / std::f64::consts::SQRT_2, 0.0)
        .map_err(candle_err)?;
    let erf = inner.erf().map_err(candle_err)?;
    // 0.5 * x * (1 + erf(x/sqrt(2)))
    x.mul(&erf.affine(0.5, 0.5).map_err(candle_err)?)
        .map_err(candle_err)
}

/// GeGLU: `Wo(act(first_half(Wi(x))) * second_half(Wi(x)))`.
pub fn geglu(x: &Tensor, wi: &Tensor, wo: &Tensor) -> Result<Tensor, LayaError> {
    let fused = linear(x, wi, None)?;
    let inter = fused.dims().last().copied().unwrap_or(0) / 2;
    let input = fused
        .narrow(candle_core::D::Minus1, 0, inter)
        .map_err(candle_err)?;
    let gate = fused
        .narrow(candle_core::D::Minus1, inter, inter)
        .map_err(candle_err)?;
    let hidden = gelu_erf(&input)?.mul(&gate).map_err(candle_err)?;
    linear(&hidden, wo, None)
}

/// RoPE cosine/sine tables, (L, head_dim), with the frequency vector duplicated
/// across both halves exactly as the HF reference does.
fn rope_tables(
    seq_len: usize,
    head_dim: usize,
    theta: f64,
    device: &Device,
) -> Result<(Tensor, Tensor), LayaError> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(i as f64 * 2.0 / head_dim as f64) as f32)
        .collect();
    let mut freqs = Vec::with_capacity(seq_len * half);
    for pos in 0..seq_len {
        for f in &inv_freq {
            freqs.push(pos as f32 * f);
        }
    }
    let freqs = Tensor::from_slice(&freqs, (seq_len, half), device).map_err(candle_err)?;
    let emb = Tensor::cat(&[&freqs, &freqs], candle_core::D::Minus1).map_err(candle_err)?;
    Ok((
        emb.cos().map_err(candle_err)?,
        emb.sin().map_err(candle_err)?,
    ))
}

/// `x * cos + rotate_half(x) * sin`.
///
/// Accepts either `(B, heads, L, head_dim)` or the folded `(B*heads, L, head_dim)`; the
/// tables are reshaped to broadcast over whichever leading dimensions are present.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, LayaError> {
    let dims = x.dims().to_vec();
    let (l, head_dim) = (dims[dims.len() - 2], dims[dims.len() - 1]);
    let lead: Vec<usize> = dims[..dims.len() - 2].to_vec();
    let mut shape = lead;
    shape.push(l);
    shape.push(head_dim);
    let cos = cos
        .broadcast_as(shape.as_slice())
        .map_err(candle_err)?
        .to_dtype(DType::F32)
        .map_err(candle_err)?;
    let sin = sin
        .broadcast_as(shape.as_slice())
        .map_err(candle_err)?
        .to_dtype(DType::F32)
        .map_err(candle_err)?;
    let rotated = rotate_half(x)?;
    let a = x.broadcast_mul(&cos).map_err(candle_err)?;
    let b = rotated.broadcast_mul(&sin).map_err(candle_err)?;
    a.add(&b).map_err(candle_err)
}

fn rotate_half(x: &Tensor) -> Result<Tensor, LayaError> {
    let head_dim = x.dims().last().copied().unwrap_or(0);
    let x1 = x
        .narrow(candle_core::D::Minus1, 0, head_dim / 2)
        .map_err(candle_err)?;
    let x2 = x
        .narrow(candle_core::D::Minus1, head_dim / 2, head_dim / 2)
        .map_err(candle_err)?;
    Tensor::cat(
        &[&x2.neg().map_err(candle_err)?, &x1],
        candle_core::D::Minus1,
    )
    .map_err(candle_err)
}

/// Additive (L, L) sliding-window mask: keep `|i - j| <= window`.
fn sliding_mask(seq_len: usize, window: usize, device: &Device) -> Result<Tensor, LayaError> {
    let mut mask = vec![0f32; seq_len * seq_len];
    for (i, row) in mask.chunks_mut(seq_len).enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            if (i as i32 - j as i32).abs() > window as i32 {
                *cell = MASK_VALUE;
            }
        }
    }
    Tensor::from_slice(&mask, (1, 1, seq_len, seq_len), device).map_err(candle_err)
}

/// Map a candle error into ours, keeping the message.
pub fn candle_err(e: candle_core::Error) -> LayaError {
    LayaError::Tensor(e.to_string())
}

/// Re-export so callers can build tensors without depending on candle directly.
pub use candle_core::D;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn config_reads_rope_theta_from_the_top_level_keys() {
        let cfg = Config::from_value(&json!({
            "vocab_size": 100, "hidden_size": 64, "num_hidden_layers": 4,
            "num_attention_heads": 4, "intermediate_size": 128,
            "local_attention": 128, "global_attn_every_n_layers": 3,
            "global_rope_theta": 160000.0,
            "local_rope_theta": 10000.0
        }))
        .unwrap();
        assert_eq!(cfg.global_rope_theta, 160000.0);
        assert_eq!(cfg.local_rope_theta, 10000.0);
        assert_eq!(cfg.sliding_window(), 64);
        assert_eq!(cfg.head_dim(), 16);
        // layer_types absent -> derived from global_attn_every_n_layers
        assert_eq!(
            cfg.layer_types,
            vec![
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention"
            ]
        );
    }

    #[test]
    fn rope_parameters_is_ignored_like_transformers_does() {
        // The shipped multilingual checkpoint carries rope_parameters claiming 160000 for
        // sliding layers, but transformers 4.57 uses local_rope_theta (default 10000).
        // Pinned here so a future "helpful" change cannot silently break parity.
        let cfg = Config::from_value(&json!({
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
            "rope_parameters": {
                "full_attention": {"rope_theta": 160000.0},
                "sliding_attention": {"rope_theta": 160000.0}
            }
        }))
        .unwrap();
        assert_eq!(cfg.global_rope_theta, 160000.0);
        assert_eq!(cfg.local_rope_theta, 10000.0);
    }

    #[test]
    fn config_prefers_explicit_layer_types() {
        let cfg = Config::from_value(&json!({
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
            "layer_types": ["sliding_attention", "full_attention"]
        }))
        .unwrap();
        assert_eq!(cfg.layer_types, vec!["sliding_attention", "full_attention"]);
    }

    #[test]
    fn rope_tables_duplicate_the_frequency_vector() {
        let (cos, sin) = rope_tables(3, 8, 10000.0, &Device::Cpu).unwrap();
        assert_eq!(cos.dims(), &[3, 8]);
        assert_eq!(sin.dims(), &[3, 8]);
        // position 0 is all ones/zeros
        let c = cos.to_vec2::<f32>().unwrap();
        let sn = sin.to_vec2::<f32>().unwrap();
        assert!(c[0].iter().all(|&v| (v - 1.0).abs() < 1e-6), "{:?}", c[0]);
        assert!(sn[0].iter().all(|&v| v.abs() < 1e-6), "{:?}", sn[0]);
        // the two halves are identical (HF cats freqs with itself)
        assert_eq!(c[1][..4], c[1][4..]);
    }

    #[test]
    fn sliding_window_is_inclusive_at_64() {
        let m = sliding_mask(200, 64, &Device::Cpu).unwrap();
        let v = m.reshape((200, 200)).unwrap().to_vec2::<f32>().unwrap();
        // |i-j| <= 64 kept, beyond masked
        assert_eq!(v[0][64], 0.0);
        assert_eq!(v[0][65], MASK_VALUE);
        assert_eq!(v[100][36], 0.0);
        assert_eq!(v[100][35], MASK_VALUE);
    }

    #[test]
    fn gelu_erf_matches_the_reference() {
        let x = Tensor::new(&[-2.0f32, -0.5, 0.0, 0.5, 2.0][..], &Device::Cpu).unwrap();
        let got = gelu_erf(&x).unwrap().to_vec1::<f32>().unwrap();
        // 0.5 * x * (1 + erf(x / sqrt(2)))
        let want: Vec<f32> = [-2.0, -0.5, 0.0, 0.5, 2.0]
            .iter()
            .map(|&v| 0.5 * v * (1.0 + libm_erf(v / std::f64::consts::SQRT_2 as f32)))
            .collect();
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "{} vs {}", g, w);
        }
    }

    fn libm_erf(x: f32) -> f32 {
        // Abramowitz & Stegun 7.1.26, good to ~1.5e-7 -- enough for a unit test.
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let y = 1.0
            - (((((1.0614054 * t - 1.453152) * t) + 1.4214137) * t - 0.2844967) * t + 0.2548296)
                * t
                * (-x * x).exp();
        if x < 0.0 {
            -y
        } else {
            y
        }
    }
}
