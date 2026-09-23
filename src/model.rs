//! The decision model: ModernBERT encoder plus the typed decision head.
//!
//! Rust port of `DecisionModel` in `laya/common.py`. Two details that are easy to miss:
//!
//! * the head's feed-forward uses **ReLU**, not GELU -- `nn.TransformerEncoderLayer`'s
//!   default activation is `F.relu` and `laya` never overrides it;
//! * the head's attention is plain bidirectional attention with only a key-padding mask
//!   (no RoPE, no sliding window), and `norm_first=True` puts the norms before each block.

use candle_core::{DType, Tensor};

use crate::modernbert::{gelu_erf, linear, ModernBert};
use crate::weights::Weights;
use crate::LayaError;

/// One `nn.TransformerEncoderLayer(d, nhead, 4*d, batch_first=True, norm_first=True)`.
struct HeadLayer {
    norm1_w: Tensor,
    norm1_b: Tensor,
    norm2_w: Tensor,
    norm2_b: Tensor,
    in_proj_w: Tensor,
    in_proj_b: Tensor,
    out_proj_w: Tensor,
    out_proj_b: Tensor,
    linear1_w: Tensor,
    linear1_b: Tensor,
    linear2_w: Tensor,
    linear2_b: Tensor,
}

/// `nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))`.
struct Scorer {
    norm_w: Tensor,
    norm_b: Tensor,
    l1_w: Tensor,
    l1_b: Tensor,
    l2_w: Tensor,
    l2_b: Tensor,
}

/// `nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))`.
struct ActHead {
    l1_w: Tensor,
    l1_b: Tensor,
    l2_w: Tensor,
    l2_b: Tensor,
}

/// The full model: encoder + decision head.
pub struct DecisionModel {
    encoder: ModernBert,
    head: Vec<HeadLayer>,
    type_emb: Tensor,
    scorer: Scorer,
    act_head: ActHead,
    temperature: Tensor,
    hidden: usize,
    head_layers: usize,
}

/// Outputs of one forward pass.
pub struct ForwardOutput {
    /// (N, kmax) raw option logits, masked entries set to -1e4.
    pub logits: Tensor,
    /// (N, n_act) action-head logits.
    pub act_logits: Tensor,
}

impl DecisionModel {
    /// The full list of `(weight key, expected shape)` this architecture needs.
    ///
    /// This is the single source of truth for expected shapes: [`Weights::verify_compatibility`]
    /// checks a checkpoint against it before anything is loaded, so a mismatched checkpoint
    /// fails with a clear message instead of a candle `matmul` error deep inside the forward
    /// pass. Mirrors what `Agent._verify_compatibility` does in Python.
    pub fn weight_plan(
        encoder_cfg: &serde_json::Value,
        agent_cfg: &serde_json::Value,
    ) -> Result<Vec<(String, Vec<usize>)>, LayaError> {
        let enc = crate::modernbert::Config::from_value(encoder_cfg)?;
        let hidden = enc.hidden_size;
        let inter = enc.intermediate_size;
        let head_layers = agent_cfg
            .get("head_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        // Python: len(cfg.get("act_costs", {})) + 1 -- absent *and* empty both give 1.
        let n_act = agent_cfg
            .get("act_costs")
            .and_then(|v| v.as_object())
            .map(|m| m.len() + 1)
            .unwrap_or(1);

        let mut plan: Vec<(String, Vec<usize>)> = Vec::new();
        let mut add = |key: String, shape: Vec<usize>| plan.push((key, shape));

        // ---- encoder
        add(
            "encoder.embeddings.tok_embeddings.weight".into(),
            vec![enc.vocab_size, hidden],
        );
        add("encoder.embeddings.norm.weight".into(), vec![hidden]);
        for i in 0..enc.num_hidden_layers {
            let p = format!("encoder.layers.{}.", i);
            // layer 0 has no attn_norm: the embedding output is already normed
            if i > 0 {
                add(format!("{}attn_norm.weight", p), vec![hidden]);
            }
            add(format!("{}attn.Wqkv.weight", p), vec![3 * hidden, hidden]);
            add(format!("{}attn.Wo.weight", p), vec![hidden, hidden]);
            add(format!("{}mlp_norm.weight", p), vec![hidden]);
            add(format!("{}mlp.Wi.weight", p), vec![2 * inter, hidden]);
            add(format!("{}mlp.Wo.weight", p), vec![hidden, inter]);
        }
        // final_norm comes after the layers, matching Python's named_parameters() order so
        // the "first 5 mismatches" in an error message list the same tensors.
        add("encoder.final_norm.weight".into(), vec![hidden]);

        // ---- decision head (PyTorch TransformerEncoderLayer, norm_first)
        for i in 0..head_layers {
            let p = format!("head.layers.{}.", i);
            add(format!("{}norm1.weight", p), vec![hidden]);
            add(format!("{}norm1.bias", p), vec![hidden]);
            add(format!("{}norm2.weight", p), vec![hidden]);
            add(format!("{}norm2.bias", p), vec![hidden]);
            add(
                format!("{}self_attn.in_proj_weight", p),
                vec![3 * hidden, hidden],
            );
            add(format!("{}self_attn.in_proj_bias", p), vec![3 * hidden]);
            add(
                format!("{}self_attn.out_proj.weight", p),
                vec![hidden, hidden],
            );
            add(format!("{}self_attn.out_proj.bias", p), vec![hidden]);
            add(format!("{}linear1.weight", p), vec![4 * hidden, hidden]);
            add(format!("{}linear1.bias", p), vec![4 * hidden]);
            add(format!("{}linear2.weight", p), vec![hidden, 4 * hidden]);
            add(format!("{}linear2.bias", p), vec![hidden]);
        }

        // ---- scorer and action head
        add("scorer.0.weight".into(), vec![hidden]);
        add("scorer.0.bias".into(), vec![hidden]);
        add("scorer.1.weight".into(), vec![hidden, hidden]);
        add("scorer.1.bias".into(), vec![hidden]);
        add("scorer.3.weight".into(), vec![1, hidden]);
        add("scorer.3.bias".into(), vec![1]);
        add("act_head.0.weight".into(), vec![256, hidden + 4]);
        add("act_head.0.bias".into(), vec![256]);
        add("act_head.2.weight".into(), vec![n_act, 256]);
        add("act_head.2.bias".into(), vec![n_act]);
        add("type_emb.weight".into(), vec![3, hidden]);
        add("temperature".into(), vec![3]);

        Ok(plan)
    }

    pub fn load(
        encoder_cfg: &serde_json::Value,
        agent_cfg: &serde_json::Value,
        weights: &mut Weights,
    ) -> Result<Self, LayaError> {
        let encoder =
            ModernBert::load(crate::modernbert::Config::from_value(encoder_cfg)?, weights)?;
        let hidden = encoder.config().hidden_size;
        let head_layers = agent_cfg
            .get("head_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;

        let mut head = Vec::with_capacity(head_layers);
        for i in 0..head_layers {
            let p = format!("head.layers.{}.", i);
            head.push(HeadLayer {
                norm1_w: weights.take(&format!("{}norm1.weight", p))?,
                norm1_b: weights.take(&format!("{}norm1.bias", p))?,
                norm2_w: weights.take(&format!("{}norm2.weight", p))?,
                norm2_b: weights.take(&format!("{}norm2.bias", p))?,
                in_proj_w: weights.take(&format!("{}self_attn.in_proj_weight", p))?,
                in_proj_b: weights.take(&format!("{}self_attn.in_proj_bias", p))?,
                out_proj_w: weights.take(&format!("{}self_attn.out_proj.weight", p))?,
                out_proj_b: weights.take(&format!("{}self_attn.out_proj.bias", p))?,
                linear1_w: weights.take(&format!("{}linear1.weight", p))?,
                linear1_b: weights.take(&format!("{}linear1.bias", p))?,
                linear2_w: weights.take(&format!("{}linear2.weight", p))?,
                linear2_b: weights.take(&format!("{}linear2.bias", p))?,
            });
        }

        let scorer = Scorer {
            norm_w: weights.take("scorer.0.weight")?,
            norm_b: weights.take("scorer.0.bias")?,
            l1_w: weights.take("scorer.1.weight")?,
            l1_b: weights.take("scorer.1.bias")?,
            l2_w: weights.take("scorer.3.weight")?,
            l2_b: weights.take("scorer.3.bias")?,
        };
        let act_head = ActHead {
            l1_w: weights.take("act_head.0.weight")?,
            l1_b: weights.take("act_head.0.bias")?,
            l2_w: weights.take("act_head.2.weight")?,
            l2_b: weights.take("act_head.2.bias")?,
        };
        let type_emb = weights.take("type_emb.weight")?;
        let temperature = weights.take("temperature")?;

        Ok(DecisionModel {
            encoder,
            head,
            type_emb,
            scorer,
            act_head,
            temperature,
            hidden,
            head_layers,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// The encoder's final hidden states, for diagnostics and parity harnesses.
    pub fn encoder_forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor, LayaError> {
        self.encoder.forward(input_ids, attention_mask)
    }

    pub fn head_layers(&self) -> usize {
        self.head_layers
    }

    /// The temperature buffer, as loaded from the checkpoint.
    pub fn temperature(&self) -> &Tensor {
        &self.temperature
    }

    /// One forward pass over a collated batch.
    ///
    /// * `input_ids` (N, L) i64
    /// * `attention_mask` (N, L) f32
    /// * `marker_pos` (N, kmax) i64 -- option marker positions
    /// * `marker_mask` (N, kmax) bool -- which markers are real
    /// * `qtype` (N,) i64 -- question type index
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        marker_pos: &Tensor,
        marker_mask: &Tensor,
        qtype: &Tensor,
    ) -> Result<ForwardOutput, LayaError> {
        let (n, l) = input_ids.dims2().map_err(crate::modernbert::candle_err)?;
        let kmax = marker_pos.dims().last().copied().unwrap_or(0);

        let mut h = self.encoder.forward(input_ids, attention_mask)?;
        // h = h + type_emb(qtype)[:, None, :]
        let emb = self
            .type_emb
            .index_select(
                &qtype.flatten_all().map_err(crate::modernbert::candle_err)?,
                0,
            )
            .map_err(crate::modernbert::candle_err)?
            .reshape((n, 1, self.hidden))
            .map_err(crate::modernbert::candle_err)?;
        h = h
            .broadcast_add(&emb)
            .map_err(crate::modernbert::candle_err)?;

        if !self.head.is_empty() {
            // src_key_padding_mask: additive, 0 for real tokens and -1e9 for padding
            let pad = attention_mask
                .eq(0f32)
                .map_err(crate::modernbert::candle_err)?
                .to_dtype(DType::F32)
                .map_err(crate::modernbert::candle_err)?;
            let pad_mask = pad
                .reshape((n, 1, 1, l))
                .map_err(crate::modernbert::candle_err)?
                .affine(-1e9, 0.0)
                .map_err(crate::modernbert::candle_err)?;
            for layer in &self.head {
                h = self.head_layer(layer, &h, &pad_mask, n, l)?;
            }
        }

        // gather the marker positions
        let idx = marker_pos
            .clamp(0i64, i64::MAX)
            .map_err(crate::modernbert::candle_err)?
            .reshape((n, kmax, 1))
            .map_err(crate::modernbert::candle_err)?
            .broadcast_as((n, kmax, self.hidden))
            .map_err(crate::modernbert::candle_err)?
            .contiguous()
            .map_err(crate::modernbert::candle_err)?;
        let m = h
            .contiguous()
            .map_err(crate::modernbert::candle_err)?
            .gather(&idx, 1)
            .map_err(crate::modernbert::candle_err)?;

        // logits = scorer(m).squeeze(-1), masked entries -> -1e4
        let mut logits = self.scorer_forward(&m)?;
        let mut squeezed: Vec<usize> = logits.dims().to_vec();
        squeezed.pop();
        logits = logits
            .reshape(squeezed)
            .map_err(crate::modernbert::candle_err)?;
        // Python's masked_fill(~marker_mask, -1e4): *set* the padded entries, do not offset them.
        // drop is 1 where the marker is padding, so logits*(1-drop) - 1e4*drop does exactly that.
        let drop = marker_mask
            .affine(-1.0, 1.0)
            .map_err(crate::modernbert::candle_err)?;
        let keep = marker_mask.clone();
        logits = logits
            .broadcast_mul(&keep)
            .map_err(crate::modernbert::candle_err)?
            .broadcast_add(
                &drop
                    .affine(-1e4, 0.0)
                    .map_err(crate::modernbert::candle_err)?,
            )
            .map_err(crate::modernbert::candle_err)?;

        // p = softmax(logits) over the whole kmax axis (masked entries are ~0)
        let p = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)
            .map_err(crate::modernbert::candle_err)?;
        let k = marker_mask
            .sum(candle_core::D::Minus1)
            .map_err(crate::modernbert::candle_err)?
            .clamp(2f64, f64::INFINITY)
            .map_err(crate::modernbert::candle_err)?;
        let log_p = p
            .clamp(1e-9f64, f64::INFINITY)
            .map_err(crate::modernbert::candle_err)?
            .log()
            .map_err(crate::modernbert::candle_err)?;
        let ent = p
            .mul(&log_p)
            .map_err(crate::modernbert::candle_err)?
            .neg()
            .map_err(crate::modernbert::candle_err)?
            .sum(candle_core::D::Minus1)
            .map_err(crate::modernbert::candle_err)?
            .div(&k.log().map_err(crate::modernbert::candle_err)?)
            .map_err(crate::modernbert::candle_err)?;
        // top2 over the whole axis
        let (sorted, _) = p
            .sort_last_dim(false)
            .map_err(crate::modernbert::candle_err)?;
        let top1 = sorted
            .narrow(candle_core::D::Minus1, 0, 1)
            .map_err(crate::modernbert::candle_err)?;
        let top2 = sorted
            .narrow(candle_core::D::Minus1, 1, 1)
            .map_err(crate::modernbert::candle_err)?;
        let margin = top1
            .broadcast_sub(&top2)
            .map_err(crate::modernbert::candle_err)?;
        let k_scaled = k
            .affine(1.0 / 255.0, 0.0)
            .map_err(crate::modernbert::candle_err)?;
        let feats = Tensor::cat(
            &[
                &top1
                    .reshape((n, 1))
                    .map_err(crate::modernbert::candle_err)?,
                &margin
                    .reshape((n, 1))
                    .map_err(crate::modernbert::candle_err)?,
                &ent.reshape((n, 1)).map_err(crate::modernbert::candle_err)?,
                &k_scaled
                    .reshape((n, 1))
                    .map_err(crate::modernbert::candle_err)?,
            ],
            candle_core::D::Minus1,
        )
        .map_err(crate::modernbert::candle_err)?;

        // pooled = h[:, 0]
        let pooled = h
            .narrow(1, 0, 1)
            .map_err(crate::modernbert::candle_err)?
            .reshape((n, self.hidden))
            .map_err(crate::modernbert::candle_err)?;
        let act_in = Tensor::cat(&[&pooled, &feats], candle_core::D::Minus1)
            .map_err(crate::modernbert::candle_err)?;
        let act_logits = self.act_head_forward(&act_in)?;

        Ok(ForwardOutput { logits, act_logits })
    }

    fn head_layer(
        &self,
        layer: &HeadLayer,
        x: &Tensor,
        pad_mask: &Tensor,
        n: usize,
        l: usize,
    ) -> Result<Tensor, LayaError> {
        let heads = (self.hidden / 64).max(1);
        let head_dim = self.hidden / heads;
        let normed = layer_norm(x, &layer.norm1_w, &layer.norm1_b, 1e-5)?;
        let attn = self.multi_head_attention(
            &normed,
            &layer.in_proj_w,
            &layer.in_proj_b,
            &layer.out_proj_w,
            &layer.out_proj_b,
            pad_mask,
            n,
            l,
            heads,
            head_dim,
        )?;
        let x = x.add(&attn).map_err(crate::modernbert::candle_err)?;
        let normed = layer_norm(&x, &layer.norm2_w, &layer.norm2_b, 1e-5)?;
        // ReLU: nn.TransformerEncoderLayer's default activation
        let ffn = linear(
            &linear(&normed, &layer.linear1_w, Some(&layer.linear1_b))?
                .relu()
                .map_err(crate::modernbert::candle_err)?,
            &layer.linear2_w,
            Some(&layer.linear2_b),
        )?;
        x.add(&ffn).map_err(crate::modernbert::candle_err)
    }

    #[allow(clippy::too_many_arguments)]
    fn multi_head_attention(
        &self,
        x: &Tensor,
        in_w: &Tensor,
        in_b: &Tensor,
        out_w: &Tensor,
        out_b: &Tensor,
        pad_mask: &Tensor,
        n: usize,
        l: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Tensor, LayaError> {
        let qkv = linear(x, in_w, Some(in_b))?;
        let split = |offset: usize| -> Result<Tensor, LayaError> {
            qkv.narrow(
                candle_core::D::Minus1,
                offset * heads * head_dim,
                heads * head_dim,
            )
            .map_err(crate::modernbert::candle_err)?
            .reshape((n, l, heads, head_dim))
            .map_err(crate::modernbert::candle_err)?
            .permute((0, 2, 1, 3))
            .map_err(crate::modernbert::candle_err)?
            .contiguous()
            .map_err(crate::modernbert::candle_err)
        };
        let q = split(0)?;
        let k = split(1)?;
        let v = split(2)?;
        let scale = (head_dim as f64).sqrt();
        let scores = q
            .matmul(&k.t().map_err(crate::modernbert::candle_err)?)
            .map_err(crate::modernbert::candle_err)?
            .affine(1.0 / scale, 0.0)
            .map_err(crate::modernbert::candle_err)?
            .broadcast_add(pad_mask)
            .map_err(crate::modernbert::candle_err)?;
        let attn =
            candle_nn::ops::softmax_last_dim(&scores).map_err(crate::modernbert::candle_err)?;
        let out = attn
            .matmul(&v)
            .map_err(crate::modernbert::candle_err)?
            .permute((0, 2, 1, 3))
            .map_err(crate::modernbert::candle_err)?
            .reshape((n, l, heads * head_dim))
            .map_err(crate::modernbert::candle_err)?;
        linear(&out, out_w, Some(out_b))
    }

    fn scorer_forward(&self, m: &Tensor) -> Result<Tensor, LayaError> {
        let x = layer_norm(m, &self.scorer.norm_w, &self.scorer.norm_b, 1e-5)?;
        let x = linear(&x, &self.scorer.l1_w, Some(&self.scorer.l1_b))?;
        let x = gelu_erf(&x)?;
        linear(&x, &self.scorer.l2_w, Some(&self.scorer.l2_b))
    }

    fn act_head_forward(&self, x: &Tensor) -> Result<Tensor, LayaError> {
        let x = linear(x, &self.act_head.l1_w, Some(&self.act_head.l1_b))?;
        let x = gelu_erf(&x)?;
        linear(&x, &self.act_head.l2_w, Some(&self.act_head.l2_b))
    }
}

/// LayerNorm with bias, as `nn.LayerNorm` defaults to.
pub fn layer_norm(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> Result<Tensor, LayaError> {
    let x = x
        .to_dtype(DType::F32)
        .map_err(crate::modernbert::candle_err)?;
    let mean = x
        .mean_keepdim(candle_core::D::Minus1)
        .map_err(crate::modernbert::candle_err)?;
    let centered = x
        .broadcast_sub(&mean)
        .map_err(crate::modernbert::candle_err)?;
    let var = centered
        .sqr()
        .map_err(crate::modernbert::candle_err)?
        .mean_keepdim(candle_core::D::Minus1)
        .map_err(crate::modernbert::candle_err)?;
    let denom = var
        .affine(1.0, eps)
        .map_err(crate::modernbert::candle_err)?
        .sqrt()
        .map_err(crate::modernbert::candle_err)?;
    centered
        .broadcast_div(&denom)
        .map_err(crate::modernbert::candle_err)?
        .broadcast_mul(weight)
        .map_err(crate::modernbert::candle_err)?
        .broadcast_add(bias)
        .map_err(crate::modernbert::candle_err)
}

#[cfg(test)]
mod plan_tests {
    use super::*;
    use serde_json::json;
    use serde_json::Value;

    fn english_like() -> Value {
        json!({
            "vocab_size": 1000, "hidden_size": 64, "num_hidden_layers": 4,
            "num_attention_heads": 4, "intermediate_size": 128,
            "local_attention": 128, "global_attn_every_n_layers": 3
        })
    }

    fn agent_cfg() -> Value {
        json!({"encoder": "some/repo", "head_layers": 2, "act_costs": {"escalate": 0.5}})
    }

    fn shape_of(plan: &[(String, Vec<usize>)], key: &str) -> Vec<usize> {
        plan.iter()
            .find(|(k, _)| k == key)
            .map(|(_, s)| s.clone())
            .unwrap_or_else(|| panic!("plan has no {}", key))
    }

    #[test]
    fn plan_covers_every_tensor_the_checkpoint_has() {
        let plan = DecisionModel::weight_plan(&english_like(), &agent_cfg()).unwrap();
        let keys: Vec<&str> = plan.iter().map(|(k, _)| k.as_str()).collect();
        // 4 layers x 6 tensors, minus layer 0's attn_norm, plus embeddings/final_norm
        let encoder = keys.iter().filter(|k| k.starts_with("encoder.")).count();
        assert_eq!(encoder, 3 + 4 * 6 - 1, "encoder tensor count");
        // 2 head layers x 12 tensors
        assert_eq!(
            keys.iter()
                .filter(|k| k.starts_with("head.layers."))
                .count(),
            24
        );
        // scorer (6) + act_head (4) + type_emb + temperature
        assert_eq!(keys.iter().filter(|k| k.starts_with("scorer.")).count(), 6);
        assert_eq!(
            keys.iter().filter(|k| k.starts_with("act_head.")).count(),
            4
        );
        assert!(keys.contains(&"type_emb.weight"));
        assert!(keys.contains(&"temperature"));
    }

    #[test]
    fn plan_shapes_match_the_architecture() {
        let plan = DecisionModel::weight_plan(&english_like(), &agent_cfg()).unwrap();
        assert_eq!(
            shape_of(&plan, "encoder.embeddings.tok_embeddings.weight"),
            vec![1000, 64]
        );
        assert_eq!(
            shape_of(&plan, "encoder.layers.0.attn.Wqkv.weight"),
            vec![192, 64]
        );
        assert_eq!(
            shape_of(&plan, "encoder.layers.0.mlp.Wi.weight"),
            vec![256, 64]
        );
        assert_eq!(
            shape_of(&plan, "encoder.layers.0.mlp.Wo.weight"),
            vec![64, 128]
        );
        // layer 0 has no attn_norm, layers 1.. do
        assert!(!plan
            .iter()
            .any(|(k, _)| k == "encoder.layers.0.attn_norm.weight"));
        assert_eq!(
            shape_of(&plan, "encoder.layers.1.attn_norm.weight"),
            vec![64]
        );
        // the head's FFN is 4x wide, and act_head takes hidden + 4 features
        assert_eq!(
            shape_of(&plan, "head.layers.0.linear1.weight"),
            vec![256, 64]
        );
        assert_eq!(
            shape_of(&plan, "head.layers.0.linear2.weight"),
            vec![64, 256]
        );
        assert_eq!(shape_of(&plan, "act_head.0.weight"), vec![256, 68]);
        assert_eq!(shape_of(&plan, "act_head.2.weight"), vec![2, 256]);
        assert_eq!(shape_of(&plan, "scorer.3.weight"), vec![1, 64]);
    }

    #[test]
    fn plan_tracks_head_layers_and_act_costs() {
        let cfg = json!({"encoder": "r", "head_layers": 0});
        let plan = DecisionModel::weight_plan(&english_like(), &cfg).unwrap();
        assert!(!plan.iter().any(|(k, _)| k.starts_with("head.layers.")));
        // Python: len(cfg.get("act_costs", {})) + 1 -- absent gives 1, not 2
        assert_eq!(shape_of(&plan, "act_head.2.weight"), vec![1, 256]);

        let cfg = json!({"encoder": "r", "head_layers": 0, "act_costs": {}});
        let plan = DecisionModel::weight_plan(&english_like(), &cfg).unwrap();
        assert_eq!(shape_of(&plan, "act_head.2.weight"), vec![1, 256]);

        let cfg = json!({"encoder": "r", "head_layers": 3,
                         "act_costs": {"a": 1.0, "b": 2.0, "c": 3.0}});
        let plan = DecisionModel::weight_plan(&english_like(), &cfg).unwrap();
        assert_eq!(
            plan.iter()
                .filter(|(k, _)| k.starts_with("head.layers."))
                .count(),
            36
        );
        assert_eq!(shape_of(&plan, "act_head.2.weight"), vec![4, 256]);
    }

    /// The plan must describe the real checkpoints, not just a synthetic config.
    /// Skipped when the weights are not downloaded locally.
    #[test]
    fn plan_matches_the_real_checkpoints() {
        let home = std::env::var("HOME").unwrap_or_default();
        for (name, dir) in [
            ("english", format!("{}/laya_models/laya", home)),
            (
                "multilingual",
                format!("{}/laya_models/laya-multilingual", home),
            ),
        ] {
            let enc_path = format!("{}/encoder/config.json", dir);
            let agent_path = format!("{}/rl_agent_config.json", dir);
            let weights_path = format!("{}/model.safetensors", dir);
            if !std::path::Path::new(&weights_path).exists() {
                eprintln!("skipping {}: no local checkpoint", name);
                continue;
            }
            let enc = crate::weights::read_json(&enc_path).unwrap();
            let agent = crate::weights::read_json(&agent_path).unwrap();
            let plan = DecisionModel::weight_plan(&enc, &agent).unwrap();
            let weights =
                crate::weights::Weights::load(&weights_path, &candle_core::Device::Cpu).unwrap();
            weights
                .verify_compatibility(&plan, &dir)
                .unwrap_or_else(|e| panic!("{}: {}", name, e));
            // and the plan must account for every tensor in the file
            assert_eq!(
                plan.len(),
                weights.len(),
                "{}: plan vs checkpoint tensor count",
                name
            );
        }
    }
}
