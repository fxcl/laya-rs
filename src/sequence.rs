//! Prompt construction: turning a state plus a question into token ids.
//!
//! Rust port of `build_sequence` / `collate_items` from `laya/common.py`. The layout is
//!
//! ```text
//! [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
//! ```
//!
//! with `head_max_len` capping the question head and `max_len` capping the whole sequence.

use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer as HfTokenizer;

use crate::criteria::{render_options, Question};
use crate::pyjson::dumps;
use crate::LayaError;

/// A loaded fast tokenizer plus the special-token ids the prompt layout needs.
pub struct Tokenizer {
    inner: HfTokenizer,
    pub cls_id: u32,
    pub mask_id: u32,
    pub sep_id: u32,
    pub pad_id: u32,
}

impl Tokenizer {
    /// Load `tokenizer.json` and read the special tokens from `tokenizer_config.json`.
    pub fn load(tokenizer_dir: &str) -> Result<Self, LayaError> {
        let json_path = format!("{}/tokenizer.json", tokenizer_dir);
        let inner = HfTokenizer::from_file(&json_path)
            .map_err(|e| LayaError::ConfigUnreadable(json_path.clone(), e.to_string()))?;
        let cfg = crate::weights::read_json(&format!("{}/tokenizer_config.json", tokenizer_dir))
            .or_else(|_| crate::weights::read_json(&json_path))?;
        // resolve every special token before moving `inner` into the struct
        let id_of = |name: &str, tok: &HfTokenizer| -> Result<u32, LayaError> {
            let token = cfg
                .get(format!("{}_token", name))
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    LayaError::ConfigUnreadable(
                        tokenizer_dir.to_string(),
                        format!("tokenizer_config.json has no {}_token", name),
                    )
                })?
                .to_string();
            tok.token_to_id(&token).ok_or_else(|| {
                LayaError::ConfigUnreadable(
                    tokenizer_dir.to_string(),
                    format!("tokenizer has no token {:?}", token),
                )
            })
        };
        let cls_id = id_of("cls", &inner)?;
        let mask_id = id_of("mask", &inner)?;
        let sep_id = id_of("sep", &inner)?;
        let pad_id = id_of("pad", &inner)?;
        Ok(Tokenizer {
            inner,
            cls_id,
            mask_id,
            sep_id,
            pad_id,
        })
    }

    /// Encode without adding special tokens, as `add_special_tokens=False` does.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, LayaError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| LayaError::Tensor(format!("tokenizer failed on {:?}: {}", text, e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn mask_token(&self) -> String {
        self.inner
            .id_to_token(self.mask_id)
            .unwrap_or_else(|| "[MASK]".to_string())
    }
}

/// `serialize_state`: strings pass through, everything else becomes JSON.
pub fn serialize_state(state: &serde_json::Value) -> String {
    match state {
        serde_json::Value::String(s) => s.clone(),
        other => dumps(other),
    }
}

/// One question's token sequence and the positions of its option markers.
pub struct Sequence {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
}

/// Build the token sequence for one question.
pub fn build_sequence(
    tok: &Tokenizer,
    state: &serde_json::Value,
    q: &Question,
    max_len: usize,
    head_max_len: usize,
) -> Result<Sequence, LayaError> {
    let mask_tok = tok.mask_token();
    let opts = render_options(q)?;
    let ins = q.ins.replace(&mask_tok, " ");
    let mut head_ids = tok.encode(&format!("{} question: {}", q.t.name(), ins))?;

    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(opts.len());
    for opt in &opts {
        let text = format!(" {}", opt.replace(&mask_tok, " "));
        let mut ids = tok.encode(&text)?;
        ids.truncate(48);
        let mut o = vec![tok.mask_id];
        o.extend(ids);
        opt_ids.push(o);
    }

    let total: usize = opt_ids.iter().map(|o| o.len()).sum();
    let mut opt_budget = head_max_len as isize - total as isize;
    if opt_budget < 16 {
        let per = ((head_max_len.saturating_sub(16)) / opt_ids.len().max(1)).max(4);
        for o in opt_ids.iter_mut() {
            o.truncate(per);
        }
        let total: usize = opt_ids.iter().map(|o| o.len()).sum();
        opt_budget = head_max_len as isize - total as isize;
    }
    let head_cap = (opt_budget.max(8) as usize).max(8);
    head_ids.truncate(head_cap);

    let mut ids = vec![tok.cls_id];
    ids.extend(head_ids.iter().copied());
    ids.push(tok.sep_id);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend(o.iter().copied());
    }
    ids.push(tok.sep_id);

    let room = max_len.saturating_sub(ids.len() + 1);
    let state_text = serialize_state(state).replace(&mask_tok, " ");
    let mut st = tok.encode(&state_text)?;
    st.truncate(room);
    ids.extend(st);
    ids.push(tok.sep_id);
    ids.truncate(max_len);

    Ok(Sequence {
        ids,
        markers: markers.into_iter().filter(|m| *m < max_len).collect(),
    })
}

/// A collated batch: padded ids, attention mask, marker positions and marker mask.
pub struct Batch {
    pub input_ids: Tensor,
    pub attention_mask: Tensor,
    pub marker_pos: Tensor,
    pub marker_mask: Tensor,
    pub qtype: Tensor,
    /// Number of real markers per row, in row order.
    pub marker_counts: Vec<usize>,
}

/// `collate_items`: pad to the longest sequence in the batch.
pub fn collate(
    seqs: &[Sequence],
    qtypes: &[usize],
    pad_id: u32,
    device: &Device,
) -> Result<Batch, LayaError> {
    let n = seqs.len();
    if n == 0 {
        return Err(LayaError::Tensor(
            "cannot collate an empty batch".to_string(),
        ));
    }
    let l = seqs.iter().map(|s| s.ids.len()).max().unwrap_or(0);
    let kmax = seqs.iter().map(|s| s.markers.len()).max().unwrap_or(0);

    let mut ids = vec![pad_id as i64; n * l];
    let mut att = vec![0f32; n * l];
    let mut mpos = vec![0i64; n * kmax];
    let mut mmask = vec![0f32; n * kmax];
    let mut counts = Vec::with_capacity(n);

    for (i, seq) in seqs.iter().enumerate() {
        for (j, id) in seq.ids.iter().enumerate() {
            ids[i * l + j] = *id as i64;
            att[i * l + j] = 1.0;
        }
        for (j, m) in seq.markers.iter().enumerate() {
            mpos[i * kmax + j] = *m as i64;
            mmask[i * kmax + j] = 1.0;
        }
        counts.push(seq.markers.len());
    }

    Ok(Batch {
        input_ids: Tensor::from_slice(&ids, (n, l), device)
            .map_err(crate::modernbert::candle_err)?,
        attention_mask: Tensor::from_slice(&att, (n, l), device)
            .map_err(crate::modernbert::candle_err)?,
        marker_pos: Tensor::from_slice(&mpos, (n, kmax), device)
            .map_err(crate::modernbert::candle_err)?,
        marker_mask: Tensor::from_slice(&mmask, (n, kmax), device)
            .map_err(crate::modernbert::candle_err)?,
        qtype: Tensor::from_slice(
            &qtypes.iter().map(|&q| q as i64).collect::<Vec<_>>(),
            (n,),
            device,
        )
        .map_err(crate::modernbert::candle_err)?,
        marker_counts: counts,
    })
}

/// The dtype a batch's masks use, for callers that need to build matching tensors.
pub const BATCH_DTYPE: DType = DType::F32;
