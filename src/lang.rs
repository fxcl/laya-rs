//! Dependency-free language/script detection used to route between Laya checkpoints.
//!
//! Rust port of `laya/lang.py`. Routing only needs one decision: *is this English
//! Latin text, or is it something the English checkpoint cannot read?* Script
//! detection is exact; the Latin-script language guess is a stopword/diacritic
//! heuristic and is explicitly best-effort.

use serde_json::{Map, Value};

/// Unicode blocks that the English (ModernBERT-large, 50k English BPE) checkpoint cannot read.
const SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    (
        "cyrillic",
        &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    ),
    ("armenian", &[(0x0530, 0x058F)]),
    ("hebrew", &[(0x0590, 0x05FF)]),
    (
        "arabic",
        &[
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ],
    ),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("gurmukhi", &[(0x0A00, 0x0A7F)]),
    ("gujarati", &[(0x0A80, 0x0AFF)]),
    ("oriya", &[(0x0B00, 0x0B7F)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("telugu", &[(0x0C00, 0x0CFF)]),
    ("kannada", &[(0x0C80, 0x0CFF)]),
    ("malayalam", &[(0x0D00, 0x0D7F)]),
    ("sinhala", &[(0x0D80, 0x0DFF)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    ("lao", &[(0x0E80, 0x0EFF)]),
    ("tibetan", &[(0x0F00, 0x0FFF)]),
    ("myanmar", &[(0x1000, 0x109F)]),
    ("georgian", &[(0x10A0, 0x10FF)]),
    ("ethiopic", &[(0x1200, 0x137F)]),
    ("khmer", &[(0x1780, 0x17FF)]),
    (
        "hangul",
        &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    ),
    (
        "kana",
        &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    ),
    (
        "han",
        &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    ),
];

/// Function words. Latin-script languages overlap heavily (de/la/le/un/e/que), so each hit is
/// weighted and a margin is required before calling something non-English.
const STOP_EN: &[&str] = &[
    "the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that", "this",
    "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from", "will", "can",
    "would", "there", "their", "what", "which", "please", "we", "i",
];
const STOP_FR: &[&str] = &[
    "le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur", "pas",
    "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
];
const STOP_DE: &[&str] = &[
    "der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für", "auf",
    "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber",
];
const STOP_ES: &[&str] = &[
    "el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como", "pero",
    "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus",
];
const STOP_PT: &[&str] = &[
    "os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da", "dos",
    "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela",
];
const STOP_IT: &[&str] = &[
    "il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
    "questo", "questa", "anche", "come", "più", "sono", "nella", "alla",
];
const STOP_NL: &[&str] = &[
    "het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan", "door",
    "maar", "ook", "worden", "deze", "naar", "wordt",
];

/// Iteration order of the Python `_STOP` dict; the first maximum wins, so this order is load-bearing.
const STOP_LANGS: &[&str] = &["en", "fr", "de", "es", "pt", "it", "nl"];

fn stopwords(lang: &str) -> &'static [&'static str] {
    match lang {
        "en" => STOP_EN,
        "fr" => STOP_FR,
        "de" => STOP_DE,
        "es" => STOP_ES,
        "pt" => STOP_PT,
        "it" => STOP_IT,
        "nl" => STOP_NL,
        _ => &[],
    }
}

const NON_EN_DIACRITICS: &str = "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö";

/// `[^\W\d_]+` -- Python's word pattern minus digits and underscore.
fn word_re() -> &'static regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"[^\W\d_]+").expect("valid regex"))
}

/// Round to 4 decimals the way Python's `round(x, 4)` does (correctly rounded, ties to even).
fn round4(x: f64) -> f64 {
    format!("{:.4}", x).parse().unwrap_or(x)
}

/// Collect the string leaves of a state (str / dict / list), so detection sees real content.
fn iter_text(state: &Value, depth: usize, out: &mut Vec<String>) {
    if depth > 6 {
        return;
    }
    match state {
        Value::String(s) => out.push(s.clone()),
        Value::Object(map) => {
            for (_, v) in map {
                iter_text(v, depth + 1, out);
            }
        }
        Value::Array(items) => {
            for v in items {
                iter_text(v, depth + 1, out);
            }
        }
        _ => {}
    }
}

/// Flatten a state into the text used for detection (keys are ignored: they are usually English).
pub fn state_text(state: &Value) -> String {
    let mut parts = Vec::new();
    iter_text(state, 0, &mut parts);
    parts.join(" ").chars().take(4000).collect()
}

/// Count letters per script, in first-encounter order, with `latin` appended last.
/// Mirrors the insertion order of the Python `counts` dict, which decides ties.
fn script_counts(text: &str, latin_first: bool) -> Vec<(String, u32)> {
    let mut counts: Vec<(String, u32)> = Vec::new();
    if latin_first {
        counts.push(("latin".to_string(), 0));
    }
    let mut latin: u32 = 0;
    for ch in text.chars() {
        if !ch.is_alphabetic() {
            continue;
        }
        let cp = ch as u32;
        if cp < 0x0250 || (0x1E00..=0x1EFF).contains(&cp) {
            latin += 1;
            continue;
        }
        for (name, ranges) in SCRIPT_RANGES {
            if ranges.iter().any(|&(lo, hi)| lo <= cp && cp <= hi) {
                match counts.iter_mut().find(|(k, _)| k == name) {
                    Some(entry) => entry.1 += 1,
                    None => counts.push((name.to_string(), 1)),
                }
                break;
            }
        }
    }
    if !latin_first {
        counts.push(("latin".to_string(), latin));
    } else {
        if let Some(entry) = counts.iter_mut().find(|(k, _)| k == "latin") {
            entry.1 = latin;
        }
    }
    counts
}

/// Dominant script of `text`: 'latin', 'han', 'devanagari', ... or 'unknown' if there are no letters.
pub fn detect_script(text: &str) -> String {
    let counts = script_counts(text, false);
    let total: u32 = counts.iter().map(|(_, c)| c).sum();
    if total == 0 {
        return "unknown".to_string();
    }
    // Python's max() keeps the FIRST maximal item in iteration order.
    let mut best = &counts[0];
    for c in &counts[1..] {
        if c.1 > best.1 {
            best = c;
        }
    }
    best.0.clone()
}

/// Fraction of alphabetic characters belonging to each detected script.
pub fn script_profile(text: &str) -> Vec<(String, f64)> {
    let counts = script_counts(text, true);
    let total: u32 = counts.iter().map(|(_, c)| c).sum();
    if total == 0 {
        return Vec::new();
    }
    counts
        .iter()
        .filter(|(_, c)| *c > 0)
        .map(|(k, c)| (k.clone(), *c as f64 / total as f64))
        .collect()
}

/// Best-effort language code for Latin-script text, or None when undecided.
///
/// Scores function-word hits per language and requires the winner to beat English by a margin,
/// so ordinary English is never misrouted. Short inputs usually return None on purpose.
pub fn guess_latin_language(text: &str) -> Option<String> {
    let words: Vec<String> = word_re()
        .find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .collect();
    if words.len() < 4 {
        return None;
    }
    let scores: Vec<(&str, usize)> = STOP_LANGS
        .iter()
        .map(|&lg| {
            (
                lg,
                words
                    .iter()
                    .filter(|w| stopwords(lg).contains(&w.as_str()))
                    .count(),
            )
        })
        .collect();
    let lowered = text.to_lowercase();
    let diac = lowered
        .chars()
        .filter(|c| NON_EN_DIACRITICS.contains(*c))
        .count();
    let diac_rate = diac as f64 / lowered.chars().count().max(1) as f64;
    let en = scores
        .iter()
        .find(|(lg, _)| *lg == "en")
        .map(|(_, s)| *s)
        .unwrap_or(0);
    let mut best_lg: Option<&str> = None;
    let mut best = 0usize;
    for &(lg, s) in scores.iter().filter(|(lg, _)| *lg != "en") {
        if s > best {
            best = s;
            best_lg = Some(lg);
        }
    }
    if best == 0 && diac_rate < 0.02 {
        return if en > 0 { Some("en".to_string()) } else { None };
    }
    // a non-English language needs a clear margin over English function words
    if let Some(lg) = best_lg {
        if best >= 2.max(en + 2) {
            return Some(lg.to_string());
        }
    }
    if diac_rate >= 0.04 && best_lg.is_some() && best >= en {
        return best_lg.map(|s| s.to_string());
    }
    if en > 0 {
        Some("en".to_string())
    } else {
        None
    }
}

/// Full detection result for a state.
#[derive(Clone, Debug, PartialEq)]
pub struct Analysis {
    pub script: String,
    pub script_profile: Vec<(String, f64)>,
    pub language: Option<String>,
    pub is_english: bool,
    pub non_latin_fraction: f64,
}

/// Full detection result for a state.
pub fn analyse(state: &Value) -> Analysis {
    let text = state_text(state);
    let prof = script_profile(&text);
    let script = detect_script(&text);
    let non_latin = if prof.is_empty() {
        0.0
    } else {
        round4(
            1.0 - prof
                .iter()
                .find(|(k, _)| k == "latin")
                .map(|(_, v)| *v)
                .unwrap_or(0.0),
        )
    };
    if script == "unknown" {
        return Analysis {
            script,
            script_profile: prof,
            language: None,
            is_english: true,
            non_latin_fraction: 0.0,
        };
    }
    if script != "latin" {
        return Analysis {
            script,
            script_profile: prof,
            language: None,
            is_english: false,
            non_latin_fraction: non_latin,
        };
    }
    let lang = guess_latin_language(&text);
    let is_english = matches!(lang.as_deref(), None | Some("en"));
    Analysis {
        script,
        script_profile: prof,
        language: lang,
        is_english,
        non_latin_fraction: non_latin,
    }
}

/// True when the English checkpoint can be expected to read this state.
pub fn is_english(state: &Value) -> bool {
    analyse(state).is_english
}

impl Analysis {
    /// The dict form Python's `analyse()` returns.
    pub fn to_value(&self) -> Value {
        let mut profile = Map::new();
        for (k, v) in &self.script_profile {
            profile.insert(k.clone(), json_number(*v));
        }
        let mut m = Map::new();
        m.insert("script".to_string(), Value::String(self.script.clone()));
        m.insert("script_profile".to_string(), Value::Object(profile));
        m.insert(
            "language".to_string(),
            match &self.language {
                Some(l) => Value::String(l.clone()),
                None => Value::Null,
            },
        );
        m.insert("is_english".to_string(), Value::Bool(self.is_english));
        m.insert(
            "non_latin_fraction".to_string(),
            json_number(self.non_latin_fraction),
        );
        Value::Object(m)
    }
}

/// Build a JSON number, keeping integers integral (Python's json does the same).
fn json_number(v: f64) -> Value {
    serde_json::Number::from_f64(v)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn s(text: &str) -> Value {
        Value::String(text.to_string())
    }

    // ------------------------------------------------------------- script detection
    #[test]
    fn script_detection_table() {
        let cases = [
            (
                "english",
                "The customer was charged twice and wants a refund.",
                "latin",
            ),
            ("armenian", "Հայերեն", "armenian"),
            ("armenian uppercase", "ՀԱՅԵՐԵՆ", "armenian"),
            ("armenian punctuation only", "։֊", "unknown"),
            (
                "french",
                "Le client a été facturé deux fois et demande un remboursement.",
                "latin",
            ),
            (
                "hindi",
                "ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।",
                "devanagari",
            ),
            (
                "japanese",
                "お客様は二重に請求されたため返金を希望しています。",
                "kana",
            ),
            ("chinese", "客户被重复扣款要求退款", "han"),
            ("korean", "고객이 두 번 청구되어 환불을 원합니다", "hangul"),
            (
                "arabic",
                "تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال",
                "arabic",
            ),
            ("tamil", "வாடிக்கையாளரிடம் இருமுறை கட்டணம் வசூலிக்கப்பட்டது", "tamil"),
            (
                "russian",
                "С клиента дважды сняли деньги и он хочет возврат",
                "cyrillic",
            ),
            ("thai", "ลูกค้าถูกเรียกเก็บเงินสองครั้งและต้องการเงินคืน", "thai"),
            (
                "greek",
                "Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων",
                "greek",
            ),
            ("hebrew", "הלקוח חויב פעמיים ורוצה החזר כספי", "hebrew"),
            ("empty", "", "unknown"),
            ("digits only", "12345 6789", "unknown"),
        ];
        for (label, text, want) in cases {
            assert_eq!(detect_script(text), want, "script/{}", label);
        }
    }

    // ------------------------------------------------------------- english vs not
    #[test]
    fn is_english_table() {
        let cases = [
            (
                "plain english",
                "Please refund the duplicate charge on invoice 4411 today.",
                true,
            ),
            ("armenian", "Հայերեն", false),
            ("english short", "refund me", true),
            ("hindi", "ग्राहक से दो बार शुल्क लिया गया", false),
            ("japanese", "お客様は二重に請求されました", false),
            ("russian", "С клиента дважды сняли деньги", false),
            (
                "french long",
                "Le client a été facturé deux fois et il demande un remboursement pour la \
                 facture qui a été payée le mois dernier avec la carte de crédit",
                false,
            ),
            (
                "german long",
                "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die \
                 Rechnung die nicht korrekt ist und auch nicht bezahlt wurde",
                false,
            ),
        ];
        for (label, text, want) in cases {
            assert_eq!(is_english(&s(text)), want, "is_english/{}", label);
        }
    }

    // ------------------------------------------------------------- Latin language guess
    #[test]
    fn latin_language_guess_table() {
        let cases = [
            (
                "english",
                "The customer was charged twice and wants a refund for this invoice",
                Some("en"),
            ),
            (
                "french",
                "Le client a ete facture deux fois et il demande un remboursement pour la facture",
                Some("fr"),
            ),
            (
                "german",
                "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung",
                Some("de"),
            ),
            (
                "spanish",
                "El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura",
                Some("es"),
            ),
            ("too short", "refund", None),
        ];
        for (label, text, want) in cases {
            assert_eq!(
                guess_latin_language(text).as_deref(),
                want,
                "latin_lang/{}",
                label
            );
        }
        // a non-English guess must never fire on ordinary English
        assert_eq!(
            guess_latin_language(
                "Please refund the duplicate charge on invoice 4411 today because \
                 we have been waiting for three days and nobody has replied to us"
            ),
            Some("en".to_string())
        );
    }

    // ------------------------------------------------------------- state flattening
    #[test]
    fn state_flattening() {
        assert!(state_text(&json!({"body": "charged twice", "n": 3})).contains("charged twice"));
        assert!(state_text(&json!({"a": {"b": ["deep"]}})).contains("deep"));
        assert!(state_text(&json!(["x", {"y": "z"}])).contains('x'));
        assert_eq!(state_text(&Value::Null), "");
        // keys must not drive detection: English keys around Hindi content stay non-English
        assert!(
            !analyse(&json!({"subject": "नमस्ते", "body": "ग्राहक से दो बार शुल्क लिया गया"})).is_english
        );
    }

    #[test]
    fn script_profile_numbers() {
        assert_eq!(
            script_profile("Հայերեն"),
            vec![("armenian".to_string(), 1.0)]
        );
        assert_eq!(analyse(&s("Հայերեն abc")).non_latin_fraction, 0.7);
    }
}
