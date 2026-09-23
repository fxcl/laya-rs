//! Email utilities for cleaning and structuring email inputs.
//!
//! Rust port of `laya/email.py` (the `clean_email_body` / `email_state` half; the
//! duplicate `email_questions` in that file is dead code and lives in
//! [`crate::presets`] instead).

use regex::Regex;
use std::sync::OnceLock;

fn quote_headers() -> &'static [Regex] {
    static RE: OnceLock<Vec<Regex>> = OnceLock::new();
    RE.get_or_init(|| {
        vec![
            Regex::new(r"(?i)^\s*On .{0,300}wrote:\s*$").unwrap(),
            Regex::new(r"(?i)^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}").unwrap(),
            Regex::new(r"^\s*_{8,}\s*$").unwrap(),
            Regex::new(r"(?i)^\s*From:\s.+$").unwrap(),
        ]
    })
}

fn signature_markers() -> &'static [Regex] {
    static RE: OnceLock<Vec<Regex>> = OnceLock::new();
    RE.get_or_init(|| {
        vec![
            Regex::new(r"^\s*--\s*$").unwrap(),
            Regex::new(r"(?i)^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\w ,!.]*$")
                .unwrap(),
            Regex::new(r"(?i)^\s*sent from my (iphone|android|mobile|ipad)").unwrap(),
        ]
    })
}

fn disclaimer() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|if you (have )?received this (e-?mail|message) in error)",
        )
        .unwrap()
    })
}

/// Remove quoted email history, signatures and disclaimers to keep input focused.
pub fn clean_email_body(body: Option<&str>, max_chars: usize) -> String {
    let text = body
        .unwrap_or("")
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace("\\n", "\n");
    let mut lines: Vec<&str> = Vec::new();
    for line in text.split('\n') {
        if !lines.is_empty() && quote_headers().iter().any(|p| p.is_match(line)) {
            break;
        }
        if line.trim_start().starts_with('>') {
            continue;
        }
        lines.push(line.trim_end());
    }
    let mut cut = lines.len();
    // Python: range(max(1, min(int(len(lines)*0.6), len(lines)-8)), len(lines))
    let start = ((lines.len() as f64 * 0.6) as i64)
        .min(lines.len() as i64 - 8)
        .max(1) as usize;
    for (i, line) in lines.iter().enumerate().skip(start) {
        if line.trim().chars().count() <= 40 && signature_markers().iter().any(|p| p.is_match(line))
        {
            cut = i;
            break;
        }
    }
    let kept = &lines[..cut];
    let joined = kept.join("\n");
    // Python: re.split(r"\n\s*\n", ...) then drop paragraphs matching the disclaimer.
    let paragraphs: Vec<&str> = blank_line()
        .split(&joined)
        .filter(|p| !disclaimer().is_match(p))
        .collect();
    let text = paragraphs
        .iter()
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    collapse_spaces(&text).chars().take(max_chars).collect()
}

/// `\n\s*\n` -- the blank-line paragraph separator.
fn blank_line() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\n\s*\n").unwrap())
}

/// `re.sub(r"[ \t]+", " ", text)`
fn collapse_spaces(text: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[ \t]+").unwrap())
        .replace_all(text, " ")
        .into_owned()
}

/// Construct a clean state dictionary for email classification.
///
/// Returns ordered `(key, value)` pairs: `subject`, `body`, optional `from`, then
/// any extra fields whose value is not `None`.
pub fn email_state(
    subject: Option<&str>,
    body: Option<&str>,
    sender: Option<&str>,
    clean: bool,
    extra: &[(String, Option<String>)],
) -> Vec<(String, String)> {
    let mut state = vec![
        (
            "subject".to_string(),
            subject.unwrap_or("").trim().to_string(),
        ),
        (
            "body".to_string(),
            if clean {
                clean_email_body(body, 3000)
            } else {
                body.unwrap_or("").to_string()
            },
        ),
    ];
    if let Some(s) = sender {
        if !s.is_empty() {
            state.push(("from".to_string(), s.to_string()));
        }
    }
    for (k, v) in extra {
        if let Some(v) = v {
            state.push((k.clone(), v.clone()));
        }
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_quoted_history_and_signature() {
        let body = "Hi, we were billed twice for invoice 4411 in March.\n\
                    Could you refund the duplicate? Thanks.\n\n\
                    On Mon, Mar 3 2026 at 10:00, billing@acme.com wrote:\n\
                    > Your invoice is attached.\n\n\
                    --\n\
                    Jane Doe\n\
                    Account Manager, Acme\n\
                    sent from my iPhone";
        let out = clean_email_body(Some(body), 3000);
        assert!(out.contains("billed twice"), "{}", out);
        assert!(!out.contains("Your invoice is attached"), "{}", out);
        assert!(!out.contains("Jane Doe"), "{}", out);
        assert!(!out.contains("sent from my iPhone"), "{}", out);
    }

    #[test]
    fn drops_confidentiality_disclaimer() {
        let body = "Please refund the duplicate charge.\n\n\
                    CONFIDENTIALITY NOTICE: This email is intended solely for the addressee.";
        let out = clean_email_body(Some(body), 3000);
        assert!(out.contains("Please refund"), "{}", out);
        assert!(!out.contains("CONFIDENTIALITY"), "{}", out);
    }

    #[test]
    fn normalises_newlines_and_truncates() {
        assert_eq!(clean_email_body(Some("a\\nb"), 3000), "a\nb");
        assert_eq!(clean_email_body(Some("abcdef"), 3), "abc");
        assert_eq!(clean_email_body(None, 3000), "");
    }

    #[test]
    fn email_state_shape() {
        let state = email_state(
            Some("  Invoice 4411  "),
            Some("we were billed twice"),
            Some("ap@acme.com"),
            true,
            &[
                ("tier".to_string(), Some("enterprise".to_string())),
                ("skip".to_string(), None),
            ],
        );
        assert_eq!(
            state,
            vec![
                ("subject".to_string(), "Invoice 4411".to_string()),
                ("body".to_string(), "we were billed twice".to_string()),
                ("from".to_string(), "ap@acme.com".to_string()),
                ("tier".to_string(), "enterprise".to_string()),
            ]
        );
    }

    #[test]
    fn email_state_without_sender_or_clean() {
        let state = email_state(Some("s"), Some("b"), None, false, &[]);
        assert_eq!(
            state,
            vec![
                ("subject".to_string(), "s".to_string()),
                ("body".to_string(), "b".to_string())
            ]
        );
    }
}
