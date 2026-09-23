//! JSON stringification that matches Python's `json.dumps(..., ensure_ascii=False)`.
//!
//! Python's default separators are `(", ", ": ")` -- a space after every comma
//! and colon -- while `serde_json`'s compact form emits none. Criteria text is
//! fed straight into the model prompt, so the spacing has to match byte for byte.

use std::io::{self, Write};

use serde::Serialize;
use serde_json::ser::{Formatter, Serializer};
use serde_json::Value;

/// Formatter emitting `", "` between items and `": "` between keys and values.
pub struct PyJsonFormatter;

impl Formatter for PyJsonFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + Write,
    {
        writer.write_all(b": ")
    }
}

/// Serialize `value` the way `json.dumps(value, ensure_ascii=False)` would.
pub fn dumps(value: &Value) -> String {
    let mut buf = Vec::new();
    let mut ser = Serializer::with_formatter(&mut buf, PyJsonFormatter);
    value
        .serialize(&mut ser)
        .expect("serde_json::Value is always serializable");
    String::from_utf8(buf).expect("serde_json emits valid utf-8")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn object_spacing_matches_python() {
        assert_eq!(
            dumps(&json!({"desc": "phishing"})),
            "{\"desc\": \"phishing\"}"
        );
    }

    #[test]
    fn array_spacing_matches_python() {
        assert_eq!(dumps(&json!(["a", "b"])), "[\"a\", \"b\"]");
    }

    #[test]
    fn scalars_match_python() {
        assert_eq!(dumps(&json!(3)), "3");
        assert_eq!(dumps(&json!(3.5)), "3.5");
        assert_eq!(dumps(&json!(false)), "false");
        assert_eq!(dumps(&Value::Null), "null");
    }

    #[test]
    fn non_ascii_is_not_escaped() {
        assert_eq!(dumps(&json!({"d": "münchen"})), "{\"d\": \"münchen\"}");
    }

    #[test]
    fn nested_structures() {
        assert_eq!(
            dumps(&json!({"a": [1, {"b": 2}]})),
            "{\"a\": [1, {\"b\": 2}]}"
        );
    }
}
