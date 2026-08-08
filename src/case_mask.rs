//! Per-token case masks for shape-conditioned generation (v0.5.0).
//!
//! The content model is trained case-folded (lowercase), so a generated candidate
//! is a sequence of lowercase-content *tokens*. A `CaseMask` assigns a case op to
//! each token slot, applied at decode time as each token's bytes are appended — so
//! the token is the unit of casing, not the character position. This natively
//! expresses what char-level masks cannot:
//!
//!   tokens ["spring","field"]  +  mask cap,cap   -> "SpringField"  (camelCase)
//!   tokens ["spring","field"]  +  mask upper      -> "SPRINGFIELD"  (all-caps)
//!   tokens ["spring","19"]     +  mask cap        -> "Spring19"     (U1L5D2)
//!   tokens ["spring","field"]  +  mask lower      -> "springfield"  (consumer)
//!
//! A run may carry several masks; each terminal candidate is emitted once per mask
//! (the operator controls multiplicity). With no `--case-shape`, `case_masks` is
//! empty and the generator keeps its exact prior behavior.

/// Case operation applied to one token's decoded bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CaseOp {
    /// Leave as-is (content is already lowercase; a true no-op).
    Lower,
    /// Capitalize the first ASCII-alphabetic byte of the token; rest untouched.
    Cap,
    /// Uppercase every ASCII-alphabetic byte of the token.
    Upper,
}

/// Per-token-slot case pattern. `ops[i]` applies to token i; positions beyond
/// `ops.len()` use `tail` (so "capitalize first token, lowercase rest" is
/// `ops=[Cap], tail=Lower`, and "title case every token" is `ops=[], tail=Cap`).
#[derive(Clone, Debug)]
pub struct CaseMask {
    pub ops: Vec<CaseOp>,
    pub tail: CaseOp,
    pub label: String,
}

impl CaseMask {
    #[inline]
    pub fn op_for(&self, pos: usize) -> CaseOp {
        self.ops.get(pos).copied().unwrap_or(self.tail)
    }

    /// Parse one pattern token. Accepts a named shortcut, or a hashcat-style
    /// `?`-sequence of per-slot ops (`?l`=lower, `?c`=cap-first-letter, `?u`=upper);
    /// the last op repeats as the tail.
    fn parse_one(spec: &str) -> Result<CaseMask, String> {
        let s = spec.trim();
        let lc = s.to_ascii_lowercase();
        let named = match lc.as_str() {
            "lower" | "l" => Some((vec![], CaseOp::Lower)),
            // capitalize first token only, lowercase the rest (the U1L* family)
            "cap1" | "capfirst" | "cap" => Some((vec![CaseOp::Cap], CaseOp::Lower)),
            // capitalize every token (Title Case across tokens)
            "title" | "capall" => Some((vec![], CaseOp::Cap)),
            "upper" | "allcaps" => Some((vec![], CaseOp::Upper)),
            _ => None,
        };
        if let Some((ops, tail)) = named {
            return Ok(CaseMask { ops, tail, label: lc });
        }
        // hashcat-style `?`-sequence, e.g. "?c?l?l" or "?c?u"
        if !s.starts_with('?') {
            return Err(format!(
                "unknown case-shape '{spec}' — use a named shortcut (lower/cap1/title/upper) \
                 or a `?`-sequence like ?c?l?u (?l=lower, ?c=cap-first, ?u=upper)"));
        }
        let bytes = s.as_bytes();
        let mut ops = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] != b'?' {
                return Err(format!("expected '?' before '{}' in case-shape '{spec}'",
                    bytes[i] as char));
            }
            i += 1;
            if i >= bytes.len() {
                return Err(format!("dangling '?' at end of case-shape '{spec}'"));
            }
            let op = match bytes[i].to_ascii_lowercase() {
                b'l' => CaseOp::Lower,
                b'c' => CaseOp::Cap,
                b'u' => CaseOp::Upper,
                other => return Err(format!(
                    "unknown case op '?{}' in case-shape '{spec}' (use ?l, ?c, or ?u)",
                    other as char)),
            };
            ops.push(op);
            i += 1;
        }
        if ops.is_empty() {
            return Err(format!("empty case-shape '{spec}'"));
        }
        let tail = *ops.last().unwrap();
        Ok(CaseMask { ops, tail, label: s.to_string() })
    }

    /// Parse a `--case-shape` spec: one or more patterns separated by `;`.
    /// e.g. "lower;cap1;upper" -> three masks emitted per candidate.
    pub fn parse_spec(spec: &str) -> Result<Vec<CaseMask>, String> {
        let masks: Result<Vec<_>, _> = spec
            .split(';')
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .map(CaseMask::parse_one)
            .collect();
        let masks = masks?;
        if masks.is_empty() {
            return Err(format!("no patterns parsed from case-shape '{spec}'"));
        }
        Ok(masks)
    }
}

/// Apply a case op in place to one token's decoded bytes.
#[inline]
pub fn apply_op(bytes: &mut [u8], op: CaseOp) {
    match op {
        CaseOp::Lower => {} // content is already lowercase — no-op
        CaseOp::Upper => {
            for b in bytes.iter_mut() {
                b.make_ascii_uppercase();
            }
        }
        CaseOp::Cap => {
            for b in bytes.iter_mut() {
                if b.is_ascii_alphabetic() {
                    b.make_ascii_uppercase();
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cased(tokens: &[&str], mask: &CaseMask) -> String {
        let mut buf = Vec::new();
        for (pos, t) in tokens.iter().enumerate() {
            let start = buf.len();
            buf.extend_from_slice(t.as_bytes());
            let end = buf.len();
            apply_op(&mut buf[start..end], mask.op_for(pos));
        }
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn named_patterns() {
        let toks = ["spring", "field"];
        assert_eq!(cased(&toks, &CaseMask::parse_spec("lower").unwrap()[0]), "springfield");
        assert_eq!(cased(&toks, &CaseMask::parse_spec("cap1").unwrap()[0]), "Springfield");
        assert_eq!(cased(&toks, &CaseMask::parse_spec("title").unwrap()[0]), "SpringField");
        assert_eq!(cased(&toks, &CaseMask::parse_spec("upper").unwrap()[0]), "SPRINGFIELD");
    }

    #[test]
    fn per_slot_and_digits() {
        // cap first token, leave the digit token (no alpha -> Cap is a no-op)
        let m = &CaseMask::parse_spec("cap1").unwrap()[0];
        assert_eq!(cased(&["spring", "19"], m), "Spring19");
        // explicit `?`-sequence per-slot, tail repeats last op
        let m = &CaseMask::parse_spec("?c?u").unwrap()[0];
        assert_eq!(cased(&["mc", "donald", "smith"], m), "McDONALDSMITH");
        // three explicit slots: cap, lower, upper (tail=upper)
        let m = &CaseMask::parse_spec("?c?l?u").unwrap()[0];
        assert_eq!(cased(&["a", "b", "c", "d"], m), "AbCD");
    }

    #[test]
    fn question_syntax_errors() {
        assert!(CaseMask::parse_spec("c,l,u").is_err());   // old comma syntax rejected
        assert!(CaseMask::parse_spec("?").is_err());       // dangling ?
        assert!(CaseMask::parse_spec("?x").is_err());      // unknown op
        assert!(CaseMask::parse_spec("?cl").is_err());     // missing ? before l
    }

    #[test]
    fn multi_pattern_spec() {
        let masks = CaseMask::parse_spec("lower;cap1;upper").unwrap();
        assert_eq!(masks.len(), 3);
        // mixed named + `?`-sequence
        let masks = CaseMask::parse_spec("lower;?c?l").unwrap();
        assert_eq!(masks.len(), 2);
    }
}
