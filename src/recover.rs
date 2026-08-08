//! Recover password lines stored in legacy 8-bit / codepage encodings back to
//! UTF-8, so a wordlist or training corpus that predates UTF-8 is used in full
//! instead of being dropped. Leaked-password dumps commonly carry a small tail
//! of entries in the codepage of whatever site they came from — e.g. Spanish
//! `contrase\xf1a` (`contraseña`) in windows-1252, or Thai entries in
//! windows-874 — which are not valid UTF-8 and would otherwise be discarded.
//!
//! A line that is already valid UTF-8 passes through untouched (the hot path).
//! A non-UTF-8 line is decoded against a small set of common legacy codepages,
//! and the most plausible result is re-encoded as UTF-8 so it is byte-consistent
//! with the rest of the corpus (which is UTF-8). A line that no candidate
//! encoding turns into plausible text is reported and skipped.

use std::collections::BTreeMap;

use encoding_rs::{Encoding, GBK, SHIFT_JIS, WINDOWS_1251, WINDOWS_1252, WINDOWS_874};

/// Candidate legacy encodings, ordered by prior probability in real leaked
/// password corpora: Western-European (windows-1252) dominates, then Thai
/// (windows-874), Cyrillic (windows-1251), Japanese (shift_jis), Chinese (gbk).
/// A byte sequence valid in several encodings (e.g. `0xF1` = `ñ` in
/// windows-1252 or `๑` in windows-874) resolves to the earlier, more-likely one.
const CODEPAGES: &[&Encoding] = &[WINDOWS_1252, WINDOWS_874, WINDOWS_1251, SHIFT_JIS, GBK];

/// Minimum fraction of text-like characters for a decode to be accepted.
const MIN_PLAUSIBILITY: f32 = 0.7;

/// Score a decoded string: negative if it carries characters that betray a
/// wrong-encoding guess (control chars / replacement chars), otherwise the
/// fraction of characters that look like real password text.
fn plausibility(s: &str) -> f32 {
    let mut good = 0usize;
    let mut total = 0usize;
    for c in s.chars() {
        total += 1;
        let u = c as u32;
        // C0 controls (except tab), C1 controls (U+0080..=U+009F), and the
        // replacement char never occur in a correctly-decoded password — their
        // presence means we picked the wrong codepage.
        if (u < 0x20 && c != '\t') || (0x80..=0x9f).contains(&u) || c == '\u{FFFD}' {
            return -1.0;
        }
        if c.is_alphanumeric() || " .-_!@#$%&*+".contains(c) {
            good += 1;
        }
    }
    if total == 0 {
        return -1.0;
    }
    good as f32 / total as f32
}

/// Outcome of decoding one line of raw bytes.
pub enum Decoded {
    /// The line was already valid UTF-8.
    Utf8(String),
    /// The line was recovered from a legacy encoding (name kept for reporting).
    Recovered { text: String, encoding: &'static str },
    /// No candidate encoding produced plausible text — the caller should skip it.
    Undecodable,
}

impl Decoded {
    /// The usable text, if any (UTF-8 or recovered); `None` means skip.
    pub fn into_text(self) -> Option<String> {
        match self {
            Decoded::Utf8(s) | Decoded::Recovered { text: s, .. } => Some(s),
            Decoded::Undecodable => None,
        }
    }
}

/// Decode one line of raw bytes, recovering from a legacy codepage if needed.
pub fn decode_line(bytes: &[u8]) -> Decoded {
    if let Ok(s) = std::str::from_utf8(bytes) {
        return Decoded::Utf8(s.to_owned());
    }
    let mut best: Option<(f32, String, &'static str)> = None;
    for enc in CODEPAGES {
        let (cow, had_errors) = enc.decode_without_bom_handling(bytes);
        if had_errors {
            continue; // an unmappable byte for this codepage — not a match
        }
        let score = plausibility(&cow);
        // `>` (not `>=`) so an earlier, more-likely codepage wins on a tie.
        if score >= MIN_PLAUSIBILITY && best.as_ref().map_or(true, |(b, _, _)| score > *b) {
            best = Some((score, cow.into_owned(), enc.name()));
        }
    }
    match best {
        Some((_, text, encoding)) => Decoded::Recovered { text, encoding },
        None => Decoded::Undecodable,
    }
}

/// Running tally of recovery outcomes for an end-of-run report.
#[derive(Default)]
pub struct RecoveryStats {
    recovered: BTreeMap<&'static str, u64>,
    skipped:   u64,
}

impl RecoveryStats {
    pub fn note(&mut self, d: &Decoded) {
        match d {
            Decoded::Recovered { encoding, .. } => *self.recovered.entry(encoding).or_default() += 1,
            Decoded::Undecodable => self.skipped += 1,
            Decoded::Utf8(_) => {}
        }
    }

    fn total_recovered(&self) -> u64 {
        self.recovered.values().sum()
    }

    /// One-line human report, or `None` if nothing was recovered or skipped
    /// (the all-valid-UTF-8 common case, which stays silent).
    pub fn report(&self) -> Option<String> {
        if self.total_recovered() == 0 && self.skipped == 0 {
            return None;
        }
        let by_enc: Vec<String> = self
            .recovered
            .iter()
            .map(|(enc, n)| format!("{}={}", enc, n))
            .collect();
        let by_enc = if by_enc.is_empty() { "none".to_string() } else { by_enc.join(" ") };
        Some(format!(
            "recovered {} legacy-encoded line(s) [{}]; skipped {} undecodable",
            self.total_recovered(),
            by_enc,
            self.skipped
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_utf8_passes_through() {
        assert!(matches!(decode_line("password123".as_bytes()), Decoded::Utf8(s) if s == "password123"));
        // legitimate UTF-8 Thai/CJK is untouched
        assert!(matches!(decode_line("密码".as_bytes()), Decoded::Utf8(_)));
    }

    #[test]
    fn recovers_windows_1252_spanish() {
        // b"contrase\xf1a" -> "contraseña"
        match decode_line(b"contrase\xf1a") {
            Decoded::Recovered { text, encoding } => {
                assert_eq!(text, "contraseña");
                assert_eq!(encoding, "windows-1252");
            }
            _ => panic!("expected recovery"),
        }
    }

    #[test]
    fn recovers_thai_windows_874() {
        // repeating 0xA2 prefix is a windows-874 (TIS-620) Thai string, not cp1252
        match decode_line(b"\xa2\xb8\xa2\xb7") {
            Decoded::Recovered { encoding, .. } => assert_eq!(encoding, "windows-874"),
            _ => panic!("expected Thai recovery"),
        }
    }
}
