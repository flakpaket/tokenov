//! Enterprise-policy compliance + the L0 minimal-repair ladder.
//!
//! ## Policy (Windows AD complexity semantics)
//!
//! A candidate is compliant when it is **>= 8 characters** (Unicode codepoints,
//! not bytes) AND draws from **at least 3 of these 5 categories**, matching how
//! Windows Active Directory "password must meet complexity requirements"
//! actually classifies characters:
//!
//! 1. **Uppercase** letters — Unicode `Uppercase` property (`A`..`Z`, plus
//!    accented `Á`, Greek, Cyrillic `П`, …).
//! 2. **Lowercase** letters — Unicode `Lowercase` property (`a`..`z`, plus `ñ`,
//!    `é`, Cyrillic `п`, …).
//! 3. **Digits** — base-10 ASCII `0`..`9` only.
//! 4. **Special** — the specific non-alphanumeric ASCII set AD counts:
//!    ASCII punctuation + space. (Currency symbols and other Unicode symbols are
//!    NOT special, per Microsoft.)
//! 5. **Other alphabetic** — a Unicode character that is alphabetic but neither
//!    uppercase nor lowercase (category `Lo`/`Lt`/`Lm`: CJK, etc.).
//!
//! Characters in none of the five (e.g. the symbol `♥` = U+2665, category `So`,
//! or control bytes) contribute to **length only**, never to a class. Invalid
//! UTF-8 is treated as non-compliant — it cannot be a password a user could set.
//!
//! This is the corrected rule. An earlier version classified by raw byte
//! (`[a-zA-Z0-9]` vs "everything else = special", length in bytes), which
//! over-counted compliance for non-ASCII content: it read `ñ` (a lowercase
//! letter) and even a multi-byte symbol like `♥` as "special", and it measured
//! length in bytes. Both errors made non-ASCII candidates look compliant when a
//! real domain would reject them (e.g. `Aguilña` = {upper, lower} = 2 classes,
//! 7 chars; `I♥chris` = {upper, lower} = 2 classes, 7 chars). Impact was ~0.02%
//! of the stream (non-ASCII candidates are rare) but the definition was wrong.
//!
//! ## L0 ("cap-only") — the minimal length-preserving repair
//!
//!   0. already compliant                     -> emit as-is
//!   1. first char is ASCII `[a-z]` and no
//!      uppercase present yet                  -> capitalize it (+0 length),
//!      re-check; emit only if now compliant
//!   2. otherwise                             -> drop (do not emit)
//!
//! Capitalizing is the only transform (zero-length, closes the dominant "missing
//! uppercase" gap). It is applied **only to a leading ASCII lowercase letter** —
//! the overwhelmingly common case, and length-preserving as a single-byte flip.
//! A candidate led by a non-ASCII lowercase letter that would need only a capital
//! is dropped rather than Unicode-uppercased in place (Unicode case changes are
//! not always length-preserving, which would break the emit buffer's byte
//! bookkeeping); this is a negligible miss.

/// Minimum compliant length, in characters.
pub const MIN_LEN: usize = 8;

/// Class bits.
pub const CL_L: u8 = 1; // lowercase letter
pub const CL_U: u8 = 2; // uppercase letter
pub const CL_D: u8 = 4; // ASCII digit
pub const CL_S: u8 = 8; // AD special (ASCII punctuation + space)
pub const CL_O: u8 = 16; // other alphabetic (Unicode Lo/Lt/Lm)

/// AD's "special" category: ASCII punctuation + space. Currency and other
/// Unicode symbols deliberately excluded (they are not letters and not in AD's
/// special set, so they contribute nothing).
#[inline]
fn is_ad_special(c: char) -> bool {
    c == ' ' || c.is_ascii_punctuation()
}

/// Class of a single character (0 if it belongs to none of the five).
#[inline]
fn class_of(c: char) -> u8 {
    if c.is_ascii_digit() {
        CL_D
    } else if c.is_uppercase() {
        CL_U
    } else if c.is_lowercase() {
        CL_L
    } else if is_ad_special(c) {
        CL_S
    } else if c.is_alphabetic() {
        CL_O
    } else {
        0
    }
}

/// Bitset of the AD classes present in `s`.
#[inline]
pub fn classes_str(s: &str) -> u8 {
    let mut m = 0u8;
    for c in s.chars() {
        m |= class_of(c);
        if m == CL_L | CL_U | CL_D | CL_S | CL_O {
            break;
        }
    }
    m
}

/// Count of distinct classes in a class bitset.
#[inline]
pub fn n_classes(m: u8) -> u32 {
    m.count_ones()
}

// `compliant` / `compliant_str` are the plain boolean policy predicate, used only
// as the test oracle below (production takes the AsIs/Cap/Drop path via `decide`).
// Gated to test builds so they don't ship as dead code in the release binary.

/// Enterprise policy on a decoded string: >= 8 chars AND >= 3 of the 5 classes.
#[cfg(test)]
#[inline]
pub fn compliant_str(s: &str) -> bool {
    s.chars().count() >= MIN_LEN && n_classes(classes_str(s)) >= 3
}

/// Enterprise policy on raw bytes. Invalid UTF-8 is non-compliant.
#[cfg(test)]
#[inline]
pub fn compliant(b: &[u8]) -> bool {
    std::str::from_utf8(b).map(compliant_str).unwrap_or(false)
}

/// L0 decision for one candidate. The caller performs the mutation for `Cap`
/// (uppercase `b[0]`, guaranteed to be an ASCII lowercase letter), so this
/// function never mutates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Already compliant — emit unchanged.
    AsIs,
    /// Emit with `b[0]` ASCII-uppercased (guaranteed compliant after the flip).
    Cap,
    /// Not repairable by cap-only — drop.
    Drop,
}

/// Decide the L0 action for `b` without mutating it.
#[inline]
pub fn decide(b: &[u8]) -> Decision {
    let s = match std::str::from_utf8(b) {
        Ok(s) => s,
        Err(_) => return Decision::Drop, // not a settable password
    };
    // Cap adds no length, so a sub-floor candidate can never be repaired.
    if s.chars().count() < MIN_LEN {
        return Decision::Drop;
    }
    let cls = classes_str(s);
    if n_classes(cls) >= 3 {
        return Decision::AsIs;
    }
    // Cap only helps when the first byte is an ASCII lowercase letter and no
    // uppercase is present yet. Uppercasing turns that leading L into a U, so the
    // honest post-cap class set is `classes(tail) | U` — recomputed on `&s[1..]`
    // (valid boundary: an ASCII byte is a whole char). This is what stops
    // `a2345678` (post-cap `A2345678` = {U,D} = 2) from being emitted.
    if b[0].is_ascii_lowercase() && (cls & CL_U == 0) {
        let post = classes_str(&s[1..]) | CL_U;
        if n_classes(post) >= 3 {
            return Decision::Cap;
        }
    }
    Decision::Drop
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_compliance_unchanged() {
        assert!(!compliant(b"password")); // 1 class
        assert!(!compliant(b"password1")); // {L,D} = 2
        assert!(!compliant(b"short1A")); // 7 chars
        assert!(compliant(b"Password1")); // {L,U,D}, 9
        assert!(compliant(b"passw0rd!")); // {L,D,S}, 9
    }

    #[test]
    fn accented_letter_is_a_letter_not_special() {
        // ñ is a lowercase letter (Unicode Ll), NOT special.
        // "Aguilña" = A(U) + guilña(L) = 2 classes, and 7 chars -> NOT compliant.
        assert!(!compliant("Aguilña".as_bytes()));
        assert_eq!(decide("Aguilña".as_bytes()), Decision::Drop);
        // A legitimately compliant accented password IS kept:
        // "Contraseña1" = C(U) + ontraseña(L) + 1(D) = {U,L,D} = 3, 11 chars.
        assert!(compliant("Contraseña1".as_bytes()));
        assert_eq!(decide("Contraseña1".as_bytes()), Decision::AsIs);
        // lowercase form gets capped: "contraseña1" -> "Contraseña1".
        assert_eq!(decide("contraseña1".as_bytes()), Decision::Cap);
    }

    #[test]
    fn unicode_symbol_contributes_no_class() {
        // ♥ (U+2665, category So) is neither a letter nor an AD special.
        // "I♥chris" = I(U) + chris(L) = 2 classes, and 7 chars -> NOT compliant.
        assert!(!compliant("I♥chris".as_bytes()));
        assert_eq!(decide("I♥chris".as_bytes()), Decision::Drop);
        // Even at length >= 8 it stays 2 classes:
        assert!(!compliant("I♥christopher".as_bytes())); // {U,L} = 2
    }

    #[test]
    fn cyrillic_cased_letters_count_as_case() {
        // Cyrillic П is uppercase (Lu), пароль lowercase (Ll).
        // "Пароль12" = {U,L,D} = 3, 8 chars -> compliant.
        assert!(compliant("Пароль12".as_bytes()));
        // all-lowercase Cyrillic + digit = {L,D} = 2 -> not compliant.
        assert!(!compliant("пароль12".as_bytes()));
        // and it gets capped (first byte is NOT ASCII lowercase, so it can't be
        // capped in place) -> Drop, not Cap.
        assert_eq!(decide("пароль12".as_bytes()), Decision::Drop);
    }

    #[test]
    fn cjk_is_the_other_alphabetic_class() {
        // 字 is alphabetic (Lo) but neither upper nor lower -> the O category.
        // "字Abcdef1" = O + U + L + D = 4 classes, 8 chars -> compliant.
        assert!(compliant("字Abcdef1".as_bytes()));
        // pure CJK is one class only.
        assert!(!compliant("字字字字字字字字".as_bytes())); // {O} = 1
    }

    #[test]
    fn length_is_in_characters_not_bytes() {
        // "Añ1" style: byte length can exceed char length. "Añ1señor" is 8 chars
        // (A ñ 1 s e ñ o r) = {U,L,D} = 3 -> compliant; a 7-char multibyte string
        // whose byte length >= 8 must still be rejected.
        assert!(compliant("Añ1señor".as_bytes())); // 8 chars, {U,L,D}
        assert!(!compliant("Añ1seño".as_bytes())); // 7 chars, byte-len 9
    }

    #[test]
    fn cap_recheck_is_honest_single_lowercase() {
        // a2345678 = {L,D}, 8 chars, no upper, ASCII-lowercase first. Cap removes
        // the only L: A2345678 = {U,D} = 2 -> Drop (not Cap).
        assert_eq!(decide(b"a2345678"), Decision::Drop);
        assert!(!compliant(b"A2345678"));
        // michael1 = {L,D}, 8 chars -> cap -> Michael1 {L,U,D} = 3.
        assert_eq!(decide(b"michael1"), Decision::Cap);
    }

    #[test]
    fn invalid_utf8_is_dropped() {
        assert_eq!(decide(&[0xff, 0xfe, b'a', b'b', b'c', b'1', b'2', b'3']), Decision::Drop);
        assert!(!compliant(&[0xff, 0xfe]));
    }

    #[test]
    fn every_cap_emission_is_compliant() {
        for w in [
            &b"michael1"[..],
            &b"password9"[..],
            &b"abcdef.1"[..],
            "contraseña1".as_bytes(),
        ] {
            if decide(w) == Decision::Cap {
                let mut m = w.to_vec();
                m[0] = m[0].to_ascii_uppercase();
                assert!(compliant(&m), "cap emission not compliant: {:?}", w);
            }
        }
    }
}
