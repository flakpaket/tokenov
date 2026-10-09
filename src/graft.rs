//! Seed-chunk graft generator: turn a target wordlist into rarity-weighted
//! candidate combinations.
//!
//! Pipeline: for each wordlist entry (seed) we compute its **surprisal**
//! (`-log P_model(seed)`) as a rarity signal, pick **anchor chunks** (the whole
//! seed + single-token distinctive sub-chunks), and combine each anchor with an
//! **affix pool** (model-probable tokens + other seeds' distinctive chunks).
//! Budget is allocated ∝ surprisal at the seed AND chunk level, and blocks are
//! emitted rarest-seed-first. No per-candidate scoring — order comes from
//! rarity-first blocks + a frequency-ordered affix pool.
//!
//! This module is OPT-IN, via `--float` or `--prepend-only`. Bare `--wordlist`
//! (and its explicit spelling `--append-only`) runs the Markov enumerator's
//! seeded path instead, NOT this module. It is fully isolated from the
//! enumerator (it does not touch the seeds/bias/heap paths), so standard mode
//! and the legacy `weighted|seeded|combined` modes stay byte-identical.

use crate::case_mask::{apply_op, CaseMask};
use crate::variant::EnumModel;
use crate::{Ctx, GenerateArgs, Model};
use anyhow::Result;

#[inline]
fn pack(a: u32, b: u32) -> Ctx {
    ((a as u64) << 32) | (b as u64)
}

/// Floor log-prob for a transition unreachable through every KN tier. Keeps any
/// sequence (incl. never-seen OSINT terms) scorable → high surprisal → max
/// budget, never dropped. ~exp(-35) is far below any real token prob.
const UNI_FLOOR: f32 = -35.0;

/// `log P(t | a, b)` with 3-tier KN back-off: trigram → bigram → unigram floor.
/// Mirrors `build_seeds`' `log_p_step` but adds the unigram tier so it is total
/// (never `None`) — the "no gate" rule: rank everything, exclude nothing.
fn log_p_step(em: &EnumModel, uni_cont: &[f32], a: u32, b: u32, t: u32) -> f32 {
    let ctx = pack(a, b);
    if let Some(children) = em.trigram.get(&ctx) {
        if let Some(lp) = children.iter().find(|(id, _)| *id == t).map(|(_, lp)| *lp) {
            return lp;
        }
        if em.is_kn {
            let log_lam = em.log_lambda.get(&ctx).copied().unwrap_or(0.0);
            if let Some(bi) = em.bigram.get(&b) {
                if let Some(lp) = bi.iter().find(|(id, _)| *id == t).map(|(_, lp)| *lp) {
                    return log_lam + lp;
                }
            }
        }
    } else if em.is_kn {
        if let Some(bi) = em.bigram.get(&b) {
            if let Some(lp) = bi.iter().find(|(id, _)| *id == t).map(|(_, lp)| *lp) {
                return lp;
            }
        }
    }
    uni_cont
        .get(t as usize)
        .copied()
        .filter(|p| *p > 0.0)
        .map(|p| p.ln())
        .unwrap_or(UNI_FLOOR)
}

/// Joint log-prob of a token sequence from the START,START context.
fn seq_logprob(em: &EnumModel, uni_cont: &[f32], start_id: u32, seq: &[u32]) -> f32 {
    let (mut a, mut b) = (start_id, start_id);
    let mut lp = 0.0;
    for &t in seq {
        lp += log_p_step(em, uni_cont, a, b, t);
        a = b;
        b = t;
    }
    lp
}

/// Rarity signal: `-log P_model(seq)`. Rare/novel → high.
fn surprisal(em: &EnumModel, uni_cont: &[f32], start_id: u32, seq: &[u32]) -> f32 {
    -seq_logprob(em, uni_cont, start_id, seq)
}

fn decode_ids(decode: &[Vec<u8>], ids: &[u32]) -> String {
    let mut b = Vec::new();
    for &id in ids {
        if let Some(bytes) = decode.get(id as usize) {
            b.extend_from_slice(bytes);
        }
    }
    String::from_utf8_lossy(&b).into_owned()
}

/// A token is a usable affix if it decodes to clean ASCII and is either all
/// digits (classic suffixes 1/12/2024) or ≥3 chars (real words, not BPE
/// fragments like `j`/`ec`).
fn good_affix(decode: &[Vec<u8>], t: u32) -> bool {
    let s = decode_ids(decode, &[t]);
    if s.is_empty() || !s.is_ascii() { return false; }
    if s.chars().any(|c| c.is_ascii_control() || c == ' ') { return false; }
    s.chars().all(|c| c.is_ascii_digit()) || s.chars().count() >= 3
}

/// Minimum decoded length (chars) for a sub-chunk to stand on its own as an anchor.
const MIN_CHUNK_CHARS: usize = 4;
/// Minimum decoded length (chars) required of EVERY piece of a seed before we
/// trust the tokenizer's split as a word boundary.
const MIN_PIECE_CHARS: usize = 3;

/// Anchor chunks for a seed: the whole seed (always), plus single-token
/// sub-chunks — but only when the tokenizer split the seed at what look like
/// word boundaries.
///
/// A sub-chunk only earns a guess if it is a word in its own right: `johnsmith`
/// → `john`/`smith` and `northwind` → `north`/`wind` are worth cracking with,
/// `cisco` → `isco` and `summer` → `sum`/`mer` are not. BPE splits an atomic
/// word mid-morpheme, and the tell is a SHORT piece — `cisco` is `c`+`isco`,
/// `summer` is `sum`+`mer`, `acmecorp` is `ac`+`m`+`ec`+`orp`, whereas
/// `johnsmith` is `john`+`smith`. So require every piece to reach
/// MIN_PIECE_CHARS before trusting any of them, then keep the pieces that are
/// themselves >= MIN_CHUNK_CHARS.
///
/// Rarity deliberately does NOT gate admission. The rarest pieces are exactly
/// the mid-morpheme splinters, so the old `surprisal >= 0.6 * whole` test kept
/// `isco` and threw away `wind`. Surprisal still orders the chunks and sizes
/// their budgets in `run` — it just no longer decides membership.
fn anchor_chunks(decode: &[Vec<u8>], seq: &[u32]) -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> = vec![seq.to_vec()];
    if seq.len() < 2 { return out; }
    let piece_chars = |t: u32| decode_ids(decode, &[t]).chars().count();
    if !seq.iter().all(|&t| piece_chars(t) >= MIN_PIECE_CHARS) { return out; }
    for &t in seq {
        if piece_chars(t) < MIN_CHUNK_CHARS { continue; }
        let chunk = vec![t];
        if !out.contains(&chunk) { out.push(chunk); }
    }
    out
}

/// Build the prefix and suffix affix pools from the model:
///   prefix = most-probable password-leading tokens = children of (START,START)
///   suffix = most-frequent tokens overall (raw counts) — digits/specials land here
/// Both filtered by `good_affix` and capped at `n`.
fn build_pools(em: &EnumModel, model: &Model, start_id: u32, n: usize) -> (Vec<u32>, Vec<u32>) {
    // Prefix: P(t | START,START) from the trigram start context.
    let mut prefix: Vec<u32> = Vec::new();
    if let Some(children) = em.trigram.get(&pack(start_id, start_id)) {
        let mut c: Vec<(u32, f32)> = children.clone();
        c.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (t, _) in c {
            if good_affix(&model.decode, t) {
                prefix.push(t);
            }
            if prefix.len() >= n { break; }
        }
    }
    // Suffix: rank by raw unigram count desc.
    let mut idx: Vec<u32> = (0..model.unigram_raw.len() as u32).collect();
    idx.sort_by(|&a, &b| model.unigram_raw[b as usize].cmp(&model.unigram_raw[a as usize]));
    let mut suffix: Vec<u32> = Vec::new();
    for t in idx {
        if good_affix(&model.decode, t) {
            suffix.push(t);
        }
        if suffix.len() >= n { break; }
    }
    (prefix, suffix)
}

/// Allocate a total budget across weights proportionally (∝ weight), min 1 each.
///
/// Floors each share and then hands the rounding remainder to the largest
/// fractional parts (largest-remainder apportionment), so the budgets sum to
/// exactly `total` rather than drifting short — rounding each share
/// independently left `--count 500` over 200 seeds allocating only 498.
///
/// The min-1 floor still makes the sum EXCEED `total` when there are more
/// weights than budget (200 seeds, `--count 10`). That is what the emitter's
/// global `--count` stop is for; allocation alone cannot express it.
fn alloc_budget(weights: &[f32], total: u64) -> Vec<u64> {
    let n = weights.len();
    if n == 0 { return Vec::new(); }
    let sum: f64 = weights.iter().map(|w| *w as f64).sum::<f64>().max(1e-9);
    let exact: Vec<f64> = weights.iter().map(|w| (total as f64) * (*w as f64) / sum).collect();
    let mut out: Vec<u64> = exact.iter().map(|e| e.floor().max(1.0) as u64).collect();
    let assigned: u64 = out.iter().sum();
    let mut rem = total.saturating_sub(assigned);
    if rem == 0 { return out; }
    // Largest fractional part first; ties keep input order (stable sort).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let (fa, fb) = (exact[a] - exact[a].floor(), exact[b] - exact[b].floor());
        fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
    });
    // One extra each, largest remainder first. `rem < n` after a floor pass
    // unless min-1 clamping consumed the budget, so spread any surplus evenly.
    let each = rem / n as u64;
    if each > 0 {
        for v in out.iter_mut() { *v += each; }
        rem -= each * n as u64;
    }
    for &i in order.iter().take(rem as usize) {
        out[i] += 1;
    }
    out
}

/// Streaming candidate writer with decoded-string dedup (first-occurrence), a
/// length filter, the `--count` ceiling, and the two post-decode shapers
/// (`--case-shape`, `--enterprise`). Owns the Sink so the emission loop stays
/// borrow-clean.
struct Emitter<'a> {
    sink: crate::Sink,
    seen: rustc_hash::FxHashSet<String>,
    decode: &'a [Vec<u8>],
    min_len: usize,
    max_len: usize,
    min_tokens: usize,
    /// `--count` ceiling. `u64::MAX` when the flag is omitted (emit everything).
    /// Enforced HERE as well as in the loops: the per-seed/per-chunk budgets are
    /// proportional allocations with a min of 1 each, so on a wordlist with more
    /// entries than `--count` they sum well past it. This is the hard stop.
    cap: u64,
    /// `--case-shape` patterns. Empty = one plain emission per candidate.
    masks: &'a [CaseMask],
    /// `--enterprise`: policy gate + capitalize-first repair.
    enterprise: bool,
    emitted: u64,
    dupes: u64,
}

impl<'a> Emitter<'a> {
    #[inline]
    fn full(&self) -> bool { self.emitted >= self.cap }

    /// Decode, shape, length-check, dedup, write. Returns Ok(true) if at least
    /// one NEW line was written, Ok(false) if filtered/duplicate/at-cap (so the
    /// caller doesn't count it against the budget — budget is a target of
    /// distinct emitted lines).
    fn emit(&mut self, cand: &[u32]) -> Result<bool> {
        if self.full() { return Ok(false); }
        // Token-count floor (--min-tokens): drop grafts built from too few tokens.
        if cand.len() < self.min_tokens {
            return Ok(false);
        }
        if self.masks.is_empty() {
            let s = decode_ids(self.decode, cand);
            return self.emit_one(s);
        }
        // Shape-conditioned path: one emission per case mask, each slot's op
        // applied to its own token's bytes (the token is the unit of casing) —
        // same contract as the enumerator's --case-shape path.
        let masks = self.masks;
        let mut wrote = false;
        for mask in masks {
            if self.full() { break; }
            let s = self.render(cand, mask);
            if self.emit_one(s)? { wrote = true; }
        }
        Ok(wrote)
    }

    /// Decode `cand` applying `mask`'s per-token case op.
    fn render(&self, cand: &[u32], mask: &CaseMask) -> String {
        let mut b = Vec::new();
        for (pos, &id) in cand.iter().enumerate() {
            if let Some(bytes) = self.decode.get(id as usize) {
                let start = b.len();
                b.extend_from_slice(bytes);
                let end = b.len();
                apply_op(&mut b[start..end], mask.op_for(pos));
            }
        }
        String::from_utf8_lossy(&b).into_owned()
    }

    fn emit_one(&mut self, s: String) -> Result<bool> {
        let len = s.chars().count();
        if len < self.min_len || len > self.max_len {
            return Ok(false);
        }
        // --enterprise: emit only policy-compliant candidates, capitalizing the
        // first byte when that is the sole missing class. Dropped candidates do
        // NOT count toward `emitted`, matching the enumerator's L0 path.
        let s = if self.enterprise {
            match crate::enterprise::decide(s.as_bytes()) {
                crate::enterprise::Decision::AsIs => s,
                crate::enterprise::Decision::Cap => {
                    let mut b = s.into_bytes();
                    b[0] = b[0].to_ascii_uppercase();
                    String::from_utf8_lossy(&b).into_owned()
                }
                crate::enterprise::Decision::Drop => return Ok(false),
            }
        } else {
            s
        };
        if self.seen.contains(&s) {
            self.dupes += 1;
            return Ok(false);
        }
        self.sink.write_line(s.as_bytes())?;
        self.seen.insert(s);
        self.emitted += 1;
        Ok(true)
    }
}

/// Entry point for the default --wordlist graft generator. Emits rarity-weighted combinations:
/// rarest seed first; per-seed and per-chunk budget ∝ surprisal; each anchor
/// combined with the affix pool (+ other seeds' distinctive chunks for mixing).
pub fn run(args: &GenerateArgs, em: &EnumModel, model: &Model, entry_seqs: &[Vec<u32>],
           case_masks: &[CaseMask]) -> Result<()> {
    let start_id = model.start_id;
    let uni = &model.unigram_kn_cont;
    let decode = &model.decode;
    let prepend_only = args.prepend_only;

    // Per-seed: surprisal + anchor chunks. Sort rarest-first.
    let mut seeds: Vec<(f32, Vec<Vec<u32>>)> = entry_seqs
        .iter()
        .map(|s| (surprisal(em, uni, start_id, s), anchor_chunks(decode, s)))
        .collect();
    seeds.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Affix pieces: model-probable tokens (single-token) + every seed's
    // distinctive anchor chunks (multi-token allowed) for wordlist mixing.
    // NOT raw seed sub-tokens — that reintroduces fragment junk (`loveorp`).
    let (prefix_toks, suffix_toks) = build_pools(em, model, start_id, 250);
    let mut mixing: Vec<Vec<u32>> = Vec::new();
    for (_, chunks) in &seeds {
        for c in chunks {
            if !mixing.contains(c) {
                mixing.push(c.clone());
            }
        }
    }
    let prefix_pieces: Vec<Vec<u32>> =
        prefix_toks.iter().map(|&t| vec![t]).chain(mixing.iter().cloned()).collect();
    let suffix_pieces: Vec<Vec<u32>> =
        suffix_toks.iter().map(|&t| vec![t]).chain(mixing.iter().cloned()).collect();

    crate::log_msg(&format!(
        "[graft] {} seeds, {} prefix + {} suffix affix pieces, prepend_only={}",
        seeds.len(), prefix_pieces.len(), suffix_pieces.len(), prepend_only
    ));

    // Seed budgets ∝ surprisal (∞ / emit-all when --count omitted).
    let weights: Vec<f32> = seeds.iter().map(|(s, _)| *s).collect();
    let seed_budgets: Vec<u64> = match args.count {
        Some(total) => alloc_budget(&weights, total),
        None => vec![u64::MAX; seeds.len()],
    };

    let mut em_out = Emitter {
        sink: crate::Sink::open(args.output.as_deref())?,
        seen: rustc_hash::FxHashSet::default(),
        decode,
        min_len: args.min_len,
        max_len: args.max_len,
        min_tokens: args.min_tokens,
        cap: args.count.unwrap_or(u64::MAX),
        masks: case_masks,
        enterprise: args.enterprise,
        emitted: 0,
        dupes: 0,
    };

    for (si, (_, chunks)) in seeds.iter().enumerate() {
        if em_out.full() { break; }
        let sb = seed_budgets[si];
        // Chunk order + budget: rarest chunk first (whole seed = rarest → leads
        // its block), budget ∝ chunk surprisal (recursive rarity).
        let chunk_surs: Vec<f32> = chunks.iter().map(|c| surprisal(em, uni, start_id, c)).collect();
        let mut order: Vec<usize> = (0..chunks.len()).collect();
        order.sort_by(|&a, &b| chunk_surs[b].partial_cmp(&chunk_surs[a]).unwrap_or(std::cmp::Ordering::Equal));
        let ordered_surs: Vec<f32> = order.iter().map(|&i| chunk_surs[i]).collect();
        let chunk_budgets: Vec<u64> = if sb == u64::MAX {
            vec![u64::MAX; chunks.len()]
        } else {
            alloc_budget(&ordered_surs, sb)
        };

        let mut seed_emitted: u64 = 0;
        for (rank, &ci) in order.iter().enumerate() {
            if seed_emitted >= sb || em_out.full() { break; }
            let anchor = &chunks[ci];
            let cb = chunk_budgets[rank].min(sb.saturating_sub(seed_emitted));
            let mut ce: u64 = 0;

            // Emission order within a chunk (affixes iterated in pool = frequency
            // order): bare anchor → append → prepend → both. prepend_only skips
            // the append/both passes.
            let mut buf: Vec<u32> = Vec::with_capacity(16);
            // 1. bare anchor
            if ce < cb && em_out.emit(anchor)? { ce += 1; seed_emitted += 1; }
            // 2. anchor + suffix
            if !prepend_only {
                'suf: for s in &suffix_pieces {
                    if ce >= cb || em_out.full() { break; }
                    buf.clear(); buf.extend_from_slice(anchor); buf.extend_from_slice(s);
                    if em_out.emit(&buf)? { ce += 1; seed_emitted += 1; }
                    if seed_emitted >= sb { break 'suf; }
                }
            }
            // 3. prefix + anchor
            'pre: for p in &prefix_pieces {
                if ce >= cb || em_out.full() { break; }
                buf.clear(); buf.extend_from_slice(p); buf.extend_from_slice(anchor);
                if em_out.emit(&buf)? { ce += 1; seed_emitted += 1; }
                if seed_emitted >= sb { break 'pre; }
            }
            // 4. prefix + anchor + suffix
            if !prepend_only {
                'both: for p in &prefix_pieces {
                    if ce >= cb || em_out.full() { break; }
                    for s in &suffix_pieces {
                        if ce >= cb || em_out.full() { break; }
                        buf.clear();
                        buf.extend_from_slice(p); buf.extend_from_slice(anchor); buf.extend_from_slice(s);
                        if em_out.emit(&buf)? { ce += 1; seed_emitted += 1; }
                        if seed_emitted >= sb { break 'both; }
                    }
                }
            }
        }
    }

    em_out.sink.finish()?;
    crate::log_msg(&format!(
        "[graft] emitted {} candidates ({} duplicates skipped)",
        em_out.emitted, em_out.dupes
    ));
    // --enterprise drops everything that cannot reach 3 character classes, and
    // the graft affix pools are finite — so a policy run can legitimately come
    // up short, or empty. Say so: silently emitting nothing looks like a crash.
    // (The enumerator has no such ceiling; it just enumerates deeper.) Fires
    // with or without --count, since `--float --enterprise` and no budget
    // emitting nothing is the very case this exists to explain.
    let policy_shortfall = match args.count {
        Some(target) => em_out.emitted < target,
        None        => em_out.emitted == 0,
    };
    if args.enterprise && policy_shortfall {
        let wanted = match args.count {
            Some(t) => format!(" of the {t} requested"),
            None    => String::new(),
        };
        let why = if prepend_only {
            "--prepend-only only adds a PREFIX, so most candidates never reach 3 of \
             {lower, upper, digit, special}. "
        } else { "" };
        crate::warn_msg(&format!(
            "warning: --enterprise kept only {}{} candidates — the graft affix pools \
             ran out of policy-compliant combinations. {}Use the default --wordlist \
             enumerator (drop {}) to keep enumerating deeper until the budget is met, \
             or widen the pools with more seeds.",
            em_out.emitted, wanted, why,
            if prepend_only { "--prepend-only" } else { "--float" }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a decode table + token sequence from the pieces a tokenizer would
    /// produce, so the admission rule can be tested on real BPE splits. Equal
    /// pieces share one id, as in a real vocabulary.
    fn seed(pieces: &[&str]) -> (Vec<Vec<u8>>, Vec<u32>) {
        let mut decode: Vec<Vec<u8>> = Vec::new();
        let mut seq: Vec<u32> = Vec::new();
        for p in pieces {
            let b = p.as_bytes().to_vec();
            let id = decode.iter().position(|d| *d == b).unwrap_or_else(|| {
                decode.push(b);
                decode.len() - 1
            });
            seq.push(id as u32);
        }
        (decode, seq)
    }

    fn chunks_of(pieces: &[&str]) -> Vec<String> {
        let (decode, seq) = seed(pieces);
        anchor_chunks(&decode, &seq).iter().map(|c| decode_ids(&decode, c)).collect()
    }

    // A seed BPE split mid-morpheme leaves a short piece; none of its pieces
    // may become an anchor. These are the real tokenov_v1 splits.
    #[test]
    fn atomic_words_yield_no_sub_chunks() {
        assert_eq!(chunks_of(&["c", "isco"]), vec!["cisco"]);           // `isco` is junk
        assert_eq!(chunks_of(&["sum", "mer"]), vec!["summer"]);         // 3-char pieces
        assert_eq!(chunks_of(&["ac", "m", "ec", "orp"]), vec!["acmecorp"]);
        assert_eq!(chunks_of(&["tes", "la"]), vec!["tesla"]);
    }

    // A compound split at word boundaries contributes each word-sized piece.
    #[test]
    fn compound_words_yield_word_sub_chunks() {
        assert_eq!(chunks_of(&["john", "smith"]), vec!["johnsmith", "john", "smith"]);
        assert_eq!(chunks_of(&["north", "wind"]), vec!["northwind", "north", "wind"]);
    }

    #[test]
    fn single_token_seed_yields_only_itself() {
        assert_eq!(chunks_of(&["microsoft"]), vec!["microsoft"]);
    }

    // The whole seed always leads, so it keeps the budget its rarity earned.
    #[test]
    fn whole_seed_is_always_first() {
        for pieces in [vec!["john", "smith"], vec!["c", "isco"], vec!["microsoft"]] {
            assert_eq!(chunks_of(&pieces)[0], pieces.concat());
        }
    }

    // A repeated piece must not produce a duplicate anchor.
    #[test]
    fn repeated_piece_is_deduped() {
        assert_eq!(chunks_of(&["bank", "bank"]), vec!["bankbank", "bank"]);
    }

    #[test]
    fn alloc_budget_is_proportional_and_never_zero() {
        let b = alloc_budget(&[3.0, 1.0], 100);
        assert_eq!(b, vec![75, 25]);
        // Min 1 each: this is why `--count` needs a separate global stop (the
        // sum here is 4, not 2).
        let b = alloc_budget(&[1.0, 1.0, 1.0, 1.0], 2);
        assert_eq!(b, vec![1, 1, 1, 1]);
        assert!(alloc_budget(&[0.0, 0.0], 10).iter().all(|&x| x >= 1));
    }

    // The case that made `--count 500` emit 498: equal shares that each round
    // down. Largest-remainder must put the budget back.
    #[test]
    fn alloc_budget_sums_to_total_exactly() {
        for (n, total) in [(200usize, 500u64), (3, 10), (7, 1000), (1, 1), (13, 13)] {
            let w = vec![1.0f32; n];
            let got: u64 = alloc_budget(&w, total).iter().sum();
            assert_eq!(got, total, "n={n} total={total}");
        }
        // Uneven weights still sum exactly.
        let got: u64 = alloc_budget(&[5.0, 3.0, 1.1, 0.4], 997).iter().sum();
        assert_eq!(got, 997);
    }
}
