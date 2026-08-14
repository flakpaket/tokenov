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
//! This module is the default for `--wordlist` (no --mode) and is fully isolated from the
//! Markov enumerator (it does not touch the seeds/bias/heap paths), so standard
//! mode and the legacy `weighted|seeded|combined` modes stay byte-identical.

use crate::enterprise::{Decision, LengthBounds};
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

/// Anchor chunks for a seed: the whole seed (always) + single-token sub-chunks
/// that are distinctive (decoded len ≥ K chars AND surprisal ≥ THRESH × the
/// whole seed's). Single-token is load-bearing: it keeps `cisco`'s `isco` but
/// rejects the multi-token splices (`ecorp`, `mecorp`) that shatter a messy
/// unique term like `acmecorp`.
fn anchor_chunks(
    em: &EnumModel,
    uni_cont: &[f32],
    decode: &[Vec<u8>],
    start_id: u32,
    seq: &[u32],
    k: usize,
    thresh: f32,
) -> Vec<Vec<u32>> {
    let whole_sur = surprisal(em, uni_cont, start_id, seq);
    let mut out: Vec<Vec<u32>> = vec![seq.to_vec()];
    if seq.len() > 1 {
        for &t in seq {
            if decode_ids(decode, &[t]).chars().count() < k {
                continue;
            }
            if surprisal(em, uni_cont, start_id, &[t]) >= thresh * whole_sur {
                let chunk = vec![t];
                if !out.contains(&chunk) {
                    out.push(chunk);
                }
            }
        }
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
fn alloc_budget(weights: &[f32], total: u64) -> Vec<u64> {
    let sum: f64 = weights.iter().map(|w| *w as f64).sum::<f64>().max(1e-9);
    weights
        .iter()
        .map(|w| ((total as f64) * (*w as f64) / sum).round().max(1.0) as u64)
        .collect()
}

/// Streaming candidate writer with decoded-string dedup (first-occurrence) and
/// a length filter. Owns the Sink so the emission loop stays borrow-clean.
struct Emitter<'a> {
    sink: crate::Sink,
    seen: rustc_hash::FxHashSet<String>,
    decode: &'a [Vec<u8>],
    min_len: usize,
    max_len: usize,
    min_tokens: usize,
    enterprise_bounds: Option<LengthBounds>,
    emitted: u64,
    dupes: u64,
}

impl<'a> Emitter<'a> {
    /// Decode, length-check, dedup, write. Returns Ok(true) if a NEW line was
    /// written, Ok(false) if filtered/duplicate (so the caller doesn't count it
    /// against the budget — budget is a target of distinct emitted lines).
    fn emit(&mut self, cand: &[u32]) -> Result<bool> {
        // Token-count floor (--min-tokens): drop grafts built from too few tokens.
        if cand.len() < self.min_tokens {
            return Ok(false);
        }
        let mut bytes = decode_ids(self.decode, cand).into_bytes();
        let len = bytes.len();
        if len < self.min_len || len > self.max_len {
            return Ok(false);
        }

        if let Some(bounds) = self.enterprise_bounds {
            match crate::enterprise::decide_with_bounds(&bytes, bounds) {
                Decision::AsIs => {}
                Decision::Cap => {
                    bytes[0] = bytes[0].to_ascii_uppercase();
                }
                Decision::Drop => return Ok(false),
            }
        }

        // Repair before dedup so repaired collisions are emitted only once.
        let s = String::from_utf8(bytes).expect("decode_ids returned valid UTF-8");
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
pub fn run(
    args: &GenerateArgs,
    em: &EnumModel,
    model: &Model,
    entry_seqs: &[Vec<u32>],
    enterprise_bounds: Option<LengthBounds>,
) -> Result<()> {
    let start_id = model.start_id;
    let uni = &model.unigram_kn_cont;
    let decode = &model.decode;
    const K: usize = 2;
    const THRESH: f32 = 0.6;
    let prepend_only = args.prepend_only;

    // Per-seed: surprisal + anchor chunks. Sort rarest-first.
    let mut seeds: Vec<(f32, Vec<Vec<u32>>)> = entry_seqs
        .iter()
        .map(|s| (surprisal(em, uni, start_id, s), anchor_chunks(em, uni, decode, start_id, s, K, THRESH)))
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
        enterprise_bounds,
        emitted: 0,
        dupes: 0,
    };

    for (si, (_, chunks)) in seeds.iter().enumerate() {
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
            if seed_emitted >= sb { break; }
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
                    if ce >= cb { break; }
                    buf.clear(); buf.extend_from_slice(anchor); buf.extend_from_slice(s);
                    if em_out.emit(&buf)? { ce += 1; seed_emitted += 1; }
                    if seed_emitted >= sb { break 'suf; }
                }
            }
            // 3. prefix + anchor
            'pre: for p in &prefix_pieces {
                if ce >= cb { break; }
                buf.clear(); buf.extend_from_slice(p); buf.extend_from_slice(anchor);
                if em_out.emit(&buf)? { ce += 1; seed_emitted += 1; }
                if seed_emitted >= sb { break 'pre; }
            }
            // 4. prefix + anchor + suffix
            if !prepend_only {
                'both: for p in &prefix_pieces {
                    if ce >= cb { break; }
                    for s in &suffix_pieces {
                        if ce >= cb { break; }
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
    Ok(())
}
