//! Wordlist targeting (`generate --wordlist`): the rarity-first hybrid generator.
//!
//! Every wordlist entry becomes an **anchor** — a locked token sequence that
//! appears verbatim in each candidate built from it. Candidates are
//! `[prefix] + anchor + [suffix]`, where prefix and suffix are walked by the
//! same Markov model the standard enumerator uses, so both sides go to any
//! depth (`cisco2024!`, `ilovecisco`) instead of drawing from a fixed pool.
//!
//! Ordering:
//!
//! * **Conditional cost.** A candidate is charged for its affixes, not for the
//!   anchor itself: suffix tokens cost `-log P(token | context)` as usual, the
//!   prefix costs `-log P(prefix)` from START, and the seam costs only how much
//!   less likely the anchor became after that prefix than at the start of a
//!   password (`junction`, clipped at 0). A common word gets no head start just
//!   for being common.
//! * **One stream per anchor.** Each anchor is its own resumable level sweep
//!   over that conditional cost, so within an anchor candidates come out in
//!   probability order (`cisco`, `cisco1`, `cisco123`, …).
//! * **Rarity-weighted interleave.** A weighted-fair scheduler merges the
//!   streams: it always advances the stream furthest behind its share, where an
//!   anchor's share ∝ `S^p` (`S = -log P_model(anchor)`, its surprisal; `p` =
//!   `--rarity-weight`). Rare, target-specific words get most of the output and
//!   lead it; common words still appear from the start. Within one worker the
//!   shares hold at every point of its stream, with or without `--count`; across
//!   fast-mode workers lines arrive in throughput order. (A fixed per-anchor level
//!   offset was tried first and measured 86 % for one rare word where its share
//!   was 56 %: a rare anchor's sparse back-off contexts grow its candidate count
//!   per level much faster than a common anchor's, so level order alone cannot
//!   hold a share.)
//! * **Per-anchor units.** Fast mode deals anchors round-robin (rarest first)
//!   across threads, runs the same scheduler inside each thread, and gives each
//!   thread a `--count` quota ∝ the summed `S^p` of its anchors, so the split
//!   survives the per-thread quota.
//!
//! `--append-only` / `--prepend-only` narrow the walk to one side, `--mix` lets
//! another wordlist entry stand in as a single affix piece, and word splitting
//! (`johnsmith` → `john`, `smith`; off with `--no-split`) adds a compound's
//! word-sized pieces as anchors of their own.
//!
//! Isolated from `enumerate_to_sink`, which dispatches here only when a plan is
//! present, so standard mode and the legacy `--mode` paths are untouched.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use rustc_hash::FxHashMap;

use crate::case_mask::{apply_op, CaseMask};
use crate::variant::{EnumModel, Variant};
use crate::{
    enterprise, finalize_decoded, finalized_range, lp_to_level, pack, BoundedChildCache,
    CacheMode, ChildCache, Children, Ctx, DecodeKind, HeapEntry, Model, ThreadCkpt,
};

/// Level charged for a `--mix` piece (another wordlist entry used as an affix).
/// ≈ ln(250): the same priority the old graft generator gave mixed pieces, which
/// it queued behind a 250-entry affix pool.
pub const MIX_LEVEL: u32 = 6;

/// Ceiling on the junction penalty, in levels. A word that almost only ever
/// starts a password (`Community` — capitalized words) pays a large junction
/// after nearly every prefix, and the prefix walk must visit every stem below
/// the target to find the few that fit. Capped, prefixes the word fits after
/// still rank first; the rest are ordered by their own probability, and the
/// walk wastes less (measured on a 1.8k-word web list, 8 threads: 680k lines/s
/// capped at 3 vs 480k/s uncapped).
const JUNCTION_CAP: u32 = 3;

/// DFS steps one stream may take per scheduler turn before yielding without a
/// line (~0.1 s). A yielded turn is charged like a line, so the scheduler
/// shares *work* too: a stream that needs a million steps per line can no
/// longer starve the rest of its worker.
const TURN_STEPS: u64 = 1 << 20;

/// A stream that takes this many DFS steps (~2 min) without writing a single
/// line is dropped with a warning — everything it can still reach is filtered
/// out (length, `--enterprise`, `--min-tokens`) or unreachably deep. Counted
/// in steps, not seconds, so `--strict` stays byte-reproducible.
const DEAD_STEPS: u64 = 1 << 30;

/// Outcome of one scheduler turn for one stream.
enum Turn {
    /// Wrote this many lines (several only under `--case-shape`).
    Lines(u64),
    /// Spent `TURN_STEPS` without a line; position kept for the next turn.
    Yield,
    /// The stream's whole space is exhausted.
    Done,
}

/// Floor log-prob for a token unreachable through every KN tier. Keeps any
/// sequence (incl. never-seen OSINT terms) scorable → high surprisal, never
/// dropped. ~exp(-35) is far below any real token prob.
const UNI_FLOOR: f32 = -35.0;

/// Which side(s) of the anchor may carry affixes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// Prefix and/or suffix (the default).
    Float,
    /// Suffix only — the anchor always leads.
    Append,
    /// Prefix only — the anchor always ends the candidate.
    Prepend,
}

pub struct Anchor {
    pub toks:      Vec<u32>,
    /// `-log P_model(anchor)` in nats — the rarity signal.
    pub surprisal: f32,
    /// `surprisal^p` — this anchor's share weight (scheduler + thread quotas).
    pub weight:    f64,
    /// Σ level of the anchor's tokens walked from START: the junction baseline.
    start_levels:  u32,
    /// Lower bound on the levels any candidate still owes once the anchor is
    /// placed. Exact for a multi-token anchor, whose following context is its
    /// own last two tokens wherever it sits; for a single-token anchor, the
    /// minimum over every context ending in it. Can be 0: a step the model
    /// gives probability 1 (END after `Resources`) costs no level. Lets the
    /// prefix walk stop at stems that cannot reach the current level instead
    /// of re-walking them every pass.
    tail_min:      u32,
    /// Token cap for candidates on this anchor: `--max-tokens`, raised to the
    /// anchor's own length so an over-long anchor is still emitted bare.
    pub cap:       usize,
    /// Stand-in continuations for right after the anchor, set only when the
    /// model has none there — the anchor ends in a token it never saw followed
    /// by anything (common for capitalized web words: `Government`). Without
    /// it no candidate on the anchor can ever end, and the prefix walk deepens
    /// forever looking for one. Taken from the anchor's lowercase form when
    /// that one is known (`government`), else just END (bare / prefixed only).
    post:          Option<Arc<[(u32, f32)]>>,
}

pub struct Plan {
    /// Sorted rarest-first (surprisal desc, then input order).
    pub anchors: Vec<Anchor>,
    pub side:    Side,
    pub mix:     bool,
    /// `unigram_kn_cont`, the last-resort tier of the junction lookup.
    uni_cont:    Vec<f32>,
    end_id:      u32,
}

pub struct PlanOpts {
    pub side:          Side,
    pub mix:           bool,
    pub split:         bool,
    pub rarity_weight: f32,
    pub max_tokens:    usize,
}

// ============================================================================
// Rarity — model surprisal of a token sequence
// ============================================================================

/// `log P(t | a, b)` with 3-tier KN back-off: trigram → bigram → unigram floor.
/// Total (never `None`): rank everything, exclude nothing. Reads
/// `em.trigram`, so it is only valid before the FULL-cache trigram drop —
/// `build_plan` runs there.
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
    uni_lp(uni_cont, t)
}

#[inline]
fn uni_lp(uni_cont: &[f32], t: u32) -> f32 {
    uni_cont.get(t as usize).copied().filter(|p| *p > 0.0).map(|p| p.ln()).unwrap_or(UNI_FLOOR)
}

/// Rarity signal: `-log P_model(seq)` from the START,START context.
fn surprisal(em: &EnumModel, uni_cont: &[f32], start_id: u32, seq: &[u32]) -> f32 {
    let (mut a, mut b) = (start_id, start_id);
    let mut lp = 0.0;
    for &t in seq {
        lp += log_p_step(em, uni_cont, a, b, t);
        a = b;
        b = t;
    }
    -lp
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
/// Length alone is not enough on long words: `Providers` is `Prov`+`iders` and
/// `protection` is `prot`+`ection`, every piece 4+ chars. So every piece must
/// also be something passwords are seen to start with (`starts_word`). A
/// suffix fragment like `iders` never starts one, and because the model finds
/// it so improbable, rarity weighting would otherwise put it at the head of the
/// stream. If any piece fails, none is used — `Prov` from `Providers` is no
/// more a word than `iders` is.
fn anchor_chunks(decode: &[Vec<u8>], seq: &[u32], starts_word: &dyn Fn(u32) -> bool)
                 -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> = vec![seq.to_vec()];
    if seq.len() < 2 { return out; }
    let piece_chars = |t: u32| decode_ids(decode, &[t]).chars().count();
    if !seq.iter().all(|&t| piece_chars(t) >= MIN_PIECE_CHARS) { return out; }
    if !seq.iter().all(|&t| starts_word(t)) { return out; }
    for &t in seq {
        if piece_chars(t) < MIN_CHUNK_CHARS { continue; }
        let chunk = vec![t];
        if !out.contains(&chunk) { out.push(chunk); }
    }
    out
}

/// Allocate a total budget across weights proportionally (∝ weight), min 1 each.
///
/// Floors each share and hands the rounding remainder to the largest fractional
/// parts (largest-remainder apportionment), so the budgets sum to exactly
/// `total` — unless the min-1 floor pushes it over, when there are more weights
/// than budget. `u64::MAX` (no `--count`) passes through to every share.
pub fn alloc_budget(weights: &[f64], total: u64) -> Vec<u64> {
    let n = weights.len();
    if n == 0 { return Vec::new(); }
    if total == u64::MAX { return vec![u64::MAX; n]; }
    let sum: f64 = weights.iter().sum::<f64>().max(1e-9);
    let exact: Vec<f64> = weights.iter().map(|w| (total as f64) * w / sum).collect();
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

// ============================================================================
// Plan — anchors, rarity offsets, work units
// ============================================================================

/// Find `t` among `children` (the variant's ranked continuations of a context);
/// fall back to the unigram tier when the variant's list leaves it out.
/// Only ever reads data that is resident in every cache mode, so the junction —
/// and therefore the output — is identical across FULL / BOUNDED / LAZY.
#[inline]
fn child_lp(children: &[(u32, f32)], uni_cont: &[f32], t: u32) -> f32 {
    children.iter().find(|(id, _)| *id == t).map(|(_, lp)| *lp)
        .unwrap_or_else(|| uni_lp(uni_cont, t))
}

/// Levels of the cheapest move from a context with children `ch` once the
/// anchor is placed: END for prepend-only (none → unreachable), else the first
/// (cheapest, lists are descending-lp) child, or a mix piece. May be 0: a
/// step with P = 1 costs no level, and assuming ≥ 1 there pruned candidates at
/// their own level and skipped them as already emitted at every later one.
fn tail_of(ch: &[(u32, f32)], side: Side, mix: bool, end_id: u32) -> u32 {
    match side {
        Side::Prepend => ch.iter().find(|(id, _)| *id == end_id)
            .map(|(_, lp)| lp_to_level(*lp)).unwrap_or(crate::LEVEL_MAX),
        _ => {
            let first = ch.first().map(|(_, lp)| lp_to_level(*lp)).unwrap_or(crate::LEVEL_MAX);
            if mix { first.min(MIX_LEVEL) } else { first }
        }
    }
}

/// Build the anchor set: tokenized entries (+ word-sized sub-chunks), deduped,
/// scored for rarity, sorted rarest-first. Must run BEFORE the FULL-cache
/// trigram drop (surprisal reads `em.trigram`).
pub fn build_plan(em: &EnumModel, variant: &dyn Variant, model: &Model,
                  entry_seqs: &[Vec<u32>], lowercase_ids: &dyn Fn(&str) -> Vec<u32>,
                  opts: &PlanOpts) -> Result<Plan> {
    let start_id = model.start_id;
    let uni = &model.unigram_kn_cont;

    // Tokens training passwords start with: the START,START trigram children
    // with real probability. Back-off gives every token some mass, so it does
    // not count; nor do listed children whose probability discounting took to
    // zero (`prepare` clamps those to ln(f32::MIN_POSITIVE) ≈ -87; real
    // entries are above -20). A piece qualifies if it, or its lowercase form
    // as one token, is one.
    let zero_lp = f32::MIN_POSITIVE.ln() + 1.0;
    let starters: FxHashMap<u32, ()> = em.trigram.get(&pack(start_id, start_id))
        .map(|l| l.iter().filter(|&&(_, lp)| lp > zero_lp).map(|&(id, _)| (id, ())).collect())
        .unwrap_or_default();
    let starts_word = |t: u32| -> bool {
        if starters.contains_key(&t) { return true; }
        let lower = lowercase_ids(&decode_ids(&model.decode, &[t]));
        lower.len() == 1 && starters.contains_key(&lower[0])
    };
    let mut seqs: Vec<Vec<u32>> = Vec::new();
    let mut seen: FxHashMap<Vec<u32>, ()> = FxHashMap::default();
    for s in entry_seqs {
        if s.is_empty() { continue; }
        let chunks = if opts.split { anchor_chunks(&model.decode, s, &starts_word) }
                     else { vec![s.clone()] };
        for c in chunks {
            if seen.insert(c.clone(), ()).is_none() { seqs.push(c); }
        }
    }
    if seqs.is_empty() {
        bail!("--wordlist has no usable entries (every line was empty after tokenization)");
    }

    let t_plan = Instant::now();
    let p = opts.rarity_weight;
    let surs: Vec<f32> = seqs.iter().map(|s| surprisal(em, uni, start_id, s).max(1e-3)).collect();

    let mut overlong = 0usize;
    let (mut stand_in_lower, mut stand_in_end) = (0usize, 0usize);
    // Child lists by context. Anchors share contexts (every one starts at
    // START,START, the largest list), and `get_children` rebuilds and sorts the
    // list on every call: uncached, that was ~6.5 ms per anchor.
    let mut kids: FxHashMap<Ctx, Arc<[(u32, f32)]>> = FxHashMap::default();
    let mut children = |a: u32, b: u32| -> Arc<[(u32, f32)]> {
        kids.entry(pack(a, b))
            .or_insert_with(|| Arc::from(variant.get_children(em, pack(a, b), b).into_boxed_slice()))
            .clone()
    };
    let mut anchors: Vec<Anchor> = seqs.into_iter().zip(surs).map(|(toks, s)| {
        // Junction baseline: the anchor walked from START through the same
        // child lists the DFS will use for every prefix.
        let (mut a, mut b) = (start_id, start_id);
        let mut start_levels = 0u32;
        for &t in &toks {
            let ch = children(a, b);
            start_levels += lp_to_level(child_lp(&ch, uni, t));
            a = b;
            b = t;
        }
        if toks.len() > opts.max_tokens { overlong += 1; }

        // What may follow the anchor. For a 1-token anchor the context also
        // holds the previous token, but if (START, t) has no children then t
        // has no bigram continuations, and no other context ending in t has
        // any either.
        let last = toks[toks.len() - 1];
        let before = if toks.len() >= 2 { toks[toks.len() - 2] } else { start_id };
        let own = children(before, last);
        let post: Option<Arc<[(u32, f32)]>> = if own.is_empty() {
            let lower = lowercase_ids(&decode_ids(&model.decode, &toks));
            let alt = if !lower.is_empty() && lower != toks {
                let (x, y) = crate::context_from_prefix(&lower, start_id);
                Some(children(x, y))
            } else { None };
            if let Some(alt) = alt.filter(|l| !l.is_empty()) {
                stand_in_lower += 1;
                Some(alt)
            } else {
                // Nothing known after it in any form: allow END only, so the
                // anchor is still emitted bare and with prefixes. Its level is
                // irrelevant across anchors (the scheduler shares by lines).
                stand_in_end += 1;
                Some(Arc::from(vec![(model.end_id, -1.0f32)].into_boxed_slice()))
            }
        } else { None };

        // Exact when the following list is fixed: a multi-token anchor's own
        // context, or any stand-in. A plain 1-token anchor's depends on the
        // token before it, so it gets the minimum over every context ending in
        // it (filled in below, after one shared pass over the model).
        let fixed: Option<&[(u32, f32)]> = match &post {
            Some(l) => Some(l),
            None if toks.len() >= 2 => Some(&own),
            None => None,
        };
        let tail_min = match fixed {
            Some(ch) => tail_of(ch, opts.side, opts.mix, model.end_id),
            None => u32::MAX, // placeholder: 1-token anchor, see below
        };
        Anchor {
            cap: opts.max_tokens.max(toks.len()),
            weight: (s as f64).powf(p as f64),
            toks, surprisal: s, start_levels, tail_min, post,
        }
    }).collect();
    // Tail bound for plain 1-token anchors: the cheapest next step over every
    // context (x, t) the anchor's token t can end. Trigram contexts are
    // enumerated directly; every other x lands on the variant's back-off list,
    // which one probe with a context that is never a trigram key reproduces.
    // Without this the bound was 1, and a word like `Community` (whose real
    // next step costs ~5 levels) walked a 5-level band of prefix stems that
    // could never produce a line — ~150x wasted work per line.
    {
        let singles: FxHashMap<u32, ()> = anchors.iter()
            .filter(|a| a.tail_min == u32::MAX)
            .map(|a| (a.toks[0], ()))
            .collect();
        if !singles.is_empty() {
            use rayon::prelude::*;
            let (side, mix, end_id) = (opts.side, opts.mix, model.end_id);
            let mut best: FxHashMap<u32, u32> = em.trigram.par_iter()
                .filter_map(|(&ctx, _)| {
                    let b = (ctx & 0xFFFF_FFFF) as u32;
                    singles.contains_key(&b).then(|| {
                        (b, tail_of(&variant.get_children(em, ctx, b), side, mix, end_id))
                    })
                })
                .fold(FxHashMap::default, |mut m: FxHashMap<u32, u32>, (b, t)| {
                    let e = m.entry(b).or_insert(t);
                    *e = (*e).min(t);
                    m
                })
                .reduce(FxHashMap::default, |mut x, y| {
                    for (b, t) in y {
                        let e = x.entry(b).or_insert(t);
                        *e = (*e).min(t);
                    }
                    x
                });
            for &b in singles.keys() {
                let backoff = tail_of(&variant.get_children(em, pack(u32::MAX, b), b), side, mix, end_id);
                let e = best.entry(b).or_insert(backoff);
                *e = (*e).min(backoff);
            }
            for a in anchors.iter_mut().filter(|a| a.tail_min == u32::MAX) {
                a.tail_min = best.get(&a.toks[0]).copied().unwrap_or(0);
            }
        }
    }

    let n_stand_in = stand_in_lower + stand_in_end;
    if n_stand_in > 0 {
        crate::warn_msg(&format!(
            "warning: {n_stand_in} wordlist word(s) end in something the model has never \
             seen followed by anything. For {stand_in_lower} of them, suffixes are based on \
             the word's lowercase form; the other {stand_in_end} are printed on their own \
             and with prefixes only."));
    }
    // Rarest first. Stable, so equal anchors keep wordlist order — the strict
    // stream is byte-reproducible.
    anchors.sort_by(|x, y| y.surprisal.partial_cmp(&x.surprisal).unwrap_or(std::cmp::Ordering::Equal));

    if overlong > 0 {
        crate::warn_msg(&format!(
            "warning: {overlong} wordlist entr{} more tokens than --max-tokens {}; \
             printed on their own, with nothing added. Raise --max-tokens to give them room.",
            if overlong == 1 { "y has" } else { "ies have" }, opts.max_tokens));
    }
    crate::log_msg(&format!(
        "[wordlist] {} words, side={:?}, mix={}, split={}, rarity-weight={} (prepared in {:.2}s)",
        anchors.len(), opts.side, opts.mix, opts.split, p, t_plan.elapsed().as_secs_f64()));
    let w_sum: f64 = anchors.iter().map(|a| a.weight).sum();
    for a in anchors.iter().take(40) {
        crate::log_msg(&format!("[wordlist]   {:<24} tokens={} rarity={:>6.1} share={:>5.1}%",
            decode_ids(&model.decode, &a.toks), a.toks.len(), a.surprisal,
            100.0 * a.weight / w_sum));
    }
    if anchors.len() > 40 {
        crate::log_msg(&format!("[wordlist]   … {} more", anchors.len() - 40));
    }

    Ok(Plan { anchors, side: opts.side, mix: opts.mix, uni_cont: uni.clone(), end_id: model.end_id })
}

/// One initial state per anchor, rarest first. The anchor index rides in
/// `prefix[0]` with `prefix_len = 0`: the wordlist DFS builds its own token
/// prefix and orders by its scheduler, not by `log_prob`.
pub fn initial_states(plan: &Plan) -> Vec<HeapEntry> {
    (0..plan.anchors.len()).map(|i| {
        let mut prefix = [0u32; 32];
        prefix[0] = i as u32;
        HeapEntry { log_prob: 0.0, prefix_len: 0, prefix }
    }).collect()
}

/// Deal anchors round-robin (they arrive rarest-first) so every worker gets a
/// similar mix of rare and common words. Deterministic for a given `n`, which
/// checkpoint resume relies on.
pub fn partition(states: Vec<HeapEntry>, n: usize) -> Vec<Vec<HeapEntry>> {
    let n = n.max(1);
    let mut parts: Vec<Vec<HeapEntry>> = (0..n).map(|_| Vec::new()).collect();
    for (i, s) in states.into_iter().enumerate() {
        parts[i % n].push(s);
    }
    parts
}

/// Per-worker `--count` quota ∝ the summed `S^p` of the worker's anchors.
/// An equal split would hand a thread holding one common word as much output
/// as a thread holding one rare word, undoing the rarity tilt.
pub fn quotas(plan: &Plan, parts: &[Vec<HeapEntry>], total: u64) -> Vec<u64> {
    let active: Vec<usize> = (0..parts.len()).filter(|&i| !parts[i].is_empty()).collect();
    let weights: Vec<f64> = active.iter()
        .map(|&i| parts[i].iter().map(|s| plan.anchors[s.prefix[0] as usize].weight).sum())
        .collect();
    let shares = alloc_budget(&weights, total);
    let mut out = vec![0u64; parts.len()];
    for (k, &i) in active.iter().enumerate() { out[i] = shares[k]; }
    out
}

// ============================================================================
// DFS
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Walking a prefix from START. `after_mix`: the last piece was a `--mix`
    /// entry, so the only remaining move is committing the anchor.
    Pre { after_mix: bool },
    /// After the anchor. `mix_used`: the suffix side already holds a mix piece.
    Suf { mix_used: bool },
}

struct Frame {
    children:  Children,
    idx:       usize,
    base_lp:   f32,
    acc_level: u32,
    phase:     Phase,
    /// Tokens pushed on entering this frame (popped on leaving it).
    pushed:    usize,
    /// Number of `--mix` items this frame offers.
    n_mix:     usize,
    /// Whether this frame's model children are items (prefix room / phase).
    kids:      bool,
}

/// What a frame's item `i` is. Item order is fixed so a checkpoint's idx stack
/// replays exactly:
///   Pre: [commit anchor] [mix pieces…] [model children…]
///   Suf: [mix pieces…] [model children…]
#[derive(Clone, Copy)]
enum Item { Commit, Mix(usize), Child(usize) }

impl Frame {
    fn n_items(&self) -> usize {
        let kids = if self.kids { self.children.len() } else { 0 };
        match self.phase {
            Phase::Pre { .. } => 1 + self.n_mix + kids,
            Phase::Suf { .. } => self.n_mix + kids,
        }
    }
    fn item(&self, i: usize) -> Item {
        match self.phase {
            Phase::Pre { .. } => {
                if i == 0 { Item::Commit }
                else if i <= self.n_mix { Item::Mix(i - 1) }
                else { Item::Child(i - 1 - self.n_mix) }
            }
            Phase::Suf { .. } => {
                if i < self.n_mix { Item::Mix(i) } else { Item::Child(i - self.n_mix) }
            }
        }
    }
    /// First item index of the model-children block.
    fn kids_start(&self) -> usize {
        match self.phase {
            Phase::Pre { .. } => 1 + self.n_mix,
            Phase::Suf { .. } => self.n_mix,
        }
    }
}

/// Child-list source with the same three cache modes as `enumerate_to_sink`'s
/// `cached_get!` (and the same guarantee: the mode never changes the output).
struct Source<'a> {
    em:          &'a EnumModel,
    variant:     &'a dyn Variant,
    child_cache: ChildCache,
    mode:        CacheMode,
    local:       FxHashMap<Ctx, Arc<[(u32, f32)]>>,
    bounded:     Option<BoundedChildCache>,
    misses:      usize,
}

impl Source<'_> {
    const LOCAL_CACHE_MAX: usize = 262144;

    fn get(&mut self, ctx: Ctx, b: u32) -> Children {
        let (em, variant) = (self.em, self.variant);
        match self.mode {
            CacheMode::Bounded(_) => Children::Shared(
                self.bounded.as_mut().unwrap().get_or(ctx, || variant.get_children(em, ctx, b))),
            CacheMode::Lazy => Children::Owned(variant.get_children(em, ctx, b).into_boxed_slice()),
            CacheMode::Full => {
                if let Some(ch) = self.child_cache.get(&ctx) {
                    Children::Ref(&ch[..])
                } else if let Some(a) = self.local.get(&ctx) {
                    Children::Shared(a.clone())
                } else {
                    self.misses += 1;
                    if self.local.len() >= Self::LOCAL_CACHE_MAX { self.local.clear(); }
                    let a: Arc<[(u32, f32)]> =
                        Arc::from(variant.get_children(em, ctx, b).into_boxed_slice());
                    self.local.insert(ctx, a.clone());
                    Children::Shared(a)
                }
            }
        }
    }
}

/// Token prefix + incrementally decoded bytes (mirrors `enumerate_to_sink`).
#[derive(Default)]
struct Path {
    toks:  Vec<u32>,
    bytes: Vec<u8>,
    lens:  Vec<usize>,
    nl:    usize,
}

impl Path {
    fn clear(&mut self) { self.toks.clear(); self.bytes.clear(); self.lens.clear(); self.nl = 0; }
    fn push(&mut self, id: u32, decode: &[Vec<u8>], has_nl: &[bool]) {
        self.toks.push(id);
        self.lens.push(self.bytes.len());
        if let Some(b) = decode.get(id as usize) { self.bytes.extend_from_slice(b); }
        if has_nl.get(id as usize).copied().unwrap_or(false) { self.nl += 1; }
    }
    fn pop_n(&mut self, n: usize, has_nl: &[bool]) {
        for _ in 0..n {
            let Some(id) = self.toks.pop() else { return };
            if let Some(old) = self.lens.pop() { self.bytes.truncate(old); }
            if has_nl.get(id as usize).copied().unwrap_or(false) { self.nl -= 1; }
        }
    }
    fn ctx(&self, start_id: u32) -> (u32, u32, u32) {
        let (a, b) = crate::context_from_prefix(&self.toks, start_id);
        (a, b, self.toks.last().copied().unwrap_or(start_id))
    }
}

/// One anchor's resumable level sweep. `stack` empty and `!done` means "about
/// to start the pass at `target`".
struct Stream {
    /// Index in this worker's state list — the stream's checkpoint identity.
    slot:     usize,
    anchor:   usize,
    weight:   f64,
    target:   u32,
    /// The current pass held something back for a higher level.
    deferred: bool,
    emitted:  u64,
    /// Turns that ended on the work budget with no line written. Counted
    /// like lines by the scheduler, so a stream pays for work, not just output.
    charge:   u64,
    /// DFS steps since this stream last wrote a line (dead-stream cutoff).
    idle:     u64,
    done:     bool,
    stack:    Vec<Frame>,
    path:     Path,
}

impl Stream {
    fn snapshot(&self) -> StreamCkpt {
        StreamCkpt {
            slot: self.slot, target: self.target, deferred: self.deferred,
            emitted: self.emitted, charge: self.charge, idle: self.idle, done: self.done,
            prefix: self.path.toks.clone(),
            idx_stack: self.stack.iter().map(|f| f.idx).collect(),
        }
    }
}

/// A stream's checkpointed position: one `stream=` line in its thread section.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamCkpt {
    pub slot:      usize,
    pub target:    u32,
    pub deferred:  bool,
    pub emitted:   u64,
    pub charge:    u64,
    pub idle:      u64,
    pub done:      bool,
    pub prefix:    Vec<u32>,
    pub idx_stack: Vec<usize>,
}

impl StreamCkpt {
    pub fn to_field(&self) -> String {
        let csv = |v: Vec<String>| v.join(",");
        format!("{};{};{};{};{};{};{};{};{}", self.slot, self.target, self.deferred as u8,
            self.emitted, self.charge, self.idle, self.done as u8,
            csv(self.prefix.iter().map(|x| x.to_string()).collect()),
            csv(self.idx_stack.iter().map(|x| x.to_string()).collect()))
    }

    pub fn from_field(v: &str) -> Result<Self> {
        let f: Vec<&str> = v.split(';').collect();
        if f.len() != 9 { bail!("checkpoint: bad stream entry '{}'", v); }
        fn list<T: std::str::FromStr>(s: &str) -> Result<Vec<T>> {
            if s.is_empty() { return Ok(Vec::new()); }
            s.split(',').map(|x| x.parse::<T>().map_err(|_| anyhow!("checkpoint: bad list '{}'", s)))
                .collect()
        }
        let flag = |s: &str| -> Result<bool> {
            match s { "0" => Ok(false), "1" => Ok(true), _ => bail!("checkpoint: bad flag '{}'", s) }
        };
        Ok(StreamCkpt {
            slot: f[0].parse()?, target: f[1].parse()?, deferred: flag(f[2])?,
            emitted: f[3].parse()?, charge: f[4].parse()?, idle: f[5].parse()?,
            done: flag(f[6])?, prefix: list(f[7])?, idx_stack: list(f[8])?,
        })
    }
}

/// Emission settings + the caller's sink, shared by every stream of a worker.
struct Out<'a, F> {
    emit:          F,
    decode_buf:    Vec<u8>,
    kind:          DecodeKind,
    min_len:       usize,
    max_len:       usize,
    min_tokens:    usize,
    enterprise_on: bool,
    case_masks:    &'a [CaseMask],
    /// Lines still allowed before the worker's quota (caps a case-mask burst).
    remaining:     u64,
}

/// Model access for one worker; streams own their DFS state and pass it in.
struct Walk<'a> {
    plan:     &'a Plan,
    src:      Source<'a>,
    decode:   &'a [Vec<u8>],
    has_nl:   &'a [bool],
    start_id: u32,
    /// Junction computed by the DFS's level check, reused by `open`.
    junction: Option<(u32, f32, u32)>,
    /// DFS steps taken (telemetry: walk work per emitted line).
    steps:    u64,
    /// Junction memo keyed by (anchor, prefix context). The junction depends on
    /// the prefix only through its last two tokens, and the level sweep revisits
    /// every stem on every pass, so without this each pass repeats the
    /// child-list scans (measured: --prepend-only 17x slower than --append-only).
    jmemo:    FxHashMap<(u32, Ctx), (u32, f32, u32)>,
    /// Prepend-only: END's log-prob per context (`None` = END not a child).
    endmemo:  FxHashMap<Ctx, Option<f32>>,
}

impl Walk<'_> {
    fn push_toks(&self, p: &mut Path, toks: &[u32]) {
        for &t in toks { p.push(t, self.decode, self.has_nl); }
    }

    /// `post_of`: this frame sits right after anchor (or suffix mix piece) `k`,
    /// where that anchor's stand-in continuations (if any) replace the empty list.
    fn new_frame(&mut self, p: &Path, anchor: usize, phase: Phase, pushed: usize,
                 base_lp: f32, acc_level: u32, post_of: Option<usize>) -> Frame {
        let plan = self.plan;
        let a = &plan.anchors[anchor];
        let (pa, pb, bv) = p.ctx(self.start_id);
        let stand_in = post_of.and_then(|k| plan.anchors[k].post.as_ref());
        let (n_mix, kids) = match phase {
            Phase::Pre { after_mix } => {
                let room = p.toks.len() + 1 + a.toks.len() <= a.cap;
                let mix = plan.mix && !after_mix && plan.side != Side::Append;
                (if mix { plan.anchors.len() } else { 0 }, !after_mix && room)
            }
            Phase::Suf { mix_used } => {
                let mix = plan.mix && !mix_used && plan.side != Side::Prepend;
                (if mix { plan.anchors.len() } else { 0 }, true)
            }
        };
        let children = if !kids {
            // A Pre frame that only commits never reads its children.
            Children::Owned(Box::new([]))
        } else if let Some(list) = stand_in {
            if plan.side == Side::Prepend {
                match list.iter().find(|(id, _)| *id == plan.end_id) {
                    Some(&e) => Children::Owned(Box::new([e])),
                    None => Children::Owned(Box::new([])),
                }
            } else {
                Children::Shared(list.clone())
            }
        } else if plan.side == Side::Prepend && matches!(phase, Phase::Suf { .. }) {
            // Prepend-only: after the anchor the only legal move is END, so
            // hand the frame just that child instead of scanning the whole
            // list on every stem of every pass.
            let ctx = pack(pa, pb);
            let end = match self.endmemo.get(&ctx) {
                Some(&e) => e,
                None => {
                    let ch = self.src.get(ctx, bv);
                    let e = ch.iter().find(|(id, _)| *id == plan.end_id).map(|(_, lp)| *lp);
                    if self.endmemo.len() >= (1 << 22) { self.endmemo.clear(); }
                    self.endmemo.insert(ctx, e);
                    e
                }
            };
            match end {
                Some(lp) => Children::Owned(Box::new([(plan.end_id, lp)])),
                None => Children::Owned(Box::new([])),
            }
        } else {
            self.src.get(pack(pa, pb), bv)
        };
        Frame { children, idx: 0, base_lp, acc_level, phase, pushed, n_mix, kids }
    }

    /// The root frame of a pass: a prefix walk from START, or (append-only)
    /// the anchor already committed.
    fn root(&mut self, p: &mut Path, anchor: usize) -> Frame {
        p.clear();
        let plan = self.plan;
        if plan.side == Side::Append {
            let toks = &plan.anchors[anchor].toks;
            self.push_toks(p, toks);
            self.new_frame(p, anchor, Phase::Suf { mix_used: false }, toks.len(), 0.0, 0, Some(anchor))
        } else {
            self.new_frame(p, anchor, Phase::Pre { after_mix: false }, 0, 0.0, 0, None)
        }
    }

    /// Junction cost of committing `anchor` after the current prefix: the extra
    /// levels the anchor costs here versus at the start of a password, clipped
    /// at 0 (a prefix can make an anchor cheaper, but never earns a bonus — that
    /// keeps every step non-negative, so the level-sweep pruning stays exact).
    /// Returns `(junction levels, anchor log-prob here, exact tail levels)`.
    fn junction(&mut self, p: &Path, anchor: usize) -> (u32, f32, u32) {
        const JMEMO_MAX: usize = 1 << 22;
        let plan = self.plan;
        let a = &plan.anchors[anchor];
        let (mut pa, mut pb, _) = p.ctx(self.start_id);
        let key = (anchor as u32, pack(pa, pb));
        if let Some(&v) = self.jmemo.get(&key) { return v; }
        let (mut lv, mut lp) = (0u32, 0f32);
        for &t in &a.toks {
            let ch = self.src.get(pack(pa, pb), pb);
            let l = child_lp(&ch, &plan.uni_cont, t);
            lv += lp_to_level(l);
            lp += l;
            pa = pb;
            pb = t;
        }
        // Exact cheapest next step from the context the anchor leaves behind
        // here (the per-anchor `tail_min` is only a bound for 1-token anchors).
        let ch = match &a.post {
            Some(list) => Children::Shared(list.clone()),
            None => self.src.get(pack(pa, pb), pb),
        };
        let tail = if plan.side == Side::Prepend {
            ch.iter().find(|(id, _)| *id == plan.end_id)
                .map(|(_, l)| lp_to_level(*l)).unwrap_or(crate::LEVEL_MAX)
        } else {
            let first = ch.first().map(|(_, l)| lp_to_level(*l)).unwrap_or(crate::LEVEL_MAX);
            if plan.mix { first.min(MIX_LEVEL) } else { first }
        };
        let v = (lv.saturating_sub(a.start_levels).min(JUNCTION_CAP), lp, tail);
        if self.jmemo.len() >= JMEMO_MAX { self.jmemo.clear(); }
        self.jmemo.insert(key, v);
        v
    }

    /// Descend into item `it` of `f` without any level check, returning the
    /// child frame. Shared by the DFS (after its checks) and checkpoint replay.
    fn open(&mut self, p: &mut Path, anchor: usize, f: &Frame, it: Item) -> Option<Frame> {
        let plan = self.plan;
        match (f.phase, it) {
            (Phase::Pre { .. }, Item::Commit) => {
                let (j, lp, _) = match self.junction.take() {
                    Some(v) => v,
                    None => self.junction(p, anchor),
                };
                let toks = &plan.anchors[anchor].toks;
                self.push_toks(p, toks);
                Some(self.new_frame(p, anchor, Phase::Suf { mix_used: false }, toks.len(),
                    f.base_lp + lp, f.acc_level + j, Some(anchor)))
            }
            (phase, Item::Mix(m)) => {
                if m == anchor { return None; }
                let toks = &plan.anchors[m].toks;
                self.push_toks(p, toks);
                let next = match phase {
                    Phase::Pre { .. } => Phase::Pre { after_mix: true },
                    Phase::Suf { .. } => Phase::Suf { mix_used: true },
                };
                Some(self.new_frame(p, anchor, next, toks.len(), f.base_lp, f.acc_level + MIX_LEVEL,
                    matches!(next, Phase::Suf { .. }).then_some(m)))
            }
            (phase, Item::Child(c)) => {
                let (t, lp) = f.children[c];
                p.push(t, self.decode, self.has_nl);
                let next = match phase {
                    Phase::Pre { .. } => Phase::Pre { after_mix: false },
                    s => s,
                };
                Some(self.new_frame(p, anchor, next, 1, f.base_lp + lp, f.acc_level + lp_to_level(lp), None))
            }
            (Phase::Suf { .. }, Item::Commit) => None,
        }
    }

    /// Room for a mix piece: the piece plus whatever must still follow it.
    fn mix_fits(&self, p: &Path, anchor: usize, f: &Frame, m: usize) -> bool {
        let a = &self.plan.anchors[anchor];
        let need = self.plan.anchors[m].toks.len()
            + if matches!(f.phase, Phase::Pre { .. }) { a.toks.len() } else { 0 };
        p.toks.len() + need <= a.cap
    }

    /// Rebuild a stream from its checkpoint: replay the saved idx stack through
    /// `open` (the code that built it), then check the token prefix matches.
    fn restore(&mut self, s: &mut Stream, ck: &StreamCkpt, label: &str) -> Result<()> {
        s.target = ck.target;
        s.deferred = ck.deferred;
        s.emitted = ck.emitted;
        s.charge = ck.charge;
        s.idle = ck.idle;
        s.done = ck.done;
        s.stack.clear();
        s.path.clear();
        if ck.done || ck.idx_stack.is_empty() { return Ok(()); }
        let root = self.root(&mut s.path, s.anchor);
        s.stack.push(root);
        s.stack[0].idx = ck.idx_stack[0];
        for k in 1..ck.idx_stack.len() {
            let i = ck.idx_stack[k - 1].checked_sub(1).ok_or_else(|| anyhow!(
                "{}resume: stream {} frame {} idx underflow", label, ck.slot, k - 1))?;
            let parent = s.stack.pop().unwrap();
            if i >= parent.n_items() {
                bail!("{}resume: stream {} item {} out of range in frame {}", label, ck.slot, i, k - 1);
            }
            let child = self.open(&mut s.path, s.anchor, &parent, parent.item(i));
            s.stack.push(parent);
            let mut child = child.ok_or_else(|| anyhow!(
                "{}resume: stream {} frame {} item {} does not open", label, ck.slot, k - 1, i))?;
            child.idx = ck.idx_stack[k];
            s.stack.push(child);
        }
        if s.path.toks != ck.prefix {
            bail!("{}resume: stream {} replayed prefix differs from checkpoint \
                   (model/threads mismatch?)", label, ck.slot);
        }
        Ok(())
    }

    /// Run stream `s` until it emits at least one line, is exhausted, or uses
    /// up this turn's work budget (`TURN_STEPS`). The budget is what keeps one
    /// stream that emits slowly — or never — from holding the whole worker.
    fn advance<F>(&mut self, s: &mut Stream, out: &mut Out<F>) -> Result<Turn>
    where
        F: FnMut(u32, u32, &[u8]) -> Result<()>,
    {
        let plan = self.plan;
        let anchor = s.anchor;
        let cap = plan.anchors[anchor].cap;
        let tail = plan.anchors[anchor].tail_min;
        let mut budget = TURN_STEPS;
        loop {
            if s.done { return Ok(Turn::Done); }
            if budget == 0 { return Ok(Turn::Yield); }
            budget -= 1;
            s.idle += 1;
            self.steps += 1;
            if s.stack.is_empty() {
                let root = self.root(&mut s.path, anchor);
                s.stack.push(root);
            }
            let target = s.target;
            let f = s.stack.last_mut().unwrap();
            if f.idx >= f.n_items() {
                let n = f.pushed;
                s.stack.pop();
                if s.stack.is_empty() {
                    // Pass over. Level pruning is the only reason a candidate is
                    // held back, so a pass that deferred nothing exhausted the
                    // anchor's whole space.
                    if !s.deferred || target >= crate::LEVEL_MAX {
                        s.done = true;
                        s.path.clear();
                        return Ok(Turn::Done);
                    }
                    s.target += 1;
                    s.deferred = false;
                } else {
                    s.path.pop_n(n, self.has_nl);
                }
                continue;
            }
            let i = f.idx;
            f.idx += 1;
            let it = f.item(i);
            let (acc, phase) = (f.acc_level, f.phase);

            match it {
                Item::Commit => {
                    let j = self.junction(&s.path, anchor);
                    // The suffix side still owes at least its cheapest next
                    // step from here (may be 0).
                    if acc + j.0 + j.2 > target { s.deferred = true; continue; }
                    self.junction = Some(j);
                }
                Item::Mix(m) => {
                    if acc + MIX_LEVEL > target {
                        // Every mix piece costs the same: skip the block.
                        s.deferred = true;
                        let f = s.stack.last_mut().unwrap();
                        f.idx = f.kids_start();
                        continue;
                    }
                    if m == anchor || !self.mix_fits(&s.path, anchor, s.stack.last().unwrap(), m) {
                        continue;
                    }
                }
                Item::Child(c) => {
                    let (t, lp) = f.children[c];
                    let lvl = acc + lp_to_level(lp);
                    match phase {
                        Phase::Pre { .. } => {
                            if t == out_end(plan) { continue; }
                            // A prefix still owes the anchor (junction ≥ 0) and
                            // `tail`, so a stem past `target - tail` can't
                            // contribute this pass; later siblings cost more.
                            if lvl + tail > target {
                                s.deferred = true;
                                f.idx = f.n_items();
                                continue;
                            }
                        }
                        Phase::Suf { .. } => {
                            if lvl > target {
                                s.deferred = true;
                                f.idx = f.n_items();
                                continue;
                            }
                            if t == out_end(plan) {
                                if lvl == target && s.path.toks.len() >= out.min_tokens {
                                    let sort_key = (-(f.base_lp + lp)).to_bits();
                                    let n = emit_candidate(&s.path, self.decode, out, target, sort_key)?;
                                    if n > 0 { s.idle = 0; return Ok(Turn::Lines(n)); }
                                }
                                continue;
                            }
                            if plan.side == Side::Prepend { continue; }
                            // (No "lvl == target" cut: a zero-level step — P = 1 — can
                            // still end a candidate at this very level.)
                            if s.path.toks.len() >= cap { continue; }
                        }
                    }
                }
            }
            let parent = s.stack.pop().unwrap();
            let child = self.open(&mut s.path, anchor, &parent, it);
            s.stack.push(parent);
            if let Some(child) = child { s.stack.push(child); }
        }
    }
}

#[inline]
fn out_end(plan: &Plan) -> u32 { plan.end_id }

/// Rarity-weighted interleave of this worker's anchor streams. Same contract
/// as `enumerate_to_sink` (which forwards here): `emit(level, sort_key, bytes)`
/// per candidate, `target_count` as this worker's quota, and a resumable
/// position published through `on_checkpoint`.
///
/// Scheduling is weighted fair queuing: the next stream to advance is the one
/// with the smallest `(emitted + 1) / weight`, ties to the rarer anchor (lower
/// slot). It depends only on emission counts, so `--strict` stays
/// byte-reproducible and a checkpoint of every stream's position resumes it.
#[allow(clippy::too_many_arguments)]
pub fn enumerate<F, H>(
    plan: &Plan,
    enum_model: &EnumModel,
    variant: &dyn Variant,
    child_cache: ChildCache,
    cache_mode: CacheMode,
    decode_table: &[Vec<u8>],
    kind: DecodeKind,
    start_id: u32,
    initial_states: Vec<HeapEntry>,
    min_tokens: usize,
    min_len: usize,
    max_len: usize,
    min_level: u32,
    target_count: u64,
    enterprise_on: bool,
    case_masks: &[CaseMask],
    emit: F,
    thread_label: &str,
    resume: Option<ThreadCkpt>,
    checkpoint_every: Option<Duration>,
    mut on_checkpoint: H,
) -> Result<()>
where
    F: FnMut(u32, u32, &[u8]) -> Result<()>,
    H: FnMut(&ThreadCkpt),
{
    let t0 = Instant::now();
    let mut last_log = Instant::now();
    let mut last_ckpt = Instant::now();
    let has_nl: Vec<bool> = decode_table.iter()
        .map(|b| b.iter().any(|&c| c == b'\n' || c == b'\r'))
        .collect();

    let mut w = Walk {
        plan,
        src: Source {
            em: enum_model, variant, child_cache, mode: cache_mode,
            local: FxHashMap::default(),
            bounded: match cache_mode {
                CacheMode::Bounded(cap) => Some(BoundedChildCache::new(cap)),
                _ => None,
            },
            misses: 0,
        },
        decode: decode_table,
        has_nl: &has_nl,
        start_id,
        junction: None,
        steps: 0,
        jmemo: FxHashMap::default(),
        endmemo: FxHashMap::default(),
    };
    let mut out = Out {
        emit, decode_buf: Vec::with_capacity(64), kind, min_len, max_len, min_tokens,
        enterprise_on, case_masks, remaining: target_count,
    };

    let mut streams: Vec<Stream> = initial_states.iter().enumerate().map(|(slot, st)| {
        let anchor = st.prefix[0] as usize;
        Stream {
            slot, anchor, weight: plan.anchors[anchor].weight, target: min_level,
            deferred: false, emitted: 0, charge: 0, idle: 0, done: false,
            stack: Vec::new(), path: Path::default(),
        }
    }).collect();

    let mut total: u64 = 0;
    if let Some(r) = &resume {
        if r.streams.len() != streams.len() {
            bail!("{}resume: checkpoint has {} anchor streams, this partition has {} \
                   (--threads or wordlist mismatch?)", thread_label, r.streams.len(), streams.len());
        }
        total = r.emitted;
        for ck in &r.streams {
            let s = streams.get_mut(ck.slot).ok_or_else(|| anyhow!(
                "{}resume: stream slot {} out of range", thread_label, ck.slot))?;
            w.restore(s, ck, thread_label)?;
        }
    }

    // Virtual finish time of a stream's next turn: lines written plus turns
    // charged for work. Weights are > 0 and finite, so the f64 bits order like
    // the values.
    let key = |s: &Stream| -> u64 { ((s.emitted + s.charge + 1) as f64 / s.weight).to_bits() };
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = streams.iter()
        .filter(|s| !s.done)
        .map(|s| Reverse((key(s), s.slot)))
        .collect();

    let mut picks: u64 = 0;
    while total < target_count {
        let Some(Reverse((_, i))) = heap.pop() else { break };
        out.remaining = target_count - total;
        let mut slow_turn = false;
        match w.advance(&mut streams[i], &mut out)? {
            Turn::Done => continue, // exhausted: drop from the schedule
            Turn::Lines(n) => {
                streams[i].emitted += n;
                total += n;
            }
            Turn::Yield => {
                slow_turn = true;
                let s = &mut streams[i];
                if s.idle >= DEAD_STEPS {
                    s.done = true;
                    s.stack.clear();
                    crate::warn_msg(&format!(
                        "{}warning: wordlist word {:?} produced no candidate in {} attempts \
                         (everything it reaches is filtered out or too long); dropped.",
                        thread_label, decode_ids(decode_table, &plan.anchors[s.anchor].toks),
                        s.idle));
                    continue;
                }
                s.charge += 1;
            }
        }
        heap.push(Reverse((key(&streams[i]), i)));

        picks = picks.wrapping_add(1);
        if slow_turn || (picks & 0x3FF) == 0 {
            if let Some(dur) = checkpoint_every {
                if last_ckpt.elapsed() >= dur {
                    on_checkpoint(&ThreadCkpt {
                        target_level: 0, init_idx: 0, prefix: Vec::new(), idx_stack: Vec::new(),
                        emitted: total,
                        streams: streams.iter().map(Stream::snapshot).collect(),
                        wordlist_pos: String::new(),
                    });
                    last_ckpt = Instant::now();
                }
            }
            if last_log.elapsed().as_secs() >= 30 {
                let live = streams.iter().filter(|s| !s.done).count();
                crate::log_msg(&format!("{}  emitted={} live_anchors={}/{} local_misses={} elapsed={:.0}s",
                    thread_label, total, live, streams.len(), w.src.misses, t0.elapsed().as_secs_f64()));
                last_log = Instant::now();
            }
        }
    }

    crate::log_msg(&format!("{}[wordlist] DONE emitted={} anchors_exhausted={}/{} dfs_steps={} \
        local_misses={} in {:.0}s",
        thread_label, total, streams.iter().filter(|s| s.done).count(), streams.len(),
        w.steps, w.src.misses, t0.elapsed().as_secs_f64()));
    let mut order: Vec<&Stream> = streams.iter().collect();
    order.sort_by_key(|s| std::cmp::Reverse(s.idle));
    for s in order.into_iter().take(8) {
        crate::log_msg(&format!("{}[wordlist]   slot {} {:?} emitted={} level={} idle={} tail={} cap={}{}",
            thread_label, s.slot, decode_ids(decode_table, &plan.anchors[s.anchor].toks), s.emitted, s.target, s.idle,
            plan.anchors[s.anchor].tail_min, plan.anchors[s.anchor].cap, if s.done { " (exhausted)" } else { "" }));
    }
    Ok(())
}

/// Length / newline / policy filters + case shaping, then `emit`. Mirrors the
/// terminal branch of `enumerate_to_sink`. Returns the number of lines written
/// (case shaping can write several; never more than `out.remaining`).
fn emit_candidate<F>(path: &Path, decode: &[Vec<u8>], out: &mut Out<F>, level: u32,
                     sort_key: u32) -> Result<u64>
where
    F: FnMut(u32, u32, &[u8]) -> Result<()>,
{
    if out.case_masks.is_empty() {
        let r = finalized_range(&path.bytes, out.kind);
        let s = &path.bytes[r];
        let has_nl = path.nl > 0 && s.iter().any(|&c| c == b'\n' || c == b'\r');
        if s.is_empty() || s.len() < out.min_len || s.len() > out.max_len || has_nl { return Ok(0); }
        if out.enterprise_on {
            match enterprise::decide(s) {
                enterprise::Decision::AsIs => (out.emit)(level, sort_key, s)?,
                enterprise::Decision::Cap => {
                    out.decode_buf.clear();
                    out.decode_buf.extend_from_slice(s);
                    out.decode_buf[0] = out.decode_buf[0].to_ascii_uppercase();
                    (out.emit)(level, sort_key, &out.decode_buf)?;
                }
                enterprise::Decision::Drop => return Ok(0),
            }
        } else {
            (out.emit)(level, sort_key, s)?;
        }
        return Ok(1);
    }
    let mut n = 0u64;
    for mask in out.case_masks {
        if n >= out.remaining { break; }
        out.decode_buf.clear();
        for (pos, &id) in path.toks.iter().enumerate() {
            if let Some(b) = decode.get(id as usize) {
                let start = out.decode_buf.len();
                out.decode_buf.extend_from_slice(b);
                let end = out.decode_buf.len();
                apply_op(&mut out.decode_buf[start..end], mask.op_for(pos));
            }
        }
        finalize_decoded(&mut out.decode_buf, out.kind);
        let len = out.decode_buf.len();
        if !out.decode_buf.is_empty() && len >= out.min_len && len <= out.max_len
            && !out.decode_buf.iter().any(|&c| c == b'\n' || c == b'\r')
        {
            (out.emit)(level, sort_key, &out.decode_buf)?;
            n += 1;
        }
    }
    Ok(n)
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
        chunks_with(pieces, &[])
    }

    /// As `chunks_of`, but the pieces in `never_start` are ones no training
    /// password starts with.
    fn chunks_with(pieces: &[&str], never_start: &[&str]) -> Vec<String> {
        let (decode, seq) = seed(pieces);
        let starts = |t: u32| !never_start.iter().any(|n| n.as_bytes() == decode[t as usize].as_slice());
        anchor_chunks(&decode, &seq, &starts).iter().map(|c| decode_ids(&decode, c)).collect()
    }

    // A long word split mid-morpheme can pass the length rule. A piece no
    // password starts with marks the split as not a word boundary, and then no
    // piece is used — not even the plausible-looking stem.
    #[test]
    fn fragment_that_never_starts_a_password_blocks_the_split() {
        assert_eq!(chunks_with(&["Prov", "iders"], &["iders"]), vec!["Providers"]);
        assert_eq!(chunks_with(&["prot", "ection"], &["ection"]), vec!["protection"]);
        assert_eq!(chunks_with(&["Smart", "Screen"], &["iders"]),
                   vec!["SmartScreen", "Smart", "Screen"]);
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

    // The whole seed always leads its chunk list.
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
        assert_eq!(alloc_budget(&[3.0, 1.0], 100), vec![75, 25]);
        assert_eq!(alloc_budget(&[1.0, 1.0, 1.0, 1.0], 2), vec![1, 1, 1, 1]);
        assert!(alloc_budget(&[0.0, 0.0], 10).iter().all(|&x| x >= 1));
    }

    #[test]
    fn alloc_budget_sums_to_total_exactly() {
        for (n, total) in [(200usize, 500u64), (3, 10), (7, 1000), (1, 1), (13, 13)] {
            let got: u64 = alloc_budget(&vec![1.0f64; n], total).iter().sum();
            assert_eq!(got, total, "n={n} total={total}");
        }
        let got: u64 = alloc_budget(&[5.0, 3.0, 1.1, 0.4], 997).iter().sum();
        assert_eq!(got, 997);
    }

    // No --count: every worker runs unbounded.
    #[test]
    fn alloc_budget_passes_unbounded_through() {
        assert_eq!(alloc_budget(&[1.0, 9.0], u64::MAX), vec![u64::MAX, u64::MAX]);
    }

    #[test]
    fn stream_ckpt_round_trips() {
        let ck = StreamCkpt { slot: 3, target: 17, deferred: true, emitted: 123456789,
                              charge: 12, idle: 4_000_000, done: false,
                              prefix: vec![5, 900, 41], idx_stack: vec![2, 1, 7, 0] };
        assert_eq!(StreamCkpt::from_field(&ck.to_field()).unwrap(), ck);
        let fresh = StreamCkpt { slot: 0, target: 0, deferred: false, emitted: 0, charge: 0,
                                 idle: 0, done: true, prefix: vec![], idx_stack: vec![] };
        assert_eq!(StreamCkpt::from_field(&fresh.to_field()).unwrap(), fresh);
        assert!(StreamCkpt::from_field("1;2;3").is_err());
        assert!(StreamCkpt::from_field("0;0;2;0;0;0;0;;").is_err()); // bad flag
    }

    // Round-robin keeps rarest-first order inside each partition and spreads
    // the rare anchors across workers instead of stacking them on one.
    #[test]
    fn partition_deals_round_robin() {
        let states: Vec<HeapEntry> = (0..5u32).map(|i| {
            let mut prefix = [0u32; 32];
            prefix[0] = i;
            HeapEntry { log_prob: 0.0, prefix_len: 0, prefix }
        }).collect();
        let parts = partition(states, 2);
        let ids: Vec<Vec<u32>> = parts.iter().map(|p| p.iter().map(|s| s.prefix[0]).collect()).collect();
        assert_eq!(ids, vec![vec![0, 2, 4], vec![1, 3]]);
    }
}
