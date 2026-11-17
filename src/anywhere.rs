//! The default `--wordlist` generator: each wordlist word anywhere in the
//! candidate, `[prefix] + word + [suffix]`, ranked at the plain model's cost.
//! (`--prepend-only` is the same generator with the word last: no suffix, the
//! candidate ends right after the word.)
//!
//! The prefix is walked from START and the suffix after the word, both through
//! the variant's child lists, as the enumerator walks any password (so
//! `--unigram-tail` and the tail variants apply to them). The word is priced
//! as one block in the prefix's context: `ceil(-log P(word | last two prefix
//! tokens))` with the per-step transition `build_seeds` uses (trigram, else
//! lambda + KN bigram), plus the unigram fallback below. With an empty prefix
//! and no fallback that is exactly the seed level of `--append-only`.
//!
//! One level sweep over everything: pass `T` emits every (prefix, word,
//! suffix) whose level is exactly `T`, across all words, before pass `T + 1`.
//! Output lines are deduplicated across the whole run (the same string can be
//! built from different prefix/word splits; it is emitted at the first, i.e.
//! lowest, level that reaches it). The dedupe key is the line as written
//! (after --case-shape / --enterprise), so the output never repeats a line.
//!
//! Pruning (all lower bounds, so nothing at or below `T` is skipped):
//! * per word `k(E)`: the cheapest the word's tokens after its first can cost
//!   after any prefix (best second-token probability over every context ending
//!   in the first token; fixed contexts after that) plus the cheapest first
//!   step after the word (END included; END only with `--prepend-only`);
//! * per prefix context: the word-first-token candidates in probability order,
//!   with early stop on `L + ceil(-lp0 + k) > T`;
//! * prefix descent: stop when `L + gfloor > T`, `gfloor` = the cheapest any
//!   word (plus its suffix bound) can cost after any context.
//!
//! Seam overlap: the last prefix token may end with a proper prefix (head) of
//! the word, `il` = `i` + head `l`; the rest of the word (`ove`, its best
//! context-free split) is a remainder entry priced in the context `(token
//! before, overlap token)` exactly as a word is. Bounds: a remainder after
//! prefix token `b` costs at least `-best_in[r0] + k` (best first step in any
//! context), so `g_b[b]` gates each node, the per-context `onode` bound and the
//! `-lp0 + k` early stops mirror `insert_part`, and `gfloor` is lowered to the
//! smallest `g_b`, so it still bounds every emission below a prefix node.
//!
//! Unigram fallback for word steps (weight F, `WORD_UNIGRAM_F`): a step of a
//! word's own tokens (the seam step into the word, the steps inside it, and the
//! remainder's steps) that has no trigram or bigram probability is priced
//! `log lambda(ctx) + ln F + ln P_uni(t)` instead of making the word
//! unreachable. `lambda(ctx)` is the KN missing mass the bigram tier already
//! gets (0 in log when the context is no trigram key), `P_uni(t) = (c(t) + 1) /
//! (N + V)` over the full vocabulary from the raw training counts: the floor
//! `score` applies. The prefix and suffix walks are unchanged, so a word whose
//! last token has no next step in any context stays unplaceable. Bounds: every
//! bound above is a maximum of log P over contexts; each gains the tier's
//! ceiling `lam_max + ln F + ln P_uni(t)`: `best_in` (so `gfloor`, `g_b`), the
//! second-step `best1` when the pair has no bigram probability (so `k`), each
//! node's `g` (and `onode`'s) through `gu`, and the per-context merges take
//! the tier as a third stream, one global list in descending `P_uni`, skipping
//! tokens a higher tier prices there. Off for models without KN tiers or
//! unigram counts.
//!
//! Resume: the sweep's position is (level, the set of finished work units at
//! that level). A unit is marked finished only after its output is written, and
//! a checkpoint is taken under the output lock, so (position, lines written,
//! bytes written) always agree. Resuming restores the dedupe set from the lines
//! already written (from `--output`, or from a hash log next to the checkpoint
//! when writing to stdout), redoes the unfinished units and filters out what
//! was already written: a strict run continues byte for byte.

use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::Hasher;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AO};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::case_mask::{apply_op, CaseMask};
use crate::variant::{EnumModel, Variant};
use crate::{enterprise, finalize_decoded, finalized_range, log_msg, lp_to_level, pack, warn_msg,
            CkptSlot, Ctx, DecodeKind, GenerateArgs, MergerStats, Sink, ThreadCkpt, LEVEL_MAX};

/// Slack for f32 rounding between a bound and the exact sequential sum.
const EPS: f32 = 1e-3;
const SHARDS: usize = 64;
/// Work units the empty prefix is split into (by word-first-token rank).
const ROOT_PARTS: usize = 64;
/// No token / no head.
const NONE: u32 = u32::MAX;
/// Lists up to this long are searched linearly by `Tlp`.
const LONG: usize = 24;
/// Weight F of the unigram fallback for a word's own steps (chosen by a crack
/// test; flat between 0.5 and 1.5).
pub const WORD_UNIGRAM_F: f32 = 1.0;
/// Debug hook (undocumented): every lower bound is 0, so the sweep prunes only
/// on exact costs and its output shows anything the bounds would skip.
pub const NO_PRUNE_ENV: &str = "TOKENOV_WORDLIST_NO_PRUNE";

/// Smallest level `ceil(c)` can take for any `c >= x` (up to f32 rounding).
#[inline]
fn lb_level(x: f32, np: bool) -> u32 {
    if !x.is_finite() { return LEVEL_MAX + 1; }
    if np { return 0; }
    if x <= EPS { 0 } else { (x - EPS).ceil() as u32 }
}

#[inline]
fn shard_of(k: u64) -> usize { (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) as usize }

#[inline]
fn line_hash(b: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    h.write(b);
    h.finish()
}

struct Entry {
    toks:  Vec<u32>,
    /// KN continuation `log P_cont(t1 | t0)` (bigram tier), for the second step.
    bi1:   Option<f32>,
    /// Steps 2.. (contexts inside the word, fixed wherever it sits).
    fixed: Vec<f32>,
    /// Lower bound (nats, float) on the word's cost beyond its first token plus
    /// the levels its cheapest next step costs. Sort key within a first token.
    k:     f32,
    /// Unigram fallback: `ln F + ln P_uni(toks[1])`, the second step's
    /// fallback when the pair has no bigram probability (else `None`).
    uni1:  Option<f32>,
    /// `NONE`: the whole word. Otherwise a seam-overlap remainder: `toks` spell
    /// the word minus its first `head_len[head]` bytes, which the last prefix
    /// token supplies.
    head:  u32,
}

/// The unigram fallback tier for a word's own steps.
struct Uni {
    /// Per token id: `ln F + ln P_uni(t)`, `P_uni(t) = (c(t) + 1) / (N + V)`.
    lu:      Vec<f32>,
    /// `max(0, max log lambda)`: the most the tier's `lambda(ctx)` can add.
    lam_max: f32,
}

impl Uni {
    fn from_counts(counts: &[u64], em: &EnumModel, frac: f32) -> Self {
        let total: f64 = counts.iter().map(|&c| c as f64).sum();
        let v = counts.len() as f64;
        let k = crate::SCORE_ADD_K;
        let lnf = frac.ln();
        let lu = counts.iter().map(|&c| lnf + (((c as f64 + k) / (total + k * v)).ln() as f32)).collect();
        let lam_max = em.log_lambda.values().copied().fold(0.0f32, f32::max);
        Uni { lu, lam_max }
    }

    #[inline]
    fn lu(&self, t: u32) -> f32 { self.lu.get(t as usize).copied().unwrap_or(f32::NEG_INFINITY) }
}

/// Transition log-probs with exactly `transition_logprob`'s tiers and values,
/// through per-context hash maps for long child lists (setup only).
struct Tlp<'a> {
    em:  &'a EnumModel,
    tri: Memo<Arc<FxHashMap<u32, f32>>>,
    bi:  Memo<Arc<FxHashMap<u32, f32>>>,
}

impl<'a> Tlp<'a> {
    fn new(em: &'a EnumModel) -> Self { Tlp { em, tri: Memo::new(20_000_000), bi: Memo::new(20_000_000) } }

    fn look(memo: &Memo<Arc<FxHashMap<u32, f32>>>, key: Ctx, list: &[(u32, f32)], t: u32) -> Option<f32> {
        if list.len() <= LONG { return list.iter().find(|(id, _)| *id == t).map(|(_, l)| *l); }
        memo.get_or(key, || {
            let mut m: FxHashMap<u32, f32> = FxHashMap::default();
            m.reserve(list.len());
            for &(id, l) in list { m.entry(id).or_insert(l); }
            (Arc::new(m), list.len())
        }).get(&t).copied()
    }

    /// KN continuation `log P_cont(t | b)` (no lambda).
    fn bi(&self, b: u32, t: u32) -> Option<f32> {
        self.em.bigram.get(&b).and_then(|l| Self::look(&self.bi, b as Ctx, l, t))
    }

    /// Same value as `transition_logprob(em, a, b, t)`.
    fn tlp(&self, a: u32, b: u32, t: u32) -> Option<f32> {
        let em = self.em;
        let ctx = pack(a, b);
        if let Some(ch) = em.trigram.get(&ctx) {
            if let Some(lp) = Self::look(&self.tri, ctx, ch, t) { return Some(lp); }
            if em.is_kn {
                let lam = em.log_lambda.get(&ctx).copied().unwrap_or(0.0);
                if let Some(x) = self.bi(b, t) { return Some(lam + x); }
            }
            None
        } else if em.is_kn { self.bi(b, t) } else { None }
    }

    /// `tlp`, else (with `uni`) the unigram fallback `lambda(ctx) + ln F + ln P_uni(t)`.
    fn tlp_fb(&self, uni: Option<&Uni>, a: u32, b: u32, t: u32) -> Option<f32> {
        if let Some(lp) = self.tlp(a, b, t) { return Some(lp); }
        let u = uni?;
        let lam = self.em.log_lambda.get(&pack(a, b)).copied().unwrap_or(0.0);
        let lp = lam + u.lu(t);
        if lp.is_finite() { Some(lp) } else { None }
    }
}

/// `lat[i]` = every `(j, id)` whose token bytes are exactly `w[i..j]`.
fn lattice(w: &[u8], vocab: &FxHashMap<Vec<u8>, u32>, maxlen: usize) -> Vec<Vec<(usize, u32)>> {
    (0..w.len()).map(|i| (i + 1..=(i + maxlen).min(w.len()))
        .filter_map(|j| vocab.get(&w[i..j]).map(|&id| (j, id))).collect()).collect()
}

fn path_cmp(x: &(f32, Vec<u32>), y: &(f32, Vec<u32>)) -> std::cmp::Ordering {
    x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| x.1.cmp(&y.1))
}

/// The `k` cheapest ways to spell the lattice's word as at most `max_tok`
/// tokens. Cost = sum of `step(a, b, t)` (nats, `None` = impossible) from
/// context `(a0, b0)`, plus `fin(a, b)` after the last token. Exact k-best over
/// states (position, last two tokens), except that the token cap is applied as
/// paths grow. Ties break on the token ids, so the result is deterministic.
fn kbest(lat: &[Vec<(usize, u32)>], k: usize, max_tok: usize, a0: u32, b0: u32,
         step: &dyn Fn(u32, u32, u32) -> Option<f32>, fin: &dyn Fn(u32, u32) -> Option<f32>)
         -> Vec<(f32, Vec<u32>)> {
    let n = lat.len();
    if n == 0 || k == 0 || max_tok == 0 { return Vec::new(); }
    type Paths = Vec<(f32, Vec<u32>)>;
    let mut st: Vec<FxHashMap<(u32, u32), Paths>> = (0..=n).map(|_| FxHashMap::default()).collect();
    st[0].insert((a0, b0), vec![(0.0, Vec::new())]);
    for i in 0..n {
        let mut cur: Vec<((u32, u32), Paths)> = std::mem::take(&mut st[i]).into_iter().collect();
        cur.sort_by(|x, y| x.0.cmp(&y.0));
        for ((a, b), paths) in cur {
            for &(j, t) in &lat[i] {
                let Some(c) = step(a, b, t) else { continue };
                if !c.is_finite() { continue; }
                let slot = st[j].entry((b, t)).or_default();
                for (pc, p) in &paths {
                    if p.len() >= max_tok { continue; }
                    let mut q = Vec::with_capacity(p.len() + 1);
                    q.extend_from_slice(p);
                    q.push(t);
                    slot.push((pc + c, q));
                }
                if slot.len() > k { slot.sort_by(path_cmp); slot.truncate(k); }
            }
        }
    }
    let mut out: Paths = Vec::new();
    for ((a, b), paths) in std::mem::take(&mut st[n]) {
        let Some(f) = fin(a, b) else { continue };
        if !f.is_finite() { continue; }
        for (pc, p) in paths { out.push((pc + f, p)); }
    }
    out.sort_by(path_cmp);
    out.truncate(k);
    out
}

/// Heads (proper prefixes of words) that are proper suffixes of `tok`: the
/// seam overlaps a prefix token with these bytes can supply.
fn overlap_heads(tok: &[u8], heads: &FxHashMap<Vec<u8>, u32>) -> Vec<u32> {
    (1..tok.len()).filter_map(|i| heads.get(&tok[i..]).copied()).collect()
}

/// Which stream holds the next token in descending log P: 0 trigram (value
/// as is), 1 bigram, 2 unigram (both `lam +` value), 3 none. Ties go to the
/// earlier tier; with the unigram stream empty this is the two-tier merge.
#[inline]
fn pick3(tri: Option<(u32, f32)>, bi: Option<(u32, f32)>, uni: Option<(u32, f32)>, lam: f32) -> u8 {
    let (mut pick, mut best) = (3u8, f32::NEG_INFINITY);
    if let Some(x) = tri { pick = 0; best = x.1; }
    if let Some(y) = bi { let v = lam + y.1; if pick == 3 || v > best { pick = 1; best = v; } }
    if let Some(z) = uni { let v = lam + z.1; if pick == 3 || v > best { pick = 2; } }
    pick
}

/// Exact log P of an entry placed after context `(_, b)` whose first token
/// `t = e.toks[0]` costs `lp0` there, summed in the order build_seeds sums it.
#[inline]
fn block_lp(em: &EnumModel, b: u32, t: u32, lp0: f32, e: &Entry) -> Option<f32> {
    let mut lpe: f32 = 0.0;
    lpe += lp0;
    if e.toks.len() >= 2 {
        let c1 = pack(b, t);
        let s1 = match em.trigram.get(&c1) {
            Some(list) => list.iter().find(|(id, _)| *id == e.toks[1]).map(|(_, l)| *l)
                .or_else(|| e.bi1.map(|x| em.log_lambda.get(&c1).copied().unwrap_or(0.0) + x))
                .or_else(|| e.uni1.map(|x| em.log_lambda.get(&c1).copied().unwrap_or(0.0) + x)),
            None => e.bi1.or(e.uni1),
        };
        lpe += s1?;
        for &f in &e.fixed { lpe += f; }
    }
    if lpe.is_finite() { Some(lpe) } else { None }
}

/// Word-first-token candidates of one prefix context, trigram tier only (the
/// bigram tier is shared per last token, see `Prep::bi_f`).
struct NodeInfo {
    /// `(token, log P)` for first tokens of words, descending.
    tri_f: Box<[(u32, f32)]>,
    /// The same ids, sorted, for the "already in the trigram tier" test.
    ids:   Box<[u32]>,
    /// `log lambda(ctx)` (0 when the context is no trigram key).
    lam:   f32,
    /// Lower bound on `ceil(word cost) + suffix bound` over all words here.
    g:     u32,
}

/// Sharded, size-capped memo shared by all workers. Past `max` elements it
/// stops inserting and computes on every miss (output is unaffected).
struct Memo<V> {
    shards: Vec<RwLock<FxHashMap<Ctx, V>>>,
    size:   AtomicUsize,
    max:    usize,
}

impl<V: Clone> Memo<V> {
    fn new(max: usize) -> Self {
        Memo { shards: (0..SHARDS).map(|_| RwLock::new(FxHashMap::default())).collect(),
               size: AtomicUsize::new(0), max }
    }
    fn get_or(&self, k: Ctx, f: impl FnOnce() -> (V, usize)) -> V {
        let s = &self.shards[shard_of(k)];
        if let Some(v) = s.read().unwrap().get(&k) { return v.clone(); }
        let (v, w) = f();
        if self.size.load(AO::Relaxed) < self.max {
            s.write().unwrap().insert(k, v.clone());
            self.size.fetch_add(w.max(1), AO::Relaxed);
        }
        v
    }
}

/// Seam overlap: remainder entries placed after a prefix token whose bytes
/// end with the word's head (`il` then `ove` for `love`).
struct Overlap {
    rentries:  Vec<Entry>,
    /// `(head, first remainder token)` -> `rentries` indices, ascending `k`.
    groups:    FxHashMap<(u32, u32), Box<[u32]>>,
    /// Per prefix token `b`: the heads its bytes end with (properly).
    heads_of:  Vec<Box<[u32]>>,
    /// Per `b`: min `k` over its heads' entries (float).
    kmin_b:    Vec<f32>,
    /// Per `b`: lower bound on any remainder's level after `b` (best first
    /// step over every context, plus `k`).
    g_b:       Vec<u32>,
    /// Per `b`: bigram tier `(r0, log P_cont)` for its heads' first tokens.
    bov:       Memo<Arc<[(u32, f32)]>>,
    /// Per context: the trigram tier of the same, with `lam` and a bound.
    onodes:    Memo<Arc<NodeInfo>>,
    /// Unigram fallback: per head, its remainders' first tokens.
    r0_of:     Vec<Box<[u32]>>,
    /// Per `b`: unigram tier `(r0, ln F + ln P_uni)` for its heads' first
    /// tokens without a bigram probability after `b`, descending.
    uov:       Memo<Arc<[(u32, f32)]>>,
}

impl Overlap {
    #[inline]
    fn has_r0(&self, b: u32, r0: u32) -> bool {
        self.heads_of[b as usize].iter().any(|&h| self.groups.contains_key(&(h, r0)))
    }

    fn bov(&self, em: &EnumModel, b: u32) -> Arc<[(u32, f32)]> {
        self.bov.get_or(b as Ctx, || {
            let mut f: Vec<(u32, f32)> = match em.bigram.get(&b) {
                Some(l) => l.iter().filter(|(t, _)| self.has_r0(b, *t)).copied().collect(),
                None => Vec::new(),
            };
            f.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal).then(x.0.cmp(&y.0)));
            let n = f.len();
            (Arc::from(f.into_boxed_slice()), n)
        })
    }

    fn uov(&self, em: &EnumModel, uni: Option<&Uni>, b: u32) -> Arc<[(u32, f32)]> {
        let Some(u) = uni else { return Arc::from(Vec::new().into_boxed_slice()) };
        self.uov.get_or(b as Ctx, || {
            let bov = self.bov(em, b);
            let mut f: Vec<(u32, f32)> = Vec::new();
            for &h in self.heads_of[b as usize].iter() {
                for &r0 in self.r0_of[h as usize].iter() {
                    if bov.iter().any(|x| x.0 == r0) || f.iter().any(|x| x.0 == r0) { continue; }
                    let v = u.lu(r0);
                    if v.is_finite() { f.push((r0, v)); }
                }
            }
            f.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal).then(x.0.cmp(&y.0)));
            let n = f.len();
            (Arc::from(f.into_boxed_slice()), n)
        })
    }

    fn onode(&self, em: &EnumModel, uni: Option<&Uni>, np: bool, a: u32, b: u32) -> Arc<NodeInfo> {
        let ctx = pack(a, b);
        self.onodes.get_or(ctx, || {
            let (mut tri_f, lam) = match em.trigram.get(&ctx) {
                Some(l) => (l.iter().filter(|(t, _)| self.has_r0(b, *t)).copied().collect::<Vec<_>>(),
                            em.log_lambda.get(&ctx).copied().unwrap_or(0.0)),
                None => (Vec::new(), 0.0),
            };
            tri_f.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal)
                .then(x.0.cmp(&y.0)));
            // Bound: the best first step here (either tier) plus the smallest
            // `k` of any remainder after `b`.
            let km = self.kmin_b[b as usize];
            let mut g = tri_f.first().map(|&(_, lp)| lb_level(-lp + km, np)).unwrap_or(LEVEL_MAX + 1);
            if em.is_kn {
                if let Some(&(_, lp)) = self.bov(em, b).first() { g = g.min(lb_level(-(lam + lp) + km, np)); }
            }
            if let Some(&(_, lu)) = self.uov(em, uni, b).first() { g = g.min(lb_level(-(lam + lu) + km, np)); }
            let mut ids: Vec<u32> = tri_f.iter().map(|&(t, _)| t).collect();
            ids.sort_unstable();
            let w = tri_f.len() * 3 + 4;
            (Arc::new(NodeInfo { tri_f: tri_f.into_boxed_slice(), ids: ids.into_boxed_slice(), lam, g }), w)
        })
    }
}

/// What a run emits, besides the model.
pub struct Opts {
    pub max_tokens:   usize,
    pub min_tokens:   usize,
    pub min_len:      usize,
    pub max_len:      usize,
    pub enterprise:   bool,
    /// `--case-shape` patterns; empty = candidates as generated.
    pub case_masks:   Vec<CaseMask>,
    /// `--prepend-only`: the word ends the candidate (no suffix walk).
    pub word_at_end:  bool,
    /// Weight F of the unigram fallback for word steps; `None` = off.
    pub word_unigram: Option<f32>,
    /// Every lower bound 0 (debug / tests).
    pub no_prune:     bool,
}

/// The model side of a run: what the tables are built from.
pub struct Inputs<'a> {
    pub em:          &'a EnumModel,
    pub variant:     &'a dyn Variant,
    pub decode:      &'a [Vec<u8>],
    pub kind:        DecodeKind,
    pub start_id:    u32,
    pub end_id:      u32,
    pub unigram_raw: &'a [u64],
}

/// The read-only tables of a run, shared by all workers.
pub struct Prep<'a> {
    em:          &'a EnumModel,
    variant:     &'a dyn Variant,
    decode:      &'a [Vec<u8>],
    has_nl:      Vec<bool>,
    kind:        DecodeKind,
    start_id:    u32,
    end_id:      u32,
    entries:     Vec<Entry>,
    ov:          Option<Overlap>,
    by_first:    Vec<Vec<u32>>,
    kmin_t:      Vec<f32>,
    kmin_all:    f32,
    /// Per last token `b`: the bigram tier's `(t, log P_cont)` for word-first
    /// tokens, descending.
    bi_f:        Vec<Box<[(u32, f32)]>>,
    /// Per `b`: min over `bi_f[b]` of `-lp + kmin_t` (float).
    gbi:         Vec<f32>,
    gfloor:      u32,
    /// Unigram fallback: the tier, the word-first tokens in descending
    /// `ln F + ln P_uni`, the sorted ids of each `bi_f[b]` (the tokens the
    /// bigram tier prices after `b`), and min over `uf` of `-lu + kmin_t`.
    uni:         Option<Uni>,
    uf:          Vec<(u32, f32)>,
    bi_ids:      Vec<Box<[u32]>>,
    gu:          f32,
    kids:        Memo<Arc<[(u32, f32)]>>,
    nodes:       Memo<Arc<NodeInfo>>,
    opts:        Opts,
}

impl Prep<'_> {
    fn kids(&self, a: u32, b: u32) -> Arc<[(u32, f32)]> {
        let ctx = pack(a, b);
        self.kids.get_or(ctx, || {
            let v = self.variant.get_children(self.em, ctx, b);
            let n = v.len();
            (Arc::from(v.into_boxed_slice()), n)
        })
    }

    fn node(&self, a: u32, b: u32) -> Arc<NodeInfo> {
        let ctx = pack(a, b);
        let np = self.opts.no_prune;
        self.nodes.get_or(ctx, || {
            let em = self.em;
            let (mut tri_f, lam) = match em.trigram.get(&ctx) {
                Some(l) => (l.iter().filter(|(t, _)| self.is_first(*t)).copied().collect::<Vec<_>>(),
                            em.log_lambda.get(&ctx).copied().unwrap_or(0.0)),
                None => (Vec::new(), 0.0),
            };
            // Already descending (trigram lists are sorted); keep it explicit.
            tri_f.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal)
                .then(x.0.cmp(&y.0)));
            let mut g = tri_f.iter().map(|&(t, lp)| lb_level(-lp + self.kmin_t[t as usize], np))
                .min().unwrap_or(LEVEL_MAX + 1);
            if em.is_kn {
                g = g.min(lb_level(-lam + self.gbi[b as usize], np));
            }
            if self.uni.is_some() { g = g.min(lb_level(-lam + self.gu, np)); }
            let mut ids: Vec<u32> = tri_f.iter().map(|&(t, _)| t).collect();
            ids.sort_unstable();
            let w = tri_f.len() * 3 + 4;
            (Arc::new(NodeInfo { tri_f: tri_f.into_boxed_slice(), ids: ids.into_boxed_slice(), lam, g }), w)
        })
    }

    #[inline]
    fn is_first(&self, t: u32) -> bool {
        self.kmin_t.get(t as usize).map(|k| k.is_finite()).unwrap_or(false)
    }

    #[inline]
    fn lb(&self, x: f32) -> u32 { lb_level(x, self.opts.no_prune) }

    /// Work units of one pass: the empty prefix (in `ROOT_PARTS` slices),
    /// then each first prefix token.
    #[cfg(test)]
    fn n_units(&self) -> usize { ROOT_PARTS + self.kids(self.start_id, self.start_id).len() }
}

/// Build the tables. `entry_seqs` = the wordlist's token sequences, in file order.
pub fn prepare<'a>(inp: Inputs<'a>, entry_seqs: &[Vec<u32>], opts: Opts) -> Result<Prep<'a>> {
    let t_setup = Instant::now();
    let Inputs { em, variant, decode, kind, start_id, end_id, unigram_raw } = inp;
    let np = opts.no_prune;
    let lbl = |x: f32| lb_level(x, np);
    let at_end = opts.word_at_end;
    let n_ids = (start_id.max(end_id) as usize + 1).max(decode.len());
    let uni: Option<Uni> = match opts.word_unigram {
        Some(f) if f > 0.0 && f.is_finite() && em.is_kn && !unigram_raw.is_empty() => {
            let u = Uni::from_counts(unigram_raw, em, f);
            log_msg(&format!("[wordlist] unigram fallback for word steps: F={f} lam_max={:.4} over {} tokens",
                u.lam_max, u.lu.len()));
            Some(u)
        }
        Some(_) => {
            log_msg("[wordlist] unigram fallback for word steps: off (the model has no Kneser-Ney tiers \
                     or no unigram counts)");
            None
        }
        None => None,
    };

    // Words: dedupe by token sequence, keep input order, drop over-long ones
    // (as build_seeds does).
    let mut seen_seq: FxHashSet<Vec<u32>> = FxHashSet::default();
    let mut seqs: Vec<Vec<u32>> = Vec::new();
    let mut overlong = 0usize;
    for s in entry_seqs {
        if s.is_empty() { continue; }
        if s.len() > opts.max_tokens { overlong += 1; continue; }
        if seen_seq.insert(s.clone()) { seqs.push(s.clone()); }
    }
    drop(seen_seq);

    // Word ids = canonical sequences. Every entry sequence below carries its
    // word id and its head (`NONE` = the whole word).
    let n_words = seqs.len();
    let words: Vec<Vec<u8>> = seqs.iter().map(|s| {
        let mut w = Vec::new();
        for &t in s { if let Some(b) = decode.get(t as usize) { w.extend_from_slice(b); } }
        w
    }).collect();
    let mut head_of: Vec<u32> = vec![NONE; n_words];
    let mut head_ids: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
    let mut head_len: Vec<u32> = Vec::new();
    let tlp = Tlp::new(em);
    let has_nl: Vec<bool> = decode.iter().map(|b| b.iter().any(|&c| c == b'\n' || c == b'\r')).collect();

    // Vocab by bytes, for splitting words (lowest id wins a byte collision).
    let mut vocab: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
    let mut maxlen = 0usize;
    let mut collide = 0usize;
    for (id, b) in decode.iter().enumerate() {
        let id = id as u32;
        if id >= start_id || id == end_id || b.is_empty() || has_nl[id as usize] { continue; }
        if vocab.contains_key(b) { collide += 1; continue; }
        vocab.insert(b.clone(), id);
        maxlen = maxlen.max(b.len());
    }
    // Best log P of every token in any context (either tier): the first step
    // of the context-free split ranking.
    let best_any: Vec<f32> = {
        let lists: Vec<&Vec<(u32, f32)>> = em.trigram.values().chain(em.bigram.values()).collect();
        let chunk = (lists.len() / 64).max(1);
        lists.par_chunks(chunk).map(|part| {
            let mut v = vec![f32::NEG_INFINITY; n_ids];
            for l in part { for &(t, lp) in l.iter() { if (t as usize) < n_ids && lp > v[t as usize] { v[t as usize] = lp; } } }
            v
        }).reduce(|| vec![f32::NEG_INFINITY; n_ids], |mut x, y| {
            for i in 0..x.len() { if y[i] > x[i] { x[i] = y[i]; } }
            x
        })
    };
    // Context-free split ranking (best first step anywhere, KN continuation
    // for the second, exact after that; a ranking heuristic only).
    let cf_step = |a: u32, b: u32, t: u32| -> Option<f32> {
        if b == NONE { best_any.get(t as usize).filter(|v| v.is_finite()).map(|v| -v) }
        else if a == NONE { tlp.bi(b, t).map(|x| -x) }
        else { tlp.tlp(a, b, t).map(|x| -x) }
    };
    let fin0 = |_: u32, _: u32| Some(0.0f32);
    // With the unigram fallback: the same ranking, else the tier at lambda 1.
    let cf_fb = |a: u32, b: u32, t: u32| -> Option<f32> {
        cf_step(a, b, t).or_else(|| uni.as_ref().map(|u| -u.lu(t)).filter(|v| v.is_finite()))
    };

    // Seam overlap: for every head of a word that some token ends with
    // (properly), the best context-free split of the rest of the word.
    let mut wid_of: Vec<u32> = (0..n_words as u32).collect();
    {
        let mut sfx: FxHashSet<&[u8]> = FxHashSet::default();
        for b in vocab.keys() { for i in 1..b.len() { sfx.insert(&b[i..]); } }
        let rems: Vec<Vec<(usize, Vec<Vec<u32>>)>> = (0..n_words).into_par_iter().map(|i| {
            let w = &words[i];
            let mut out = Vec::new();
            for j in 1..w.len() {
                if !sfx.contains(&w[..j]) { continue; }
                let lat = lattice(&w[j..], &vocab, maxlen);
                let mut ps: Vec<Vec<u32>> = kbest(&lat, 1, opts.max_tokens.saturating_sub(1), NONE, NONE, &cf_step, &fin0)
                    .into_iter().map(|x| x.1).collect();
                // Unigram fallback: only a head with no split at all retries,
                // so the splits chosen without it stay as they are.
                if ps.is_empty() && uni.is_some() {
                    ps = kbest(&lat, 1, opts.max_tokens.saturating_sub(1), NONE, NONE, &cf_fb, &fin0)
                        .into_iter().map(|x| x.1).collect();
                }
                if !ps.is_empty() { out.push((j, ps)); }
            }
            out
        }).collect();
        for (i, list) in rems.into_iter().enumerate() {
            for (j, ps) in list {
                let n = head_ids.len() as u32;
                let h = *head_ids.entry(words[i][..j].to_vec()).or_insert_with(|| { head_len.push(j as u32); n });
                for p in ps { seqs.push(p); wid_of.push(i as u32); head_of.push(h); }
            }
        }
    }
    drop(wid_of);
    log_msg(&format!("[wordlist] splits: vocab {} byte strings ({} id collisions), {} words -> {} entry sequences \
                      ({} remainders over {} heads) in {:.2}s",
        vocab.len(), collide, n_words, seqs.len(), head_of.iter().filter(|&&h| h != NONE).count(),
        head_len.len(), t_setup.elapsed().as_secs_f64()));

    let mut is_first = vec![false; n_ids];
    let mut single = vec![false; n_ids];
    let mut multi_first = vec![false; n_ids];
    let mut pair_idx: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    for s in &seqs {
        is_first[s[0] as usize] = true;
        if s.len() == 1 { single[s[0] as usize] = true; }
        else {
            multi_first[s[0] as usize] = true;
            let n = pair_idx.len() as u32;
            pair_idx.entry((s[0], s[1])).or_insert(n);
        }
    }
    let n_pairs = pair_idx.len();

    // One pass over the trigram tier: best log P of each word-first token in
    // any context, best second step for each (t0, t1) over contexts ending in
    // t0, and the cheapest first child (END with --prepend-only) after a
    // 1-token word in any context.
    let tri: Vec<(&Ctx, &Vec<(u32, f32)>)> = em.trigram.iter().collect();
    let chunk = (tri.len() / 64).max(1);
    let (mut best_in, mut best1, mut sufmin1) = tri.par_chunks(chunk).map(|part| {
        let mut bi = vec![f32::NEG_INFINITY; n_ids];
        let mut b1 = vec![f32::NEG_INFINITY; n_pairs];
        let mut sm = vec![u32::MAX; n_ids];
        for &(&ctx, list) in part {
            let b = (ctx & 0xFFFF_FFFF) as u32;
            for &(t, lp) in list.iter() {
                if is_first[t as usize] && lp > bi[t as usize] { bi[t as usize] = lp; }
            }
            if (b as usize) < n_ids && multi_first[b as usize] {
                for &(u, lp) in list.iter() {
                    if let Some(&pi) = pair_idx.get(&(b, u)) {
                        if lp > b1[pi as usize] { b1[pi as usize] = lp; }
                    }
                }
            }
            if (b as usize) < n_ids && single[b as usize] {
                let next = if at_end { list.iter().find(|x| x.0 == end_id) } else { list.first() };
                if let Some(&(_, lp)) = next {
                    let v = lp_to_level(lp);
                    if v < sm[b as usize] { sm[b as usize] = v; }
                }
            }
        }
        (bi, b1, sm)
    }).reduce(|| (vec![f32::NEG_INFINITY; n_ids], vec![f32::NEG_INFINITY; n_pairs], vec![u32::MAX; n_ids]),
              |mut x, y| {
        for i in 0..x.0.len() { if y.0[i] > x.0[i] { x.0[i] = y.0[i]; } }
        for i in 0..x.1.len() { if y.1[i] > x.1[i] { x.1[i] = y.1[i]; } }
        for i in 0..x.2.len() { if y.2[i] < x.2[i] { x.2[i] = y.2[i]; } }
        x
    });
    drop(tri);

    // Bigram tier: same three, plus the per-`b` word-first lists.
    let mut bi1v: Vec<Option<f32>> = vec![None; n_pairs];
    let mut bi_f: Vec<Box<[(u32, f32)]>> = (0..n_ids).map(|_| Box::<[(u32, f32)]>::default()).collect();
    if em.is_kn {
        for (&b, list) in em.bigram.iter() {
            let bu = b as usize;
            if bu >= n_ids { continue; }
            let mut f: Vec<(u32, f32)> = Vec::new();
            for &(t, lp) in list.iter() {
                if is_first[t as usize] {
                    f.push((t, lp));
                    if lp > best_in[t as usize] { best_in[t as usize] = lp; }
                }
                if multi_first[bu] {
                    if let Some(&pi) = pair_idx.get(&(b, t)) {
                        bi1v[pi as usize] = Some(lp);
                        if lp > best1[pi as usize] { best1[pi as usize] = lp; }
                    }
                }
            }
            if single[bu] {
                let next = if at_end { list.iter().find(|x| x.0 == end_id) } else { list.first() };
                if let Some(&(_, lp)) = next {
                    let v = lp_to_level(lp);
                    if v < sufmin1[bu] { sufmin1[bu] = v; }
                }
            }
            f.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal).then(x.0.cmp(&y.0)));
            bi_f[bu] = f.into_boxed_slice();
        }
        // Tail variants (`--unigram-tail`, cap-tail) add unigram children in
        // every context, at most `max log lambda + unigram_logw + lp` (log
        // lambda <= 0), so the step after a 1-token word can be that cheap
        // anywhere. Without a tail `unigram_top` is empty and nothing changes.
        if !em.unigram_top.is_empty() {
            let lam_max = em.log_lambda.values().copied().fold(0.0f32, f32::max);
            let best = if at_end {
                em.unigram_top.iter().find(|x| x.0 == end_id).map(|x| x.1)
            } else {
                em.unigram_top.iter().map(|x| x.1).reduce(f32::max)
            };
            if let Some(lp) = best {
                let v = lp_to_level(lam_max + em.unigram_logw + lp);
                for (t, sm) in sufmin1.iter_mut().enumerate() {
                    if single[t] && v < *sm { *sm = v; }
                }
            }
        }
    }

    // Unigram fallback: a first token can also be priced by the tier, at most
    // `lam_max + ln F + ln P_uni(t)` in any context.
    if let Some(u) = &uni {
        for t in 0..n_ids {
            if is_first[t] { let v = u.lam_max + u.lu(t as u32); if v > best_in[t] { best_in[t] = v; } }
        }
    }

    // Per-word data. Words no context can ever complete are dropped
    // (--append-only drops the unreachable ones too).
    // Cheapest first child per context (END with --prepend-only).
    let first_kid: Memo<Option<f32>> = Memo::new(50_000_000);
    let built: Vec<Option<Entry>> = seqs.par_iter().enumerate().map(|(si, s)| {
        let head = head_of[si];
        if s.len() == 1 {
            let sm = sufmin1[s[0] as usize];
            if sm == u32::MAX { return None; }
            return Some(Entry { toks: s.clone(), bi1: None, fixed: Vec::new(), k: sm as f32, uni1: None, head });
        }
        let pi = pair_idx[&(s[0], s[1])] as usize;
        // The fallback prices the second step only when the pair has no
        // bigram probability (the bigram tier catches it otherwise).
        let uni1 = match &uni { Some(u) if bi1v[pi].is_none() => Some(u.lu(s[1])).filter(|v| v.is_finite()), _ => None };
        let b1 = match (&uni, uni1) { (Some(u), Some(x)) => best1[pi].max(u.lam_max + x), _ => best1[pi] };
        if !b1.is_finite() { return None; }
        let mut fixed = Vec::with_capacity(s.len() - 2);
        for k in 2..s.len() {
            match tlp.tlp_fb(uni.as_ref(), s[k - 2], s[k - 1], s[k]) {
                Some(lp) => fixed.push(lp),
                None => return None,
            }
        }
        let (x, y) = (s[s.len() - 2], s[s.len() - 1]);
        let first = first_kid.get_or(pack(x, y), || {
            let ch = variant.get_children(em, pack(x, y), y);
            let next = if at_end { ch.iter().find(|c| c.0 == end_id) } else { ch.first() };
            (next.map(|c| c.1), 1)
        });
        let first = first?;
        let rem: f32 = -b1 - fixed.iter().sum::<f32>();
        Some(Entry { toks: s.clone(), bi1: bi1v[pi], fixed, k: rem.max(0.0) + lp_to_level(first) as f32, uni1, head })
    }).collect();
    drop(first_kid);
    let n_dropped = built[..n_words].iter().filter(|b| b.is_none()).count();
    let mut entries: Vec<Entry> = Vec::new();
    let mut rentries: Vec<Entry> = Vec::new();
    for e in built.into_iter().flatten() {
        if e.head == NONE { entries.push(e) } else { rentries.push(e) }
    }
    if entries.is_empty() && rentries.is_empty() {
        bail!("none of the {} --wordlist entries can be generated with this model{}",
            entry_seqs.len(),
            if overlong > 0 { format!(" ({overlong} are longer than --max-tokens)") } else { String::new() });
    }

    let mut kmin_t = vec![f32::INFINITY; n_ids];
    let mut by_first: Vec<Vec<u32>> = vec![Vec::new(); n_ids];
    for (i, e) in entries.iter().enumerate() {
        let t = e.toks[0] as usize;
        by_first[t].push(i as u32);
        if e.k < kmin_t[t] { kmin_t[t] = e.k; }
    }
    for l in by_first.iter_mut() {
        l.sort_by(|&x, &y| entries[x as usize].k.partial_cmp(&entries[y as usize].k)
            .unwrap_or(std::cmp::Ordering::Equal).then(x.cmp(&y)));
    }
    let kmin_all = kmin_t.iter().copied().fold(f32::INFINITY, f32::min);
    let gbi: Vec<f32> = bi_f.iter().map(|l| l.iter()
        .map(|&(t, lp)| -lp + kmin_t[t as usize]).fold(f32::INFINITY, f32::min)).collect();
    let mut gfloor = (0..n_ids).filter(|&t| kmin_t[t].is_finite())
        .map(|t| lbl(-best_in[t] + kmin_t[t])).min().unwrap_or(LEVEL_MAX + 1);
    // Seam overlap: group the remainders by (head, first token) and bound them
    // per prefix token. A remainder after prefix token `b` costs at least its
    // first token's best log P in any context plus its `k`, so `g_b[b]` (and
    // `gfloor`, lowered to cover it) stay lower bounds.
    let ov = {
        let n_heads = head_len.len();
        let mut groups: FxHashMap<(u32, u32), Vec<u32>> = FxHashMap::default();
        let mut hk = vec![f32::INFINITY; n_heads];
        let mut hg = vec![f32::INFINITY; n_heads];
        for (ri, e) in rentries.iter().enumerate() {
            let (h, r0) = (e.head as usize, e.toks[0]);
            groups.entry((e.head, r0)).or_default().push(ri as u32);
            hk[h] = hk[h].min(e.k);
            let bi = best_in.get(r0 as usize).copied().unwrap_or(f32::NEG_INFINITY);
            hg[h] = hg[h].min(-bi + e.k);
        }
        let groups: FxHashMap<(u32, u32), Box<[u32]>> = groups.into_iter().map(|(key, mut l)| {
            l.sort_by(|&x, &y| rentries[x as usize].k.partial_cmp(&rentries[y as usize].k)
                .unwrap_or(std::cmp::Ordering::Equal).then(x.cmp(&y)));
            (key, l.into_boxed_slice())
        }).collect();
        let mut heads_of: Vec<Box<[u32]>> = Vec::with_capacity(n_ids);
        let mut kmin_b = vec![f32::INFINITY; n_ids];
        let mut g_b = vec![LEVEL_MAX + 1; n_ids];
        for b in 0..n_ids {
            let hs: Vec<u32> = match decode.get(b) {
                Some(bytes) if (b as u32) < start_id && b as u32 != end_id =>
                    overlap_heads(bytes, &head_ids).into_iter().filter(|&h| hk[h as usize].is_finite()).collect(),
                _ => Vec::new(),
            };
            for &h in &hs {
                kmin_b[b] = kmin_b[b].min(hk[h as usize]);
                g_b[b] = g_b[b].min(lbl(hg[h as usize]));
            }
            heads_of.push(hs.into_boxed_slice());
        }
        let mut r0_of: Vec<Vec<u32>> = vec![Vec::new(); n_heads];
        if uni.is_some() {
            for &(h, r0) in groups.keys() { r0_of[h as usize].push(r0); }
            for l in r0_of.iter_mut() { l.sort_unstable(); }
        }
        let gov = g_b.iter().copied().min().unwrap_or(LEVEL_MAX + 1);
        log_msg(&format!("[wordlist] seam overlap: {} remainders in {} groups, {} prefix tokens with heads, \
                          floor {} (word floor {})",
            rentries.len(), groups.len(), heads_of.iter().filter(|h| !h.is_empty()).count(), gov, gfloor));
        gfloor = gfloor.min(gov);
        Some(Overlap { rentries, groups, heads_of, kmin_b, g_b,
                       bov: Memo::new(30_000_000), onodes: Memo::new(30_000_000),
                       r0_of: r0_of.into_iter().map(|l| l.into_boxed_slice()).collect(),
                       uov: Memo::new(30_000_000) })
    };
    // Bound vectors no longer needed.
    drop(best_in); drop(best1); drop(sufmin1);
    // Re-filter the bigram lists to tokens that still start a usable word.
    for l in bi_f.iter_mut() {
        if l.iter().any(|(t, _)| !kmin_t[*t as usize].is_finite()) {
            *l = l.iter().filter(|(t, _)| kmin_t[*t as usize].is_finite()).copied().collect();
        }
    }
    // Unigram fallback streams: word-first tokens by descending tier value,
    // and per `b` the ids the bigram tier already prices.
    let (mut uf, mut bi_ids, mut gu) = (Vec::new(), Vec::new(), f32::INFINITY);
    if let Some(u) = &uni {
        uf = (0..n_ids as u32).filter(|&t| kmin_t[t as usize].is_finite())
            .map(|t| (t, u.lu(t))).filter(|x| x.1.is_finite()).collect();
        uf.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal).then(x.0.cmp(&y.0)));
        gu = uf.iter().map(|&(t, lu)| -lu + kmin_t[t as usize]).fold(f32::INFINITY, f32::min);
        bi_ids = bi_f.iter().map(|l| {
            let mut v: Vec<u32> = l.iter().map(|x| x.0).collect();
            v.sort_unstable();
            v.into_boxed_slice()
        }).collect();
        log_msg(&format!("[wordlist] unigram fallback: {} first tokens, gu={:.2}", uf.len(), gu));
    }
    log_msg(&format!(
        "[wordlist] {} words placed ({} duplicate/over-long skipped: {} over --max-tokens; {} cannot be \
         generated with this model), kmin={:.2} gfloor={} word_at_end={} (prepared in {:.2}s)",
        entries.len(), entry_seqs.len() - n_words, overlong, n_dropped, kmin_all, gfloor, at_end,
        t_setup.elapsed().as_secs_f64()));

    Ok(Prep {
        em, variant, decode, has_nl, kind, start_id, end_id,
        entries, ov, by_first, kmin_t, kmin_all, bi_f, gbi, gfloor,
        uni, uf, bi_ids, gu,
        kids: Memo::new(120_000_000),
        nodes: Memo::new(30_000_000),
        opts,
    })
}

// ----------------------------------------------------------------------------
// Run state and the sweep
// ----------------------------------------------------------------------------

/// Output side of a run, behind one lock: the sink, what has been written to
/// it, the hash log (stdout runs with a checkpoint) and the progress sidecar.
struct Out {
    w:     Box<dyn Write + Send>,
    lines: u64,
    bytes: u64,
    log:   Option<BufWriter<File>>,
    side:  Option<Sidecar>,
}

/// `<output>.progress`, written inline when the run keeps no checkpoint.
struct Sidecar {
    path:       PathBuf,
    fp:         String,
    last:       Instant,
    last_lines: u64,
}

/// Where the sweep is: the level, which of its units are finished (output
/// written), and whether any finished unit held work back for a later level.
#[derive(Clone, Debug, Default, PartialEq)]
struct Pos {
    level:    u32,
    units:    usize,
    done:     Vec<bool>,
    deferred: bool,
}

struct Rt {
    seen:        Vec<Mutex<FxHashSet<u64>>>,
    emitted:     AtomicU64,
    target:      u64,
    stop:        AtomicBool,
    out:         Mutex<Out>,
    pos:         Mutex<Pos>,
    stats:       Arc<MergerStats>,
    flush_bytes: usize,
}

impl Rt {
    fn new(w: Box<dyn Write + Send>, target: u64, flush_bytes: usize) -> Self {
        let seen_cap = if target < (1 << 32) { (target as usize) / SHARDS + 1 } else { 0 };
        Rt {
            seen: (0..SHARDS).map(|_| Mutex::new(FxHashSet::with_capacity_and_hasher(seen_cap, Default::default()))).collect(),
            emitted: AtomicU64::new(0), target, stop: AtomicBool::new(false),
            out: Mutex::new(Out { w, lines: 0, bytes: 0, log: None, side: None }),
            pos: Mutex::new(Pos::default()),
            stats: Arc::new(MergerStats::default()),
            flush_bytes: flush_bytes.max(4096),
        }
    }

    fn insert_seen(&self, hv: u64) { self.seen[shard_of(hv)].lock().unwrap().insert(hv); }

    /// A consistent checkpoint: the output flushed to the OS, and the position
    /// read under the same lock, so every line counted is written and every
    /// unit marked finished has all its lines counted.
    fn snapshot(&self) -> Result<(Pos, u64, u64)> {
        let mut o = self.out.lock().unwrap();
        o.w.flush()?;
        if let Some(l) = o.log.as_mut() { l.flush()?; }
        let p = self.pos.lock().unwrap().clone();
        Ok((p, o.lines, o.bytes))
    }
}

/// One worker's DFS state for the current pass.
#[derive(Default)]
struct Worker {
    path:     Vec<u32>,
    bytes:    Vec<u8>,
    lens:     Vec<usize>,
    nl:       usize,
    out:      Vec<u8>,
    hashes:   Vec<u64>,
    buf:      Vec<u8>,
    target:   u32,
    deferred: bool,
    nodes:    u64,
    probes:   u64,
    steps:    u64,
    /// This pass: seconds spent inside units, the longest unit and its index.
    busy:     f64,
    max_unit: (f64, usize),
}

impl Worker {
    fn push(&mut self, p: &Prep, id: u32) {
        self.path.push(id);
        self.lens.push(self.bytes.len());
        if let Some(b) = p.decode.get(id as usize) { self.bytes.extend_from_slice(b); }
        if p.has_nl.get(id as usize).copied().unwrap_or(false) { self.nl += 1; }
    }
    fn pop(&mut self, p: &Prep) {
        let Some(id) = self.path.pop() else { return };
        if let Some(old) = self.lens.pop() { self.bytes.truncate(old); }
        if p.has_nl.get(id as usize).copied().unwrap_or(false) { self.nl -= 1; }
    }
    fn ctx(&self, start_id: u32) -> (u32, u32) { crate::context_from_prefix(&self.path, start_id) }

    /// Write the buffered lines (and their hashes) to the sink.
    fn flush(&mut self, rt: &Rt) -> Result<()> {
        if self.out.is_empty() { return Ok(()); }
        let (n, nb) = (self.hashes.len() as u64, self.out.len() as u64);
        rt.stats.writer_blocked.store(true, AO::Relaxed);
        {
            let mut g = rt.out.lock().unwrap();
            let o = &mut *g;
            o.w.write_all(&self.out)?;
            if let Some(l) = o.log.as_mut() {
                for h in &self.hashes { l.write_all(&h.to_le_bytes())?; }
            }
            o.lines += n;
            o.bytes += nb;
            if let Some(sc) = o.side.as_mut() {
                if o.lines - sc.last_lines >= 1_000_000 && sc.last.elapsed() >= Duration::from_secs(1) {
                    o.w.flush()?;
                    let _ = crate::write_progress(&sc.path, o.lines, o.bytes, &sc.fp);
                    sc.last = Instant::now();
                    sc.last_lines = o.lines;
                }
            }
        }
        rt.stats.writer_blocked.store(false, AO::Relaxed);
        rt.stats.emitted.fetch_add(n, AO::Relaxed);
        rt.stats.bytes_written.fetch_add(nb, AO::Relaxed);
        self.out.clear();
        self.hashes.clear();
        Ok(())
    }

    /// The line in `buf`: dedupe, count, buffer. True when the run is over.
    fn accept(&mut self, rt: &Rt) -> Result<bool> {
        let hv = line_hash(&self.buf);
        if !rt.seen[shard_of(hv)].lock().unwrap().insert(hv) { return Ok(false); }
        let n = rt.emitted.fetch_add(1, AO::Relaxed);
        if n >= rt.target {
            rt.stop.store(true, AO::Relaxed);
            return Ok(true);
        }
        self.out.extend_from_slice(&self.buf);
        self.out.push(b'\n');
        self.hashes.push(hv);
        if self.out.len() >= rt.flush_bytes { self.flush(rt)?; }
        if n + 1 >= rt.target {
            rt.stop.store(true, AO::Relaxed);
            return Ok(true);
        }
        Ok(false)
    }

    /// The finished candidate on `path`: filters, --enterprise / --case-shape,
    /// then `accept`. True when the run is over.
    fn emit(&mut self, p: &Prep, rt: &Rt) -> Result<bool> {
        let o = &p.opts;
        if o.case_masks.is_empty() {
            let r = finalized_range(&self.bytes, p.kind);
            let (rs, re) = (r.start, r.end);
            let len = re - rs;
            if len == 0 || len < o.min_len || len > o.max_len { return Ok(false); }
            if self.nl > 0 && self.bytes[rs..re].iter().any(|&c| c == b'\n' || c == b'\r') { return Ok(false); }
            self.buf.clear();
            self.buf.extend_from_slice(&self.bytes[rs..re]);
            if o.enterprise {
                match enterprise::decide(&self.buf) {
                    enterprise::Decision::AsIs => {}
                    enterprise::Decision::Cap => { self.buf[0] = self.buf[0].to_ascii_uppercase(); }
                    enterprise::Decision::Drop => return Ok(false),
                }
            }
            return self.accept(rt);
        }
        // --case-shape: once per pattern, each token re-cased by its position
        // in the whole candidate (prefix, word and suffix tokens alike), as the
        // enumerator does.
        for mask in o.case_masks.iter() {
            self.buf.clear();
            for (pos, &id) in self.path.iter().enumerate() {
                if let Some(b) = p.decode.get(id as usize) {
                    let s = self.buf.len();
                    self.buf.extend_from_slice(b);
                    apply_op(&mut self.buf[s..], mask.op_for(pos));
                }
            }
            finalize_decoded(&mut self.buf, p.kind);
            let len = self.buf.len();
            if self.buf.is_empty() || len < o.min_len || len > o.max_len
                || self.buf.iter().any(|&c| c == b'\n' || c == b'\r') { continue; }
            if self.accept(rt)? { return Ok(true); }
        }
        Ok(false)
    }

    /// After a placed word: walk suffixes (`acc` = levels so far), or with
    /// --prepend-only take END only.
    fn suffix(&mut self, p: &Prep, rt: &Rt, acc: u32) -> Result<bool> {
        let (a, b) = self.ctx(p.start_id);
        let ch = p.kids(a, b);
        let t_lvl = self.target;
        if p.opts.word_at_end {
            self.steps += 1;
            if let Some(&(_, lp)) = ch.iter().find(|x| x.0 == p.end_id) {
                let nl = acc + lp_to_level(lp);
                if nl > t_lvl { self.deferred = true; }
                else if nl == t_lvl && self.path.len() >= p.opts.min_tokens && self.emit(p, rt)? { return Ok(true); }
            }
            return Ok(false);
        }
        for &(t, lp) in ch.iter() {
            self.steps += 1;
            let nl = acc + lp_to_level(lp);
            if nl > t_lvl { self.deferred = true; break; }
            if t == p.end_id {
                if nl == t_lvl && self.path.len() >= p.opts.min_tokens && self.emit(p, rt)? { return Ok(true); }
                continue;
            }
            if self.path.len() < p.opts.max_tokens {
                self.push(p, t);
                let stop = self.suffix(p, rt, nl)?;
                self.pop(p);
                if stop { return Ok(true); }
            }
        }
        Ok(false)
    }

    /// Place every word whose candidates can reach the current level after the
    /// current prefix (`lvl` levels), restricted to the word-first tokens whose
    /// rank here is `off` mod `stride` (splits the empty prefix across units).
    fn insert_part(&mut self, p: &Prep, rt: &Rt, lvl: u32, stride: usize, off: usize) -> Result<bool> {
        let (a, b) = self.ctx(p.start_id);
        let info = p.node(a, b);
        let t_lvl = self.target;
        if lvl.saturating_add(info.g) > t_lvl { self.deferred = true; return Ok(false); }
        let em = p.em;
        let tri = &info.tri_f;
        let bi: &[(u32, f32)] = if em.is_kn { &p.bi_f[b as usize] } else { &[] };
        let uf: &[(u32, f32)] = &p.uf;
        let bids: &[u32] = if p.uni.is_some() { &p.bi_ids[b as usize] } else { &[] };
        let (mut i, mut j, mut u) = (0usize, 0usize, 0usize);
        let plen = self.path.len();
        let mut rank = 0usize;
        loop {
            // Next word-first token in descending log P across the tiers.
            while j < bi.len() && info.ids.binary_search(&bi[j].0).is_ok() { j += 1; }
            // Unigram tier: skip tokens a higher tier prices here; once even
            // the stream's head is past this level, the rest is too.
            while u < uf.len() {
                if lvl + p.lb(-(info.lam + uf[u].1) + p.kmin_all) > t_lvl { self.deferred = true; u = uf.len(); break; }
                let x = uf[u].0;
                if info.ids.binary_search(&x).is_ok() || bids.binary_search(&x).is_ok() { u += 1; } else { break; }
            }
            let (t, lp0) = match pick3(tri.get(i).copied(), bi.get(j).copied(), uf.get(u).copied(), info.lam) {
                0 => { i += 1; tri[i - 1] }
                1 => { j += 1; (bi[j - 1].0, info.lam + bi[j - 1].1) }
                2 => { u += 1; (uf[u - 1].0, info.lam + uf[u - 1].1) }
                _ => break,
            };
            if lvl + p.lb(-lp0 + p.kmin_all) > t_lvl { self.deferred = true; break; }
            rank += 1;
            if (rank - 1) % stride != off { continue; }
            if lvl + p.lb(-lp0 + p.kmin_t[t as usize]) > t_lvl { self.deferred = true; continue; }
            for &ei in &p.by_first[t as usize] {
                let e = &p.entries[ei as usize];
                if lvl + p.lb(-lp0 + e.k) > t_lvl { self.deferred = true; break; }
                if plen + e.toks.len() > p.opts.max_tokens { continue; }
                self.probes += 1;
                // Exact block cost.
                let Some(lpe) = block_lp(em, b, t, lp0, e) else { continue };
                let acc = lvl + lp_to_level(lpe);
                if acc > t_lvl { self.deferred = true; continue; }
                for &x in &e.toks { self.push(p, x); }
                let stop = self.suffix(p, rt, acc)?;
                for _ in 0..e.toks.len() { self.pop(p); }
                if stop { return Ok(true); }
            }
        }
        Ok(false)
    }

    /// A prefix node: place words here, then extend the prefix.
    fn prefix(&mut self, p: &Prep, rt: &Rt, lvl: u32) -> Result<bool> {
        self.nodes += 1;
        if rt.stop.load(AO::Relaxed) { return Ok(true); }
        if self.insert_part(p, rt, lvl, 1, 0)? { return Ok(true); }
        if p.ov.is_some() && self.overlap(p, rt, lvl)? { return Ok(true); }
        self.descend(p, rt, lvl)
    }

    /// Seam overlap: place remainders whose head the last prefix token's
    /// bytes end with. Same merge and bounds as `insert_part`, over the
    /// remainders' first tokens.
    fn overlap(&mut self, p: &Prep, rt: &Rt, lvl: u32) -> Result<bool> {
        let Some(ov) = &p.ov else { return Ok(false) };
        let (a, b) = self.ctx(p.start_id);
        let bu = b as usize;
        if bu >= ov.heads_of.len() || ov.heads_of[bu].is_empty() { return Ok(false); }
        let t_lvl = self.target;
        if lvl.saturating_add(ov.g_b[bu]) > t_lvl { self.deferred = true; return Ok(false); }
        let em = p.em;
        let info = ov.onode(em, p.uni.as_ref(), p.opts.no_prune, a, b);
        if lvl.saturating_add(info.g) > t_lvl { self.deferred = true; return Ok(false); }
        let bi_arc = if em.is_kn { Some(ov.bov(em, b)) } else { None };
        let bi: &[(u32, f32)] = bi_arc.as_deref().unwrap_or(&[]);
        let uf_arc = if p.uni.is_some() { Some(ov.uov(em, p.uni.as_ref(), b)) } else { None };
        let uf: &[(u32, f32)] = uf_arc.as_deref().unwrap_or(&[]);
        let tri = &info.tri_f;
        let km = ov.kmin_b[bu];
        let (mut i, mut j, mut u) = (0usize, 0usize, 0usize);
        let plen = self.path.len();
        loop {
            while j < bi.len() && info.ids.binary_search(&bi[j].0).is_ok() { j += 1; }
            while u < uf.len() && info.ids.binary_search(&uf[u].0).is_ok() { u += 1; }
            let (t, lp0) = match pick3(tri.get(i).copied(), bi.get(j).copied(), uf.get(u).copied(), info.lam) {
                0 => { i += 1; tri[i - 1] }
                1 => { j += 1; (bi[j - 1].0, info.lam + bi[j - 1].1) }
                2 => { u += 1; (uf[u - 1].0, info.lam + uf[u - 1].1) }
                _ => break,
            };
            if lvl + p.lb(-lp0 + km) > t_lvl { self.deferred = true; break; }
            for &h in ov.heads_of[bu].iter() {
                let Some(list) = ov.groups.get(&(h, t)) else { continue };
                for &ri in list.iter() {
                    let e = &ov.rentries[ri as usize];
                    if lvl + p.lb(-lp0 + e.k) > t_lvl { self.deferred = true; break; }
                    if plen + e.toks.len() > p.opts.max_tokens { continue; }
                    self.probes += 1;
                    let Some(lpe) = block_lp(em, b, t, lp0, e) else { continue };
                    let acc = lvl + lp_to_level(lpe);
                    if acc > t_lvl { self.deferred = true; continue; }
                    for &x in &e.toks { self.push(p, x); }
                    let stop = self.suffix(p, rt, acc)?;
                    for _ in 0..e.toks.len() { self.pop(p); }
                    if stop { return Ok(true); }
                }
            }
        }
        Ok(false)
    }

    fn descend(&mut self, p: &Prep, rt: &Rt, lvl: u32) -> Result<bool> {
        if self.path.len() + 2 > p.opts.max_tokens { return Ok(false); }
        let (a, b) = self.ctx(p.start_id);
        let ch = p.kids(a, b);
        for &(t, lp) in ch.iter() {
            if t == p.end_id { continue; }
            let nl = lvl + lp_to_level(lp);
            if nl + p.gfloor > self.target { self.deferred = true; break; }
            self.push(p, t);
            let stop = self.prefix(p, rt, nl)?;
            self.pop(p);
            if stop { return Ok(true); }
        }
        Ok(false)
    }

    /// Unit `u` of the current pass. True when the run is over.
    fn unit(&mut self, p: &Prep, rt: &Rt, root: &[(u32, f32)], u: usize) -> Result<bool> {
        self.path.clear(); self.bytes.clear(); self.lens.clear(); self.nl = 0;
        if u < ROOT_PARTS {
            self.nodes += 1;
            return self.insert_part(p, rt, 0, ROOT_PARTS, u);
        }
        let (t, lp) = root[u - ROOT_PARTS];
        if t == p.end_id { return Ok(false); }
        let lvl = lp_to_level(lp);
        if lvl + p.gfloor > self.target { self.deferred = true; return Ok(false); }
        self.push(p, t);
        self.prefix(p, rt, lvl)
    }
}

/// Run the level sweep from `start` until --count, exhaustion or LEVEL_MAX.
/// Returns the last level swept.
fn sweep(p: &Prep, rt: &Rt, n_threads: usize, start: Pos) -> Result<u32> {
    let root = p.kids(p.start_id, p.start_id);
    let n_units = ROOT_PARTS + root.len();
    if !start.done.is_empty() && start.done.len() != n_units {
        bail!("--resume: the checkpoint has {} work units per level, this run has {} \
               (different model or settings?)", start.done.len(), n_units);
    }
    let mut workers: Vec<Worker> = (0..n_threads.max(1)).map(|_| Worker::default()).collect();
    let t0 = Instant::now();
    let mut last_level = start.level;
    let mut first = true;
    for target_level in start.level..=LEVEL_MAX {
        last_level = target_level;
        let skip: Vec<bool> = if first && !start.done.is_empty() { start.done.clone() } else { vec![false; n_units] };
        {
            let mut g = rt.pos.lock().unwrap();
            *g = Pos { level: target_level, units: n_units, done: skip.clone(),
                       deferred: first && start.deferred };
        }
        first = false;
        let next_unit = AtomicUsize::new(0);
        let tp = Instant::now();
        let before = rt.emitted.load(AO::Relaxed).min(rt.target);
        let results: Vec<Result<()>> = std::thread::scope(|s| {
            let hs: Vec<_> = workers.iter_mut().map(|w| {
                let (next_unit, root, skip) = (&next_unit, &root, &skip);
                s.spawn(move || -> Result<()> {
                    w.target = target_level;
                    w.busy = 0.0;
                    w.max_unit = (0.0, 0);
                    loop {
                        if rt.stop.load(AO::Relaxed) { break; }
                        let u = next_unit.fetch_add(1, AO::Relaxed);
                        if u >= n_units { break; }
                        if skip[u] { continue; }
                        let tu = Instant::now();
                        w.deferred = false;
                        let over = w.unit(p, rt, root, u)?;
                        // Output first, then the mark: a checkpoint never
                        // records a unit whose lines are not all written.
                        w.flush(rt)?;
                        if !over {
                            let mut g = rt.pos.lock().unwrap();
                            g.done[u] = true;
                            g.deferred |= w.deferred;
                        }
                        let d = tu.elapsed().as_secs_f64();
                        w.busy += d;
                        if d > w.max_unit.0 { w.max_unit = (d, u); }
                        if over { break; }
                    }
                    w.flush(rt)
                })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow!("worker panicked")))).collect()
        });
        for r in results { r?; }
        let after = rt.emitted.load(AO::Relaxed).min(rt.target);
        let any_deferred = rt.pos.lock().unwrap().deferred;
        let (nodes, probes, steps) = workers.iter().fold((0, 0, 0), |a, w| (a.0 + w.nodes, a.1 + w.probes, a.2 + w.steps));
        // Load balance: share of worker time spent outside units (waiting at
        // the end-of-level barrier), and the longest single unit.
        let wall = tp.elapsed().as_secs_f64();
        let busy: f64 = workers.iter().map(|w| w.busy).sum();
        let (mu, mi) = workers.iter().map(|w| w.max_unit)
            .fold((0.0, 0), |a, b| if b.0 > a.0 { b } else { a });
        let unit = if mi < ROOT_PARTS { format!("empty-prefix slice {mi}") }
                   else { format!("prefix {:?}", String::from_utf8_lossy(&p.decode[root[mi - ROOT_PARTS].0 as usize])) };
        log_msg(&format!(
            "[wordlist] level {} +{} (total {}) in {:.1}s; cumulative prefix_nodes={} word_probes={} suffix_steps={}; \
             barrier_wait={:.0}% longest_unit={:.2}s ({})",
            target_level, after - before, after, wall, nodes, probes, steps,
            100.0 * (1.0 - busy / (wall * workers.len() as f64).max(1e-9)), mu, unit));
        if rt.stop.load(AO::Relaxed) || !any_deferred { break; }
    }
    log_msg(&format!("[wordlist] DONE emitted={} level={} kids_memo={} node_memo={} in {:.1}s",
        rt.emitted.load(AO::Relaxed).min(rt.target), last_level, p.kids.size.load(AO::Relaxed),
        p.nodes.size.load(AO::Relaxed), t0.elapsed().as_secs_f64()));
    Ok(last_level)
}

// ----------------------------------------------------------------------------
// Checkpoint encoding
// ----------------------------------------------------------------------------

/// `level`, `lines` (= emitted) and `bytes` written, finished units as ranges.
fn encode_pos(p: &Pos, lines: u64, bytes: u64) -> String {
    let mut ranges: Vec<String> = Vec::new();
    let mut i = 0;
    while i < p.done.len() {
        if !p.done[i] { i += 1; continue; }
        let s = i;
        while i < p.done.len() && p.done[i] { i += 1; }
        ranges.push(if i - 1 == s { s.to_string() } else { format!("{}-{}", s, i - 1) });
    }
    format!("units={};deferred={};lines={};bytes={};done={}",
        p.units, u8::from(p.deferred), lines, bytes, ranges.join(","))
}

fn decode_pos(level: u32, s: &str) -> Result<(Pos, u64, u64)> {
    let mut units: Option<usize> = None;
    let (mut deferred, mut lines, mut bytes) = (false, 0u64, 0u64);
    let mut done_s = "";
    for kv in s.split(';') {
        let (k, v) = kv.split_once('=').ok_or_else(|| anyhow!("checkpoint: bad wordlist field '{kv}'"))?;
        match k {
            "units"    => units = Some(v.parse()?),
            "deferred" => deferred = v == "1",
            "lines"    => lines = v.parse()?,
            "bytes"    => bytes = v.parse()?,
            "done"     => done_s = v,
            _ => {}
        }
    }
    let units = units.ok_or_else(|| anyhow!("checkpoint: wordlist field has no unit count"))?;
    let mut done = vec![false; units];
    for r in done_s.split(',').filter(|x| !x.is_empty()) {
        let (a, b) = match r.split_once('-') {
            Some((a, b)) => (a.parse::<usize>()?, b.parse::<usize>()?),
            None => { let a = r.parse::<usize>()?; (a, a) }
        };
        if b >= units || a > b { bail!("checkpoint: unit range {r} out of 0..{units}"); }
        for x in &mut done[a..=b] { *x = true; }
    }
    Ok((Pos { level, units, done, deferred }, lines, bytes))
}

fn seen_log_path(ckpt: &Path) -> PathBuf {
    let mut s = ckpt.as_os_str().to_owned();
    s.push(".seen");
    PathBuf::from(s)
}

/// Rebuild the dedupe set from the first `bytes` bytes of `out` (which must
/// hold exactly `lines` lines there).
fn restore_from_output(rt: &Rt, out: &Path, lines: u64, bytes: u64) -> Result<()> {
    let f = File::open(out).with_context(|| format!("open {}", out.display()))?;
    let mut r = BufReader::with_capacity(8 << 20, f.take(bytes));
    let mut buf: Vec<u8> = Vec::new();
    let mut n = 0u64;
    loop {
        buf.clear();
        let k = std::io::BufRead::read_until(&mut r, b'\n', &mut buf)?;
        if k == 0 { break; }
        if buf.last() != Some(&b'\n') { bail!("--resume: {} does not end on a full line at {} bytes", out.display(), bytes); }
        buf.pop();
        rt.insert_seen(line_hash(&buf));
        n += 1;
    }
    if n != lines {
        bail!("--resume: {} holds {} lines in its first {} bytes, the checkpoint says {} \
               (the file was changed since)", out.display(), n, bytes, lines);
    }
    Ok(())
}

/// Rebuild the dedupe set from the hash log, cut to its first `lines` entries.
fn restore_from_log(rt: &Rt, log: &Path, lines: u64) -> Result<()> {
    let need = lines * 8;
    let len = std::fs::metadata(log).with_context(|| format!("stat {}", log.display()))?.len();
    if len < need {
        bail!("--resume: {} is shorter ({}B) than the checkpoint needs ({}B)", log.display(), len, need);
    }
    if len > need {
        std::fs::OpenOptions::new().write(true).open(log).and_then(|f| f.set_len(need))
            .with_context(|| format!("truncate {}", log.display()))?;
    }
    let mut r = BufReader::with_capacity(8 << 20, File::open(log)?);
    let mut b = [0u8; 8];
    for _ in 0..lines {
        r.read_exact(&mut b)?;
        rt.insert_seen(u64::from_le_bytes(b));
    }
    Ok(())
}

/// `Sink` as a `Write` for the output lock.
struct SinkW(Sink);

impl Write for SinkW {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write_raw(buf).map_err(std::io::Error::other)?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { self.0.flush_buffered().map_err(std::io::Error::other) }
}

/// Checkpoint and resume settings, resolved by `run_generate` as for the
/// enumerator.
pub struct ResumeCfg {
    pub checkpoint_to:  Option<PathBuf>,
    pub resume_from:    Option<PathBuf>,
    pub sidecar_resume: bool,
    pub ckpt_is_default: bool,
    pub ckpt_fp:        String,
    pub args_fp:        String,
    pub progress_path:  Option<PathBuf>,
    pub run_id:         String,
    pub n_threads:      usize,
}

/// The whole `--wordlist` run: resume, tables, sweep, checkpoints, telemetry.
pub fn run(args: &GenerateArgs, inp: Inputs, entry_seqs: &[Vec<u32>], opts: Opts,
           rc: ResumeCfg, model_path: &Path) -> Result<()> {
    let strict = args.strict;
    let n_threads = rc.n_threads.max(1);
    let target = args.count.unwrap_or(u64::MAX);

    // Where to start: a checkpoint, the progress sidecar, or level --min-level.
    // (pos, lines, bytes) of what the resumed output already holds.
    let mut start = Pos { level: args.min_level, ..Pos::default() };
    let mut have: Option<(u64, u64)> = None;
    if let Some(rp) = &rc.resume_from {
        if !rp.exists() {
            bail!("--resume: no checkpoint at {} — nothing to resume. Run without --resume to start fresh.",
                rp.display());
        }
        let cf = crate::read_checkpoint(rp)?;
        if cf.fingerprint != rc.ckpt_fp {
            bail!("--resume: checkpoint fingerprint mismatch, refusing to resume.\n  \
                   saved:   {}\n  current: {}\n  \
                   (model, args, --threads, and tokenov version must all match)", cf.fingerprint, rc.ckpt_fp);
        }
        match cf.slots.into_iter().next() {
            Some(CkptSlot::Done) => {
                log_msg("[gen] previous run already complete — nothing to resume");
                return Ok(());
            }
            Some(CkptSlot::InProgress(c)) => {
                let (pos, lines, bytes) = decode_pos(c.target_level, &c.wordlist_pos)?;
                if pos.level < args.min_level {
                    bail!("--resume: checkpoint is at level {} but --min-level is {} — \
                           resuming it would re-emit levels below the floor", pos.level, args.min_level);
                }
                log_msg(&format!("[gen] resume: level {} with {} of {} units finished, {} candidates written",
                    pos.level, pos.done.iter().filter(|&&d| d).count(), pos.units, lines));
                start = pos;
                have = Some((lines, bytes));
            }
            _ => {}
        }
    } else if rc.sidecar_resume {
        let pp = rc.progress_path.as_ref().ok_or_else(|| anyhow!(
            "--resume requires --output FILE (no sidecar exists for stdout)"))?;
        if !pp.exists() {
            bail!("--resume: no progress sidecar at {} — run without --resume to start fresh", pp.display());
        }
        let saved = crate::read_progress(pp)?;
        if saved.fingerprint != rc.args_fp {
            bail!("--resume: args fingerprint mismatch, refusing to append.\n\
                   saved:   {}\n  current: {}\n\
                   Re-run without --resume to start a fresh output.", saved.fingerprint, rc.args_fp);
        }
        log_msg(&format!("[gen] resume: re-running and skipping the {} candidates already written", saved.emitted));
        have = Some((saved.emitted, saved.byte_offset));
    } else {
        // Fresh run: clear stale state a later --resume could pick up.
        if let Some(pp) = &rc.progress_path { let _ = std::fs::remove_file(pp); }
        if let Some(cp) = &rc.checkpoint_to {
            let _ = std::fs::remove_file(cp);
            let _ = std::fs::remove_file(seen_log_path(cp));
        }
    }

    let prep = prepare(inp, entry_seqs, opts)?;

    // Output: append to a resumed --output (cut back to the recorded offset),
    // otherwise create it.
    let resuming = have.is_some();
    if resuming && args.output.is_none() && rc.resume_from.is_none() {
        bail!("--resume requires --output FILE (no sidecar exists for stdout)");
    }
    let sink = match (&args.output, have) {
        (Some(out), Some((_, bytes))) => {
            crate::truncate_output_to(out, bytes)?;
            Sink::open_append(out)?
        }
        _ => Sink::open(args.output.as_deref())?,
    };
    let rt = Rt::new(Box::new(SinkW(sink)), target, args.flush_bytes);
    if let Some((lines, bytes)) = have {
        let t = Instant::now();
        match (&args.output, &rc.resume_from) {
            (Some(out), _) => restore_from_output(&rt, out, lines, bytes)?,
            (None, Some(rf)) => {
                let from = seen_log_path(rf);
                restore_from_log(&rt, &from, lines).with_context(||
                    "if the run being resumed wrote to --output FILE, resume it with the same --output")?;
                // Writing to another checkpoint: carry the hashes over.
                if let Some(cp) = &rc.checkpoint_to {
                    let to = seen_log_path(cp);
                    if to != from {
                        std::fs::copy(&from, &to).with_context(|| format!("copy {} to {}", from.display(), to.display()))?;
                    }
                }
            }
            (None, None) => bail!("--resume requires --output FILE (no sidecar exists for stdout)"),
        }
        rt.emitted.store(lines, AO::Relaxed);
        let mut o = rt.out.lock().unwrap();
        o.lines = lines;
        o.bytes = bytes;
        log_msg(&format!("[gen] resume: {} candidates restored in {:.1}s", lines, t.elapsed().as_secs_f64()));
    }
    {
        let mut o = rt.out.lock().unwrap();
        // Stdout cannot be re-read on resume: keep the hashes of what was
        // written next to the checkpoint instead.
        if let (None, Some(cp)) = (&args.output, &rc.checkpoint_to) {
            let lp = seen_log_path(cp);
            let f = if resuming { std::fs::OpenOptions::new().create(true).append(true).open(&lp) }
                    else { File::create(&lp) };
            o.log = Some(BufWriter::with_capacity(1 << 20, f.with_context(|| format!("open {}", lp.display()))?));
        }
        if rc.checkpoint_to.is_none() {
            if let Some(pp) = &rc.progress_path {
                o.side = Some(Sidecar { path: pp.clone(), fp: rc.args_fp.clone(), last: Instant::now(),
                                        last_lines: o.lines });
            }
        }
    }

    // Session record (fast mode with a checkpoint, as for the enumerator).
    let session_id: Option<String> = match (&rc.checkpoint_to, strict) {
        (Some(cp), false) => {
            let found = if rc.resume_from.is_some() {
                crate::load_sessions().into_iter().find(|r| Path::new(&r.checkpoint_file) == cp.as_path())
            } else { None };
            match found {
                Some(mut rec) => {
                    rec.pid = std::process::id();
                    rec.status = "running".into();
                    rec.updated_at = crate::now_secs();
                    let _ = crate::write_session(&rec);
                    Some(rec.id)
                }
                None => crate::session_start(model_path, args.count, n_threads, cp, &rc.run_id),
            }
        }
        _ => None,
    };

    log_msg(&format!("[gen] wordlist: threads={} count={:?} min_level={} max_tokens={} min_len={} max_len={}",
        n_threads, args.count, start.level, prep.opts.max_tokens, prep.opts.min_len, prep.opts.max_len));
    let stats_stop = Arc::new(AtomicBool::new(false));
    let ckpt_stop = AtomicBool::new(false);
    let result: Result<u32> = std::thread::scope(|scope| {
        // Telemetry, as fast mode prints it (strict prints none, as before).
        let telemetry = if !strict && args.stats_interval_ms > 0 {
            let s = Arc::clone(&rt.stats);
            let stop = Arc::clone(&stats_stop);
            let csv = match (&args.stats_csv, &args.output) {
                (Some(p), _) => Some(p.clone()),
                (None, Some(o)) => { let mut s = o.as_os_str().to_owned(); s.push(".stats.csv"); Some(PathBuf::from(s)) }
                (None, None) => None,
            };
            let interval = Duration::from_millis(args.stats_interval_ms);
            let (v, pe, j) = (crate::verbose(), Duration::from_secs(args.progress_secs), args.json);
            Some(scope.spawn(move || crate::run_telemetry(s, stop, interval, csv, v, pe, j)))
        } else { None };
        let writer = rc.checkpoint_to.as_ref().map(|cp| {
            let (rt, stop, fp, sid) = (&rt, &ckpt_stop, &rc.ckpt_fp, &session_id);
            let secs = args.checkpoint_secs.max(1);
            scope.spawn(move || {
                loop {
                    let mut waited = 0u64;
                    while waited < secs * 1000 {
                        if stop.load(AO::Relaxed) { return; }
                        std::thread::sleep(Duration::from_millis(200));
                        waited += 200;
                    }
                    if stop.load(AO::Relaxed) { return; }
                    if let Ok((pos, lines, bytes)) = rt.snapshot() {
                        let _ = write_pos(cp, fp, &pos, lines, bytes);
                        if let Some(id) = sid { crate::session_update(id, lines, "running"); }
                    }
                }
            })
        });
        let r = sweep(&prep, &rt, n_threads, start);
        ckpt_stop.store(true, AO::Relaxed);
        if let Some(h) = writer { let _ = h.join(); }
        stats_stop.store(true, AO::Relaxed);
        if let Some(h) = telemetry { let _ = h.join(); }
        r
    });
    let (pos, lines, bytes) = rt.snapshot()?;
    match &result {
        Ok(_) => {
            if let Some(pp) = &rc.progress_path { let _ = std::fs::remove_file(pp); }
            if let Some(cp) = &rc.checkpoint_to {
                let _ = std::fs::remove_file(seen_log_path(cp));
                if rc.ckpt_is_default {
                    let _ = std::fs::remove_file(cp);
                } else {
                    let _ = crate::write_checkpoint(cp, &rc.ckpt_fp, &[CkptSlot::Done]);
                }
            }
            if let Some(id) = &session_id { crate::session_update(id, lines, "done"); }
        }
        Err(_) => {
            // Leave a resumable position behind.
            if let Some(cp) = &rc.checkpoint_to { let _ = write_pos(cp, &rc.ckpt_fp, &pos, lines, bytes); }
        }
    }
    result?;
    log_msg(&format!("[gen] wordlist: total emitted = {}", lines));
    if args.count.is_some() && lines < target {
        if let Some(w) = crate::wordlist_shortfall_warning(args.enterprise, false, args.count, lines,
                                                          prep.opts.word_at_end) {
            warn_msg(&w);
        } else {
            warn_msg(&format!("warning: --wordlist produced {lines} of the {target} candidates requested: \
                               every candidate within --max-tokens / --max-len is used up"));
        }
    } else if let Some(w) = crate::wordlist_shortfall_warning(args.enterprise, false, args.count, lines,
                                                              prep.opts.word_at_end) {
        warn_msg(&w);
    }
    Ok(())
}

fn write_pos(cp: &Path, fp: &str, pos: &Pos, lines: u64, bytes: u64) -> Result<()> {
    crate::write_checkpoint(cp, fp, &[CkptSlot::InProgress(ThreadCkpt {
        target_level: pos.level, init_idx: 0, prefix: Vec::new(), idx_stack: Vec::new(),
        emitted: lines, streams: Vec::new(), wordlist_pos: encode_pos(pos, lines, bytes),
    })])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(toks: &[&str]) -> (FxHashMap<Vec<u8>, u32>, usize) {
        let mut v = FxHashMap::default();
        let mut m = 0;
        for (i, t) in toks.iter().enumerate() { v.insert(t.as_bytes().to_vec(), i as u32); m = m.max(t.len()); }
        (v, m)
    }

    /// Every split of `w` into vocab tokens (test oracle for `kbest`).
    fn all_splits(w: &[u8], vocab: &FxHashMap<Vec<u8>, u32>, maxlen: usize, max_tok: usize) -> Vec<Vec<u32>> {
        fn go(lat: &[Vec<(usize, u32)>], i: usize, cur: &mut Vec<u32>, out: &mut Vec<Vec<u32>>, max_tok: usize) {
            if i == lat.len() { out.push(cur.clone()); return; }
            if cur.len() >= max_tok { return; }
            for &(j, t) in &lat[i] {
                cur.push(t);
                go(lat, j, cur, out, max_tok);
                cur.pop();
            }
        }
        let lat = lattice(w, vocab, maxlen);
        let mut out = Vec::new();
        go(&lat, 0, &mut Vec::new(), &mut out, max_tok);
        out
    }

    #[test]
    fn lattice_lists_every_token_at_each_position() {
        let (v, m) = vocab(&["l", "o", "v", "e", "lo", "ove", "love", "il"]);
        let lat = lattice(b"love", &v, m);
        assert_eq!(lat.len(), 4);
        let mut l0 = lat[0].clone(); l0.sort();
        assert_eq!(l0, vec![(1, 0), (2, 4), (4, 6)]);
        let mut l1 = lat[1].clone(); l1.sort();
        assert_eq!(l1, vec![(2, 1), (4, 5)]);
        assert_eq!(all_splits(b"love", &v, m, 12).len(), 4); // l.o.v.e, l.ove, lo.v.e, love
        assert_eq!(all_splits(b"love", &v, m, 2), vec![vec![0, 5], vec![6]]);
    }

    /// Brute force over all splits must agree with the k-best search,
    /// including the trigram-state dependence of the step cost.
    #[test]
    fn kbest_matches_brute_force() {
        let toks = ["a", "b", "c", "ab", "bc", "abc", "ca", "cab"];
        let (v, m) = vocab(&toks);
        let step = |a: u32, b: u32, t: u32| -> Option<f32> {
            if a == 1 && t == 2 { return None; }
            Some(((a.wrapping_mul(7) ^ b.wrapping_mul(13) ^ t.wrapping_mul(31)) % 17) as f32 * 0.5 + 0.25)
        };
        let fin = |a: u32, b: u32| -> Option<f32> { if b == 0 { None } else { Some(((a ^ b) % 5) as f32) } };
        for w in ["abcab", "cabcab", "abcabcab", "bcc"] {
            let all = all_splits(w.as_bytes(), &v, m, 6);
            let mut scored: Vec<(f32, Vec<u32>)> = all.into_iter().filter_map(|p| {
                let (mut a, mut b, mut c) = (NONE - 1, NONE - 1, 0f32);
                for &t in &p { c += step(a, b, t)?; a = b; b = t; }
                Some((c + fin(a, b)?, p))
            }).collect();
            scored.sort_by(path_cmp);
            let lat = lattice(w.as_bytes(), &v, m);
            for k in [1, 3, 10] {
                let got = kbest(&lat, k, 6, NONE - 1, NONE - 1, &step, &fin);
                let want: Vec<_> = scored.iter().take(k).cloned().collect();
                assert_eq!(got, want, "word {w} k {k}");
            }
        }
    }

    /// Tokens 0..6; trigram ctx (0,1) has child 2; bigram after 1 has 3; after
    /// 2 has 4. Token 5 is seen nowhere and has zero training count.
    fn tiny_em() -> EnumModel {
        let mut trigram = FxHashMap::default();
        trigram.insert(pack(0, 1), vec![(2u32, -0.5f32)]);
        trigram.insert(pack(1, 2), vec![(4u32, -0.25f32)]);
        let mut bigram = FxHashMap::default();
        bigram.insert(1u32, vec![(2u32, -1.0f32), (3u32, -1.5f32)]);
        bigram.insert(2u32, vec![(4u32, -0.75f32)]);
        let mut log_lambda = FxHashMap::default();
        log_lambda.insert(pack(0, 1), -2.0f32);
        log_lambda.insert(pack(1, 2), -1.0f32);
        EnumModel { is_kn: true, trigram, bigram, log_lambda, unigram_top: Vec::new(),
                    unigram_logw: 0.0, start_id: 6 }
    }
    const COUNTS: [u64; 6] = [10, 20, 30, 15, 25, 0];

    #[test]
    fn word_unigram_tier_only_when_trigram_and_bigram_miss() {
        let em = tiny_em();
        let tlp = Tlp::new(&em);
        let u = Uni::from_counts(&COUNTS, &em, 0.1);
        assert_eq!(tlp.tlp_fb(Some(&u), 0, 1, 2), Some(-0.5));
        assert_eq!(tlp.tlp_fb(Some(&u), 0, 1, 3), Some(-2.0 + -1.5));
        assert_eq!(tlp.tlp_fb(Some(&u), 4, 1, 3), Some(-1.5));
        assert_eq!(tlp.tlp_fb(None, 0, 1, 4), None);
        let n = 100.0 + 6.0; // add-1 over the full vocabulary
        let lu4 = 0.1f32.ln() + ((26.0f64 / n).ln() as f32);
        assert!((tlp.tlp_fb(Some(&u), 0, 1, 4).unwrap() - (-2.0 + lu4)).abs() < 1e-5);
        assert!((tlp.tlp_fb(Some(&u), 3, 3, 4).unwrap() - lu4).abs() < 1e-5);
        let lu5 = u.lu(5);
        assert!(lu5.is_finite());
        assert!((lu5 - (0.1f32.ln() + ((1.0f64 / n).ln() as f32))).abs() < 1e-5);
        assert!((0..5).all(|t| u.lu(t) > lu5));
        assert_eq!(u.lam_max, 0.0);
    }

    #[test]
    fn pick3_is_the_two_tier_merge_without_a_unigram_stream() {
        for &(x, y, lam) in &[(-1.0f32, -0.5f32, -0.5f32), (-1.0, -0.4, -0.5), (-2.0, -0.1, -1.0)] {
            let two = if x >= lam + y { 0 } else { 1 };
            assert_eq!(pick3(Some((1, x)), Some((2, y)), None, lam), two);
        }
        assert_eq!(pick3(None, None, None, 0.0), 3);
        assert_eq!(pick3(None, None, Some((3, -9.0)), -1.0), 2);
        assert_eq!(pick3(Some((1, -3.0)), None, Some((3, -2.0)), -1.0), 0);
        assert_eq!(pick3(None, Some((2, -2.0)), Some((3, -2.0)), -1.0), 1);
        assert_eq!(pick3(Some((1, -5.0)), Some((2, -3.0)), Some((3, -2.0)), -1.0), 2);
    }

    /// The exact block cost (`block_lp` with `uni1`) equals the sequential
    /// fallback sum in every context, and the setup's second-step bound is
    /// never below it.
    #[test]
    fn fallback_block_cost_is_exact_and_bounded() {
        let em = tiny_em();
        let tlp = Tlp::new(&em);
        let u = Uni::from_counts(&COUNTS, &em, 0.1);
        let toks = vec![2u32, 5, 4];
        let bi1 = tlp.bi(2, 5);
        assert!(bi1.is_none());
        let uni1 = Some(u.lu(5));
        let fixed = vec![tlp.tlp_fb(Some(&u), 2, 5, 4).unwrap()];
        let e = Entry { toks: toks.clone(), bi1, fixed: fixed.clone(), k: 0.0, uni1, head: NONE };
        let b1 = u.lam_max + u.lu(5);
        for a in 0..6u32 { for b in 0..6u32 {
            let Some(lp0) = tlp.tlp_fb(Some(&u), a, b, 2) else { continue };
            let got = block_lp(&em, b, 2, lp0, &e).unwrap();
            let s1 = tlp.tlp_fb(Some(&u), b, 2, 5).unwrap();
            let want = lp0 + s1 + fixed[0];
            assert!((got - want).abs() < 1e-5, "ctx ({a},{b})");
            assert!(s1 <= b1 + 1e-6, "ctx ({a},{b})");
        } }
        let off = Entry { toks, bi1: None, fixed, k: 0.0, uni1: None, head: NONE };
        assert!(block_lp(&em, 1, 2, -0.5, &off).is_none());
    }

    #[test]
    fn overlap_heads_are_proper_suffixes_only() {
        let mut heads = FxHashMap::default();
        heads.insert(b"l".to_vec(), 0u32);
        heads.insert(b"il".to_vec(), 1u32);
        heads.insert(b"ov".to_vec(), 2u32);
        assert_eq!(overlap_heads(b"il", &heads), vec![0]);
        let mut h = overlap_heads(b"mil", &heads); h.sort();
        assert_eq!(h, vec![0, 1]);
        assert!(overlap_heads(b"l", &heads).is_empty());
        assert!(overlap_heads(b"ove", &heads).is_empty());
    }

    #[test]
    fn position_round_trips_through_the_checkpoint_field() {
        let mut done = vec![false; 70];
        for i in [0, 1, 2, 5, 9, 10, 69] { done[i] = true; }
        let p = Pos { level: 17, units: 70, done, deferred: true };
        let s = encode_pos(&p, 1234, 56789);
        assert!(s.contains("done=0-2,5,9-10,69"), "{s}");
        let (q, lines, bytes) = decode_pos(17, &s).unwrap();
        assert_eq!((q, lines, bytes), (p, 1234, 56789));
        let none = Pos { level: 3, units: 4, done: vec![false; 4], deferred: false };
        assert_eq!(decode_pos(3, &encode_pos(&none, 0, 0)).unwrap().0, none);
        assert!(decode_pos(3, "units=4;done=2-9").is_err());
        assert!(decode_pos(3, "lines=4").is_err());
    }

    // ---- Whole-sweep tests on a synthetic model ----------------------------

    /// A small KN model over the vocab `TOKS` with every trigram/bigram tier
    /// generated from a fixed pseudo-random rule, END = 9, START = 10.
    const TOKS: [&str; 9] = ["a", "b", "1", "ab", "ba", "12", "!", "i", "il"];
    const END: u32 = 9;
    const START: u32 = 10;

    fn synth_em(tail: bool) -> EnumModel {
        let n = 11u32;
        let mut trigram: FxHashMap<Ctx, Vec<(u32, f32)>> = FxHashMap::default();
        let mut log_lambda = FxHashMap::default();
        let mut bigram: FxHashMap<u32, Vec<(u32, f32)>> = FxHashMap::default();
        let h = |x: u32| (x.wrapping_mul(2654435761) >> 7) % 97;
        for a in 0..n { for b in 0..n {
            if a == END || b == END || (b == START && a != START) { continue; }
            if h(a * 31 + b) % 3 == 0 { continue; }
            let mut l: Vec<(u32, f32)> = (0..=END)
                .filter(|&t| if t == END { h(a + b) % 2 == 0 } else { h(a * 131 + b * 17 + t) % 4 != 0 })
                .map(|t| (t, -0.3 - (h(a * 7 + b * 11 + t * 13) as f32) / 25.0)).collect();
            if l.is_empty() { continue; }
            l.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then(x.0.cmp(&y.0)));
            trigram.insert(pack(a, b), l);
            log_lambda.insert(pack(a, b), -0.7 - (h(a + 3 * b) as f32) / 60.0);
        } }
        for b in 0..n {
            if b == END { continue; }
            let mut l: Vec<(u32, f32)> = (0..10u32).filter(|&t| h(b * 19 + t) % 5 != 0)
                .map(|t| (t, -0.5 - (h(b * 23 + t * 5) as f32) / 30.0)).collect();
            l.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then(x.0.cmp(&y.0)));
            bigram.insert(b, l);
        }
        let unigram_top = if tail { vec![(5u32, -1.2f32), (0, -1.5), (6, -2.5)] } else { Vec::new() };
        EnumModel { is_kn: true, trigram, bigram, log_lambda, unigram_top,
                    unigram_logw: if tail { 0.5f32.ln() } else { -2.302585 }, start_id: START }
    }

    fn synth_decode() -> Vec<Vec<u8>> {
        let mut d: Vec<Vec<u8>> = TOKS.iter().map(|t| t.as_bytes().to_vec()).collect();
        d.push(Vec::new()); // END
        d.push(Vec::new()); // START
        d
    }

    struct MemW(Arc<Mutex<Vec<u8>>>);
    impl Write for MemW {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> { self.0.lock().unwrap().extend_from_slice(b); Ok(b.len()) }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }

    fn opts(at_end: bool, np: bool) -> Opts {
        Opts { max_tokens: 6, min_tokens: 1, min_len: 1, max_len: 30, enterprise: false,
               case_masks: Vec::new(), word_at_end: at_end, word_unigram: Some(1.0), no_prune: np }
    }

    /// Run a strict sweep to `count` from `start`, after restoring `prior`.
    fn sweep_to_vec(em: &EnumModel, variant: &dyn Variant, o: Opts, count: u64, start: Pos,
                    prior: &[&[u8]]) -> Vec<u8> {
        let decode = synth_decode();
        let counts: Vec<u64> = (0..11u64).map(|i| (i * 7) % 5).collect();
        let inp = Inputs { em, variant, decode: &decode, kind: DecodeKind::Plain, start_id: START,
                           end_id: END, unigram_raw: &counts };
        // Words: "ab", "1", "b!" (b + !), "ia" (i + a), "ab1" (a + b + 1).
        let seqs = vec![vec![3u32], vec![2], vec![1, 6], vec![7, 0], vec![0, 1, 2]];
        let p = prepare(inp, &seqs, o).unwrap();
        let buf = Arc::new(Mutex::new(Vec::new()));
        let rt = Rt::new(Box::new(MemW(buf.clone())), count, 4096);
        for l in prior { rt.insert_seen(line_hash(l)); }
        rt.emitted.store(prior.len() as u64, AO::Relaxed);
        sweep(&p, &rt, 1, start).unwrap();
        let v = buf.lock().unwrap().clone();
        v
    }

    fn lines(v: &[u8]) -> Vec<&[u8]> { v.split(|&c| c == b'\n').filter(|l| !l.is_empty()).collect() }

    /// The bounds only skip work: with every bound at 0 the sweep emits the
    /// same bytes, for the plain model, a unigram-tail model, and the
    /// word-at-end mode.
    #[test]
    fn pruned_sweep_equals_unpruned_sweep() {
        for tail in [false, true] {
            let em = synth_em(tail);
            let var: &dyn Variant = if tail { &crate::variant_b::B } else { &crate::variant_a::A };
            for at_end in [false, true] {
                let a = sweep_to_vec(&em, var, opts(at_end, false), 400, Pos::default(), &[]);
                let b = sweep_to_vec(&em, var, opts(at_end, true), 400, Pos::default(), &[]);
                assert!(lines(&a).len() > 50, "tail={tail} at_end={at_end}: only {} lines", lines(&a).len());
                assert_eq!(String::from_utf8_lossy(&a), String::from_utf8_lossy(&b), "tail={tail} at_end={at_end}");
                let mut u = lines(&a);
                u.sort();
                let n = u.len();
                u.dedup();
                assert_eq!(u.len(), n, "duplicate lines");
            }
        }
    }

    /// With --prepend-only every line ends with a word.
    #[test]
    fn word_at_end_lines_end_with_a_word() {
        let em = synth_em(false);
        let v = sweep_to_vec(&em, &crate::variant_a::A, opts(true, false), 300, Pos::default(), &[]);
        let words: [&[u8]; 4] = [b"ab", b"1", b"b!", b"ia"];
        for l in lines(&v) {
            assert!(words.iter().any(|w| l.ends_with(w)), "{}", String::from_utf8_lossy(l));
        }
    }

    /// Resume: restore the first lines into the dedupe set and re-run from the
    /// start; the output continues exactly where the uninterrupted run was.
    #[test]
    fn resume_from_written_lines_continues_byte_for_byte() {
        let em = synth_em(false);
        let full = sweep_to_vec(&em, &crate::variant_a::A, opts(false, false), 300, Pos::default(), &[]);
        let ls = lines(&full);
        for cut in [0usize, 1, 37, 150, 299] {
            let rest = sweep_to_vec(&em, &crate::variant_a::A, opts(false, false), 300, Pos::default(), &ls[..cut]);
            let mut want: Vec<u8> = Vec::new();
            for l in &ls[cut..] { want.extend_from_slice(l); want.push(b'\n'); }
            assert_eq!(String::from_utf8_lossy(&rest), String::from_utf8_lossy(&want), "cut {cut}");
        }
    }

    /// Resuming at a level with some units marked finished skips them.
    #[test]
    fn finished_units_are_skipped() {
        let em = synth_em(false);
        let decode = synth_decode();
        let counts = vec![1u64; 11];
        let inp = Inputs { em: &em, variant: &crate::variant_a::A, decode: &decode, kind: DecodeKind::Plain,
                           start_id: START, end_id: END, unigram_raw: &counts };
        let p = prepare(inp, &[vec![3u32], vec![2]], opts(false, false)).unwrap();
        let n = p.n_units();
        let start = Pos { level: 0, units: n, done: vec![true; n], deferred: false };
        let buf = Arc::new(Mutex::new(Vec::new()));
        let rt = Rt::new(Box::new(MemW(buf.clone())), 100, 4096);
        // Every unit finished and nothing deferred: the sweep ends at once.
        assert_eq!(sweep(&p, &rt, 1, start).unwrap(), 0);
        assert!(buf.lock().unwrap().is_empty());
        let bad = Pos { level: 0, units: n + 1, done: vec![false; n + 1], deferred: false };
        assert!(sweep(&p, &rt, 1, bad).is_err());
    }

    /// Apply `f` (None = drop) to each line of `plain`, keep first occurrences.
    fn derive(plain: &[u8], f: &dyn Fn(&[u8]) -> Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        let mut seen: FxHashSet<Vec<u8>> = FxHashSet::default();
        let mut out = Vec::new();
        for l in lines(plain) { for x in f(l) { if seen.insert(x.clone()) { out.push(x); } } }
        out
    }

    fn take(v: &[u8]) -> Vec<Vec<u8>> { lines(v).into_iter().map(|l| l.to_vec()).collect() }

    /// --enterprise, --case-shape and the length filters change only what a
    /// candidate is written as: each run equals its transform applied to the
    /// plain stream, deduplicated, in the same order.
    #[test]
    fn transforms_are_the_plain_stream_transformed() {
        let em = synth_em(false);
        let a: &dyn Variant = &crate::variant_a::A;
        let plain = sweep_to_vec(&em, a, opts(false, false), 20_000, Pos::default(), &[]);
        // --enterprise (min length 8 here, so long lines only).
        let mut o = opts(false, false);
        o.enterprise = true;
        let got = take(&sweep_to_vec(&em, a, o, 40, Pos::default(), &[]));
        let want = derive(&plain, &|l: &[u8]| match enterprise::decide(l) {
            enterprise::Decision::AsIs => vec![l.to_vec()],
            enterprise::Decision::Cap => { let mut x = l.to_vec(); x[0] = x[0].to_ascii_uppercase(); vec![x] }
            enterprise::Decision::Drop => vec![],
        });
        assert_eq!(got.len(), 40);
        assert_eq!(got[..], want[..40]);
        // --case-shape "lower;upper": both forms of each candidate, in order
        // (`lower` leaves a token as it is, as in the enumerator).
        let mut o = opts(false, false);
        o.case_masks = CaseMask::parse_spec("lower;upper").unwrap();
        let got = take(&sweep_to_vec(&em, a, o, 300, Pos::default(), &[]));
        let want = derive(&plain, &|l: &[u8]| vec![l.to_vec(), l.to_ascii_uppercase()]);
        assert_eq!(got.len(), 300);
        assert_eq!(got[..], want[..300]);
        // --min-len / --max-len filter the same stream.
        let mut o = opts(false, false);
        o.min_len = 4; o.max_len = 6;
        let got = take(&sweep_to_vec(&em, a, o, 300, Pos::default(), &[]));
        let want = derive(&plain, &|l: &[u8]| if (4..=6).contains(&l.len()) { vec![l.to_vec()] } else { vec![] });
        assert_eq!(got.len(), 300);
        assert_eq!(got[..], want[..300]);
    }

    /// --case-shape: each pattern re-cases by token position over the whole
    /// candidate; lines stay unique.
    #[test]
    fn case_shape_applies_per_token_and_dedupes() {
        let em = synth_em(false);
        let mut o = opts(false, false);
        o.case_masks = CaseMask::parse_spec("lower;?u?l").unwrap();
        let v = sweep_to_vec(&em, &crate::variant_a::A, o, 200, Pos::default(), &[]);
        let ls = lines(&v);
        assert!(ls.iter().any(|l| l.iter().any(|c| c.is_ascii_uppercase())));
        let mut u = ls.clone();
        u.sort();
        u.dedup();
        assert_eq!(u.len(), ls.len());
    }
}
