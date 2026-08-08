//! Variant B — extends Variant A with a unigram backoff
//! tier sourced from raw token frequencies. After trigram + bigram-KN are
//! exhausted, emit up to `MAX_UNIGRAM_BACKOFF_CHILDREN` unigram-tier
//! candidates dedup'd against both higher tiers, weighted by
//! `log_lambda(a, b) + LOG_LAMBDA_BIGRAM`.
//!
//! Mechanistic motivation: PCFG has no continuation-count discounting,
//! while KN's continuation-count formula demotes tokens that appear in few
//! distinct contexts (e.g. `123`). Raw frequency restores those tokens to
//! the candidate stream when bigram tiers exhaust.
//!
//! Empirical results (rockyou + best66 @ 500M, cap=32):
//!   raw 500M total: +3,759 cracks (+0.75% across 20 corpora)
//!   +best66 rockyou: +0.14pp (62.05% → 62.19%)
//!   +best66 enterprise_union: +0.22pp (21.45% → 21.66%)

use std::cmp::Ordering;

use crate::{Ctx, Model, MAX_KN_BIGRAM_CHILDREN};
use crate::variant::{EnumModel, Variant};

/// Max children to add from the unigram-tier backoff per (a, b) context.
/// cap=128 is the memory ceiling on a 32 GB machine; larger caps OOM during
/// child_cache construction. cap-sweep showed quality scales super-linearly
/// (+0.07% at cap=32 → +0.36% at cap=128 on rockyou 100M).
pub const MAX_UNIGRAM_BACKOFF_CHILDREN: usize = 128;

/// Log-weight applied to unigram-tier emissions in addition to log_lambda(a, b).
/// Effective weight = exp(log_lambda(a, b)) * 0.1; the unigram tier gets ~10%
/// of the bigram tier's missing-mass budget. Constant for now; could become a
/// per-bigram dynamic lambda_bigram(b) in a future revision. log(0.1) ≈ -2.302585.
pub const LOG_LAMBDA_BIGRAM: f32 = -2.302585;

/// Variant B — KN with raw-frequency unigram backoff tier.
pub struct B;

impl Variant for B {
    fn name(&self) -> &'static str { "freq-tail" }

    fn prepare(&self, model: &Model) -> EnumModel {
        // Trigram + bigram + log_lambda are identical to Variant A. Reuse
        // by delegating to Variant A's prepare, then add unigram_top.
        let mut em = crate::variant_a::A.prepare(model);

        // Build top-K raw-frequency unigram backoff list. Used by
        // get_children as a third tier after trigram + bigram. The
        // unigram_raw array on the model is captured at training time and
        // persists in every NGRMv002 model — no rebuild needed.
        let total_unigram: u64 = model.unigram_raw.iter().sum();
        // Experiment knobs (default = canonical Variant B): TOKENOV_UNIGRAM_CAP
        // overrides the top-K cap (e.g. huge = "uncapped"); TOKENOV_UNIGRAM_FLOOR
        // adds an add-1 floor so zero-count tokens are included too (full smoothing).
        let cap = std::env::var("TOKENOV_UNIGRAM_CAP").ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(MAX_UNIGRAM_BACKOFF_CHILDREN);
        let use_floor = std::env::var("TOKENOV_UNIGRAM_FLOOR").is_ok();
        em.unigram_top = if total_unigram > 0 {
            let vocab = model.unigram_raw.len() as f64;
            let add_k = 1.0_f64;
            let denom = total_unigram as f64 + if use_floor { add_k * vocab } else { 0.0 };
            let mut entries: Vec<(u32, f32)> = model.unigram_raw.iter().enumerate()
                .filter_map(|(i, &c)| {
                    if c == 0 && !use_floor { return None; }
                    let num = c as f64 + if use_floor { add_k } else { 0.0 };
                    let p = (num / denom) as f32;
                    if p <= 0.0 { return None; }
                    Some((i as u32, p.ln()))
                })
                .collect();
            entries.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
            });
            entries.truncate(cap);
            entries
        } else {
            Vec::new()
        };
        crate::log_msg(&format!(
            "[enum] Variant B unigram backoff tier: {} entries (cap {}), \
             top log_prob = {:.3}, bottom log_prob = {:.3}",
            em.unigram_top.len(),
            MAX_UNIGRAM_BACKOFF_CHILDREN,
            em.unigram_top.first().map(|(_, lp)| *lp).unwrap_or(f32::NAN),
            em.unigram_top.last().map(|(_, lp)| *lp).unwrap_or(f32::NAN),
        ));
        em
    }

    fn get_children(&self, em: &EnumModel, ctx: Ctx, b: u32) -> Vec<(u32, f32)> {
        let mut out: Vec<(u32, f32)> = Vec::new();
        let mut tri_seen: Vec<u32> = Vec::new();
        let mut bi_seen: Vec<u32> = Vec::new();
        if let Some(tri) = em.trigram.get(&ctx) {
            for &(id, lp) in tri {
                out.push((id, lp));
                tri_seen.push(id);
            }
        }
        if em.is_kn {
            if let Some(bi) = em.bigram.get(&b) {
                let log_lam = em.log_lambda.get(&ctx).copied().unwrap_or(0.0);
                let mut n = 0usize;
                for &(id, lp_cont) in bi {
                    if n >= MAX_KN_BIGRAM_CHILDREN { break; }
                    if !tri_seen.contains(&id) {
                        out.push((id, log_lam + lp_cont));
                        bi_seen.push(id);
                        n += 1;
                    }
                }
            }
            // Unigram-tier backoff. After trigram + bigram are exhausted,
            // emit up to MAX_UNIGRAM_BACKOFF_CHILDREN entries from the
            // precomputed top-K, dedup'd against both higher tiers.
            // Weight = log_lambda(a, b) + LOG_LAMBDA_BIGRAM.
            // unigram_top is already truncated to the (env-configurable) cap in
            // prepare, so emit all of it (deduped against higher tiers).
            if !em.unigram_top.is_empty() {
                let log_lam_ab = em.log_lambda.get(&ctx).copied().unwrap_or(0.0);
                for &(id, lp_uni) in &em.unigram_top {
                    if !tri_seen.contains(&id) && !bi_seen.contains(&id) {
                        out.push((id, log_lam_ab + LOG_LAMBDA_BIGRAM + lp_uni));
                    }
                }
            }
        }
        out.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        out
    }
}
