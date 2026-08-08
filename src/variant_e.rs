//! Variant E — capital-biased unigram backoff tier.
//!
//! v3/v4 attempted case-variant expansion at the trigram/bigram (head) tier
//! — both lost cracks because expansion DISPLACES head candidates at fixed
//! budget, and training data is overwhelmingly lowercase. The displacement
//! cost exceeded the capital coverage gain.
//!
//! v5 inverts the approach: leave head untouched (Variant A) and add a
//! Variant-B-style unigram tail with case-aware ranking. Variant B's tail
//! ranks tokens by raw c(t) — almost all top-128 entries are lowercase
//! because lowercase tokens dominate training counts. Capital-first variants
//! never make the cut.
//!
//! v5 ranks by case-GROUP frequency (sum of c(t') over case-collapsed
//! group), so capital-first variants of common lowercase tokens are admitted
//! to the tail. For each top group, the lowercase head and capital-first
//! sibling are emitted adjacently in unigram_top, so the per-context
//! emission cap (`MAX_UNIGRAM_BACKOFF_CHILDREN`) admits both before saturating.
//!
//! This combines Variant B's "fill unused tail capacity" mechanism with
//! capital coverage — additive only, no head displacement at fixed budget.
//!
//! Mass: each tail entry emits at log(c(t)/total_unigram), its true unigram
//! probability. Only ranking is case-aware; per-token mass is unchanged.

use std::cmp::Ordering;
use rustc_hash::FxHashMap;

use crate::{Ctx, Model, MAX_KN_BIGRAM_CHILDREN};
use crate::variant::{EnumModel, Variant};

/// Same as Variant B — per-(a, b) cap on emitted unigram-tier children.
pub const MAX_UNIGRAM_BACKOFF_CHILDREN: usize = 128;

/// Same as Variant B — log(0.1), unigram tier gets 10% of bigram missing-mass.
pub const LOG_LAMBDA_BIGRAM: f32 = -2.302585;

/// Build-time cap on unigram_top length. Twice the emit cap so each top
/// group's lowercase + capital pair both make the list before truncation.
const UNIGRAM_TOP_BUILD_CAP: usize = 256;

/// Variant E — KN with capital-biased unigram backoff tier.
pub struct E;

fn case_collapse(bytes: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(bytes.len());
    for &b in bytes {
        if b.is_ascii_uppercase() {
            v.push(b.to_ascii_lowercase());
        } else {
            v.push(b);
        }
    }
    v
}

impl Variant for E {
    fn name(&self) -> &'static str { "cap-tail" }

    fn prepare(&self, model: &Model) -> EnumModel {
        // Trigram + bigram + log_lambda are identical to Variant A. Reuse
        // by delegating; v5 does not modify the head tiers.
        let mut em = crate::variant_a::A.prepare(model);

        let total_unigram: u64 = model.unigram_raw.iter().sum();
        if total_unigram == 0 {
            crate::log_msg("[enum] Variant E v5: total_unigram=0, no tail tier built");
            return em;
        }
        let inv_total = 1.0_f64 / total_unigram as f64;

        // Group tokens by case-collapsed bytes. Each group's members keep
        // their (id, c(t)) so we can rank within-group by c(t) desc.
        let vocab_size = model.unigram_raw.len();
        let mut groups: FxHashMap<Vec<u8>, Vec<(u32, u64)>> = FxHashMap::default();
        for tid in 0..vocab_size {
            let c = model.unigram_raw[tid];
            if c == 0 { continue; }
            let bytes = if tid < model.decode.len() { &model.decode[tid] } else { continue };
            if bytes.is_empty() { continue; }
            let key = case_collapse(bytes);
            groups.entry(key).or_default().push((tid as u32, c));
        }

        // For each group, compute group_freq = sum c(t') and sort members by
        // c(t) desc (lowercase head first, capital-first sibling next).
        let mut group_list: Vec<(Vec<(u32, u64)>, u64)> = groups.into_iter()
            .map(|(_, mut members)| {
                members.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                let gf: u64 = members.iter().map(|m| m.1).sum();
                (members, gf)
            })
            .collect();
        // Sort groups by group_freq desc — top groups emit first.
        group_list.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        // Build interleaved unigram_top. For each group (in group-freq order),
        // emit members (in c(t) order). This places lowercase heads and
        // capital-first siblings adjacently in unigram_top, so when
        // get_children walks unigram_top with a 128-emission cap, both
        // make it past the cap (sample math: top-64 case-groups × 2 members
        // ≈ 128 emissions, of which ~half are capital).
        let mut unigram_top: Vec<(u32, f32)> = Vec::new();
        let mut groups_emitted = 0usize;
        let mut multi_member_groups = 0usize;
        for (members, _gf) in &group_list {
            if unigram_top.len() >= UNIGRAM_TOP_BUILD_CAP { break; }
            groups_emitted += 1;
            if members.len() > 1 { multi_member_groups += 1; }
            for &(tid, c) in members {
                if unigram_top.len() >= UNIGRAM_TOP_BUILD_CAP { break; }
                let p = (c as f64 * inv_total) as f32;
                if p > 0.0 {
                    unigram_top.push((tid, p.ln()));
                }
            }
        }

        let n_capital_in_top = unigram_top.iter()
            .filter(|(id, _)| {
                let i = *id as usize;
                if i >= model.decode.len() { return false; }
                model.decode[i].first().is_some_and(|&b| b.is_ascii_uppercase())
            })
            .count();

        em.unigram_top = unigram_top;

        crate::log_msg(&format!(
            "[enum] Variant E v5 capital-biased tail: {} entries (build-cap {}, emit-cap {}), \
             {} groups walked, {} multi-member groups, {} capital-leading entries, \
             top log_prob = {:.3}, bottom log_prob = {:.3}",
            em.unigram_top.len(),
            UNIGRAM_TOP_BUILD_CAP,
            MAX_UNIGRAM_BACKOFF_CHILDREN,
            groups_emitted,
            multi_member_groups,
            n_capital_in_top,
            em.unigram_top.first().map(|(_, lp)| *lp).unwrap_or(f32::NAN),
            em.unigram_top.last().map(|(_, lp)| *lp).unwrap_or(f32::NAN),
        ));
        em
    }

    fn get_children(&self, em: &EnumModel, ctx: Ctx, b: u32) -> Vec<(u32, f32)> {
        // Identical to Variant B's get_children. The difference is in
        // em.unigram_top's CONTENTS (case-aware ranking), not in the loop.
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
            if !em.unigram_top.is_empty() {
                let log_lam_ab = em.log_lambda.get(&ctx).copied().unwrap_or(0.0);
                let mut n = 0usize;
                for &(id, lp_uni) in &em.unigram_top {
                    if n >= MAX_UNIGRAM_BACKOFF_CHILDREN { break; }
                    if !tri_seen.contains(&id) && !bi_seen.contains(&id) {
                        out.push((id, log_lam_ab + LOG_LAMBDA_BIGRAM + lp_uni));
                        n += 1;
                    }
                }
            }
        }
        out.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        out
    }
}
