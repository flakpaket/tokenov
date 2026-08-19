//! Variant A — the original baseline. Trigram + bigram-KN backoff, no
//! unigram tier — the original baseline implementation, kept as the
//! reference point for all subsequent variants.

use std::cmp::Ordering;
use rustc_hash::FxHashMap;

use crate::{Ctx, Model, MAX_KN_BIGRAM_CHILDREN};
use crate::variant::{EnumModel, Variant};

/// Variant A — KN-smoothed trigram + bigram backoff. No unigram tier.
pub struct A;

impl Variant for A {
    fn name(&self) -> &'static str { "baseline" }

    fn prepare(&self, model: &Model, unigram_logw: f32) -> EnumModel {
        let mut trigram: FxHashMap<Ctx, Vec<(u32, f32)>> = FxHashMap::default();
        trigram.reserve(model.contexts.len());
        for (&ctx, (ids, cum)) in &model.contexts {
            let mut entries: Vec<(u32, f32)> = Vec::with_capacity(ids.len());
            let mut prev: f32 = 0.0;
            for (i, &id) in ids.iter().enumerate() {
                let p = cum[i] - prev;
                prev = cum[i];
                let p_clamped = if p > 0.0 { p } else { f32::MIN_POSITIVE };
                entries.push((id, p_clamped.ln()));
            }
            entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            trigram.insert(ctx, entries);
        }

        if !model.is_kn {
            return EnumModel {
                is_kn: false, trigram,
                bigram: FxHashMap::default(),
                log_lambda: FxHashMap::default(),
                unigram_top: Vec::new(),
                unigram_logw,
                start_id: model.start_id,
            };
        }

        let mut bigram: FxHashMap<u32, Vec<(u32, f32)>> = FxHashMap::default();
        bigram.reserve(model.bigram_kn.len());
        for (&b, (ids, cum)) in &model.bigram_kn {
            let mut entries: Vec<(u32, f32)> = Vec::with_capacity(ids.len());
            let mut prev: f32 = 0.0;
            for (i, &id) in ids.iter().enumerate() {
                let p = cum[i] - prev;
                prev = cum[i];
                let p_clamped = if p > 0.0 { p } else { f32::MIN_POSITIVE };
                entries.push((id, p_clamped.ln()));
            }
            entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            bigram.insert(b, entries);
        }
        let mut log_lambda: FxHashMap<Ctx, f32> = FxHashMap::default();
        log_lambda.reserve(model.lambda.len());
        for (&ctx, &lam) in &model.lambda {
            let lam_clamped = if lam > 0.0 { lam } else { f32::MIN_POSITIVE };
            log_lambda.insert(ctx, lam_clamped.ln());
        }
        EnumModel {
            is_kn: true,
            trigram,
            bigram,
            log_lambda,
            unigram_top: Vec::new(),
            unigram_logw,
            start_id: model.start_id,
        }
    }

    fn get_children(&self, em: &EnumModel, ctx: Ctx, b: u32) -> Vec<(u32, f32)> {
        let mut out: Vec<(u32, f32)> = Vec::new();
        let mut tri_seen: Vec<u32> = Vec::new();
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
                        n += 1;
                    }
                }
            }
        }
        out.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        out
    }
}
