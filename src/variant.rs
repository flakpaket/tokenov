//! Variant trait and `EnumModel` struct.
//!
//! Each tokenov variant (algorithm) implements this trait. The `Variant` impl
//! lives in its own module (`variant_a.rs`, `variant_b.rs`, etc.) so each
//! variant's code is physically isolated — a change to one variant's source
//! cannot accidentally affect another's compiled output.
//!
//! `EnumModel` carries runtime state derived from the saved `Model`. It has
//! optional fields (currently `unigram_top`) that variants populate as needed.
//! Variants that don't need a particular field leave it empty / default.

use crate::{Ctx, Model};
use std::sync::Arc;

#[allow(dead_code)] // some fields are read directly via field access from main.rs
pub struct EnumModel {
    pub is_kn: bool,
    /// Trigram children with their per-child log-prob.
    /// V1: sum exp(log_p) per ctx = 1.
    /// V2: sum exp(log_p) per ctx = (1 - lambda(ctx)); rest goes to bigram.
    pub trigram: rustc_hash::FxHashMap<Ctx, Vec<(u32, f32)>>,
    /// KN-only: per-bigram-context (the second token of the trigram context),
    /// sorted (id, log_prob) for KN-continuation P_cont(t | b).
    pub bigram: rustc_hash::FxHashMap<u32, Vec<(u32, f32)>>,
    /// KN-only: log of lambda(a, b) per trigram context. Used to scale
    /// bigram-back-off children's log-prob: log(lambda) + log(P_cont).
    pub log_lambda: rustc_hash::FxHashMap<Ctx, f32>,
    /// Variant B/E: top-K entries from a unigram backoff distribution.
    /// Empty for Variant A (no unigram tier).
    pub unigram_top: Vec<(u32, f32)>,
    /// Log-weight applied to unigram-tier emissions, on top of log_lambda(a, b).
    /// Set from `--unigram-tail [FRACTION]` (ln of the fraction); defaults to
    /// ln(0.1) — the tier gets 10% of the bigram tier's missing-mass budget.
    /// Unused by variants without a unigram tier.
    pub unigram_logw: f32,
    /// Start-of-sequence sentinel token id. Variants that need to detect
    /// the start context (e.g., Variant E for start-position-only case
    /// expansion) read this. Set by `Variant::prepare`.
    pub start_id: u32,
}

/// Tokenov variant — encapsulates the algorithm-specific parts of
/// `prepare_enum_model` (build runtime state) and `get_children` (emit
/// child candidates per context).
///
/// All shared infrastructure (model load, sink, merger, DFS, level sweep)
/// is in `main.rs` and is variant-agnostic. The trait is `Send + Sync` so
/// `Arc<dyn Variant>` can be shared across worker threads.
pub trait Variant: Send + Sync {
    /// Short identifier. Must match the `--variant` CLI flag value.
    fn name(&self) -> &'static str;

    /// Build the runtime `EnumModel` from the saved `Model`. Variant-specific
    /// fields (e.g. `unigram_top`) are populated here.
    fn prepare(&self, model: &Model, unigram_logw: f32) -> EnumModel;

    /// Return the children to emit for context (a, b). The returned vec is
    /// sorted descending by log-prob. Per-context cap is variant-specific.
    fn get_children(&self, em: &EnumModel, ctx: Ctx, b: u32) -> Vec<(u32, f32)>;
}

/// Resolve a variant name to a shared trait object. The canonical names are
/// descriptive; the single letters are back-compat aliases for the old
/// `a`/`b`/`e` values.
pub fn dispatch(name: &str) -> anyhow::Result<Arc<dyn Variant>> {
    match name {
        "baseline" | "a" => Ok(Arc::new(crate::variant_a::A)),
        "freq-tail" | "b" => Ok(Arc::new(crate::variant_b::B)),
        "cap-tail" | "e" => Ok(Arc::new(crate::variant_e::E)),
        other => anyhow::bail!(
            "unknown variant: {:?}. Known variants: baseline, freq-tail, cap-tail", other),
    }
}
