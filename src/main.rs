mod mem_monitor;
use mem_monitor::{MemMonitor, MemMonitorConfig, MemReader, ProcMemReader};

// tokenov — token n-gram Markov password generator with optional wordlist targeting.
//
// Two subcommands:
//   tokenov build    — fit a model file from a tokenizer + password list
//   tokenov generate — emit candidates (default if no subcommand)
//
// Ranked enumeration via a bounded max-heap with select_nth_unstable_by prune,
// plus wordlist-targeting modes (weighted / seeded / combined) and an explicit
// `build` subcommand.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded as channel_bounded, Receiver, Sender};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use tokenizers::Tokenizer;

// ============================================================================
// Variant modules — algorithm-specific code lives in dedicated files so each
// variant's source is physically isolated from the others. See `variant.rs`
// for the trait definition. Adding a new variant: create variant_X.rs, impl
// `Variant` for it, and add to `variant::dispatch()`.
// ============================================================================

mod variant;
mod variant_a;
mod variant_b;
mod variant_e;
mod case_mask;
mod enterprise;
mod registry;
mod bootstrap;
mod graft;
mod recover;

use case_mask::{apply_op, CaseMask};

use variant::{EnumModel, Variant};

// ============================================================================
// Constants and types
// ============================================================================

const MODEL_MAGIC_V1: &[u8; 8] = b"NGRMv001";
const MODEL_MAGIC_V2: &[u8; 8] = b"NGRMv002";
// V3 = V2/KN body verbatim, preceded by a length-prefixed provenance blob that
// embeds the tokenizer.json bytes + build metadata. A v3 model is
// self-describing: `generate --wordlist` loads its embedded tokenizer with no
// sidecar, env var, or registry entry. v3 always implies KN (is_kn = true) —
// the only body `model train` produces.
const MODEL_MAGIC_V3: &[u8; 8] = b"NGRMv003";
const DEFAULT_MAX_TOKENS: usize = 12; // heap-level token-count cap

/// Default Kneser-Ney discount (D). Standard NLP value; could be estimated from
/// data via D = n1 / (n1 + 2*n2), but the empirical-vs-fixed difference is
/// usually <5% on crack rates. We compute the data-driven value during build
/// and write it into the header; the runtime uses the stored value.
const DEFAULT_KN_DISCOUNT: f32 = 0.75;

/// Level-sweep enumeration scale. Each transition log-prob lp is discretised as
/// ceil(-lp * LEVEL_SCALE) → a non-negative integer level. A candidate's total
/// level = sum of per-step levels. The outer loop sweeps levels 0, 1, 2, …,
/// emitting all candidates at level L before moving to L+1. This gives
/// approximately rank-ordered output (all level-L candidates precede level-L+1)
/// with O(stack-depth × branching) memory — no heap, no prune cycles.
const LEVEL_SCALE: f32 = 1.0;

/// Practical upper bound on level. At SCALE=1.0 a single transition contributes
/// ceil(|lp|) levels. With 12 max tokens at up to ~50 nats each → ≤ 600 levels.
/// The sweep terminates early once target_count is reached.
const LEVEL_MAX: u32 = 700;

/// Maximum bigram back-off children returned by get_children for KN models.
/// Bigram entries are already sorted descending by continuation probability, so
/// this effectively truncates to the top-N most probable back-offs. Limits the
/// branching factor of the KN enumeration tree and caps ctx_cache entry size.
pub const MAX_KN_BIGRAM_CHILDREN: usize = 200;

/// Per-channel bound: aim for this many items in flight regardless of
/// `chunk_size`. With items at 16 B each + ~16 B/item of bytes-arena
/// (typical password length), 64K items ≈ 2 MB per channel — 20 channels
/// ≈ 40 MB peak buffer pressure. Capacity in CHUNKS is derived from this:
/// `chunks = max(2, MERGE_CHANNEL_BUFFER_ITEMS / chunk_size)`.
const MERGE_CHANNEL_BUFFER_ITEMS: usize = 65_536;


/// Default producer-side chunk size (items per `MergeChunk`), used by the STRICT
/// (merged) path when chunk size isn't pinned. Set to the canonical 262144: large
/// enough that per-chunk overhead is negligible (the merger's real cost is
/// per-item), so the default already captures the chunk-tuning win without
/// calibration. Fast mode (default) ignores this entirely. Override with
/// `--merge-chunk-size`.
const DEFAULT_MERGE_CHUNK_SIZE: usize = 262144;

/// One emission flowing from a producer thread to the merger. `level` and
/// `sort_key` come from the OMEN-style level sweep (see `lp_to_level`); the
/// merger orders by (level asc, sort_key asc, thread_idx asc) to produce a
/// globally rank-ordered output stream. `byte_offset` + `byte_len` index into
/// the parent `MergeChunk.bytes` arena — no per-item heap allocation.
#[derive(Clone, Copy)]
struct MergeItem {
    level:       u32,
    sort_key:    u32,
    byte_offset: u32,
    byte_len:    u16,
}

/// A locally rank-ordered batch of items from one producer. Producers
/// accumulate up to `chunk_size` items in a `MergeChunk` and send the chunk
/// as one channel op. The merger orders chunks by their *first* item's
/// (level, sort_key) — the chunk with the highest-prob head wins. After
/// popping a chunk, the merger drains all of it before consulting the heap
/// again, accepting up to `chunk_size` items of imperfect global ordering
/// (chunk tails can be lower-prob than the next chunk's head).
///
/// Storage layout: per-item metadata in `items`, candidate bytes packed
/// contiguously in `bytes` and addressed by (offset, len). One `Vec<u8>`
/// allocation per chunk regardless of how many items it holds, vs the prior
/// design's K separate `Vec<u8>`s — eliminates the cross-thread free pattern
/// for tiny per-candidate Vecs.
struct MergeChunk {
    items: Vec<MergeItem>,
    bytes: Vec<u8>,
}

impl MergeChunk {
    fn with_capacity(items_cap: usize, bytes_cap: usize) -> Self {
        Self {
            items: Vec::with_capacity(items_cap),
            bytes: Vec::with_capacity(bytes_cap),
        }
    }
    #[inline]
    fn is_empty(&self) -> bool { self.items.is_empty() }
}

/// Producer-side accumulator: buffers items and flushes a full chunk to the
/// channel. Drop alone is not a flush — the worker must call `flush` after
/// `enumerate_to_sink` returns so any partial trailing chunk reaches the
/// merger. The channel sender drops on `ChunkSender` drop, signaling
/// end-of-stream to the merger.
///
/// `chunk_size` is shared via `Arc<AtomicUsize>` so a controller thread (e.g.
/// the calibration loop) can change it mid-stream. Each `push` reads the
/// current value with a relaxed atomic load (~1 ns); on detection of a change
/// the sender flushes any partial chunk so the next chunk respects the new
/// size. Different producers can hold different `Arc`s and use different K
/// independently — the merger does not care.
/// Per-item byte budget used when sizing a fresh chunk's `bytes` arena.
/// Average decoded password length is well below this; over-allocation is
/// cheap (one Vec) and avoids `bytes` growing inside the hot push loop.
const CHUNK_BYTES_PER_ITEM_HINT: usize = 32;

struct ChunkSender {
    tx:         Sender<MergeChunk>,
    buf:        MergeChunk,
    chunk_size: Arc<AtomicUsize>,
    last_seen:  usize,                              // cached value to skip atomic read once per emit
}

impl ChunkSender {
    fn new(tx: Sender<MergeChunk>, chunk_size: Arc<AtomicUsize>) -> Self {
        let initial = chunk_size.load(AtomicOrdering::Relaxed).max(1);
        Self {
            tx,
            buf: MergeChunk::with_capacity(initial, initial * CHUNK_BYTES_PER_ITEM_HINT),
            chunk_size,
            last_seen: initial,
        }
    }

    fn push(&mut self, level: u32, sort_key: u32, bytes: &[u8]) -> Result<()> {
        // u16 byte_len ceiling: 65 535 bytes/candidate is far above any password
        // length we'd ever emit; truncating here would be a bug elsewhere, so
        // assert in debug and clamp in release.
        debug_assert!(bytes.len() <= u16::MAX as usize,
            "candidate length {} exceeds u16 byte_len ceiling", bytes.len());
        let len = bytes.len().min(u16::MAX as usize) as u16;
        let off = self.buf.bytes.len() as u32;
        self.buf.bytes.extend_from_slice(&bytes[..len as usize]);
        self.buf.items.push(MergeItem {
            level, sort_key, byte_offset: off, byte_len: len,
        });
        if self.buf.items.len() >= self.last_seen {
            // Time to ship a chunk — also a natural moment to reload K.
            // The atomic load is one-per-K-items, so cost is ~zero.
            let new_k = self.chunk_size.load(AtomicOrdering::Relaxed).max(1);
            self.last_seen = new_k;
            let fresh = MergeChunk::with_capacity(new_k, new_k * CHUNK_BYTES_PER_ITEM_HINT);
            let full = std::mem::replace(&mut self.buf, fresh);
            self.tx.send(full)
                .map_err(|_| anyhow!("merger channel closed (consumer hung up)"))?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if !self.buf.is_empty() {
            let fresh = MergeChunk::with_capacity(self.last_seen.max(1),
                self.last_seen.max(1) * CHUNK_BYTES_PER_ITEM_HINT);
            let chunk = std::mem::replace(&mut self.buf, fresh);
            self.tx.send(chunk)
                .map_err(|_| anyhow!("merger channel closed (consumer hung up)"))?;
        }
        Ok(())
    }
}

pub type Ctx = u64; // (id_a, id_b) packed as (a << 32) | b

#[inline(always)]
fn pack(a: u32, b: u32) -> Ctx { ((a as u64) << 32) | (b as u64) }

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser)]
#[command(name = "tokenov", version, about = "token n-gram Markov password generator")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Deprecated: use `tokenov model`. Hidden alias, still functional.
    #[arg(long, global = true, hide = true)]
    list_models: bool,

    /// Verbose stderr: model-load, tuning, and progress detail. Default is quiet
    /// (only warnings/errors print; stdout still carries candidates). Also sets the
    /// telemetry `[stats]` line to every tick.
    #[arg(short = 'v', long, global = true, default_value_t = false)]
    verbose: bool,

    // Allow the bare `tokenov [generate-flags]` form by hoisting Generate flags
    // up. clap does not natively support "default subcommand" in derive mode;
    // we route through the command handling in main() below.
    #[command(flatten)]
    generate_args: GenerateArgs,

    // Declared after the flatten so the "Generation" heading leads `-h` (heading
    // order follows first-encounter). These two join the "Resume & checkpointing"
    // block via the shared heading regardless of position.
    /// List recent generation sessions and exit (newest first).
    #[arg(long, help_heading = "Resume & checkpointing")]
    sessions: bool,

    /// Resume a recorded session by ID (see `--sessions`).
    #[arg(long, value_name = "ID", help_heading = "Resume & checkpointing")]
    resume_session: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Manage models: list (default), train, register, delete.
    Model(ModelArgs),
    /// Manage tokenizers: list/status (default), get, add, delete.
    Tokenizer(TokenizerArgs),
    /// Emit candidates (default if no subcommand is given).
    Generate(GenerateArgs),
    /// Score candidate password(s): report each candidate's Rarity (surprisal,
    /// in bits) under the model. Reads candidates from args, --file, or stdin.
    Score(ScoreArgs),
    /// Quickstart: fetch a tokenizer + RockYou, then train + register a model.
    Bootstrap(bootstrap::BootstrapArgs),

    /// Measure merge throughput and write a chunk-size sidecar. Rarely needed.
    #[command(hide = true)]
    Calibrate(CalibrateArgs),

    // ── Deprecated top-level aliases (hidden; warn + route to the new homes).
    //    Kept so existing scripts keep working; removed in a later release.
    #[command(hide = true)]
    Build(BuildArgs),
    #[command(hide = true, alias = "rm")]
    Delete(DeleteArgs),
    #[command(hide = true)]
    Register(RegisterArgs),
    #[command(hide = true)]
    Fetch(bootstrap::FetchArgs),
}

#[derive(clap::Args, Debug, Clone)]
pub struct ModelArgs {
    #[command(subcommand)]
    cmd: Option<ModelCmd>,
}

#[derive(Subcommand, Debug, Clone)]
enum ModelCmd {
    /// Fit a model from a tokenizer and a password list.
    Train(BuildArgs),
    /// Register an existing model file in the registry.
    Register(RegisterArgs),
    /// Remove a model (registry entry + on-disk file).
    #[command(alias = "rm")]
    Delete(DeleteArgs),
    /// List registered models (default).
    List,
    /// Show a model's embedded provenance (tokenizer, train corpus, build info).
    Info(InfoArgs),
    /// Verify a registered model's SHA-256 against the recorded hash (integrity check).
    Verify(VerifyArgs),
}

#[derive(clap::Args, Debug, Clone)]
struct InfoArgs {
    /// Registered model name, or a path to a .ngram file.
    model: PathBuf,
}

#[derive(clap::Args, Debug, Clone)]
struct VerifyArgs {
    /// Registered model name to verify. Omit to verify every registered model.
    name: Option<String>,
    /// Record the current file hash for entries that have none yet (backfill
    /// models built before hashing). Never overwrites a recorded hash on mismatch.
    #[arg(long)]
    update: bool,
}

#[derive(clap::Args, Debug, Clone)]
pub struct TokenizerArgs {
    #[command(subcommand)]
    cmd: Option<TokenizerCmd>,
}

#[derive(Subcommand, Debug, Clone)]
enum TokenizerCmd {
    /// Train a BPE tokenizer from a corpus, writing a `tokenizer.json`.
    Train(TokenizerTrainArgs),
    /// Download tokenizer(s) by alias (or --all).
    Get(bootstrap::FetchArgs),
    /// Add a tokenizer to the manifest: alias + URL-or-file [+ note].
    Add(bootstrap::AddArgs),
    /// Remove a downloaded tokenizer (and a user-added manifest entry).
    #[command(alias = "rm")]
    Delete(bootstrap::TokDeleteArgs),
    /// Set the default tokenizer/model alias (used by `generate` with no --model).
    #[command(alias = "default")]
    SetDefault(bootstrap::SetDefaultArgs),
    /// List tokenizers with download status (default).
    List,
}

#[derive(clap::Args, Debug, Clone)]
struct TokenizerTrainArgs {
    /// Training corpus: one entry (password) per line, UTF-8.
    #[arg(long, value_name = "FILE")]
    corpus: PathBuf,

    /// Where to write the trained tokenizer.json.
    #[arg(long, value_name = "FILE")]
    output: PathBuf,

    /// Target vocabulary size (number of tokens).
    ///
    /// Smaller vocabularies (~1k–8k) often crack password data better than the
    /// 30k+ typical of web-text tokenizers, and the sweet spot grows with the
    /// amount of unique training data — sweep it for your corpus. Must be >= 256
    /// (the byte-level alphabet alone fills 256 slots).
    #[arg(long, default_value_t = 8000, value_name = "N")]
    vocab_size: usize,

    /// Minimum pair frequency for a BPE merge to be learned.
    #[arg(long, default_value_t = 2, value_name = "N")]
    min_frequency: u64,

    /// Overwrite --output if it already exists.
    #[arg(long, default_value_t = false)]
    force: bool,
}

#[derive(clap::Args, Debug, Clone)]
struct RegisterArgs {
    /// Path to an existing .ngram model file.
    path: PathBuf,

    /// Registry name. Defaults to the file stem.
    #[arg(long)]
    name: Option<String>,

    /// Overwrite an existing registry entry of the same name without confirming.
    #[arg(short = 'y', long)]
    yes: bool,
}

#[derive(clap::Args, Debug, Clone)]
struct DeleteArgs {
    /// Registered model name (or a path). Required unless --missing is given.
    name: Option<String>,

    /// Remove every registry entry whose file no longer exists (the
    /// `⚠ MISSING` rows). Ignores NAME.
    #[arg(long)]
    missing: bool,

    /// Deregister only — leave the model file on disk.
    #[arg(long)]
    keep_file: bool,

    /// Delete the file without confirmation.
    #[arg(short = 'y', long)]
    yes: bool,
}

#[derive(clap::Args, Debug, Clone)]
struct BuildArgs {
    /// HuggingFace tokenizer file (typically tokenizer.json) defining the
    /// model's token alphabet.
    #[arg(long)]
    tokenizer: PathBuf,

    /// Password list, one per line, UTF-8. Used to fit conditional probability
    /// distributions over n-grams of tokens by counting.
    #[arg(long)]
    train: PathBuf,

    /// Where to write the model file. Conventional extension: .ngram.
    /// Optional: if omitted, writes to the default model store
    /// (`$XDG_DATA_HOME/tokenov/models/<name>.ngram`, else
    /// `~/.local/share/tokenov/models/<name>.ngram`), where <name> is --name if
    /// given, else the --tokenizer basename. The registry name matches the
    /// output file stem.
    #[arg(long)]
    output: Option<PathBuf>,

    /// N-gram order. Default 3 (trigram). Higher orders are sparser; fitting a
    /// 4-gram from <10M passwords typically over-memorizes the train set.
    /// (Accepts the legacy spelling `--order` as a hidden alias.)
    #[arg(long = "ngram", short = 'n', alias = "order", default_value_t = 3)]
    order: usize,

    /// Human-readable identifier embedded in logs (not stored in the file).
    /// Defaults to the basename of --tokenizer.
    #[arg(long)]
    name: Option<String>,

    /// Cap the model to the first N token IDs of the tokenizer. Default: full
    /// tokenizer vocab.
    #[arg(long)]
    max_vocab: Option<u32>,

    /// Overwrite an existing --output path. Default refuses to clobber.
    #[arg(long)]
    force: bool,
}

/// Parse a candidate count: a plain non-negative integer, or a number with a
/// trailing magnitude suffix (case-insensitive) K=10^3, M=10^6, B=10^9 (billion),
/// T=10^12. Decimals are accepted only when the result is a whole number of
/// candidates (`1.5M` -> 1_500_000; `1.2345K` is rejected). Overflow of u64 and
/// malformed input are rejected rather than silently wrapped. Exact fixed-point
/// arithmetic (no float rounding), so values like `1.1M` -> 1_100_000 are precise.
fn parse_count(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty count".to_string());
    }
    let last = s.chars().last().unwrap();
    let mult: u64 = match last.to_ascii_lowercase() {
        'k' => 1_000,
        'm' => 1_000_000,
        'b' => 1_000_000_000,
        't' => 1_000_000_000_000,
        _ => {
            // No suffix: a plain non-negative integer.
            return s.parse::<u64>().map_err(|_| {
                format!("invalid count '{s}': expected a non-negative integer, optionally suffixed with K/M/B/T")
            });
        }
    };
    let num = &s[..s.len() - last.len_utf8()];
    if num.is_empty() {
        return Err(format!("invalid count '{s}': missing number before '{last}'"));
    }
    // Split into integer/fraction digits and evaluate as fixed-point in u128 so
    // both `100k` and `1.5M` go through the same exact path (no f64 rounding).
    let (int_part, frac_part) = num.split_once('.').unwrap_or((num, ""));
    let digits = format!("{int_part}{frac_part}");
    let mantissa: u128 = digits.parse().map_err(|_| {
        format!("invalid count '{s}': '{num}' is not a valid number")
    })?;
    let scale = 10u128.pow(frac_part.len() as u32);
    let numer = mantissa
        .checked_mul(mult as u128)
        .ok_or_else(|| format!("count '{s}' overflows (max is {})", u64::MAX))?;
    if numer % scale != 0 {
        return Err(format!(
            "invalid count '{s}': {num}{last} is not a whole number of candidates"
        ));
    }
    u64::try_from(numer / scale)
        .map_err(|_| format!("count '{s}' overflows (max is {})", u64::MAX))
}

#[derive(clap::Args, Debug, Clone, Default)]
struct GenerateArgs {
    /// Path to a target word list, one entry per line. Without this flag the
    /// tool runs in standard mode using only the supplied or bundled model.
    #[arg(long, help_heading = "Generation")]
    wordlist: Option<PathBuf>,

    /// Model to generate from: a path to a `.ngram` file, or a registered model
    /// name (see `tokenov model`). If omitted, uses the built-in default model.
    #[arg(long, help_heading = "Generation")]
    model: Option<PathBuf>,

    /// Append affixes to the seed (seed stays the prefix, e.g. `cisco2024`).
    ///
    /// This is the DEFAULT for --wordlist. Kept as an explicit flag for clarity;
    /// mutually exclusive with --prepend-only / --float.
    #[arg(long, default_value_t = false, help_heading = "Wordlist targeting")]
    append_only: bool,

    /// Prepend affixes instead (seed becomes the suffix, e.g. `ilovecisco`).
    #[arg(long, default_value_t = false, help_heading = "Wordlist targeting")]
    prepend_only: bool,

    /// Place affixes on EITHER side of the seed (rarity-weighted graft).
    ///
    /// The pre-0.20 default. Mutually exclusive with --append-only / --prepend-only.
    #[arg(long, default_value_t = false, help_heading = "Wordlist targeting")]
    float: bool,

    /// LEGACY targeting mode (weighted|seeded|combined). Rarely needed; the
    /// default append/prepend/float generators supersede it.
    #[arg(long, value_enum, hide = true)]
    mode: Option<Mode>,

    /// Strength of word-list influence (legacy weighted/combined modes only).
    /// Higher tilts harder toward word-list tokens; 1.0 is a no-op. Must be > 0.
    #[arg(long, default_value_t = 2.0, hide = true)]
    bias: f32,

    /// How seeds are derived in the legacy seeded / combined modes.
    ///   entry  — one seed per word-list entry (default; targets entries)
    ///   token  — one seed per unique token from the union of tokenizations
    #[arg(long, value_enum, default_value_t = SeedMode::Entry, hide = true)]
    seed_mode: SeedMode,

    /// [DEPRECATED — no-op] Wordlist expansion was retired. Expand your seed list
    /// with separate tooling and pass the result to --wordlist. Accepted but
    /// ignored (warns) for script compatibility.
    #[arg(long, default_value_t = 0, hide = true)]
    skipgram_expand: usize,

    /// [DEPRECATED — no-op] Companion to --skipgram-expand; retired.
    #[arg(long, value_enum, default_value_t = SkipgramDirection::Both, hide = true)]
    skipgram_direction: SkipgramDirection,

    /// Stop after emitting this many candidates. Omit to run until interrupted.
    ///
    /// Accepts a plain integer or a magnitude suffix (case-insensitive): K = thousand,
    /// M = million, B = billion, T = trillion — so `-c 100k` = 100000, `-c 1.5M` =
    /// 1500000, `-c 1B` = 1000000000. Decimals are allowed when the result is a whole
    /// number of candidates.
    #[arg(short = 'c', long, value_name = "N", value_parser = parse_count, help_heading = "Generation")]
    count: Option<u64>,

    /// Minimum candidate length in bytes (post-decode). 0 disables.
    #[arg(long, default_value_t = 4, value_name = "N", help_heading = "Generation")]
    min_len: usize,

    /// Maximum candidate length in bytes (post-decode). >=256 effectively
    /// disables.
    #[arg(long, default_value_t = 30, value_name = "N", help_heading = "Generation")]
    max_len: usize,

    /// Maximum number of tokens in any heap-explored path. Heap-level cap.
    #[arg(long, default_value_t = DEFAULT_MAX_TOKENS, value_name = "N", help_heading = "Generation")]
    max_tokens: usize,

    /// Minimum number of tokens per candidate. 1 (default) is a no-op.
    ///
    /// Emit only candidates built from >= N tokens — steers toward multi-token /
    /// compound structures and away from single-token trivial candidates. A
    /// drop-at-emit filter (survivors keep their rank order); dropped candidates
    /// don't count toward --count, so generation enumerates deeper to reach it.
    #[arg(long, default_value_t = 1, value_name = "N", help_heading = "Generation")]
    min_tokens: usize,

    /// Start enumeration at probability level N, skipping the more-probable
    /// shells below it. 0 (default) is a no-op.
    ///
    /// Enumeration walks levels 0, 1, 2, … in order, so --count is a ceiling on
    /// depth and this is the matching floor. The stream is an exact suffix of the
    /// full run: same candidates, same order, minus the skipped levels. Long
    /// candidates concentrate at higher levels, but the overlap is wide — this is
    /// a stream-volume knob, not a length filter; use it with --min-len. Run with
    /// -v to see the level a run stopped at.
    #[arg(long, default_value_t = 0, value_name = "N", help_heading = "Generation")]
    min_level: u32,

    /// Write candidates (plain UTF-8, one per line) here instead of stdout.
    ///
    /// Output is uncompressed; pipe through gzip/zstd/7z to compress at rest.
    #[arg(long, help_heading = "Generation")]
    output: Option<PathBuf>,

    /// Parallel enumeration threads. Default: detected CPU count.
    #[arg(long, value_name = "N", help_heading = "Generation")]
    threads: Option<usize>,

    /// Resume the previous run — continues where the last run left off.
    ///
    /// Resumes from the checkpoint state file (`--checkpoint-file` if given, else
    /// the rolling default). Both modes restore the saved position in O(depth) and
    /// continue — no re-enumeration from candidate 0. Fast mode can resume a killed
    /// `tokenov | hashcat` pipe with NO `--output`; strict mode restores its single
    /// DFS position and appends to `--output` (so strict resume needs `--output`).
    /// With `--no-checkpoint` there is no state file, and `--resume` instead re-runs
    /// and skips the first N already-written candidates via the `<output>.progress`
    /// sidecar. Args must match the original run; if nothing is resumable, run
    /// without `--resume`.
    #[arg(long, default_value_t = false, help_heading = "Resume & checkpointing")]
    resume: bool,

    /// Merger chunk size (strict mode only): items batched per channel send.
    ///
    /// Higher = less overhead; lower = tighter ordering; 1 = per-item. If
    /// omitted, uses a `.tune.toml` sidecar if present, else auto-tunes.
    #[arg(long, value_name = "N", help_heading = "Merger tuning")]
    merge_chunk_size: Option<usize>,

    /// Skip inline auto-tune (strict mode, no sidecar); use the default chunk size.
    ///
    /// Uses the default chunk size instead of calibrating first.
    #[arg(long, default_value_t = false, help_heading = "Merger tuning")]
    no_auto_tune: bool,

    /// Force re-calibration even if a `.tune.toml` sidecar exists.
    ///
    /// The new measurements overwrite the `<model>.ngram.tune.toml` sidecar.
    #[arg(long, default_value_t = false, help_heading = "Merger tuning")]
    retune: bool,

    /// Telemetry tick interval (ms) for the merger stats thread. 0 disables.
    ///
    /// Each tick writes one row of (t, total emitted, delta, instantaneous c/s,
    /// avg c/s, total bytes) to the stats CSV (see `--stats-csv`) and one stderr
    /// line. Cost is two `Relaxed` atomic stores per chunk drain on the merger;
    /// effectively zero. 0 disables telemetry entirely.
    #[arg(long, default_value_t = 500, value_name = "MS", help_heading = "Telemetry")]
    stats_interval_ms: u64,

    /// Where to write the per-tick telemetry CSV. Default: `<output>.stats.csv`
    /// when `--output` is set; no file otherwise (stderr only).
    #[arg(long, value_name = "FILE", help_heading = "Telemetry")]
    stats_csv: Option<PathBuf>,

    /// Seconds between stderr `[stats]` progress lines when not `--verbose`.
    ///
    /// The first and final ticks always print. Default: 60.
    #[arg(long, default_value_t = 60, value_name = "SEC", help_heading = "Telemetry")]
    progress_secs: u64,

    /// Emit telemetry as JSONL on stderr instead of the human `[stats]` line.
    ///
    /// One JSON object per line, for consumption by wrapper tools. Fields:
    /// `elapsed_s, emitted, delta, inst_cps, avg_cps, bytes, blocked, stalled`.
    /// Cadence is still set by `--verbose` / `--progress-secs`. (stdout stays
    /// candidates-only.)
    #[arg(short = 'j', long, default_value_t = false, help_heading = "Telemetry")]
    json: bool,

    /// [no-op] Fast mode is the default; accepted for explicitness only.
    ///
    /// Fast mode: each thread writes its partition directly (N-way interleaved,
    /// each partition internally rank-ordered) for maximum throughput. This is
    /// already the default, so the flag changes nothing.
    #[arg(long, default_value_t = false, hide = true)]
    fast: bool,

    /// Strict mode: globally rank-order the candidate stream (best-first).
    ///
    /// Emits one exactly rank-ordered (best-first) stream — the canonical,
    /// byte-reproducible mode. Always single-threaded: the global merge is serial,
    /// so more threads give no speedup, and only single-threaded output is
    /// byte-canonical (--threads > 1 is ignored in strict mode). Slower than fast
    /// mode, but the right choice for slow hashes where you only test your top
    /// candidates and ordering beats raw volume. The candidate SET is identical to
    /// fast mode; only the order differs.
    #[arg(long, default_value_t = false, help_heading = "Generation")]
    strict: bool,

    /// Checkpoint the DFS position(s) to a SPECIFIC file.
    ///
    /// Generation checkpoints by DEFAULT to a rolling state file, so every run is
    /// resumable with `--resume` and no forethought. Pass this to point the
    /// checkpoint at a named file instead — keep several long tasks going and return
    /// to a specific one by resuming its file. Resume with `--resume` (+ the same
    /// `--checkpoint-file`) or `--resume-state <file>`; the saved position is
    /// reconstructed in O(depth) and continues, instead of re-enumerating from
    /// candidate 0. (Fast mode saves one position per worker; strict mode, being
    /// single-threaded, saves one.)
    #[arg(long, value_name = "FILE", help_heading = "Resume & checkpointing")]
    checkpoint_file: Option<PathBuf>,

    /// Disable the default checkpoint state file.
    ///
    /// Generation writes a rolling checkpoint by default so any interrupted run is
    /// resumable in O(depth). Pass this to skip it entirely (no state file written);
    /// `--resume` then falls back to re-running and skipping the already-written
    /// candidates via the progress sidecar.
    #[arg(long, default_value_t = false, help_heading = "Resume & checkpointing")]
    no_checkpoint: bool,

    /// Checkpoint cadence in seconds. Default 300.
    ///
    /// The cadence is also the resume safety margin (the last checkpoint lags the
    /// crash, so resume re-tests an overlap region rather than skipping it).
    #[arg(long, default_value_t = 300, value_name = "SEC", help_heading = "Resume & checkpointing")]
    checkpoint_secs: u64,

    /// Resume a killed run from a SPECIFIC checkpoint file.
    ///
    /// Explicit form of `--resume` for a named checkpoint (the resumed run keeps
    /// checkpointing back to the same file, so a second crash is still resumable —
    /// no separate `--checkpoint-file` needed). The model, args, `--threads`, and
    /// tokenov version must match the checkpointed run (enforced via the checkpoint
    /// fingerprint). Keep the SAME hashcat `--potfile-path` across restarts so the
    /// re-tested overlap is deduped cheaply.
    #[arg(long, value_name = "FILE", help_heading = "Resume & checkpointing")]
    resume_state: Option<PathBuf>,

    /// Algorithm variant (experimental). Default `baseline`.
    ///
    ///   baseline   — trigram + Kneser-Ney bigram backoff (the default)
    ///   freq-tail  — baseline plus a raw-frequency unigram backoff tail
    ///   cap-tail   — baseline plus a capital-biased unigram backoff tail
    #[arg(long, default_value = "baseline", hide = true)]
    variant: String,

    /// Per-token case shaping — modify each token's case AFTER it's selected.
    ///
    /// Use with a case-folded (lowercase-trained) content model. Unlike a hashcat
    /// mask (which constrains which characters are *selected*), this does not
    /// influence token selection — it re-cases the decoded bytes of each already-
    /// chosen token. The token is the unit of casing.
    ///
    /// Syntax is hashcat-like: `?` followed by a case op per token slot —
    /// `?l`=lowercase, `?c`=capitalize its first letter, `?u`=uppercase. The last
    /// op repeats for any further tokens. Separate multiple patterns with `;`; each
    /// terminal candidate is emitted once per pattern. Named shortcuts: `lower`,
    /// `cap1` (cap first token only), `title` (cap every token), `upper`.
    ///
    ///   --case-shape cap1            -> Spring19
    ///   --case-shape "lower;cap1"    -> springfield AND Springfield
    ///   --case-shape "?c?l?u"        -> cap tok0, lower tok1, upper tok2+
    ///
    /// Omit for the default (single lowercase emission; byte-identical to prior
    /// behavior).
    #[arg(long = "case-shape", value_name = "SPEC", help_heading = "Case shaping")]
    case_shape: Option<String>,

    /// Enterprise-policy compliance mode: emit only policy-compliant candidates,
    /// applying the minimal repair (capitalize-first) to each one.
    ///
    /// Policy = byte-length >= 8 AND at least 3 of {lowercase, uppercase, digit,
    /// special}. Per candidate:
    ///   1. already compliant             -> emit as-is
    ///   2. first char [a-z], no capital  -> capitalize it (+0 length), re-check;
    ///                                        emit if now compliant
    ///   3. otherwise                     -> drop
    ///
    /// One guess per candidate (1-in-1-out), unlike hashcat rule files (~66x).
    /// Capitalizing is the only transform — it closes the single dominant gap
    /// (most non-compliant candidates lack *only* an uppercase letter), at zero
    /// length cost. Drops roughly two-thirds of the default stream, so
    /// generation keeps enumerating deeper to reach `--count` compliant guesses.
    ///
    /// For POLICY targets only (a set where min-length + complexity are
    /// enforced). Do NOT use on consumer corpora — forcing compliance wrecks
    /// lowercase-dominated sets. Mutually exclusive with --case-shape.
    #[arg(long, help_heading = "Case shaping",
          conflicts_with_all = ["case_shape"])]
    enterprise: bool,

    // ── Memory safety ─────────────────────────────────────────────────────────

    /// Abort if process RSS exceeds this many GiB. Default: 75% of MemTotal.
    ///
    /// The default is stable and hardware-derived — not MemAvailable, which
    /// fluctuates with cache. Also read from the TOKENOV_MAX_RSS_GB env var (flag
    /// takes precedence). Set to 0 to disable the RSS trigger entirely.
    #[arg(long, value_name = "GiB", help_heading = "Memory safety")]
    max_rss_gb: Option<f64>,

    /// Abort when system MemAvailable / MemTotal drops below this fraction.
    ///
    /// Default: 0.10 (10%). Set to 0 to disable.
    #[arg(long, default_value_t = 0.10, value_name = "FRAC", help_heading = "Memory safety")]
    mem_pressure_threshold: f64,

    /// Memory monitor sample interval (ms). Default: 3000.
    #[arg(long, default_value_t = 3000, value_name = "MS", help_heading = "Memory safety")]
    mem_sample_ms: u64,

    /// FORCE the on-demand (lazy) child path, overriding automatic selection.
    ///
    /// By DEFAULT tokenov chooses automatically (see the child_cache note below):
    /// it builds the resident child_cache when its projected size comfortably
    /// fits available RAM, and falls back to computing Kneser-Ney bigram-backoff
    /// children on demand when it wouldn't — so you never have to pass a flag to
    /// avoid an OOM. This flag pins the lazy path unconditionally (e.g. to
    /// minimize RSS regardless of headroom). Output is **byte-identical** either
    /// way — the child_cache is purely a performance cache. Mutually exclusive
    /// with `--force-child-cache`.
    #[arg(long = "lazy", default_value_t = false,
          conflicts_with = "force_child_cache", help_heading = "Memory safety")]
    lazy_children: bool,

    /// FORCE building the resident child_cache, overriding automatic selection.
    ///
    /// Pins the fast (cached) path even when automatic selection would fall back
    /// to lazy on RAM grounds. Use only when you know the cache fits and want max
    /// throughput; on a model whose cache exceeds RAM this will OOM/self-abort.
    /// Mutually exclusive with `--lazy`.
    #[arg(long, default_value_t = false, help_heading = "Memory safety")]
    force_child_cache: bool,

    /// FORCE the BOUNDED per-thread cache with this cap (entries/thread), overriding auto.
    ///
    /// Pins a partial child cache of the N most-recently-used contexts per worker
    /// thread, recomputing on a miss. Bounds RAM (~2·N entries/thread
    /// resident) while recovering most of the recompute cost via the DFS's
    /// locality. Output is byte-identical to FULL/LAZY. Mainly a benchmarking /
    /// power knob; normally the automatic selector sizes this from the RAM budget.
    /// Mutually exclusive with `--lazy` / `--force-child-cache`.
    #[arg(long, value_name = "N", conflicts_with_all = ["lazy_children", "force_child_cache"],
          help_heading = "Memory safety")]
    bounded_cap: Option<usize>,

    /// Enable the background runtime chunk-size auto-tuner (experimental).
    ///
    /// Strict mode, unpinned chunk only. Off by default and rarely useful — the
    /// default already captures the tuning win. Kept for experimentation.
    #[arg(long, default_value_t = false, hide = true)]
    runtime_tune: bool,

    /// Deprecated no-op: the runtime auto-tuner is now off by default, so this
    /// has no effect. Accepted for backward compatibility; use `--runtime-tune`.
    #[arg(long, default_value_t = false, hide = true)]
    no_runtime_tune: bool,
}

#[derive(clap::Args, Debug, Clone)]
struct CalibrateArgs {
    /// Model file to calibrate (`.ngram` from `tokenov model train`).
    #[arg(long)]
    model: PathBuf,

    /// K values to test, comma-separated.
    #[arg(long, value_delimiter = ',', default_values_t = vec![1024usize, 2048, 4096, 8192, 16384])]
    chunk_sizes: Vec<usize>,

    /// Seconds to measure each K (after a settle period).
    #[arg(long, default_value_t = 30)]
    measure_secs: u64,

    /// Seconds to wait after switching K before sampling — lets the new K
    /// reach steady state in the channels.
    #[arg(long, default_value_t = 5)]
    settle_secs: u64,

    /// Number of producer threads (default: rayon's detected count).
    #[arg(long)]
    threads: Option<usize>,

    /// Reject K values whose peak RSS exceeds this many MB. The recommended
    /// K is the highest-throughput value within the budget. Note: RSS
    /// includes the model + child_cache baseline (often several GB), so
    /// this is a process-total ceiling, not a per-K delta. If all K values
    /// exceed the budget, the highest-throughput value is picked anyway.
    #[arg(long, default_value_t = 16384)]
    max_memory_mb: usize,

    /// Path to write the sidecar (default: `<model>.tune.toml` next to the model).
    #[arg(long)]
    output: Option<PathBuf>,

    /// Overwrite an existing sidecar (default: refuses to clobber).
    #[arg(long, default_value_t = false)]
    force: bool,

    /// Algorithm variant. Same flag as `tokenov generate --variant`. Default
    /// `baseline`. Calibrate with the variant you intend to generate with —
    /// different variants have different child-cache sizes.
    #[arg(long, default_value = "baseline")]
    variant: String,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
enum Mode {
    #[default]
    Weighted,
    Seeded,
    Combined,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
enum SkipgramDirection {
    Forward,
    Inverse,
    #[default]
    Both,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
enum SeedMode {
    #[default]
    Entry,
    Token,
}

// ============================================================================
// Logging helper (stderr; stdout is reserved for candidates)
// ============================================================================

/// Global verbosity. Off by default: informational `log_msg` output is suppressed
/// so a normal run is quiet (stdout still carries candidates). `-v/--verbose` turns
/// it on. Genuine warnings use `warn_msg` and print regardless.
static VERBOSE: AtomicBool = AtomicBool::new(false);

pub fn set_verbose(on: bool) {
    VERBOSE.store(on, AtomicOrdering::Relaxed);
}

pub fn verbose() -> bool {
    VERBOSE.load(AtomicOrdering::Relaxed)
}

/// Informational stderr (model-load / tuning / progress). Silent unless `--verbose`.
pub fn log_msg(msg: &str) {
    if !VERBOSE.load(AtomicOrdering::Relaxed) { return; }
    let now = chrono_now();
    eprintln!("{} {}", now, msg);
}

/// A genuine warning — always printed on stderr, regardless of `--verbose`.
pub fn warn_msg(msg: &str) {
    eprintln!("{}", msg);
}

fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("[{:02}:{:02}:{:02}]", h, m, s)
}

// ============================================================================
// Model: in-memory and on-disk
// ============================================================================

/// In-memory representation of a fitted model.
///
/// Two flavors share the struct, distinguished by `is_kn`:
///   - V1 (legacy, no smoothing): only `contexts` + `decode` populated. Trigram
///     children's cum sums to 1.0 per context; no back-off.
///   - V2 (Kneser-Ney): `contexts` stores KN-discounted trigram cum-probs that
///     sum to `1 - lambda` per context (the missing mass goes to bigram via
///     back-off). `lambda`, `bigram_kn`, `unigram_raw`, and `unigram_kn_cont`
///     are populated.
pub struct Model {
    /// Per-trigram-context (children_ids, cum_probs).
    /// V1: cum sums to 1.0 per context.
    /// V2: cum sums to (1 - lambda(a,b)) per context.
    pub contexts: FxHashMap<Ctx, (Vec<u32>, Vec<f32>)>,
    /// Per-token-id decoded byte string. Length = vocab_size.
    pub decode: Vec<Vec<u8>>,
    pub start_id: u32,
    pub end_id: u32,

    // V2-only fields. Empty / default for V1.
    pub is_kn: bool,
    pub discount: f32,
    /// Per-trigram-context Kneser-Ney back-off weight. λ(a,b) = D * |distinct
    /// trigram children of (a,b)| / c2(a,b). Mass given to bigram back-off.
    pub lambda: FxHashMap<Ctx, f32>,
    /// Per-bigram-context KN-continuation distribution.
    /// (children_ids, cum_probs) where cum sums to 1.0 per b. Each child's
    /// per-step probability is log(P_cont(t | b)).
    pub bigram_kn: FxHashMap<u32, (Vec<u32>, Vec<f32>)>,
    /// Per-token raw count (from training). Length = vocab_size + 2 (sentinels
    /// included). Useful for KN-vs-raw comparison and as the source for
    /// Variant B's unigram-tier backoff.
    pub unigram_raw: Vec<u64>,
    /// Per-token KN-continuation probability. Length = vocab_size + 2.
    pub unigram_kn_cont: Vec<f32>,
}

/// Model provenance embedded in the v3 file header. The load-bearing
/// field is `tokenizer_json` — the exact bytes of the tokenizer that built the
/// model, so `generate --wordlist` can re-tokenize with the correct tokenizer
/// from the `.ngram` file alone. The rest is informational build metadata for
/// `--list-models`/`model info` and reproducibility audits.
struct Provenance {
    /// Raw `tokenizer.json` bytes (loadable via `Tokenizer::from_bytes`).
    tokenizer_json: Vec<u8>,
    /// The `--tokenizer` path as given at build (informational).
    tok_source: String,
    /// The `--train` corpus path as given at build (informational).
    train_path: String,
    /// Build time, seconds since the Unix epoch.
    build_epoch: u64,
    /// tokenov binary version that built the model (`CARGO_PKG_VERSION`).
    binver: String,
}

/// Provenance-blob schema version (inside the v3 header, distinct from the file
/// magic). Bump when adding fields; readers skip trailing unknown bytes via the
/// blob's outer length prefix, so older readers keep working.
const PROVENANCE_SCHEMA: u32 = 1;

impl Provenance {
    /// Serialize the blob body (NOT including the outer u32 length prefix the
    /// file writes before it). All little-endian; strings are u32-len + utf8.
    fn serialize(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(self.tokenizer_json.len() + 256);
        let put_u32 = |b: &mut Vec<u8>, v: u32| b.extend_from_slice(&v.to_le_bytes());
        let put_u64 = |b: &mut Vec<u8>, v: u64| b.extend_from_slice(&v.to_le_bytes());
        let put_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u32).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        put_u32(&mut b, PROVENANCE_SCHEMA);
        put_u32(&mut b, self.tokenizer_json.len() as u32);
        b.extend_from_slice(&self.tokenizer_json);
        put_str(&mut b, &self.tok_source);
        put_str(&mut b, &self.train_path);
        put_u64(&mut b, self.build_epoch);
        put_str(&mut b, &self.binver);
        b
    }

    /// Parse a blob body (the bytes between the file's length prefix and the v2
    /// model body). Tolerant of trailing unknown fields (future schema).
    fn parse(blob: &[u8]) -> Result<Provenance> {
        let mut p = 0usize;
        let rd_u32 = |p: &mut usize, b: &[u8]| -> Result<u32> {
            if *p + 4 > b.len() { bail!("provenance blob truncated"); }
            let v = u32::from_le_bytes([b[*p], b[*p+1], b[*p+2], b[*p+3]]); *p += 4; Ok(v)
        };
        let rd_u64 = |p: &mut usize, b: &[u8]| -> Result<u64> {
            if *p + 8 > b.len() { bail!("provenance blob truncated"); }
            let v = u64::from_le_bytes([b[*p],b[*p+1],b[*p+2],b[*p+3],b[*p+4],b[*p+5],b[*p+6],b[*p+7]]); *p += 8; Ok(v)
        };
        let rd_bytes = |p: &mut usize, b: &[u8]| -> Result<Vec<u8>> {
            let n = {
                if *p + 4 > b.len() { bail!("provenance blob truncated"); }
                let v = u32::from_le_bytes([b[*p], b[*p+1], b[*p+2], b[*p+3]]) as usize; *p += 4; v
            };
            if *p + n > b.len() { bail!("provenance blob truncated"); }
            let v = b[*p..*p+n].to_vec(); *p += n; Ok(v)
        };
        let rd_str = |p: &mut usize, b: &[u8]| -> Result<String> {
            Ok(String::from_utf8_lossy(&rd_bytes(p, b)?).into_owned())
        };
        let _schema = rd_u32(&mut p, blob)?; // reserved for future field-skip logic
        let tokenizer_json = rd_bytes(&mut p, blob)?;
        let tok_source = rd_str(&mut p, blob)?;
        let train_path = rd_str(&mut p, blob)?;
        let build_epoch = rd_u64(&mut p, blob)?;
        let binver = rd_str(&mut p, blob)?;
        Ok(Provenance { tokenizer_json, tok_source, train_path, build_epoch, binver })
    }
}

/// Read ONLY a v3 model's provenance header without loading the full model body
/// (cheap — reads the file's first bytes, not the multi-hundred-MB body). Returns
/// `Ok(None)` for v1/v2 (no provenance) or a truncated/foreign file.
fn model_read_provenance(path: &Path) -> Result<Option<Provenance>> {
    use std::io::Read;
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut head = [0u8; 12];
    if f.read_exact(&mut head).is_err() { return Ok(None); }
    if &head[..8] != MODEL_MAGIC_V3 { return Ok(None); }
    let blob_len = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
    let mut blob = vec![0u8; blob_len];
    f.read_exact(&mut blob).with_context(|| "read v3 provenance blob")?;
    Ok(Some(Provenance::parse(&blob)?))
}

fn model_save(path: &Path, model: &Model, provenance: Option<&Provenance>, force: bool) -> Result<()> {
    if path.exists() && !force {
        bail!("output already exists: {} (pass --force to overwrite)", path.display());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    // tokenov owns the temp file (unique per-process so concurrent builds of the
    // same model don't clobber each other) and renames it into place only on
    // success. On any error the temp is removed, so a failed or killed build
    // never leaves a partial .ngram (or stray .tmp) behind for the caller to
    // reason about — the build is all-or-nothing from the caller's view.
    let tmp = path.with_extension(format!("ngram.tmp.{}", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let res = model_save_write(&tmp, path, model, provenance);
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

fn model_save_write(tmp: &Path, path: &Path, model: &Model, provenance: Option<&Provenance>) -> Result<()> {
    let f = File::create(tmp).with_context(|| format!("create {}", tmp.display()))?;
    let mut w = BufWriter::with_capacity(8 << 20, f);
    let t0 = Instant::now();

    // v3 = provenance blob + v2/KN body. v3 implies KN (the only body a v3-writing
    // build produces); guard so a non-KN model never claims v3.
    let magic = match (provenance, model.is_kn) {
        (Some(_), true)  => MODEL_MAGIC_V3,
        (Some(_), false) => bail!("v3 provenance requires a KN model"),
        (None, true)     => MODEL_MAGIC_V2,
        (None, false)    => MODEL_MAGIC_V1,
    };
    w.write_all(magic)?;
    if let Some(prov) = provenance {
        // u32 length prefix, then the blob; the v2 body follows verbatim.
        let blob = prov.serialize();
        w.write_all(&(blob.len() as u32).to_le_bytes())?;
        w.write_all(&blob)?;
    }
    w.write_all(&model.start_id.to_le_bytes())?;
    w.write_all(&model.end_id.to_le_bytes())?;
    let vocab_size = model.decode.len() as u32;
    w.write_all(&vocab_size.to_le_bytes())?;

    if model.is_kn {
        // V2 header includes the discount D after vocab_size.
        w.write_all(&model.discount.to_le_bytes())?;
    }

    // Decode table (v1 and v2 identical)
    for entry in &model.decode {
        let n = entry.len();
        if n > u16::MAX as usize {
            bail!("decode entry length {} exceeds u16::MAX", n);
        }
        w.write_all(&(n as u16).to_le_bytes())?;
        w.write_all(entry)?;
    }

    if model.is_kn {
        // V2: unigram raw counts and KN-continuation probs (vocab_size + 2 each
        // — sentinel ids start_id and end_id may have entries for their counts
        // although in practice we don't use them; we serialize vocab_size only).
        // We store exactly vocab_size entries of each, indexed 0..vocab_size.
        if model.unigram_raw.len() < vocab_size as usize ||
           model.unigram_kn_cont.len() < vocab_size as usize {
            bail!("v2 model: unigram arrays must have at least vocab_size entries");
        }
        for i in 0..(vocab_size as usize) {
            w.write_all(&model.unigram_raw[i].to_le_bytes())?;
        }
        for i in 0..(vocab_size as usize) {
            w.write_all(&model.unigram_kn_cont[i].to_le_bytes())?;
        }

        // V2: bigram KN-continuation. (b: u32, n: u32, ids: [u32; n], cum: [f32; n])
        let n_bi = model.bigram_kn.len() as u32;
        w.write_all(&n_bi.to_le_bytes())?;
        let mut bkeys: Vec<u32> = model.bigram_kn.keys().copied().collect();
        bkeys.sort_unstable();
        for b in bkeys {
            let (ids, cum) = &model.bigram_kn[&b];
            w.write_all(&b.to_le_bytes())?;
            w.write_all(&(ids.len() as u32).to_le_bytes())?;
            for &id in ids { w.write_all(&id.to_le_bytes())?; }
            for &c in cum  { w.write_all(&c.to_le_bytes())?; }
        }
    }

    // Trigram contexts (v1 and v2)
    let n_ctx = model.contexts.len() as u32;
    w.write_all(&n_ctx.to_le_bytes())?;
    let mut keys: Vec<Ctx> = model.contexts.keys().copied().collect();
    keys.sort_unstable();
    for ctx in keys {
        let (ids, cum) = &model.contexts[&ctx];
        w.write_all(&ctx.to_le_bytes())?;
        if model.is_kn {
            // V2 also writes lambda(a,b) per trigram context
            let lam = model.lambda.get(&ctx).copied().unwrap_or(0.0);
            w.write_all(&lam.to_le_bytes())?;
        }
        w.write_all(&(ids.len() as u32).to_le_bytes())?;
        for &id in ids { w.write_all(&id.to_le_bytes())?; }
        for &c in cum  { w.write_all(&c.to_le_bytes())?; }
    }
    w.flush()?;
    drop(w);
    std::fs::rename(tmp, path).context("atomic rename .ngram into place")?;
    let sz = std::fs::metadata(path)?.len();
    let fmt = match (provenance, model.is_kn) {
        (Some(_), _) => "v3/KN+prov",
        (None, true) => "v2/KN",
        (None, false) => "v1",
    };
    log_msg(&format!("[save] {} ({} MB, {}) in {:.1}s",
        path.display(), sz / 1_000_000, fmt,
        t0.elapsed().as_secs_f64()));
    Ok(())
}

fn model_load(path: &Path) -> Result<Model> {
    let t0 = Instant::now();
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() < 24 {
        bail!("{}: file too small", path.display());
    }
    let magic = &bytes[..8];
    // v3 carries a length-prefixed provenance blob between the magic and the v2
    // body; is_kn is true and we advance `p` past the blob so the body parser
    // below is shared byte-for-byte with v2.
    let (is_kn, body_start) = match magic {
        m if m == MODEL_MAGIC_V1 => (false, 8usize),
        m if m == MODEL_MAGIC_V2 => (true, 8usize),
        m if m == MODEL_MAGIC_V3 => {
            if bytes.len() < 12 { bail!("{}: v3 file too small", path.display()); }
            let blob_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            let body = 12usize.checked_add(blob_len)
                .filter(|&b| b <= bytes.len())
                .ok_or_else(|| anyhow!("{}: v3 provenance blob length overruns file", path.display()))?;
            (true, body)
        }
        _ => bail!("{}: bad or unrecognized magic header (expected NGRMv001, NGRMv002, or NGRMv003)", path.display()),
    };

    let mut p = body_start;
    let read_u16 = |p: &mut usize, b: &[u8]| -> Result<u16> {
        if *p + 2 > b.len() { bail!("truncated"); }
        let v = u16::from_le_bytes([b[*p], b[*p+1]]); *p += 2; Ok(v)
    };
    let read_u32 = |p: &mut usize, b: &[u8]| -> Result<u32> {
        if *p + 4 > b.len() { bail!("truncated"); }
        let v = u32::from_le_bytes([b[*p], b[*p+1], b[*p+2], b[*p+3]]); *p += 4; Ok(v)
    };
    let read_u64 = |p: &mut usize, b: &[u8]| -> Result<u64> {
        if *p + 8 > b.len() { bail!("truncated"); }
        let v = u64::from_le_bytes([b[*p], b[*p+1], b[*p+2], b[*p+3], b[*p+4], b[*p+5], b[*p+6], b[*p+7]]); *p += 8; Ok(v)
    };
    let read_f32 = |p: &mut usize, b: &[u8]| -> Result<f32> {
        if *p + 4 > b.len() { bail!("truncated"); }
        let v = f32::from_le_bytes([b[*p], b[*p+1], b[*p+2], b[*p+3]]); *p += 4; Ok(v)
    };

    let start_id   = read_u32(&mut p, &bytes)?;
    let end_id     = read_u32(&mut p, &bytes)?;
    let vocab_size = read_u32(&mut p, &bytes)?;
    let discount = if is_kn { read_f32(&mut p, &bytes)? } else { 0.0 };

    let mut decode: Vec<Vec<u8>> = Vec::with_capacity(vocab_size as usize);
    for _ in 0..vocab_size {
        let n = read_u16(&mut p, &bytes)? as usize;
        if p + n > bytes.len() { bail!("decode-table truncated"); }
        decode.push(bytes[p..p + n].to_vec());
        p += n;
    }

    let mut unigram_raw: Vec<u64> = Vec::new();
    let mut unigram_kn_cont: Vec<f32> = Vec::new();
    let mut bigram_kn: FxHashMap<u32, (Vec<u32>, Vec<f32>)> = FxHashMap::default();
    let mut lambda: FxHashMap<Ctx, f32> = FxHashMap::default();

    if is_kn {
        // Unigram raw counts
        unigram_raw.reserve(vocab_size as usize);
        for _ in 0..vocab_size { unigram_raw.push(read_u64(&mut p, &bytes)?); }
        // Unigram KN-continuation
        unigram_kn_cont.reserve(vocab_size as usize);
        for _ in 0..vocab_size { unigram_kn_cont.push(read_f32(&mut p, &bytes)?); }
        // Bigram KN-continuation
        let n_bi = read_u32(&mut p, &bytes)? as usize;
        bigram_kn.reserve(n_bi);
        for _ in 0..n_bi {
            let b = read_u32(&mut p, &bytes)?;
            let n = read_u32(&mut p, &bytes)? as usize;
            let mut ids: Vec<u32> = Vec::with_capacity(n);
            for _ in 0..n { ids.push(read_u32(&mut p, &bytes)?); }
            let mut cum: Vec<f32> = Vec::with_capacity(n);
            for _ in 0..n { cum.push(read_f32(&mut p, &bytes)?); }
            bigram_kn.insert(b, (ids, cum));
        }
    }

    // Trigram contexts
    let n_ctx = read_u32(&mut p, &bytes)? as usize;
    let mut contexts: FxHashMap<Ctx, (Vec<u32>, Vec<f32>)> = FxHashMap::default();
    contexts.reserve(n_ctx);
    if is_kn { lambda.reserve(n_ctx); }
    for _ in 0..n_ctx {
        let ctx = read_u64(&mut p, &bytes)?;
        if is_kn {
            let lam = read_f32(&mut p, &bytes)?;
            lambda.insert(ctx, lam);
        }
        let n = read_u32(&mut p, &bytes)? as usize;
        let mut ids: Vec<u32> = Vec::with_capacity(n);
        for _ in 0..n { ids.push(read_u32(&mut p, &bytes)?); }
        let mut cum: Vec<f32> = Vec::with_capacity(n);
        for _ in 0..n { cum.push(read_f32(&mut p, &bytes)?); }
        contexts.insert(ctx, (ids, cum));
    }
    if p != bytes.len() {
        warn_msg(&format!("warning: {} trailing bytes after model parse", bytes.len() - p));
    }
    let fmt = if magic == MODEL_MAGIC_V3 { "v3/KN+prov" }
              else if is_kn { "v2/KN" } else { "v1" };
    log_msg(&format!("[load] {} ({} ctx, {} MB, {}) in {:.1}s",
        path.display(), n_ctx, bytes.len() / 1_000_000, fmt,
        t0.elapsed().as_secs_f64()));
    Ok(Model {
        contexts, decode, start_id, end_id,
        is_kn, discount, lambda, bigram_kn, unigram_raw, unigram_kn_cont,
    })
}

// ============================================================================
// Decode-table construction
// ============================================================================

/// Method-specific cleanup of `tokenizer.decode([id])` output.
///   - Plain: keep as-is.
///   - StripBpeSpace: strip ASCII space and UTF-8 'Ġ' (0xC4 0xA0); for
///     byte-level BPE tokenizers (like Cracken's rockyou_bpe).
#[derive(Copy, Clone)]
enum DecodeKind { Plain, StripBpeSpace }

fn classify_tokenizer(tokenizer: &Tokenizer) -> DecodeKind {
    // Heuristic: peek at a few likely byte-level tokens. If decoding token id
    // 0 produces something with 'Ġ' or a leading space, assume byte-level BPE
    // and strip them. Otherwise Plain (LLaMA-style SentencePiece, where decode
    // already substitutes ▁ → space).
    for id in 0..tokenizer.get_vocab_size(false).min(256) as u32 {
        if let Ok(s) = tokenizer.decode(&[id], true) {
            let bytes = s.as_bytes();
            if bytes.windows(2).any(|w| w == [0xC4, 0xA0]) {
                return DecodeKind::StripBpeSpace;
            }
        }
    }
    DecodeKind::Plain
}

fn precompute_decode_table(tokenizer: &Tokenizer, vocab_size: u32, kind: DecodeKind) -> Result<Vec<Vec<u8>>> {
    let mut table: Vec<Vec<u8>> = Vec::with_capacity(vocab_size as usize);
    for id in 0..vocab_size {
        let s = tokenizer.decode(&[id], true).unwrap_or_default();
        let bytes = match kind {
            DecodeKind::Plain => s.into_bytes(),
            DecodeKind::StripBpeSpace => {
                let raw = s.as_bytes();
                let mut out = Vec::with_capacity(raw.len());
                let mut i = 0;
                while i < raw.len() {
                    if raw[i] == b' ' { i += 1; continue; }
                    if i + 1 < raw.len() && raw[i] == 0xC4 && raw[i + 1] == 0xA0 { i += 2; continue; }
                    out.push(raw[i]);
                    i += 1;
                }
                out
            }
        };
        table.push(bytes);
    }
    Ok(table)
}

#[inline(always)]
fn finalize_decoded(buf: &mut Vec<u8>, kind: DecodeKind) {
    match kind {
        DecodeKind::Plain => {
            // Strip leading ASCII spaces (LLaMA-style decode artifact).
            let mut k = 0;
            while k < buf.len() && buf[k] == b' ' { k += 1; }
            if k > 0 { buf.drain(..k); }
        }
        DecodeKind::StripBpeSpace => {
            // Per-token strip already removed inner spaces; trim ends to be safe.
            let mut k = 0;
            while k < buf.len() && (buf[k] as char).is_ascii_whitespace() { k += 1; }
            if k > 0 { buf.drain(..k); }
            while buf.last().map_or(false, |b| (*b as char).is_ascii_whitespace()) {
                buf.pop();
            }
        }
    }
}

/// Non-mutating equivalent of `finalize_decoded`: returns the sub-range of `buf`
/// that `finalize_decoded` would leave after trimming, so a caller holding an
/// incrementally-maintained byte buffer can emit `&buf[range]` with zero copy.
/// MUST stay byte-for-byte equivalent to `finalize_decoded` above.
#[inline]
fn finalized_range(buf: &[u8], kind: DecodeKind) -> std::ops::Range<usize> {
    match kind {
        DecodeKind::Plain => {
            let mut start = 0;
            while start < buf.len() && buf[start] == b' ' { start += 1; }
            start..buf.len()
        }
        DecodeKind::StripBpeSpace => {
            let mut start = 0;
            while start < buf.len() && (buf[start] as char).is_ascii_whitespace() { start += 1; }
            let mut end = buf.len();
            while end > start && (buf[end - 1] as char).is_ascii_whitespace() { end -= 1; }
            start..end
        }
    }
}

// ============================================================================
// `tokenov build` — fit a model from a tokenizer + password list
// ============================================================================

fn run_build(args: BuildArgs) -> Result<()> {
    if args.order != 3 {
        // We hard-code 3-gram for v1; --order is exposed for future-compat
        // but only 3 is supported. Document & fail clearly.
        bail!("only --ngram 3 (trigram) is supported in v1; got {}", args.order);
    }
    let name = args.name.clone().unwrap_or_else(|| args.tokenizer.file_stem()
        .and_then(|s| s.to_str()).unwrap_or("model").to_string());

    // Resolve the output path: explicit --output, else default model store
    // (tokenov-owned) at <store>/<name>.ngram.
    let out_path = args.output.clone()
        .unwrap_or_else(|| registry::models_dir().join(format!("{name}.ngram")));
    if out_path.exists() && !args.force {
        bail!("output already exists: {} (pass --force to overwrite)", out_path.display());
    }

    log_msg(&format!("[build] name={} tokenizer={} train={} output={}",
        name, args.tokenizer.display(), args.train.display(), out_path.display()));
    let t_overall = Instant::now();

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow!("load tokenizer {}: {}", args.tokenizer.display(), e))?;
    let full_vocab_size = tokenizer.get_vocab_size(false) as u32;
    let vocab_size = args.max_vocab.map(|v| v.min(full_vocab_size)).unwrap_or(full_vocab_size);
    if args.max_vocab.is_some() {
        log_msg(&format!("[build] using max_vocab={} (tokenizer has {})", vocab_size, full_vocab_size));
    }
    if vocab_size > 50_000 {
        log_msg(&format!("[build] note: vocab_size={} is large; if --train has fewer than ~10M lines, the model will be very sparse", vocab_size));
    }

    let start_id = vocab_size;
    let end_id = vocab_size + 1;
    log_msg(&format!("[build] vocab_size={} start_id={} end_id={}", vocab_size, start_id, end_id));

    // Stream the training file in chunks. Each chunk is tokenized and folded
    // into the trigram count map, then dropped — the only structure that
    // grows with N is `counts`. This replaces the older "read whole file,
    // collect Vec<Vec<u32>>, then count" path, which held the entire
    // tokenized corpus in memory and OOM'd at ~30 M lines on a 31 GB box.
    log_msg("[build] streaming tokenize + count...");
    let t0 = Instant::now();
    let mut counts: FxHashMap<Ctx, FxHashMap<u32, u64>> = FxHashMap::default();
    counts.reserve(8_000_000);
    let mut total_tokens: u64 = 0;
    let mut total_lines: u64 = 0;
    // Recovery of non-UTF-8 passwords (e.g. legacy-codepage entries in a leaked
    // dump). Recovered to UTF-8 where a plausible encoding exists, skipped only
    // when none does. Valid-UTF-8 corpora recover/skip nothing.
    let mut rec_stats = recover::RecoveryStats::default();

    // Batch size: large enough to amortize encode_batch's Rayon overhead,
    // small enough to keep per-batch memory bounded. 100K lines × avg ~10 B
    // = ~1 MB raw text per batch; Encoding objects are a few hundred MB peak
    // (then dropped). Profiled empirically to be near-optimal on the 12-core
    // workload; smaller batches lose throughput, larger ones gain nothing.
    const BATCH: usize = 100_000;

    let train_file = std::fs::File::open(&args.train)
        .with_context(|| format!("read --train {}", args.train.display()))?;
    let mut reader = std::io::BufReader::with_capacity(8 << 20, train_file);
    let mut line_bytes: Vec<u8> = Vec::with_capacity(256);
    let mut batch_strings: Vec<String> = Vec::with_capacity(BATCH);

    // Helper: fold a single tokenized sequence (ids) into the counts map.
    // start_id / end_id flank the sequence so contexts at the boundary
    // also get counted. This matches the prior implementation exactly.
    let fold_seq = |counts: &mut FxHashMap<Ctx, FxHashMap<u32, u64>>,
                    total_tokens: &mut u64,
                    ids: &[u32]| {
        if ids.is_empty() { return; }
        *total_tokens += ids.len() as u64;
        let mut a = start_id;
        let mut b = start_id;
        for &id in ids.iter().chain(std::iter::once(&end_id)) {
            *counts.entry(pack(a, b)).or_default().entry(id).or_insert(0) += 1;
            a = b;
            b = id;
        }
    };

    loop {
        // Drain a batch of up to BATCH non-empty lines.
        batch_strings.clear();
        while batch_strings.len() < BATCH {
            line_bytes.clear();
            let n = std::io::BufRead::read_until(&mut reader, b'\n', &mut line_bytes)?;
            if n == 0 { break; } // EOF
            while matches!(line_bytes.last(), Some(b'\n') | Some(b'\r')) { line_bytes.pop(); }
            if line_bytes.is_empty() { continue; }
            // Tokenizers need &str. A non-UTF-8 password (e.g. a legacy-codepage
            // entry in a leaked dump) is recovered to UTF-8 where possible, and
            // only skipped if no candidate encoding yields plausible text.
            let d = recover::decode_line(&line_bytes);
            rec_stats.note(&d);
            if let Some(s) = d.into_text() { batch_strings.push(s); }
        }
        if batch_strings.is_empty() { break; }
        total_lines += batch_strings.len() as u64;

        // Encode_batch on this batch's strings, fold each sequence into counts
        // immediately, drop the Encodings.
        let inputs: Vec<tokenizers::EncodeInput> =
            batch_strings.iter().map(|s| s.clone().into()).collect();
        let encs = tokenizer.encode_batch(inputs, false)
            .map_err(|e| anyhow!("encode_batch: {}", e))?;
        let mut ids_scratch: Vec<u32> = Vec::with_capacity(64);
        for e in encs {
            ids_scratch.clear();
            ids_scratch.extend(e.get_ids().iter()
                .filter(|&&id| id < vocab_size).copied());
            fold_seq(&mut counts, &mut total_tokens, &ids_scratch);
        }
    }
    log_msg(&format!("[build] loaded {} train passwords", total_lines));
    if let Some(msg) = rec_stats.report() {
        log_msg(&format!("[build] {}", msg));
    }
    log_msg(&format!("[build] streamed in {:.1}s (total tokens: {}; {} trigram contexts)",
        t0.elapsed().as_secs_f64(), total_tokens, counts.len()));

    // ----- Kneser-Ney statistics -----
    // Pass 1: compute unigram raw counts, n1 / n2 (for D estimation), and the
    // type counts N1+(•, b, t) (number of distinct preceding `a` such that
    // c3(a, b, t) > 0).
    log_msg("[build] computing Kneser-Ney statistics...");
    let t0 = Instant::now();
    let mut unigram_raw: Vec<u64> = vec![0u64; vocab_size as usize];
    let mut n1_trigram_types_count1 = 0u64;
    let mut n2_trigram_types_count2 = 0u64;
    // N1+(b, t) — for each pair (b, t), count distinct `a` s.t. c3(a, b, t) > 0.
    // Stored as FxHashMap<(b,t)_packed, count>.
    let mut n1plus_bt: FxHashMap<u64, u32> = FxHashMap::default();
    n1plus_bt.reserve(8_000_000);
    for (&packed_ab, children) in &counts {
        let b = (packed_ab & 0xFFFF_FFFF) as u32;
        for (&t, &c) in children {
            // Unigram raw (skip sentinels — they're not in the decode table)
            if t < vocab_size {
                unigram_raw[t as usize] += c;
            }
            if c == 1 { n1_trigram_types_count1 += 1; }
            if c == 2 { n2_trigram_types_count2 += 1; }
            *n1plus_bt.entry(pack(b, t)).or_insert(0) += 1;
        }
    }
    // Estimate discount D from data; fall back to default if degenerate.
    let discount: f32 = if n1_trigram_types_count1 + 2 * n2_trigram_types_count2 > 0 {
        let d_est = n1_trigram_types_count1 as f64
                  / (n1_trigram_types_count1 + 2 * n2_trigram_types_count2) as f64;
        let d_est = d_est as f32;
        if (0.1..0.95).contains(&d_est) { d_est } else { DEFAULT_KN_DISCOUNT }
    } else {
        DEFAULT_KN_DISCOUNT
    };
    log_msg(&format!(
        "[build] discount D = {:.4} (n1={}, n2={}); fallback if outside 0.1..0.95",
        discount, n1_trigram_types_count1, n2_trigram_types_count2));

    // Pass 2: build trigram cumdists with KN-discounted probs and per-context lambda.
    // For each (a, b): c2 = sum_t c3(a, b, t).
    //   KN-trigram-prob(t | a, b) = max(c3 - D, 0) / c2
    //   lambda(a, b)              = D * |{ t : c3(a,b,t) > 0 }| / c2
    //   sum over t of KN-prob = (c2 - D * N+) / c2 = 1 - lambda
    log_msg("[build] computing trigram KN-discounted cumdists + lambdas...");
    let mut contexts: FxHashMap<Ctx, (Vec<u32>, Vec<f32>)> = FxHashMap::default();
    contexts.reserve(counts.len());
    let mut lambda: FxHashMap<Ctx, f32> = FxHashMap::default();
    lambda.reserve(counts.len());
    for (ctx, cnt) in counts {
        let mut entries: Vec<(u32, u64)> = cnt.into_iter().collect();
        entries.sort_unstable_by_key(|x| x.0);
        let total_c2: u64 = entries.iter().map(|x| x.1).sum();
        let n_plus = entries.len() as f64; // number of distinct children
        let inv_c2 = 1.0_f64 / total_c2 as f64;
        let lam = (discount as f64 * n_plus * inv_c2) as f32;
        // KN-discounted prob per child: (c - D) / c2 if positive else 0.
        let mut ids: Vec<u32> = Vec::with_capacity(entries.len());
        let mut cum: Vec<f32> = Vec::with_capacity(entries.len());
        let mut acc: f64 = 0.0;
        for (id, c) in entries {
            let kn_prob = ((c as f64 - discount as f64).max(0.0)) * inv_c2;
            acc += kn_prob;
            ids.push(id);
            cum.push(acc as f32);
        }
        // Numerical: per-context cum should reach (1 - lam). Snap last value
        // exactly to avoid roundoff drift compounding across deep heap walks.
        if let Some(last) = cum.last_mut() {
            *last = (1.0_f32 - lam).max(0.0);
        }
        contexts.insert(ctx, (ids, cum));
        lambda.insert(ctx, lam);
    }
    log_msg(&format!(
        "[build] trigram + lambda built in {:.0}s",
        t0.elapsed().as_secs_f64()));

    // Pass 3: build bigram KN-continuation distribution.
    // P_cont(t | b) = max(N1+(•, b, t) - D, 0) / N1+(•, b, •)
    //   where N1+(•, b, •) = sum over t of N1+(•, b, t) = total trigram types
    //   ending in (b, _).
    // Cum sums to 1 per b (or close to it; we snap last to 1.0 for numerical
    // safety). For an unseen b (no trigram type ending after b), the bigram
    // entry doesn't exist; the enumerator falls through to no-back-off.
    log_msg("[build] computing bigram KN-continuation distributions...");
    let t0 = Instant::now();
    let mut per_b: FxHashMap<u32, Vec<(u32, u32)>> = FxHashMap::default();
    for (&packed_bt, &n1plus) in &n1plus_bt {
        let b = (packed_bt >> 32) as u32;
        let t = (packed_bt & 0xFFFF_FFFF) as u32;
        per_b.entry(b).or_default().push((t, n1plus));
    }
    let mut bigram_kn: FxHashMap<u32, (Vec<u32>, Vec<f32>)> = FxHashMap::default();
    bigram_kn.reserve(per_b.len());
    for (b, mut entries) in per_b {
        entries.sort_unstable_by_key(|x| x.0);
        let total_types: u64 = entries.iter().map(|x| x.1 as u64).sum();
        if total_types == 0 { continue; }
        let inv = 1.0_f64 / total_types as f64;
        let mut ids: Vec<u32> = Vec::with_capacity(entries.len());
        let mut cum: Vec<f32> = Vec::with_capacity(entries.len());
        let mut acc: f64 = 0.0;
        for (t, types) in entries {
            // KN-continuation discount: (types - D) / total_types if positive.
            let p = ((types as f64 - discount as f64).max(0.0)) * inv;
            acc += p;
            ids.push(t);
            cum.push(acc as f32);
        }
        if let Some(last) = cum.last_mut() { *last = 1.0_f32; }
        bigram_kn.insert(b, (ids, cum));
    }
    log_msg(&format!(
        "[build] bigram KN-cont built in {:.0}s ({} bigram contexts)",
        t0.elapsed().as_secs_f64(), bigram_kn.len()));

    // Unigram KN-continuation: N1+(•, t) / N1+(•, •)
    //   N1+(•, t) = number of distinct b such that some (b, t) trigram type exists
    //   N1+(•, •) = total number of (b, t) trigram types
    let mut n1plus_t: FxHashMap<u32, u32> = FxHashMap::default();
    for &packed_bt in n1plus_bt.keys() {
        let t = (packed_bt & 0xFFFF_FFFF) as u32;
        *n1plus_t.entry(t).or_insert(0) += 1;
    }
    let n1plus_total: u64 = n1plus_t.values().map(|&v| v as u64).sum();
    let inv_total = if n1plus_total > 0 { 1.0_f64 / n1plus_total as f64 } else { 0.0 };
    let mut unigram_kn_cont: Vec<f32> = vec![0.0_f32; vocab_size as usize];
    for (t, n) in n1plus_t {
        if (t as u32) < vocab_size {
            unigram_kn_cont[t as usize] = (n as f64 * inv_total) as f32;
        }
    }
    drop(n1plus_bt);

    // Decode table
    let kind = classify_tokenizer(&tokenizer);
    log_msg(&format!("[build] precomputing decode table (kind={:?})", match kind {
        DecodeKind::Plain => "plain",
        DecodeKind::StripBpeSpace => "byte-level-bpe",
    }));
    let t0 = Instant::now();
    let decode = precompute_decode_table(&tokenizer, vocab_size, kind)?;
    log_msg(&format!("[build] decode table in {:.1}s", t0.elapsed().as_secs_f64()));

    // Save (NGRMv003: v2/KN body + embedded tokenizer provenance).
    let model = Model {
        contexts, decode, start_id, end_id,
        is_kn: true, discount,
        lambda, bigram_kn, unigram_raw, unigram_kn_cont,
    };
    // Embed the tokenizer.json bytes so the model is self-describing: `generate
    // --wordlist` loads it straight from the .ngram, no sidecar/env/registry. If
    // the tokenizer bytes can't be read for some reason, fall back to a v2 model
    // (the sidecar path still resolves it) rather than failing the build.
    let provenance = match std::fs::read(&args.tokenizer) {
        Ok(tokenizer_json) => Some(Provenance {
            tokenizer_json,
            tok_source: args.tokenizer.display().to_string(),
            train_path: args.train.display().to_string(),
            build_epoch: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs()).unwrap_or(0),
            binver: env!("CARGO_PKG_VERSION").to_string(),
        }),
        Err(e) => {
            warn_msg(&format!("[build] warning: could not read tokenizer to embed ({e}); \
                              saving v2 model without provenance"));
            None
        }
    };
    model_save(&out_path, &model, provenance.as_ref(), args.force)?;

    // Co-locate a copy of the tokenizer next to the model as <stem>.tokenizer.json
    // so `generate --wordlist` can auto-resolve it without TOKENOV_TOKENIZER, and
    // so the two always travel together. Best-effort: a copy
    // failure doesn't invalidate the built model — wordlist mode can still fall
    // back to env/default-alias resolution.
    let tok_sidecar = out_path.with_extension("tokenizer.json");
    // Guard against src==dest: the sidecar is named `<stem>.tokenizer.json`, so a
    // rebuild that points --tokenizer at a model's own co-located sidecar would
    // make src and dest the same path — and `std::fs::copy(p, p)` truncates the
    // file to zero before reading it, wiping the tokenizer. Skip the no-op copy.
    let same_file = std::fs::canonicalize(&args.tokenizer).ok()
        == std::fs::canonicalize(&tok_sidecar).ok()
        && tok_sidecar.exists();
    if same_file {
        log_msg(&format!("[build] tokenizer already co-located at {}", tok_sidecar.display()));
    } else {
        match std::fs::copy(&args.tokenizer, &tok_sidecar) {
            Ok(_)  => log_msg(&format!("[build] co-located tokenizer -> {}", tok_sidecar.display())),
            Err(e) => warn_msg(&format!("[build] warning: could not co-locate tokenizer at {}: {}",
                tok_sidecar.display(), e)),
        }
    }

    // Record the built model in ~/.config/tokenov/models.toml (non-fatal).
    // Registration happens AFTER the model is in place at its final path, so the
    // registry can never point at a temp file. The registry name honors --name
    // if given, else the (clean) output file stem.
    let reg_name = args.name.clone().unwrap_or_else(|| out_path.file_stem()
        .and_then(|s| s.to_str()).unwrap_or("model").to_string());
    let reg_size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    match registry::register(&reg_name, &out_path, reg_size) {
        Ok(p)  => log_msg(&format!("[registry] recorded '{}' in {}", reg_name, p.display())),
        Err(e) => warn_msg(&format!("[registry] warning: could not record model: {}", e)),
    }

    // Completion confirmation (always shown — this is the command's result). The
    // detailed breakdown stays behind --verbose.
    let n_ctx = model.contexts.len();
    let mb = std::fs::metadata(&out_path)?.len() / 1_000_000;
    println!("Trained model '{}' → {} ({} MB, {} contexts)", reg_name, out_path.display(), mb, n_ctx);
    log_msg(&format!(
        "[build] DONE: name={} vocab_size={} train_lines={} total_tokens={} contexts={} output={} ({} MB) total_time={:.1}s",
        name, vocab_size, total_lines, total_tokens, n_ctx,
        out_path.display(), mb,
        t_overall.elapsed().as_secs_f64()
    ));
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// `tokenov delete` — remove a model's registry entry and (optionally) its file
// ─────────────────────────────────────────────────────────────────────────────
fn run_delete(args: DeleteArgs) -> Result<()> {
    // Bulk cleanup of dangling entries — nothing to delete on disk, no confirm.
    if args.missing {
        let removed = registry::remove_missing()?;
        if removed.is_empty() {
            println!("No MISSING entries to remove.");
        } else {
            println!("Removed {} MISSING registry entr{}:", removed.len(),
                if removed.len() == 1 { "y" } else { "ies" });
            for n in &removed {
                println!("  {n}");
            }
        }
        return Ok(());
    }

    let name = args.name.as_deref().ok_or_else(||
        anyhow!("delete: specify a model NAME, or use --missing to clear dangling entries"))?;

    // Look up by registry name first; fall back to treating NAME as a path.
    let (reg_name, file_path): (Option<String>, Option<PathBuf>) = match registry::find(name) {
        Some(e) => (Some(e.name), Some(PathBuf::from(e.path))),
        None => {
            let p = PathBuf::from(name);
            if p.exists() {
                (None, Some(p))
            } else {
                bail!("'{}' is neither a registered model name nor an existing file \
                       (see `tokenov --list-models`)", name);
            }
        }
    };

    // Delete the file unless --keep-file. Confirm first unless -y or the file is
    // already gone (a dangling entry — nothing to lose).
    if !args.keep_file {
        if let Some(p) = &file_path {
            if p.exists() {
                if !args.yes && !confirm(&format!("Delete model file {}?", p.display()))? {
                    println!("Aborted; nothing changed.");
                    return Ok(());
                }
                std::fs::remove_file(p).with_context(|| format!("delete {}", p.display()))?;
                println!("Deleted file {}", p.display());
            }
        }
    }

    if let Some(rn) = reg_name {
        registry::deregister(&rn)?;
        println!("Deregistered '{}'.", rn);
    } else {
        println!("(No registry entry for that path; file handled above.)");
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// `tokenov register` — add an existing model file to the registry
// ─────────────────────────────────────────────────────────────────────────────
fn run_register(args: RegisterArgs) -> Result<()> {
    if !args.path.exists() {
        bail!("no such file: {}", args.path.display());
    }
    let name = args.name.clone().unwrap_or_else(|| args.path.file_stem()
        .and_then(|s| s.to_str()).unwrap_or("model").to_string());

    // Warn before replacing an existing entry that points at a different file.
    if let Some(e) = registry::find(&name) {
        let new_abs = args.path.canonicalize().unwrap_or_else(|_| args.path.clone());
        if PathBuf::from(&e.path) != new_abs {
            if !args.yes && !confirm(&format!(
                "'{}' already registered → {}\n  replace with {}?",
                name, e.path, new_abs.display()))? {
                println!("Aborted; nothing changed.");
                return Ok(());
            }
        }
    }

    let size = std::fs::metadata(&args.path).map(|m| m.len()).unwrap_or(0);
    let p = registry::register(&name, &args.path, size)?;
    println!("Registered '{}' → {} (in {})", name, args.path.display(), p.display());
    Ok(())
}

/// Prompt the user for a y/N confirmation on stdin. Returns false on EOF / non-tty.
fn confirm(prompt: &str) -> Result<bool> {
    use std::io::Write as _;
    print!("{prompt} [y/N] ");
    std::io::stdout().flush().ok();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line)? == 0 {
        return Ok(false); // EOF (e.g. piped/non-interactive) → treat as "no"
    }
    Ok(matches!(line.trim(), "y" | "Y" | "yes" | "Yes"))
}

// ============================================================================
// `tokenov generate` — emit candidates, with optional wordlist targeting
// ============================================================================

// ============================================================================
// Skipgram-style wordlist expansion via bigram-distribution cosine similarity.
//
// Two tokens s, t are "similar" if their KN-continuation distributions
// P_cont(* | s) and P_cont(* | t) are similar in shape (cosine over the sparse
// per-token bigram entries). Tokens that follow similar things are similar in
// behavior; in password contexts this captures structural variants (e.g.
// 'password' ↔ 'passwd' both followed by digits, suffixes, END at similar
// rates).
//
// Expansion: for each token in the wordlist, find the top-K most-similar
// tokens by cosine similarity over their bigram-cont distributions, and add
// those to the W set used by weighted/combined bias. Implementation is sparse:
// per-token vectors live as (id, prob) pairs from the model's bigram_kn map;
// dot products use sparse intersection (both sides sorted by id, merge walk).
//
// Cost: O(|W_explicit| × n_bigram_contexts × avg_children) for the sweep.
// For 100-token wordlist + 30K bigram-contexts + 50 avg children: ~150M ops,
// ~150ms wall clock.
// ============================================================================

fn rebuild_per_token_probs(model: &Model) -> FxHashMap<u32, Vec<(u32, f32)>> {
    // Forward direction: for each token b (the "second token" / preceding
    // token), the vector is its P_cont(t | b) distribution over t.
    //
    // Convert each bigram_kn cumdist back to per-child raw probability so we
    // can compute dot products cleanly.
    let mut out: FxHashMap<u32, Vec<(u32, f32)>> = FxHashMap::default();
    out.reserve(model.bigram_kn.len());
    for (&b, (ids, cum)) in &model.bigram_kn {
        let mut entries: Vec<(u32, f32)> = Vec::with_capacity(ids.len());
        let mut prev: f32 = 0.0;
        for (i, &id) in ids.iter().enumerate() {
            let p = cum[i] - prev;
            prev = cum[i];
            entries.push((id, p));
        }
        // Already sorted by id (build pass sorts before serializing).
        out.insert(b, entries);
    }
    out
}

/// Inverse direction: for each token t (the "third token" / current token),
/// the vector is the distribution of P_cont(t | b) values across all b that
/// have t in their forward distribution. Tokens t1, t2 are "inverse-similar"
/// if they appear in the continuation distributions of similar b's with
/// similar weights — captures preceding-context similarity (prefix-role).
///
/// The sparse cosine of these vectors works the same as the forward direction;
/// the values aren't strictly P(b | t) but the cosine just needs comparable
/// vectors and the cosine-similarity scale is the same shape.
fn rebuild_per_token_probs_inverse(model: &Model) -> FxHashMap<u32, Vec<(u32, f32)>> {
    let mut out: FxHashMap<u32, Vec<(u32, f32)>> = FxHashMap::default();
    for (&b, (ids, cum)) in &model.bigram_kn {
        let mut prev: f32 = 0.0;
        for (i, &t) in ids.iter().enumerate() {
            let p = cum[i] - prev;
            prev = cum[i];
            out.entry(t).or_default().push((b, p));
        }
    }
    // sparse_cosine requires entries sorted by id (the b values here).
    for v in out.values_mut() {
        v.sort_by_key(|x| x.0);
    }
    out
}

fn precompute_bigram_norms(per_token: &FxHashMap<u32, Vec<(u32, f32)>>) -> FxHashMap<u32, f32> {
    let mut out: FxHashMap<u32, f32> = FxHashMap::default();
    out.reserve(per_token.len());
    for (&b, entries) in per_token {
        let s: f64 = entries.iter().map(|(_, p)| (*p as f64) * (*p as f64)).sum();
        out.insert(b, s.sqrt() as f32);
    }
    out
}

/// Sparse cosine over two id-sorted (id, prob) lists. Linear in their combined
/// length via merge walk.
fn sparse_cosine(
    a_entries: &[(u32, f32)], a_norm: f32,
    b_entries: &[(u32, f32)], b_norm: f32,
) -> f32 {
    if a_norm == 0.0 || b_norm == 0.0 { return 0.0; }
    let mut i = 0usize;
    let mut j = 0usize;
    let mut dot: f64 = 0.0;
    while i < a_entries.len() && j < b_entries.len() {
        let (ai, ap) = a_entries[i];
        let (bi, bp) = b_entries[j];
        if ai == bi {
            dot += (ap as f64) * (bp as f64);
            i += 1; j += 1;
        } else if ai < bi {
            i += 1;
        } else {
            j += 1;
        }
    }
    (dot / (a_norm as f64 * b_norm as f64)) as f32
}

/// Inner sweep: for each w in `wordlist_tokens`, find the top-K most-similar
/// tokens by cosine over `per_token[*]` and add them to `out`. Returns
/// (n_added, n_not_found).
fn skipgram_sweep(
    per_token: &FxHashMap<u32, Vec<(u32, f32)>>,
    norms: &FxHashMap<u32, f32>,
    wordlist_tokens: &[u32],
    top_k: usize,
    out: &mut FxHashMap<u32, ()>,
) -> (usize, usize) {
    let mut not_found = 0usize;
    let mut summed_added = 0usize;
    for &w in wordlist_tokens {
        let w_entries = match per_token.get(&w) {
            Some(v) => v,
            None => { not_found += 1; continue; }
        };
        let w_norm = norms[&w];
        let mut topk: Vec<(f32, u32)> = Vec::with_capacity(top_k + 1);
        for (&t, t_entries) in per_token {
            if t == w { continue; }
            let t_norm = norms[&t];
            let sim = sparse_cosine(w_entries, w_norm, t_entries, t_norm);
            if topk.len() < top_k {
                topk.push((sim, t));
                if topk.len() == top_k {
                    topk.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
                }
            } else if sim > topk[0].0 {
                topk[0] = (sim, t);
                topk.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            }
        }
        for (_sim, t) in topk {
            if !out.contains_key(&t) {
                out.insert(t, ());
                summed_added += 1;
            }
        }
    }
    (summed_added, not_found)
}

/// Return W_expanded as a set: W_explicit ∪ { top-K most-similar tokens to
/// each w in W_explicit by bigram-distribution cosine }. Skips wordlist tokens
/// that have no bigram entry (no contexts seen in training; rare).
///
/// Direction:
///   - Forward: similarity over P(t | w) — "what comes after w". Captures
///     suffix-role similarity (e.g., common base words followed by digit
///     suffixes).
///   - Inverse: similarity over P(b | w) — "what comes before w". Captures
///     prefix-role similarity (e.g., named-entity / brand tokens occupying
///     the same starting positions).
///   - Both: union of top-K from each direction.
fn expand_via_skipgram(
    model: &Model,
    wordlist_tokens: &[u32],
    top_k: usize,
    direction: SkipgramDirection,
) -> Vec<u32> {
    if top_k == 0 || !model.is_kn {
        return wordlist_tokens.to_vec();
    }

    let mut out: FxHashMap<u32, ()> = FxHashMap::default();
    for &w in wordlist_tokens { out.insert(w, ()); }

    let do_fwd = matches!(direction, SkipgramDirection::Forward | SkipgramDirection::Both);
    let do_inv = matches!(direction, SkipgramDirection::Inverse | SkipgramDirection::Both);

    let mut total_added = 0usize;

    if do_fwd {
        let t0 = Instant::now();
        let per_token = rebuild_per_token_probs(model);
        let norms = precompute_bigram_norms(&per_token);
        log_msg(&format!(
            "[skipgram] forward index: {} bigram contexts in {:.1}s",
            norms.len(), t0.elapsed().as_secs_f64()));
        let t1 = Instant::now();
        let (added, not_found) = skipgram_sweep(&per_token, &norms, wordlist_tokens, top_k, &mut out);
        total_added += added;
        log_msg(&format!(
            "[skipgram] forward sweep: +{} tokens (top_k={}, {} skipped no-fwd-entry) in {:.1}s",
            added, top_k, not_found, t1.elapsed().as_secs_f64()));
    }

    if do_inv {
        let t0 = Instant::now();
        let per_token_inv = rebuild_per_token_probs_inverse(model);
        let norms_inv = precompute_bigram_norms(&per_token_inv);
        log_msg(&format!(
            "[skipgram] inverse index: {} tokens with precursors in {:.1}s",
            norms_inv.len(), t0.elapsed().as_secs_f64()));
        let t1 = Instant::now();
        let (added, not_found) = skipgram_sweep(&per_token_inv, &norms_inv, wordlist_tokens, top_k, &mut out);
        total_added += added;
        log_msg(&format!(
            "[skipgram] inverse sweep: +{} tokens (top_k={}, {} skipped no-inv-entry) in {:.1}s",
            added, top_k, not_found, t1.elapsed().as_secs_f64()));
    }

    log_msg(&format!(
        "[skipgram] direction={:?}, expanded {} explicit tokens by {} new total",
        direction, wordlist_tokens.len(), total_added));

    // Debug: print the expanded set (decoded) up to a cap, so the user can
    // verify which tokens were brought in. Suppressed when TOKENOV_QUIET=1.
    if std::env::var("TOKENOV_QUIET").is_err() {
        let mut all: Vec<u32> = out.keys().copied().collect();
        all.sort_unstable();
        let cap = 50usize;
        let preview: Vec<String> = all.iter().take(cap).map(|&id| {
            if (id as usize) < model.decode.len() {
                String::from_utf8_lossy(&model.decode[id as usize]).into_owned()
            } else {
                format!("<id={}>", id)
            }
        }).collect();
        log_msg(&format!(
            "[skipgram] expanded W (first {} of {}): {:?}",
            preview.len(), all.len(), preview));
    }

    out.keys().copied().collect()
}

/// Apply per-context multiplicative bias to a cum-distribution model (Scheme A).
/// **Use only on V1 (unsmoothed) models.** For V2/KN models, use
/// `apply_weighted_bias_kn_aware` instead — this function operates on the
/// trigram tier alone, which over-amplifies trigram mass in KN models where
/// trigram sums to (1 - lambda) and the bigram-backoff tier carries lambda
/// worth of mass (left untouched here, leading to per-context emission >1).
///
/// Per-context renormalization preserves distribution-summing-to-1.
fn apply_weighted_bias(
    model: &Model,
    w_set: &FxHashMap<u32, ()>,
    bias: f32,
) -> FxHashMap<Ctx, (Vec<u32>, Vec<f32>)> {
    let mut out: FxHashMap<Ctx, (Vec<u32>, Vec<f32>)> = FxHashMap::default();
    out.reserve(model.contexts.len());
    for (&ctx, (ids, cum)) in &model.contexts {
        // Rebuild raw probs from cumdist
        let n = ids.len();
        let mut raw = Vec::with_capacity(n);
        let mut prev = 0.0_f32;
        for i in 0..n {
            raw.push(cum[i] - prev);
            prev = cum[i];
        }
        // Bias multiply
        for (i, &id) in ids.iter().enumerate() {
            if w_set.contains_key(&id) {
                raw[i] *= bias;
            }
        }
        // Renormalize (per context)
        let total: f32 = raw.iter().sum();
        if total <= 0.0 { continue; }
        let inv = 1.0_f32 / total;
        let mut new_cum = Vec::with_capacity(n);
        let mut acc = 0.0_f32;
        for r in &raw {
            acc += r * inv;
            new_cum.push(acc);
        }
        if let Some(last) = new_cum.last_mut() { *last = 1.0_f32; }
        out.insert(ctx, (ids.clone(), new_cum));
    }
    out
}

/// KN-aware multiplicative bias (Scheme A, joint-distribution variant).
///
/// For V2/KN models, the trigram tier sums to `(1 - lambda)` per context, with
/// the remaining `lambda` worth of mass flowing through the bigram-backoff
/// tier. Biasing the trigram tier alone (as `apply_weighted_bias` does) yields
/// per-context emission probabilities that sum to >1 because the bigram tier
/// is unmodified — the model is no longer a proper distribution.
///
/// This function instead computes the **joint emission distribution** per
/// context (the deduped union of trigram entries with weight `P_tri(t)` and
/// bigram-backoff entries with weight `λ × P_cont(t)`), applies the bias to
/// W-tokens in that joint table, and renormalizes the whole context to sum
/// to 1. Output is a single-tier per-context distribution.
///
/// Bigram-only contexts (where `(a,b)` has no trigram entry but `b` has bigram
/// entries) are handled by the second return value: a globally pre-biased
/// `bigram_kn` keyed on `b` with `bias_factor[t]` applied per token then
/// renormalized per b. The caller wires this into `post_bias_model.bigram_kn`
/// and sets `lambda` to empty so the enumerator's bigram-fallback path emits
/// these entries with `log_lam = 0` (full-backoff weight).
///
/// For V1 (non-KN) models this function falls back to the trigram-only
/// behavior (which is correct in that regime) — see `apply_weighted_bias`.
fn apply_weighted_bias_kn_aware(
    model: &Model,
    w_set: &FxHashMap<u32, ()>,
    bias: f32,
) -> (FxHashMap<Ctx, (Vec<u32>, Vec<f32>)>,
      FxHashMap<u32, (Vec<u32>, Vec<f32>)>) {
    if !model.is_kn {
        // V1 path: bigram tier doesn't exist, trigram-only bias is correct.
        let contexts = apply_weighted_bias(model, w_set, bias);
        return (contexts, FxHashMap::default());
    }

    // V2/KN: build per-context joint emission, bias, renormalize.
    //
    // Pre-decompose bigram_kn[b] into (ids, raw_probs) for fast lookup. This
    // dictionary is read-only during the per-context parallel sweep.
    let mut bigram_raw: FxHashMap<u32, Vec<(u32, f32)>> = FxHashMap::default();
    bigram_raw.reserve(model.bigram_kn.len());
    for (&b, (ids, cum)) in &model.bigram_kn {
        let mut entries = Vec::with_capacity(ids.len());
        let mut prev = 0.0_f32;
        for (i, &id) in ids.iter().enumerate() {
            let p = cum[i] - prev;
            prev = cum[i];
            entries.push((id, p));
        }
        bigram_raw.insert(b, entries);
    }

    // Parallel per-context biasing. Each context is independent; collect into
    // a Vec<(Ctx, ...)> then assemble the output FxHashMap. Using `par_iter`
    // on `model.contexts` requires it to be a HashMap of materialized entries,
    // which it is.
    //
    // Note: `tri_seen` membership check uses linear scan over a small Vec
    // rather than a HashMap. Per-context entry count is typically <100; linear
    // scan beats HashMap allocation overhead × millions of contexts.
    let entries: Vec<(Ctx, (Vec<u32>, Vec<f32>))> = model.contexts
        .par_iter()
        .filter_map(|(&ctx, (tri_ids, tri_cum))| {
            // ctx = pack(a, b) = (a << 32) | b — extract b (low 32 bits).
            let b = (ctx & 0xFFFF_FFFFu64) as u32;
            let lam = model.lambda.get(&ctx).copied().unwrap_or(0.0);

            // Build joint = trigram entries + λ * bigram \ trigram.
            let mut joint: Vec<(u32, f32)> = Vec::with_capacity(
                tri_ids.len() + MAX_KN_BIGRAM_CHILDREN);
            let mut prev = 0.0_f32;
            for (i, &id) in tri_ids.iter().enumerate() {
                let p = tri_cum[i] - prev;
                prev = tri_cum[i];
                joint.push((id, p));
            }
            // tri_seen as a sorted view over the trigram ids (already sorted
            // by id at build time), used via binary search.
            let tri_seen_n = tri_ids.len();
            if let Some(bi_entries) = bigram_raw.get(&b) {
                for &(id, q) in bi_entries.iter().take(MAX_KN_BIGRAM_CHILDREN) {
                    // tri_ids is sorted ascending by id (build-time invariant);
                    // binary_search is O(log n) per lookup vs O(n) for Vec.
                    if tri_ids[..tri_seen_n].binary_search(&id).is_err() {
                        joint.push((id, lam * q));
                    }
                }
            }

            // Apply bias to W-tokens.
            for entry in joint.iter_mut() {
                if w_set.contains_key(&entry.0) {
                    entry.1 *= bias;
                }
            }

            // Renormalize per context.
            let total: f32 = joint.iter().map(|(_, p)| *p).sum();
            if total <= 0.0 { return None; }
            let inv = 1.0_f32 / total;

            // Sort by id for deterministic ordering (not required by enumerator
            // — prepare_enum_model re-sorts by log-prob — but keeps output
            // stable).
            joint.sort_unstable_by_key(|x| x.0);

            let mut ids = Vec::with_capacity(joint.len());
            let mut cum = Vec::with_capacity(joint.len());
            let mut acc = 0.0_f32;
            for (id, p) in &joint {
                acc += p * inv;
                ids.push(*id);
                cum.push(acc);
            }
            if let Some(last) = cum.last_mut() { *last = 1.0_f32; }
            Some((ctx, (ids, cum)))
        })
        .collect();

    let mut out_contexts: FxHashMap<Ctx, (Vec<u32>, Vec<f32>)> = FxHashMap::default();
    out_contexts.reserve(entries.len());
    for (ctx, e) in entries {
        out_contexts.insert(ctx, e);
    }

    // Pre-biased bigram_kn for bigram-only contexts. Apply per-token bias and
    // renormalize per b. The enumerator emits these for any context not in
    // out_contexts (i.e. bigram-only) via the bigram-fallback code path with
    // log_lam = 0 (lambda unset).
    let bigram_entries: Vec<(u32, (Vec<u32>, Vec<f32>))> = model.bigram_kn
        .par_iter()
        .filter_map(|(&b, (ids, cum))| {
            let mut raw = Vec::with_capacity(ids.len());
            let mut prev = 0.0_f32;
            for (i, &id) in ids.iter().enumerate() {
                let p = cum[i] - prev;
                prev = cum[i];
                let bias_factor = if w_set.contains_key(&id) { bias } else { 1.0 };
                raw.push(p * bias_factor);
            }
            let total: f32 = raw.iter().sum();
            if total <= 0.0 { return None; }
            let inv = 1.0_f32 / total;
            let mut new_cum = Vec::with_capacity(ids.len());
            let mut acc = 0.0_f32;
            for r in &raw {
                acc += r * inv;
                new_cum.push(acc);
            }
            if let Some(last) = new_cum.last_mut() { *last = 1.0_f32; }
            Some((b, (ids.clone(), new_cum)))
        })
        .collect();
    let mut out_bigram: FxHashMap<u32, (Vec<u32>, Vec<f32>)> = FxHashMap::default();
    out_bigram.reserve(bigram_entries.len());
    for (b, e) in bigram_entries {
        out_bigram.insert(b, e);
    }

    (out_contexts, out_bigram)
}

// `EnumModel` and `prepare_enum_model` moved to `variant.rs` / `variant_a.rs`.
// Each variant module owns its own `prepare` implementation, called via the
// `Variant` trait at runtime. See `mod variant; mod variant_a;` near the top
// of this file.

#[derive(Clone, Copy)]
struct HeapEntry {
    log_prob: f32,
    prefix_len: u8,
    prefix: [u32; 32], // big enough for any practical max-tokens; spec caps at 12
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool { self.log_prob == other.log_prob }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { self.log_prob.partial_cmp(&other.log_prob) }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering { self.log_prob.partial_cmp(&other.log_prob).unwrap_or(Ordering::Equal) }
}

/// Precomputed children map shared across all generation threads.
/// Built once from the trigram model; threads fall back to per-thread local
/// caches for contexts not present (rare KN bigram-only paths).
///
/// Storage: `Box<[(u32, f32)]>` per ctx (16 B header) instead of the older
/// `Arc<Vec<(u32, f32)>>` (8 B Arc ptr + 16 B ArcInner header + 24 B Vec
/// header per ctx). Saves ~32 B/ctx plus per-Arc allocator overhead. At the
/// 100M-line training scale (13.83M ctxs) that's ~660 MB.
///
/// Lifetime: the map is `Box::leak`'d once at build time, giving the rest of
/// the process `&'static` access without ref-counting. The leaked memory is
/// reclaimed by the OS on process exit — the gen path runs once per process
/// invocation, so this is fine.
type ChildCacheMap = FxHashMap<Ctx, Box<[(u32, f32)]>>;
type ChildCache = &'static ChildCacheMap;

/// Pre-built gen setup shared between run_generate and do_calibration.
///
/// Before this struct existed, do_calibration would model_load + variant.prepare
/// + build_child_cache internally, even when invoked inline from run_generate
/// which had ALREADY built those structures. The two child_cache instances
/// then coexisted in memory during calibration, ~3 GB each at the 50M-line
/// model scale, blowing past the soft memory cap and tripping the mem_monitor
/// abort right after calibration finished.
///
/// CalibSetup lets run_generate pass its already-built structures into
/// resolve_chunk_size → do_calibration. The standalone `calibrate` subcommand
/// (run_calibrate) still has to build the setup itself, but only one copy
/// exists at a time in that flow.
struct CalibSetup {
    enum_model: Arc<EnumModel>,
    child_cache: ChildCache,
    kind: DecodeKind,
    start_id: u32,
    end_id: u32,
    decode_table: Arc<Vec<Vec<u8>>>,
    first_level: Vec<HeapEntry>,
}

/// Child-cache mode for one generation run. Cheap to
/// Copy; passed to every worker's `enumerate_to_sink`. `Bounded(cap)` carries
/// the PER-THREAD cache cap in entries.
#[derive(Clone, Copy, Debug, PartialEq)]
enum CacheMode {
    /// Whole resident child_cache built up front; hits are zero-copy `&'static`.
    Full,
    /// Per-thread bounded cache of `cap` hot contexts; miss ⇒ recompute.
    Bounded(usize),
    /// No cache; every context recomputed on demand.
    Lazy,
}

/// A frame's children slice, sourced one of three ways depending on `CacheMode`.
/// All three `Deref` to `&[(u32, f32)]`, so the DFS reads them uniformly.
enum Children {
    /// Zero-copy `'static` slice — FULL-cache hit or a leaked local-miss entry.
    Ref(&'static [(u32, f32)]),
    /// Refcounted slice from a BOUNDED per-thread cache. The `Arc` keeps the
    /// data alive for the frame's lifetime even if the cache evicts the entry
    /// (eviction just drops the cache's `Arc`), so eviction can never dangle.
    Shared(Arc<[(u32, f32)]>),
    /// Owned slice — a LAZY recompute, freed when the frame pops.
    Owned(Box<[(u32, f32)]>),
}
impl std::ops::Deref for Children {
    type Target = [(u32, f32)];
    #[inline]
    fn deref(&self) -> &[(u32, f32)] {
        match self {
            Children::Ref(s) => s,
            Children::Shared(a) => a,
            Children::Owned(b) => b,
        }
    }
}

/// Per-thread bounded child cache — a two-generation ("segmented")
/// approximation of LRU. NO shared state ⇒ zero cross-thread lock contention:
/// each worker owns its own instance, so the hot path is a plain `FxHashMap`
/// lookup + an `Arc` clone, never a lock.
///
/// Policy: inserts land in `young`; when `young` reaches `cap`, it becomes `old`
/// (wholesale) and a fresh empty `young` starts — so the least-recently-touched
/// generation is evicted in one O(1) swap+clear. A hit in `old` is promoted back
/// to `young` (recency). Resident entries are bounded by `young.len()+old.len()`
/// ≤ `2*cap`. Because it is only a performance cache (a miss recomputes the exact
/// same children), the eviction policy cannot affect output — only hit rate.
struct BoundedChildCache {
    cap: usize,
    young: FxHashMap<Ctx, Arc<[(u32, f32)]>>,
    old: FxHashMap<Ctx, Arc<[(u32, f32)]>>,
    hits: u64,
    misses: u64,
}
impl BoundedChildCache {
    fn new(cap: usize) -> Self {
        let cap = cap.max(1);
        BoundedChildCache {
            cap,
            young: FxHashMap::with_capacity_and_hasher(cap + 1, Default::default()),
            old: FxHashMap::default(),
            hits: 0,
            misses: 0,
        }
    }
    /// Return the children for `ctx`, computing + caching them on a miss.
    #[inline]
    fn get_or(&mut self, ctx: Ctx, compute: impl FnOnce() -> Vec<(u32, f32)>) -> Arc<[(u32, f32)]> {
        if let Some(a) = self.young.get(&ctx) {
            self.hits += 1;
            return a.clone();
        }
        if let Some(a) = self.old.remove(&ctx) {
            self.hits += 1;
            self.young.insert(ctx, a.clone()); // promote to young (recency)
            self.rotate_if_full();
            return a;
        }
        self.misses += 1;
        let a: Arc<[(u32, f32)]> = Arc::from(compute().into_boxed_slice());
        self.young.insert(ctx, a.clone());
        self.rotate_if_full();
        a
    }
    #[inline]
    fn rotate_if_full(&mut self) {
        if self.young.len() >= self.cap {
            std::mem::swap(&mut self.young, &mut self.old); // old ⇐ full young (evicts prev old)
            self.young.clear(); // fresh empty young, reusing its capacity
        }
    }
}

struct DfsFrame {
    /// Children of this frame's context — see `Children` for the three sources.
    children:  Children,
    idx:       usize,
    base_lp:   f32,
    acc_level: u32,
}

/// Read a candidate/wordlist source line-by-line, recovering legacy-encoded
/// (non-UTF-8) lines to UTF-8 rather than aborting or silently dropping them.
/// Reads raw bytes (never `BufRead::lines()`, which errors on the first
/// non-UTF-8 byte) and reports how many lines were recovered/skipped.
fn read_lines_recovered<R: BufRead>(mut r: R, tag: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut stats = recover::RecoveryStats::default();
    let mut buf: Vec<u8> = Vec::with_capacity(256);
    loop {
        buf.clear();
        let n = r.read_until(b'\n', &mut buf)?;
        if n == 0 { break; } // EOF
        while matches!(buf.last(), Some(b'\n') | Some(b'\r')) { buf.pop(); }
        if buf.is_empty() { continue; }
        let d = recover::decode_line(&buf);
        stats.note(&d);
        if let Some(s) = d.into_text() { out.push(s); }
    }
    if let Some(msg) = stats.report() { log_msg(&format!("[{}] {}", tag, msg)); }
    Ok(out)
}

fn read_wordlist(path: &Path) -> Result<Vec<String>> {
    let f = File::open(path).with_context(|| format!("read --wordlist {}", path.display()))?;
    read_lines_recovered(BufReader::with_capacity(1 << 20, f), "wordlist")
}

/// Output sink — stdout, a plain file, or `Discard` (calibration).
///
/// Tokenov writes plain UTF-8 text only. Compression (gzip, 7z, etc.) is
/// the caller's responsibility — pipe through your tool of choice or run
/// it as a post-step. Plain-text output is the only mode that supports
/// `--resume` cleanly (a streaming compressor's archive can't be appended
/// to after a kill), so dropping in-process compression is what makes
/// resume universally work.
enum Sink {
    // Unlocked Stdout handle (vs StdoutLock<'static>) because Sink moves into
    // the merger thread in parallel mode and StdoutLock is !Send. Each write
    // through BufWriter takes the stdout lock internally; the BufWriter
    // batches writes so the lock cost is amortized over ~8 MB chunks.
    Stdout(BufWriter<std::io::Stdout>),
    File(BufWriter<File>),
    /// Counts emits and discards bytes — used by `tokenov calibrate` to
    /// measure throughput without writing GB of /dev/null traffic.
    Discard {
        emit_count: Arc<AtomicU64>,
    },
}

impl Sink {
    fn open(out: Option<&Path>) -> Result<Self> {
        Self::open_inner(out, false)
    }
    /// Open a sink in append mode (used for `--resume`). Only valid for a
    /// plain file path — stdout isn't appendable.
    fn open_append(out: &Path) -> Result<Self> {
        Self::open_inner(Some(out), true)
    }
    fn open_inner(out: Option<&Path>, append: bool) -> Result<Self> {
        match out {
            None => {
                if append { bail!("--resume requires --output FILE (cannot append to stdout)"); }
                Ok(Sink::Stdout(BufWriter::with_capacity(8 << 20, std::io::stdout())))
            }
            Some(p) => {
                if let Some(parent) = p.parent() { std::fs::create_dir_all(parent).ok(); }
                let f = if append {
                    std::fs::OpenOptions::new()
                        .create(true).append(true)
                        .open(p)
                        .with_context(|| format!("open append {}", p.display()))?
                } else {
                    File::create(p).with_context(|| format!("create {}", p.display()))?
                };
                Ok(Sink::File(BufWriter::with_capacity(8 << 20, f)))
            }
        }
    }
    /// Open a `Discard` sink that counts emits but writes no bytes. Used by
    /// `tokenov calibrate` so the merger has somewhere to send items while
    /// throughput is measured without producing GB of garbage.
    fn open_discard(emit_count: Arc<AtomicU64>) -> Self {
        Sink::Discard { emit_count }
    }
    fn write_line(&mut self, bytes: &[u8]) -> Result<()> {
        match self {
            Sink::Stdout(w) => { w.write_all(bytes)?; w.write_all(b"\n")?; }
            Sink::File(w)   => { w.write_all(bytes)?; w.write_all(b"\n")?; }
            Sink::Discard { emit_count } => {
                emit_count.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
        Ok(())
    }
    /// Write a pre-built buffer (already candidate+'\n' concatenated) in one
    /// call. Used by --fast where each worker batches its own output.
    fn write_raw(&mut self, buf: &[u8]) -> Result<()> {
        match self {
            Sink::Stdout(w) => { w.write_all(buf)?; }
            Sink::File(w)   => { w.write_all(buf)?; }
            Sink::Discard { .. } => {}
        }
        Ok(())
    }
    /// Write a pre-built chunk buffer (candidates + '\n' already
    /// concatenated, in order) in a single call. `n_items` is the candidate
    /// count in the buffer — used only by Discard to count emits, since that
    /// count isn't recoverable from the bytes.
    fn write_chunk(&mut self, buf: &[u8], n_items: u64) -> Result<()> {
        match self {
            Sink::Stdout(w) => { w.write_all(buf)?; }
            Sink::File(w)   => { w.write_all(buf)?; }
            Sink::Discard { emit_count } => {
                emit_count.fetch_add(n_items, AtomicOrdering::Relaxed);
            }
        }
        Ok(())
    }
    /// Flush user-space buffers to the OS. Used by the merger right before
    /// writing the progress sidecar so a kill-after-progress-write leaves the
    /// file with at least `emitted` lines on disk. (Note: anything still in
    /// the BufWriter's user-space buffer at kill time is lost — that's why
    /// the progress sidecar also records `byte_offset`, which the resume
    /// path uses to truncate any over-spillage.)
    fn flush_buffered(&mut self) -> Result<()> {
        match self {
            Sink::Stdout(w)      => { w.flush()?; }
            Sink::File(w)        => { w.flush()?; }
            Sink::Discard { .. } => {}
        }
        Ok(())
    }
    fn finish(self) -> Result<()> {
        match self {
            Sink::Stdout(mut w)  => { w.flush()?; }
            Sink::File(mut w)    => { w.flush()?; }
            Sink::Discard { .. } => {}
        }
        Ok(())
    }
}

/// A fast-mode worker's local output buffer + running counts. Shared (via a
/// RefCell) between the emit path and the checkpoint callback so BOTH can flush.
struct Emitter {
    buf:   Vec<u8>,
    n:     u64,  // running emit count
    b:     u64,  // running byte count
    rep_n: u64,  // last counts published to telemetry
    rep_b: u64,
}

/// Flush a worker's buffered candidates to the shared sink and publish counts.
/// `durable`: also `flush_buffered()` the sink's 8 MB BufWriter down to the OS
/// (pipe/page-cache) so the bytes survive a process kill. The hot 256 KB emit
/// path passes `durable=false` (let the sink's BufWriter batch OS writes); the
/// checkpoint callback passes `durable=true` so the saved DFS position never sits
/// ahead of durably-written output — resume then re-tests an overlap instead of
/// skipping the unflushed window.
fn emitter_flush(
    e: &mut Emitter,
    sink: &std::sync::Mutex<Sink>,
    stats: &MergerStats,
    durable: bool,
) -> Result<()> {
    if e.buf.is_empty() && !durable { return Ok(()); }
    stats.writer_blocked.store(true, AtomicOrdering::Relaxed);
    {
        let mut s = sink.lock().unwrap();
        if !e.buf.is_empty() { s.write_raw(&e.buf)?; }
        if durable { s.flush_buffered()?; }
    }
    stats.writer_blocked.store(false, AtomicOrdering::Relaxed);
    if !e.buf.is_empty() {
        stats.emitted.fetch_add(e.n - e.rep_n, AtomicOrdering::Relaxed);
        stats.bytes_written.fetch_add(e.b - e.rep_b, AtomicOrdering::Relaxed);
        e.rep_n = e.n; e.rep_b = e.b;
        e.buf.clear();
    }
    Ok(())
}

fn expand_first_level(enum_model: &EnumModel, start_id: u32) -> Vec<HeapEntry> {
    let ctx = pack(start_id, start_id);
    let mut out: Vec<HeapEntry> = Vec::new();
    let mut seen: Vec<u32> = Vec::new();
    if let Some(children) = enum_model.trigram.get(&ctx) {
        for &(tok, lp) in children {
            let mut prefix = [0u32; 32];
            prefix[0] = tok;
            out.push(HeapEntry { log_prob: lp, prefix_len: 1, prefix });
            seen.push(tok);
        }
    }
    if enum_model.is_kn {
        if let Some(bi) = enum_model.bigram.get(&start_id) {
            let log_lam = enum_model.log_lambda.get(&ctx).copied().unwrap_or(0.0);
            for &(tok, lp_cont) in bi {
                if seen.contains(&tok) { continue; }
                let mut prefix = [0u32; 32];
                prefix[0] = tok;
                out.push(HeapEntry { log_prob: log_lam + lp_cont, prefix_len: 1, prefix });
            }
        }
    }
    out.sort_by(|a, b| b.log_prob.partial_cmp(&a.log_prob).unwrap_or(Ordering::Equal));
    out
}

fn assign_partitions(first_level: Vec<HeapEntry>, n: usize) -> Vec<Vec<HeapEntry>> {
    let mut parts: Vec<Vec<HeapEntry>> = (0..n).map(|_| Vec::new()).collect();
    for (rank, entry) in first_level.into_iter().enumerate() {
        parts[rank % n].push(entry);
    }
    parts
}

/// Distribute `first_level` tokens across `n_domains` buckets (round-robin by
/// rank) and return them as a `VecDeque` for use as a work-stealing queue.
/// Empty buckets are dropped. Each domain is a contiguous, non-overlapping
/// slice of the first-level token tree — callers process them independently
/// on separate channels so the global sorted-merge invariant is preserved.
#[allow(dead_code)] // kept for a possible future work-stealing redesign
fn build_work_queue(first_level: Vec<HeapEntry>, n_domains: usize) -> VecDeque<Vec<HeapEntry>> {
    let n = n_domains.max(1);
    let mut buckets: Vec<Vec<HeapEntry>> = (0..n).map(|_| Vec::new()).collect();
    for (rank, entry) in first_level.into_iter().enumerate() {
        buckets[rank % n].push(entry);
    }
    buckets.into_iter().filter(|b| !b.is_empty()).collect()
}

/// Partition seeds by their first token, distributing token-groups across N
/// threads. All seeds sharing a first token go to the same thread so their
/// subtrees stay disjoint with other threads' subtrees (which is what makes
/// the parallel level-sweep produce non-overlapping output).
///
/// Distribution is greedy: largest groups first, each placed on the
/// least-loaded thread (by total seed count).
fn partition_seeds_by_first_token(seeds: Vec<HeapEntry>, n: usize) -> Vec<Vec<HeapEntry>> {
    if n == 0 {
        return vec![seeds];
    }
    // Group by first token. Empty-prefix seeds go under key = 0 (shouldn't
    // happen in practice — build_seeds rejects zero-length entries — but
    // robust to that).
    let mut groups: FxHashMap<u32, Vec<HeapEntry>> = FxHashMap::default();
    for seed in seeds {
        let key = if seed.prefix_len > 0 { seed.prefix[0] } else { 0 };
        groups.entry(key).or_default().push(seed);
    }
    // Order groups by size descending so the bin-packing is reasonable.
    let mut group_vec: Vec<(u32, Vec<HeapEntry>)> = groups.into_iter().collect();
    group_vec.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));

    let mut parts: Vec<Vec<HeapEntry>> = (0..n).map(|_| Vec::new()).collect();
    let mut sizes: Vec<usize> = vec![0; n];
    for (_, members) in group_vec {
        // Place this group on the least-loaded thread.
        let target = sizes.iter().enumerate().min_by_key(|(_, s)| **s).unwrap().0;
        sizes[target] += members.len();
        parts[target].extend(members);
    }
    parts
}

// (Removed: temp-file merge is replaced by the streaming in-memory k-way
// merge in `run_merger` above. See progress sidecar + --resume for crash
// recovery instead of partial-file recovery.)

/// `--min-level` is a floor on the enumerator's level sweep. Reject, rather than
/// ignore, the two settings where it would silently do nothing: past `LEVEL_MAX`
/// the sweep is empty and the run emits not one candidate, and the graft
/// generator has no level sweep to floor at all.
fn validate_min_level(min_level: u32, run_graft: bool, float: bool) -> Result<()> {
    if min_level > LEVEL_MAX {
        bail!("--min-level ({}) exceeds the maximum level {} — nothing would be emitted",
            min_level, LEVEL_MAX);
    }
    if min_level > 0 && run_graft {
        bail!("--min-level has no effect with {} (that generator ranks seeds by \
               surprisal instead of sweeping levels)",
            if float { "--float" } else { "--prepend-only" });
    }
    Ok(())
}

fn run_generate(mut args: GenerateArgs) -> Result<()> {
    // Validate
    if args.bias <= 0.0 {
        bail!("--bias must be positive (got {})", args.bias);
    }
    if args.bias > 1000.0 {
        warn_msg(&format!("warning: --bias={} is very large; numerical sensitivity in renorm", args.bias));
    }

    // DEPRECATED: --skipgram-expand / --skipgram-direction. The old bigram-
    // continuation cosine-similarity expansion was retired — it captured
    // structural role rather than semantic theme, only perturbed the weighted-
    // bias set (a no-op for seeded mode), and carried edge-case bugs. Expand
    // seeds with separate tooling and feed the result to --wordlist instead. We
    // force the flag off and warn rather than erroring so old scripts still run.
    if args.skipgram_expand > 0 {
        warn_msg("warning: --skipgram-expand is DEPRECATED and now a no-op (ignored). \
                 Expand your seed list separately and pass the result to --wordlist.");
        args.skipgram_expand = 0;
    }
    if args.min_tokens < 1 {
        bail!("--min-tokens must be >= 1");
    }
    if args.min_tokens > args.max_tokens {
        bail!("--min-tokens ({}) cannot exceed --max-tokens ({})",
            args.min_tokens, args.max_tokens);
    }
    // A --wordlist defaults to APPEND-ONLY (seed stays the prefix, affixes
    // appended) — the seeded walk, routed through the seeded machinery. The
    // graft generator (rarity-weighted, affixes on either side) is opt-in via
    // --float; --prepend-only is the prepend graft. --mode is the legacy opt-in
    // (weighted/seeded/combined).
    if [args.append_only, args.prepend_only, args.float].iter().filter(|&&x| x).count() > 1 {
        bail!("--append-only, --prepend-only, and --float are mutually exclusive");
    }
    let run_graft = args.wordlist.is_some() && args.mode.is_none()
        && (args.prepend_only || args.float);
    validate_min_level(args.min_level, run_graft, args.float)?;
    // Concrete mode for the legacy machinery + the graft generator's unbiased
    // model prep. None → Seeded (unbiased; --append-only is exactly seeded).
    let legacy_mode: Mode = args.mode.unwrap_or(Mode::Seeded);

    if args.wordlist.is_none() {
        // Mode flag is meaningless without --wordlist. We still allow the
        // default to pass through (so a bare `tokenov generate` works), but
        // explicit non-default --mode without --wordlist is an error.
        // (Default value can't be distinguished from explicit-default in
        // clap derive; we treat any --mode without --wordlist as benign.)
    }

    // Resolve the algorithm variant from the --variant flag.
    let variant = variant::dispatch(&args.variant)?;
    log_msg(&format!("[gen] variant: {}", variant.name()));

    let model_path = match &args.model {
        Some(p) => registry::resolve_model(p)?,  // accept a registered name or a path
        None => default_model_path()?,
    };
    let mut model = model_load(&model_path)?;

    // Decode kind classification — for the bundled/default flow the model
    // file doesn't store decode kind, but the decode table was already built
    // method-aware at build time; for finalize_decoded we need to know the
    // kind. Heuristic: if any decode entry contains a leading space, this is
    // Plain (LLaMA-style); otherwise StripBpeSpace. This is informational only;
    // both finalizers strip leading whitespace at minimum.
    let kind = guess_kind_from_decode(&model.decode);

    // Phase 1: compute post-bias model + entry_seqs / w_set if wordlist.
    //
    // Standard-mode path (wordlist is None) is bit-identical to v1: we clone
    // the loaded model and prepare_enum_model runs over the unmodified KN
    // tiers. No bias / seed code is reachable.
    //
    // Wordlist-Some path: in addition to producing tokenized entries and the
    // W set, we materialize a `post_bias_model` whose `contexts`, `bigram_kn`,
    // and `lambda` reflect the bias (for weighted/combined) or are the
    // unmodified original (for seeded). For KN models in weighted/combined,
    // the bias is computed via `apply_weighted_bias_kn_aware` so the joint
    // emission distribution per context is properly renormalized to sum to 1.
    let (post_bias_model, entry_seqs_opt, _w_set): (Model,
                                                    Option<Vec<Vec<u32>>>,
                                                    FxHashMap<u32, ()>) = match args.wordlist.as_ref() {
        None => {
            // Standard mode — clone the loaded model unchanged. No bias.
            let m = Model {
                contexts: model.contexts.clone(),
                decode: model.decode.clone(),
                start_id: model.start_id,
                end_id: model.end_id,
                is_kn: model.is_kn,
                discount: model.discount,
                lambda: model.lambda.clone(),
                bigram_kn: model.bigram_kn.clone(),
                unigram_raw: model.unigram_raw.clone(),
                unigram_kn_cont: model.unigram_kn_cont.clone(),
            };
            (m, None, FxHashMap::default())
        }
        Some(p) => {
            let wordlist = read_wordlist(p)?;
            log_msg(&format!("[gen] wordlist: {} entries from {}", wordlist.len(), p.display()));
            // Wordlist must be tokenized with the same tokenizer used at build
            // time. v3 models embed it (self-describing); v1/v2 resolve it
            // out-of-band (env override → sidecar → default-alias). See
            // load_wordlist_tokenizer.
            let tokenizer = load_wordlist_tokenizer(&model_path, args.model.is_none())?;
            let entry_seqs: Vec<Vec<u32>> = wordlist.iter().map(|s| {
                tokenizer.encode(s.as_str(), false)
                    .map(|e| e.get_ids().iter().filter(|&&id| id < model.start_id).copied().collect::<Vec<_>>())
                    .unwrap_or_default()
            }).filter(|s| !s.is_empty()).collect();
            log_msg(&format!("[gen] wordlist tokenized: {} non-empty entries", entry_seqs.len()));

            let mut w_set: FxHashMap<u32, ()> = FxHashMap::default();
            for seq in &entry_seqs {
                for &id in seq { w_set.insert(id, ()); }
            }
            log_msg(&format!("[gen] wordlist token-set size: {}", w_set.len()));

            // Optional skipgram-style expansion via bigram-distribution
            // cosine similarity. Only the W set used for bias is expanded;
            // seeds (in seeded/combined modes) remain the literal wordlist
            // entries — see CLI flag docs for the rationale.
            let w_set_for_bias: FxHashMap<u32, ()> = if args.skipgram_expand > 0 {
                if !model.is_kn {
                    bail!("--skipgram-expand requires a v2/KN model (NGRMv002); \
                           the loaded model is v1. Rebuild via `tokenov model train`.");
                }
                let explicit: Vec<u32> = w_set.keys().copied().collect();
                let expanded_ids = expand_via_skipgram(&model, &explicit,
                    args.skipgram_expand, args.skipgram_direction);
                let mut s: FxHashMap<u32, ()> = FxHashMap::default();
                for id in expanded_ids { s.insert(id, ()); }
                s
            } else {
                w_set.clone()
            };
            log_msg(&format!("[gen] wordlist token-set size for bias: {} (explicit={}, +expansion={})",
                w_set_for_bias.len(),
                w_set.len(),
                w_set_for_bias.len().saturating_sub(w_set.len())));

            // Build post-bias model. For weighted/combined under KN, the
            // joint-distribution-aware bias collapses the trigram + bigram
            // tiers into a per-context single-tier distribution and produces a
            // pre-biased global bigram_kn for bigram-only contexts. Lambda is
            // emptied so the enumerator's bigram-fallback path runs with
            // log_lam = 0 (effective lambda = 1) for any context not in
            // post-bias trigram. For seeded mode (no bias), the model is
            // unchanged.
            let m = match legacy_mode {
                Mode::Weighted | Mode::Combined => {
                    log_msg(&format!("[gen] applying KN-aware bias={} (mode={:?})", args.bias, legacy_mode));
                    let (post_contexts, post_bigram) =
                        apply_weighted_bias_kn_aware(&model, &w_set_for_bias, args.bias);
                    Model {
                        contexts: post_contexts,
                        decode: model.decode.clone(),
                        start_id: model.start_id,
                        end_id: model.end_id,
                        // Keep is_kn=true so bigram-only contexts use the
                        // bigram-fallback path. Lambda is empty → log_lam=0.
                        is_kn: model.is_kn,
                        discount: model.discount,
                        lambda: FxHashMap::default(),
                        bigram_kn: post_bigram,
                        unigram_raw: model.unigram_raw.clone(),
                        unigram_kn_cont: model.unigram_kn_cont.clone(),
                    }
                }
                // Seeded (and the default graft generator) run the model
                // unbiased. The graft path branches out below before enumeration.
                Mode::Seeded => Model {
                    contexts: model.contexts.clone(),
                    decode: model.decode.clone(),
                    start_id: model.start_id,
                    end_id: model.end_id,
                    is_kn: model.is_kn,
                    discount: model.discount,
                    lambda: model.lambda.clone(),
                    bigram_kn: model.bigram_kn.clone(),
                    unigram_raw: model.unigram_raw.clone(),
                    unigram_kn_cont: model.unigram_kn_cont.clone(),
                },
            };
            (m, Some(entry_seqs), w_set)
        }
    };
    let mut enum_model = variant.prepare(&post_bias_model);

    // Seed-chunk graft generator: the DEFAULT for --wordlist (unless a
    // legacy --mode was requested, or --append-only). A self-contained path that
    // reads the model + wordlist and emits rarity-weighted combinations directly,
    // bypassing the Markov enumerator. Branch here (before post_bias_model is
    // dropped) so it can read the model's unigram/decode tiers.
    if run_graft {
        return match &entry_seqs_opt {
            Some(entry_seqs) => graft::run(&args, &enum_model, &post_bias_model, entry_seqs),
            None => unreachable!("run_graft implies wordlist present"),
        };
    }

    drop(post_bias_model);

    // Drop the heavy fields of `model` now. variant.prepare() transformed
    // model.contexts/bigram_kn/lambda into enum_model.trigram/bigram/log_lambda;
    // post_bias_model (a copy or transform of model) is already dropped. The
    // small fields (decode, start_id, end_id, is_kn, discount, unigram_*) stay
    // — they're read later by enumerate_to_sink and the worker loop.
    //
    // Saves ~225 B/ctx (contexts) + ~25 B/ctx (lambda) + a few MB for
    // bigram_kn. At the 100M-line training scale (13.83M ctxs) that's ~3.3 GB
    // freed BEFORE build_child_cache spawns its parallel workers, so the peak
    // memory during child_cache build is also lower.
    {
        let n_ctx_before = model.contexts.len();
        model.contexts = rustc_hash::FxHashMap::default();
        model.bigram_kn = rustc_hash::FxHashMap::default();
        model.lambda = rustc_hash::FxHashMap::default();
        log_msg(&format!(
            "[gen] dropped model.contexts + bigram_kn + lambda ({} ctxs) — covered by enum_model",
            n_ctx_before));
    }

    // Phase 3: derive seeds from wordlist (if seeded/combined) using the
    // EnumModel (so back-off is honored when computing joint log-probs).
    let seeds: Vec<HeapEntry> = match (&entry_seqs_opt, legacy_mode) {
        (Some(entry_seqs), Mode::Seeded) | (Some(entry_seqs), Mode::Combined) => {
            build_seeds(&enum_model, entry_seqs, &_w_set, args.seed_mode,
                model.start_id, args.max_tokens)
        }
        _ => Vec::new(),
    };
    log_msg(&format!("[gen] seeds: {}", seeds.len()));

    let target_count = args.count.unwrap_or(u64::MAX);
    let mut n_threads = args.threads.unwrap_or_else(rayon::current_num_threads).max(1);
    // Strict mode runs a single producer thread. The strict k-way merge is
    // inherently serial (N producers feed one merger thread), so extra threads
    // give NO throughput gain — measured slightly slower — while the approximate
    // ±chunk_size merge makes the fine order depend on producer timing / memory
    // footprint, so multithreaded strict is only reproducible per-file, not
    // canonical across models. Single-threaded strict is the exact global
    // rank order and is byte-reproducible. Clamping here routes strict through
    // the single-thread enumerator below (strict already forbids checkpoint/
    // resume, so it always takes that clean path).
    if args.strict && n_threads != 1 {
        if let Some(t) = args.threads {
            if t > 1 {
                log_msg(&format!(
                    "[gen] --strict runs single-threaded (ignoring --threads {t}): the global \
                     rank-merge is serial, so more threads give no speedup, and only \
                     single-threaded output is the canonical byte-reproducible order."));
            }
        }
        n_threads = 1;
    }
    log_msg(&format!(
        "[gen] enumerating: max_tokens={} min_len={} max_len={} count={:?} kn={} threads={}",
        args.max_tokens, args.min_len, args.max_len, args.count, enum_model.is_kn,
        n_threads));

    // Build shared child cache once; all threads share it via Arc.
    // Covers all trigram contexts (~6.6 M for the llama model); threads use a
    // small local fallback for the rare KN bigram-only misses.
    //
    // FULL builds the whole cache; BOUNDED/LAZY hand the workers
    // an EMPTY shared cache and recompute children on demand (BOUNDED memoizes
    // hot ones per-thread). Trades the ~27 GB FULL cache (COMB, ~16.8M ctx) for a
    // sized partial cache + CPU, or ~0 extra RAM in LAZY. Output is byte-identical
    // in all modes. `decide_*` picks the mode automatically from projected cache
    // size vs the RAM budget (manual flags override), so a large-context model
    // never OOMs without a flag.
    let cache_mode = decide_child_cache_mode(&args, &enum_model, n_threads);
    let child_cache: ChildCache = match cache_mode {
        CacheMode::Full => build_child_cache(&enum_model, &*variant),
        CacheMode::Bounded(_) | CacheMode::Lazy => Box::leak(Box::new(ChildCacheMap::default())),
    };
    log_msg(&format!("[gen] child_cache: {} ctx (mode={:?})", child_cache.len(), cache_mode));
    // NOTE: enum_model.trigram and enum_model.log_lambda are dropped further
    // down, AFTER expand_first_level has had a chance to read trigram in the
    // seeds-empty branch. See the "DROP TRIGRAM" block below.

    // Progress sidecar path. Only meaningful when we have a file output;
    // stdout has nothing to attach to. With plain-text-only output (since
    // the .7z mode was removed), `--resume` works uniformly whenever a
    // sidecar exists.
    let progress_path: Option<PathBuf> = args.output.as_ref().map(|p| {
        let mut s = p.as_os_str().to_owned();
        s.push(".progress");
        PathBuf::from(s)
    });

    // Args fingerprint: must match between the saved progress sidecar and the
    // current invocation for `--resume` to be safe (otherwise the merged
    // candidate sequence wouldn't be deterministic and skip-N would produce
    // garbage).
    let args_fp = build_args_fingerprint(&args, &model_path, n_threads)?;

    // ── Resume / checkpoint resolution. Two mechanisms:
    //   • SIDECAR (strict / single-thread clean path): re-runs the deterministic
    //     stream, skips the first N already-emitted, appends. Needs --output.
    //   • CHECKPOINT (fast mode): each worker saves its DFS position; resume
    //     reconstructs and continues. Works with stdout (the piped `| hashcat` case).
    // Fast mode checkpoints by DEFAULT to a rolling state file so any interrupted run
    // is resumable without forethought; --checkpoint-file overrides the path;
    // --no-checkpoint opts out. Strict mode leaves both effective values None and
    // keeps its sidecar behavior untouched.
    if args.fast && args.strict {
        bail!("--fast and --strict are mutually exclusive (fast is the default)");
    }
    let explicit_resume: Option<PathBuf> = args.resume_state.clone();
    let user_named_ckpt = args.checkpoint_file.is_some() || explicit_resume.is_some();
    // Where to WRITE checkpoints — a rolling state file so any interrupted run is
    // resumable without forethought (both fast and strict; None only with
    // --no-checkpoint). Strict is single-threaded, so its checkpoint is a single
    // DFS-position slot restored in O(depth) on resume — same machinery as a fast
    // worker, no re-enumeration from candidate 0.
    let checkpoint_to: Option<PathBuf> = if args.no_checkpoint {
        None
    } else {
        Some(args.checkpoint_file.clone()
            .or_else(|| explicit_resume.clone())
            .unwrap_or_else(default_checkpoint_path))
    };
    // Where to READ a resume from: explicit --resume-state, else --resume resolves
    // to the checkpoint we'd write to (named or default).
    let resume_from: Option<PathBuf> = explicit_resume
        .or_else(|| if args.resume { checkpoint_to.clone() } else { None });
    // The SIDECAR resume path (re-run + skip-N) applies only when no checkpoint
    // resume does — i.e. --no-checkpoint --resume. This is what the skip-N block
    // below keys on.
    let sidecar_resume = args.resume && resume_from.is_none();
    // Checkpoint fingerprint: binds model+args+threads (args_fp) AND the tokenov
    // version (enumeration order is version-sensitive, so resuming a saved DFS
    // stack under a different binary would corrupt it). Used for the checkpoint
    // file in both fast and strict modes; the progress sidecar keeps args_fp.
    let ckpt_fp = format!("{} tokenov={}", args_fp, env!("CARGO_PKG_VERSION"));
    // Rolling default checkpoint (not a user-named file) → remove on clean completion.
    let ckpt_is_default = checkpoint_to.is_some() && !user_named_ckpt;

    // Strict checkpoint resume (single-thread O(depth) restore). When resuming a
    // strict run, load the saved DFS position from the checkpoint file; the
    // enumerator jumps to it and emits only *new* candidates, so nothing is
    // skipped at the sink (skip_first stays 0). The paired byte offset (progress
    // sidecar, written atomically with the checkpoint) truncates the output to a
    // clean boundary. None for fresh runs, non-strict runs, and the sidecar path.
    let strict_ckpt_resume: Option<ThreadCkpt> = if args.strict && resume_from.is_some() {
        let rp = resume_from.as_ref().unwrap();
        if !rp.exists() {
            bail!("--resume: no checkpoint at {} — run without --resume to start fresh",
                rp.display());
        }
        let cf = read_checkpoint(rp)?;
        if cf.fingerprint != ckpt_fp {
            bail!("--resume: checkpoint fingerprint mismatch, refusing to resume.\n  \
                   saved:   {}\n  current: {}\n  \
                   (model, args, and tokenov version must all match)",
                   cf.fingerprint, ckpt_fp);
        }
        match cf.slots.into_iter().next() {
            Some(CkptSlot::InProgress(c)) => Some(c),
            Some(CkptSlot::Done) => {
                log_msg("[gen] previous strict run already complete — nothing to resume");
                return Ok(());
            }
            _ => None, // NotStarted / empty → start fresh
        }
    } else { None };
    // --resume wins over --min-level: a checkpoint already encodes a start
    // position, and its target_level can only sit at or past the floor the run
    // began with (--min-level is part of the fingerprint, which must match). A
    // checkpoint below the floor therefore means a hand-edited or foreign state
    // file — refuse rather than silently re-emit the skipped shells.
    if let Some(c) = &strict_ckpt_resume {
        if c.target_level < args.min_level {
            bail!("--resume: checkpoint is at level {} but --min-level is {} — \
                   resuming it would re-emit shells below the floor",
                c.target_level, args.min_level);
        }
    }

    // Compute (skip_first, initial_bytes) from the progress sidecar if resuming.
    let (skip_first, initial_bytes): (u64, u64) = if strict_ckpt_resume.is_some() {
        // Byte side of the paired checkpoint: read the offset the progress sidecar
        // recorded at the same `emitted` as the saved DFS position, and truncate
        // the output to it. The DFS restore (above) handles skipping — skip_first=0.
        let pp = progress_path.as_ref().ok_or_else(|| anyhow!(
            "--resume requires --output FILE (no sidecar exists for stdout)"))?;
        if !pp.exists() {
            bail!("--resume: checkpoint present but no progress sidecar at {} — \
                   run without --resume to start fresh", pp.display());
        }
        let saved = read_progress(pp)?;
        if saved.fingerprint != args_fp {
            bail!("--resume: args fingerprint mismatch, refusing to append.\n  \
                   saved:   {}\n  current: {}", saved.fingerprint, args_fp);
        }
        truncate_output_to(args.output.as_ref().unwrap(), saved.byte_offset)?;
        log_msg(&format!("[gen] resume: restoring DFS position at {} candidates ({}B)",
            saved.emitted, saved.byte_offset));
        (0, saved.byte_offset)
    } else if sidecar_resume {
        let pp = progress_path.as_ref().ok_or_else(|| anyhow!(
            "--resume requires --output FILE (no sidecar exists for stdout)"))?;
        if !pp.exists() {
            bail!("--resume: no progress sidecar at {} — \
                   run without --resume to start fresh", pp.display());
        }
        let saved = read_progress(pp)?;
        if saved.fingerprint != args_fp {
            bail!("--resume: args fingerprint mismatch, refusing to append.\n\
                   saved:   {}\n  current: {}\n\
                   Re-run without --resume to start a fresh output.",
                   saved.fingerprint, args_fp);
        }
        // Truncate the output to byte_offset — the on-disk file may be longer than
        // `emitted` lines (BufWriter spilled past the last progress write).
        truncate_output_to(args.output.as_ref().unwrap(), saved.byte_offset)?;
        log_msg(&format!("[gen] resume: skipping first {} candidates from merged stream",
            saved.emitted));
        (saved.emitted, saved.byte_offset)
    } else if let Some(pp) = &progress_path {
        // Fresh run — wipe any stale progress sidecar from a prior aborted
        // run so a future --resume doesn't accidentally pick up wrong state.
        let _ = std::fs::remove_file(pp);
        (0, 0)
    } else { (0, 0) };

    // Per-token case shaping. Empty = default single lowercase emission
    // (byte-identical to the prior path).
    let case_masks: Vec<CaseMask> = match &args.case_shape {
        Some(spec) => {
            let m = CaseMask::parse_spec(spec).map_err(|e| anyhow::anyhow!(e))?;
            log_msg(&format!("[gen] case-shape: {} pattern(s): [{}]", m.len(),
                m.iter().map(|x| x.label.as_str()).collect::<Vec<_>>().join(", ")));
            m
        }
        None => Vec::new(),
    };

    // Single-thread direct-write path — no channels. This is the exact global
    // rank order and is byte-reproducible. Strict is clamped to one thread and
    // always routes here (never through the merger, which would change the
    // bytes). A non-strict 1-thread run with no checkpoint/resume lands here too
    // (the hooks below are inert then); a non-strict 1-thread run that wants a
    // checkpoint goes through the multi-thread path, which handles n_threads==1.
    //
    // Resume is O(depth): the checkpoint restores the saved DFS position (prefix
    // + idx_stack) and continues, exactly as a fast worker does — no
    // re-enumeration from candidate 0. The checkpoint file (DFS position) and the
    // progress sidecar (byte offset) are written together in on_checkpoint, so
    // they pair at the same `emitted`; resume restores the position and truncates
    // the output to the paired offset. Single-thread + a synchronous callback is
    // what makes that capture atomic.
    if n_threads == 1 && (args.strict || (checkpoint_to.is_none() && resume_from.is_none())) {
        let resuming = sidecar_resume || strict_ckpt_resume.is_some();
        let sink = if resuming {
            Sink::open_append(args.output.as_ref().unwrap())?
        } else {
            // Fresh run: wipe any stale checkpoint so a later --resume can't read a
            // position from a prior aborted run (the progress sidecar is wiped above).
            if let Some(cp) = &checkpoint_to { let _ = std::fs::remove_file(cp); }
            Sink::open(args.output.as_deref())?
        };
        let initial_states: Vec<HeapEntry> = if seeds.is_empty() {
            vec![HeapEntry { log_prob: 0.0, prefix_len: 0, prefix: [0u32; 32] }]
        } else { seeds };

        // Checkpoint cadence: Some only when a checkpoint file is in play (strict).
        // When None, on_checkpoint never fires and progress is written inline
        // (sidecar / --no-checkpoint mode, byte-identical to the prior path).
        let ckpt_every = checkpoint_to.as_ref()
            .map(|_| Duration::from_secs(args.checkpoint_secs.max(1)));
        let progress_inline = ckpt_every.is_none();

        // Sink + running byte count shared between the emit path and the
        // checkpoint callback (single thread ⇒ RefCell, no lock).
        let shared = std::cell::RefCell::new((sink, initial_bytes));
        let mut emitted: u64 = 0;
        let mut last_progress = Instant::now();
        let result = enumerate_to_sink(
            &enum_model, &*variant, child_cache, cache_mode, &model.decode, kind,
            model.start_id, model.end_id, initial_states,
            args.max_tokens, args.min_tokens, args.min_len, args.max_len,
            args.min_level,
            target_count,
            args.enterprise,
            &case_masks,
            |_lvl, _sk, bytes| {
                {
                    let mut g = shared.borrow_mut();
                    if emitted >= skip_first {
                        g.0.write_line(bytes)?;
                        g.1 += bytes.len() as u64 + 1;
                    }
                }
                emitted += 1;
                if progress_inline
                    && emitted % 1_000_000 == 0
                    && last_progress.elapsed().as_secs() >= 1
                {
                    if let Some(pp) = &progress_path {
                        let mut g = shared.borrow_mut();
                        g.0.flush_buffered()?;
                        let _ = write_progress(pp, emitted, g.1, &args_fp);
                    }
                    last_progress = Instant::now();
                }
                Ok(())
            },
            "",
            strict_ckpt_resume,
            ckpt_every,
            // Atomic paired capture: flush the sink, then persist the DFS position
            // (checkpoint file, version-bound fingerprint) and the byte offset
            // (progress sidecar) for the *same* emitted count.
            |ck: &ThreadCkpt| {
                let bytes = {
                    let mut g = shared.borrow_mut();
                    let _ = g.0.flush_buffered();
                    g.1
                };
                if let Some(cp) = &checkpoint_to {
                    let _ = write_checkpoint(cp, &ckpt_fp,
                        &[CkptSlot::InProgress(ck.clone())]);
                }
                if let Some(pp) = &progress_path {
                    let _ = write_progress(pp, ck.emitted, bytes, &args_fp);
                }
            },
        );
        let (sink, _) = shared.into_inner();
        sink.finish()?;
        if result.is_ok() {
            if let Some(pp) = &progress_path { let _ = std::fs::remove_file(pp); }
            // Rolling default checkpoint → remove on clean completion; a
            // user-named one → mark Done so a later --resume reports "complete".
            if let Some(cp) = &checkpoint_to {
                if ckpt_is_default {
                    let _ = std::fs::remove_file(cp);
                } else {
                    let _ = write_checkpoint(cp, &ckpt_fp, &[CkptSlot::Done]);
                }
            }
        }
        return result;
    }

    // Parallel path: domain-decomp producers → bounded channels → in-memory
    // k-way merge → sink. Sink is stdout or a plain file — the merger
    // writes uniformly to either.
    //
    // Work-stealing design: we build 4×N domains from the first-level token
    // list. Each domain gets its own channel so the per-channel sorted invariant
    // (required by the k-way merger) is preserved even when a single worker
    // thread processes multiple domains sequentially. Workers pull (states, tx)
    // pairs from a shared Mutex<VecDeque>; a thread that finishes its domain
    // early picks up the next one rather than going idle.
    //
    // Memory-pressure soft retirement: when the mem_monitor detects RSS > soft
    // (Soft thread retirement is inert in the reverted static-spawn path — workers
    // run one partition to completion, so there is no per-domain pull point to
    // retire at. The hard OOM abort via the mem_monitor's abort_flag still applies.)

    // Build first_level once (needed by both calibration and run_generate's
    // own work). expand_first_level reads enum_model.trigram, so it MUST run
    // before the trigram drop below. In seed mode we skip it (seeds are the
    // domains).
    let first_level: Vec<HeapEntry> = if seeds.is_empty() {
        expand_first_level(&enum_model, model.start_id)
    } else {
        Vec::new()
    };

    // DROP TRIGRAM (memory). At this point:
    //   - build_child_cache snapshotted every (a,b) ctx into child_cache.
    //   - expand_first_level (the only other reader of enum_model.trigram in
    //     the gen path) has already run.
    //   - Worker-thread enumerate_to_sink calls variant.get_children() only
    //     for cache misses, which are by construction NOT in trigram — the
    //     `em.trigram.get(&ctx)` check inside get_children() returns None
    //     for those whether trigram is populated or empty.
    //   - log_lambda is read for cache-miss ctxs too; those ctxs have no
    //     log_lambda entry either (log_lambda is built only for trigram ctxs),
    //     so dropping it is equivalent to .get() always returning None →
    //     fallback default 0.
    // Net: ~225 B (trigram) + ~25 B (log_lambda) per ctx freed. At 13.83M ctxs
    // (100M-line model) that's ~3.4 GB freed before resolve_chunk_size /
    // calibration runs — so the calibration spike doesn't double-pay.
    // In BOUNDED/LAZY mode the shared child_cache is EMPTY and
    // workers read enum_model.trigram/log_lambda for EVERY context via
    // get_children, so these tiers must stay resident — drop them only in FULL
    // mode (where the cache covers them). The coverage-bail below would also trip
    // on an empty cache, so it's inside the FULL branch.
    if cache_mode == CacheMode::Full {
        let n_tri = enum_model.trigram.len();
        let n_cache = child_cache.len();
        if n_tri != n_cache {
            anyhow::bail!(
                "internal: child_cache covers {} of {} trigram ctxs; refusing to drop trigram",
                n_cache, n_tri);
        }
        enum_model.trigram = rustc_hash::FxHashMap::default();
        enum_model.log_lambda = rustc_hash::FxHashMap::default();
        log_msg(&format!(
            "[gen] dropped enum_model.trigram + log_lambda ({} ctxs covered by child_cache)",
            n_cache));
    } else {
        log_msg(&format!(
            "[gen] {:?}: keeping enum_model.trigram + log_lambda ({} ctxs) resident \
             for on-demand child computation", cache_mode, enum_model.trigram.len()));
    }

    // Arc the now-settled enum_model and extract small fields. This is hoisted
    // BEFORE resolve_chunk_size so calibration (if triggered) can share the
    // same structures — no second model_load + variant.prepare + build_child_cache.
    let enum_model    = Arc::new(enum_model);
    let decode_table  = Arc::new(model.decode.clone());
    let start_id      = model.start_id;
    let end_id        = model.end_id;

    // Static partition: exactly one partition per worker (n_threads), each
    // assigned deterministically to its own channel below. REVERTED the
    // 4×n_threads work-stealing queue:
    // its non-deterministic pull could put 2 domains on one channel (breaking the
    // per-channel level-sort the k-way merger requires) and 0 on another, which
    // corrupted probability order and HALVED small-budget crack rate (1e7 rockyou
    // 12.03% → 6.64%). Static spawn (one partition ⇒ one level-sorted channel) is
    // 0.3.0's known-good design.
    let (partitions, log_header): (Vec<Vec<HeapEntry>>, String) = if seeds.is_empty() {
        let h = format!("[gen] domain decomp: {} first-level tokens → {} static partitions",
            first_level.len(), n_threads);
        (assign_partitions(first_level.clone(), n_threads), h)
    } else {
        let h = format!("[gen] domain decomp: {} seeds → {} static partitions",
            seeds.len(), n_threads);
        (partition_seeds_by_first_token(seeds, n_threads), h)
    };
    log_msg(&log_header);

    // Per-worker quota. Each worker tracks its own emission count locally
    // (a plain u64 in the closure — no shared atomic, no cache contention).
    // total ≈ n_active × thread_target = target_count (slight overshoot by
    // at most n_active candidates at boundaries, which is fine).
    //
    // Divide by the number of NON-EMPTY partitions, not n_threads: in seeded
    // mode `partition_seeds_by_first_token` leaves a partition empty whenever
    // there are fewer distinct first-tokens than threads, and an empty
    // partition's worker emits nothing. Dividing by n_threads there throttled
    // total output to (n_nonempty / n_threads) × target_count — e.g. a 1-seed
    // wordlist on 8 threads emitted target/8. In standard mode
    // assign_partitions fills all n_threads partitions, so n_active == n_threads
    // and this is byte-identical to the previous behaviour.
    let n_active = partitions.iter().filter(|p| !p.is_empty()).count().max(1);
    let thread_target = target_count.div_ceil(n_active as u64);

    // active_target: how many workers should remain alive. Starts at n_threads;
    // the mem_monitor decrements it on soft-RSS breach; workers >= active_target
    // retire after finishing their current domain.
    let active_target = Arc::new(AtomicUsize::new(n_threads));

    let sink = if sidecar_resume {
        Sink::open_append(args.output.as_ref().unwrap())?
    } else {
        Sink::open(args.output.as_deref())?
    };

    // Channels: N total, one per worker (not per domain). A worker writes all
    // its domains to its own channel. The per-chunk sort invariant is preserved
    // by flushing the ChunkSender between domains so no chunk crosses a boundary.
    //
    // Resolution order for chunk_size:
    //   1. --merge-chunk-size N (explicit) → use N
    //   2. <model>.ngram.tune.toml sidecar exists + valid → use cached K
    //   3. --no-auto-tune set → use DEFAULT_MERGE_CHUNK_SIZE
    //   4. Otherwise → run inline calibration, write sidecar, use result
    // For (4): we pass our already-built CalibSetup so calibration shares the
    // same model/enum_model/child_cache instead of building duplicates.
    let calib_setup = CalibSetup {
        enum_model:   Arc::clone(&enum_model),
        child_cache,
        kind,
        start_id, end_id,
        decode_table: Arc::clone(&decode_table),
        first_level,
    };
    let initial_chunk_size = resolve_chunk_size(&model_path, &args, cache_mode, Some(calib_setup))?;
    let chunk_size_atomic = Arc::new(AtomicUsize::new(initial_chunk_size));
    let channel_chunks = (MERGE_CHANNEL_BUFFER_ITEMS / initial_chunk_size).max(2);
    log_msg(&format!(
        "[gen] merge: chunk_size={} channel_capacity={} chunks/channel ({} items/channel)",
        initial_chunk_size, channel_chunks, channel_chunks * initial_chunk_size));

    // N channels — one per worker thread. Workers write all their domains to
    // their own channel; the channel stays open until the worker exits.
    // The merger's blocking pre-fetch (one recv per channel) is compatible
    // because every worker will produce ≥1 chunk unless quota fires before
    // the first emit (in which case the channel closes immediately and the
    // merger treats it as an empty source).
    //
    // Work queue holds only domain states (no per-domain senders). Workers
    // flush their ChunkSender between domains to ensure no chunk spans a
    // boundary — preserving the per-chunk sort invariant the k-way merge needs.
    let mut receivers: Vec<Receiver<MergeChunk>> = Vec::with_capacity(n_threads);
    let mut worker_senders: Vec<(Sender<MergeChunk>, usize)> = Vec::with_capacity(n_threads);
    for i in 0..n_threads {
        let (tx, rx) = channel_bounded::<MergeChunk>(channel_chunks);
        worker_senders.push((tx, i));
        receivers.push(rx);
    }
    // enum_model, decode_table, start_id, end_id are already in scope from the
    // earlier hoist (just before resolve_chunk_size). Continuing with worker setup.
    let max_tokens    = args.max_tokens;
    let min_tokens    = args.min_tokens;
    let min_len       = args.min_len;
    let max_len       = args.max_len;
    let min_level     = args.min_level;
    let enterprise    = args.enterprise;

    // (--fast/--strict and strict+checkpoint validation, plus checkpoint_to/resume_from
    // resolution, done up front near the args fingerprint.)

    // ── Mode select. FAST (default): no global merge — each thread writes its
    // partition directly to a shared sink (batched, brief lock per ~256 KB),
    // N-way interleaved. Removes the single-merger bottleneck (~3.4× at 8
    // threads). STRICT (--strict): globally rank-ordered, byte-reproducible —
    // forced single-threaded (n_threads clamped to 1 above), so it takes the
    // single-thread enumerator path, NOT this multi-thread merger. (A
    // multithreaded strict merge is only reproducible per-file, not canonical
    // across models, because its ±chunk_size approximate merge order shifts with
    // producer timing / memory footprint; and it is no faster, since the merge is
    // serial.) Identical candidate SET either way; only the order differs. Fast
    // mode is inherently low-memory (≈256 KB/thread of output buffer, no chunk
    // arena), so it skips the merged path's chunk-pressure mem-monitor.
    if !args.strict {
        // ── Crash-resume wiring. `ckpt_fp` (defined above) binds model+args+threads
        // and the tokenov version — enumeration order is version-sensitive, so
        // resuming with a different binary would corrupt the saved idx_stack.
        let ckpt_path = checkpoint_to.clone();
        let ckpt_secs = args.checkpoint_secs.max(1);
        let resume_slots: Vec<CkptSlot> = if let Some(rp) = &resume_from {
            if !rp.exists() {
                bail!("--resume: no checkpoint at {} — nothing to resume. \
                       Run without --resume to start fresh.", rp.display());
            }
            let cf = read_checkpoint(rp)?;
            if cf.fingerprint != ckpt_fp {
                bail!("--resume-state: fingerprint mismatch, refusing to resume.\n  \
                       saved:   {}\n  current: {}\n  \
                       (model, args, --threads, and tokenov version must all match)",
                    cf.fingerprint, ckpt_fp);
            }
            if cf.slots.len() != n_threads {
                bail!("--resume-state: checkpoint has {} thread slots but --threads={}; \
                       pin --threads to the checkpointed value", cf.slots.len(), n_threads);
            }
            let n_done = cf.slots.iter().filter(|s| matches!(s, CkptSlot::Done)).count();
            let n_prog = cf.slots.iter().filter(|s| matches!(s, CkptSlot::InProgress(_))).count();
            log_msg(&format!("[gen] resuming from {} ({} threads: {} done, {} in-progress)",
                rp.display(), cf.slots.len(), n_done, n_prog));
            if n_done == cf.slots.len() {
                log_msg("[gen] previous run already complete — nothing to resume");
            }
            cf.slots
        } else {
            vec![CkptSlot::NotStarted; n_threads]
        };
        // Per-worker resume state + the live slots the writer thread serializes.
        let worker_resume: Vec<Option<ThreadCkpt>> = resume_slots.iter().map(|s| match s {
            CkptSlot::InProgress(c) => Some(c.clone()),
            _ => None,
        }).collect();
        let worker_done: Vec<bool> =
            resume_slots.iter().map(|s| matches!(s, CkptSlot::Done)).collect();
        // Same floor check as the strict path, over EVERY worker slot — checked
        // here, before any worker spawns, so a bad state file can't leave a run
        // half-started.
        for (i, ck) in worker_resume.iter().enumerate() {
            let Some(c) = ck else { continue };
            if c.target_level < args.min_level {
                bail!("--resume: worker {} checkpoint is at level {} but --min-level is {} — \
                       resuming it would re-emit shells below the floor",
                    i, c.target_level, args.min_level);
            }
        }
        let slots: Arc<Vec<std::sync::Mutex<CkptSlot>>> =
            Arc::new(resume_slots.into_iter().map(std::sync::Mutex::new).collect());
        let ckpt_stop = Arc::new(AtomicBool::new(false));

        // Session registry: a checkpointed run is a resumable "session".
        // On resume, continue the existing session record (match by checkpoint file)
        // rather than spawning a duplicate; otherwise create a fresh one.
        let session_id: Option<String> = ckpt_path.as_ref().and_then(|cp| {
            if resume_from.is_some() {
                if let Some(mut rec) = load_sessions().into_iter()
                    .find(|r| Path::new(&r.checkpoint_file) == cp.as_path())
                {
                    rec.pid = std::process::id();
                    rec.status = "running".into();
                    rec.updated_at = now_secs();
                    let _ = write_session(&rec);
                    return Some(rec.id);
                }
            }
            session_start(&model_path, args.count, n_threads, cp)
        });

        let telemetry_enabled = args.stats_interval_ms > 0;
        // Always materialize stats so the final total is accurate even with
        // telemetry off; the telemetry thread is only spawned when enabled.
        let stats = Arc::new(MergerStats::default());
        let stats_stop = Arc::new(AtomicBool::new(false));
        let stats_csv_path: Option<PathBuf> = if telemetry_enabled {
            match (&args.stats_csv, &args.output) {
                (Some(p), _)              => Some(p.clone()),
                (None,    Some(out_path)) => {
                    let mut s = out_path.as_os_str().to_owned();
                    s.push(".stats.csv");
                    Some(PathBuf::from(s))
                }
                (None, None) => None,
            }
        } else { None };

        let sink = Arc::new(std::sync::Mutex::new(sink));
        let scope_result: Result<()> = std::thread::scope(|scope| {
            // Telemetry thread — same [stats] / --json / back-pressure as merged.
            let telemetry_handle = if telemetry_enabled {
                let s        = Arc::clone(&stats);
                let stop     = Arc::clone(&stats_stop);
                let csv      = stats_csv_path.clone();
                let interval = Duration::from_millis(args.stats_interval_ms);
                let v  = verbose();
                let pe = Duration::from_secs(args.progress_secs);
                let j  = args.json;
                Some(scope.spawn(move || run_telemetry(s, stop, interval, csv, v, pe, j)))
            } else { None };

            // Checkpoint-writer thread: every ckpt_secs, snapshot all worker slots
            // and atomically write the combined checkpoint file. Sleeps in short
            // increments so it shuts down promptly when workers finish.
            let ckpt_writer = ckpt_path.as_ref().map(|cp| {
                let slots_r = Arc::clone(&slots);
                let stop    = Arc::clone(&ckpt_stop);
                let path    = cp.clone();
                let fp      = ckpt_fp.clone();
                let secs    = ckpt_secs;
                let sid     = session_id.clone();
                let sstats  = Arc::clone(&stats);
                scope.spawn(move || {
                    let snapshot = |slots: &[std::sync::Mutex<CkptSlot>]| -> Vec<CkptSlot> {
                        slots.iter().map(|m| m.lock().unwrap().clone()).collect()
                    };
                    loop {
                        let mut waited = 0u64;
                        while waited < secs * 1000 {
                            if stop.load(AtomicOrdering::Relaxed) { break; }
                            std::thread::sleep(Duration::from_millis(200));
                            waited += 200;
                        }
                        let _ = write_checkpoint(&path, &fp, &snapshot(&slots_r));
                        if let Some(id) = &sid {
                            session_update(id, sstats.emitted.load(AtomicOrdering::Relaxed), "running");
                        }
                        if stop.load(AtomicOrdering::Relaxed) { break; }
                    }
                    // Final flush reflects the terminal state (all Done on clean exit).
                    let _ = write_checkpoint(&path, &fp, &snapshot(&slots_r));
                })
            });

            // One worker per partition, each writing directly to the shared sink.
            let mut handles = Vec::with_capacity(n_threads);
            for (thread_id, states) in partitions.into_iter().enumerate() {
                let em    = Arc::clone(&enum_model);
                let cc    = child_cache;
                let dt    = Arc::clone(&decode_table);
                let var   = Arc::clone(&variant);
                let cm    = case_masks.clone();
                let sink  = Arc::clone(&sink);
                let stats = Arc::clone(&stats);
                let my_resume = worker_resume[thread_id].clone();
                let done      = worker_done[thread_id];
                let slots_w   = Arc::clone(&slots);
                let cp_every  = ckpt_path.as_ref().map(|_| Duration::from_secs(ckpt_secs));
                let ckpt_on   = ckpt_path.is_some();
                handles.push(scope.spawn(move || -> Result<()> {
                    let label = format!("[t{}] ", thread_id);
                    if done {
                        // This partition was already exhausted before the checkpointed
                        // crash — nothing left to emit.
                        return Ok(());
                    }
                    const FLUSH: usize = 256 * 1024;
                    // Emit buffer + counts, shared with the checkpoint callback so it
                    // can flush durably before recording the DFS position.
                    let em_cell = std::cell::RefCell::new(Emitter {
                        buf: Vec::with_capacity(FLUSH + 1024), n: 0, b: 0, rep_n: 0, rep_b: 0,
                    });
                    let res = enumerate_to_sink(
                        &em, &*var, cc, cache_mode, &dt, kind, start_id, end_id,
                        states, max_tokens, min_tokens, min_len, max_len, min_level, thread_target, enterprise, &cm,
                        |_lvl, _sk, bytes| {
                            let mut e = em_cell.borrow_mut();
                            e.buf.extend_from_slice(bytes);
                            e.buf.push(b'\n');
                            e.n += 1;
                            e.b += bytes.len() as u64 + 1;
                            if e.buf.len() >= FLUSH {
                                emitter_flush(&mut e, &sink, &stats, false)?; // hot path: no OS flush
                            }
                            Ok(())
                        },
                        &label,
                        my_resume,
                        cp_every,
                        |ck: &ThreadCkpt| {
                            // Flush buffered output down to the OS BEFORE recording the
                            // position, so the checkpoint never sits ahead of durable
                            // output (resume overlaps instead of skipping).
                            let _ = emitter_flush(&mut em_cell.borrow_mut(), &sink, &stats, true);
                            *slots_w[thread_id].lock().unwrap() = CkptSlot::InProgress(ck.clone());
                        },
                    );
                    if res.is_ok() {
                        // Push the final partial buffer (the outer scope flushes the
                        // sink to the OS after all workers join).
                        emitter_flush(&mut em_cell.borrow_mut(), &sink, &stats, false)?;
                    }
                    // Publish any residual counts left after the last flush.
                    {
                        let mut e = em_cell.borrow_mut();
                        stats.emitted.fetch_add(e.n - e.rep_n, AtomicOrdering::Relaxed);
                        stats.bytes_written.fetch_add(e.b - e.rep_b, AtomicOrdering::Relaxed);
                        e.rep_n = e.n; e.rep_b = e.b;
                    }
                    // Partition exhausted → mark Done so a later resume skips it.
                    if res.is_ok() && ckpt_on {
                        *slots_w[thread_id].lock().unwrap() = CkptSlot::Done;
                    }
                    res
                }));
            }

            let mut worker_errors: Vec<anyhow::Error> = Vec::new();
            for h in handles {
                match h.join() {
                    Err(_)     => worker_errors.push(anyhow!("worker panicked")),
                    Ok(Err(e)) => worker_errors.push(e),
                    Ok(Ok(())) => {}
                }
            }
            stats_stop.store(true, AtomicOrdering::Relaxed);
            if let Some(h) = telemetry_handle { let _ = h.join(); }
            // Stop the checkpoint writer; its final flush records the terminal state
            // (all Done on a clean finish; the last in-progress slots otherwise).
            ckpt_stop.store(true, AtomicOrdering::Relaxed);
            if let Some(h) = ckpt_writer { let _ = h.join(); }
            // Mark the session done on a clean finish; on error leave it "running" so
            // --sessions reports it interrupted (its pid is gone) and it stays resumable.
            if let Some(id) = &session_id {
                if worker_errors.is_empty() {
                    session_update(id, stats.emitted.load(AtomicOrdering::Relaxed), "done");
                }
            }
            if !worker_errors.is_empty() {
                return Err(worker_errors.remove(0));
            }
            Ok(())
        });
        scope_result?;
        sink.lock().unwrap().flush_buffered()?;
        // Clean completion: drop the rolling DEFAULT checkpoint (nothing left to
        // resume). A user-named --checkpoint-file is left as a Done record.
        if ckpt_is_default {
            if let Some(p) = &checkpoint_to { let _ = std::fs::remove_file(p); }
        }
        log_msg(&format!("[gen] fast mode: total emitted = {}",
            stats.emitted.load(AtomicOrdering::Relaxed)));
        return Ok(());
    }

    let progress_path_for_merger = progress_path.clone();
    let args_fp_for_merger       = args_fp.clone();

    // Memory monitor. Starts a background thread that polls /proc every
    // mem_sample_ms and sets the abort flag if RSS or global memory pressure
    // thresholds are breached. The merger checks the flag at each chunk boundary.
    //
    // Soft cap: triggers graceful thread retirement rather than an immediate abort.
    // Set adaptively as the midpoint between current RSS (post-child_cache) and the
    // hard cap. This avoids firing during model load — the child_cache can be
    // 5–20× the on-disk model size, so a fixed percentage of the hard cap would
    // trip immediately on large models before any enumeration starts.
    let max_rss_kb = {
        let from_flag = args.max_rss_gb.map(|gb| (gb * 1024.0 * 1024.0) as u64);
        let from_env  = std::env::var("TOKENOV_MAX_RSS_GB").ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map(|gb| (gb * 1024.0 * 1024.0) as u64);
        from_flag.or(from_env).unwrap_or_else(MemMonitorConfig::default_rss_cap_kb)
    };
    let baseline_rss_kb = ProcMemReader.snapshot()
        .map(|s| s.rss_kb)
        .unwrap_or(0);
    let soft_rss_kb = if max_rss_kb > baseline_rss_kb {
        // Midpoint between current (post-load) RSS and the hard cap.
        baseline_rss_kb + (max_rss_kb - baseline_rss_kb) / 2
    } else {
        0 // model already exceeds hard cap; disable soft threshold
    };
    let mem_cfg = MemMonitorConfig {
        max_rss_kb,
        soft_rss_kb,
        pressure_threshold: args.mem_pressure_threshold,
        interval: std::time::Duration::from_millis(args.mem_sample_ms),
    };
    log_msg(&format!(
        "[gen] mem_monitor: rss_cap={:.2} GiB  soft_cap={:.2} GiB  \
         pressure_threshold={:.0}%  sample={}ms",
        max_rss_kb as f64 / (1024.0 * 1024.0),
        soft_rss_kb as f64 / (1024.0 * 1024.0),
        args.mem_pressure_threshold * 100.0,
        args.mem_sample_ms,
    ));
    let mem_monitor = MemMonitor::start(
        mem_cfg, ProcMemReader, Arc::clone(&active_target), 1);
    let abort_flag  = mem_monitor.abort_flag();

    // Runtime auto-tune is now OFF by default: the 262144 default
    // already captures the chunk-tuning win, and the tuner measures the
    // consumer's rate when piped. Opt in with --runtime-tune; it still needs an
    // unpinned chunk size to have anything to tune.
    if args.no_runtime_tune {
        log_msg("[gen] note: --no-runtime-tune is the default now (runtime tuner off); \
                 use --runtime-tune to enable");
    }
    let runtime_tune_enabled = args.runtime_tune && args.merge_chunk_size.is_none();

    // Telemetry plumbing. `stats_atomics` is shared between the merger
    // (writer) and the telemetry/auto-tune threads (readers). The auto-tuner
    // needs the emission counter regardless of whether the user enabled
    // stderr/CSV telemetry, so we materialize the stats whenever either
    // consumer is on.
    let telemetry_enabled = args.stats_interval_ms > 0;
    let stats_atomics: Option<Arc<MergerStats>> = if telemetry_enabled || runtime_tune_enabled {
        Some(Arc::new(MergerStats::default()))
    } else {
        None
    };
    let stats_stop: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    let stats_csv_path: Option<PathBuf> = if telemetry_enabled {
        match (&args.stats_csv, &args.output) {
            (Some(p), _)              => Some(p.clone()),
            (None,    Some(out_path)) => {
                let mut s = out_path.as_os_str().to_owned();
                s.push(".stats.csv");
                Some(PathBuf::from(s))
            }
            (None, None) => None, // stdout output → stderr-only telemetry
        }
    } else {
        None
    };
    let stats_for_merger = stats_atomics.clone();

    let scope_result: Result<u64> = std::thread::scope(|scope| {
        // Spawn the telemetry thread (if enabled). It owns its own clones of
        // the stats Arc and stop signal; outer scope sets stop after the
        // merger joins, which lets the telemetry loop fall out cleanly with
        // one final tick capturing terminal counter values.
        let telemetry_handle = if telemetry_enabled {
            stats_atomics.clone().map(|s| {
                let stop_for_t = Arc::clone(&stats_stop);
                let csv = stats_csv_path.clone();
                let interval = Duration::from_millis(args.stats_interval_ms);
                let tverbose = verbose();
                let tstderr_every = Duration::from_secs(args.progress_secs);
                let tjson = args.json;
                scope.spawn(move || run_telemetry(s, stop_for_t, interval, csv, tverbose, tstderr_every, tjson))
            })
        } else {
            None
        };

        // Spawn the runtime K auto-tuner (if enabled). Reads the same
        // `MergerStats.emitted` the telemetry uses; writes to `chunk_size_atomic`
        // which producers re-read at chunk boundaries.
        let auto_tune_handle = if runtime_tune_enabled {
            stats_atomics.clone().map(|s| {
                let stop_for_at = Arc::clone(&stats_stop);
                let cs          = Arc::clone(&chunk_size_atomic);
                let sidecar     = Some(sidecar_path_for(&model_path));
                let mp          = model_path.clone();
                let nt          = n_threads;
                scope.spawn(move || run_auto_tuner(s, cs, stop_for_at, sidecar, mp, nt))
            })
        } else {
            None
        };

        // Spawn the merger; it consumes all receivers and the sink.
        let abort_for_merger = Arc::clone(&abort_flag);
        let merger = scope.spawn(move || -> Result<u64> {
            run_merger(sink, receivers, skip_first, initial_bytes,
                progress_path_for_merger.as_deref(), &args_fp_for_merger,
                stats_for_merger, abort_for_merger)
        });

        // Spawn one worker per partition (static, deterministic). Each worker
        // owns exactly one partition and one channel, runs a SINGLE level-sweep
        // to its quota (thread_target = target/n_threads), then exits. One
        // partition ⇒ one channel ⇒ the channel is level-sorted across its whole
        // stream, which is the invariant the k-way merger relies on. (This is the
        // known-good static-spawn design.)
        let mut worker_handles = Vec::with_capacity(n_threads);
        for ((tx, thread_id), states) in worker_senders.into_iter().zip(partitions) {
            let em  = Arc::clone(&enum_model);
            let cc  = child_cache;  // &'static, copy is free — no Arc to clone
            let dt  = Arc::clone(&decode_table);
            let cs  = Arc::clone(&chunk_size_atomic);
            let var = Arc::clone(&variant);
            let cm  = case_masks.clone();  // tiny per-worker copy of the case masks
            worker_handles.push(scope.spawn(move || -> Result<()> {
                let label = format!("[t{}] ", thread_id);
                let mut chunk_sender = ChunkSender::new(tx, cs);
                let res = enumerate_to_sink(
                    &em, &*var, cc, cache_mode, &dt, kind, start_id, end_id,
                    states, max_tokens, min_tokens, min_len, max_len,
                    min_level,
                    thread_target,
                    enterprise,
                    &cm,
                    |lvl, sk, bytes| chunk_sender.push(lvl, sk, bytes),
                    &label,
                    None, None, |_: &ThreadCkpt| {},
                );
                // Flush the trailing partial chunk before the ChunkSender (and
                // its channel sender) drops, signaling end-of-stream. Don't
                // shadow an enumerate error with a flush error.
                if res.is_ok() {
                    chunk_sender.flush()?;
                }
                res
            }));
        }

        // Collect worker errors. Worker exit (normal or error) drops the channel
        // sender, signaling end-of-stream to the merger for that channel.
        let mut worker_errors: Vec<anyhow::Error> = Vec::new();
        for h in worker_handles {
            match h.join() {
                Err(_)        => worker_errors.push(anyhow!("worker panicked")),
                Ok(Err(e))    => worker_errors.push(e),
                Ok(Ok(()))    => {}
            }
        }

        let merger_result = merger.join().map_err(|_| anyhow!("merger panicked"))?;
        // Tell telemetry + auto-tuner to wrap up. Joining them here keeps
        // any final stderr line ordered before our own log lines.
        stats_stop.store(true, AtomicOrdering::Relaxed);
        if let Some(h) = telemetry_handle { let _ = h.join(); }
        if let Some(h) = auto_tune_handle { let _ = h.join(); }
        if !worker_errors.is_empty() {
            return Err(worker_errors.remove(0));
        }
        merger_result
    });

    let total_emitted = scope_result?;
    log_msg(&format!("[gen] merger drained: total emitted = {}", total_emitted));
    // Signal monitor thread to stop and log peak RSS.
    // (The thread is detached so we can't join it, but it will exit the next
    // time it wakes and sees the abort flag set — or when the process exits.)
    mem_monitor.abort_flag().store(true, AtomicOrdering::Relaxed);
    let peak_rss_kb = mem_monitor.peak_rss_kb();
    if peak_rss_kb > 0 {
        log_msg(&format!("[gen] peak RSS: {:.2} GiB", peak_rss_kb as f64 / (1024.0 * 1024.0)));
    }
    if let Some(pp) = &progress_path { let _ = std::fs::remove_file(pp); }
    Ok(())
}

// ============================================================================
// Streaming k-way merger (parallel mode).
// ============================================================================

/// Atomic counters published by the merger and read by the telemetry thread.
/// Two stores per chunk drain (Relaxed) — undetectable next to the disk write
/// + sink flush already happening per chunk. The telemetry thread only loads,
/// never stores, so there's zero contention against the writer.
#[derive(Default)]
struct MergerStats {
    emitted:       AtomicU64,
    bytes_written: AtomicU64,
    /// Set while the merger is inside the chunk write loop (i.e. potentially
    /// blocked in `sink.write_line` flushing to a full downstream pipe). Lets
    /// telemetry distinguish "stalled on back-pressure" (healthy — the consumer
    /// is busy) from a real hang. One pair of Relaxed stores per chunk drain.
    writer_blocked: AtomicBool,
}

/// Telemetry thread loop. Sleeps `interval`, reads atomics, writes one row to
/// the stats CSV (if any) and one stderr line, repeats until `stop` is set.
/// One extra tick fires after `stop` is set so the final row captures the
/// terminal counter values.
fn run_telemetry(
    stats:        Arc<MergerStats>,
    stop:         Arc<AtomicBool>,
    interval:     Duration,
    csv_path:     Option<PathBuf>,
    verbose:      bool,
    stderr_every: Duration,
    json:         bool,
) {
    let t0 = Instant::now();
    let mut writer: Option<BufWriter<File>> = match csv_path.as_ref() {
        Some(p) => match File::create(p) {
            Ok(f) => {
                let mut w = BufWriter::new(f);
                let _ = writeln!(w,
                    "t_seconds,emitted_total,delta_emit,inst_c_s,avg_c_s,bytes_total");
                let _ = w.flush();
                Some(w)
            }
            Err(e) => {
                eprintln!("[stats] cannot create {}: {}", p.display(), e);
                None
            }
        },
        None => None,
    };

    let mut prev_emitted: u64       = 0;
    let mut prev_t:       Duration  = Duration::ZERO;
    // Separate accounting for the (throttled) stderr line so its Δ / inst
    // reflect the period since the last *printed* line, not the last tick.
    let mut last_se_emit: u64            = 0;
    let mut last_se_t:    Option<Duration> = None;

    loop {
        std::thread::sleep(interval);
        let elapsed   = t0.elapsed();
        let emitted   = stats.emitted.load(AtomicOrdering::Relaxed);
        let bytes     = stats.bytes_written.load(AtomicOrdering::Relaxed);
        let delta     = emitted.saturating_sub(prev_emitted);
        let dt_secs   = (elapsed - prev_t).as_secs_f64();
        let inst      = if dt_secs > 0.0 { delta as f64 / dt_secs } else { 0.0 };
        let t_secs    = elapsed.as_secs_f64();
        let avg       = if t_secs   > 0.0 { emitted as f64 / t_secs } else { 0.0 };

        let is_final = stop.load(AtomicOrdering::Relaxed);
        // stderr: verbose → every tick; else first tick, every `stderr_every`,
        // and the final tick. CSV (below) still records every tick.
        // Quiet by default: the human `[stats]` line only prints with --verbose.
        // `--json` is an explicit opt-in for wrapper tools, so its stream keeps the
        // periodic + final cadence regardless of verbosity.
        let due_stderr = if json {
            verbose || is_final
                || last_se_t.map_or(true, |t| elapsed.saturating_sub(t) >= stderr_every)
        } else {
            verbose
        };
        if due_stderr {
            let prev_t_se = last_se_t.unwrap_or(Duration::ZERO);
            let se_delta  = emitted.saturating_sub(last_se_emit);
            let se_dt     = (elapsed - prev_t_se).as_secs_f64();
            let se_inst   = if se_dt > 0.0 { se_delta as f64 / se_dt } else { 0.0 };
            let blocked   = stats.writer_blocked.load(AtomicOrdering::Relaxed);
            let stalled   = se_delta == 0;
            if json {
                // JSONL for wrapper tools. No strings → no escaping needed.
                eprintln!(
                    "{{\"elapsed_s\":{:.1},\"emitted\":{},\"delta\":{},\"inst_cps\":{:.0},\"avg_cps\":{:.0},\"bytes\":{},\"blocked\":{},\"stalled\":{}}}",
                    t_secs, emitted, se_delta, se_inst, avg, bytes, blocked, stalled,
                );
            } else {
                // Explain a stall: blocked writing to a full downstream pipe
                // (consumer busy — healthy) vs waiting on the generator threads.
                let note = if stalled {
                    if blocked {
                        "  STALLED: blocked writing to downstream consumer (back-pressure — normal with high-multiplier rules; not a hang)"
                    } else {
                        "  STALLED: no output this interval (generator catching up)"
                    }
                } else { "" };
                eprintln!("[stats] t={:6.1}s emit={:>13} Δ={:>10} inst={:>10.0}/s avg={:>10.0}/s bytes={}{}",
                    t_secs, emitted, se_delta, se_inst, avg, bytes, note);
            }
            last_se_emit = emitted;
            last_se_t    = Some(elapsed);
        }
        if let Some(w) = writer.as_mut() {
            let _ = writeln!(w, "{:.3},{},{},{:.0},{:.0},{}",
                t_secs, emitted, delta, inst, avg, bytes);
            let _ = w.flush();
        }

        prev_emitted = emitted;
        prev_t       = elapsed;

        if is_final {
            break;
        }
    }
    if let Some(mut w) = writer {
        let _ = w.flush();
    }
}

// ── Runtime K auto-tuner ────────────────────────────────────────────────────

/// Lower bound for runtime K probing. Below this the per-chunk overhead
/// dominates throughput; calibration sweeps confirm K=1024 is a safe floor
/// for the production workloads.
const RUNTIME_TUNE_MIN_K: usize = 256;

/// Upper bound for runtime K probing. Channel capacity is sized at startup
/// from `initial_chunk_size`; if K grows past this, individual chunks would
/// hold more items but the channel still buffers `channel_chunks` of them,
/// so total in-flight memory grows linearly with K. Cap at 16,384 to keep
/// per-channel buffer headroom comfortable.
const RUNTIME_TUNE_MAX_K: usize = 16_384;

/// Throughput improvement required to keep a probed K. The level-sweep
/// rate is highly non-stationary (varies 1.5–2× across short windows); a
/// 5 % improvement easily falls inside that envelope. Require ≥10 % so
/// only meaningful K-effects pass the gate.
const RUNTIME_TUNE_ACCEPT_RATIO: f64 = 1.10;

/// Number of short samples we take per phase (baseline / probe). What
/// matters statistically is total phase time × emission rate (= total
/// candidates counted); slicing into N sub-samples adds nothing on its
/// own. We slice anyway because MAD-based outlier rejection needs enough
/// data points to compute a meaningful median + MAD — N=7 gives 7-3=4
/// kept samples even after dropping a couple of outliers.
const RUNTIME_TUNE_SAMPLES_PER_PHASE: usize = 7;

/// Length of one sample within a phase. Longer samples have lower per-
/// sample variance because each one averages over more underlying chunk-
/// emission cycles. We saw 10 s windows span 35–120 K c/s on the same K
/// (CoV ~50 %); 30 s windows should land closer to ~30 % CoV. The
/// trimmed mean of 7 such samples then has standard error ~30%/√7 ≈ 11 %,
/// which is what we need to detect a 10 % K-effect with confidence.
const RUNTIME_TUNE_SAMPLE_SECS: u64 = 30;

/// MAD (median absolute deviation) multiplier for outlier rejection. Any
/// sample > k × MAD from the median is treated as an outlier and dropped
/// from the phase mean. k=2.5 corresponds to roughly the 99 % envelope
/// for normally-distributed data, but the test is distribution-free —
/// what matters is "this sample is far from the rest", and MAD adapts to
/// the actual spread of the data without needing a parametric assumption.
const RUNTIME_TUNE_MAD_K: f64 = 2.5;

/// Maximum allowed MAD/median ratio for a phase to count as "stable enough"
/// to base a K decision on. The level-sweep enumeration has structurally
/// different emission rates at different level depths; when a phase spans
/// a level-depth transition, MAD inflates because the "samples" are really
/// sampling different regimes, not noise around a single mean. Above this
/// threshold the phase isn't a useful estimate of K's effect — defer the
/// decision rather than chase a number dominated by regime change.
///
/// Applied to both the baseline phase (skip the probe entirely) and the
/// probe phase (revert K and defer to next cycle, since we don't know if
/// the probed K helped or hurt).
///
/// Empirical: stable single-level phases run 3–5 % MAD/median; phases that
/// straddle a transition run 15–25 %. A 10 % gate cleanly separates them.
const RUNTIME_TUNE_MAX_PHASE_NOISE: f64 = 0.10;

/// Median of a slice (no panic on empty — returns 0.0). Sorts a copy.
fn median_of(samples: &[f64]) -> f64 {
    if samples.is_empty() { return 0.0; }
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = s.len();
    if n % 2 == 0 { (s[n/2 - 1] + s[n/2]) / 2.0 } else { s[n/2] }
}

/// Median absolute deviation around a given center.
fn median_absolute_deviation(samples: &[f64], center: f64) -> f64 {
    if samples.is_empty() { return 0.0; }
    let dev: Vec<f64> = samples.iter().map(|x| (x - center).abs()).collect();
    median_of(&dev)
}

/// MAD-trimmed mean: drop samples > k × MAD from the median, average the rest.
/// Falls back to the median if MAD is exactly zero (all samples identical) or
/// if every sample gets rejected.
fn robust_mean(samples: &[f64], k: f64) -> f64 {
    if samples.is_empty() { return 0.0; }
    let median = median_of(samples);
    let mad    = median_absolute_deviation(samples, median);
    if mad == 0.0 {
        return median;
    }
    let kept: Vec<f64> = samples.iter().copied()
        .filter(|x| (*x - median).abs() <= k * mad)
        .collect();
    if kept.is_empty() {
        median
    } else {
        kept.iter().sum::<f64>() / kept.len() as f64
    }
}

/// Read an `f64` env var with a default fallback. Used so the auto-tune
/// gate thresholds can be loosened or tightened without a rebuild.
fn env_f64_or(key: &str, default: f64) -> f64 {
    std::env::var(key).ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

/// Hill-climbing K auto-tuner with robust per-phase measurement.
///
/// The level-sweep enumeration's per-second rate is highly non-stationary
/// (varies 3–4× across short windows depending on heap depth, thermal state,
/// and which level the sweep is emitting from), so a single throughput
/// measurement is unreliable. Each phase here takes N short samples
/// (`RUNTIME_TUNE_SAMPLES_PER_PHASE` × `RUNTIME_TUNE_SAMPLE_SECS`) and
/// reports the **trimmed mean** of the middle samples — drop the highest
/// and lowest, average the rest. With N=5 that's middle-3-of-5, which
/// rejects single fortunate or unfortunate windows that would otherwise
/// dominate a one-shot reading.
///
/// Algorithm:
///   1. Wait `warmup_secs` for the model to reach steady state.
///   2. Sample N×T at current K → trimmed-mean baseline rate.
///   3. Pick a probe direction (alternates ×2 / ÷2). Switch K, settle.
///   4. Sample N×T at new K → trimmed-mean probe rate.
///   5. If probe / baseline > ACCEPT_RATIO (1.05) → keep new K, persist;
///      else → revert and flip direction next round.
///   6. Wait `cooldown_secs` and repeat.
///
/// On accept, writes the new value back to the sidecar so future runs start
/// from there. The sidecar's original `measurements` vector is preserved
/// (calibration data); only `recommended_chunk_size` and `calibrated_at`
/// are updated, plus one appended measurement entry.
#[allow(clippy::too_many_arguments)]
fn run_auto_tuner(
    stats:        Arc<MergerStats>,
    chunk_size:   Arc<AtomicUsize>,
    stop:         Arc<AtomicBool>,
    sidecar_path: Option<PathBuf>,
    model_path:   PathBuf,
    n_threads:    usize,
) {
    const WARMUP_SECS:   u64 = 60;
    const SETTLE_SECS:   u64 = 10;
    const COOLDOWN_SECS: u64 = 30;

    let phase_secs = RUNTIME_TUNE_SAMPLES_PER_PHASE as u64 * RUNTIME_TUNE_SAMPLE_SECS;

    // Sleep with stop check; returns true if stop was signaled.
    let sleep_or_stop = |secs: u64, stop: &Arc<AtomicBool>| -> bool {
        for _ in 0..secs {
            std::thread::sleep(Duration::from_secs(1));
            if stop.load(AtomicOrdering::Relaxed) { return true; }
        }
        false
    };

    // Take a single short rate sample.
    let sample_rate = |stop: &Arc<AtomicBool>| -> Option<f64> {
        let start_emitted = stats.emitted.load(AtomicOrdering::Relaxed);
        let start = Instant::now();
        if sleep_or_stop(RUNTIME_TUNE_SAMPLE_SECS, stop) { return None; }
        let end_emitted = stats.emitted.load(AtomicOrdering::Relaxed);
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed > 0.0 && end_emitted > start_emitted {
            Some((end_emitted - start_emitted) as f64 / elapsed)
        } else {
            Some(0.0)
        }
    };

    // Take N samples and return (robust_mean, raw_samples, n_kept).
    //
    // Robust mean = MAD-trimmed mean:
    //   1. Compute the median M of all samples.
    //   2. Compute MAD = median(|x - M|).
    //   3. Drop any sample where |x - M| > k × MAD (k = RUNTIME_TUNE_MAD_K).
    //   4. Return the arithmetic mean of the kept samples.
    //
    // Why MAD rather than fixed-position trimming: the level-sweep rate has
    // heavy tails — most samples cluster, but occasionally a sample lands
    // 3-4× higher or lower than the cluster. Fixed-position trim (drop top-
    // 1 + bottom-1) only removes 2 of N samples regardless of how extreme
    // they are. MAD-based trim adapts: if one sample is 5× off, it gets
    // dropped; if all samples are tight, none get dropped. Distribution-
    // free, no parametric assumption.
    let sample_phase = |stop: &Arc<AtomicBool>| -> Option<(f64, Vec<f64>, usize)> {
        let mut samples = Vec::with_capacity(RUNTIME_TUNE_SAMPLES_PER_PHASE);
        for _ in 0..RUNTIME_TUNE_SAMPLES_PER_PHASE {
            samples.push(sample_rate(stop)?);
        }
        let mean = robust_mean(&samples, RUNTIME_TUNE_MAD_K);
        let kept = samples.iter()
            .filter(|x| {
                let median = median_of(&samples);
                let mad = median_absolute_deviation(&samples, median);
                if mad == 0.0 { true } else { (**x - median).abs() <= RUNTIME_TUNE_MAD_K * mad }
            })
            .count();
        Some((mean, samples, kept))
    };

    let format_samples = |samples: &[f64]| -> String {
        samples.iter()
            .map(|r| format!("{:.0}", r))
            .collect::<Vec<_>>()
            .join(",")
    };

    // Override the noise gate and accept ratio at runtime via env vars.
    // Useful if the defaults turn out too strict (deferring forever) or too
    // loose (accepting noise) on a particular workload.
    let noise_gate    = env_f64_or("TOKENOV_AUTOTUNE_NOISE_GATE", RUNTIME_TUNE_MAX_PHASE_NOISE);
    let accept_ratio  = env_f64_or("TOKENOV_AUTOTUNE_ACCEPT_RATIO", RUNTIME_TUNE_ACCEPT_RATIO);

    log_msg(&format!(
        "[auto-tune] starting (warmup={}s phase={}×{}s={}s settle={}s cooldown={}s \
         noise_gate={:.0}% accept_ratio={:.2})",
        WARMUP_SECS, RUNTIME_TUNE_SAMPLES_PER_PHASE, RUNTIME_TUNE_SAMPLE_SECS,
        phase_secs, SETTLE_SECS, COOLDOWN_SECS,
        noise_gate * 100.0, accept_ratio));
    if sleep_or_stop(WARMUP_SECS, &stop) { return; }

    // Alternate probe direction: try up first, then down, repeat.
    let directions = [2.0_f64, 0.5];
    let mut dir_idx = 0;
    let mut consecutive_reverts = 0;
    let mut best_persisted_k = chunk_size.load(AtomicOrdering::Relaxed);

    loop {
        if stop.load(AtomicOrdering::Relaxed) { return; }

        let current_k = chunk_size.load(AtomicOrdering::Relaxed);
        let (baseline, baseline_samples, baseline_kept) = match sample_phase(&stop) {
            Some(x) => x,
            None    => return,
        };
        if baseline <= 0.0 {
            log_msg("[auto-tune] no emissions during baseline phase; pausing");
            if sleep_or_stop(COOLDOWN_SECS, &stop) { return; }
            continue;
        }

        // Baseline stability gate: if the baseline phase straddles a level-
        // depth transition (or any other regime change), the within-phase MAD
        // dwarfs any real K-effect we could detect. Defer the probe and
        // re-measure on the next cycle, when we're hopefully back in a
        // single stable regime.
        let baseline_median = median_of(&baseline_samples);
        let baseline_mad    = median_absolute_deviation(&baseline_samples, baseline_median);
        let baseline_noise  = if baseline_median > 0.0 {
            baseline_mad / baseline_median
        } else { 1.0 };
        if baseline_noise > noise_gate {
            log_msg(&format!(
                "[auto-tune] baseline too noisy (MAD/median {:.0}% > {:.0}% gate, samples [{}]) — \
                 deferring probe (probably mid level-transition)",
                baseline_noise * 100.0, noise_gate * 100.0,
                format_samples(&baseline_samples)));
            if sleep_or_stop(COOLDOWN_SECS, &stop) { return; }
            continue;
        }

        let direction = directions[dir_idx];
        let new_k = ((current_k as f64 * direction) as usize)
            .clamp(RUNTIME_TUNE_MIN_K, RUNTIME_TUNE_MAX_K);
        if new_k == current_k {
            dir_idx = 1 - dir_idx;
            if sleep_or_stop(COOLDOWN_SECS, &stop) { return; }
            continue;
        }

        log_msg(&format!(
            "[auto-tune] probing K={} → K={} (baseline {:.0} c/s, kept {}/{} samples [{}])",
            current_k, new_k, baseline, baseline_kept, baseline_samples.len(),
            format_samples(&baseline_samples)));
        chunk_size.store(new_k, AtomicOrdering::Relaxed);
        if sleep_or_stop(SETTLE_SECS, &stop) { return; }

        let (probe, probe_samples, probe_kept) = match sample_phase(&stop) {
            Some(x) => x,
            None    => return,
        };
        if probe <= 0.0 {
            log_msg("[auto-tune] no emissions during probe phase; reverting");
            chunk_size.store(current_k, AtomicOrdering::Relaxed);
            consecutive_reverts += 1;
            if sleep_or_stop(COOLDOWN_SECS, &stop) { return; }
            continue;
        }

        // Probe stability gate: if the probe phase straddles a regime change,
        // its mean isn't a meaningful estimate of the new K's effect. Even
        // when MAD-trim drops a couple of outliers, a smooth upward / downward
        // drift across all samples will pass the trim but bias the mean. We
        // detect this via the same MAD/median ratio applied to the raw probe
        // samples. On breach: revert K (we don't trust the comparison) and
        // wait one cooldown — the next cycle will re-measure.
        let probe_median = median_of(&probe_samples);
        let probe_mad    = median_absolute_deviation(&probe_samples, probe_median);
        let probe_noise  = if probe_median > 0.0 {
            probe_mad / probe_median
        } else { 1.0 };
        if probe_noise > noise_gate {
            log_msg(&format!(
                "[auto-tune] probe phase too noisy (MAD/median {:.0}% > {:.0}% gate, samples [{}]) — \
                 reverting K and deferring (probably mid level-transition)",
                probe_noise * 100.0, noise_gate * 100.0,
                format_samples(&probe_samples)));
            chunk_size.store(current_k, AtomicOrdering::Relaxed);
            // Don't count this as a "consecutive revert" — it's an aborted
            // measurement, not a rejection of the probed K.
            if sleep_or_stop(COOLDOWN_SECS, &stop) { return; }
            continue;
        }

        let ratio = probe / baseline;
        if ratio > accept_ratio {
            log_msg(&format!(
                "[auto-tune] ACCEPT K={} → {} (baseline {:.0} kept {}/{} [{}], probe {:.0} kept {}/{} [{}], +{:.1}%)",
                current_k, new_k,
                baseline, baseline_kept, baseline_samples.len(), format_samples(&baseline_samples),
                probe, probe_kept, probe_samples.len(), format_samples(&probe_samples),
                (ratio - 1.0) * 100.0));
            consecutive_reverts = 0;
            if let Some(sp) = sidecar_path.as_ref() {
                if new_k != best_persisted_k {
                    if let Err(e) = persist_runtime_tune(sp, &model_path, new_k, n_threads, probe) {
                        log_msg(&format!(
                            "[auto-tune] sidecar writeback failed: {} — continuing in-memory only", e));
                    } else {
                        best_persisted_k = new_k;
                    }
                }
            }
        } else {
            log_msg(&format!(
                "[auto-tune] REVERT K={} → {} (baseline {:.0} kept {}/{} [{}], probe {:.0} kept {}/{} [{}], ratio {:.3})",
                new_k, current_k,
                baseline, baseline_kept, baseline_samples.len(), format_samples(&baseline_samples),
                probe, probe_kept, probe_samples.len(), format_samples(&probe_samples),
                ratio));
            chunk_size.store(current_k, AtomicOrdering::Relaxed);
            consecutive_reverts += 1;
            dir_idx = 1 - dir_idx;
        }

        // Back off if we've been reverting consistently — current K is
        // probably already near optimal. Wait longer between probes.
        let cooldown = if consecutive_reverts >= 2 {
            COOLDOWN_SECS * 4 // 2 min instead of 30 s
        } else {
            COOLDOWN_SECS
        };
        if sleep_or_stop(cooldown, &stop) { return; }
    }
}

/// Update the recommended K in the sidecar without disturbing the original
/// calibration measurements. Reads the existing sidecar if any and updates
/// only `recommended_chunk_size` + `calibrated_at`. If no sidecar exists,
/// writes a minimal one with no measurement curve.
fn persist_runtime_tune(
    sidecar_path: &Path,
    model_path:   &Path,
    new_k:        usize,
    n_threads:    usize,
    measured_rate: f64,
) -> Result<()> {
    let mut sc = match read_tune_sidecar(sidecar_path)? {
        Some(s) => s,
        None => {
            // No prior sidecar — synthesize a minimal one. Measurement curve
            // intentionally empty (we only have one data point).
            let (size, mtime) = file_size_mtime(model_path).unwrap_or((0, 0));
            TuneSidecar {
                schema_version: 1,
                model_path: model_path.to_path_buf(),
                model_size_bytes: size,
                model_mtime_unix: mtime,
                calibrated_at: timestamp_now(),
                hostname: hostname(),
                cpu_count: rayon::current_num_threads(),
                threads_used: n_threads,
                recommended_chunk_size: new_k,
                measurements: Vec::new(),
                noise_warning: None,
            }
        }
    };
    sc.recommended_chunk_size = new_k;
    sc.calibrated_at = timestamp_now();
    sc.threads_used = n_threads;
    // Append a single-point measurement so future readers can see this came
    // from runtime tuning rather than the initial sweep.
    sc.measurements.push(TuneMeasurement {
        chunk_size: new_k,
        emit_rate_per_sec: measured_rate,
        peak_rss_mb: read_peak_rss_mb(),
    });
    write_tune_sidecar(sidecar_path, &sc)?;
    Ok(())
}

/// Merger thread: drain N producer channels into one globally rank-ordered
/// stream. Producers send `MergeChunk`s — locally rank-ordered batches of
/// items — and the merger orders chunks by their head item's
/// (level, sort_key). The pop-loop drains an entire chunk before consulting
/// the heap again, so heap and channel ops cost amortizes over `chunk_size`
/// items. Cross-chunk ordering is approximate (chunk tails can be lower-prob
/// than the next chunk's head); within a chunk ordering is exact.
///
/// Updates the progress sidecar periodically. Honors `skip_first` for resume
/// — counts but doesn't write the first N candidates (already on disk).
///
/// Exits when all producer channels close (returning emitted count) or when
/// a sink write fails (returning Err; receivers drop on function exit, which
/// unblocks any producers stuck on full channels).
fn run_merger(
    mut sink:        Sink,
    receivers:       Vec<Receiver<MergeChunk>>,
    skip_first:      u64,
    initial_bytes:   u64, // byte_offset on resume; 0 fresh
    progress_path:   Option<&Path>,
    args_fp:         &str,
    stats:           Option<Arc<MergerStats>>,
    abort:           Arc<AtomicBool>,
) -> Result<u64> {
    let mut emitted:       u64 = 0;
    let mut bytes_written: u64 = initial_bytes;
    // Reusable per-chunk output buffer (one batched write per chunk).
    let mut write_buf: Vec<u8> = Vec::new();
    let mut last_progress = Instant::now();
    let mut last_log      = Instant::now();
    let t0 = Instant::now();

    // Pre-fetch one chunk from each channel. recv() blocks until the first
    // chunk arrives or the channel closes (Err means the worker emitted
    // nothing — possible for tiny --count runs satisfied by other threads).
    let mut heads: Vec<Option<MergeChunk>> =
        receivers.iter().map(|rx| rx.recv().ok()).collect();
    let mut heap: BinaryHeap<Reverse<(u32, u32, usize)>> = BinaryHeap::new();
    for (i, h) in heads.iter().enumerate() {
        if let Some(chunk) = h {
            // Empty chunks shouldn't happen — ChunkSender::flush only sends
            // non-empty buffers — but guard defensively.
            if let Some(first) = chunk.items.first() {
                heap.push(Reverse((first.level, first.sort_key, i)));
            }
        }
    }

    while let Some(Reverse((_lvl, _sk, idx))) = heap.pop() {
        let chunk = heads[idx].take().expect("heap entry without chunk");
        let chunk_len = chunk.items.len();

        // Drain the entire chunk in one go. Within the chunk items are
        // already locally rank-ordered (the producer's level sweep emitted
        // them that way), so we just walk the index. Bytes live in the chunk's
        // shared `bytes` arena and are addressed by (offset, len).
        // Batch the whole chunk into one buffer (candidates + '\n'
        // concatenated, in order) and issue a single write, instead of two
        // write calls per candidate. Identical bytes + order; far fewer calls.
        // `skip_first` (resume) suppresses a leading prefix — the written items
        // are always a contiguous suffix of the chunk.
        let n_items = chunk.items.len() as u64;
        let start_i = if emitted >= skip_first {
            0
        } else {
            (skip_first - emitted).min(n_items) as usize
        };
        write_buf.clear();
        for item in &chunk.items[start_i..] {
            let off = item.byte_offset as usize;
            let end = off + item.byte_len as usize;
            write_buf.extend_from_slice(&chunk.bytes[off..end]);
            write_buf.push(b'\n');
        }
        let written_items = (chunk.items.len() - start_i) as u64;
        // Mark the write phase: if the downstream pipe is full (consumer busy),
        // write_chunk blocks here and emit stalls — telemetry reads this to
        // label it as back-pressure rather than a hang.
        if let Some(s) = &stats { s.writer_blocked.store(true, AtomicOrdering::Relaxed); }
        sink.write_chunk(&write_buf, written_items)?;
        bytes_written += write_buf.len() as u64;
        emitted += n_items;
        let _ = chunk_len; // for future per-chunk diagnostics

        // Publish counters to telemetry. Relaxed stores per chunk drain;
        // the telemetry thread only loads, no contention with the writer.
        if let Some(s) = &stats {
            s.writer_blocked.store(false, AtomicOrdering::Relaxed);
            s.emitted.store(emitted, AtomicOrdering::Relaxed);
            s.bytes_written.store(bytes_written, AtomicOrdering::Relaxed);
        }

        // Check OOM abort flag. Relaxed load: the monitor thread uses SeqCst
        // on the store, so we will see it within one sample interval.
        if abort.load(AtomicOrdering::Relaxed) {
            sink.flush_buffered()?;
            if let Some(pp) = progress_path {
                let _ = write_progress(pp, emitted, bytes_written, args_fp);
            }
            bail!("aborted by memory monitor (progress saved — restart with --resume)");
        }

        // Refill from this channel: pull next chunk; on close, drop from
        // rotation.
        if let Ok(next) = receivers[idx].recv() {
            if let Some(first) = next.items.first() {
                let lvl = first.level;
                let sk  = first.sort_key;
                heads[idx] = Some(next);
                heap.push(Reverse((lvl, sk, idx)));
            }
        }

        // Periodic progress + log. Time-check only when emitted has crossed
        // a 1M boundary to avoid clock-call overhead on the hot path. With
        // chunked drain we may step over multiple millions in one chunk;
        // emit one progress write per drain that crosses a boundary.
        if emitted >= last_progress_threshold(emitted, chunk_len as u64) {
            if last_progress.elapsed().as_secs() >= 1 {
                if let Some(pp) = progress_path {
                    sink.flush_buffered()?;
                    let _ = write_progress(pp, emitted, bytes_written, args_fp);
                }
                last_progress = Instant::now();
            }
            if last_log.elapsed().as_secs() >= 5 {
                log_msg(&format!("[merge] emitted={} elapsed={:.2}s",
                    emitted, t0.elapsed().as_secs_f64()));
                last_log = Instant::now();
            }
        }
    }

    sink.finish()?;
    Ok(emitted)
}

/// Returns the next 1M-multiple boundary if the current chunk crossed one,
/// else returns u64::MAX (so the caller's `emitted >= ...` check is false).
/// This lets the merger emit one progress/log update per drain that crosses
/// any 1M boundary, regardless of how many millions a single chunk spans.
#[inline]
fn last_progress_threshold(emitted_after: u64, chunk_len: u64) -> u64 {
    let emitted_before = emitted_after.saturating_sub(chunk_len);
    let prev_boundary  = emitted_before / 1_000_000;
    let next_boundary  = (prev_boundary + 1) * 1_000_000;
    if emitted_after >= next_boundary { next_boundary } else { u64::MAX }
}

// ============================================================================
// Progress sidecar (resume support).
// ============================================================================

struct ProgressSnapshot {
    emitted:     u64,
    /// Byte offset in the output file as of the last successful flush.
    /// On resume we truncate the file to exactly this size before appending
    /// new content — handles the case where BufWriter auto-flushed extra
    /// content past `emitted` to the OS before the process was killed.
    byte_offset: u64,
    fingerprint: String,
}

fn build_args_fingerprint(
    args:       &GenerateArgs,
    model_path: &Path,
    n_threads:  usize,
) -> Result<String> {
    use std::fmt::Write;
    let meta = std::fs::metadata(model_path)
        .with_context(|| format!("stat {}", model_path.display()))?;
    let size = meta.len();
    let mtime = meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut s = String::new();
    write!(s, "model={} size={} mtime={} ",
        model_path.display(), size, mtime).unwrap();
    write!(s, "threads={} count={:?} ", n_threads, args.count).unwrap();
    write!(s, "min_len={} max_len={} max_tokens={} min_tokens={} ",
        args.min_len, args.max_len, args.max_tokens, args.min_tokens).unwrap();
    write!(s, "mode={:?} bias={} seed_mode={:?} prepend_only={} append_only={} float={} ",
        args.mode, args.bias, args.seed_mode, args.prepend_only, args.append_only, args.float).unwrap();
    write!(s, "skipgram_expand={} skipgram_direction={:?} ",
        args.skipgram_expand, args.skipgram_direction).unwrap();
    write!(s, "wordlist={:?}", args.wordlist).unwrap();
    // Appended only when set, so a default run's fingerprint is unchanged from
    // 1.0.0 and an in-flight `--no-checkpoint` sidecar resume still matches.
    // (The checkpoint file's own fingerprint already carries the crate version,
    // so it never crosses a release boundary anyway.)
    if args.min_level > 0 {
        write!(s, " min_level={}", args.min_level).unwrap();
    }
    Ok(s)
}

fn write_progress(path: &Path, emitted: u64, byte_offset: u64, fingerprint: &str) -> Result<()> {
    let tmp = {
        let mut s = path.as_os_str().to_owned();
        s.push(".tmp");
        PathBuf::from(s)
    };
    {
        let mut f = File::create(&tmp)
            .with_context(|| format!("create {}", tmp.display()))?;
        writeln!(f, "emitted={}",     emitted)?;
        writeln!(f, "byte_offset={}", byte_offset)?;
        writeln!(f, "fingerprint={}", fingerprint)?;
        f.sync_data().ok();
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Truncate the resumed output down to `byte_offset` — the boundary the progress
/// sidecar recorded at the last checkpoint. The on-disk file can be *longer* than
/// that (the BufWriter spilled bytes past the last progress write before the
/// kill); truncating brings it back to a known line boundary before we append.
/// Bails if the file is *shorter* (externally truncated → refusing to append into
/// a hole).
fn truncate_output_to(out: &Path, byte_offset: u64) -> Result<()> {
    let actual_size = std::fs::metadata(out)
        .with_context(|| format!("stat {}", out.display()))?.len();
    if actual_size < byte_offset {
        bail!("--resume: output {} is shorter ({}B) than recorded byte_offset \
               ({}B). File likely truncated externally; refusing to append.",
               out.display(), actual_size, byte_offset);
    }
    if actual_size > byte_offset {
        log_msg(&format!("[gen] resume: truncating {} from {}B to {}B \
                          (buffered output past last checkpoint)",
            out.display(), actual_size, byte_offset));
        std::fs::OpenOptions::new()
            .write(true).open(out)
            .and_then(|f| f.set_len(byte_offset))
            .with_context(|| format!("truncate {}", out.display()))?;
    }
    Ok(())
}

fn read_progress(path: &Path) -> Result<ProgressSnapshot> {
    let txt = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    let mut emitted:     Option<u64>    = None;
    let mut byte_offset: Option<u64>    = None;
    let mut fingerprint: Option<String> = None;
    for line in txt.lines() {
        if let Some(v) = line.strip_prefix("emitted=") {
            emitted = Some(v.parse().with_context(|| {
                format!("parse 'emitted' in {}", path.display())
            })?);
        } else if let Some(v) = line.strip_prefix("byte_offset=") {
            byte_offset = Some(v.parse().with_context(|| {
                format!("parse 'byte_offset' in {}", path.display())
            })?);
        } else if let Some(v) = line.strip_prefix("fingerprint=") {
            fingerprint = Some(v.to_string());
        }
    }
    Ok(ProgressSnapshot {
        emitted:     emitted.ok_or_else(|| anyhow!(
            "missing 'emitted' in {}", path.display()))?,
        byte_offset: byte_offset.ok_or_else(|| anyhow!(
            "missing 'byte_offset' in {}", path.display()))?,
        fingerprint: fingerprint.ok_or_else(|| anyhow!(
            "missing 'fingerprint' in {}", path.display()))?,
    })
}

// ============================================================================
// Crash-resume checkpoints for piped `tokenov | hashcat` jobs.
//
// Each fast-mode worker enumerates an independent partition (no merger), so its
// position is fully captured by {target_level, init_idx, prefix, idx_stack,
// emitted} — see enumerate_to_sink. Workers publish that periodically; a writer
// thread flushes all slots to ONE file atomically. On --resume-state each worker
// reconstructs its DFS stack in O(depth) from its record and continues, instead
// of re-enumerating from candidate 0. Requires pinned --threads (partitioning is
// deterministic in n_threads). The checkpoint cadence is also the resume safety
// margin: the last checkpoint lags the crash, so a resumed slow-hash job
// re-tests an overlap region (never gaps) and the persistent hashcat potfile
// makes the re-test cheap.
// ============================================================================

const CKPT_VERSION: u32 = 1;

/// One worker's resumable DFS position.
#[derive(Clone, Debug)]
struct ThreadCkpt {
    target_level: u32,
    init_idx:     usize,
    prefix:       Vec<u32>,
    idx_stack:    Vec<usize>,
    emitted:      u64,
}

/// Per-thread slot the writer serializes. `NotStarted` = worker hasn't published
/// (resume restarts its whole partition from target_level 0); `Done` = partition
/// exhausted (resume emits nothing for it).
#[derive(Clone, Debug)]
enum CkptSlot {
    NotStarted,
    InProgress(ThreadCkpt),
    Done,
}

fn csv_join<T: std::fmt::Display>(v: &[T]) -> String {
    v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
}
fn parse_csv_u32(s: &str) -> Result<Vec<u32>> {
    if s.is_empty() { return Ok(vec![]); }
    s.split(',').map(|x| x.trim().parse::<u32>().context("parse u32 list")).collect()
}
fn parse_csv_usize(s: &str) -> Result<Vec<usize>> {
    if s.is_empty() { return Ok(vec![]); }
    s.split(',').map(|x| x.trim().parse::<usize>().context("parse usize list")).collect()
}

/// Atomically write the full checkpoint (header + one section per thread).
fn write_checkpoint(path: &Path, fingerprint: &str, slots: &[CkptSlot]) -> Result<()> {
    use std::fmt::Write as _;
    let mut s = String::new();
    writeln!(s, "# tokenov checkpoint").unwrap();
    writeln!(s, "version={}", CKPT_VERSION).unwrap();
    writeln!(s, "threads={}", slots.len()).unwrap();
    writeln!(s, "fingerprint={}", fingerprint).unwrap();
    for (i, slot) in slots.iter().enumerate() {
        writeln!(s, "[thread={}]", i).unwrap();
        match slot {
            CkptSlot::NotStarted    => { writeln!(s, "status=not_started").unwrap(); }
            CkptSlot::Done          => { writeln!(s, "status=done").unwrap(); }
            CkptSlot::InProgress(c) => {
                writeln!(s, "status=in_progress").unwrap();
                writeln!(s, "target_level={}", c.target_level).unwrap();
                writeln!(s, "init_idx={}", c.init_idx).unwrap();
                writeln!(s, "emitted={}", c.emitted).unwrap();
                writeln!(s, "prefix={}", csv_join(&c.prefix)).unwrap();
                writeln!(s, "idx_stack={}", csv_join(&c.idx_stack)).unwrap();
            }
        }
    }
    let tmp = { let mut t = path.as_os_str().to_owned(); t.push(".tmp"); PathBuf::from(t) };
    {
        let mut f = File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        f.write_all(s.as_bytes())?;
        f.sync_data().ok();
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

struct CheckpointFile {
    fingerprint: String,
    slots:       Vec<CkptSlot>,
}

fn read_checkpoint(path: &Path) -> Result<CheckpointFile> {
    let txt = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut fingerprint = String::new();
    let mut threads: usize = 0;
    let mut slots: Vec<CkptSlot> = Vec::new();
    // Per-section accumulator for the [thread=N] currently being parsed.
    let mut sec_status: Option<String> = None;
    let mut sec_tl = 0u32; let mut sec_ii = 0usize; let mut sec_em = 0u64;
    let mut sec_prefix: Vec<u32> = vec![]; let mut sec_idx: Vec<usize> = vec![];
    let flush = |status: &Option<String>, tl, ii, em, prefix: &Vec<u32>, idx: &Vec<usize>,
                 slots: &mut Vec<CkptSlot>| -> Result<()> {
        match status.as_deref() {
            None => {}
            Some("not_started") => slots.push(CkptSlot::NotStarted),
            Some("done")        => slots.push(CkptSlot::Done),
            Some("in_progress") => slots.push(CkptSlot::InProgress(ThreadCkpt {
                target_level: tl, init_idx: ii, emitted: em,
                prefix: prefix.clone(), idx_stack: idx.clone(),
            })),
            Some(other) => bail!("checkpoint: unknown status '{}'", other),
        }
        Ok(())
    };
    for line in txt.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        if line.starts_with("[thread=") {
            flush(&sec_status, sec_tl, sec_ii, sec_em, &sec_prefix, &sec_idx, &mut slots)?;
            sec_status = None; sec_tl = 0; sec_ii = 0; sec_em = 0;
            sec_prefix.clear(); sec_idx.clear();
            continue;
        }
        let (k, v) = line.split_once('=').ok_or_else(|| anyhow!("checkpoint: bad line '{}'", line))?;
        match k {
            "version" => { let ver: u32 = v.parse()?; if ver != CKPT_VERSION {
                bail!("checkpoint version {} != supported {}", ver, CKPT_VERSION); } }
            "threads"      => threads = v.parse()?,
            "fingerprint"  => fingerprint = v.to_string(),
            "status"       => sec_status = Some(v.to_string()),
            "target_level" => sec_tl = v.parse()?,
            "init_idx"     => sec_ii = v.parse()?,
            "emitted"      => sec_em = v.parse()?,
            "prefix"       => sec_prefix = parse_csv_u32(v)?,
            "idx_stack"    => sec_idx = parse_csv_usize(v)?,
            _              => {} // tolerate unknown keys
        }
    }
    flush(&sec_status, sec_tl, sec_ii, sec_em, &sec_prefix, &sec_idx, &mut slots)?;
    if threads != 0 && slots.len() != threads {
        bail!("checkpoint: header threads={} but found {} thread sections", threads, slots.len());
    }
    Ok(CheckpointFile { fingerprint, slots })
}

// ============================================================================
// Session registry. Track the last MAX_SESSIONS checkpointed runs so
// the user can `--sessions` to list them and resume the one they want. A "session"
// is one `generate` run with --checkpoint-file (the checkpoint is its resume
// handle). One file per session under the state dir; ring-buffer at MAX_SESSIONS.
// ============================================================================

// A session record is ~500 bytes today (a few KB once the checkpoint moves into it),
// so retaining a deep history costs single-digit MB. Keep it large: evicting a record
// still holding a live resume position loses the run, and a short throwaway run must
// never be able to push out a long one.
const MAX_SESSIONS: usize = 1000;

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn sessions_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_STATE_HOME").map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("state")))?;
    let d = base.join("tokenov").join("sessions");
    std::fs::create_dir_all(&d).ok()?;
    Some(d)
}

/// Rolling default checkpoint path used when fast-mode generation checkpoints
/// without an explicit `--checkpoint-file`. One "last run" slot in the state dir;
/// name a `--checkpoint-file` to keep several tasks in parallel. Falls back to the
/// current directory if neither XDG_STATE_HOME nor HOME is set.
fn default_checkpoint_path() -> PathBuf {
    let base = std::env::var_os("XDG_STATE_HOME").map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("state")))
        .unwrap_or_else(|| PathBuf::from("."));
    let d = base.join("tokenov");
    let _ = std::fs::create_dir_all(&d);
    d.join("generate.state")
}

#[derive(Clone, Debug)]
struct SessionRecord {
    id:              String,
    started_at:      u64,
    updated_at:      u64,
    pid:             u32,
    model:           String,
    count:           String,
    threads:         usize,
    checkpoint_file: String,
    status:          String, // "running" | "done"
    emitted:         u64,
    argv:            Vec<String>, // original generate args (after the binary name)
}

fn session_path(id: &str) -> Option<PathBuf> { sessions_dir().map(|d| d.join(format!("{}.session", id))) }

fn write_session(rec: &SessionRecord) -> Result<()> {
    use std::fmt::Write as _;
    let path = session_path(&rec.id).ok_or_else(|| anyhow!("no session dir (set HOME or XDG_STATE_HOME)"))?;
    let mut s = String::new();
    writeln!(s, "id={}", rec.id).unwrap();
    writeln!(s, "started_at={}", rec.started_at).unwrap();
    writeln!(s, "updated_at={}", rec.updated_at).unwrap();
    writeln!(s, "pid={}", rec.pid).unwrap();
    writeln!(s, "model={}", rec.model).unwrap();
    writeln!(s, "count={}", rec.count).unwrap();
    writeln!(s, "threads={}", rec.threads).unwrap();
    writeln!(s, "checkpoint_file={}", rec.checkpoint_file).unwrap();
    writeln!(s, "status={}", rec.status).unwrap();
    writeln!(s, "emitted={}", rec.emitted).unwrap();
    for a in &rec.argv { writeln!(s, "arg={}", a).unwrap(); }
    let tmp = { let mut t = path.as_os_str().to_owned(); t.push(".tmp"); PathBuf::from(t) };
    { let mut f = File::create(&tmp)?; f.write_all(s.as_bytes())?; f.sync_data().ok(); }
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

fn read_session(path: &Path) -> Result<SessionRecord> {
    let txt = std::fs::read_to_string(path)?;
    let mut r = SessionRecord {
        id: String::new(), started_at: 0, updated_at: 0, pid: 0, model: String::new(),
        count: String::new(), threads: 0, checkpoint_file: String::new(),
        status: String::new(), emitted: 0, argv: Vec::new(),
    };
    for line in txt.lines() {
        let (k, v) = match line.split_once('=') { Some(x) => x, None => continue };
        match k {
            "id" => r.id = v.into(), "started_at" => r.started_at = v.parse().unwrap_or(0),
            "updated_at" => r.updated_at = v.parse().unwrap_or(0), "pid" => r.pid = v.parse().unwrap_or(0),
            "model" => r.model = v.into(), "count" => r.count = v.into(),
            "threads" => r.threads = v.parse().unwrap_or(0), "checkpoint_file" => r.checkpoint_file = v.into(),
            "status" => r.status = v.into(), "emitted" => r.emitted = v.parse().unwrap_or(0),
            "arg" => r.argv.push(v.into()), _ => {}
        }
    }
    Ok(r)
}

fn load_sessions() -> Vec<SessionRecord> {
    let dir = match sessions_dir() { Some(d) => d, None => return vec![] };
    let mut recs: Vec<SessionRecord> = std::fs::read_dir(&dir).ok().into_iter().flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |x| x == "session"))
        .filter_map(|e| read_session(&e.path()).ok())
        .collect();
    recs.sort_by(|a, b| b.started_at.cmp(&a.started_at)); // newest first
    recs
}

/// True if a pid is alive (so a `status=running` record whose pid is dead = interrupted).
fn pid_alive(pid: u32) -> bool {
    pid != 0 && Path::new(&format!("/proc/{}", pid)).exists()
}

/// Create a session record at gen start and prune to the newest MAX_SESSIONS.
fn session_start(model: &Path, count: Option<u64>, threads: usize, checkpoint_file: &Path) -> Option<String> {
    let id = format!("s{}-{}", now_secs(), std::process::id());
    let argv: Vec<String> = std::env::args().skip(1)
        // drop any prior --resume-state/--resume-session so re-resume re-derives it cleanly
        .scan(false, |skip_next, a| {
            if *skip_next { *skip_next = false; return Some(None); }
            if a == "--resume-state" || a == "--resume-session" { *skip_next = true; return Some(None); }
            if a.starts_with("--resume-state=") || a.starts_with("--resume-session=") { return Some(None); }
            Some(Some(a))
        }).flatten().collect();
    let rec = SessionRecord {
        id: id.clone(), started_at: now_secs(), updated_at: now_secs(), pid: std::process::id(),
        model: model.display().to_string(),
        count: count.map(|c| c.to_string()).unwrap_or_else(|| "unbounded".into()),
        threads, checkpoint_file: checkpoint_file.display().to_string(),
        status: "running".into(), emitted: 0, argv,
    };
    write_session(&rec).ok()?;
    // Ring-buffer: keep newest MAX_SESSIONS, delete older.
    let all = load_sessions();
    for old in all.into_iter().skip(MAX_SESSIONS) {
        if let Some(p) = session_path(&old.id) { let _ = std::fs::remove_file(p); }
    }
    Some(id)
}

fn session_update(id: &str, emitted: u64, status: &str) {
    if let Some(p) = session_path(id) {
        if let Ok(mut rec) = read_session(&p) {
            rec.emitted = emitted; rec.status = status.into(); rec.updated_at = now_secs();
            let _ = write_session(&rec);
        }
    }
}

fn fmt_ago(secs: u64) -> String {
    let now = now_secs();
    let d = now.saturating_sub(secs);
    if d < 90 { format!("{}s ago", d) }
    else if d < 5400 { format!("{}m ago", d / 60) }
    else if d < 129600 { format!("{}h ago", d / 3600) }
    else { format!("{}d ago", d / 86400) }
}

/// How many sessions `--sessions` prints. Retention (MAX_SESSIONS) is deliberately
/// much larger — a resumable position should outlive the listing that shows it — so
/// the default view is trimmed to the recent ones and the rest stay on disk.
const SESSION_LIST_LIMIT: usize = 20;

/// `--sessions`: print the most recent sessions, newest first.
fn list_sessions() -> Result<()> {
    let recs = load_sessions();
    if recs.is_empty() {
        println!("no sessions recorded (sessions are created for --checkpoint-file runs)");
        return Ok(());
    }
    println!("{:<22} {:<10} {:>10} {:<8} {:<10} model", "ID", "WHEN", "EMITTED", "THREADS", "STATUS");
    let shown = recs.len().min(SESSION_LIST_LIMIT);
    for r in recs.iter().take(shown) {
        let status = if r.status == "done" { "done".to_string() }
            else if pid_alive(r.pid) { "running".to_string() }
            else { "interrupted".to_string() };
        let model = Path::new(&r.model).file_name().and_then(|s| s.to_str()).unwrap_or(&r.model);
        println!("{:<22} {:<10} {:>10} {:<8} {:<10} {}",
            r.id, fmt_ago(r.started_at), r.emitted, r.threads, status, model);
    }
    if recs.len() > shown {
        println!("... and {} older (all still resumable)", recs.len() - shown);
    }
    println!("\nresume one with:  tokenov --resume-session <ID> | hashcat …");
    Ok(())
}

/// `--resume-session <id>`: re-exec the original command + --resume-state <checkpoint>.
fn resume_session(id: &str) -> Result<()> {
    let p = session_path(id).ok_or_else(|| anyhow!("no session dir"))?;
    if !p.exists() { bail!("no session '{}' (see --sessions)", id); }
    let rec = read_session(&p)?;
    if rec.status == "done" {
        bail!("session '{}' already completed (status=done); nothing to resume", id);
    }
    if !Path::new(&rec.checkpoint_file).exists() {
        bail!("session '{}' checkpoint file missing: {}", id, rec.checkpoint_file);
    }
    let exe = std::env::current_exe().context("current_exe")?;
    log_msg(&format!("[sessions] resuming {} → {} {} --resume-state {}",
        id, exe.display(), rec.argv.join(" "), rec.checkpoint_file));
    use std::os::unix::process::CommandExt;
    let err = std::process::Command::new(exe)
        .args(&rec.argv)
        .arg("--resume-state").arg(&rec.checkpoint_file)
        .exec(); // replaces this process on success
    Err(anyhow!("exec failed: {}", err))
}

/// Convert a (negative) transition log-prob to its discrete level.
/// All transitions have lp < 0, so level ≥ 1; lp ≥ 0 is clamped to 0.
#[inline]
fn lp_to_level(lp: f32) -> u32 {
    if lp >= 0.0 { 0 } else { (-lp * LEVEL_SCALE).ceil() as u32 }
}

#[inline]
fn context_from_prefix(prefix: &[u32], start_id: u32) -> (u32, u32) {
    match prefix.len() {
        0 => (start_id, start_id),
        1 => (start_id, prefix[0]),
        n => (prefix[n - 2], prefix[n - 1]),
    }
}

// `get_children` moved to per-variant modules. Dispatch via `&dyn Variant`.

/// Automatic child_cache mode selection.
///
/// Three-way: FULL (build the whole resident cache) if it comfortably fits;
/// else BOUNDED (a per-thread partial cache sized to the leftover RAM budget)
/// if that budget buys a worthwhile cache; else LAZY (pure recompute). Manual
/// flags override. Projects cache size from the model shape without building it,
/// so a large-context model (e.g. COMB) never OOMs and the user never needs a
/// flag. `n_threads` sizes the per-thread bounded cap.
fn decide_child_cache_mode(args: &GenerateArgs, enum_model: &EnumModel, n_threads: usize) -> CacheMode {
    if args.lazy_children {
        log_msg("[gen] child_cache: LAZY (forced by --lazy)");
        return CacheMode::Lazy;
    }
    if args.force_child_cache {
        log_msg("[gen] child_cache: FULL (forced by --force-child-cache)");
        return CacheMode::Full;
    }
    if let Some(cap) = args.bounded_cap {
        let cap = cap.max(1);
        log_msg(&format!("[gen] child_cache: BOUNDED (forced by --bounded-cap {})", cap));
        return CacheMode::Bounded(cap);
    }
    // Project the resident FULL cache size from the model shape, WITHOUT building
    // it. The cache is FxHashMap<Ctx, Box<[(u32,f32)]>>:
    //   - one 8-byte (u32,f32) slot per child: trigram children (T) plus up to
    //     MAX_KN_BIGRAM_CHILDREN KN bigram-backoff children per KN context (E).
    //   - per-context overhead: boxed-slice fat ptr + map slot + malloc header.
    // Use the 200-child upper bound for E — a deliberate over-estimate so the
    // error biases toward BOUNDED/LAZY (safe), never toward a cache that OOMs.
    let contexts = enum_model.trigram.len() as u64;
    let t_children: u64 = enum_model.trigram.par_iter()
        .map(|(_, v)| v.len() as u64).sum();
    let e_children: u64 = if enum_model.is_kn {
        contexts.saturating_mul(MAX_KN_BIGRAM_CHILDREN as u64)
    } else { 0 };
    const PER_CTX_OVERHEAD: u64 = 48; // box fat ptr(16) + map slot(~24) + malloc header
    let total_child_slots = t_children + e_children;
    let projected = 8u64.saturating_mul(total_child_slots)
        + contexts.saturating_mul(PER_CTX_OVERHEAD);
    // Average bytes per cached context (for sizing the bounded cap).
    let bytes_per_ctx = (projected / contexts.max(1)).max(1);

    // Effective RSS cap — resolved exactly as the mem_monitor does below.
    let cap_kb = {
        let from_flag = args.max_rss_gb.map(|gb| (gb * 1024.0 * 1024.0) as u64);
        let from_env  = std::env::var("TOKENOV_MAX_RSS_GB").ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map(|gb| (gb * 1024.0 * 1024.0) as u64);
        from_flag.or(from_env).unwrap_or_else(MemMonitorConfig::default_rss_cap_kb)
    };
    let snap = ProcMemReader.snapshot().ok();
    let avail   = snap.as_ref().map(|s| s.avail_kb * 1024).unwrap_or(u64::MAX);
    let cur_rss = snap.as_ref().map(|s| s.rss_kb * 1024).unwrap_or(0);
    let cap     = cap_kb.saturating_mul(1024); // 0 = disabled
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);

    // Spendable RAM budget for a cache: leave >=40% of available as headroom, and
    // stay under ~85% of the RSS cap (so the mem_monitor won't self-abort).
    const AVAIL_HEADROOM_FRAC: f64 = 0.60;
    const CAP_HEADROOM_FRAC:   f64 = 0.85;
    let budget_avail = (AVAIL_HEADROOM_FRAC * avail as f64) as u64;
    let budget_cap = if cap == 0 { u64::MAX }
        else { ((CAP_HEADROOM_FRAC * cap as f64) as u64).saturating_sub(cur_rss) };
    let budget = budget_avail.min(budget_cap);

    if projected <= budget {
        log_msg(&format!(
            "[gen] child_cache: FULL (auto) — projected {:.1} GiB fits budget {:.1} GiB \
             (avail {:.1}, cur RSS {:.1}, cap {:.1} GiB, {} ctx)",
            gib(projected), gib(budget), gib(avail), gib(cur_rss), gib(cap), contexts));
        return CacheMode::Full;
    }

    // BOUNDED sizing. The bounded cache stores Arc<[(u32,f32)]> in a two-generation
    // map (≤2*cap resident) per thread, with more per-entry overhead than the FULL
    // Box map (Arc control block + two hashmap slots). Fold that in with a 1.6x
    // factor, then split the total budget across threads and the 2 generations.
    const BOUNDED_ENTRY_SLACK: f64 = 1.6;
    let eff_bytes_per_ctx = ((bytes_per_ctx as f64) * BOUNDED_ENTRY_SLACK) as u64;
    let total_cacheable = budget / eff_bytes_per_ctx.max(1);      // entries the budget affords
    // Per-thread `cap` (young fills to this before rotating; resident ≤ 2*cap).
    let per_thread_cap = total_cacheable / (2 * n_threads as u64).max(1);

    // Worth it only if a thread can hold a non-trivial hot set.
    const MIN_BOUNDED_CAP: u64 = 50_000; // entries/thread; below this, just go lazy
    if per_thread_cap >= MIN_BOUNDED_CAP {
        // Resident estimate for logging: 2 * per_thread_cap * n_threads entries.
        let resident = 2 * per_thread_cap * n_threads as u64 * eff_bytes_per_ctx;
        log_msg(&format!(
            "[gen] child_cache: BOUNDED (auto) — FULL {:.1} GiB > budget {:.1} GiB; \
             per-thread cap {} ctx (~{:.1} GiB resident across {} threads) \
             (avail {:.1}, cur RSS {:.1}, cap {:.1} GiB, {} ctx)",
            gib(projected), gib(budget), per_thread_cap, gib(resident), n_threads,
            gib(avail), gib(cur_rss), gib(cap), contexts));
        CacheMode::Bounded(per_thread_cap as usize)
    } else {
        log_msg(&format!(
            "[gen] child_cache: LAZY (auto) — FULL {:.1} GiB > budget {:.1} GiB and budget \
             affords only {} ctx/thread (< {} min) (avail {:.1}, cur RSS {:.1}, cap {:.1} GiB, {} ctx)",
            gib(projected), gib(budget), per_thread_cap, MIN_BOUNDED_CAP,
            gib(avail), gib(cur_rss), gib(cap), contexts));
        CacheMode::Lazy
    }
}

/// Pre-compute get_children for every context in the trigram model.
/// Shared across threads; thread-local fallback handles rare misses.
fn build_child_cache(enum_model: &EnumModel, variant: &dyn Variant) -> ChildCache {
    // Parallel construction. Each context's children are computed
    // independently via `variant.get_children` (which does its own per-call
    // dedup). Collect into a Vec then assemble the FxHashMap; rayon's collect
    // into FxHashMap isn't directly supported without a custom implementation.
    //
    // Per-entry: `Box<[(u32, f32)]>` (16 B header) — no Arc wrapping. The
    // assembled map is leaked once via `Box::leak`, giving callers `&'static`
    // access without ref-counting. The leaked memory is reclaimed by the OS
    // on process exit.
    let entries: Vec<(Ctx, Box<[(u32, f32)]>)> = enum_model.trigram
        .par_iter()
        .map(|(&ctx, _)| {
            let b = (ctx & 0xFFFFFFFF) as u32;
            let v = variant.get_children(enum_model, ctx, b);
            (ctx, v.into_boxed_slice())
        })
        .collect();
    let mut map: ChildCacheMap =
        FxHashMap::with_capacity_and_hasher(entries.len(), Default::default());
    for (ctx, v) in entries {
        map.insert(ctx, v);
    }
    Box::leak(Box::new(map))
}

#[allow(clippy::too_many_arguments)]
/// Run the level-sweep enumeration, calling `emit(level, sort_key, bytes)` for
/// each candidate. The closure decides what to do with the emission — write
/// directly to a `Sink` (single-thread mode) or send a `MergeItem` to the
/// merger (multi-thread mode). Returning `Err` from the closure aborts
/// enumeration (used by the merger when its consumer hangs up).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn enumerate_to_sink<F, H>(
    enum_model: &EnumModel,
    variant: &dyn Variant,
    child_cache: ChildCache,
    // Full: hits come from the shared `child_cache`. Bounded/Lazy: `child_cache`
    // is empty and children are recomputed on demand (a per-thread bounded cache
    // memoizes hot ones in Bounded mode). All modes are byte-identical — the
    // cache only trades RAM for CPU.
    cache_mode: CacheMode,
    decode_table: &[Vec<u8>],
    kind: DecodeKind,
    start_id: u32,
    end_id: u32,
    initial_states: Vec<HeapEntry>,
    max_tokens: usize,
    min_tokens: usize,
    min_len: usize,
    max_len: usize,
    // Floor on the level sweep (`--min-level`): shells below it are never walked,
    // so the stream is an exact suffix of the same run at 0. Ignored when `resume`
    // is Some — the checkpoint's own target_level already sits at or past the floor.
    min_level: u32,
    target_count: u64,
    enterprise: bool,
    case_masks: &[CaseMask],
    mut emit: F,
    thread_label: &str,
    // Crash-resume (fast mode): `resume` jumps this worker to a saved DFS position
    // and continues; `checkpoint_every`+`on_checkpoint` publish the live position
    // on that cadence. All three are inert (None/None/no-op) for non-resumable runs.
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
    let mut iter_ctr: u64 = 0;
    let (resume_tl, resume_ii) = resume.as_ref()
        .map(|r| (r.target_level, r.init_idx)).unwrap_or((min_level, 0));
    let mut resume_taken = resume.is_none();
    let mut emitted: u64 = resume.as_ref().map(|r| r.emitted).unwrap_or(0);
    let mut decode_buf: Vec<u8> = Vec::with_capacity(64); // case-mask path only

    // #2 + TokenMeta: incremental decoded-byte stack mirroring `prefix`, so
    // terminals emit a slice instead of re-decoding the whole prefix.
    // byte_lens[i] = bytes_buf.len() before prefix[i]'s bytes (truncate target on
    // pop). nl_count = number of prefix tokens whose decode contains \n/\r, so the
    // common case (0) skips the newline scan entirely. byte_lens always has
    // exactly prefix.len() entries.
    let mut bytes_buf: Vec<u8> = Vec::with_capacity(64);
    let mut byte_lens: Vec<usize> = Vec::with_capacity(max_tokens + 2);
    let mut nl_count: usize;  // reset per init-state below before any read
    let token_has_nl: Vec<bool> = decode_table.iter()
        .map(|b| b.iter().any(|&c| c == b'\n' || c == b'\r'))
        .collect();

    // child_cache is pre-built and shared (`'static`); covers all trigram
    // contexts. local_cache handles the rare KN bigram-only contexts not in
    // child_cache. It is BOUNDED (cleared past LOCAL_CACHE_MAX) and Arc-backed
    // so memory stays flat even on a KN model with many distinct bigram-only
    // ctxs — the old code `Box::leak`'d each miss, which grew unbounded and
    // could OOM at high cap. The Arc keeps a slice alive
    // for any frame still referencing it even after a clear evicts it, so
    // eviction can never dangle. Output is unaffected (a miss recomputes the
    // identical children).
    const LOCAL_CACHE_MAX: usize = 262144;
    let mut local_cache: FxHashMap<Ctx, Arc<[(u32, f32)]>> = FxHashMap::default();
    let mut local_misses: usize = 0;
    // Per-thread bounded cache — only allocated in Bounded mode.
    let mut bounded: Option<BoundedChildCache> = match cache_mode {
        CacheMode::Bounded(cap) => Some(BoundedChildCache::new(cap)),
        _ => None,
    };
    // Returns a `Children` (all three variants Deref to `&[(u32,f32)]`):
    //   - Full  : `Ref` into the shared child_cache; the rare KN bigram-only
    //             miss goes through a BOUNDED per-thread `local_cache` as a
    //             `Shared` Arc (bounded, non-leaking).
    //   - Bounded: `Shared` Arc from the per-thread bounded cache (hit = clone;
    //             miss = recompute + insert + maybe-evict). Arc keeps the slice
    //             alive across the frame's lifetime even if evicted meanwhile.
    //   - Lazy  : `Owned` recompute, freed when the frame pops (no cache/leak).
    // All three yield identical bytes — the cache never changes output.
    macro_rules! cached_get {
        ($ctx:expr_2021, $b:expr_2021) => {{
            let ctx = $ctx;
            let b   = $b;
            match cache_mode {
                CacheMode::Bounded(_) => {
                    let a = bounded.as_mut().unwrap()
                        .get_or(ctx, || variant.get_children(enum_model, ctx, b));
                    Children::Shared(a)
                }
                CacheMode::Lazy => {
                    Children::Owned(variant.get_children(enum_model, ctx, b).into_boxed_slice())
                }
                CacheMode::Full => {
                    if let Some(ch) = child_cache.get(&ctx) {
                        Children::Ref(&ch[..])
                    } else if let Some(a) = local_cache.get(&ctx) {
                        Children::Shared(a.clone())
                    } else {
                        local_misses += 1;
                        if local_cache.len() >= LOCAL_CACHE_MAX { local_cache.clear(); }
                        let a: Arc<[(u32, f32)]> =
                            Arc::from(variant.get_children(enum_model, ctx, b).into_boxed_slice());
                        local_cache.insert(ctx, a.clone());
                        Children::Shared(a)
                    }
                }
            }
        }};
    }

    // Sort initial states by level (ascending) so the `break` below fires early.
    let mut states = initial_states;
    states.sort_unstable_by_key(|s| lp_to_level(s.log_prob));

    // Level sweep: process all candidates at level L before moving to L+1.
    // For each pass, run a DFS that:
    //   - prunes branches whose accumulated level exceeds target_level
    //   - emits terminals where accumulated level == target_level exactly
    //   - skips terminals at lower levels (already emitted in prior passes)
    // Memory: O(max_tokens × avg_branching) stack + ctx_cache.
    if let Some(r) = &resume {
        if r.init_idx >= states.len() {
            bail!("{}resume: init_idx {} >= partition size {} (--threads mismatch vs checkpoint?)",
                thread_label, r.init_idx, states.len());
        }
    }
    // Shell this worker was in when it stopped — reported on the DONE line so an
    // operator who ended a run with --count can see where to point --min-level.
    let mut last_level = resume_tl;
    'main: for target_level in resume_tl..=LEVEL_MAX {
        last_level = target_level;
        // On resume the first pass starts at the saved init_idx (earlier inits in
        // that pass already completed pre-checkpoint); later passes start at 0.
        let init_start = if target_level == resume_tl { resume_ii } else { 0 };
        for ii in init_start..states.len() {
            let init = &states[ii];
            let init_level = lp_to_level(init.log_prob);
            if init_level > target_level {
                break; // states are sorted; no further state can contribute here
            }

            let resuming_here = !resume_taken && target_level == resume_tl && ii == resume_ii;
            let mut prefix: Vec<u32>;
            let mut stack: Vec<DfsFrame>;

            if resuming_here {
                // O(depth) restore: rebuild prefix + byte stack, then one DfsFrame
                // per saved idx by recomputing each frame's children from the prefix
                // context (the child slices aren't serialized). base_lp/acc_level are
                // re-accumulated along the path; the child descended into at each
                // parent sits at idx-1 (idx points one past it after `frame.idx += 1`).
                let r = resume.as_ref().unwrap();
                resume_taken = true;
                if r.idx_stack.is_empty() { continue; } // nothing to restore; defensive
                let pl = init.prefix_len as usize;
                if r.prefix.len() < pl || r.prefix[..pl] != init.prefix[..pl] {
                    bail!("{}resume: checkpoint prefix doesn't extend init {} (model/threads mismatch?)",
                        thread_label, ii);
                }
                if r.prefix.len() != pl + r.idx_stack.len() - 1 {
                    bail!("{}resume: prefix len {} inconsistent with idx_stack len {} (pl={})",
                        thread_label, r.prefix.len(), r.idx_stack.len(), pl);
                }
                prefix = r.prefix.clone();
                bytes_buf.clear(); byte_lens.clear(); nl_count = 0;
                for &id in &prefix {
                    byte_lens.push(bytes_buf.len());
                    if (id as usize) < decode_table.len() {
                        bytes_buf.extend_from_slice(&decode_table[id as usize]);
                    }
                    if (id as usize) < token_has_nl.len() && token_has_nl[id as usize] { nl_count += 1; }
                }
                stack = Vec::with_capacity(r.idx_stack.len());
                let mut acc_lp = init.log_prob;
                let mut acc_lvl = init_level;
                {
                    let bval = prefix[..pl].last().copied().unwrap_or(start_id);
                    let (a, b) = context_from_prefix(&prefix[..pl], start_id);
                    let ch = cached_get!(pack(a, b), bval);
                    stack.push(DfsFrame { children: ch, idx: r.idx_stack[0], base_lp: acc_lp, acc_level: acc_lvl });
                }
                for k in 1..r.idx_stack.len() {
                    let ci = r.idx_stack[k - 1].checked_sub(1).ok_or_else(|| anyhow!(
                        "{}resume: frame {} idx underflow", thread_label, k - 1))?;
                    let (cid, clp) = stack[k - 1].children.get(ci).copied().ok_or_else(|| anyhow!(
                        "{}resume: child idx {} out of range in frame {}", thread_label, ci, k - 1))?;
                    if cid != prefix[pl + k - 1] {
                        bail!("{}resume: prefix/idx mismatch at depth {} ({} != {})",
                            thread_label, k, cid, prefix[pl + k - 1]);
                    }
                    acc_lp += clp;
                    acc_lvl += lp_to_level(clp);
                    let d = pl + k;
                    let bval = prefix[d - 1];
                    let (a, b) = context_from_prefix(&prefix[..d], start_id);
                    let ch = cached_get!(pack(a, b), bval);
                    stack.push(DfsFrame { children: ch, idx: r.idx_stack[k], base_lp: acc_lp, acc_level: acc_lvl });
                }
            } else {
                prefix = init.prefix[..init.prefix_len as usize].to_vec();
                // (#2) seed the byte stack from the init prefix — one byte_lens entry
                // per prefix token, mirroring the bounds guard used at decode time.
                bytes_buf.clear(); byte_lens.clear(); nl_count = 0;
                for &id in &prefix {
                    byte_lens.push(bytes_buf.len());
                    if (id as usize) < decode_table.len() {
                        bytes_buf.extend_from_slice(&decode_table[id as usize]);
                    }
                    if (id as usize) < token_has_nl.len() && token_has_nl[id as usize] { nl_count += 1; }
                }
                let b_val0 = prefix.last().copied().unwrap_or(start_id);
                let (a0, b0) = context_from_prefix(&prefix, start_id);
                let root_children = cached_get!(pack(a0, b0), b_val0);
                // (#1) the frame-skip prune below is valid only if children are in
                // descending-lp (nondecreasing-level) order.
                debug_assert!(
                    root_children.windows(2).all(|w| lp_to_level(w[0].1) <= lp_to_level(w[1].1)),
                    "child_cache children must be in descending-lp order",
                );
                stack = vec![DfsFrame {
                    children:  root_children,
                    idx:       0,
                    base_lp:   init.log_prob,
                    acc_level: init_level,
                }];
            }

            'dfs: loop {
                // Publish a checkpoint on cadence (cheap iter-gated wall-clock check).
                iter_ctr = iter_ctr.wrapping_add(1);
                if let Some(dur) = checkpoint_every {
                    if (iter_ctr & ((1 << 18) - 1)) == 0 && last_ckpt.elapsed() >= dur {
                        let idx_stack: Vec<usize> = stack.iter().map(|f| f.idx).collect();
                        on_checkpoint(&ThreadCkpt {
                            target_level, init_idx: ii,
                            prefix: prefix.clone(), idx_stack, emitted,
                        });
                        last_ckpt = Instant::now();
                    }
                }
                let frame = match stack.last_mut() {
                    Some(f) => f,
                    None => break 'dfs,
                };
                if frame.idx >= frame.children.len() {
                    stack.pop();
                    if let Some(popped) = prefix.pop() {
                        // (#2) mirror the pop on the byte stack.
                        if let Some(old) = byte_lens.pop() { bytes_buf.truncate(old); }
                        if (popped as usize) < token_has_nl.len() && token_has_nl[popped as usize] {
                            nl_count -= 1;
                        }
                    }
                    continue 'dfs;
                }

                let (next_id, child_lp) = frame.children[frame.idx];
                frame.idx += 1;
                let new_level = frame.acc_level + lp_to_level(child_lp);

                if new_level > target_level {
                    // Prune: lp_to_level ≥ 1 per transition, so deeper extensions
                    // also exceed target_level. And since children are descending-lp
                    // (nondecreasing level), every remaining sibling in this frame
                    // exceeds it too — skip the whole frame this pass. Deferred
                    // children resurface in a later target_level pass (multi-pass
                    // sweep), so output is unchanged; this just avoids walking the
                    // doomed tail of each frame.
                    frame.idx = frame.children.len();
                    continue 'dfs;
                }

                let new_lp = frame.base_lp + child_lp;

                if next_id == end_id {
                    // Token-count floor (--min-tokens): drop candidates built from
                    // fewer than min_tokens tokens (prefix holds the content tokens;
                    // END is not pushed). Drop-and-continue — not counted toward the
                    // budget, survivors keep rank order. min_tokens==1 is a no-op.
                    if new_level == target_level && prefix.len() >= min_tokens {
                        let sort_key = (-new_lp).to_bits();
                        if case_masks.is_empty() {
                            // Default path — bytes_buf already holds the
                            // decoded prefix (maintained incrementally), so emit a
                            // finalized slice with no re-decode. Byte-identical to
                            // the prior decode-from-scratch path (byte-identical).
                            let range = finalized_range(&bytes_buf, kind);
                            let (rs, re) = (range.start, range.end);
                            let len = re - rs;
                            // Newline scan only when some prefix token actually
                            // carries \n/\r (nl_count>0) — matches the original
                            // post-finalize scan exactly, but skips it in the
                            // overwhelmingly common newline-free case.
                            let has_nl = nl_count > 0
                                && bytes_buf[rs..re].iter().any(|&c| c == b'\n' || c == b'\r');
                            if len > 0 && len >= min_len && len <= max_len && !has_nl {
                                if enterprise {
                                    // Enterprise-policy mode: emit only compliant
                                    // candidates, capitalizing the first byte when
                                    // that is the sole missing class; drop the rest.
                                    // Dropped candidates do NOT count toward
                                    // `emitted`, so the DFS keeps pulling deeper to
                                    // reach the target budget (drop-and-continue).
                                    match enterprise::decide(&bytes_buf[rs..re]) {
                                        enterprise::Decision::AsIs => {
                                            emit(target_level, sort_key, &bytes_buf[rs..re])?;
                                            emitted += 1;
                                            if emitted >= target_count { break 'main; }
                                        }
                                        enterprise::Decision::Cap => {
                                            // Same (level, sort_key) so the
                                            // capitalized form slots into the merger
                                            // exactly where the original would have.
                                            let first = bytes_buf[rs];
                                            bytes_buf[rs] = first.to_ascii_uppercase();
                                            emit(target_level, sort_key, &bytes_buf[rs..re])?;
                                            bytes_buf[rs] = first; // restore prefix buffer
                                            emitted += 1;
                                            if emitted >= target_count { break 'main; }
                                        }
                                        enterprise::Decision::Drop => {}
                                    }
                                } else {
                                    // Emit immediately — no level_emits buffer to avoid OOM at
                                    // large levels. DFS already visits children in desc lp order,
                                    // giving approximate within-level ordering.
                                    emit(target_level, sort_key, &bytes_buf[rs..re])?;
                                    emitted += 1;
                                    if emitted >= target_count { break 'main; }
                                }
                            }
                        } else {
                            // Shape-conditioned path: emit once per case mask, applying
                            // each slot's case op to its token's bytes as it decodes (the
                            // token is the unit of casing). Same (level, sort_key) so all
                            // variants of one candidate merge adjacently.
                            for mask in case_masks {
                                decode_buf.clear();
                                for (pos, &id) in prefix.iter().enumerate() {
                                    if (id as usize) < decode_table.len() {
                                        let start = decode_buf.len();
                                        decode_buf.extend_from_slice(&decode_table[id as usize]);
                                        let end = decode_buf.len();
                                        apply_op(&mut decode_buf[start..end], mask.op_for(pos));
                                    }
                                }
                                finalize_decoded(&mut decode_buf, kind);
                                let len = decode_buf.len();
                                if !decode_buf.is_empty()
                                    && len >= min_len && len <= max_len
                                    && !decode_buf.iter().any(|&c| c == b'\n' || c == b'\r')
                                {
                                    emit(target_level, sort_key, &decode_buf)?;
                                    emitted += 1;
                                    if emitted >= target_count { break 'main; }
                                }
                            }
                        }
                    }
                    // new_level < target_level → emitted in a prior pass; skip.
                } else if prefix.len() < max_tokens {
                    prefix.push(next_id);
                    // (#2) mirror the push on the byte stack.
                    byte_lens.push(bytes_buf.len());
                    if (next_id as usize) < decode_table.len() {
                        bytes_buf.extend_from_slice(&decode_table[next_id as usize]);
                    }
                    if (next_id as usize) < token_has_nl.len() && token_has_nl[next_id as usize] {
                        nl_count += 1;
                    }
                    let (a, b) = context_from_prefix(&prefix, start_id);
                    let ch = cached_get!(pack(a, b), next_id);
                    debug_assert!(
                        ch.windows(2).all(|w| lp_to_level(w[0].1) <= lp_to_level(w[1].1)),
                        "child_cache children must be in descending-lp order",
                    );
                    stack.push(DfsFrame {
                        children:  ch,
                        idx:       0,
                        base_lp:   new_lp,
                        acc_level: new_level,
                    });
                }
            }
        }

        if last_log.elapsed().as_secs() >= 30 {
            log_msg(&format!("{}  level={} emitted={} local_misses={} elapsed={:.0}s",
                thread_label, target_level, emitted, local_misses, t0.elapsed().as_secs_f64()));
            last_log = Instant::now();
        }
    }

    let bounded_stats = bounded.as_ref().map(|b| {
        let tot = b.hits + b.misses;
        let hr = if tot > 0 { 100.0 * b.hits as f64 / tot as f64 } else { 0.0 };
        format!(" bounded[hits={} misses={} hit_rate={:.1}% resident={}]",
            b.hits, b.misses, hr, b.young.len() + b.old.len())
    }).unwrap_or_default();
    log_msg(&format!("{}[gen] DONE emitted={} local_misses={}{} in {:.0}s level={}",
        thread_label, emitted, local_misses, bounded_stats, t0.elapsed().as_secs_f64(),
        last_level));
    Ok(())
}

fn build_seeds(
    enum_model: &EnumModel,
    entry_seqs: &[Vec<u32>],
    _w_set: &FxHashMap<u32, ()>,
    seed_mode: SeedMode,
    start_id: u32,
    max_tokens: usize,
) -> Vec<HeapEntry> {
    let mut seen: FxHashMap<u64, ()> = FxHashMap::default();
    let mut seeds: Vec<HeapEntry> = Vec::new();
    let mut zero_prob_skipped = 0u64;
    let mut overlong_skipped = 0u64;

    // Joint log-prob of the next-token transition. Uses the trigram if it has
    // the transition; falls back to bigram-back-off (with log_lambda penalty)
    // for KN models. Returns None only if no path through any tier exists.
    let log_p_step = |a: u32, b: u32, t: u32| -> Option<f32> {
        let ctx = pack(a, b);
        if let Some(children) = enum_model.trigram.get(&ctx) {
            if let Some(lp) = children.iter().find(|(id, _)| *id == t).map(|(_, lp)| *lp) {
                return Some(lp);
            }
            // Fall through to bigram-back-off
            if enum_model.is_kn {
                let log_lam = enum_model.log_lambda.get(&ctx).copied().unwrap_or(0.0);
                if let Some(bi) = enum_model.bigram.get(&b) {
                    if let Some(lp_cont) = bi.iter().find(|(id, _)| *id == t).map(|(_, lp)| *lp) {
                        return Some(log_lam + lp_cont);
                    }
                }
            }
            None
        } else if enum_model.is_kn {
            // Unseen trigram context — pure bigram back-off (no lambda penalty
            // because there's no trigram mass to discount).
            enum_model.bigram.get(&b)
                .and_then(|bi| bi.iter().find(|(id, _)| *id == t).map(|(_, lp)| *lp))
        } else {
            None
        }
    };

    match seed_mode {
        SeedMode::Entry => {
            for seq in entry_seqs {
                if seq.is_empty() { continue; }
                if seq.len() > max_tokens {
                    overlong_skipped += 1;
                    continue;
                }
                // Compute joint log-prob along the prefix
                let mut a = start_id;
                let mut b = start_id;
                let mut lp: f32 = 0.0;
                let mut ok = true;
                for &t in seq {
                    match log_p_step(a, b, t) {
                        Some(step_lp) => { lp += step_lp; a = b; b = t; }
                        None => { ok = false; break; }
                    }
                }
                if !ok || !lp.is_finite() {
                    zero_prob_skipped += 1;
                    continue;
                }
                // Dedup
                let mut hkey = 0u64;
                for &t in seq { hkey = hkey.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(t as u64); }
                if seen.contains_key(&hkey) { continue; }
                seen.insert(hkey, ());
                let mut prefix = [0u32; 32];
                for (i, &t) in seq.iter().enumerate() { prefix[i] = t; }
                seeds.push(HeapEntry { log_prob: lp, prefix_len: seq.len() as u8, prefix });
            }
        }
        SeedMode::Token => {
            // One seed per unique token across all entries' tokenizations
            let mut tokens: FxHashMap<u32, ()> = FxHashMap::default();
            for seq in entry_seqs {
                for &t in seq { tokens.insert(t, ()); }
            }
            for &t in tokens.keys() {
                if let Some(lp) = log_p_step(start_id, start_id, t) {
                    if !lp.is_finite() { zero_prob_skipped += 1; continue; }
                    let mut prefix = [0u32; 32];
                    prefix[0] = t;
                    seeds.push(HeapEntry { log_prob: lp, prefix_len: 1, prefix });
                } else {
                    zero_prob_skipped += 1;
                }
            }
        }
    }

    if zero_prob_skipped > 0 {
        log_msg(&format!("[seeds] {} skipped (zero joint prob / unreachable transition)", zero_prob_skipped));
    }
    if overlong_skipped > 0 {
        log_msg(&format!("[seeds] {} skipped (entry > --max-tokens)", overlong_skipped));
    }
    seeds
}

// ============================================================================
// Score — report a candidate's Rarity (surprisal) under the model
// ============================================================================
//
// "Rarity" is the user-facing name; the precise quantity is *surprisal*,
// `-log2 P(candidate)`, reported in bits (positive; higher = rarer = stronger).
// This is the token-level analog of FLA's neural strength meter: tokenov is a
// generative probability model, so a candidate's joint log-prob under the model
// IS its score. `transition_logprob` below is a verbatim copy of the generator's
// per-step ranking transition (`log_p_step` in `build_seeds`), so the scorer is
// consistent with the generator at the TOKEN-PATH level. At the STRING level it
// is only *concordant, not identical*: the generator dedups on the best path it
// reaches under a pruned level-sweep while the scorer uses one greedy
// tokenization, so string-emission order and Rarity order agree ~loosely (~60%
// rank-concordance on a strict top-K probe), not exactly. Closing that gap is
// the deferred segmentation-marginalization work.
//
// Known caveats of the current scorer:
//  - Score is over the tokenizer's canonical (greedy) segmentation; a
//    fully-correct P(string) would marginalize over all segmentations.
//  - Variant A (the default, matching `generate`) has NO unigram tier, so a
//    transition absent from both the trigram and the KN bigram-continuation is a
//    true zero: such candidates get Rarity = +inf / in_vocab=false. This is a
//    real property of the backbone, not a bug — some plausible-looking strings
//    are genuinely unreachable. A finite-everywhere meter would need a unigram
//    floor (score under variant B/E); left as a follow-up.
//  - Guess-rank estimation (the crackability-correlated number) is a follow-up.

#[derive(clap::Args, Debug, Clone)]
pub struct ScoreArgs {
    /// Candidate password(s) to score. If none are given and --file is unset,
    /// candidates are read from stdin (one per line).
    candidates: Vec<String>,

    /// Model to score under: a registered name or a path to a .ngram file.
    /// Defaults to the default model (same resolution as `generate`).
    #[arg(long, value_name = "NAME|PATH")]
    model: Option<PathBuf>,

    /// Read candidates from FILE (one per line) instead of args/stdin.
    #[arg(long, value_name = "FILE")]
    file: Option<PathBuf>,

    /// Output format: table (human, default), tsv, or jsonl (for analysis).
    #[arg(long, value_enum, default_value_t = ScoreFormat::Table)]
    format: ScoreFormat,

    /// Algorithm variant to score under. Default `a` — the same variant
    /// `generate` ranks with, so Rarity stays consistent with emission order.
    /// Experimental; leave as `a`.
    #[arg(long, default_value = "a", hide = true)]
    variant: String,

    /// Report `+inf` for candidates the model assigns zero mass (what `generate`
    /// in its default mode can never emit), instead of the default finite floored Rarity.
    /// Default scoring backs off to a full-vocab add-k unigram floor so every
    /// password gets a finite probability; `--reachable` restores the old
    /// zero-mass reachability behavior (needed to reproduce +inf-gap analyses).
    #[arg(long)]
    reachable: bool,

    /// Add a per-position `Segment Score` column to the table: each token (and
    /// the terminal END) annotated with the surprisal it contributes in bits,
    /// with a `*` marking off-support (floored) steps. Off by default to keep
    /// the table compact. No effect on `--format tsv`/`jsonl`, which always
    /// carry the per-position data.
    #[arg(short = 'd', long)]
    detailed: bool,
}

/// Add-k (Laplace) constant for the score-time unigram floor: every token gets
/// `(count + k) / (total + k*vocab)`, so a zero-count token still has mass and no
/// candidate scores `+inf` in the default (finite) mode.
const SCORE_ADD_K: f64 = 1.0;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
enum ScoreFormat {
    /// Aligned human-readable table (default).
    Table,
    /// Tab-separated: candidate, n_tok, surprisal_bits, surprisal_per_tok_bits,
    /// in_vocab, segmentation, per_pos_surprisal_bits, per_pos_floored (the last
    /// two comma-joined, one entry per transition incl. the trailing END step).
    Tsv,
    /// One JSON object per line (surprisal_bits/_per_tok in bits; +inf → null).
    /// Includes a `positions` array of per-transition {label, surprisal_bits, floored}.
    Jsonl,
}

/// log P(t | a, b) in *nats* under a prepared EnumModel — an exact copy of the
/// per-step transition the generator's ranking heap uses (the `log_p_step`
/// closure in `build_seeds`). Trigram child if present; else KN bigram back-off
/// (with log-lambda when the trigram context exists but lacks `t`); `None` if
/// the transition is unreachable through both tiers. Keeping this identical to
/// the generator is what makes Rarity consistent with emission order.
fn transition_logprob(em: &EnumModel, a: u32, b: u32, t: u32) -> Option<f32> {
    let ctx = pack(a, b);
    if let Some(children) = em.trigram.get(&ctx) {
        if let Some(lp) = children.iter().find(|(id, _)| *id == t).map(|(_, lp)| *lp) {
            return Some(lp);
        }
        if em.is_kn {
            let log_lam = em.log_lambda.get(&ctx).copied().unwrap_or(0.0);
            if let Some(bi) = em.bigram.get(&b) {
                if let Some(lp_cont) = bi.iter().find(|(id, _)| *id == t).map(|(_, lp)| *lp) {
                    return Some(log_lam + lp_cont);
                }
            }
        }
        None
    } else if em.is_kn {
        em.bigram
            .get(&b)
            .and_then(|bi| bi.iter().find(|(id, _)| *id == t).map(|(_, lp)| *lp))
    } else {
        None
    }
}

/// Per-position score for one transition of the candidate's token path: the
/// token predicted at this step (or `END`), the surprisal it contributes in
/// bits, and whether that surprisal came from the unigram floor (an off-support
/// transition the trigram/bigram tiers couldn't cover). The step surprisals sum
/// to the candidate's joint Rarity.
struct StepScore {
    /// Token string predicted at this step, or `END` for the completion step.
    label: String,
    /// Surprisal this step contributes, bits (`-log2 P(step)`).
    surprisal_bits: f64,
    /// True iff this step fell through to the add-k unigram floor (off-support).
    floored: bool,
}

struct Scored {
    candidate: String,
    n_tok: usize,
    /// Surprisal of the whole candidate, bits. Finite via the add-k unigram floor
    /// by default; `+inf` only under `--reachable` when the model can't emit it.
    surprisal_bits: f64,
    /// Per-token surprisal (cross-entropy), bits/token.
    surprisal_per_tok_bits: f64,
    /// True iff fully on the trigram/bigram support (no floor used). In finite
    /// mode a floored candidate is still finite but reports `in_vocab = false`.
    in_vocab: bool,
    segmentation: Vec<String>,
    /// One entry per transition (each token, then `END`); surprisals sum to
    /// `surprisal_bits`. Drives the per-position display and per-position floor
    /// markers.
    steps: Vec<StepScore>,
}

fn score_candidate(em: &EnumModel, model: &Model, tokenizer: &Tokenizer, cand: &str, finite: bool, total_unigram: f64) -> Scored {
    let start_id = model.start_id;
    // Canonical greedy tokenization; drop any sentinel/special ids (>= start_id),
    // exactly as the seeded-generate path does.
    let ids: Vec<u32> = tokenizer
        .encode(cand, false)
        .map(|e| e.get_ids().iter().filter(|&&id| id < start_id).copied().collect())
        .unwrap_or_default();
    let segmentation: Vec<String> = ids
        .iter()
        .map(|&id| String::from_utf8_lossy(&model.decode[id as usize]).into_owned())
        .collect();
    let n_tok = ids.len();

    // Joint log-prob (nats), accumulated in f32 exactly as `build_seeds` does so
    // the value matches the generator's ranking key / seeded-mode heap init.
    // When `finite`, any transition unreachable through trigram + bigram-KN backs
    // off to a full-vocab raw-frequency unigram tier with an add-k floor, weighted
    // `log_lambda(a,b) + LOG_LAMBDA_BIGRAM` — the same formulation as Variant B's
    // generation tier, but uncapped and floored so EVERY candidate is finite.
    // `floored` records whether that floor fired at any step (⇒ off-support).
    let vocab = model.unigram_raw.len() as f64;
    let floor_step = |a: u32, b: u32, t: u32| -> f32 {
        let log_lam = em.log_lambda.get(&pack(a, b)).copied().unwrap_or(0.0);
        let c = *model.unigram_raw.get(t as usize).unwrap_or(&0) as f64;
        let p = (c + SCORE_ADD_K) / (total_unigram + SCORE_ADD_K * vocab);
        log_lam + crate::variant_b::LOG_LAMBDA_BIGRAM + (p.ln() as f32)
    };
    // Per-step surprisal (bits) for each transition, in path order (each token,
    // then END). Captured as we walk so the per-position view and floor markers
    // fall out of the same pass that computes the joint — nats -> bits is
    // `-(step)/ln2`. `steps` sums to the joint surprisal (modulo f32/f64 rounding).
    let nats_to_bits = |s: f32| -(s as f64) / std::f64::consts::LN_2;
    let mut a = start_id;
    let mut b = start_id;
    let mut lp = 0f32;
    let mut ok = n_tok > 0;
    let mut floored = false;
    let mut steps: Vec<StepScore> = Vec::with_capacity(n_tok + 1);
    if ok {
        for (i, &t) in ids.iter().enumerate() {
            let (step, this_floored) = match transition_logprob(em, a, b, t) {
                Some(s) => (s, false),
                None if finite => { floored = true; (floor_step(a, b, t), true) }
                None => {
                    // --reachable: mark exactly which transition is unreachable,
                    // then stop — the whole candidate is +inf.
                    ok = false;
                    steps.push(StepScore { label: segmentation[i].clone(), surprisal_bits: f64::INFINITY, floored: false });
                    break;
                }
            };
            lp += step;
            steps.push(StepScore { label: segmentation[i].clone(), surprisal_bits: nats_to_bits(step), floored: this_floored });
            a = b;
            b = t;
        }
    }
    // Final transition to the end-of-password sentinel. P(password) is the joint
    // over START, t1..tn, END — the model generates a *complete* password, so the
    // completion step belongs in the score (this is what makes Rarity consistent
    // with the generator's emission order, and matches FLA's end-symbol handling).
    if ok {
        match transition_logprob(em, a, b, model.end_id) {
            Some(step) => { lp += step; steps.push(StepScore { label: "END".into(), surprisal_bits: nats_to_bits(step), floored: false }); }
            None if finite => { floored = true; let step = floor_step(a, b, model.end_id); lp += step; steps.push(StepScore { label: "END".into(), surprisal_bits: nats_to_bits(step), floored: true }); }
            None => { ok = false; steps.push(StepScore { label: "END".into(), surprisal_bits: f64::INFINITY, floored: false }); }
        }
    }

    // `in_vocab` = fully on the trigram/bigram support (no floor). In finite mode
    // surprisal is finite even when floored; only `--reachable` yields `+inf`.
    let (surprisal_bits, in_vocab) = if ok && lp.is_finite() {
        // nats -> bits, and negate: surprisal = -log2 P = -(ln P) / ln 2.
        (-(lp as f64) / std::f64::consts::LN_2, !floored)
    } else {
        (f64::INFINITY, false)
    };
    let surprisal_per_tok_bits = if surprisal_bits.is_finite() && n_tok > 0 {
        surprisal_bits / n_tok as f64
    } else {
        f64::INFINITY
    };

    Scored {
        candidate: cand.to_string(),
        n_tok,
        surprisal_bits,
        surprisal_per_tok_bits,
        in_vocab,
        segmentation,
        steps,
    }
}

/// Render the per-position segmentation for the human table: `START` anchor,
/// then each predicted token / `END` annotated with its surprisal in bits, with
/// a trailing `*` on any step that fell through to the unigram floor. The
/// parenthetical bits sum to the candidate's Rarity, e.g.
/// `START|hack(7.0)|the(2.5)|planet(11.0)|END(1.6)`.
fn fmt_positional_segmentation(steps: &[StepScore]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(steps.len() + 1);
    parts.push("START".to_string());
    for st in steps {
        let star = if st.floored { "*" } else { "" };
        parts.push(format!("{}({}{})", st.label, fmt_bits(st.surprisal_bits, 1), star));
    }
    parts.join("|")
}

fn fmt_bits(v: f64, prec: usize) -> String {
    if v.is_finite() {
        format!("{:.*}", prec, v)
    } else {
        "inf".to_string()
    }
}

fn truncate_display(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n.saturating_sub(1)).collect();
        format!("{head}…")
    }
}

/// Minimal JSON string escaping (tokenov has no serde_json dependency).
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_num(v: f64) -> String {
    if v.is_finite() {
        format!("{:.6}", v)
    } else {
        "null".to_string()
    }
}

fn run_score(args: ScoreArgs) -> Result<()> {
    let model_path = match &args.model {
        Some(p) => registry::resolve_model(p)?,
        None => default_model_path()?,
    };
    let model = model_load(&model_path)?;
    let tokenizer = load_wordlist_tokenizer(&model_path, args.model.is_none())?;
    let variant = variant::dispatch(&args.variant)?;
    let em = variant.prepare(&model);
    // Full-vocab add-k unigram floor: finite scoring unless --reachable, and only
    // when the model actually carries unigram counts (NGRMv002).
    let total_unigram: f64 = model.unigram_raw.iter().map(|&c| c as f64).sum();
    let finite = !args.reachable && total_unigram > 0.0;

    // Candidate source precedence: positional args > --file > stdin.
    let candidates: Vec<String> = if !args.candidates.is_empty() {
        args.candidates.clone()
    } else if let Some(f) = &args.file {
        read_wordlist(f)?
    } else {
        read_lines_recovered(std::io::stdin().lock(), "score-stdin")?
    };
    if candidates.is_empty() {
        bail!("no candidates given (pass as args, via --file, or on stdin)");
    }

    let stdout = std::io::stdout();
    let mut w = BufWriter::new(stdout.lock());

    match args.format {
        ScoreFormat::Table => {
            if args.detailed {
                // Compact columns + the per-position Segment Score breakdown.
                writeln!(
                    w,
                    "{:<20} {:>6}  {:>12}  {:<24}  {}",
                    "Candidate", "Tokens", "Rarity(bits)", "Segmentation", "Segment Score"
                )?;
                for c in &candidates {
                    let s = score_candidate(&em, &model, &tokenizer, c, finite, total_unigram);
                    writeln!(
                        w,
                        "{:<20} {:>6}  {:>12}  {:<24}  {}",
                        truncate_display(&s.candidate, 20),
                        s.n_tok,
                        fmt_bits(s.surprisal_bits, 1),
                        s.segmentation.join("|"),
                        fmt_positional_segmentation(&s.steps)
                    )?;
                }
            } else {
                // Default: plain segmentation only. `-d`/`--detailed` adds the
                // per-position Segment Score column.
                writeln!(
                    w,
                    "{:<20} {:>6}  {:>12}  {}",
                    "Candidate", "Tokens", "Rarity(bits)", "Segmentation"
                )?;
                for c in &candidates {
                    let s = score_candidate(&em, &model, &tokenizer, c, finite, total_unigram);
                    writeln!(
                        w,
                        "{:<20} {:>6}  {:>12}  {}",
                        truncate_display(&s.candidate, 20),
                        s.n_tok,
                        fmt_bits(s.surprisal_bits, 1),
                        s.segmentation.join("|")
                    )?;
                }
            }
        }
        ScoreFormat::Tsv => {
            // per_pos_* carry one entry per transition (each token, then the
            // trailing END step) — so they have n_tok+1 entries; the last is END.
            // per_pos_surprisal_bits sums to surprisal_bits; per_pos_floored is
            // 0/1 flags marking off-support (floored) steps.
            writeln!(
                w,
                "candidate\tn_tok\tsurprisal_bits\tsurprisal_per_tok_bits\tin_vocab\tsegmentation\tper_pos_surprisal_bits\tper_pos_floored"
            )?;
            for c in &candidates {
                let s = score_candidate(&em, &model, &tokenizer, c, finite, total_unigram);
                let per_pos_bits = s.steps.iter().map(|st| fmt_bits(st.surprisal_bits, 4)).collect::<Vec<_>>().join(",");
                let per_pos_floored = s.steps.iter().map(|st| if st.floored { "1" } else { "0" }).collect::<Vec<_>>().join(",");
                writeln!(
                    w,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    s.candidate,
                    s.n_tok,
                    fmt_bits(s.surprisal_bits, 4),
                    fmt_bits(s.surprisal_per_tok_bits, 4),
                    s.in_vocab,
                    s.segmentation.join("|"),
                    per_pos_bits,
                    per_pos_floored
                )?;
            }
        }
        ScoreFormat::Jsonl => {
            for c in &candidates {
                let s = score_candidate(&em, &model, &tokenizer, c, finite, total_unigram);
                let seg = s
                    .segmentation
                    .iter()
                    .map(|t| json_str(t))
                    .collect::<Vec<_>>()
                    .join(",");
                // Per-position path (each token, then END); surprisal_bits sum to
                // the joint. +inf floored/unreachable steps serialize as null.
                let positions = s
                    .steps
                    .iter()
                    .map(|st| format!(
                        "{{\"label\":{},\"surprisal_bits\":{},\"floored\":{}}}",
                        json_str(&st.label),
                        json_num(st.surprisal_bits),
                        st.floored
                    ))
                    .collect::<Vec<_>>()
                    .join(",");
                writeln!(
                    w,
                    "{{\"candidate\":{},\"n_tok\":{},\"surprisal_bits\":{},\"surprisal_per_tok_bits\":{},\"in_vocab\":{},\"segmentation\":[{}],\"positions\":[{}]}}",
                    json_str(&s.candidate),
                    s.n_tok,
                    json_num(s.surprisal_bits),
                    json_num(s.surprisal_per_tok_bits),
                    s.in_vocab,
                    seg,
                    positions
                )?;
            }
        }
    }
    w.flush()?;
    Ok(())
}

// ============================================================================
// Default-model lookup
// ============================================================================

fn default_model_path() -> Result<PathBuf> {
    // The default model is the registered model whose name matches the
    // manifest's `default_alias` (set by `tokenizer set-default`, shipped as
    // `tokenov_v1`). `tokenov bootstrap` builds it under exactly that name.
    let alias = bootstrap::default_alias()
        .context("resolving the default tokenizer/model from the manifest")?;
    if let Some(e) = registry::find(&alias) {
        let p = PathBuf::from(&e.path);
        if p.exists() {
            return Ok(p);
        }
        bail!(
            "default model '{}' is registered but its file is missing ({}). \
             Rebuild it with `tokenov bootstrap`, or pass --model PATH.",
            alias, e.path
        );
    }
    bail!(
        "--model not specified and the default model '{}' is not built yet.\n\
         Build it with `tokenov bootstrap` (one command), change the default with \
         `tokenov tokenizer set-default <alias>`, or pass --model PATH.",
        alias
    );
}

/// Resolve the tokenizer for `--wordlist` mode. The `.ngram` file does not (yet)
/// embed its tokenizer, so we resolve it out-of-band. Order:
///   1. `TOKENOV_TOKENIZER` env — explicit override, always wins.
///   2. Co-located sidecar `<model>.tokenizer.json` — copied next to the model at
///      build time (forward-only: present for models built after this change).
///   3. Default model only (`--model` omitted): the tokenizer `tokenov bootstrap`
///      installed for the manifest's default alias at
///      `<tokenizers_dir>/<alias>/tokenizer.json`. This is what fixes an
///      already-built bundled model without a rebuild.
/// On failure, name the paths we looked for so the user can supply one.
///
/// Out-of-band tokenizer resolution for pre-v3 (v1/v2) models. A v3 model embeds its own tokenizer
/// (see `load_wordlist_tokenizer`), so this out-of-band resolution is
/// only reached for pre-v3 models. For those, a deliberately-wrong
/// `TOKENOV_TOKENIZER` is used as-is, not validated.
/// Load the tokenizer used to build `model_path`, for encoding a `--wordlist`.
/// Precedence:
///   1. `TOKENOV_TOKENIZER` env — explicit override, always wins (even over an
///      embedded tokenizer — a deliberate re-tokenize with a variant).
///   2. Embedded tokenizer in a **v3** model — loaded in-memory, no external file
///      (the model is self-describing, survives a bare single-file copy).
///   3. **v1/v2** model — out-of-band resolution via `resolve_wordlist_tokenizer`
///      (sidecar → default-alias).
fn load_wordlist_tokenizer(model_path: &Path, model_is_default: bool) -> Result<Tokenizer> {
    if let Some(p) = std::env::var_os("TOKENOV_TOKENIZER") {
        let p = PathBuf::from(p);
        if !p.exists() {
            bail!("TOKENOV_TOKENIZER is set to {} but that file does not exist", p.display());
        }
        if matches!(model_read_provenance(model_path), Ok(Some(_))) {
            log_msg(&format!("[gen] wordlist tokenizer: TOKENOV_TOKENIZER={} \
                              (overrides the model's embedded tokenizer)", p.display()));
        } else {
            log_msg(&format!("[gen] wordlist tokenizer (TOKENOV_TOKENIZER): {}", p.display()));
        }
        return Tokenizer::from_file(&p)
            .map_err(|e| anyhow!("load tokenizer {}: {}", p.display(), e));
    }
    if let Some(prov) = model_read_provenance(model_path)? {
        log_msg(&format!("[gen] wordlist tokenizer (embedded v3, built from {})", prov.tok_source));
        return Tokenizer::from_bytes(&prov.tokenizer_json)
            .map_err(|e| anyhow!("load embedded tokenizer: {}", e));
    }
    log_msg("[gen] wordlist tokenizer: pre-v3 model (identity not embedded) — resolving out-of-band");
    let tokenizer_path = resolve_wordlist_tokenizer(model_path, model_is_default)?;
    Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("load tokenizer {}: {}", tokenizer_path.display(), e))
}

fn resolve_wordlist_tokenizer(model_path: &Path, model_is_default: bool) -> Result<PathBuf> {
    // 1. Explicit override.
    if let Some(p) = std::env::var_os("TOKENOV_TOKENIZER") {
        let p = PathBuf::from(p);
        if !p.exists() {
            bail!("TOKENOV_TOKENIZER is set to {} but that file does not exist", p.display());
        }
        log_msg(&format!("[gen] wordlist tokenizer (TOKENOV_TOKENIZER): {}", p.display()));
        return Ok(p);
    }
    // 2. Co-located sidecar next to the model.
    let sidecar = model_path.with_extension("tokenizer.json");
    if sidecar.exists() {
        log_msg(&format!("[gen] wordlist tokenizer (co-located): {}", sidecar.display()));
        return Ok(sidecar);
    }
    // 3. Default model → the tokenizer bootstrap installed for the default alias.
    let default_alias_tok = if model_is_default {
        bootstrap::default_alias().ok().map(|alias| {
            registry::tokenizers_dir().join(&alias).join("tokenizer.json")
        })
    } else {
        None
    };
    if let Some(p) = &default_alias_tok {
        if p.exists() {
            log_msg(&format!("[gen] wordlist tokenizer (default alias): {}", p.display()));
            return Ok(p.clone());
        }
    }

    let default_line = match &default_alias_tok {
        Some(p) => format!("\n  - default-alias tokenizer: {}", p.display()),
        None => String::new(),
    };
    bail!(
        "wordlist mode needs the tokenizer that built this model, but none was found.\n\
         Looked for:\n  \
         - $TOKENOV_TOKENIZER (unset)\n  \
         - co-located sidecar: {}{}\n\
         Fix: set TOKENOV_TOKENIZER=/path/to/tokenizer.json, or place the build-time \
         tokenizer at the co-located path above. (Embedding the tokenizer in the model \
         file is tracked separately.)",
        sidecar.display(), default_line,
    )
}

fn guess_kind_from_decode(decode: &[Vec<u8>]) -> DecodeKind {
    // If any decode entry starts with a space, it's plausibly a Plain
    // (SentencePiece-style) tokenizer where decode() converts ▁ → ' '.
    // Both finalizers strip leading whitespace at minimum, so this
    // heuristic is informational only; getting it wrong has negligible
    // impact on output.
    for e in decode.iter().take(2048) {
        if e.first() == Some(&b' ') { return DecodeKind::Plain; }
    }
    DecodeKind::StripBpeSpace
}

// ============================================================================
// Calibration: measure throughput across K values, write sidecar
// ============================================================================

/// Per-K throughput measurement.
struct TuneMeasurement {
    chunk_size: usize,
    emit_rate_per_sec: f64,
    peak_rss_mb: usize,
}

/// Sidecar file persisted next to the model — caches the recommended K.
struct TuneSidecar {
    schema_version: u32,
    model_path: PathBuf,
    model_size_bytes: u64,
    model_mtime_unix: u64,
    calibrated_at: String,        // ISO-ish — seconds-since-epoch or human
    hostname: String,
    cpu_count: usize,
    threads_used: usize,
    recommended_chunk_size: usize,
    measurements: Vec<TuneMeasurement>,
    /// Set when the rate-vs-K curve has 2+ direction reversals (U-shape /
    /// multi-peak), which on this hardware almost always indicates thermal
    /// throttling or measurement-order bias rather than a real allocator
    /// effect. None = curve looked clean. Empty Some is treated like None.
    noise_warning: Option<String>,
}

/// Count direction reversals (sign-changes of consecutive deltas) in a rate
/// sequence. A clean curve has 0 (monotonic) or 1 (single peak / valley)
/// reversal. Two or more is the U-shape / multi-peak pattern that the
/// gpt2 / llama runs both showed and that thermal-position bias readily
/// produces. Equal-rate runs are skipped (no direction).
fn count_direction_reversals(rates: &[f64]) -> usize {
    if rates.len() < 3 { return 0; }
    let mut reversals = 0usize;
    let mut prev_dir: Option<bool> = None; // true = increasing
    for w in rates.windows(2) {
        let d = w[1] - w[0];
        if d.abs() < f64::EPSILON { continue; }
        let dir = d > 0.0;
        if let Some(p) = prev_dir {
            if p != dir { reversals += 1; }
        }
        prev_dir = Some(dir);
    }
    reversals
}

/// Build a stderr-friendly diagnostic for a noisy rate curve. Returns None
/// when the curve looks clean.
fn noise_warning_for_curve(measurements: &[TuneMeasurement]) -> Option<String> {
    let rates: Vec<f64> = measurements.iter().map(|m| m.emit_rate_per_sec).collect();
    let reversals = count_direction_reversals(&rates);
    if reversals < 2 { return None; }
    let curve = measurements.iter()
        .map(|m| format!("K={}: {:.0} c/s", m.chunk_size, m.emit_rate_per_sec))
        .collect::<Vec<_>>().join(", ");
    Some(format!(
        "rate curve has {} direction reversals — U-shape / multi-peak pattern. \
         On a shared or thermally-throttled CPU this is almost always throttling or \
         measurement-order bias, not real allocator behavior, so the recommended K is \
         unreliable. Re-run calibration on an idle machine, or pin --merge-chunk-size \
         explicitly. Curve: [{}]",
        reversals, curve))
}

fn sidecar_path_for(model_path: &Path) -> PathBuf {
    let mut s = model_path.as_os_str().to_owned();
    s.push(".tune.toml");
    PathBuf::from(s)
}

fn file_size_mtime(p: &Path) -> Result<(u64, u64)> {
    let m = std::fs::metadata(p).with_context(|| format!("stat {}", p.display()))?;
    let mtime = m.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Ok((m.len(), mtime))
}

fn read_peak_rss_mb() -> usize {
    // Linux: /proc/self/status has "VmHWM:    NNNN kB" for peak RSS.
    let s = match std::fs::read_to_string("/proc/self/status") {
        Ok(s) => s,
        Err(_) => return 0,
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            for tok in rest.split_whitespace() {
                if let Ok(kb) = tok.parse::<usize>() {
                    return kb / 1024;
                }
            }
        }
    }
    0
}

fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn timestamp_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    format!("{}", secs)
}

/// Hand-rolled TOML writer — sidecar is small and the format is simple
/// enough that pulling in a dep is overkill.
fn write_tune_sidecar(path: &Path, sc: &TuneSidecar) -> Result<()> {
    let mut s = String::new();
    s.push_str("# tokenov calibration sidecar — generated by `tokenov calibrate`\n");
    s.push_str("# Cached chunk_size for fast subsequent generates.\n");
    s.push_str("# Delete this file (or pass --retune / --force) to re-calibrate.\n\n");
    s.push_str(&format!("[meta]\nschema_version = {}\n", sc.schema_version));
    s.push_str(&format!("model_path = \"{}\"\n", sc.model_path.display()));
    s.push_str(&format!("model_size_bytes = {}\n", sc.model_size_bytes));
    s.push_str(&format!("model_mtime_unix = {}\n", sc.model_mtime_unix));
    s.push_str(&format!("calibrated_at = \"{}\"\n\n", sc.calibrated_at));
    s.push_str("[machine]\n");
    s.push_str(&format!("hostname = \"{}\"\n", sc.hostname));
    s.push_str(&format!("cpu_count = {}\n", sc.cpu_count));
    s.push_str(&format!("threads_used = {}\n\n", sc.threads_used));
    s.push_str("[recommended]\n");
    s.push_str(&format!("chunk_size = {}\n", sc.recommended_chunk_size));
    if let Some(w) = sc.noise_warning.as_deref().filter(|w| !w.is_empty()) {
        // TOML basic-string: escape backslashes and double quotes; keep on
        // one logical line by replacing newlines with spaces.
        let escaped = w.replace('\\', "\\\\").replace('"', "\\\"")
            .replace('\n', " ").replace('\r', " ");
        s.push_str(&format!("noise_warning = \"{}\"\n", escaped));
    }
    s.push('\n');
    for m in &sc.measurements {
        s.push_str("[[measurements]]\n");
        s.push_str(&format!("chunk_size = {}\n", m.chunk_size));
        s.push_str(&format!("emit_rate_per_sec = {:.0}\n", m.emit_rate_per_sec));
        s.push_str(&format!("peak_rss_mb = {}\n\n", m.peak_rss_mb));
    }
    let tmp = {
        let mut p = path.as_os_str().to_owned();
        p.push(".tmp");
        PathBuf::from(p)
    };
    std::fs::write(&tmp, &s)
        .with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Hand-rolled TOML reader — only parses what we wrote. Returns None if
/// the file doesn't exist or doesn't have the keys we need.
fn read_tune_sidecar(path: &Path) -> Result<Option<TuneSidecar>> {
    let text = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(anyhow::Error::from(e).context(format!("read {}", path.display()))),
    };
    let mut schema_version: u32 = 0;
    let mut model_path = PathBuf::new();
    let mut model_size_bytes: u64 = 0;
    let mut model_mtime_unix: u64 = 0;
    let mut calibrated_at = String::new();
    let mut hostname_s = String::new();
    let mut cpu_count: usize = 0;
    let mut threads_used: usize = 0;
    let mut recommended: usize = 0;
    let mut noise_warning: Option<String> = None;
    let mut measurements: Vec<TuneMeasurement> = Vec::new();
    let mut current: Option<TuneMeasurement> = None;
    let mut section: &str = "";
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        if line == "[[measurements]]" {
            if let Some(m) = current.take() { measurements.push(m); }
            current = Some(TuneMeasurement { chunk_size: 0, emit_rate_per_sec: 0.0, peak_rss_mb: 0 });
            section = "measurement";
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            if let Some(m) = current.take() { measurements.push(m); }
            section = match &line[1..line.len()-1] {
                "meta"        => "meta",
                "machine"     => "machine",
                "recommended" => "recommended",
                _             => "",
            };
            continue;
        }
        let Some((k, v)) = line.split_once('=') else { continue };
        let k = k.trim();
        let v = v.trim().trim_matches('"');
        match (section, k) {
            ("meta", "schema_version") => schema_version = v.parse().unwrap_or(0),
            ("meta", "model_path")      => model_path = PathBuf::from(v),
            ("meta", "model_size_bytes")=> model_size_bytes = v.parse().unwrap_or(0),
            ("meta", "model_mtime_unix")=> model_mtime_unix = v.parse().unwrap_or(0),
            ("meta", "calibrated_at")   => calibrated_at = v.to_string(),
            ("machine", "hostname")     => hostname_s = v.to_string(),
            ("machine", "cpu_count")    => cpu_count = v.parse().unwrap_or(0),
            ("machine", "threads_used") => threads_used = v.parse().unwrap_or(0),
            ("recommended", "chunk_size") => recommended = v.parse().unwrap_or(0),
            ("recommended", "noise_warning") => {
                let unesc = v.replace("\\\"", "\"").replace("\\\\", "\\");
                noise_warning = if unesc.is_empty() { None } else { Some(unesc) };
            }
            ("measurement", key) => {
                if let Some(m) = current.as_mut() {
                    match key {
                        "chunk_size"        => m.chunk_size = v.parse().unwrap_or(0),
                        "emit_rate_per_sec" => m.emit_rate_per_sec = v.parse().unwrap_or(0.0),
                        "peak_rss_mb"       => m.peak_rss_mb = v.parse().unwrap_or(0),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(m) = current.take() { measurements.push(m); }
    if recommended == 0 {
        return Ok(None); // not a valid sidecar (or empty)
    }
    Ok(Some(TuneSidecar {
        schema_version, model_path, model_size_bytes, model_mtime_unix,
        calibrated_at, hostname: hostname_s, cpu_count, threads_used,
        recommended_chunk_size: recommended, measurements, noise_warning,
    }))
}

/// Validate that a sidecar is for the same model (size + mtime match).
fn sidecar_is_valid(sc: &TuneSidecar, model_path: &Path) -> Result<bool> {
    let (size, mtime) = file_size_mtime(model_path)?;
    Ok(sc.model_size_bytes == size && sc.model_mtime_unix == mtime)
}

/// Run the actual calibration: load model, set up parallel infrastructure,
/// loop through K values measuring throughput. Writes sidecar on success.
#[allow(clippy::too_many_arguments)]
fn do_calibration(
    model_path: &Path,
    chunk_sizes: &[usize],
    measure_secs: u64,
    settle_secs: u64,
    threads_opt: Option<usize>,
    max_memory_mb: usize,
    sidecar_out: &Path,
    variant: Arc<dyn Variant>,
    // When `Some`, reuse the caller's already-built setup instead of doing
    // model_load + variant.prepare + build_child_cache again. Inline calibration
    // from run_generate always passes Some (the dominant code path). Standalone
    // `tokenov calibrate` (run_calibrate) passes None and we build internally.
    setup: Option<CalibSetup>,
) -> Result<usize> {
    let n_threads = threads_opt.unwrap_or_else(rayon::current_num_threads).max(1);
    log_msg(&format!("[calibrate] variant={} threads={} chunk_sizes={:?} measure={}s settle={}s",
        variant.name(), n_threads, chunk_sizes, measure_secs, settle_secs));

    // Either reuse the caller's setup, or build our own.
    let (enum_model_arc, child_cache, kind, start_id, end_id, decode_table_arc, first_level) =
        if let Some(s) = setup {
            log_msg(&format!(
                "[calibrate] reusing caller's setup: child_cache={} ctx, first_level={} tokens",
                s.child_cache.len(), s.first_level.len()));
            (s.enum_model, s.child_cache, s.kind, s.start_id, s.end_id,
             s.decode_table, s.first_level)
        } else {
            // Standalone path: load model, prepare enum_model, build child cache.
            // Mirrors run_generate's standard-mode setup so the in-process memory
            // shape matches.
            let mut model = model_load(model_path)?;
            let kind  = guess_kind_from_decode(&model.decode);
            let mut enum_model = variant.prepare(&model);
            // Drop model heavy fields immediately after prepare.
            model.contexts = rustc_hash::FxHashMap::default();
            model.bigram_kn = rustc_hash::FxHashMap::default();
            model.lambda = rustc_hash::FxHashMap::default();
            let child_cache: ChildCache = build_child_cache(&enum_model, &*variant);
            log_msg(&format!("[calibrate] child_cache: {} ctx", child_cache.len()));
            let first_level = expand_first_level(&enum_model, model.start_id);
            // Drop trigram + log_lambda — covered by child_cache. See
            // run_generate's identical block for the safety argument.
            {
                let n_tri = enum_model.trigram.len();
                let n_cache = child_cache.len();
                if n_tri != n_cache {
                    anyhow::bail!(
                        "internal: child_cache covers {} of {} trigram ctxs; refusing to drop trigram",
                        n_cache, n_tri);
                }
                enum_model.trigram = rustc_hash::FxHashMap::default();
                enum_model.log_lambda = rustc_hash::FxHashMap::default();
                log_msg(&format!(
                    "[calibrate] dropped enum_model.trigram + log_lambda ({} ctxs covered by child_cache)",
                    n_cache));
            }
            let decode_table_arc = Arc::new(model.decode.clone());
            let start_id = model.start_id;
            let end_id = model.end_id;
            let enum_model_arc = Arc::new(enum_model);
            (enum_model_arc, child_cache, kind, start_id, end_id, decode_table_arc, first_level)
        };

    let partitions: Vec<Vec<HeapEntry>> = assign_partitions(first_level, n_threads);
    let n_active = partitions.iter().filter(|p| !p.is_empty()).count().max(1);
    log_msg(&format!("[calibrate] domain decomp: {} non-empty partitions", n_active));

    // Shared atomics: chunk_size (controls K), should_stop (signals workers to exit).
    let initial_k = *chunk_sizes.first().unwrap_or(&DEFAULT_MERGE_CHUNK_SIZE);
    let chunk_size_atomic = Arc::new(AtomicUsize::new(initial_k));
    let should_stop = Arc::new(AtomicBool::new(false));
    let emit_count = Arc::new(AtomicU64::new(0));

    // Channel setup. Use the smallest expected K to size capacity; producers
    // can change K mid-flight but capacity stays fixed (slack is fine).
    let smallest_k = chunk_sizes.iter().copied().min().unwrap_or(initial_k).max(1);
    let channel_chunks = (MERGE_CHANNEL_BUFFER_ITEMS / smallest_k).max(2);
    let mut workers: Vec<(Sender<MergeChunk>, Vec<HeapEntry>, usize)> = Vec::new();
    let mut receivers: Vec<Receiver<MergeChunk>> = Vec::new();
    for (i, partition) in partitions.into_iter().enumerate() {
        if partition.is_empty() { continue; }
        let (tx, rx) = channel_bounded::<MergeChunk>(channel_chunks);
        workers.push((tx, partition, i));
        receivers.push(rx);
    }

    // enum_model_arc, child_cache, kind, start_id, end_id, decode_table_arc are
    // already in scope from the setup-reuse-or-build block above.
    // Use the build-time defaults — we're measuring throughput, not actually emitting.
    let max_tokens = DEFAULT_MAX_TOKENS;
    let min_len    = 4usize;
    let max_len    = 30usize;
    let thread_target: u64 = u64::MAX;  // run forever; should_stop ends the loop

    let measurements: Vec<TuneMeasurement> = std::thread::scope(|scope| -> Result<Vec<TuneMeasurement>> {
        // Discard sink: the merger writes to it (counts emits, drops bytes).
        let sink = Sink::open_discard(Arc::clone(&emit_count));

        // Spawn merger.
        let merger = scope.spawn(move || -> Result<u64> {
            run_merger(sink, receivers, 0, 0, None, "", None,
                Arc::new(AtomicBool::new(false)))
        });

        // Spawn workers.
        let mut handles = Vec::with_capacity(workers.len());
        for (tx, states, id) in workers {
            let em = Arc::clone(&enum_model_arc);
            let cc = child_cache;  // &'static, no Arc to clone
            let dt = Arc::clone(&decode_table_arc);
            let cs = Arc::clone(&chunk_size_atomic);
            let stop = Arc::clone(&should_stop);
            let var = Arc::clone(&variant);
            handles.push(scope.spawn(move || -> Result<()> {
                let label = format!("[c{}] ", id);
                let mut chunk_sender = ChunkSender::new(tx, cs);
                let res = enumerate_to_sink(
                    // Calibration only runs in FULL mode (resolve_chunk_size
                    // returns before do_calibration for BOUNDED/LAZY), so the
                    // cache here is always populated → CacheMode::Full.
                    &em, &*var, cc, CacheMode::Full, &dt, kind, start_id, end_id,
                    states, max_tokens, 1 /* min_tokens: no filter during calibration */, min_len, max_len,
                    0, // min_level: calibration always sweeps from shell 0
                    thread_target,
                    false, // no enterprise filter during calibration
                    &[],   // no case masks during calibration
                    |lvl, sk, bytes| {
                        if stop.load(AtomicOrdering::Relaxed) {
                            return Err(anyhow!("calibration stop"));
                        }
                        chunk_sender.push(lvl, sk, bytes)
                    },
                    &label,
                    None, None, |_: &ThreadCkpt| {},
                );
                if res.is_ok() {
                    chunk_sender.flush()?;
                }
                // Calibration stop is expected — don't propagate as error.
                match res {
                    Ok(()) => Ok(()),
                    Err(e) if e.to_string().contains("calibration stop") => Ok(()),
                    Err(e) if e.to_string().contains("merger channel closed") => Ok(()),
                    Err(e) => Err(e),
                }
            }));
        }

        // Sample loop.
        let mut results = Vec::with_capacity(chunk_sizes.len());
        for &k in chunk_sizes {
            chunk_size_atomic.store(k, AtomicOrdering::Relaxed);
            log_msg(&format!("[calibrate] K={} settling for {}s", k, settle_secs));
            std::thread::sleep(Duration::from_secs(settle_secs));
            let start = emit_count.load(AtomicOrdering::Relaxed);
            let t0 = Instant::now();
            log_msg(&format!("[calibrate] K={} measuring for {}s", k, measure_secs));
            std::thread::sleep(Duration::from_secs(measure_secs));
            let end = emit_count.load(AtomicOrdering::Relaxed);
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = (end - start) as f64 / elapsed;
            let rss = read_peak_rss_mb();
            log_msg(&format!("[calibrate] K={} rate={:.0} c/s peak_rss={} MB",
                k, rate, rss));
            results.push(TuneMeasurement {
                chunk_size: k,
                emit_rate_per_sec: rate,
                peak_rss_mb: rss,
            });
        }

        // Stop producers; merger will drain and exit.
        should_stop.store(true, AtomicOrdering::Relaxed);
        for h in handles {
            let _ = h.join();
        }
        let _ = merger.join();

        Ok(results)
    })?;

    // Pick best K within memory budget. Capture the values out of the
    // borrowed reference so we can move `measurements` into the sidecar.
    let (best_k, best_rate, best_rss) = {
        let pick = measurements.iter()
            .filter(|m| m.peak_rss_mb <= max_memory_mb)
            .max_by(|a, b| a.emit_rate_per_sec.partial_cmp(&b.emit_rate_per_sec)
                           .unwrap_or(Ordering::Equal))
            .or_else(|| measurements.iter()
                .max_by(|a, b| a.emit_rate_per_sec.partial_cmp(&b.emit_rate_per_sec)
                               .unwrap_or(Ordering::Equal)))
            .ok_or_else(|| anyhow!("no measurements"))?;
        (pick.chunk_size, pick.emit_rate_per_sec, pick.peak_rss_mb)
    };

    let noise_warning = noise_warning_for_curve(&measurements);
    if let Some(w) = &noise_warning {
        warn_msg(&format!("[calibrate] WARNING: {}", w));
    }

    let (model_size, model_mtime) = file_size_mtime(model_path)?;
    let sc = TuneSidecar {
        schema_version: 1,
        model_path: model_path.to_path_buf(),
        model_size_bytes: model_size,
        model_mtime_unix: model_mtime,
        calibrated_at: timestamp_now(),
        hostname: hostname(),
        cpu_count: rayon::current_num_threads(),
        threads_used: n_threads,
        recommended_chunk_size: best_k,
        measurements,
        noise_warning,
    };
    write_tune_sidecar(sidecar_out, &sc)?;
    log_msg(&format!("[calibrate] recommended K={} (rate={:.0} c/s, rss={} MB)",
        best_k, best_rate, best_rss));
    log_msg(&format!("[calibrate] sidecar written: {}", sidecar_out.display()));

    Ok(best_k)
}

fn run_calibrate(args: CalibrateArgs) -> Result<()> {
    let model_path = registry::resolve_model(&args.model)?;  // name or path
    let sidecar_out = args.output.clone()
        .unwrap_or_else(|| sidecar_path_for(&model_path));
    if sidecar_out.exists() && !args.force {
        bail!("sidecar already exists at {} — pass --force to overwrite",
            sidecar_out.display());
    }
    let var = variant::dispatch(&args.variant)?;
    do_calibration(&model_path, &args.chunk_sizes, args.measure_secs,
                   args.settle_secs, args.threads, args.max_memory_mb,
                   &sidecar_out, var, None)?;
    Ok(())
}

/// Determine chunk_size for a generate run, in order of priority:
///   1. --merge-chunk-size N (explicit) → use N
///   2. <model>.tune.toml exists + valid → use cached recommended K
///   3. --no-auto-tune → use DEFAULT_MERGE_CHUNK_SIZE
///   4. else → run inline calibration, write sidecar, use result
fn resolve_chunk_size(
    model_path: &Path,
    args: &GenerateArgs,
    // The child_cache mode chosen for this run. Inline calibration builds/uses
    // the FULL cache, so it is only valid in FULL mode; BOUNDED/LAZY skip it.
    cache_mode: CacheMode,
    // `Some` when called inline from run_generate (caller has already built
    // model + enum_model + child_cache); `None` when called from contexts
    // without a pre-built setup. In the `Some` case we hand it through to
    // do_calibration so calibration doesn't build a duplicate.
    setup: Option<CalibSetup>,
) -> Result<usize> {
    // Fast mode (the default) has no merger — chunk size is never used. Skip the
    // sidecar read + the ~3-min inline calibration entirely. The
    // returned value is inert.
    if !args.strict {
        return Ok(DEFAULT_MERGE_CHUNK_SIZE);
    }
    if let Some(k) = args.merge_chunk_size {
        log_msg(&format!("[gen] chunk_size: explicit --merge-chunk-size={}", k));
        return Ok(k.max(1));
    }
    let sidecar = sidecar_path_for(model_path);
    if !args.retune {
        if let Some(sc) = read_tune_sidecar(&sidecar)? {
            if sidecar_is_valid(&sc, model_path)? {
                log_msg(&format!(
                    "[gen] chunk_size: {} from sidecar {} (calibrated {})",
                    sc.recommended_chunk_size, sidecar.display(), sc.calibrated_at));
                if let Some(w) = sc.noise_warning.as_deref().filter(|w| !w.is_empty()) {
                    warn_msg(&format!("[gen] WARNING: sidecar was flagged as noisy at calibration time: {}", w));
                }
                return Ok(sc.recommended_chunk_size.max(1));
            } else {
                log_msg(&format!("[gen] sidecar {} is stale (model size/mtime changed); ignoring",
                    sidecar.display()));
            }
        }
    }
    if args.no_auto_tune {
        log_msg(&format!("[gen] chunk_size: {} (default, --no-auto-tune set)",
            DEFAULT_MERGE_CHUNK_SIZE));
        return Ok(DEFAULT_MERGE_CHUNK_SIZE);
    }
    // Inline calibration builds/uses the FULL child_cache
    // (do_calibration → build_child_cache / Full-mode lookups), which is exactly
    // the large allocation BOUNDED/LAZY exist to avoid. Chunk size does not affect
    // output, so default here rather than OOM the calibration pass. (A pre-existing
    // .tune.toml sidecar, handled above, is still honored — it costs no memory.)
    if cache_mode != CacheMode::Full {
        log_msg(&format!("[gen] chunk_size: {} (default; {:?} skips inline \
                 calibration to avoid building the full child_cache)", DEFAULT_MERGE_CHUNK_SIZE, cache_mode));
        return Ok(DEFAULT_MERGE_CHUNK_SIZE);
    }
    log_msg("[gen] no calibration sidecar found and --no-auto-tune not set; \
             running inline calibration (~3 min)");
    let chunk_sizes = vec![1024usize, 2048, 4096, 8192, 16384];
    let var = variant::dispatch(&args.variant)?;
    let k = do_calibration(model_path, &chunk_sizes, 30, 5,
                           Some(rayon::current_num_threads()),
                           16384, &sidecar, var, setup)?;
    log_msg(&format!("[gen] chunk_size: {} (auto-tuned, sidecar written)", k));
    Ok(k.max(1))
}

// ============================================================================
// main
// ============================================================================

/// Sixel logo (a small raster image) shown above the TOP-LEVEL `-h`/`--help`.
const LOGO_SIXEL: &str = include_str!("logo.txt");

/// True where the logo can render safely: an interactive terminal, not
/// `TERM=dumb`, and not opted out via NO_COLOR / TOKENOV_NO_LOGO. Piped/redirected
/// output (`--help | less`, CI, `> file`) fails the isatty check, and terminals
/// without Sixel support silently consume the DCS image block.
fn logo_ok() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
        && std::env::var_os("TOKENOV_NO_LOGO").is_none()
        && std::env::var_os("NO_COLOR").is_none()
        && std::env::var("TERM").map(|t| t != "dumb").unwrap_or(false)
}

/// The branded logo block: "A project by" then the Sixel image. The asset's own
/// trailing line advance (the `\n` before show-cursor) is stripped so surrounding
/// text sits as close to the image as possible — the remaining gap is the
/// intrinsic sub-row padding of the raster (image height rounds up to whole text
/// rows), which no escape trims.
fn logo_block() -> String {
    let logo = LOGO_SIXEL.replace("\n\x1b[?25h", "\x1b[?25h");
    format!("A project by\n{logo}")
}

/// Decorate the TOP-LEVEL `-h/--help` (logo above the help) and `-V/--version`
/// (logo below the version) with the logo, when it can render. A subcommand before
/// the flag (e.g. `generate --help`) is not top-level and gets no logo. For
/// `--version` we print the version + logo ourselves and exit, since clap would
/// otherwise print the plain version and exit before we could append anything.
fn maybe_decorate_help_version() {
    use std::io::Write;
    let mut args = std::env::args().skip(1);
    let first = args.next();
    let is_toplevel_help = matches!(first.as_deref(), Some("-h") | Some("--help"))
        || (first.as_deref() == Some("help") && args.next().is_none());
    let is_toplevel_version = matches!(first.as_deref(), Some("-V") | Some("--version"));

    if is_toplevel_help && logo_ok() {
        print!("{}", logo_block()); // clap renders the help body right after
        let _ = std::io::stdout().flush();
    } else if is_toplevel_version && logo_ok() {
        // Match clap's version line, then the logo below it, then exit — bypassing
        // clap's plain-version handler. (Piped/non-tty falls through to clap.)
        println!("tokenov {}", env!("CARGO_PKG_VERSION"));
        print!("{}", logo_block());
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    }
}

fn main() -> Result<()> {
    maybe_decorate_help_version();
    let cli = Cli::parse();
    set_verbose(cli.verbose);
    if cli.list_models {
        registry::list();
        return Ok(());
    }
    if cli.sessions {
        return list_sessions();
    }
    if let Some(id) = &cli.resume_session {
        return resume_session(id);
    }
    match cli.command {
        Some(Command::Model(args))     => run_model(args),
        Some(Command::Tokenizer(args)) => run_tokenizer(args),
        Some(Command::Generate(args))  => run_generate(args),
        Some(Command::Score(args))     => run_score(args),
        Some(Command::Bootstrap(args)) => bootstrap::run_bootstrap(args),
        Some(Command::Calibrate(args)) => run_calibrate(args),
        // Deprecated aliases: warn (stderr only — stdout stays candidates-only)
        // then route to the new handler.
        Some(Command::Build(args))     => { deprecate("build", "model train"); run_build(args) }
        Some(Command::Delete(args))    => { deprecate("delete", "model delete"); run_delete(args) }
        Some(Command::Register(args))  => { deprecate("register", "model register"); run_register(args) }
        Some(Command::Fetch(args))     => { deprecate("fetch", "tokenizer get"); bootstrap::run_fetch(args) }
        None => run_generate(cli.generate_args),
    }
}

/// Print a one-line deprecation notice for an old top-level subcommand to
/// stderr (never stdout — the candidate stream must stay clean).
fn deprecate(old: &str, new: &str) {
    eprintln!("warning: 'tokenov {old}' is deprecated; use 'tokenov {new}'");
}

fn run_model(args: ModelArgs) -> Result<()> {
    match args.cmd {
        Some(ModelCmd::Train(a))    => run_build(a),
        Some(ModelCmd::Register(a)) => run_register(a),
        Some(ModelCmd::Delete(a))   => run_delete(a),
        Some(ModelCmd::Info(a))     => run_model_info(a),
        Some(ModelCmd::Verify(a))   => run_model_verify(a),
        Some(ModelCmd::List) | None => { registry::list(); Ok(()) }
    }
}

fn run_model_verify(args: VerifyArgs) -> Result<()> {
    let entries = registry::read_registry();
    let targets: Vec<&registry::ModelEntry> = match &args.name {
        Some(n) => entries.iter().filter(|e| &e.name == n).collect(),
        None => entries.iter().collect(),
    };
    if targets.is_empty() {
        match &args.name {
            Some(n) => anyhow::bail!("no registered model named '{}'", n),
            None => {
                println!("no registered models");
                return Ok(());
            }
        }
    }
    let short = |h: &str| h.chars().take(12).collect::<String>();
    let mut mismatches = 0usize;
    let mut backfilled = 0usize;
    for e in &targets {
        let p = std::path::Path::new(&e.path);
        if !p.exists() {
            println!("⚠ MISSING  {}  ({})", e.name, e.path);
            continue;
        }
        let actual = match registry::sha256_file(p) {
            Ok(h) => h,
            Err(err) => {
                println!("✗ ERROR    {}: {}", e.name, err);
                mismatches += 1;
                continue;
            }
        };
        match &e.sha256 {
            Some(rec) if rec == &actual => println!("✓ OK       {}", e.name),
            Some(rec) => {
                println!(
                    "✗ MISMATCH {}  (registry {}… file {}…)",
                    e.name,
                    short(rec),
                    short(&actual)
                );
                mismatches += 1;
            }
            None => {
                if args.update {
                    registry::set_entry_hash(&e.name, &actual)?;
                    println!("+ RECORDED {}  ({}…)", e.name, short(&actual));
                    backfilled += 1;
                } else {
                    println!(
                        "? NO-HASH  {}  (run `tokenov model verify {} --update` to record)",
                        e.name, e.name
                    );
                }
            }
        }
    }
    if backfilled > 0 {
        eprintln!("recorded {} previously-unhashed model(s)", backfilled);
    }
    if mismatches > 0 {
        anyhow::bail!("{} model(s) failed the integrity check", mismatches);
    }
    Ok(())
}

fn run_model_info(args: InfoArgs) -> Result<()> {
    let path = registry::resolve_model(&args.model)?;
    println!("model: {}", path.display());
    match model_read_provenance(&path)? {
        Some(prov) => {
            // A v3 model's embedded tokenizer identifies itself — decode its
            // reported vocab size as a light integrity signal.
            let vocab = Tokenizer::from_bytes(&prov.tokenizer_json)
                .map(|t| t.get_vocab_size(false).to_string())
                .unwrap_or_else(|_| "<unreadable>".to_string());
            println!("format:            NGRMv003 (self-describing)");
            println!("tokenizer:         embedded, {} bytes, vocab_size {}",
                prov.tokenizer_json.len(), vocab);
            println!("tokenizer source:  {}", prov.tok_source);
            println!("train corpus:      {}", prov.train_path);
            println!("built:             {} (epoch {})", fmt_epoch_utc(prov.build_epoch), prov.build_epoch);
            println!("tokenov version:   {}", prov.binver);
        }
        None => {
            println!("format:            NGRMv001/002 (older format — no embedded tokenizer)");
            println!("note:              built by an older tokenov; the tokenizer is not embedded.");
            println!("                   Rebuild with a current tokenov to embed it, or for");
            println!("                   --wordlist mode supply the tokenizer via TOKENOV_TOKENIZER");
            println!("                   or a co-located <model>.tokenizer.json file.");
        }
    }
    Ok(())
}

/// epoch seconds → "YYYY-MM-DD HH:MM UTC" (delegates to the registry's formatter).
fn fmt_epoch_utc(secs: u64) -> String {
    registry::fmt_epoch_public(secs)
}

/// Train a byte-level BPE tokenizer from a corpus and write a `tokenizer.json`
/// that `model train` (and `generate --wordlist`) can consume directly. Matches
/// the pipeline's tokenizer shape: BPE model + ByteLevel pre-tokenizer, no
/// normalizer/decoder, no special tokens (tokenov supplies START/END internally).
fn run_tokenizer_train(args: TokenizerTrainArgs) -> Result<()> {
    use tokenizers::models::bpe::{BpeTrainer, BPE};
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;

    if !args.corpus.exists() {
        bail!("corpus not found: {}", args.corpus.display());
    }
    if args.output.exists() && !args.force {
        bail!("{} already exists (use --force to overwrite)", args.output.display());
    }
    if args.vocab_size < 256 {
        bail!("--vocab-size must be >= 256 (the byte-level alphabet alone is 256 tokens)");
    }

    log_msg(&format!(
        "[tok-train] BPE vocab_size={} min_frequency={} corpus={}",
        args.vocab_size, args.min_frequency, args.corpus.display()));
    let t0 = Instant::now();

    // BPE model + ByteLevel pre-tokenizer (add_prefix_space=false, trim_offsets=true,
    // use_regex=true) — the same shape as the bundled/reference tokenizers.
    let mut tokenizer = Tokenizer::new(BPE::default());
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));

    // The high-level Tokenizer trains via ModelWrapper, so wrap the BpeTrainer in
    // the crate's TrainerWrapper (whose associated Model is ModelWrapper).
    let mut trainer: tokenizers::models::TrainerWrapper = BpeTrainer::builder()
        .vocab_size(args.vocab_size)
        .min_frequency(args.min_frequency)
        .initial_alphabet(ByteLevel::alphabet())
        .special_tokens(vec![])
        .show_progress(verbose())
        .build()
        .into();

    let corpus = args.corpus.to_string_lossy().to_string();
    tokenizer
        .train_from_files(&mut trainer, vec![corpus])
        .map_err(|e| anyhow!("tokenizer training failed: {}", e))?;

    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    tokenizer
        .save(&args.output, false)
        .map_err(|e| anyhow!("could not write tokenizer to {}: {}", args.output.display(), e))?;

    let vs = tokenizer.get_vocab_size(true);
    log_msg(&format!("[tok-train] done in {:.1}s", t0.elapsed().as_secs_f64()));
    println!("Trained tokenizer → {} (BPE, {} tokens)", args.output.display(), vs);
    println!("Next:  tokenov model train --tokenizer {} --train <corpus> --name <name>",
        args.output.display());
    Ok(())
}

fn run_tokenizer(args: TokenizerArgs) -> Result<()> {
    match args.cmd {
        Some(TokenizerCmd::Train(a))  => run_tokenizer_train(a),
        Some(TokenizerCmd::Get(a))    => bootstrap::run_fetch(a),
        Some(TokenizerCmd::Add(a))    => bootstrap::run_add(a),
        Some(TokenizerCmd::Delete(a)) => bootstrap::run_tok_delete(a),
        Some(TokenizerCmd::SetDefault(a)) => bootstrap::run_set_default(a),
        Some(TokenizerCmd::List) | None => bootstrap::run_list_status(None, None),
    }
}

#[cfg(test)]
mod prov_tests {
    use super::*;

    #[test]
    fn provenance_round_trips() {
        let p = Provenance {
            tokenizer_json: b"{\"fake\":\"tokenizer\",\"bytes\":[0,255,128]}".to_vec(),
            tok_source: "tokenizers/tokenov_v1/tokenizer.json".to_string(),
            train_path: "/data/rockyou_train_freq.txt".to_string(),
            build_epoch: 1_785_459_182,
            binver: "0.18.0".to_string(),
        };
        let blob = p.serialize();
        let q = Provenance::parse(&blob).expect("parse");
        assert_eq!(q.tokenizer_json, p.tokenizer_json);
        assert_eq!(q.tok_source, p.tok_source);
        assert_eq!(q.train_path, p.train_path);
        assert_eq!(q.build_epoch, p.build_epoch);
        assert_eq!(q.binver, p.binver);
    }

    #[test]
    fn provenance_parse_tolerates_trailing_future_fields() {
        // A newer schema appends extra bytes after the known fields; an older
        // reader must still parse the known prefix (the outer length prefix is
        // what bounds the blob, so trailing bytes are simply not read).
        let p = Provenance {
            tokenizer_json: vec![1, 2, 3, 4],
            tok_source: "t".into(), train_path: "r".into(),
            build_epoch: 42, binver: "0.18.0".into(),
        };
        let mut blob = p.serialize();
        blob.extend_from_slice(b"\x07\x00\x00\x00FUTUREF"); // a bogus future field
        let q = Provenance::parse(&blob).expect("parse ignores trailing");
        assert_eq!(q.tokenizer_json, vec![1, 2, 3, 4]);
        assert_eq!(q.binver, "0.18.0");
    }

    #[test]
    fn empty_tokenizer_bytes_round_trip() {
        let p = Provenance {
            tokenizer_json: Vec::new(),
            tok_source: String::new(), train_path: String::new(),
            build_epoch: 0, binver: String::new(),
        };
        let q = Provenance::parse(&p.serialize()).expect("parse");
        assert!(q.tokenizer_json.is_empty());
        assert_eq!(q.build_epoch, 0);
    }

    #[test]
    fn truncated_blob_errors_not_panics() {
        let p = Provenance {
            tokenizer_json: vec![9; 32],
            tok_source: "x".into(), train_path: "y".into(),
            build_epoch: 1, binver: "z".into(),
        };
        let blob = p.serialize();
        // Chop the blob mid-field: parse must return Err, never panic/OOB.
        assert!(Provenance::parse(&blob[..blob.len() - 5]).is_err());
        assert!(Provenance::parse(&blob[..3]).is_err());
    }
}

#[cfg(test)]
mod count_parse_tests {
    use super::parse_count;

    #[test]
    fn bare_integers_unchanged() {
        assert_eq!(parse_count("0"), Ok(0));
        assert_eq!(parse_count("100000"), Ok(100_000));
        assert_eq!(parse_count("1000000000"), Ok(1_000_000_000));
        assert_eq!(parse_count(&u64::MAX.to_string()), Ok(u64::MAX));
    }

    #[test]
    fn suffixes_case_insensitive() {
        for (s, v) in [
            ("100k", 100_000u64), ("100K", 100_000),
            ("1m", 1_000_000), ("1M", 1_000_000),
            ("1b", 1_000_000_000), ("1B", 1_000_000_000),
            ("1t", 1_000_000_000_000), ("1T", 1_000_000_000_000),
        ] {
            assert_eq!(parse_count(s), Ok(v), "{s}");
        }
    }

    #[test]
    fn suffix_matches_bare_equivalent() {
        assert_eq!(parse_count("100k"), parse_count("100000"));
        assert_eq!(parse_count("100K"), parse_count("100000"));
    }

    #[test]
    fn decimals_that_are_whole_numbers() {
        assert_eq!(parse_count("1.5M"), Ok(1_500_000));
        assert_eq!(parse_count("2.5k"), Ok(2_500));
        assert_eq!(parse_count("1.1M"), Ok(1_100_000)); // exact fixed-point, no float drift
        assert_eq!(parse_count("1.50M"), Ok(1_500_000));
        assert_eq!(parse_count("0.5k"), Ok(500));
    }

    #[test]
    fn whitespace_trimmed() {
        assert_eq!(parse_count("  100k  "), Ok(100_000));
    }

    #[test]
    fn rejects_non_integer_result() {
        assert!(parse_count("1.2345k").is_err()); // 1234.5 candidates
        assert!(parse_count("0.0005k").is_err()); // 0.5 candidates
        // sanity: a decimal that IS whole must still succeed
        assert_eq!(parse_count("1.3k"), Ok(1_300));
    }

    #[test]
    fn rejects_malformed() {
        assert!(parse_count("100X").is_err());
        assert!(parse_count("1.2.3K").is_err());
        assert!(parse_count("").is_err());
        assert!(parse_count("K").is_err());
        assert!(parse_count("-5k").is_err());
        assert!(parse_count("abc").is_err());
    }

    #[test]
    fn rejects_overflow() {
        assert!(parse_count("99999999999999999999T").is_err());
        assert!(parse_count("18446744073709551616").is_err()); // u64::MAX + 1
    }
}

#[cfg(test)]
mod min_level_tests {
    use super::{validate_min_level, LEVEL_MAX};

    #[test]
    fn default_and_in_range_accepted() {
        assert!(validate_min_level(0, false, false).is_ok());
        assert!(validate_min_level(30, false, false).is_ok());
        assert!(validate_min_level(LEVEL_MAX, false, false).is_ok());
    }

    #[test]
    fn rejects_past_level_max() {
        assert!(validate_min_level(LEVEL_MAX + 1, false, false).is_err());
    }

    #[test]
    fn rejects_graft_combination_but_not_the_default() {
        assert!(validate_min_level(30, true, true).is_err());   // --float
        assert!(validate_min_level(30, true, false).is_err());  // --prepend-only
        assert!(validate_min_level(0,  true, true).is_ok());    // unset ⇒ no conflict
    }
}
