//! `tokenov tokenizer {get,add,delete,list}` and `tokenov bootstrap`.
//!
//! Quickstart surfaces sharing one alias→URL manifest:
//!
//! * `tokenizer get` — download tokenizer.json files by alias (or `--all`); the
//!   in-binary replacement for a `fetch_tokenizers.sh` helper. (`tokenov fetch`
//!   is a hidden deprecated alias.) `add`/`delete` edit the manifest.
//! * `bootstrap` — the zero-input chain: fetch the default tokenizer, download +
//!   frequency-expand RockYou into a training corpus, then `model train` +
//!   register a trigram model. From a fresh clone to a usable model in one go.
//!
//! The alias→URL table is NOT hardcoded in Rust: it lives in `tokenizers.toml`,
//! embedded at compile time via `include_str!`, and is overridable at runtime
//! with `--manifest`. Network/decompression is delegated to `curl` and `tar`
//! (tokenov also shells out to `7z`/`hashcat`); both must be on PATH.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};
use tokenizers::Tokenizer;

use crate::log_msg;
use crate::registry;

/// The default manifest, baked into the binary so a fresh box needs no external
/// file. Overridable with `--manifest <path>`.
const DEFAULT_MANIFEST: &str = include_str!("../tokenizers.toml");

/// "Tokenov tokenizer v1" — the bundled default, embedded in the binary so a
/// fresh install resolves it with **zero network**.
///
/// ATTRIBUTION: this tokenizer is a derivative of **OpenAI GPT-2** (the 50,257-token
/// byte-level BPE vocabulary + merges from `openai-community/gpt2`, MIT-licensed).
/// The ONLY modification is the digit pre-tokenizer: a `Split(\p{N}{1,2})` is
/// prepended so digit runs tokenize in ≤2-digit groups (`2007` -> `20|07`), which
/// tested as a strong, license-clean alphabet. The alphabetic vocabulary is
/// GPT-2's, unchanged.
/// See `tokenizers/tokenov_v1/ATTRIBUTION.md`.
const TOKENOV_V1_TOKENIZER: &[u8] = include_bytes!("../tokenizers/tokenov_v1/tokenizer.json");

/// Resolve a `bundled:<name>` manifest source to its embedded bytes. Returns
/// `None` for any non-`bundled:` source (URL / local path), so existing fetch
/// behaviour is untouched.
fn bundled_source(source: &str) -> Option<&'static [u8]> {
    match source.strip_prefix("bundled:")? {
        "tokenov_v1" => Some(TOKENOV_V1_TOKENIZER),
        _ => None,
    }
}

/// The previous auto-default tokenizer. Existing user manifests that still carry
/// it are migrated to the current shipped default (`tokenov_v1`) on load — Qwen
/// is no longer the default. Users who want it back: `tokenizer set-default`.
const DEPRECATED_DEFAULT_ALIAS: &str = "qwen25_7b";

/// The effective default tokenizer/model alias (from the user manifest). The
/// bootstrap model is named after this alias, so it is also the default model
/// `tokenov generate` uses when no `--model` is given.
pub fn default_alias() -> Result<String> {
    let mpath = resolve_manifest_path(None)?;
    Ok(Manifest::load(&mpath)?.default_alias)
}

/// Public RockYou-with-count mirror (SecLists). `<count> <password>` lines,
/// gzip-tar. Frequency-expanded into the bootstrap training corpus.
const DEFAULT_ROCKYOU_URL: &str =
    "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Leaked-Databases/rockyou-withcount.txt.tar.gz";

// ----------------------------------------------------------------------------
// Manifest
// ----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TokEntry {
    pub alias: String,
    pub url: String,
    pub note: String,
}

#[derive(Debug, Clone)]
pub struct Manifest {
    pub default_alias: String,
    pub entries: Vec<TokEntry>,
}

/// Resolve which manifest file to use, seeding the writable user file from the
/// embedded default on first use:
///   explicit `--manifest <path>` (must exist) > user file (seed if absent).
/// The embedded string is only a seed; once the file exists it is authoritative
/// so `tokenizer add`/`delete` persist.
fn resolve_manifest_path(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if !p.exists() {
            bail!("manifest not found: {}", p.display());
        }
        return Ok(p.to_path_buf());
    }
    let user = registry::tokenizer_manifest_path();
    if !user.exists() {
        if let Some(parent) = user.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        // Seed verbatim so the header comments survive on a fresh box.
        std::fs::write(&user, DEFAULT_MANIFEST)
            .with_context(|| format!("seed tokenizer manifest {}", user.display()))?;
        log_msg(&format!("[tokenizer] seeded manifest at {}", user.display()));
    }
    // Keep an already-seeded user manifest current with the shipped binary:
    // pull in any new built-in entries (e.g. the bundled `tokenov_v1`) and
    // migrate the deprecated `qwen25_7b` default. Idempotent — only writes on a
    // real change, so it runs at most once per upgrade.
    sync_user_manifest(&user)?;
    Ok(user)
}

/// Bring an existing user manifest up to date with the embedded built-ins:
/// add any missing built-in entries, and bump the deprecated default alias to
/// the current shipped default. Writes back only when something changed.
fn sync_user_manifest(user: &Path) -> Result<()> {
    let mut man = match Manifest::load(user) {
        Ok(m) => m,
        Err(_) => return Ok(()), // unreadable/odd manifest: leave it untouched
    };
    let embedded = Manifest::parse(DEFAULT_MANIFEST)?;
    let mut changed = false;

    // Add built-in entries the user file predates (e.g. tokenov_v1).
    for e in &embedded.entries {
        if man.find(&e.alias).is_none() {
            man.entries.push(e.clone());
            changed = true;
        }
    }
    // One-time migration of the deprecated auto-default.
    if man.default_alias == DEPRECATED_DEFAULT_ALIAS
        && embedded.default_alias != DEPRECATED_DEFAULT_ALIAS
        && man.find(&embedded.default_alias).is_some()
    {
        log_msg(&format!(
            "[tokenizer] default migrated {} -> {} (override with `tokenov tokenizer set-default <alias>`)",
            man.default_alias, embedded.default_alias
        ));
        man.default_alias = embedded.default_alias.clone();
        changed = true;
    }
    if changed {
        man.save(user)?;
    }
    Ok(())
}

/// Aliases that ship in the embedded default (i.e. "built-in", not user-added).
/// Used to decide whether `tokenizer delete` may drop a manifest entry.
fn builtin_aliases() -> Vec<String> {
    Manifest::parse(DEFAULT_MANIFEST)
        .map(|m| m.entries.into_iter().map(|e| e.alias).collect())
        .unwrap_or_default()
}

impl Manifest {
    fn load(path: &Path) -> Result<Manifest> {
        let txt = std::fs::read_to_string(path)
            .with_context(|| format!("read manifest {}", path.display()))?;
        Self::parse(&txt)
    }

    /// Write the manifest back to `path` (atomic temp+rename). Regenerates the
    /// file, so a fixed header is emitted in place of the seed's comments.
    fn save(&self, path: &Path) -> Result<()> {
        let mut s = String::from(
            "# tokenov tokenizer manifest — machine-managed by `tokenov tokenizer add/delete`.\n\
             # alias -> ungated tokenizer.json URL (or a local file path). Seeded from the\n\
             # binary's embedded default; edit by hand or via the CLI. NOTE: rubert /\n\
             # rubert_conv ship only vocab.txt (no fetchable tokenizer.json) and are\n\
             # intentionally absent — no fetchable tokenizer.json exists for them.\n\n",
        );
        s.push_str(&format!("default_alias = \"{}\"\n\n", toml_esc(&self.default_alias)));
        for e in &self.entries {
            s.push_str("[[tokenizer]]\n");
            s.push_str(&format!("alias = \"{}\"\n", toml_esc(&e.alias)));
            s.push_str(&format!("url = \"{}\"\n", toml_esc(&e.url)));
            if !e.note.is_empty() {
                s.push_str(&format!("note = \"{}\"\n", toml_esc(&e.note)));
            }
            s.push('\n');
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, s).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("install {}", path.display()))?;
        Ok(())
    }

    /// Minimal hand-rolled parser for the `[[tokenizer]]` array-of-tables +
    /// top-level `default_alias` (matches the no-serde TOML style,
    /// like `registry.rs`). Only `alias`, `url`, `note` keys are recognized.
    fn parse(txt: &str) -> Result<Manifest> {
        let mut default_alias = String::new();
        let mut entries: Vec<TokEntry> = Vec::new();
        let mut cur: Option<TokEntry> = None;
        let flush = |cur: &mut Option<TokEntry>, out: &mut Vec<TokEntry>| {
            if let Some(e) = cur.take() {
                out.push(e);
            }
        };
        for line in txt.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line == "[[tokenizer]]" {
                flush(&mut cur, &mut entries);
                cur = Some(TokEntry {
                    alias: String::new(),
                    url: String::new(),
                    note: String::new(),
                });
            } else if let Some((k, v)) = line.split_once('=') {
                let v = v.trim().trim_matches('"').to_string();
                match (cur.as_mut(), k.trim()) {
                    (None, "default_alias") => default_alias = v,
                    (Some(e), "alias") => e.alias = v,
                    (Some(e), "url") => e.url = v,
                    (Some(e), "note") => e.note = v,
                    _ => {}
                }
            }
        }
        flush(&mut cur, &mut entries);
        entries.retain(|e| !e.alias.is_empty() && !e.url.is_empty());
        if entries.is_empty() {
            bail!("manifest has no usable [[tokenizer]] entries");
        }
        if default_alias.is_empty() {
            default_alias = entries[0].alias.clone();
        }
        Ok(Manifest { default_alias, entries })
    }

    fn find(&self, alias: &str) -> Option<&TokEntry> {
        self.entries.iter().find(|e| e.alias == alias)
    }

    fn aliases(&self) -> String {
        self.entries.iter().map(|e| e.alias.as_str()).collect::<Vec<_>>().join(", ")
    }
}

fn toml_esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn looks_like_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

// ----------------------------------------------------------------------------
// fetch
// ----------------------------------------------------------------------------

#[derive(clap::Args, Debug, Clone)]
pub struct FetchArgs {
    /// Aliases to fetch (e.g. `llama qwen25_7b`). Resolved against the manifest.
    /// Omit and pass `--all` to fetch every alias.
    pub aliases: Vec<String>,

    /// Fetch every tokenizer in the manifest.
    #[arg(long)]
    pub all: bool,

    /// Print the alias → URL table and exit (no downloads).
    #[arg(long)]
    pub list: bool,

    /// Destination root. Each tokenizer lands at `<dest>/<alias>/tokenizer.json`.
    /// Default: `<XDG_DATA_HOME>/tokenov/tokenizers/`.
    #[arg(long)]
    pub dest: Option<PathBuf>,

    /// Use this manifest file instead of the built-in default table.
    #[arg(long)]
    pub manifest: Option<PathBuf>,

    /// Re-download even if the destination tokenizer.json already exists.
    #[arg(long)]
    pub force: bool,
}

pub fn run_fetch(args: FetchArgs) -> Result<()> {
    let mpath = resolve_manifest_path(args.manifest.as_deref())?;
    let man = Manifest::load(&mpath)?;

    if args.list {
        println!("Tokenizers in manifest (default: {}):\n", man.default_alias);
        let w = man.entries.iter().map(|e| e.alias.len()).max().unwrap_or(5).max(5);
        for e in &man.entries {
            println!("  {:<w$}  {}", e.alias, e.url, w = w);
        }
        return Ok(());
    }

    let dest_root = args.dest.clone().unwrap_or_else(registry::tokenizers_dir);

    // Resolve which entries to fetch.
    let targets: Vec<&TokEntry> = if args.all {
        man.entries.iter().collect()
    } else if args.aliases.is_empty() {
        bail!("nothing to fetch: pass one or more aliases, or --all (see `tokenov fetch --list`)");
    } else {
        let mut v = Vec::new();
        for a in &args.aliases {
            match man.find(a) {
                Some(e) => v.push(e),
                None => bail!("unknown alias '{}' — available: {}", a, man.aliases()),
            }
        }
        v
    };

    // Fetch resiliently: one dead URL must not abort the rest (a fresh-box
    // `--all` should bring down everything it can, then report what failed).
    let mut fetched = 0usize;
    let mut skipped = 0usize;
    let mut failures: Vec<(String, String)> = Vec::new();
    for e in &targets {
        match fetch_one(e, &dest_root, args.force) {
            Ok(FetchOutcome::Fetched) => fetched += 1,
            Ok(FetchOutcome::Cached) => skipped += 1,
            Err(err) => {
                crate::warn_msg(&format!("[fetch] {} FAILED: {}", e.alias, err));
                failures.push((e.alias.clone(), err.to_string()));
            }
        }
    }
    println!(
        "Fetched {} tokenizer(s), {} cached, {} failed ({} target(s)).",
        fetched, skipped, failures.len(), targets.len()
    );
    if !failures.is_empty() {
        let names = failures.iter().map(|(a, _)| a.as_str()).collect::<Vec<_>>().join(", ");
        bail!("{} tokenizer(s) failed to fetch: {}", failures.len(), names);
    }
    Ok(())
}

enum FetchOutcome {
    Fetched,
    Cached,
}

fn fetch_one(e: &TokEntry, dest_root: &Path, force: bool) -> Result<FetchOutcome> {
    let dir = dest_root.join(&e.alias);
    let path = dir.join("tokenizer.json");
    if path.exists() && !force {
        log_msg(&format!("[fetch] {} already present at {} (--force to refresh)", e.alias, path.display()));
        return Ok(FetchOutcome::Cached);
    }
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create {}", dir.display()))?;
    log_msg(&format!("[fetch] {} <- {}", e.alias, e.url));
    // A manifest entry's source is a URL (curl) or a local file path (copy) —
    // the latter supports `tokenizer add <alias> <file>`.
    if let Some(bytes) = bundled_source(&e.url) {
        // Bundled tokenizer (e.g. tokenov_v1): embedded in the binary, no network.
        std::fs::write(&path, bytes)
            .with_context(|| format!("write bundled tokenizer to {}", path.display()))?;
    } else if looks_like_url(&e.url) {
        curl_to_file(&e.url, &path)?;
    } else {
        let src = Path::new(&e.url);
        if !src.exists() {
            bail!("source for '{}' is neither a URL nor an existing file: {}", e.alias, e.url);
        }
        std::fs::copy(src, &path)
            .with_context(|| format!("copy {} -> {}", src.display(), path.display()))?;
    }
    // Validate it actually loads as a tokenizer; a 200-that-was-HTML would pass
    // curl but fail here, which is the failure we want to catch.
    Tokenizer::from_file(&path)
        .map_err(|err| anyhow!("fetched {} but it does not load as a tokenizer: {}", path.display(), err))?;
    log_msg(&format!("[fetch] {} ok -> {}", e.alias, path.display()));
    Ok(FetchOutcome::Fetched)
}

// ----------------------------------------------------------------------------
// tokenizer add / delete / list-status
// ----------------------------------------------------------------------------

#[derive(clap::Args, Debug, Clone)]
pub struct AddArgs {
    /// Short alias to register (e.g. `myбert`). Becomes `tokenizer get <alias>`.
    pub alias: String,
    /// tokenizer.json source: an `http(s)://…` URL or a local file path.
    pub source: String,
    /// Optional free-text note (remaining words are joined).
    pub note: Vec<String>,
    /// Overwrite an existing entry of the same alias.
    #[arg(long)]
    pub force: bool,
    /// Manifest file to edit (default: the user manifest, seeded if absent).
    #[arg(long)]
    pub manifest: Option<PathBuf>,
}

pub fn run_add(args: AddArgs) -> Result<()> {
    let mpath = resolve_manifest_path(args.manifest.as_deref())?;
    let mut man = Manifest::load(&mpath)?;
    if man.find(&args.alias).is_some() && !args.force {
        bail!("alias '{}' already in manifest (pass --force to overwrite)", args.alias);
    }
    // Normalize a local source to an absolute path so `get` works from any cwd.
    let source = if looks_like_url(&args.source) {
        args.source.clone()
    } else {
        let p = Path::new(&args.source);
        if !p.exists() {
            bail!("source is neither a URL nor an existing file: {}", args.source);
        }
        std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
            .to_string_lossy().into_owned()
    };
    man.entries.retain(|e| e.alias != args.alias);
    man.entries.push(TokEntry {
        alias: args.alias.clone(),
        url: source,
        note: args.note.join(" "),
    });
    man.save(&mpath)?;
    log_msg(&format!("[tokenizer] added '{}' to {}", args.alias, mpath.display()));
    Ok(())
}

#[derive(clap::Args, Debug, Clone)]
pub struct TokDeleteArgs {
    /// Alias to remove.
    pub alias: String,
    /// Tokenizer store root (default: `<XDG_DATA_HOME>/tokenov/tokenizers/`).
    #[arg(long)]
    pub dest: Option<PathBuf>,
    /// Manifest file to edit (default: the user manifest, seeded if absent).
    #[arg(long)]
    pub manifest: Option<PathBuf>,
}

pub fn run_tok_delete(args: TokDeleteArgs) -> Result<()> {
    let mpath = resolve_manifest_path(args.manifest.as_deref())?;
    let mut man = Manifest::load(&mpath)?;
    let dest_root = args.dest.clone().unwrap_or_else(registry::tokenizers_dir);

    // 1. Remove the downloaded file (if present).
    let file = dest_root.join(&args.alias).join("tokenizer.json");
    if file.exists() {
        std::fs::remove_file(&file).with_context(|| format!("remove {}", file.display()))?;
        // Drop the now-empty alias dir, best-effort.
        let _ = std::fs::remove_dir(file.parent().unwrap());
        log_msg(&format!("[tokenizer] removed downloaded {}", file.display()));
    } else {
        log_msg(&format!("[tokenizer] no downloaded file for '{}' (nothing to remove)", args.alias));
    }

    // 2. Drop the manifest entry — but only for user-added aliases. A built-in
    //    (seed) alias stays in the manifest; only its file is removed.
    if builtin_aliases().iter().any(|a| a == &args.alias) {
        log_msg(&format!("[tokenizer] '{}' is a built-in alias; kept in manifest (file removed only)", args.alias));
    } else if man.find(&args.alias).is_some() {
        man.entries.retain(|e| e.alias != args.alias);
        man.save(&mpath)?;
        log_msg(&format!("[tokenizer] removed '{}' from manifest {}", args.alias, mpath.display()));
    }
    Ok(())
}

#[derive(clap::Args, Debug, Clone)]
pub struct SetDefaultArgs {
    /// Alias to make the default. Must already be in the manifest. This sets
    /// both the default tokenizer (for `bootstrap`) and the default model name
    /// (for `generate` with no --model).
    pub alias: String,
    /// Manifest file to edit (default: the user manifest, seeded if absent).
    #[arg(long)]
    pub manifest: Option<PathBuf>,
}

/// `tokenov tokenizer set-default <alias>` — set the default tokenizer/model.
pub fn run_set_default(args: SetDefaultArgs) -> Result<()> {
    let mpath = resolve_manifest_path(args.manifest.as_deref())?;
    let mut man = Manifest::load(&mpath)?;
    if man.find(&args.alias).is_none() {
        bail!(
            "unknown alias '{}' — available: {}\n(add it first with `tokenov tokenizer add {} <url|file>`)",
            args.alias, man.aliases(), args.alias
        );
    }
    if man.default_alias == args.alias {
        println!("'{}' is already the default.", args.alias);
        return Ok(());
    }
    man.default_alias = args.alias.clone();
    man.save(&mpath)?;
    log_msg(&format!("[tokenizer] default set to '{}' -> {}", args.alias, mpath.display()));
    println!(
        "Default is now '{a}'. `tokenov bootstrap` will use it, and `tokenov generate` \
         (no --model) resolves model '{a}'.",
        a = args.alias
    );
    Ok(())
}

/// `tokenov tokenizer` (bare): list manifest aliases with download status.
pub fn run_list_status(manifest: Option<&Path>, dest: Option<&Path>) -> Result<()> {
    let mpath = resolve_manifest_path(manifest)?;
    let man = Manifest::load(&mpath)?;
    let dest_root = dest.map(Path::to_path_buf).unwrap_or_else(registry::tokenizers_dir);
    let builtin = builtin_aliases();

    println!("Tokenizers (manifest: {}, default: {})\n", mpath.display(), man.default_alias);
    let w = man.entries.iter().map(|e| e.alias.len()).max().unwrap_or(5).max(5);
    println!("  {:<w$}  {:<12}  {:<7}  SOURCE", "ALIAS", "DOWNLOADED", "ORIGIN", w = w);
    for e in &man.entries {
        let file = dest_root.join(&e.alias).join("tokenizer.json");
        let status = match std::fs::metadata(&file) {
            Ok(m) => format!("✓ {}", human_size(m.len())),
            Err(_) => "✗ no".to_string(),
        };
        let origin = if builtin.iter().any(|a| a == &e.alias) { "builtin" } else { "added" };
        println!("  {:<w$}  {:<12}  {:<7}  {}", e.alias, status, origin, e.url, w = w);
    }
    println!("\nDownload with:  tokenov tokenizer get <alias>   (or --all)");
    Ok(())
}

fn human_size(b: u64) -> String {
    if b >= 1 << 20 { format!("{:.1}M", b as f64 / (1u64 << 20) as f64) }
    else if b >= 1 << 10 { format!("{:.0}K", b as f64 / (1u64 << 10) as f64) }
    else { format!("{}B", b) }
}

// ----------------------------------------------------------------------------
// bootstrap
// ----------------------------------------------------------------------------

#[derive(clap::Args, Debug, Clone)]
pub struct BootstrapArgs {
    /// Tokenizer to use: a manifest alias, a tokenizer.json URL, or a local
    /// path. Default: the manifest's `default_alias`.
    #[arg(long)]
    pub tokenizer: Option<String>,

    /// Registry name for the built model. Default: `<alias>-quickstart`.
    #[arg(long)]
    pub name: Option<String>,

    /// RockYou-with-count source (gzip tarball of `<count> <password>` lines).
    #[arg(long)]
    pub rockyou_url: Option<String>,

    /// Use this manifest file instead of the built-in default table.
    #[arg(long)]
    pub manifest: Option<PathBuf>,

    /// Print the plan + equivalent manual commands, then exit without doing
    /// anything (no network, no writes).
    #[arg(long)]
    pub dry_run: bool,

    /// Skip the plan print and run silently-enough for CI.
    #[arg(short = 'y', long)]
    pub yes: bool,

    /// Re-download / re-expand / rebuild even if cached artifacts exist.
    #[arg(long)]
    pub force: bool,
}

pub fn run_bootstrap(args: BootstrapArgs) -> Result<()> {
    let mpath = resolve_manifest_path(args.manifest.as_deref())?;
    let man = Manifest::load(&mpath)?;
    let rockyou_url = args.rockyou_url.clone().unwrap_or_else(|| DEFAULT_ROCKYOU_URL.to_string());

    // Resolve the tokenizer spec: alias | URL | path.
    let spec = args.tokenizer.clone().unwrap_or_else(|| man.default_alias.clone());
    let (alias, tok_source) = resolve_tokenizer_source(&man, &spec);
    // The bootstrap model takes the tokenizer alias's name verbatim (e.g.
    // `tokenov_v1`), so it doubles as the default model `generate` resolves when
    // no --model is given (default model name == manifest default_alias).
    let model_name = args.name.clone().unwrap_or_else(|| alias.clone());

    let boot = registry::bootstrap_dir();
    let tok_path = registry::tokenizers_dir().join(&alias).join("tokenizer.json");
    let corpus_path = boot.join("rockyou_withcount_expanded.txt");
    let model_path = registry::models_dir().join(format!("{model_name}.ngram"));

    print_plan(&alias, &tok_source, &rockyou_url, &tok_path, &corpus_path, &model_name, &model_path);

    if args.dry_run {
        println!("\n[dry-run] no network calls or writes performed.");
        return Ok(());
    }
    if !args.yes {
        println!("\nProceeding automatically (Ctrl-C to abort; --dry-run to preview only).\n");
    }

    // 1. Tokenizer ----------------------------------------------------------
    let resolved_tok = match &tok_source {
        TokSource::Path(p) => {
            if !p.exists() {
                bail!("tokenizer path does not exist: {}", p.display());
            }
            p.clone()
        }
        TokSource::Url(url) => {
            if tok_path.exists() && !args.force {
                log_msg(&format!("[bootstrap] tokenizer cached: {}", tok_path.display()));
            } else {
                std::fs::create_dir_all(tok_path.parent().unwrap())?;
                log_msg(&format!("[bootstrap] fetch tokenizer {} <- {}", alias, url));
                curl_to_file(url, &tok_path)?;
            }
            Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("fetched tokenizer at {} does not load: {}", tok_path.display(), e))?;
            tok_path.clone()
        }
        // Bundled "Tokenov tokenizer v1" (GPT-2 + {1,2}): write the embedded bytes
        // to the cache, no network. Idempotent like the Url arm.
        TokSource::Bundled(bytes, label) => {
            if tok_path.exists() && !args.force {
                log_msg(&format!("[bootstrap] tokenizer cached: {}", tok_path.display()));
            } else {
                std::fs::create_dir_all(tok_path.parent().unwrap())?;
                log_msg(&format!("[bootstrap] writing bundled tokenizer '{}' -> {}", label, tok_path.display()));
                std::fs::write(&tok_path, bytes)
                    .with_context(|| format!("write bundled tokenizer to {}", tok_path.display()))?;
            }
            Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("bundled tokenizer at {} does not load: {}", tok_path.display(), e))?;
            tok_path.clone()
        }
    };

    // 2. RockYou corpus -----------------------------------------------------
    std::fs::create_dir_all(&boot)?;
    if corpus_path.exists() && !args.force {
        log_msg(&format!("[bootstrap] corpus cached: {} (--force to rebuild)", corpus_path.display()));
    } else {
        let raw = fetch_and_extract_rockyou(&rockyou_url, &boot, args.force)?;
        let (uniq, total) = expand_withcount(&raw, &corpus_path)?;
        log_msg(&format!("[bootstrap] expanded corpus: {} unique -> {} lines (with frequency dups)", uniq, total));
    }

    // 3. Build + register ---------------------------------------------------
    log_msg(&format!("[bootstrap] building trigram model '{}' ...", model_name));
    build_via_self(&resolved_tok, &corpus_path, &model_name, args.force)?;

    println!();
    log_msg(&format!("[bootstrap] done. Registered model '{}'.", model_name));
    if model_name == man.default_alias {
        println!("\nNext:  tokenov generate --count 1000000 | head   # '{}' is the default model", model_name);
    } else {
        println!("\nNext:  tokenov generate --model {} --count 1000000 | head", model_name);
    }
    Ok(())
}

enum TokSource {
    Url(String),
    Path(PathBuf),
    /// Bytes embedded in the binary (the bundled "Tokenov tokenizer v1"). The
    /// String is a human label for the plan printout.
    Bundled(&'static [u8], String),
}

impl std::fmt::Display for TokSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokSource::Url(u) => write!(f, "{u}"),
            TokSource::Path(p) => write!(f, "{}", p.display()),
            TokSource::Bundled(_, label) => write!(f, "{label} (bundled in binary)"),
        }
    }
}

/// Resolve a `--tokenizer` spec into (alias-for-naming, source). A bare alias is
/// looked up in the manifest; a URL or path is used verbatim with a derived
/// alias for output naming.
fn resolve_tokenizer_source(man: &Manifest, spec: &str) -> (String, TokSource) {
    if looks_like_url(spec) {
        return ("custom".to_string(), TokSource::Url(spec.to_string()));
    }
    let p = Path::new(spec);
    if p.exists() || spec.contains(std::path::MAIN_SEPARATOR) || spec.ends_with(".json") {
        let alias = p.file_stem().and_then(|s| s.to_str()).unwrap_or("custom").to_string();
        return (alias, TokSource::Path(p.to_path_buf()));
    }
    match man.find(spec) {
        Some(e) => match bundled_source(&e.url) {
            // A `bundled:` entry (e.g. tokenov_v1) → embedded bytes, no network.
            Some(bytes) => (e.alias.clone(), TokSource::Bundled(bytes, e.alias.clone())),
            None => (e.alias.clone(), TokSource::Url(e.url.clone())),
        },
        // Unknown bare token: treat as a path so the error surfaces at use-time
        // with a clear "does not exist" rather than a confusing alias message.
        None => (spec.to_string(), TokSource::Path(PathBuf::from(spec))),
    }
}

#[allow(clippy::too_many_arguments)]
fn print_plan(
    alias: &str,
    tok_source: &TokSource,
    rockyou_url: &str,
    tok_path: &Path,
    corpus_path: &Path,
    model_name: &str,
    model_path: &Path,
) {
    println!("tokenov bootstrap — plan:\n");
    println!("  1. Tokenizer '{}'", alias);
    match tok_source {
        TokSource::Url(u) => {
            println!("       source : {}", u);
            println!("       dest   : {}", tok_path.display());
            println!("       manual : curl -fSL '{}' -o '{}'", u, tok_path.display());
        }
        TokSource::Path(p) => {
            println!("       source : {} (local, no download)", p.display());
        }
        TokSource::Bundled(_, label) => {
            println!("       source : {} — bundled in binary (GPT-2 + {{1,2}}, no download)", label);
            println!("       dest   : {}", tok_path.display());
        }
    }
    println!("\n  2. RockYou training corpus (frequency-expanded)");
    println!("       source : {}", rockyou_url);
    println!("       dest   : {}", corpus_path.display());
    println!("       manual : curl -fSL '{}' -o ry.tar.gz && tar xzf ry.tar.gz", rockyou_url);
    println!("                # then expand each `<count> <pw>` line to `pw` repeated count×");
    println!("\n  3. Build + register trigram model '{}'", model_name);
    println!("       dest   : {}", model_path.display());
    println!("       manual : tokenov model train --tokenizer <tokenizer.json> \\");
    println!("                  --train '{}' --name {}", corpus_path.display(), model_name);
    println!("\n  End state: registered model '{}', ready for `tokenov generate`.", model_name);
}

// ----------------------------------------------------------------------------
// shell-outs + corpus expansion
// ----------------------------------------------------------------------------

/// Download `url` to `dest` with curl (follows redirects, fails on HTTP error,
/// retries transient failures). Writes to a temp path then renames so a partial
/// download never leaves a truncated file at `dest`.
fn curl_to_file(url: &str, dest: &Path) -> Result<()> {
    let tmp = dest.with_extension("download.tmp");
    let status = Command::new("curl")
        .args([
            "-fSL",
            "--retry", "3",
            "--retry-delay", "2",
            "-o",
        ])
        .arg(&tmp)
        .arg(url)
        .status()
        .with_context(|| "spawn curl (is it installed and on PATH?)")?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        bail!("curl failed for {} (exit {:?})", url, status.code());
    }
    std::fs::rename(&tmp, dest)
        .with_context(|| format!("move download into place: {}", dest.display()))?;
    Ok(())
}

/// Download the RockYou-with-count tarball and extract its `.txt`. Returns the
/// path to the extracted `<count> <password>` file.
fn fetch_and_extract_rockyou(url: &str, work: &Path, force: bool) -> Result<PathBuf> {
    let tarball = work.join("rockyou-withcount.txt.tar.gz");
    if tarball.exists() && !force {
        log_msg(&format!("[bootstrap] tarball cached: {}", tarball.display()));
    } else {
        log_msg(&format!("[bootstrap] fetch RockYou <- {}", url));
        curl_to_file(url, &tarball)?;
    }
    log_msg("[bootstrap] extracting tarball ...");
    let status = Command::new("tar")
        .arg("xzf")
        .arg(&tarball)
        .arg("-C")
        .arg(work)
        .status()
        .with_context(|| "spawn tar (is it installed and on PATH?)")?;
    if !status.success() {
        bail!("tar failed to extract {}", tarball.display());
    }
    let extracted = work.join("rockyou-withcount.txt");
    if !extracted.exists() {
        bail!("expected {} after extraction but it is missing", extracted.display());
    }
    Ok(extracted)
}

/// Frequency-expand a `<count> <password>` file: each password is written
/// `count` times, preserving the corpus's with-duplicates frequency weighting.
/// Returns (unique_lines, total_lines_written).
///
/// Parsing: split on the FIRST whitespace run only. The count is right-justified
/// with leading spaces; the password follows and may itself contain spaces — so
/// `split_whitespace().collect()` would corrupt space-containing passwords.
fn expand_withcount(src: &Path, dst: &Path) -> Result<(u64, u64)> {
    use std::io::{BufRead, BufReader, BufWriter, Write};
    let f = std::fs::File::open(src).with_context(|| format!("open {}", src.display()))?;
    let mut reader = BufReader::with_capacity(8 << 20, f);
    // Write to a temp path and rename only on success, so an interrupted expand
    // (crash / Ctrl-C / OOM / disk-full) NEVER leaves a partial `dst` that the
    // bootstrap cache check (`corpus_path.exists()`) would later trust and train
    // on — the cause of undertrained quickstart models.
    let tmp = dst.with_extension("partial");
    let out = std::fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
    let mut w = BufWriter::with_capacity(8 << 20, out);

    // Byte-oriented: RockYou contains non-UTF-8 (latin-1/binary) passwords, so we
    // must NOT decode lines as UTF-8 (`BufRead::lines()` would error "stream did
    // not contain valid UTF-8"). Reading raw bytes both fixes that and *preserves*
    // those passwords — matching a byte-preserving training corpus.
    let mut uniq = 0u64;
    let mut total = 0u64;
    let mut rec_stats = crate::recover::RecoveryStats::default();
    let mut buf: Vec<u8> = Vec::with_capacity(256);
    loop {
        buf.clear();
        if reader.read_until(b'\n', &mut buf)? == 0 {
            break;
        }
        // Strip trailing newline(s) and leading count-padding (spaces/tabs).
        while matches!(buf.last(), Some(b'\n') | Some(b'\r')) {
            buf.pop();
        }
        let start = buf.iter().position(|&b| b != b' ' && b != b'\t').unwrap_or(buf.len());
        let line = &buf[start..];
        // Split on the FIRST space/tab only: `<count> <password>`, where the
        // password may itself contain spaces.
        let Some(sep) = line.iter().position(|&b| b == b' ' || b == b'\t') else {
            continue;
        };
        // The count field is ASCII digits — safe to decode; the password is not.
        let Ok(count) = std::str::from_utf8(&line[..sep]).map_err(|_| ()).and_then(|s| s.parse::<u64>().map_err(|_| ())) else {
            continue;
        };
        let pw = &line[sep + 1..];
        if count == 0 || pw.is_empty() {
            continue;
        }
        // Recover non-UTF-8 (legacy-codepage) passwords to UTF-8 so they train
        // rather than being dropped; skip only if no encoding yields plausible
        // text. Recovered text is emitted ×count, preserving frequency weight.
        let decoded = crate::recover::decode_line(pw);
        rec_stats.note(&decoded);
        let Some(text) = decoded.into_text() else { continue };
        let out_bytes = text.as_bytes();
        uniq += 1;
        for _ in 0..count {
            w.write_all(out_bytes)?;
            w.write_all(b"\n")?;
        }
        total += count;
    }
    if let Some(msg) = rec_stats.report() {
        log_msg(&format!("[expand] {}", msg));
    }
    w.flush()?;
    drop(w);
    // Atomic publish: the complete corpus appears at `dst` in one step.
    std::fs::rename(&tmp, dst)
        .with_context(|| format!("finalize corpus {} -> {}", tmp.display(), dst.display()))?;
    Ok((uniq, total))
}

/// Run the build step by invoking this same binary's `build` subcommand. Reuses
/// the real `run_build` code path (and its auto-registration) without coupling
/// bootstrap to `BuildArgs`'s internals.
fn build_via_self(tokenizer: &Path, train: &Path, name: &str, force: bool) -> Result<()> {
    let exe = std::env::current_exe().context("locate own executable for build step")?;
    let mut cmd = Command::new(exe);
    cmd.arg("model").arg("train")
        .arg("--tokenizer").arg(tokenizer)
        .arg("--train").arg(train)
        .arg("--name").arg(name);
    if force {
        cmd.arg("--force");
    }
    let status = cmd.status().context("spawn `tokenov model train`")?;
    if !status.success() {
        bail!("build step failed (exit {:?})", status.code());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // The bundled "Tokenov tokenizer v1" must stay (a) resolvable, (b) a valid
    // tokenizer, and (c) the GPT-2 + \p{N}{1,2} build — `2007` -> `20|07`. This
    // guards the embedded artifact against an accidental swap/corruption.
    #[test]
    fn bundled_source_resolves_only_known_aliases() {
        assert!(bundled_source("bundled:tokenov_v1").is_some());
        assert!(bundled_source("bundled:nope").is_none());
        assert!(bundled_source("https://example.com/tokenizer.json").is_none());
        assert!(bundled_source("/local/path.json").is_none());
    }

    #[test]
    fn bundled_tokenov_v1_is_gpt2_with_two_digit_split() {
        let tmp = std::env::temp_dir().join("tokenov_v1_test_tokenizer.json");
        std::fs::write(&tmp, TOKENOV_V1_TOKENIZER).unwrap();
        let tok = Tokenizer::from_file(&tmp).expect("bundled tokenizer must load");
        let enc = tok.encode("whales2007", false).unwrap();
        let toks: Vec<&str> = enc.get_tokens().iter().map(|s| s.as_str()).collect();
        assert_eq!(toks, ["wh", "ales", "20", "07"], "expected GPT-2 + {{1,2}} digit split");
        let _ = std::fs::remove_file(&tmp);
    }
}
