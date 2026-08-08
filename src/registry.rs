//! Model registry — `~/.config/tokenov/models.toml`.
//!
//! Auto-managed by `tokenov build`: each successful build records the model's
//! name (output file stem), absolute on-disk path, size, and build time. `tokenov
//! --list-models` reads it back. Hand-rolled TOML (no serde/toml dep), same style as the
//! `.tune.toml` sidecar. Format:
//!
//! ```toml
//! [[model]]
//! name = "llama"
//! path = "/abs/path/llama.ngram"
//! size_bytes = 360710144
//! built_epoch = 1750000000
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct ModelEntry {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub built_epoch: u64,
    /// SHA-256 of the model file at register time. `None` for entries recorded
    /// before hashing was added (backfill via `tokenov model verify --update`).
    pub sha256: Option<String>,
}

/// SHA-256 of a file, lowercase hex. Streamed so a multi-GB `.ngram` never loads
/// whole into memory.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// `$XDG_CONFIG_HOME/tokenov/models.toml`, else `$HOME/.config/tokenov/models.toml`.
pub fn registry_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from(".config"));
    base.join("tokenov").join("models.toml")
}

/// The writable tokenizer manifest: `$XDG_CONFIG_HOME/tokenov/tokenizers.toml`,
/// alongside `models.toml`. Seeded from the binary's embedded default on first
/// use; authoritative thereafter (so `tokenizer add`/`delete` persist).
pub fn tokenizer_manifest_path() -> PathBuf {
    registry_path().with_file_name("tokenizers.toml")
}

/// Tokenov data root: `$XDG_DATA_HOME/tokenov/`, else
/// `$HOME/.local/share/tokenov/`. The parent of all tokenov-owned stores
/// (models, fetched tokenizers, bootstrap corpora).
pub fn data_dir() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))
        .unwrap_or_else(|| PathBuf::from(".local").join("share"));
    base.join("tokenov")
}

/// Default model store: `<data_dir>/models/`. This is where `tokenov build`
/// writes when `--output` is omitted, so the common case needs no path juggling.
pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

/// Where `tokenov fetch` writes downloaded tokenizer.json files, one per alias:
/// `<data_dir>/tokenizers/<alias>/tokenizer.json`.
pub fn tokenizers_dir() -> PathBuf {
    data_dir().join("tokenizers")
}

/// Scratch + output store for `tokenov bootstrap` (downloaded RockYou tarball,
/// the frequency-expanded training corpus): `<data_dir>/bootstrap/`.
pub fn bootstrap_dir() -> PathBuf {
    data_dir().join("bootstrap")
}

pub fn read_registry() -> Vec<ModelEntry> {
    match fs::read_to_string(registry_path()) {
        Ok(txt) => parse_registry(&txt),
        Err(_) => Vec::new(),
    }
}

fn parse_registry(txt: &str) -> Vec<ModelEntry> {
    let mut out = Vec::new();
    let mut cur: Option<ModelEntry> = None;
    for line in txt.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[model]]" {
            if let Some(e) = cur.take() {
                out.push(e);
            }
            cur = Some(ModelEntry {
                name: String::new(), path: String::new(),
                size_bytes: 0, built_epoch: 0, sha256: None,
            });
        } else if let (Some(e), Some((k, v))) = (cur.as_mut(), line.split_once('=')) {
            let v = v.trim().trim_matches('"');
            match k.trim() {
                "name" => e.name = v.to_string(),
                "path" => e.path = v.to_string(),
                "size_bytes" => e.size_bytes = v.parse().unwrap_or(0),
                "built_epoch" => e.built_epoch = v.parse().unwrap_or(0),
                "sha256" => e.sha256 = Some(v.to_string()),
                _ => {}
            }
        }
    }
    if let Some(e) = cur.take() {
        out.push(e);
    }
    out
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn write_registry(entries: &[ModelEntry]) -> std::io::Result<()> {
    let path = registry_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut s = String::from(
        "# tokenov model registry — auto-managed by `tokenov build`.\n\
         # Lists built models by name with their on-disk location.\n\n",
    );
    for e in entries {
        s.push_str("[[model]]\n");
        s.push_str(&format!("name = \"{}\"\n", esc(&e.name)));
        s.push_str(&format!("path = \"{}\"\n", esc(&e.path)));
        s.push_str(&format!("size_bytes = {}\n", e.size_bytes));
        s.push_str(&format!("built_epoch = {}\n", e.built_epoch));
        if let Some(h) = &e.sha256 {
            s.push_str(&format!("sha256 = \"{}\"\n", h));
        }
        s.push('\n');
    }
    let tmp = path.with_extension("toml.tmp");
    fs::write(&tmp, s)?;
    fs::rename(&tmp, &path)?;
    Ok(())
}

/// Upsert a model by name (rebuilding a model updates its entry). Returns the
/// registry path so the caller can report where it was recorded.
pub fn register(name: &str, model_path: &Path, size_bytes: u64) -> std::io::Result<PathBuf> {
    let abs = model_path
        .canonicalize()
        .unwrap_or_else(|_| model_path.to_path_buf());
    let built_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let sha256 = sha256_file(model_path).ok();
    let mut entries = read_registry();
    entries.retain(|e| e.name != name);
    entries.push(ModelEntry {
        name: name.to_string(),
        path: abs.to_string_lossy().into_owned(),
        size_bytes,
        built_epoch,
        sha256,
    });
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    write_registry(&entries)?;
    Ok(registry_path())
}

/// Record/overwrite the sha256 for a named entry (used by `verify --update` to
/// backfill models built before hashing). Returns whether the name was found.
pub fn set_entry_hash(name: &str, sha256: &str) -> std::io::Result<bool> {
    let mut entries = read_registry();
    let mut found = false;
    for e in entries.iter_mut() {
        if e.name == name {
            e.sha256 = Some(sha256.to_string());
            found = true;
        }
    }
    if found {
        write_registry(&entries)?;
    }
    Ok(found)
}

/// Remove a model's registry entry by name. Returns the entry's stored path if
/// it was present (so the caller can optionally delete the file), else `None`.
pub fn deregister(name: &str) -> std::io::Result<Option<String>> {
    let mut entries = read_registry();
    let found = entries.iter().find(|e| e.name == name).map(|e| e.path.clone());
    if found.is_some() {
        entries.retain(|e| e.name != name);
        write_registry(&entries)?;
    }
    Ok(found)
}

/// Remove every entry whose on-disk path no longer exists (the `⚠ MISSING`
/// rows). Returns the names removed. Used by `tokenov delete --missing`.
pub fn remove_missing() -> std::io::Result<Vec<String>> {
    let entries = read_registry();
    let (gone, kept): (Vec<_>, Vec<_>) =
        entries.into_iter().partition(|e| !Path::new(&e.path).exists());
    if !gone.is_empty() {
        write_registry(&kept)?;
    }
    Ok(gone.into_iter().map(|e| e.name).collect())
}

/// Look up a single entry by exact name.
pub fn find(name: &str) -> Option<ModelEntry> {
    read_registry().into_iter().find(|e| e.name == name)
}

/// Resolve a `--model` argument: an existing file path is returned as-is;
/// otherwise a bare name is looked up in the registry (so `--model llama_freq`
/// works without the full path). Errors helpfully if neither resolves.
pub fn resolve_model(arg: &Path) -> anyhow::Result<PathBuf> {
    if arg.exists() {
        return Ok(arg.to_path_buf());
    }
    let name = arg.to_string_lossy();
    // Only do a name lookup for a bare name (no path separator); a non-existent
    // explicit path should report as a missing path, not an unknown name.
    if !name.contains(std::path::MAIN_SEPARATOR) {
        if let Some(e) = read_registry().into_iter().find(|e| e.name == *name) {
            let p = PathBuf::from(&e.path);
            if p.exists() {
                return Ok(p);
            }
            anyhow::bail!(
                "model '{}' is registered but its file is missing: {} \
                 (rebuild it; see `tokenov --list-models`)",
                name, e.path
            );
        }
    }
    anyhow::bail!(
        "'{}' is neither an existing file nor a registered model name \
         (see `tokenov --list-models`)",
        name
    );
}

/// Print the registry as a table to stdout (`tokenov --list-models`).
pub fn list() {
    let path = registry_path();
    let entries = read_registry();
    if entries.is_empty() {
        println!("No models registered (registry: {}).", path.display());
        println!("Build one with:  tokenov build --tokenizer <t> --train <r> --name <name>");
        return;
    }
    println!("Models registered in {}:\n", path.display());
    let namew = entries.iter().map(|e| e.name.len()).max().unwrap_or(4).max(4);
    println!("{:<nw$}  {:>10}  {:<16}  PATH", "NAME", "SIZE", "BUILT (UTC)", nw = namew);
    for e in &entries {
        let missing = if Path::new(&e.path).exists() { "" } else { "  ⚠ MISSING" };
        println!(
            "{:<nw$}  {:>10}  {:<16}  {}{}",
            e.name, human_size(e.size_bytes), fmt_epoch(e.built_epoch), e.path, missing,
            nw = namew,
        );
    }
}

fn human_size(b: u64) -> String {
    const U: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut f = b as f64;
    let mut i = 0;
    while f >= 1024.0 && i < U.len() - 1 {
        f /= 1024.0;
        i += 1;
    }
    if i == 0 { format!("{} B", b) } else { format!("{:.1} {}", f, U[i]) }
}

/// Public wrapper over the internal epoch formatter (for `model info`).
pub fn fmt_epoch_public(secs: u64) -> String {
    fmt_epoch(secs)
}

/// epoch seconds → "YYYY-MM-DD HH:MM" UTC (Hinnant civil_from_days; no chrono dep).
fn fmt_epoch(secs: u64) -> String {
    if secs == 0 {
        return "-".to_string();
    }
    let days = (secs / 86400) as i64;
    let rem = (secs % 86400) as i64;
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    let (h, mi) = (rem / 3600, (rem % 3600) / 60);
    format!("{:04}-{:02}-{:02} {:02}:{:02}", y, m, d, h, mi)
}
