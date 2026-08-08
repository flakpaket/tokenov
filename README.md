# tokenov

Token-level n-gram Markov password candidate generator. Trains an n-gram
model over passwords tokenized with any HuggingFace tokenizer and emits
candidate passwords in approximately rank-ordered (descending
joint-probability) order via a multithreaded OMEN-style level-sweep
enumerator.

The headline use case is password cracking: feed the candidate file to
`hashcat -a 0` or any wordlist-mode cracker. The token primitive (rather
than ASCII characters) is what makes the output structurally diverse —
multi-class strings like `Stamford12`, `Bella-boo`, `KOOL-AID5` appear in
the head of the distribution rather than as low-probability tail events.

## Build

```bash
git clone <this-repo>
cd tokenov
cargo build --release
# binary lands at target/release/tokenov
```

Requires Rust ≥ 1.85 (2024 edition).

## Quick start — zero to candidates

**The one-command path.** `tokenov bootstrap` does everything below for you —
uses its **bundled default tokenizer** (`tokenov_v1`, embedded in the binary, no
download), downloads + frequency-expands RockYou, then trains and registers a
model:

```bash
tokenov bootstrap --dry-run   # preview the plan + equivalent manual commands
tokenov bootstrap             # run it; ends with a registered model
tokenov generate --count 10   # Test run to generate 10 candidates
tokenov | hashcat -a 0 ...    # Pipe candidates to Hashcat stdin
```

## Inputs you'll need

### A tokenizer

**You don't need to fetch one yourself.** tokenov ships a bundled default
tokenizer (`tokenov_v1`, embedded in the binary — no download, works offline)
plus a built-in manifest of ungated aliases it can download for you, no
HuggingFace auth token required:

```bash
tokenov tokenizer                       # list every alias + download status
tokenov tokenizer get gpt2 --dest .     # → ./gpt2/tokenizer.json
tokenov tokenizer get --all             # fetch them all
```

Bundled aliases include `tokenov_v1` (the offline default), `gpt2`, `llama3`,
`qwen25_7b`, `mistral7b`, `falcon7b`, `gemma2_2b`, and more — run `tokenov
tokenizer` for the full list. You can also add your own
(`tokenov tokenizer add <alias> <url|file> "note"`) or train one from a
password corpus (`tokenov tokenizer train`). Most workflows never touch a
tokenizer file directly: `tokenov bootstrap` and `generate` use the bundled
default, and a trained model embeds its tokenizer.

**Bring your own.** tokenov also consumes any HuggingFace `tokenizer.json`
directly — fetch it however you like (`hf download gpt2 tokenizer.json
--local-dir ./tokenizers/gpt2`, or `AutoTokenizer.save_pretrained(...)`) and
point `--tokenizer` at the file.

Tokenov will work with any HF tokenizer, but be aware that:

- **WordPiece** tokenizers (BERT) emit subword tokens with `##` prefixes that
  appear in decoded output; you may need to filter the vocab before training.
- **SentencePiece** tokenizers (e.g. Mistral) prepend `▁` (U+2581) to
  word-initial tokens; tokenov decodes these correctly to spaces.
- Tokenizers trained on non-English text will produce candidates in those
  scripts.

### A training corpus (password list)

The model learns an n-gram distribution from a list of plaintext passwords
(one per line, UTF-8). Common sources:

- **[SecLists](https://github.com/danielmiessler/SecLists)** —
  `Passwords/Leaked-Databases/rockyou.txt.tar.gz` is the canonical RockYou
  leak (~14M unique passwords). Most password-cracking research uses this
  as the training set.
- **[weakpass.com](https://weakpass.com/wordlists)** — curated wordlists
  derived from many breaches.
- **Your own breach data** — for OSINT-aware engagements, train on a leak
  closer to your target population if available.

For experimentation, RockYou is the standard:

```bash
git clone --depth 1 https://github.com/danielmiessler/SecLists
tar xf SecLists/Passwords/Leaked-Databases/rockyou.txt.tar.gz -C ./data/
# wordlist now at ./data/rockyou.txt (~140 MB, ~14M entries)
```

### (Optional) An OSINT wordlist for targeted attacks

For wordlist-targeting modes (`--wordlist`), provide a small file
(typically 5-100 entries) with target-relevant terms — employee names,
project codenames, brand words, location names, etc. One entry per line.

```text
# example: healthcare_seeds.txt — industry/region terms, not a real org
healthcare
hospital
clinic
patient
nurse
pharma
cardio
radiology
telehealth
wellness
medicaid
hipaa
```

## Manual setup and configuration

The manual walkthrough below does the same steps explicitly. It is fully
self-contained: nothing but the built `tokenov` binary, runs in seconds, and
downloads ~1.3 MB. Every path is one you create in an earlier step.

```bash
# 0. Point at the binary you built above.
TOKENOV=/path/to/tokenov/target/release/tokenov

# 1. Get a tokenizer.json. Easiest is tokenov's built-in fetcher (no HF auth):
$TOKENOV tokenizer get gpt2 --dest .
# → ./gpt2/tokenizer.json
#   (list available aliases with `$TOKENOV tokenizer`; or download any HF
#    tokenizer.json yourself, e.g. `hf download gpt2 tokenizer.json --local-dir .`)

# 2. Make a tiny training corpus (one password per line). For a real
#    attack swap this for a leak like RockYou — see step 2b below.
printf '%s\n' password 123456 password1 letmein qwerty Summer2020 \
    dragon123 iloveyou monkey master Spring2021 hello123 superman \
    princess sunshine football welcome admin123 Bella-boo KOOL-AID5 \
    Stamford12 baseball shadow michael > train.txt

# 3. Fit the n-gram Markov model. These three flags are the full minimum.
$TOKENOV model train \
    --tokenizer ./gpt2/tokenizer.json \
    --train ./train.txt \
    --output ./model.ngram
# → ./model.ngram

# 4. Generate candidates (approximately rank-ordered). Minimal: model +
#    count + output. No --threads (auto-detected), no chunk-size tuning.
$TOKENOV generate \
    --model ./model.ngram \
    --count 1000 \
    --output ./candidates.txt
# → ./candidates.txt
```

What you'll see: `head candidates.txt` shows password-like lines
(`Bella-boo`, `KOOL-AID5`, `Summer2020`, `admin123` …) — multi-class
strings appear in the *head* of the distribution, which is the whole point
of using tokens instead of characters. The toy 24-line corpus exhausts its
keyspace at ~31 candidates even though you asked for 1000; that's expected
for a tiny list. A real training list emits billions of unique candidates
before exhausting.

**2b. Use a real wordlist.** For an actual attack, replace `train.txt`
with a breach corpus. RockYou is the research standard:

```bash
git clone --depth 1 https://github.com/danielmiessler/SecLists
tar xf SecLists/Passwords/Leaked-Databases/rockyou.txt.tar.gz -C .
# now build with --train ./rockyou.txt instead (~14M passwords; build is ~25–75s)
```

Or just run `tokenov bootstrap`, which downloads RockYou and frequency-expands
it (each password repeated by its real count) into a ready-to-train corpus — a
frequency-weighted training list rather than the flat deduped `rockyou.txt` —
then trains and registers the model in one step.

**Use the candidates** as a hashcat wordlist, or pipe them straight in:

```bash
hashcat -a 0 -m 0 hashes.txt ./candidates.txt
# or, no intermediate file (multithreaded stdout works):
$TOKENOV generate --model ./model.ngram --count 1000000000 | hashcat -a 0 -m 0 hashes.txt
```

**Compression at rest.** Tokenov writes plain UTF-8 text by design (so
`--resume` and stdout piping work) — it never compresses its own output.
If you want candidates compressed on disk, run it as a separate post-step:

```bash
7z a -mx=1 candidates.7z candidates.txt && rm candidates.txt
```

**Side effect of training.** Every `model train` registers the model in
`~/.config/tokenov/models.toml` so `tokenov model` (list) can find it later,
and `generate --model <name>` resolves it by name. If you omit `--output`, the
model is written to the default store
(`$XDG_DATA_HOME/tokenov/models/<name>.ngram`, else
`~/.local/share/tokenov/models/<name>.ngram`) under the tokenizer's
basename. Passing `--output` (as above) keeps the file in your working
directory; the registry entry is still written either way.

Throughput is hardware-dependent and scales roughly with vocab size (the
decode table and per-candidate byte payloads grow with it) — see the build
table under `tokenov model train` below for the relative cost of each tokenizer.
For a real 10⁹-candidate run, just omit `--threads` and let tokenov use all
detected cores.

## Subcommands

The CLI is grouped into noun-commands. `tokenov --help` lists them; each group
(`tokenov model`, `tokenov tokenizer`) lists its verbs.

| command | what it does |
|---|---|
| `tokenov bootstrap` | zero-input quickstart: fetch tokenizer + RockYou → train + register |
| `tokenov tokenizer` | list tokenizers + download status (bare); `get`/`add`/`delete`/`train`/`set-default` verbs |
| `tokenov model` | list registered models (bare); `train`/`register`/`delete`/`info`/`verify` verbs |
| `tokenov generate` | emit candidates (also the default if no subcommand is given) |
| `tokenov score` | report each candidate's Rarity (surprisal, in bits) under the model |

> The old flat names still work as hidden aliases — `tokenov build` →
> `tokenov model train`, `tokenov fetch` → `tokenov tokenizer get`, plus
> `delete`/`register` — but print a deprecation notice. Prefer the new forms.

### `tokenov model train`

Fits an n-gram Markov model over passwords tokenized with the supplied
HuggingFace tokenizer. Default order is trigram (the empirical sweet spot
— 4-grams over-memorize training sets <10M passwords). Output is a binary
`.ngram` file, format **`NGRMv003`**: the Kneser-Ney model body (trigram tier
with bigram backoff + per-context lambda + KN-continuation distribution)
prefixed by a **provenance header** that embeds the exact `tokenizer.json`
bytes plus build metadata (source paths, build time, tokenov version).

Because the tokenizer is embedded, a v3 model is **self-describing**:
`tokenov generate --wordlist` loads the correct tokenizer straight from the
`.ngram` — no sidecar, env var, or registry entry — so the model survives a
bare single-file copy. Older `NGRMv001`/`NGRMv002` models still load; for them
the tokenizer is resolved out-of-band (`TOKENOV_TOKENIZER` env → co-located
`<model>.tokenizer.json` sidecar → default-alias tokenizer). The model math is
byte-identical across v2 and v3 — the header is purely additive.

```bash
tokenov model train \
    --tokenizer tokenizer.json \
    --train train.txt \
    --output model.ngram \
    --ngram 3
```

Inspect a model's embedded provenance:

```bash
tokenov model info model.ngram      # tokenizer, train corpus, build time, version
```

Build times for typical configs on RockYou-train (14.2M passwords):

| tokenizer | vocab | model size | build time |
|---|---:|---:|---:|
| GPT-2 (`gpt2`) | 50,257 | ~250 MB | ~30s |
| Mistral (`mistralai/Mistral-7B-v0.1`) | 32,000 | ~200 MB | ~25s |

### `tokenov tokenizer`

Manage tokenizer downloads from a built-in alias → ungated-URL manifest (no
HuggingFace auth needed). Bare `tokenov tokenizer` lists every alias with its
download status and origin.

```bash
tokenov tokenizer                       # list aliases + download status
tokenov tokenizer train --corpus rockyou.txt --vocab-size 4000 --output my.json  # train your own
tokenov tokenizer get mistral7b         # download one (or --all)
tokenov tokenizer get gpt2 --dest .     # download into ./gpt2/tokenizer.json
tokenov tokenizer add mybert <url|file> "note"   # add your own alias
tokenov tokenizer delete mybert         # remove a download (+ user-added entry)
tokenov tokenizer set-default gpt2      # change the default tokenizer/model
```

**Train your own tokenizer** (`tokenizer train`) fits a byte-level **BPE** over a
password corpus and writes a standard `tokenizer.json` — closing the loop so you can
go corpus → tokenizer → model → candidates entirely in tokenov:

```bash
tokenov tokenizer train --corpus rockyou.txt --vocab-size 4000 --output my.json
tokenov model train --tokenizer my.json --train rockyou.txt --name my_model
tokenov generate --model my_model --count 1000000 | hashcat -a 0 -m 0 hashes.txt
```

`--vocab-size` (default 8000) is the knob that matters most: smaller vocabularies
(~1k–8k) often crack password data better than the 30k+ typical of web-text
tokenizers, and the sweet spot grows with the amount of unique training data — sweep
it for your corpus. The output matches the pipeline's tokenizer shape (BPE + ByteLevel
pre-tokenizer, no special tokens), so `model train` consumes it unmodified.

`set-default <alias>` sets `default_alias` in the manifest — the tokenizer
`bootstrap` uses **and** the model name `generate` resolves when you omit
`--model`. Ships as `tokenov_v1`; existing installs that still had the old
`qwen25_7b` default are migrated to `tokenov_v1` automatically on first run.

The manifest is seeded from a built-in default into
`~/.config/tokenov/tokenizers.toml` on first use; once it exists it is
authoritative, so `add`/`delete` persist. (A few WordPiece tokenizers — e.g.
DeepPavlov RuBERT — ship only `vocab.txt` and have no fetchable
`tokenizer.json`; those aren't in the manifest.)

The default alias **`tokenov_v1`** is special: its manifest source is
`bundled:tokenov_v1`, so it is **embedded in the binary** and resolves with no
network. It is *Tokenov tokenizer v1* — OpenAI GPT-2's vocabulary (MIT) with a
`\p{N}{1,2}` digit pre-tokenizer (`2007` → `20|07`), a strong, license-clean
default. Attribution + provenance:
[`tokenizers/tokenov_v1/ATTRIBUTION.md`](tokenizers/tokenov_v1/ATTRIBUTION.md).
(Existing installs keep whatever `default_alias` their user manifest already has;
only fresh installs pick `tokenov_v1` automatically.)

### `tokenov bootstrap`

Zero-input quickstart that chains the above: use the **bundled default tokenizer**
(`tokenov_v1`, embedded in the binary — no download), download + frequency-expand
RockYou, then `model train` + register. Prints a plan (with copy-pasteable manual
commands) and proceeds.

```bash
tokenov bootstrap --dry-run    # preview only; no network or writes
tokenov bootstrap              # → registers the default model `tokenov_v1`
```

> The bootstrap model trains on the **full** RockYou (no train/test split) — a
> get-running convenience, not for measuring crack rates.

### `tokenov generate`

Emits candidate passwords. Default subcommand if none is specified. `--model`
is **optional**: omit it to use the default model (the registry entry named by
the manifest's `default_alias` — `tokenov_v1` out of the box, built by
`tokenov bootstrap`). Pass `--model <name|path>` to choose another. Two
operating modes:

**Standard mode** (no `--wordlist`): emits candidates in approximately
descending joint-probability order. Uses a multithreaded OMEN-style
level sweep — the probability space is discretized into integer levels
via `ceil(-log_prob)`; each thread level-sweeps a disjoint first-level
subtree and appends its emissions directly to the sink (stdout or a
plain file). The partitions interleave, so the stream is *approximately*
rank-ordered; `--strict` runs a single producer for an exact global
order (see below).

```bash
tokenov generate --model model.ngram --count 1000000000 --output cands.txt
```

**Wordlist-targeting mode** (with `--wordlist`): roots generation at your
OSINT-derived seeds and grows Markov continuations from them. By **default** each
seed keeps its place as the prefix and affixes are **appended** (`cisco` →
`cisco2024`) — the common shape for OSINT seeds. Affix placement:

- **default / `--append-only`** — append affixes (seed stays the prefix).
- `--prepend-only` — prepend affixes (seed becomes the suffix, e.g. `ilovecisco`).
- `--float` — rarity-weighted graft with affixes on **either** side (the pre-0.20
  default).

```bash
# OSINT-style: target a community using its known terms (append-only default)
tokenov generate \
    --model model.ngram \
    --wordlist osint_seeds.txt \
    --count 100000 \
    --output cands.txt
```

(A legacy `--mode weighted|seeded|combined` with `--bias` / `--seed-mode` still
exists but is hidden from `--help`; the append/prepend/float generators supersede it.)

Wordlist mode tokenizes the wordlist with the **same tokenizer used at build
time**. tokenov resolves it automatically — in order:

1. `TOKENOV_TOKENIZER` env var (explicit override — always wins), then
2. the tokenizer **embedded in the model** (`NGRMv003` — the normal case, needs
   no external file), then
3. for older `NGRMv001`/`NGRMv002` models only, the out-of-band fallback: a
   co-located `<model>.tokenizer.json` sidecar, then (for the **default model**,
   no `--model`) the tokenizer `tokenov bootstrap` installed for the default alias.

So `tokenov generate --wordlist seeds.txt` works with no extra flags — against
the default model *or* any copied v3 model, sidecar or not. Override only to
deliberately re-tokenize with a different tokenizer (tokenov logs that it's
overriding the embedded one):

```bash
export TOKENOV_TOKENIZER=/path/to/tokenizer.json
```

#### Output format

Candidates write to `--output PATH` as **plain UTF-8 text, one
candidate per line**. Tokenov does not compress its output. If you
want compression at rest, run a post-step:

```bash
tokenov generate --model m.ngram --count 1000000000 --output cands.txt
7z a -mx=1 cands.7z cands.txt && rm cands.txt
```

Or pipe through gzip / zstd / xz on the fly:

```bash
tokenov generate --model m.ngram --count 1000000000 \
    | gzip > cands.txt.gz
```

If `--output` is omitted, candidates go to stdout. **Multithreaded
stdout works** — the streaming merger writes to whatever sink it has,
so `tokenov ... | hashcat ...` is the recommended attack-pipeline
pattern.

(Earlier versions wrote `.7z` archives directly via a 7z subprocess.
That mode was removed because resume can't append to a streaming
compressor's archive — plain-text output makes `--resume` work
universally.)

#### Resume

Every fast-mode `generate` run is **resumable by default** — no `--output`
required, so a killed `tokenov | hashcat` pipe can be picked back up.
tokenov periodically checkpoints each worker's enumeration position to a
rolling default state file (`$XDG_STATE_HOME/tokenov/generate.state`, else
`~/.local/state/tokenov/generate.state`). After a kill, re-run the SAME
command with `--resume`:

```bash
# original piped run — checkpointed automatically
tokenov generate --model m.ngram --count 1B | hashcat -m 0 hashes.txt

# killed? resume where it left off — no --output, no extra flags
tokenov generate --model m.ngram --count 1B --resume | hashcat -m 0 hashes.txt
```

Each worker reconstructs its stack in O(depth) and continues, instead of
re-enumerating from candidate 0. The resume validates that the model, args,
`--threads`, and tokenov version all match the checkpoint (errors on
mismatch rather than corrupting the stream). On clean completion the default
state file is removed (nothing left to resume).

To run several long tasks and return to a **specific** one, give each its own
checkpoint with `--checkpoint-file <FILE>` (instead of the shared default),
then resume that file with `--resume --checkpoint-file <FILE>` or
`--resume-state <FILE>`. `--no-checkpoint` disables the state file entirely.
`--checkpoint-secs` (default 300) sets the cadence, which is also the resume
safety margin — the last checkpoint lags the crash, so resume re-tests a
short overlap.

**Strict mode** (`--strict`) resumes the same way — O(depth), not by
re-enumerating. Being single-threaded it saves one DFS position; on kill it
records that position (checkpoint file) together with the output byte offset
(`<output>.progress` sidecar) at the same emitted count, so `--resume`
restores the position, truncates the output to the paired offset, and
continues. Strict resume therefore needs `--output` (it appends to the file;
there's no clean append path for a stdout stream). With `--no-checkpoint`
there's no saved position, and `--resume` falls back to re-running the
deterministic stream and skipping the first N already-written candidates via
the sidecar.

#### Merge chunk size (strict mode only)

`--merge-chunk-size` tunes the single-producer k-way merger used by
`--strict`. Fast mode (the default) writes each thread's partition directly
to the sink and has no merger, so this flag does nothing there. It batches
emissions per merger draw to amortize per-item overhead; larger values raise
throughput at the cost of a few items of imperfect ordering across chunks.

You rarely set it by hand. If it's unset, tokenov **auto-tunes** it during
the run and records the best value in a `<model>.ngram.tune.toml` sidecar so
future runs start from a good point; `--no-auto-tune` pins the built-in
default (262144) instead.

Optimal chunk size is model-dependent: small-vocab models are merger-bound
(bigger chunks help), while large-vocab models are producer-bound (the DFS
step dominates, so bigger chunks mostly add memory-copy pressure). The
auto-tuner handles this — only pin a value after benchmarking your own model.

#### Case shaping and enterprise (policy) mode

Two flags reshape the emitted candidates at generation time:

- **`--case-shape <SPEC>`** re-cases each selected token's decoded bytes. A
  spec is a per-token-slot pattern of `?l` (lower), `?c` (capitalize first),
  or `?u` (upper) — or a named shortcut `lower` / `cap1` / `title` / `upper` —
  and multiple `;`-separated specs expand each candidate into all of them.
  Useful for forcing case variants the model wouldn't rank highly on its own.

- **`--enterprise`** emits only candidates that satisfy a common corporate
  password policy: **≥ 8 characters** and **≥ 3 of 5 character classes**
  (lowercase, uppercase, digit, special, other). When a candidate is one
  class short *only* because it lacks an uppercase letter, tokenov applies a
  minimal capitalize-first repair rather than discarding it; otherwise
  non-compliant candidates are dropped. One guess in, at most one out — no
  rule-style multiplication. Mutually exclusive with `--case-shape`. This
  targets policy-enforced systems, where the bulk of an unshaped stream is
  wasted on candidates the target would never accept.

### `tokenov score`

The inverse of `generate`: instead of emitting candidates in probability
order, it takes candidates you supply and reports how (im)probable each one
is under the model. This is the FLA-style strength-meter view (Melicher et
al., USENIX 2016) — a neural/Markov model scoring an arbitrary password.

**Rarity** is the metric: the candidate's *surprisal*, `−log₂ P(candidate)`,
in **bits**. It answers "how many bits of information does this password
carry under the model" — higher Rarity = rarer = the model would enumerate
it later (a stronger password); lower Rarity = the model expects it early (a
weaker one). `Rarity/token` is the per-token surprisal (bits per token, i.e.
the cross-entropy of the segmentation); `2^(Rarity/token)` is the model's
perplexity on that candidate.

```bash
# Score a few candidates against the default model (table output)
tokenov score 123456 password correcthorsebatterystaple 'Tr0ub4dour&3'
```

```
Candidate            Tokens  Rarity(bits)  Segmentation
123456                    3           7.0  12|34|56
password                  1           9.1  password
correcthorsebattery…      6          92.8  correct|horse|b|attery|st|aple
Tr0ub4dour&3              8          87.9  Tr|0|ub|4|d|our|&|3
```

The columns: `Tokens` is the token count, `Rarity(bits)` the joint surprisal
(`-log2 P(candidate)` in bits; higher = rarer = stronger), and `Segmentation`
the plain token split (`hack|the|planet`). Long candidates are truncated in the
first column only; the full split still shows in `Segmentation`.

Pass **`-d` / `--detailed`** to add a per-position `Segment Score` column:

```bash
tokenov score -d 123456 correcthorsebatterystaple 'Tr0ub4dour&3'
```

```
Candidate            Tokens  Rarity(bits)  Segmentation              Segment Score
123456                    3           7.0  12|34|56                  START|12(5.3)|34(0.6)|56(0.5)|END(0.6)
correcthorsebattery…      6          92.8  correct|horse|b|attery|st|aple  START|correct(18.4)|horse(20.5*)|b(10.1)|attery(14.9)|st(15.9*)|aple(12.0)|END(0.9)
Tr0ub4dour&3              8          87.9  Tr|0|ub|4|d|our|&|3       START|Tr(14.6)|0(6.8)|ub(5.9)|4(7.7)|d(9.0)|our(16.8)|&(16.9*)|3(8.8)|END(1.5)
```

`Segment Score` is the **per-position breakdown** — `START` anchor, then each
predicted token (and the terminal `END`) annotated with the surprisal *it*
contributes, in bits. The parenthetical bits sum to `Rarity(bits)`, so you can
read straight across *where* a password is predictable vs. surprising — for
`123456`, the `12` prefix carries almost all the bits and `34`/`56` are near-free
continuations. This is more informative than a single length-averaged number: the
model's "surprise" is rarely spread evenly across the string.

A trailing `*` on a `Segment Score` step (e.g. `horse(20.5*)`) marks a
**floored** transition — one the trigram/bigram tiers had no mass for, so its
surprisal came from the add-k unigram floor (see below). It pinpoints exactly
which token the model couldn't cover, rather than only flagging that the
candidate is off-support somewhere. (`tsv`/`jsonl` always carry the per-position
data regardless of `-d`.)

Candidates come from positional args, `--file <path>` (one per line), or
stdin (one per line) if neither is given. `--model` resolves exactly like
`generate` (registered name or `.ngram` path; omit for the default model).

**What the score is, precisely.** The probability scored is the joint
`P(START, t₁, …, tₙ, END)` over the candidate's token segmentation — the
same per-step KN transition probabilities the enumerator uses, including the
terminal `END` transition. Including `END` is deliberate: `generate` ranks
*completed* strings, so `123456` (a complete password) is scored as such, not
as a prefix of something longer. The per-step transition math is a verbatim
copy of the generator's, so the score is consistent with the generator **at
the token-path level**. It is only *concordant*, not identical, with
generator emission order at the string level (~60% rank-concordance),
because the tokenizer's single greedy segmentation may differ from the
path the enumerator took, and Rarity does not marginalize over alternative
segmentations of the same string.

**Floored steps, and when Rarity is `+inf`.** A token transition the trigram
*and* the KN bigram-continuation tier both lack mass for is off-support. By
default, `score` backs such a step off to a full-vocab add-k unigram floor so
every candidate gets a finite Rarity — that step is marked `*` and the whole
candidate reports `in_vocab: false` (finite, but off the learned support). Pass
`--reachable` to disable the floor and restore the strict behavior: a floored
step then makes the whole candidate unreachable — `P = 0`, Rarity `= +inf` — and
the segmentation stops at the offending token (e.g. `correct(18.4)|horse(inf)`).
`--reachable` reproduces what `generate` in its default mode can never emit;
the finite default is the everyday strength-meter view.

**Output formats** (`--format`):

- `table` (default) — aligned, human-readable, long candidates truncated for
  display only. Shows a plain `Segmentation` column; `-d`/`--detailed` adds a
  `Segment Score` column with the per-position bits + `*` floor markers.
- `tsv` — `candidate, n_tok, surprisal_bits, surprisal_per_tok_bits, in_vocab,
  segmentation, per_pos_surprisal_bits, per_pos_floored`. The last two are
  comma-joined, one entry per transition (each token, then the trailing `END`
  step, so they hold `n_tok + 1` entries); `per_pos_surprisal_bits` sums to
  `surprisal_bits`. Columns use the precise term *surprisal* (Rarity is the same
  number); untruncated; feed straight into pandas/awk.
- `jsonl` — one JSON object per line, same scalar fields plus a `positions`
  array of per-transition `{label, surprisal_bits, floored}`; `+inf` is emitted
  as `null` so it parses. For analysis pipelines (e.g. studying whether a Rarity
  threshold predicts crackability).

```bash
# Bulk-score a wordlist for analysis
tokenov score --file candidates.txt --format tsv > rarity.tsv
```

## Advanced: generate on one machine, crack on another

Generation memory and GPU cracking pull in opposite directions. Cracking wants a
GPU box; **standard (unseeded) generation wants RAM** — its footprint scales with
the model's *context count* (the Kneser-Ney bigram-backoff child cache), not with
the `.ngram` file size or the GPU. A model trained on a very large or diverse
corpus can need far more RAM to generate at full speed than your GPU box has.

The fix is to **generate on a RAM-rich (or faster-CPU) host and stream candidates
to the GPU host over a plain `ssh` pipe** — no netcat, no ports, no extra tools:

```bash
# candidates are generated on `genhost` and streamed straight into local hashcat
ssh genhost 'tokenov generate --model big.ngram --count 1000000000' \
  | hashcat -a 0 -m 100 hashes.txt -r rules/dive.rule
```

Copy the `.ngram` to `genhost` first. `NGRMv003` models are self-describing (the
tokenizer is embedded), so a bare single-file copy is enough; older models need
their `<model>.tokenizer.json` sidecar alongside. Run the **same tokenov version**
on `genhost` — enumeration order is version-sensitive, so a matched binary keeps
results reproducible.

**Why `ssh` (and not a lossy transport):** with a rule file — especially a large
one — hashcat consumes *base* candidates slowly, because each base word expands
into (rule-count) GPU-side variants that must all be hashed before it reads the
next base. Its stdin intake is roughly `GPU_hash_rate ÷ rule_count`, often only
tens of KB/s. `ssh` runs over TCP, whose **flow control propagates that
backpressure back to the remote `tokenov`**, which then self-throttles to the
cracker's pace — losslessly, with no candidate skipped and no need to guess a
rate. A transport without flow control (e.g. raw UDP) has no backpressure: the
generator runs ahead, the receiver's buffer overflows, and candidates are
**silently dropped** — passwords that leave the generator but are never tried.
Don't do that.

**Bandwidth is not the bottleneck.** Full-bore generation tops out at tens of
MB/s (single-digit MB/s for large bounded-cache models), so any ordinary LAN —
even 1 GbE — carries it with room to spare. The generator is always the limiting
stage; the network and the cracker's backpressure govern the rest.

**Tuning generation RAM on the gen host.** `tokenov` picks the child-cache mode
automatically from available RAM, so it will not OOM without a flag — but you can
pin it:

- `--force-child-cache` — build the full resident cache (fastest; needs the most
  RAM, scaling with context count).
- `--bounded-cap N` — keep a partial cache of the `N` hottest contexts per thread
  (near-full speed at a fraction of the RAM; the auto-selector uses this by
  default when the full cache won't fit).
- `--lazy` — recompute children on demand (minimal RAM, slowest).
- `--max-rss-gb G` — an RSS ceiling / guard for the run.

Output is **byte-identical across all three modes** — the cache is purely a
performance/RAM trade. So put generation on your highest-RAM box to keep it in the
full or a large bounded cache and maximize candidates/second; the GPU box just
cracks. (Add `--strict` if you need a canonical single-threaded rank-ordered
stream; the default fast mode is multithreaded and higher-throughput.)

## Algorithm sketch

**Model**: a Kneser-Ney–smoothed n-gram (default trigram) over token IDs
produced by a HuggingFace tokenizer. Each context `(a, b)`'s trigram
distribution sums to `(1 − λ(a,b))`; the remaining `λ(a,b)` mass flows
to a bigram-continuation distribution `P_cont(t | b)` shared across all
contexts ending in `b`. Special tokens (`START`, `END`) are integers
above the tokenizer vocab size.

**Enumeration**: an outer loop sweeps `target_level = 0, 1, 2, …`; an
inner DFS visits all candidates at exactly that level and prunes
branches whose accumulated `−log_prob` exceeds it. All level-`L`
candidates emit before any level-`(L+1)` candidate, so output is
approximately rank-ordered. (The relaxation from "exactly rank-ordered"
to "approximately" comes from discretizing log-prob to integer levels
via `ceil(-lp)` — within a single level, output ordering is DFS rather
than strict rank.)

**Parallelization**: domain decomposition by first-level token. In the
default **fast** mode each thread level-sweeps a disjoint first-level
subtree and appends its emissions directly to a shared sink in batches (a
brief lock per ~256 KB), N-way interleaved — no channels, no merger. This
removes the single-merger bottleneck (~3.4× faster at 8 threads). Because
the partitions interleave, the global order is *approximate*: each
partition is internally rank-ordered, but items from different partitions
land in scheduling order (a few thousand items out of place globally).

**`--strict`** trades that parallelism for an exact global order: a single
producer streams the full rank order through a k-way merger — a min-heap of
`(level, sort_key)`-ordered chunks (size `--merge-chunk-size`) drained to
the sink — giving a byte-reproducible, model-canonical stream. The merge is
inherently serial (one ordered output), so strict runs single-threaded;
`--threads > 1` is ignored, and that costs no throughput versus a parallel
merge. No temp files either way — output streams live to stdout or a file.

**Wordlist targeting**: for V2/KN models, weighted/combined modes use
a KN-aware bias formulation that computes the joint emission
distribution per context (trigram entries plus the deduped λ-weighted
bigram-backoff entries), applies the multiplicative bias to W-tokens,
and renormalizes per context to sum to 1. Output is effectively a
v1-equivalent (single-tier) post-bias model; bigram-only contexts are
handled via a globally pre-biased + per-`b` renormalized `bigram_kn`
that the enumerator falls back to with `log_lam = 0`.

## CLI reference

Top-level options on `tokenov generate` (and the implicit default):

By default every subcommand is **quiet on stderr** — only warnings and errors
print; stdout still carries candidates (or a command's results). Pass **`-v` /
`--verbose`** (global — works before or after the subcommand) to see model-load,
tuning, and progress detail, plus per-tick `[stats]` telemetry. Wrapper tools can
use `--json` to get the machine telemetry stream on stderr without `-v`.

| flag | meaning | default |
|---|---|---|
| `-v`, `--verbose` | verbose stderr (model-load / tuning / progress / `[stats]`); global | quiet |
| `--json` | emit the machine telemetry stream on stderr without `-v` | (off) |
| `--strict` | exact global rank order (single producer, single-threaded, byte-reproducible) | (off → fast mode) |
| `--model <PATH>` | model file or registered name (`tokenov model train`) | (default model) |
| `--count <N>` | stop after N candidates (accepts K/M/B/T suffixes, e.g. `1B`) | unlimited |
| `--output <PATH>` | output destination (plain UTF-8 text); compress externally if you want | stdout |
| `--threads <N>` | enumeration thread count (ignored under `--strict`, which is always single-threaded) | rayon's CPU count |
| `--min-len <N>` | minimum candidate length, post-decode bytes | 4 |
| `--max-len <N>` | maximum candidate length, post-decode bytes | 30 |
| `--max-tokens <N>` | maximum tokens per heap-explored path | 12 |
| `--min-tokens <N>` | minimum tokens per candidate (drop-at-emit floor; 1 = no-op) | 1 |
| `--case-shape <SPEC>` | re-case each token (per-slot `?l`/`?c`/`?u`, or `lower`/`cap1`/`title`/`upper`; `;`-separated) | (off) |
| `--enterprise` | emit only policy-compliant candidates (≥8 chars + ≥3 of 5 classes; capitalize-first repair) | (off) |
| `--wordlist <PATH>` | OSINT/target wordlist (one entry per line) | (none → standard mode) |
| `--append-only` | append affixes to the seed (seed stays prefix) | **default** for `--wordlist` |
| `--prepend-only` | prepend affixes (seed becomes suffix) | (off) |
| `--float` | affixes on either side (rarity-weighted graft; pre-0.20 default) | (off) |
| `--mode <weighted\|seeded\|combined>` | legacy wordlist-targeting strategy (hidden) | (unset) |
| `--bias <FLOAT>` | legacy: strength multiplier for W-tokens (weighted/combined; hidden) | 2.0 |
| `--seed-mode <entry\|token>` | legacy: per-entry vs per-token seeding (hidden) | `entry` |
| `--merge-chunk-size <N>` | strict mode only: items per merger chunk (auto-tunes if unset) | (auto-tune) |
| `--resume` | continue the previous run from its checkpoint, O(depth) (fast: no `--output` needed; strict: restores its DFS position + byte offset, needs `--output`; `--no-checkpoint` falls back to the re-run + skip-N sidecar) | (off) |
| `--checkpoint-file <FILE>` | checkpoint to a named file instead of the rolling default (fast mode) | (default state file) |
| `--no-checkpoint` | disable the default checkpoint state file (fast mode) | (off) |
| `--checkpoint-secs <SEC>` | checkpoint cadence + resume safety margin | 300 |
| `--resume-state <FILE>` | resume a specific checkpoint file (explicit form of `--resume`) | (none) |
| `--max-rss-gb <GiB>` | abort generation if the process RSS exceeds this (safety cap) | (auto: ~75% of RAM) |
| `--sessions` | list recent checkpointed sessions and exit | — |
| `--resume-session <ID>` | resume a checkpointed session by id | (none) |

Options on `tokenov score`:

| flag | meaning | default |
|---|---|---|
| `[CANDIDATES]...` | candidates to score; if none and no `--file`, read from stdin | (stdin) |
| `--model <NAME\|PATH>` | model to score under (registered name or `.ngram`) | default model |
| `--file <PATH>` | read candidates from a file, one per line | (none) |
| `--format <table\|tsv\|jsonl>` | output format (`tsv`/`jsonl` for analysis) | `table` |
| `-d`, `--detailed` | add a per-position `Segment Score` column | (off) |
| `--reachable` | report `+inf` for zero-mass candidates instead of the finite floor | (off) |

## File formats

### `.ngram` (NGRMv003)

Binary file. 8-byte magic `NGRMv003`, then a **length-prefixed provenance
blob**, then the model body.

The provenance blob (a `u32` byte-length prefix, then a little-endian record:
schema version, the embedded `tokenizer.json` bytes, the `--tokenizer` and
`--train` paths, the build time, and the tokenov version) makes a v3 model
**self-describing** — `generate --wordlist` and `model info` read the tokenizer
straight from the file, so it survives a bare single-file copy with no sidecar.
Inspect it with `tokenov model info <model.ngram>`.

The body is the Kneser-Ney model: a header (vocab size, START/END sentinels, KN
discount D, decode-table length), then per-context records (trigram cumdists,
lambda, bigram-continuation cumdists, raw and KN-continuation unigram counts).
**It is byte-identical to a v2 body** — the provenance blob is purely additive,
so v3 changes the file, not the model math.

Older formats still load: `NGRMv002` (the KN body with no provenance blob) and
`NGRMv001` (unsmoothed trigram only). For those, the tokenizer is resolved
out-of-band — `$TOKENOV_TOKENIZER`, then a co-located `<model>.tokenizer.json`
sidecar, then the bundled default's tokenizer.

### `<output>.progress` (byte-offset sidecar for strict resume)

Alongside the DFS checkpoint state file, strict mode writes a plain-text
sidecar next to a plain-file `--output`, holding the output byte offset that
pairs with the checkpointed DFS position:

```
emitted=<u64>
byte_offset=<u64>
fingerprint=<args summary>
```

Both are written together at each checkpoint (single-threaded, so the capture
is atomic — the byte offset always matches the saved position's emitted
count). On `--resume`, tokenov restores the DFS position from the checkpoint
file, validates the fingerprint (model file size+mtime, threads, count,
min/max len, mode, bias, etc.), truncates the output to `byte_offset` (handling
any BufWriter spillage past the last write), and continues appending. With
`--no-checkpoint` there's no saved position, so `--resume` instead re-runs the
deterministic stream and discards the first `emitted` items via this sidecar
alone. Both files are removed automatically on clean completion.

## Known limitations / TODO

- KN-aware bias is currently eager (materializes the post-bias model
  for all trigram contexts even if generation only visits a fraction).
  Wall-clock for weighted/combined at small budgets is dominated by
  this setup. A lazy-bias optimization is tracked for a future release.
- Skipgram-expand similarity is structural-role (bigram-distribution
  cosine), not semantic. For semantic similarity you'd want to use a
  language model's own embedding layer or a sentence-transformer model
  — not currently implemented.
- 4-gram and higher models over-fit on training sets <10M passwords;
  trigram is the empirical sweet spot. The `--ngram` flag accepts
  larger values but the resulting model will likely be worse, not
  better.

## License

Apache License 2.0 — see [LICENSE](LICENSE). The bundled default tokenizer
(`tokenov_v1`) is a derivative of OpenAI GPT-2's tokenizer (MIT); see
[`tokenizers/tokenov_v1/ATTRIBUTION.md`](tokenizers/tokenov_v1/ATTRIBUTION.md).
