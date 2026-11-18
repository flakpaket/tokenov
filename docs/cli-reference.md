# CLI reference

Flag tables for `generate` and `score`. The explanations live on the topic pages:
[generate.md](generate.md), [wordlist.md](wordlist.md), [resume.md](resume.md),
[unigram-tail.md](unigram-tail.md), [remote-generation.md](remote-generation.md)
(child-cache and RAM flags), and [score.md](score.md). For `model` and
`tokenizer` verbs see [models.md](models.md) and [tokenizers.md](tokenizers.md),
or run `tokenov <command> --help`.

By default every subcommand is **quiet on stderr** — only warnings and errors
print; stdout still carries candidates (or a command's results). Pass **`-v` /
`--verbose`** (global — works before or after the subcommand) to see model-load,
tuning, and progress detail, plus per-tick `[stats]` telemetry. Wrapper tools can
use `--json` to get the machine telemetry stream on stderr without `-v`.

Options on `tokenov generate` (also used when no subcommand is given). This
table lists the main flags; `tokenov generate --help` lists all of them.

| flag | meaning | default |
|---|---|---|
| `-v`, `--verbose` | verbose stderr (model-load / tuning / progress / `[stats]`); global | quiet |
| `--json` | emit the machine telemetry stream on stderr without `-v` | (off) |
| `--stats-interval-ms <MS>` | telemetry tick interval; 0 turns telemetry off | 500 |
| `--stats-csv <FILE>` | write one telemetry row per tick (time, emitted, rate, bytes) to this CSV | `<output>.stats.csv` with `--output`, else none |
| `--progress-secs <SEC>` | seconds between stderr `[stats]` lines when not `-v` | 60 |
| `--strict` | exact global rank order (single producer, single-threaded, byte-reproducible) | (off → fast mode) |
| `--model <PATH>` | model file or registered name (`tokenov model train`) | (default model) |
| `--count <N>` | stop after N candidates (accepts K/M/B/T suffixes, e.g. `1B`) | unlimited |
| `--output <PATH>` | output file: plain text, one candidate per line (usually but not always valid UTF-8); compress externally if you want | stdout |
| `--threads <N>` | generation thread count (ignored under `--strict`, which is always single-threaded) | number of CPU cores |
| `--min-len <N>` | minimum candidate length in bytes | 4 |
| `--max-len <N>` | maximum candidate length in bytes | 30 |
| `--max-tokens <N>` | maximum tokens per candidate | 12 |
| `--min-tokens <N>` | minimum tokens per candidate; shorter candidates are skipped | 1 |
| `--unigram-tail [FRACTION]` | also consider globally-frequent tokens at every step; FRACTION sets how strongly | (off; bare flag = 0.1) |
| `--min-level <N>` | start the level sweep at level N, skipping the more-probable levels below it (0 = no-op; with `--wordlist` see [wordlist.md](wordlist.md)) | 0 |
| `--case-shape <SPEC>` | re-case each token (per-slot `?l`/`?c`/`?u`, or `lower`/`cap1`/`title`/`upper`; `;`-separated) | (off) |
| `--enterprise` | emit only policy-compliant candidates (≥8 chars + ≥3 of 5 classes; capitalize-first repair) | (off) |
| `--wordlist <PATH>` | target wordlist (one entry per line); the model adds characters before, after or on both sides of each word, likeliest first across all words | (none → standard mode) |
| `--append-only` | with `--wordlist`: add characters after each word only; the word starts the candidate | (off) |
| `--append-only-legacy` | with `--wordlist`: add characters after each word only, as tokenov 1.3 did; reproduces 1.3-era `--wordlist` results (the output of `--wordlist` in 1.3 and `--append-only` in 1.4.0). Not with `--rare-first` | (off) |
| `--prepend-only` | with `--wordlist`: add characters before each word only; the word ends the candidate | (off) |
| `--rare-first` | rarest words get the most candidates; the model adds characters on either side (one side with `--append-only` / `--prepend-only`); to be deprecated | (off) |
| `--rarity-weight <P>` | with `--rare-first`: each word's share ∝ rarity^P (0 = equal shares) | 1.0 |
| `--mix` | with `--rare-first`: let other words from the list be used as additions (`johncisco`) | (off) |
| `--no-split` | with `--rare-first`: don't use the parts of compound words (`johnsmith` → `john`, `smith`) as words | (off) |
| `--resume` | continue the previous run where it stopped (strict mode needs `--output`); see [resume.md](resume.md) | (off) |
| `--checkpoint-file <FILE>` | checkpoint to a named file instead of this run's own | (per-run file) |
| `--no-checkpoint` | don't write a checkpoint | (off) |
| `--checkpoint-secs <SEC>` | checkpoint cadence + resume safety margin | 300 |
| `--resume-state <FILE>` | resume a specific checkpoint file (explicit form of `--resume`) | (none) |
| `--max-rss-gb <GiB>` | abort generation if the process RSS exceeds this (safety cap) | (auto: ~75% of RAM) |
| `--mem-pressure-threshold <FRAC>` | abort when free system memory falls below this fraction of total; 0 disables | 0.10 |
| `--mem-sample-ms <MS>` | how often the memory checks run | 3000 |
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

