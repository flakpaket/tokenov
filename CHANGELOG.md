# Changelog

All notable changes to tokenov are recorded here. Versions follow
[semantic versioning](https://semver.org/): MAJOR for breaking changes to the CLI
or model format, MINOR for new opt-in surface with unchanged default output,
PATCH for fixes.

## 1.2.0 — unreleased

Fixes non-ASCII candidates and three `--wordlist` defects. **Default output
changes, but only on lines that were already broken.** These are candidates
containing a token that 1.1.0 rendered as U+FFFD (`�`); every pure-ASCII
candidate is unchanged (see "Release gate").

### Fixed

- **Non-ASCII candidates are byte-exact.** The model's decode table was built
  one token at a time. A byte-level BPE token holding part of a multi-byte
  character is not valid UTF-8 on its own, so it was stored as U+FFFD. That is
  352 of 50,257 entries in the default model. As a result:
  - `--float` / `--prepend-only` emitted `�` in place of the seed's real
    characters, e.g. `ûtԱCXvvIJYZYT` came back as `ût��CXvvIJYZYT`.
  - Standard mode emitted junk such as `��`.

  The table now stores each token's raw bytes. Existing models do not need
  retraining: a model built before 1.2.0 is repaired in memory at load from its
  embedded tokenizer, and tokenov prints a one-line note. `model info` reports
  which kind of table a model has. Models remain `NGRMv003`; the provenance
  blob gains an additive `decode_exact` flag (schema 2). Older releases ignore
  the flag, still load 1.2.0 models, and read the corrected table as stored.
- **`--count` is a ceiling under `--float` / `--prepend-only`.** Each seed and
  sub-chunk was given at least one candidate and nothing checked the total, so
  a 200-entry wordlist with `-c 10` emitted 200. Budgets are now apportioned
  exactly and the total is enforced.
- **Graft seed splitting keeps words, not fragments.** Sub-chunks were admitted
  for being rare. The rarest pieces are mid-word BPE splits, so `cisco` gave
  `isco` and `summer` gave `sum` / `mer`, while `northwind` lost `wind`. A seed
  is now split only when every piece is at least 3 characters, and pieces of 4+
  characters become anchors (`johnsmith` → `john`, `smith`).
- **`--enterprise` and `--case-shape` work under `--float` / `--prepend-only`.**
  They were accepted and ignored.

### Changed

- **Graft says which flags it ignores.** `--strict`, `--threads`,
  `--max-tokens`, `--unigram-tail`, `--resume`, `--resume-state` and
  `--checkpoint-file` have never had any effect with `--float` /
  `--prepend-only`, and were accepted silently. They now print a warning naming
  the ignored flags. Output is byte-identical to running without them, so
  existing scripts keep working. `--min-level` remains an error there, as in
  1.1.0.
- **`--mode` with `--float` / `--prepend-only` is an error.** It previously
  discarded `--float` / `--prepend-only` without saying so and ran a different
  generator.
- **Output is no longer guaranteed valid UTF-8.** Candidates are the tokens'
  exact bytes. 1.1.0's output was always valid UTF-8 because every partial
  character had been replaced with `�`. Read output as bytes.

### Documentation

- README explains that the default / `--append-only` path and `--float` /
  `--prepend-only` are two different generators, and what each supports.
- README documents the decode-table repair and the `decode_exact` flag.

### Release gate

_Pending._

## 1.1.0 — 2026-09-02

Two new generation controls and four fixes. **Default output is unchanged from
1.0.0** — verified byte-for-byte at one billion candidates (see "Release gate").

### Added

- **`--unigram-tail [FRACTION]`** — also consider the most frequent tokens in the
  corpus at every step, not just tokens seen in the current context. Lets the
  generator reach candidates its context statistics alone cannot. The optional
  `FRACTION` is the share of the bigram tier's missing-mass budget the tail
  receives; a bare flag uses 0.1. Replaces `--variant freq-tail`, which still
  works and now warns.
- **`--min-level <N>`** — start the enumeration sweep at level N, skipping the
  more-probable levels below it. `--count` already expressed a ceiling on depth;
  this is the missing floor. A `--min-level N` stream is an exact suffix of the
  full stream. Errors rather than silently doing nothing when N exceeds the
  maximum level, or when combined with the graft generator (`--float` /
  `--prepend-only`), which has no level sweep.
- **`score` now reports the enumeration level** of each candidate, so the level
  to hand `--min-level` can be measured rather than guessed.

### Fixed

- `--resume` no longer truncates the output file it is continuing, and a resumed
  run whose stream exhausts before `--count` now terminates instead of hanging.
- Every run gets its own checkpoint and session record; a second run can no
  longer clobber the first one's resume position.
- The fast-mode worker flush defaults to 64 KiB.
- `--merge-chunk-size`, `--no-auto-tune` and `--runtime-tune` now say plainly
  that they have no effect on `generate` (they apply to `calibrate` only). The
  documentation no longer advertises a merger that `generate` does not run.
- The unigram tail weight is part of the checkpoint fingerprint, so a run
  checkpointed at one weight can no longer resume at another. Runs without the
  flag keep a fingerprint identical to 1.0.0 and stay resumable across the
  upgrade.

### Documentation

- `docs/FAQ.md` — why the length bounds count bytes, and why the enterprise
  policy checks five character categories rather than the usual four.
- README covers `--unigram-tail` and `--min-level`, including what the tail is
  worth with and without a rule pipeline.

### Release gate

1.0.0 versus this release, six 1e9 streams on default arguments, cracked against
rockyou (99,998 hashes) and enterprise_union (83,867) at 1e7/1e8/1e9:

| configuration | result |
|---|---|
| `--strict` | byte-identical |
| `--threads 1` | candidate set identical |
| `--threads 8` | candidate set identical |

Crack counts match exactly at full depth in every configuration.
