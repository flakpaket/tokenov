# Changelog

All notable changes to tokenov are recorded here. Versions follow
[semantic versioning](https://semver.org/): MAJOR for breaking changes to the CLI
or model format, MINOR for new opt-in surface with unchanged default output,
PATCH for fixes.

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
