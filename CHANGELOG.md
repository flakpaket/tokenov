# Changelog

All notable changes to tokenov are recorded here. Versions follow
[semantic versioning](https://semver.org/): MAJOR for breaking changes to the CLI
or model format, MINOR for new opt-in surface with unchanged default output,
PATCH for fixes.

## 1.4.0 — unreleased

A better default for `--wordlist`. **`--wordlist` output changes; add
`--append-only` to get exactly the output of earlier versions.** Output without
`--wordlist`, and `score`, `model`, `tokenizer` and `bootstrap`, are unchanged
from 1.3.0.

### Changed

- **`--wordlist` places each word anywhere in the candidate** ("Anywhere"). The
  model adds characters before the word, after it, or both (`ilovecisco`,
  `#1cisco2024`, `cisco2024!`), and every candidate is ranked by how likely the
  model finds the whole password, across all words, in one pass. Until now the
  word always started the candidate. Details:
  - the characters in front of a word are chosen for how well they lead into
    it, and may run into the word the way the tokenizer splits it
    (`il` + `ove`);
  - a step inside a word that the model has no statistics for is priced from
    how common the token is overall, so far fewer words in a list are skipped;
  - every output line is unique, and `--count` is exact;
  - every `generate` flag works, including `--strict`, `--threads`,
    `--case-shape`, `--enterprise`, `--unigram-tail`, checkpoints and
    `--resume` (a resumed run writes no line twice, and a `--strict` run
    continues byte for byte);
  - tokenov keeps a record of every line it has printed, about 16 bytes per
    candidate, so give long runs a `--count`;
  - with `--min-level`, about 1% of lines can repeat ones a full run printed
    below that level.

  In our tests on leaked passwords of staff at nine large companies, with a
  wordlist scraped from each company's website, on the passwords left after
  running that wordlist through hashcat's dive rules, it cracked about 65% more
  than the previous `--wordlist` at 100 million candidates, and more at every
  company. See [docs/wordlist.md](docs/wordlist.md).
- **`--append-only`** now selects the previous `--wordlist` generator. Its output
  is byte-identical to 1.3.0's `--wordlist` (and `--append-only`).
- **`--prepend-only`** now runs the new generator with the word at the end of
  the candidate. It used to run the separate fixed-additions generator behind
  `--float`, which is gone.
- **`--rare-first`** prints a warning that it will be deprecated in a future
  release. Its output is unchanged.
- **`--min-len` and `--max-len` count bytes everywhere.** The removed
  fixed-additions generator counted characters.
- Checkpoints from earlier versions do not resume in 1.4.0.

### Retired

These are still accepted, print a warning and run something else. See
[docs/retired.md](docs/retired.md).

- **`--float`** runs the new default (with `--rare-first` it is ignored).
- **`--mode seeded`** runs `--append-only`, with the same output.
  **`--mode weighted`** and **`--mode combined`** run the new default; the bias
  they applied to the model is gone, and **`--bias`** has no effect.
- **`--seed-mode`** is still honored by `--append-only` and ignored otherwise.

### Fixed

- `--lazy`, `--force-child-cache` and `--bounded-cap` print a warning when the
  run cannot use them (the default `--wordlist` generator) instead of being
  silently ignored.
- A checkpoint or progress file no longer resumes a run with a different
  `--case-shape` or `--enterprise` setting. Both change the output, and resuming
  across them mixed two different streams in one file.

## 1.3.0 — 2026-11-15

A second way to target a wordlist, and a fix for `--float` on large lists.
**Output without `--rare-first` is unchanged from 1.2.0**, except where 1.2.0
stalled (see "Fixed").

### Added

- **`--rare-first`**, an alternative `--wordlist` generator that gives the most
  candidates to the rarest words in your list. A word's rarity is how unlikely
  the model finds it, and each word's share of the output is proportional to its
  rarity raised to the power `--rarity-weight` (default 1). The model generates
  characters on either side of each word (`ilovecisco`, `cisco2024!`);
  `--append-only` or `--prepend-only` keeps them to one side. Compound words
  also contribute their parts (`johnsmith` → `john`, `smith`; `--no-split`
  turns this off), and `--mix` lets other words from the list be used as
  additions. Every `generate` flag works with it, including `--strict`,
  `--threads` and `--resume`.

  Bare `--wordlist` is still the default and in testing cracked more overall:
  against real leaked corporate passwords, after rockyou with hashcat's best66
  rules, it cracked about three times as many as `--rare-first`. `--rare-first`
  is there for runs where you want more of the budget on target-specific words.
  See [docs/wordlist.md](docs/wordlist.md).
- **`tokenov bootstrap --rebuild`** retrains the default model from the cached
  tokenizer and corpus without downloading anything. Use it to rebuild a model
  made before 1.2.0. If the model already exists and neither `--rebuild` nor
  `--force` is given, `bootstrap` now stops before doing any work; 1.2.0 could
  download the corpus first and then fail.
- **`tokenizer train --vocab-size` and `model train --max-vocab` accept K/M
  suffixes**, like `--count`: `--vocab-size 8K` is 8000. Plain numbers work as
  before.

### Fixed

- **`--float` and `--prepend-only` no longer stall on large wordlists.** With
  more than about 11,000 words (typical for a scraped website), 1.2.0 had
  printed nothing after five minutes. For a long word, nearly every combination
  with the fixed additions was longer than `--max-len`, and each one was built
  and rejected in turn. Such combinations are now skipped before they are
  built: 10 million candidates from an 87,000-word list take about 12 seconds.
  Where 1.2.0 did finish, the output is byte-identical.
- **Builds on Windows in MSYS2 UCRT64.** Resuming a saved session used a
  Unix-only call, so the build failed on Windows. On Windows the resumed run now
  starts as a child process and passes its exit code through. Where memory
  statistics can't be read, tokenov now says so once instead of on every check,
  and the memory limit is not enforced. Linux and WSL2 remain the supported
  platforms; see the README.

### Changed

- **The note for models built before 1.2.0 prints the exact command to rebuild
  the model**, and so does `model info`. The note's wording is also clearer.
- `--append-only`, `--prepend-only` and `--float` without `--wordlist` now print
  a warning that they are ignored. 1.2.0 ignored them silently.
- When a model name isn't found, the error now points to `tokenov model` for
  the list of registered models.

### Documentation

- The README is cut down to getting started, key concepts and upgrading. The
  detail moved to one page per topic under `docs/`, including
  `docs/retired.md` for deprecated flags and commands.
- `--help` text is rewritten in plain language. `--enterprise`, `--output` and
  `bootstrap --name` now describe what they actually do.
- `docs/unigram-tail.md` is rewritten from new measurements at 1 and 10 billion
  candidates, with and without rules. It now says when the tail is worth using
  and shows how to build a small tail-only list: the tail candidates the default
  stream did not already produce, cracked after it. The earlier figures came
  from a comparison that did not account for overlap between the two streams,
  and are replaced.

## 1.2.0 — 2026-11-01

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

1.1.0 versus this release, six 1e9 streams on default arguments, cracked against
rockyou (99,998 hashes) and enterprise_union (83,867) at 1e7/1e8/1e9.

Default output was expected to change, so the test is that it changes **only**
where the decode fix says it should:

| configuration | pure-ASCII candidates | lines only in 1.1.0 |
|---|---|---|
| `--strict` | 1.1.0's are an exact, in-order prefix of 1.2.0's | 22,242, every one containing `�` |
| `--threads 1` | none missing; 2,351 extra in 1.2.0 | 22,242, every one containing `�` |
| `--threads 8` | none missing; 2,332 extra in 1.2.0 | 21,981, every one containing `�` |

The extra ASCII candidates are the next lines of the stream. 1.2.0 emits about
2,350 fewer junk lines in its first billion, so it reaches slightly further. In
1.2.0's stream, 1,228 of the billion lines are not valid UTF-8, and 70 still
contain `�`; the latter come from vocabulary tokens that really are that
character.

| depth | rockyou 1.1.0 → 1.2.0 | enterprise_union 1.1.0 → 1.2.0 |
|---:|---:|---:|
| 1e7 | 26,898 → 26,898 | 1,954 → 1,954 |
| 1e8 | 54,787 → 54,787 | 5,956 → 5,956 |
| 1e9 | 72,619 → **72,621** | 11,404 → 11,404 |

(`--strict` and `--threads 1`; `--threads 8` gains the same 2 at 1e9.) The two
new cracks are `ESPAÑOL` and `Maturìn`. 1.1.0 could not produce them because it
rendered `Ñ` / `ì` as `�`. Resume was tested separately: a SIGKILLed fast-mode
run resumes without truncation and reproduces the uninterrupted candidate set;
strict resume is byte-identical; a 1.1.0 checkpoint is refused.

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
