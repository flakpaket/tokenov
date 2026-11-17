# tokenov

Multithreaded token-level n-gram Markov password candidate generator. Trains an
n-gram model over passwords tokenized with any HuggingFace tokenizer and outputs
candidate passwords roughly from most to least likely.

The headline use case is password cracking: feed the candidate file to
`hashcat -a 0` or any wordlist-mode cracker. The token primitive (rather
than ASCII characters) is what makes the output structurally diverse —
multi-class strings like `Stamford12`, `Bella-boo`, `KOOL-AID5` appear in
the head of the distribution rather than as low-probability tail events.

## Quick start — Install & Bootstrap

Tokenov is available on crates.io and can be installed with `cargo`. To install
Rust and Cargo, check out https://rust-lang.org/tools/install/. Once you have
Rust and Cargo, simply run:

```bash
cargo install tokenov
tokenov bootstrap
```

`bootstrap` uses the tokenizer bundled in the binary, downloads RockYou, trains
a model on it and registers it as the default (see [docs/models.md](docs/models.md)).
After that you're ready to generate candidates:

```bash
tokenov generate --count 10   # Test run: generate 10 candidates
tokenov | hashcat -a 0 ...    # Pipe candidates to Hashcat stdin
```

A few more starting points:

```bash
# Target an organization: every candidate contains one of your words
tokenov generate --wordlist osint_words.txt --count 100K

# A byte-reproducible stream in exact rank order, written to a file
tokenov generate --strict --count 1B --output cands.txt

# How guessable is a password under the model?
tokenov score password 'Tr0ub4dour&3'
```

## Build

```bash
git clone https://github.com/flakpaket/tokenov
cd tokenov
cargo build --release
# binary lands at target/release/tokenov
```

Requires Rust ≥ 1.85 (2024 edition).

**Platforms.** tokenov is supported on Linux, and on Windows through WSL2
(install Rust inside WSL and build as above). It can also be built natively on
Windows in an MSYS2 UCRT64 shell, but that isn't supported or regularly tested,
and the memory limit (`--max-rss-gb`) is not enforced there.

## Subcommands

The CLI is grouped into noun-commands. `tokenov --help` lists them; each group
(`tokenov model`, `tokenov tokenizer`) lists its verbs.

| command | what it does | docs |
|---|---|---|
| `tokenov bootstrap` | quickstart: download RockYou and train the default model with the bundled tokenizer | [models](docs/models.md) |
| `tokenov tokenizer` | list tokenizers + download status (bare); `get`/`add`/`delete`/`train`/`set-default` verbs | [tokenizers](docs/tokenizers.md) |
| `tokenov model` | list registered models (bare); `train`/`register`/`delete`/`info`/`verify` verbs | [models](docs/models.md) |
| `tokenov generate` | emit candidates (also the default if no subcommand is given) | [generate](docs/generate.md) |
| `tokenov score` | report how guessable each password is under the model (Rarity, in bits; higher = harder) and when `generate` would produce it | [score](docs/score.md) |

Every subcommand is quiet on stderr by default; `-v` shows progress. stdout
carries only candidates (or a command's results), so it is always safe to pipe.

## Key concepts

### Levels, and why the order is approximate

The model gives every candidate a probability. `generate` turns probabilities
into integer **levels** (`ceil(-ln P)`, summed per token) and sweeps them in
order — level 0, then 1, then 2 — so the most likely candidates come out first.
Within a level the order is a tree walk, not a strict ranking, which is why the
stream is *approximately* rank-ordered. `--count` stops the sweep;
`--min-level` starts it deeper.

By default the sweep runs on every core, each thread taking a disjoint slice of
the tree and writing straight to the output. The threads' output interleaves in
scheduling order, so the order differs between runs. With no `--count`, fast
mode produces exactly the same candidates as `--strict`, in a different order.
With `--count N`, each thread stops after its share, so the N candidates you get
differ slightly from `--strict`'s first N. `--strict` runs a single producer
instead: an exact rank order, byte-identical on every run, at single-thread
speed. See [docs/algorithm.md](docs/algorithm.md).

### Wordlist targeting builds on your words

`--wordlist FILE` takes a list of target-specific words (names, product names,
brands, words scraped from the target's website) and builds every candidate
around one of them intact, with characters the model adds before it, after it,
or both: `cisco1`, `ilovecisco`, `#1cisco2024`. Candidates come out likeliest
first across the whole list. `--append-only` and `--prepend-only` keep the
additions to one side. See [docs/wordlist.md](docs/wordlist.md).

### Models carry their tokenizer

A model is a single `.ngram` file that embeds the exact tokenizer it was trained
with, so it can be copied to another machine on its own and still work —
including for `--wordlist`, which must tokenize your words the same way. See
[docs/models.md](docs/models.md) and [docs/file-formats.md](docs/file-formats.md).

## What's new

**Tokenov 1.4.0:**

- **Anywhere:** `--wordlist` now places each word anywhere in the candidate.
  The model adds characters before the word, after it, or both (`ilovecisco`,
  `#1cisco2024`), and every candidate is ranked by how likely the model finds
  the whole password. In our tests it cracked about 65% more than the previous
  `--wordlist` at 100 million candidates, on leaked passwords that were mostly
  for personal accounts rather than company systems, so expect different gains
  on other data (details in [docs/wordlist.md](docs/wordlist.md)).
  **`--wordlist` output changes;**
  `--append-only` gives the old output. See [Upgrading](#upgrading).
- `--prepend-only` now uses the same generator, with the word at the end.
- Older ways of using a wordlist are retired. Old command lines still run,
  with a warning; [docs/retired.md](docs/retired.md) says what runs instead.
  `--rare-first` still works and will be deprecated in a future release.

**Tokenov 1.3.0:**

- `--rare-first`, a second way to target a wordlist that puts most of the
  candidates on the rarest words in your list.
- Wordlist runs that add characters in front of words no longer stall on large
  wordlists, such as a scraped website.
- `tokenov bootstrap --rebuild` retrains the default model without downloading
  anything again.
- New documentation: this README now covers getting started, and each topic
  has its own page under [docs/](docs/) (see [Documentation](#documentation)).

**Tokenov 1.2.0:**

- Candidates with non-ASCII characters (accented letters, other alphabets) come
  out exactly right instead of containing `�`.
- In wordlist runs that add characters in front of words, `--count` is a hard
  limit, and `--enterprise` and `--case-shape` work.

**Tokenov 1.1.0:**

- `--unigram-tail` reaches candidates the model's context statistics alone
  cannot ([docs/unigram-tail.md](docs/unigram-tail.md)).
- `--min-level` starts generating further down the list, skipping the most
  likely candidates.
- `score` reports the level at which `generate` produces each password, the
  value to use with `--min-level`.

Full details are in [CHANGELOG.md](CHANGELOG.md).

## Upgrading

**From 1.3.0:** plain `generate` (no `--wordlist`), `score`, `model`,
`tokenizer` and `bootstrap` give the same output as 1.3.0. **`--wordlist`
output changes:** it now adds characters before words as well as after
them, in a different order. To get exactly the old `--wordlist` output, add
`--append-only`. `--prepend-only` output also changes. Options retired in
1.4.0 still run, with a warning, and are listed in
[docs/retired.md](docs/retired.md).

To keep using 1.3.0 instead, install it alongside or in place of 1.4.0:

```bash
cargo install tokenov --version 1.3.0 --locked --force
# or build it from source:
git clone https://github.com/flakpaket/tokenov && cd tokenov
git checkout v1.3.0 && cargo build --release
```

**From 1.2.0:** everything under "From 1.3.0" applies. Plain `generate` output is
the same as 1.2.0. See [CHANGELOG.md](CHANGELOG.md).

**From 1.1.0:** also read the 1.2.0 entry in [CHANGELOG.md](CHANGELOG.md).
Candidates containing non-ASCII characters changed, and output is no longer
guaranteed to be valid UTF-8.

**Models built before 1.2.0** print a note on every run until they are
rebuilt. For the default model, run `tokenov bootstrap --rebuild`. For any
other model, the note prints the exact command. See
[docs/models.md](docs/models.md#rebuilding-a-model-after-upgrading).

**Checkpoints** only resume with the tokenov version that wrote them, so finish
or restart interrupted runs before upgrading.

## Documentation

| page | covers |
|---|---|
| [generate](docs/generate.md) | standard mode, output format, `--min-level`, `--case-shape`, `--enterprise` |
| [wordlist](docs/wordlist.md) | `--wordlist` targeting: the default, `--append-only`, `--prepend-only`, `--rare-first` |
| [resume](docs/resume.md) | checkpoints, `--resume`, sessions, strict-mode resume |
| [unigram-tail](docs/unigram-tail.md) | `--unigram-tail`: what it changes and when it pays off |
| [score](docs/score.md) | `tokenov score`: Rarity, levels, output formats |
| [models](docs/models.md) | `bootstrap`, `model train`, `model info`, the model registry |
| [tokenizers](docs/tokenizers.md) | the tokenizer manifest, bringing or training your own, the bundled default |
| [remote-generation](docs/remote-generation.md) | generating on one host and cracking on another; RAM and cache tuning |
| [algorithm](docs/algorithm.md) | the model, the level sweep, parallelism, known limitations |
| [cli-reference](docs/cli-reference.md) | the main `generate` and `score` flags |
| [file-formats](docs/file-formats.md) | `.ngram` and the strict-resume `.progress` sidecar |
| [retired](docs/retired.md) | deprecated flags and commands that are still accepted |
| [FAQ](docs/FAQ.md) | why lengths count bytes; how `--enterprise` classifies characters |

Release history is in [CHANGELOG.md](CHANGELOG.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE). The bundled default tokenizer
(`tokenov_v1`) is a derivative of OpenAI GPT-2's tokenizer (MIT); see
[`tokenizers/tokenov_v1/ATTRIBUTION.md`](tokenizers/tokenov_v1/ATTRIBUTION.md).
