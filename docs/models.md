# Models

Training, registering and inspecting `.ngram` models. The on-disk format is in [file-formats.md](file-formats.md).

## `tokenov bootstrap`

Zero-input quickstart: use the **bundled default tokenizer** (`tokenov_v1`,
embedded in the binary — no download), download RockYou and repeat each password by its count, then
`model train` + register. Prints a plan (with copy-pasteable manual commands) and
proceeds.

```bash
tokenov bootstrap --dry-run    # preview only; no network or writes
tokenov bootstrap              # → registers the default model `tokenov_v1`
tokenov bootstrap --rebuild    # retrain the existing model from the cached corpus
tokenov bootstrap --force      # re-download and re-expand RockYou, then retrain
```

The tokenizer and the expanded corpus are cached, so a second `bootstrap` needs
no download. Once the model exists, a plain `bootstrap` stops before doing
anything and asks for one of the two flags. `--rebuild` retrains from the cache
and downloads only what is missing. `--force` throws the cache away and starts
over.

> The bootstrap model trains on the **full** RockYou (no train/test split) — a
> get-running convenience, not for measuring crack rates.

## `tokenov model train`

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
bare single-file copy. A model file without an embedded tokenizer still loads;
tokenov finds its tokenizer separately (see
[file-formats.md](file-formats.md)).

```bash
tokenov model train \
    --tokenizer tokenizer.json \
    --train train.txt \
    --output model.ngram \
    --ngram 3
```

Inspect a model's embedded provenance:

```bash
tokenov model info model.ngram      # tokenizer, train corpus, build time, version, decode-table status
```

## Rebuilding a model after upgrading

A model built before 1.2.0 stores a few tokens wrongly: any token holding part of
a multi-byte character was saved as `�`. tokenov repairs this in memory every
time it loads such a model, so output is correct, but it prints a note on every
run until the model is retrained. Retraining the default model on the full
RockYou takes about a minute and a half.

**The default model, or any model built by `bootstrap`:**

```bash
tokenov bootstrap --rebuild
```

**Any other model:** the note, and `tokenov model info <model>`, print the exact
`model train` command that rebuilds it in place from the tokenizer and corpus
recorded in the model. It looks like this:

```bash
tokenov model train --tokenizer ~/.local/share/tokenov/tokenizers/tokenov_v1/tokenizer.json \
    --train ~/.local/share/tokenov/bootstrap/rockyou_withcount_expanded.txt \
    --name tokenov_v1 --force
```

The command keeps the model's file location and registry name (`--name` for a
model in the default store, `--output` for one elsewhere), and adds
`--max-vocab` if the model was capped. If the recorded corpus or tokenizer file
no longer exists, tokenov says so instead of printing a command that would fail.
Rebuild with the corpus you want using `--train`.

Retraining from the same inputs changes only the stored table. On the default
model, a 10M-candidate `--strict` stream is byte-identical before and after.

Every `model train` registers the model in `~/.config/tokenov/models.toml`, so
`tokenov model` lists it and `generate --model <name>` resolves it by name. With
`--output` the file stays where you put it; omit `--output` and it lands in the
default store (`$XDG_DATA_HOME/tokenov/models/<name>.ngram`, else
`~/.local/share/tokenov/models/<name>.ngram`).

Build times for typical configs on a 14.2M-password sample of RockYou:

| tokenizer | vocab | model size | build time |
|---|---:|---:|---:|
| GPT-2 (`gpt2`) | 50,257 | ~250 MB | ~30s |
| Mistral (`mistralai/Mistral-7B-v0.1`) | 32,000 | ~200 MB | ~25s |

