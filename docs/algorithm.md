# Algorithm sketch

How the model is built and how `generate` walks it. For the flags that drive
this, see [generate.md](generate.md); for the file layout, [file-formats.md](file-formats.md).

## Model

A Kneser-Ney–smoothed n-gram (default trigram) over token IDs
produced by a HuggingFace tokenizer. Each context `(a, b)`'s trigram
distribution sums to `(1 − λ(a,b))`; the remaining `λ(a,b)` mass flows
to a bigram-continuation distribution `P_cont(t | b)` shared across all
contexts ending in `b`. Special tokens (`START`, `END`) are integers
above the tokenizer vocab size. With `--unigram-tail`, a third set of
entries — the globally most frequent tokens — is appended to every
context's child list, discounted by the tail weight, so they compete for
emission alongside the context tiers rather than backing them off.

## Enumeration

An outer loop sweeps `target_level = 0, 1, 2, …`; an
inner depth-first walk of the token tree visits all candidates at exactly that level and prunes
branches whose accumulated `−log_prob` exceeds it. All level-`L`
candidates emit before any level-`(L+1)` candidate, so output is
approximately rank-ordered. (The relaxation from "exactly rank-ordered"
to "approximately" comes from discretizing log-prob to integer levels
via `ceil(-lp)` — within a single level, output follows the tree walk rather
than strict rank.)

Each candidate belongs to exactly one level, which is what lets `--count` act
as a ceiling on the sweep and `--min-level` as the matching floor — see
[Starting deeper with `--min-level`](generate.md#starting-deeper-with---min-level).

## Parallelization

Domain decomposition by first-level token. In the
default **fast** mode each thread level-sweeps a disjoint first-level
subtree and appends its emissions directly to a shared sink in batches (a
brief lock each time a thread's 64 KiB buffer fills), N-way interleaved. Because
the partitions interleave, the global order is *approximate*: each
partition is internally rank-ordered, but items from different partitions
land in scheduling order (a few thousand items out of place globally).

## Exact order (`--strict`)

`--strict` trades that parallelism for an exact global order: a single
producer streams the full rank order straight to the sink, giving a
byte-reproducible, model-canonical stream. There is one ordered output, so
strict runs single-threaded and `--threads` is ignored. No temp files either
way — output streams live to stdout or a file.

## Wordlist targeting

`--wordlist` is described in [wordlist.md](wordlist.md).

## Known limitations

- 4-gram and higher models over-fit on training sets <10M passwords;
  trigram is the empirical sweet spot. The `--ngram` flag accepts
  larger values but the resulting model will likely be worse, not
  better.

