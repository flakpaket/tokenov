# Deprecated and retired flags

tokenov keeps old command lines running. A deprecated flag or command is still
accepted, so a script written for an older version runs instead of failing on
an unknown argument. Most print a warning on stderr, as noted below. None of
these appear in `--help`.

## Old spellings (still work)

Each prints a deprecation notice except `--list-models`, which runs silently.

| old | use instead | notes |
|---|---|---|
| `tokenov build` | `tokenov model train` | |
| `tokenov fetch` | `tokenov tokenizer get` | |
| `tokenov delete` | `tokenov model delete` | |
| `tokenov register` | `tokenov model register` | |
| `--list-models` | `tokenov model` | |
| `--variant freq-tail` | `--unigram-tail` | `--unigram-tail` also takes an optional weight; see [unigram-tail.md](unigram-tail.md) |

## Accepted and ignored

| flag | what happens |
|---|---|
| `--skipgram-expand`, `--skipgram-direction` | ignored; `--skipgram-expand` prints a warning. To widen a wordlist, use separate tooling and pass the result to `--wordlist`. |
| `--merge-chunk-size`, `--no-auto-tune`, `--runtime-tune`, `--retune` | ignored with a warning. They tune an internal component that `generate` does not use. |
| `--no-runtime-tune` | ignored; prints a note only under `-v`. |
| `--fast` | ignored: fast mode is already the default. It is an error together with `--strict`. |
