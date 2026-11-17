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
| `--bias` | ignored with a warning. It set the strength of the retired `--mode weighted` / `combined`. |

## Retired wordlist generators (still accepted)

tokenov 1.4.0 made placing each word anywhere the default `--wordlist`
generator ([wordlist.md](wordlist.md)). These older ways of using a wordlist
print a warning and run the generator named here instead.

| flag | what runs instead |
|---|---|
| `--float` | the default `--wordlist` generator, which also adds characters on either side of each word. The old `--float` joined words to a fixed set of 250 common additions, rarest words first. With `--rare-first`, `--float` is ignored: adding on either side is already `--rare-first`'s default. |
| `--mode seeded` | `--append-only`: the same generator, with the same output. |
| `--mode weighted`, `--mode combined` | the default `--wordlist` generator. They tilted the whole model toward the wordlist's tokens; that bias is gone. With `--append-only`, `--append-only` runs. Without `--wordlist`, `--mode` has no effect. |
| `--seed-mode token` | still honored by `--append-only` (candidates start with each distinct token of the list instead of each word), so its output matches tokenov 1.3.0. Ignored otherwise. |
