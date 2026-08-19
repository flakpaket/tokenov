# Tokenov tokenizer v1 — attribution & provenance

`tokenizer.json` in this directory is the **default tokenizer bundled with tokenov**
(`default_alias = "tokenov_v1"`), embedded into the binary via `include_bytes!`
(`src/bootstrap.rs`) so a fresh install resolves it with zero network.

## What it is

A **derivative of OpenAI GPT-2**:

- **Base (unchanged):** the GPT-2 tokenizer from
  [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) — its
  50,257-token byte-level BPE vocabulary and merge table. GPT-2 was released by
  OpenAI under the **MIT License**.
- **The only modification:** a `Split(\p{N}{1,2})` pre-tokenizer is **prepended** so
  digit runs are tokenized in ≤2-digit groups — e.g. `2007` → `20|07`,
  `whales2007` → `wh|ales|20|07`. The alphabetic vocabulary, merges, and all other
  tokenization are GPT-2's, untouched (verified: 0/6 pure-alpha tokenizations change
  vs base; 2000/2000 byte-exact decode round-trip; all 100 `"00".."99"` are single
  tokens).

## Why this tokenizer

It benchmarked as the strongest license-clean token alphabet for the tokenov
n-gram cracker: at a 1e9-candidate budget it beats stock LLaMA-3 by **+3.22 pts**
(mean over 16 corpora, no rules) and edges an equivalent LLaMA-3 digit variant by
+0.46. The `\p{N}{1,2}` digit granularity is the lever — it lets the trigram model
compose year/date suffixes (`whales|20|07`) instead of diluting them across single
digits.

## How to regenerate

Take the upstream `openai-community/gpt2` `tokenizer.json` and prepend a
`Split` pre-tokenizer on the pattern `\p{N}{1,2}` (invert off, `isolated`
behavior), keeping the existing `ByteLevel` pre-tokenizer after it. Everything
else — vocab, merges, decoder, post-processor — is copied through unchanged.

## License note

GPT-2 and its tokenizer are MIT-licensed by OpenAI; redistributing this derivative
is permitted under MIT (retain the MIT notice). tokenov's own code is Apache-2.0
(see `LICENSE`). This file records attribution for the bundled tokenizer; it is
not itself the license grant.
