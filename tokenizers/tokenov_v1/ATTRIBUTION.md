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

Phase-18 of the password_research project benchmarked it as the strongest
**non-Chinese, license-clean** token alphabet for the tokenov n-gram cracker: at a
1e9-candidate budget it beats stock LLaMA-3 by **+3.22 pts** (mean over 16 corpora,
raw) and edges the `llama3_d12` reference by +0.46. The `\p{N}{1,2}` digit
granularity is the lever — it lets the trigram model compose year/date suffixes
(`whales|20|07`) instead of diluting them across single digits. Full writeup:
`password_research/docs/findings/phase18_license_clean_default_tokenizer.md`.

## How to regenerate

```
uv run password_research/scripts/build_digit_variant.py gpt2 --prepend
# -> outputs/vocabularies/hf_tokenizers/gpt2_d12/tokenizer.json  (copied here)
```

## License note

GPT-2 and its tokenizer are MIT-licensed by OpenAI; redistributing this derivative
is permitted under MIT (retain the MIT notice). **The formal license text / SPDX
sidecar for this bundled file, and tokenov's own code-license choice, are being
finalized separately** — see `password_research/issues/issue-015-*`. This file
records attribution; it is not itself the license grant.
