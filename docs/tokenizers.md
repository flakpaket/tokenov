# Tokenizers (`tokenov tokenizer`)

Manage tokenizers. tokenov has a built-in list of named tokenizers (aliases)
that download without a HuggingFace login. Bare `tokenov tokenizer` lists every
alias with its download status and origin.

```bash
tokenov tokenizer                       # list aliases + download status
tokenov tokenizer train --corpus rockyou.txt --vocab-size 4K --output my.json  # train your own
tokenov tokenizer get mistral7b         # download one (or --all)
tokenov tokenizer get gpt2 --dest .     # download into ./gpt2/tokenizer.json
tokenov tokenizer add mybert <url|file> "note"   # add your own alias
tokenov tokenizer delete mybert         # remove a download (+ user-added entry)
tokenov tokenizer set-default gpt2      # change the default tokenizer/model
```

**Bring your own.** tokenov also consumes any HuggingFace `tokenizer.json`
directly — point `--tokenizer` at the file. Two shapes to know about: WordPiece
tokenizers (BERT) emit `##`-prefixed subwords that show up in decoded output (you
may want to filter the vocab before training), and SentencePiece tokenizers (e.g.
Mistral) prepend `▁` (U+2581) to word-initial tokens, which tokenov decodes to
spaces. A tokenizer trained on non-English text produces candidates in that
script.

**Train your own tokenizer** (`tokenizer train`) fits a byte-level **BPE** over a
password corpus and writes a standard `tokenizer.json` — closing the loop so you can
go corpus → tokenizer → model → candidates entirely in tokenov:

```bash
tokenov tokenizer train --corpus rockyou.txt --vocab-size 4K --output my.json
tokenov model train --tokenizer my.json --train rockyou.txt --name my_model
tokenov generate --model my_model --count 1M | hashcat -a 0 -m 0 hashes.txt
```

`--vocab-size` (default 8000; accepts K suffixes like `--count`, so `8K`) is the knob that matters most: smaller vocabularies
(~1k–8k) often crack password data better than the 30k+ typical of web-text
tokenizers, and the sweet spot grows with the amount of unique training data — sweep
it for your corpus. The output matches the pipeline's tokenizer shape (BPE + ByteLevel
pre-tokenizer, no special tokens), so `model train` consumes it unmodified.

`set-default <alias>` sets `default_alias` in the manifest — the tokenizer
`bootstrap` uses **and** the model name `generate` resolves when you omit
`--model`. It is `tokenov_v1` unless you change it.

The manifest is seeded from a built-in default into
`~/.config/tokenov/tokenizers.toml` on first use; once it exists it is
authoritative, so `add`/`delete` persist. (A few WordPiece tokenizers — e.g.
DeepPavlov RuBERT — ship only `vocab.txt` and have no fetchable
`tokenizer.json`; those aren't in the manifest.)

The default alias **`tokenov_v1`** is special: its manifest source is
`bundled:tokenov_v1`, so it is **embedded in the binary** and resolves with no
network. It is *Tokenov tokenizer v1* — OpenAI GPT-2's vocabulary (MIT) with a
`\p{N}{1,2}` digit pre-tokenizer (`2007` → `20|07`), a strong, license-clean
default. Attribution + provenance:
[`tokenizers/tokenov_v1/ATTRIBUTION.md`](../tokenizers/tokenov_v1/ATTRIBUTION.md).
If you changed the default with `set-default`, upgrading tokenov keeps your
choice.

