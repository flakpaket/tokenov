# File formats

## `.ngram` (NGRMv003)

Binary file. 8-byte magic `NGRMv003`, then a **length-prefixed provenance
blob**, then the model body.

The provenance blob (a `u32` byte-length prefix, then a little-endian record:
schema version, the embedded `tokenizer.json` bytes, the `--tokenizer` and
`--train` paths, the build time, the tokenov version, and — from 1.2.0 on — a
`decode_exact` flag) makes a v3 model
**self-describing** — `generate --wordlist` and `model info` read the tokenizer
straight from the file, so it survives a bare single-file copy with no sidecar.
Inspect it with `tokenov model info <model.ngram>`.

**Decode table and the `decode_exact` flag.** The body stores, for every token,
the bytes it decodes to.

- **Models built before 1.2.0 have a lossy table.** Older builds decoded each
  token on its own, so a token holding part of a multi-byte character was stored
  as U+FFFD (`�`). That is 352 of 50,257 entries in the default model.
- **1.2.0 stores the raw bytes and sets `decode_exact`.**
- **An older v3 model is repaired at load.** With no flag, tokenov rebuilds the
  table in memory from the embedded tokenizer, in about 0.1 s, and prints a
  one-line note on stderr. The file itself is unchanged. Retrain with
  `tokenov model train` to store the fixed table and skip the rebuild.
- **Model files without an embedded tokenizer cannot be repaired**, so tokenov
  warns instead.
- **The flag is additive.** Older tokenov releases ignore it and still load
  1.2.0 models, and read the corrected table as stored.

The body is the Kneser-Ney model: a header (vocab size, START/END sentinels, KN
discount D, decode-table length), then per-context records (trigram cumulative
distributions, lambda, bigram-continuation cumulative distributions, raw and
KN-continuation unigram counts).

A model file without the provenance blob (`NGRMv002`, written only when the
tokenizer file could not be read at build time) still loads. Its tokenizer is
found separately: `$TOKENOV_TOKENIZER`, then a `<model>.tokenizer.json` file
next to the model, then the default tokenizer (`tokenov_v1` unless changed with
`tokenizer set-default`).

## `<output>.progress` (byte-offset sidecar for strict resume)

How resume uses this file is described in [resume.md](resume.md).

Alongside the checkpoint file, strict mode writes a plain-text
sidecar next to a plain-file `--output`, holding the output byte offset that
pairs with the checkpointed position:

```
emitted=<u64>
byte_offset=<u64>
fingerprint=<args summary>
```

Both are written together at each checkpoint (single-threaded, so the capture
is atomic — the byte offset always matches the saved position's emitted
count). On `--resume`, tokenov restores the position from the checkpoint
file, validates the fingerprint (model file size+mtime, threads, count,
min/max len, mode, bias, etc.), truncates the output to `byte_offset` (handling
any BufWriter spillage past the last write), and continues appending. With
`--no-checkpoint` there's no saved position, so `--resume` instead re-runs the
deterministic stream and discards the first `emitted` items via this sidecar
alone. Both files are removed automatically on clean completion.

The default `--wordlist` generator keeps the byte offset in its checkpoint file
instead, and writes this sidecar only with `--no-checkpoint` (with or without
`--strict`). On `--resume` it truncates the output to `byte_offset`, reads the
`emitted` lines back so it does not write them again, and re-runs.

