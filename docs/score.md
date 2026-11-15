# Scoring candidates (`tokenov score`)

The inverse of `generate`: instead of emitting candidates in probability
order, it takes candidates you supply and reports how (im)probable each one
is under the model. It works like a password-strength meter: it tells you how
early the model would guess a given password.

**Rarity** is the metric: `−log₂ P(candidate)`, in **bits**. Higher Rarity =
rarer = the model would generate it later (a stronger password); lower Rarity =
the model expects it early (a weaker one). `Rarity/token` is the same figure
divided by the number of tokens.

```bash
# Score a few candidates against the default model (table output)
tokenov score 123456 password correcthorsebatterystaple 'Tr0ub4dour&3'
```

```
Candidate            Tokens  Rarity(bits)  Level  Segmentation
123456                    3           7.0      7  12|34|56
password                  1           9.1      7  password
correcthorsebattery…      6          92.8     68  correct|horse|b|attery|st|aple
Tr0ub4dour&3              8          87.9     67  Tr|0|ub|4|d|our|&|3
```

The columns: `Tokens` is the token count, `Rarity(bits)` the Rarity
(`-log2 P(candidate)` in bits; higher = rarer = stronger), `Level` the
level at which `generate` produces this candidate (see below), and
`Segmentation` the plain token split (`hack|the|planet`). Long candidates are
truncated in the first column only; the full split still shows in
`Segmentation`.

Pass **`-d` / `--detailed`** to add a per-position `Segment Score` column:

```bash
tokenov score -d 123456 correcthorsebatterystaple 'Tr0ub4dour&3'
```

```
Candidate            Tokens  Rarity(bits)  Level  Segmentation              Segment Score
123456                    3           7.0      7  12|34|56                  START|12(5.3)|34(0.6)|56(0.5)|END(0.6)
correcthorsebattery…      6          92.8     68  correct|horse|b|attery|st|aple  START|correct(18.4)|horse(20.5*)|b(10.1)|attery(14.9)|st(15.9*)|aple(12.0)|END(0.9)
Tr0ub4dour&3              8          87.9     67  Tr|0|ub|4|d|our|&|3       START|Tr(14.6)|0(6.8)|ub(5.9)|4(7.7)|d(9.0)|our(16.8)|&(16.9*)|3(8.8)|END(1.5)
```

`Segment Score` is the **per-position breakdown** — `START`, then each
token (and the end of the password, `END`) annotated with the Rarity *it*
contributes, in bits. The parenthetical bits sum to `Rarity(bits)`, so you can
read straight across *where* a password is predictable vs. surprising — for
`123456`, the `12` prefix carries almost all the bits and `34`/`56` are near-free
continuations. This is more informative than a single length-averaged number: the
model's "surprise" is rarely spread evenly across the string.

A trailing `*` on a `Segment Score` step (e.g. `horse(20.5*)`) marks a
**floored** step — one the model has never seen, so it got a small fallback
probability (see below). It shows exactly which token the model couldn't
cover. (`tsv`/`jsonl` always carry the per-position
data regardless of `-d`.)

Candidates come from positional args, `--file <path>` (one per line), or
stdin (one per line) if neither is given. `--model` resolves exactly like
`generate` (registered name or `.ngram` path; omit for the default model).

**What the score is, precisely.** (For the model itself, see
[algorithm.md](algorithm.md).) Rarity uses the same probabilities as
`generate`, including the step that ends the password, so `123456` is scored
as a complete password, not as the start of something longer. It scores the
tokenizer's one way of splitting the password; `generate` may reach the same
string by a different split, so Rarity and generation order agree only
roughly.

**Level, and why it isn't the Rarity number.** `Level` is the level at which
`generate` produces the candidate — the number to hand to `--min-level`. It is
computed exactly the way `generate` computes it, which is *not* a rounding
of the Rarity figure next to it. Two things differ:

- **Unit.** Rarity is bits (`−log₂ P`); levels are nats (`−ln P`). One nat is
  1.4427 bits, so 92.8 bits of Rarity is 64.3 nats.
- **Rounding.** The level is not one rounding of the total. Each transition's
  cost is rounded **up** on its own and the results are summed (the `END`
  transition included), so a candidate with `k` tokens has been rounded up `k+1`
  times and its level sits *above* the nats total by up to that much.

So `level ≈ ceil(rarity_bits / 1.4427)`, plus up to one per transition —
`correcthorsebatterystaple` above is 92.8 bits = 64.3 nats over 7 transitions,
and lands at level 68. Rarity and level order candidates near-identically, but
neither converts cleanly into the other.

Two caveats. A candidate with a floored step (`in_vocab: false`) still gets a
level, but it is hypothetical: that step's probability came from the fallback, and
`generate` cannot produce that step. And the level is
the level of the tokenizer's split of the candidate — `generate` may reach
the same string by a different token path with a different level, the same
caveat that already applies to Rarity. An unreachable candidate
(`--reachable`, Rarity `+inf`) reports no level at all: `-` in the table, `null`
in `jsonl`.

**Floored steps, and when Rarity is `+inf`.** If the model has never seen a
token follow the previous ones, `score` gives that step a small fallback
probability so the total stays finite, marks the step `*`, and reports the
candidate as `in_vocab: false`. `--reachable` reports `+inf` instead, meaning
`generate` can never produce this password, and the segmentation stops at that
token (e.g. `correct(18.4)|horse(inf)`).

**Output formats** (`--format`):

- `table` (default) — aligned, human-readable, long candidates truncated for
  display only. Shows a plain `Segmentation` column; `-d`/`--detailed` adds a
  `Segment Score` column with the per-position bits + `*` floor markers.
- `tsv` — `candidate, n_tok, surprisal_bits, surprisal_per_tok_bits, in_vocab,
  segmentation, per_pos_surprisal_bits, per_pos_floored, level`.
  `per_pos_surprisal_bits` and `per_pos_floored` are comma-joined, one entry per
  transition (each token, then the trailing `END` step, so they hold `n_tok + 1`
  entries); `per_pos_surprisal_bits` sums to `surprisal_bits`. `level` is `-` for
  an unreachable candidate. `surprisal_bits` is the Rarity. Values are
  untruncated; feed them straight into pandas/awk.
- `jsonl` — one JSON object per line, same scalar fields (including `level`, an
  integer or `null`) plus a `positions` array of per-transition
  `{label, surprisal_bits, floored}`; `+inf` is emitted as `null` so it parses. For analysis pipelines (e.g. studying whether a Rarity
  threshold predicts crackability).

```bash
# Bulk-score a wordlist for analysis
tokenov score --file candidates.txt --format tsv > rarity.tsv
```

