# The unigram tail (`--unigram-tail`)

An option of [`tokenov generate`](generate.md). How the tail fits into the
model is in [algorithm.md](algorithm.md#model).

By default tokenov only proposes tokens it has actually seen following the
current context. `--unigram-tail` additionally offers the most frequent tokens
in the corpus at every step, letting the generator reach candidates its context
statistics alone cannot — `oatmeal77` when `77` never followed `oatmeal` in
training.

Tail tokens are weighted below the ones the model has seen in that context, so
the tail changes little at the start of the stream and more the further you go.

```bash
tokenov generate --unigram-tail --count 1B       # default weight (0.1)
tokenov generate --unigram-tail 0.3 --count 1B   # stronger tail
```

The optional `FRACTION` controls how much probability the tail gets. Higher
means tail candidates appear earlier and more often. Omitting the flag disables
the tail entirely; a bare `--unigram-tail` uses 0.1. All the measurements below
use the default weight.

## When to use it

**Short answer:** generate the default stream and crack with it first. Then
generate a tail stream of the same length, remove everything the default stream
already contained, and crack with what is left, the *tail-only list*. It is
small, and in every test it found more new passwords per candidate than
generating more default output did.

How much the tail adds depends on two things:

- **How many candidates you can afford.** The two streams differ more the
  deeper you go. In the first 20 million candidates only 0.03% of the tail stream
  is new; by 1 billion it is 0.76%, and by 10 billion 1.7%. If your budget is in
  the tens of millions, the tail changes almost nothing.
- **How many rules you run.** Rules turn many tail-only candidates back into
  passwords the default stream already finds, so the bigger the rule set, the
  smaller the gain. With no rules or a small set, the tail stream on its own beats
  the default stream; with a large set like `essential.rule` (about 24,000 rules)
  the two are close to tied. The tail-only list still adds new cracks with any
  rule set.

So the tail is most useful when you can generate a lot of candidates but cannot
apply many rules to them: a slow hash with a long time budget, or a fast hash
with a very large candidate budget. With a slow hash and a small budget, skip it.

## What it adds

Measured against HIBP SHA-1 hashes, cracking each stream to the end. "Tail
instead of default" replaces the default stream with an equal number of tail
candidates. "Tail-only list after default" runs the default stream, then the
tail-only list. The last column runs the default stream, then the same number
of further default candidates instead, for comparison.

| candidates per stream | rules | tail instead of default | tail-only list after default | same number of extra default candidates |
|---|---|---:|---:|---:|
| 1 billion | none | +1.1% | +1.4% | +0.1% |
| 1 billion | best66 | +0.6% | +0.8% | +0.07% |
| 1 billion | essential.rule | +0.1% | +0.2% | +0.03% |
| 10 billion | none | +3.8% | +4.3% | +0.2% |
| 10 billion | best66 | +1.4% | +1.8% | +0.2% |

Percentages are of what the default stream alone cracked. At 1 billion the
tail-only list was 7.6 million candidates; at 10 billion, 169 million.

The tail is a way to generalize beyond the training data, so it helps most on
passwords unlike the ones the model learned from. Against 20 other leaked
password sets (no rules, 1 and 10 billion candidates), the tail stream cracked
more than the default stream on 19 of 20. The one loss was RockYou, the set the
default model was trained on.

**Which rules.** The new cracks from the tail-only list under `essential.rule`
came from over 6,000 different rules, none of them dominant. Run the tail-only
list with the same full rule set as the default pass; a trimmed set loses most
of the gain, and the list is small enough that the full set is cheap.

## Building the tail-only list

Remove from the tail stream everything the default stream already produced,
then crack with what is left, using the same potfile as the default run.

```bash
N=1B

# 1. Default stream. Crack with it as usual (it is in likelihood order).
tokenov generate --count $N > default.txt
hashcat -m 100 -a 0 hashes.txt default.txt -r best66.rule

# 2. Sorted, deduplicated copies of both streams. The tail stream is only
#    needed sorted, so it goes straight into sort.
LC_ALL=C sort -u -S 8G -T /big/tmp default.txt > default.sorted
tokenov generate --unigram-tail --count $N | LC_ALL=C sort -u -S 8G -T /big/tmp > tail.sorted

# 3. Lines in tail.sorted that are not in default.sorted.
LC_ALL=C comm -13 default.sorted tail.sorted > tail-only.txt

# 4. Crack with the tail-only list, same rules, same potfile.
hashcat -m 100 -a 0 hashes.txt tail-only.txt -r best66.rule
```

Notes:

- `LC_ALL=C` must be set for both `sort` and `comm`, so they agree on the
  order; otherwise `comm` gives wrong results.
- `sort` needs temporary space of about two to three times the file size. `-T`
  puts it on a disk with room; `-S` sets how much memory it may use. A
  1-billion-candidate stream is about 10 GB.
- `sort -u` also removes the tail stream's own duplicates (about 0.3% of a
  1-billion stream): it can reach the same string through more than one token
  split.
- `tail-only.txt` is in byte order, not likelihood order. That doesn't matter
  when you crack the whole list, which is the point of keeping it small.
- Build `default.sorted` from the same default stream you cracked with, so the
  tail-only list holds only candidates that run never tried.

## Limitations

`--wordlist` with `--float` or `--prepend-only` warns and ignores the flag: that
generator joins words to a fixed set of additions instead of walking the model,
so the tail has nothing to act on. `score` always uses the default weight, so
scores are not directly comparable to a stream generated with a non-default
`FRACTION`.
