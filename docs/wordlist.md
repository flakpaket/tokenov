# Wordlist targeting (`--wordlist`)

`tokenov generate --wordlist FILE` builds every candidate around a word from your
own list. Use it against a specific organization: put in employee names,
product and project names, brand words, place names, or every word scraped from
the organization's website (for example with CeWL). One word per line. Lists of
thousands or tens of thousands of words are normal.

Each candidate keeps one of your words intact, and the model adds characters
before it, after it, or both: `cisco`, `cisco1`, `ilovecisco`, `#1cisco2024`.

```bash
tokenov generate --wordlist words.txt --count 10M --output cands.txt
```

Flags are listed in [cli-reference.md](cli-reference.md).

## How it works

- **The word can be anywhere.** The model adds characters in front of the
  word, after it, or on both sides, the same way it builds any password. The
  additions are whatever the model finds likely: `i`, `my`, `1`, `#1`, `2024`,
  `!`, `123`.
- **Likeliest first, across all words.** Every candidate is priced as the
  model prices any password, and candidates come out in that order across your
  whole list. The word costs what the model says it costs after the characters
  in front of it, so `ilove` + `cisco` comes out when that combination is
  likely, not just because `ilove` is common. Common words in your list
  (`love`, `team`, `summer`) and common additions produce many of the early
  candidates.
- **Additions can run into the word.** The tokenizer often splits a password
  across the join (`ilove` is `il` + `ove`). tokenov builds those splits too,
  so `i` + `love` is not lost.
- **Words the model rarely saw still come out.** A step inside a word that the
  model has no statistics for is priced from how common that token is overall,
  so an unusual spelling (`cryptocurrency`, many capitalized words from a
  website) still gets candidates, later than common words. Words that still
  cannot be built are skipped; `-v` reports how many.
- **Every line is unique.** The same password can be built from different
  words or splits (`iloveteam` from `love` and from `team`); it is printed
  once, at its likeliest. `--count` is exact.

In our tests on leaked passwords of staff at nine large companies (people with
a corporate email address, leaked mostly from third-party sites), with a
wordlist scraped from each company's website, on the passwords left after
running that wordlist through hashcat's dive rules, the default cracked about
65% more than `--append-only` (the `--wordlist` of tokenov 1.3 and earlier) at
100 million candidates, and more at every one of the nine companies.

**Memory.** To keep lines unique, tokenov remembers every candidate it has
printed: about 16 bytes each, so 1.6 GB per 100 million. A run without
`--count` keeps growing; give long runs a `--count`.

## Choosing a generator

| flags | where the added characters go | order |
|---|---|---|
| (none) | before, after, or both | likeliest first across all words |
| `--prepend-only` | before the word only: the word ends the candidate (`ilovecisco`, `2024cisco`) | likeliest first across all words |
| `--append-only` | after the word only: the word starts the candidate (`cisco2024!`). This is how `--wordlist` worked up to tokenov 1.3; the output is the same as 1.3.0's | likeliest first across all words |
| `--rare-first` | either side, or one side with `--append-only` / `--prepend-only` | the rarest words get the most candidates; to be deprecated |

**Start with the default.** Use `--append-only` to reproduce the output of
earlier versions, and `--prepend-only` when you know the word ends the
password.

## Flags with the default generator

Every `generate` flag works, with `--prepend-only` too:

- **`--strict`** gives one exactly ordered stream, byte-identical on every run.
  Without it, threads share out the work and lines arrive in the order the
  threads produce them. With `--count`, the last few candidates can differ
  between threaded runs: whichever thread gets there first fills the count.
- **`--max-tokens`** counts all tokens: the additions and the word together.
  `--min-tokens`, `--min-len` and `--max-len` apply to the whole candidate.
- **`--unigram-tail`** applies to the characters the model adds, as in
  standard generation. It does not change how the word itself is priced.
- **`--case-shape`** re-cases each token by its position in the whole
  candidate, as in standard generation. With `cap1` the first token is the
  first addition when there is one (`Ilovecisco`), not necessarily the word.
- **`--enterprise`** keeps only policy-compliant candidates, as in standard
  generation. A run that ends before `--count` says so on stderr.
- **`--min-level N`** starts at level N, skipping the likelier candidates below
  it. Unlike standard mode, the result is not exactly the tail of a full run: a
  password the full run builds below level N through one word or split can be
  built again at N or above through another. In our tests about 1% of the lines
  were such repeats.
- **Checkpoints and `--resume`** work as in [resume.md](resume.md). A resumed
  run picks up at the level it was on, re-generates the part of that level that
  was not finished, and leaves out every line already written, so a `--strict`
  run continues byte for byte. With `--output`, tokenov reads the file back to
  learn which lines are written. Writing to stdout, it keeps a list of 8 bytes
  per line next to the checkpoint instead, removed when the run finishes.
- `--lazy`, `--force-child-cache` and `--bounded-cap` tune a cache this
  generator does not use: they print a warning and are ignored.

## `--append-only`: after the word, as before 1.4

Each word is the start of the candidate, and the model adds characters after
it. Candidates come out in order of how likely the model finds the whole
password, across all words. The output is byte-identical to tokenov 1.3.0's
`--wordlist`, so use this flag to reproduce older runs. Every `generate` flag
works.

## `--prepend-only`: before the word

The default generator with nothing after the word: the model adds characters in
front of it and the candidate ends with the word (`ilovecisco`, `2024cisco`,
`mycisco`). Before tokenov 1.4.0 this flag ran a different generator with a
fixed set of additions; that generator is gone.

## `--rare-first`: rare words, model-generated additions

`--rare-first` will be deprecated in a future release and prints a warning. The
default cracked more in testing. Its output is unchanged.

`--rare-first` gives the rarest words the most candidates:

- **Additions on either side**, of any length up to `--max-tokens`:
  `cisco2024!`, `ilovecisco`, `2024cisco1`. `--append-only` keeps them after
  the word and `--prepend-only` keeps them before it.
- **Shares by rarity.** A word's rarity is how unlikely the model finds it.
  Each word's share of the output is proportional to its rarity raised to the
  power `--rarity-weight` (default 1). At 2, a word twice as rare gets four
  times the candidates. At 0, every word gets the same share. `-v` prints each
  word's share.
- **Within one word**, candidates come out in order of how likely the
  additions are. The word itself is never penalized for being rare.
- **Compound words are split.** If a word is made of other words, such as
  `johnsmith`, the parts (`john`, `smith`) are used as words too.
  `--no-split` turns that off.
- **`--mix`** lets another word from the list be used as an addition
  (`johncisco`). It is off by default because the number of pairs grows with
  the square of the list size, and on a large list the pairs would take most of
  the output.

Every `generate` flag works with `--rare-first`. Under `--strict` the shares
hold at every point in the output. Without `--strict` the threads each take
some of the words. With `--count`, the totals per word still follow the shares,
but lines arrive in whatever order the threads produce them.

Some words need special handling:

- **A word longer than `--max-tokens`** is printed on its own, with a warning.
- **A word the model has never seen anything follow** (common for capitalized
  words from a website, such as `Government`) borrows the additions of its
  lowercase form. If that is unknown too, it gets prefixes only. stderr reports
  how many words this affected.
- **With `--enterprise`**, a run that ends before `--count` (or produces
  nothing) says so on stderr.

## The tokenizer

tokenov splits your words into tokens with the same tokenizer the model was
trained with. It finds that tokenizer on its own, in this order:

1. the `TOKENOV_TOKENIZER` environment variable, if set;
2. the tokenizer stored inside the model file (models normally have one);
3. for model files without one: a `<model>.tokenizer.json` file next to the
   model, then, for the default model, the tokenizer `tokenov bootstrap`
   installed.

So `tokenov generate --wordlist words.txt` needs no other flags, with the
default model or a model copied from another machine. Set `TOKENOV_TOKENIZER`
only to deliberately use a different tokenizer; tokenov logs that it is
overriding the stored one.
