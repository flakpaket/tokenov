# FAQ

Answers to questions that come up about why tokenov behaves the way it does.

---

## Why do `--min-len` and `--max-len` count bytes instead of characters?

Because measuring bytes is free and measuring characters is not, and for almost
all password candidates the two numbers are identical.

As the generator walks its tree of tokens, it maintains a running buffer holding
the raw bytes of the candidate it is currently building. When a candidate is
finished, its length is the end position of that buffer minus the start position
— a single subtraction, with no memory read. Counting *characters* instead means
scanning through every byte of every candidate to work out where each character
begins and ends, because a character can occupy anywhere from one to four bytes.
That scan lands in the hottest loop in the program, one that routinely emits
millions of candidates per minute.

For unaccented English text the two measurements agree exactly, because every
character is one byte. They diverge only for accented letters and non-Latin
scripts. How often that happens depends entirely on the corpus your model was
trained on: with an English-dominant model, candidates containing any non-ASCII
character at all are rare enough to be a rounding error, so the extra scan would
change essentially nothing while costing something on every single candidate.
With a model trained on Spanish, German, or Japanese text it would matter much
more.

### What this means in practice

A candidate containing accented characters is measured as *longer* than it
looks. `Contraseña1` shows eleven characters on screen but occupies twelve
bytes, because `ñ` takes two bytes rather than one. So:

- `--max-len 11` will reject it, even though it is eleven characters long.
- `--min-len 12` will accept it, even though it is not twelve characters long.

If you are generating against a non-English target and your length bounds need
to be exact, allow a byte or two of slack, or filter the output afterwards.

### One exception: the word-list graft generator

Combining `--wordlist` with `--prepend-only` or `--float` selects a separate
generator, and that one currently measures **characters**, not bytes. On that
path the two bullets above are reversed: `--max-len 11` accepts `contraseña1`,
because it counts eleven characters and never looks at the byte total.

Everything else counts bytes as described, including plain `--wordlist` and
`--wordlist --append-only`, which both run through the standard generator.

This is a known inconsistency rather than a deliberate difference, and it is
being corrected. Two related limitations apply to the same `--prepend-only` /
`--float` path in the current release: `--enterprise` is accepted but has no
effect there, and raising the enterprise minimum length can return noticeably
fewer candidates than `--count` asked for without warning. Until those are
fixed, use `--append-only` (the default for `--wordlist`) if you need length
bounds and policy filtering to behave consistently.

### Why enterprise mode is different

`--enterprise` deliberately counts characters, not bytes. It is imitating a
Windows domain password policy, and a policy that says "at least eight
characters" means eight things a user can see — a domain controller does not
care how many bytes they take up. It can afford the more expensive count because
it only ever examines candidates that already survived the byte-based length
filter, so it runs on a much smaller stream.

The two checks use different units because they are doing different jobs. It is
worth knowing about, because it is easy to assume `--min-len 12` and an
enterprise minimum of 12 mean the same thing. For English candidates they do;
for anything else they do not.

---

## Why does enterprise mode check five character categories instead of four?

Because Windows Active Directory defines five, not four — and the fifth one
exists for a reason that rarely comes up until you generate non-English
candidates.

Most people describe password complexity as four categories: lowercase,
uppercase, digits, and special characters. That is the common shorthand, and for
English-only passwords it is complete. Microsoft's actual rule for "password must
meet complexity requirements" is that a password must draw from **at least three
of these five** categories:

1. **Uppercase letters** of European languages — `A`–`Z`, plus accented capitals
   like `Á` and `Ñ`, plus Greek and Cyrillic capitals.
2. **Lowercase letters** of European languages — `a`–`z`, plus accented letters
   like `é` and `ñ`, plus Greek and Cyrillic lowercase.
3. **Digits** — `0` through `9`.
4. **Non-alphanumeric characters** — punctuation and symbols. Which symbols
   count is fuzzier than it sounds; see the edge case at the end of this answer.
5. **Any other Unicode alphabetic character** — one that is a letter but is
   neither uppercase nor lowercase. Microsoft's documentation describes this as
   covering characters from Asian languages.

A single character only ever counts toward one category.

### The fifth category is not about accented letters

This is the most common misreading, so it is worth being explicit: category five
has nothing to do with `ñ`, `é`, `ü`, or any other accented European letter.
Those are ordinary lowercase and uppercase letters, and they belong in categories
one and two.

Category five exists for writing systems that have no concept of capital and
small letters at all. Japanese, Chinese, Thai, Hebrew, and Arabic characters are
unmistakably letters, but they have no uppercase form, so they cannot go in
category one or two. Without a fifth category they would have to be lumped in
with punctuation, which would be wrong — they are not symbols. Microsoft made a
category for them, and tokenov mirrors it.

### How tokenov actually classifies characters

| Character | Category |
|---|---|
| `n`, `ñ`, `é`, `ß` | lowercase letter |
| `N`, `Ñ`, `É` | uppercase letter |
| `0`–`9` | digit |
| `!`, `@`, `#`, and other ASCII punctuation, plus the space | special |
| `日`, `ก`, and other caseless letters | other alphabetic |
| `♥`, `€` | none — counts toward length only |

Note in particular that **`ñ` is a lowercase letter, not a special character.**
An easy way to confirm this is to compare `Contrasena` against `Contraseña`.
Both are ten characters, and both draw from exactly two categories (uppercase
and lowercase), so both are rejected. The `ñ` earns the password nothing that a
plain `n` would not have. If it were being miscounted as a special character,
the accented version would reach three categories and pass while the plain
version failed.

This distinction matters because getting it wrong inflates your apparent
compliance rate. Treating every non-English character as "special" makes
accented candidates look like they satisfy an extra category, so the tool emits
passwords that a real domain controller would refuse — and you spend guesses on
them.

### One edge case that is genuinely arguable

`♥` and `€` count toward a candidate's length but toward no category at all.

The reasoning is that Microsoft's older documentation spells out the special
character set explicitly, and everything on that list is ASCII punctuation — so
a currency symbol or a dingbat does not qualify. But the modern summarized
wording simply says "non-alphanumeric characters (symbols)," and a euro sign is
unarguably a non-alphanumeric symbol. A real domain controller may well accept
it under category four where tokenov does not.

In practice these characters are vanishingly rare in generated candidates, so
the effect is negligible. But the bias runs in a specific direction worth
knowing: tokenov is *stricter* than the domain might be, so it will occasionally
skip a candidate that would have been accepted. If you are targeting a
population that uses currency or symbol characters heavily, this is the one
place where the Active Directory imitation rests on soft ground.

### Sources

- [Password must meet complexity requirements (Windows Server)](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-server-2012-r2-and-2012/hh994562(v=ws.11))
- [Password must meet complexity requirements (Windows 10)](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/security-policy-settings/password-must-meet-complexity-requirements)
