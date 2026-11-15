# Generate on one machine, crack on another

Generation memory and GPU cracking pull in opposite directions. Cracking wants a
GPU box; **standard (no `--wordlist`) generation wants RAM** — its footprint
scales with the number of distinct token contexts in the model, not with
the `.ngram` file size or the GPU. A model trained on a very large or diverse
corpus can need far more RAM to generate at full speed than your GPU box has.

The fix is to **generate on a RAM-rich (or faster-CPU) host and stream candidates
to the GPU host over a plain `ssh` pipe** — no netcat, no ports, no extra tools:

```bash
# candidates are generated on `genhost` and streamed straight into local hashcat
ssh genhost 'tokenov generate --model big.ngram --count 1B' \
  | hashcat -a 0 -m 100 hashes.txt -r rules/dive.rule
```

Copy the `.ngram` to `genhost` first. The model file contains its
tokenizer, so the single file is enough. Run the **same tokenov version**
on `genhost` — enumeration order is version-sensitive, so a matched binary keeps
results reproducible.

**Why `ssh` (and not a lossy transport):** with a rule file — especially a large
one — hashcat consumes *base* candidates slowly, because each base word expands
into (rule-count) GPU-side variants that must all be hashed before it reads the
next base. Its stdin intake is roughly `GPU_hash_rate ÷ rule_count`, often only
tens of KB/s. `ssh` runs over TCP, whose **flow control propagates that
backpressure back to the remote `tokenov`**, which then self-throttles to the
cracker's pace — losslessly, with no candidate skipped and no need to guess a
rate. A transport without flow control (e.g. raw UDP) has no backpressure: the
generator runs ahead, the receiver's buffer overflows, and candidates are
**silently dropped** — passwords that leave the generator but are never tried.
Don't do that.

**Bandwidth is not the bottleneck.** Full-bore generation tops out at tens of
MB/s (single-digit MB/s for large bounded-cache models), so any ordinary LAN —
even 1 GbE — carries it with room to spare. The generator is always the limiting
stage; the network and the cracker's backpressure govern the rest.

**Tuning generation RAM on the gen host.** `tokenov` picks the child-cache mode
automatically from available RAM, so it will not OOM without a flag — but you can
pin it:

- `--force-child-cache` — build the full resident cache (fastest; needs the most
  RAM, scaling with context count).
- `--bounded-cap N` — keep a partial cache of the `N` most recently used contexts per thread
  (near-full speed at a fraction of the RAM; the auto-selector uses this by
  default when the full cache won't fit).
- `--lazy` — recompute children on demand (minimal RAM, slowest).
- `--max-rss-gb G` — an RSS ceiling / guard for the run.

Output is **byte-identical across all three modes** — the cache is purely a
performance/RAM trade. So put generation on your highest-RAM box to keep it in the
full or a large bounded cache and maximize candidates/second; the GPU box just
cracks. (Add `--strict` if you need a canonical single-threaded rank-ordered
stream; the default fast mode is multithreaded and higher-throughput.)

