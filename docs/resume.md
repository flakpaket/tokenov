# Resume and checkpoints

Every fast-mode `generate` run is **resumable by default** — no `--output`
required, so a killed `tokenov | hashcat` pipe can be picked back up.
tokenov periodically checkpoints each worker's enumeration position to a
state file **belonging to that run alone**
(`$XDG_STATE_HOME/tokenov/sessions/<session-id>.ckpt`, else
`~/.local/state/tokenov/sessions/…`). After a kill, re-run the SAME command
with `--resume`:

```bash
# original piped run — checkpointed automatically
tokenov generate --model m.ngram --count 1B | hashcat -m 0 hashes.txt

# killed? resume where it left off — no --output, no extra flags
tokenov generate --model m.ngram --count 1B --resume | hashcat -m 0 hashes.txt
```

Each thread jumps straight back to where it was and continues, instead of
re-generating from candidate 0. `--resume` locates the newest checkpoint
whose fingerprint matches the current invocation — model, args, `--threads`
and tokenov version all have to agree (it errors on a mismatch rather than
corrupting the stream). On clean completion a run removes its own checkpoint;
its session record stays in the session list (`--sessions`) as history.

**Per-run isolation is guaranteed.** A run only ever writes, reads or removes
state carrying its own session id, so concurrent and unrelated runs cannot
touch each other's resume state — not by convention, but because they never
name the same file. Resuming **appends** to `--output` and never truncates it: the
saved position always lags durable output slightly, so a resume re-emits a
short overlap rather than risking a gap.

Every run already has its own checkpoint, so several long tasks coexist
safely; `tokenov --sessions` (or `tokenov generate --sessions`) lists them and
`--resume-session <ID>` returns to a specific one. `--checkpoint-file <FILE>`
still pins a checkpoint to a path you choose, resumable with
`--resume --checkpoint-file <FILE>` or `--resume-state <FILE>`. `--no-checkpoint` disables the state file entirely.
`--checkpoint-secs` (default 300) sets the cadence, which is also the resume
safety margin — the last checkpoint lags the crash, so resume re-tests a
short overlap.

**Strict mode** (`--strict`) resumes the same way, without
re-generating. Being single-threaded it saves one position; on kill it
records that position (checkpoint file) together with the output byte offset
(`<output>.progress` sidecar — format in
[file-formats.md](file-formats.md#outputprogress-byte-offset-sidecar-for-strict-resume)) at the same emitted count, so `--resume`
restores the position, truncates the output to the paired offset, and
continues. Strict resume therefore needs `--output` (it appends to the file;
there's no clean append path for a stdout stream). With `--no-checkpoint`
there's no saved position, and `--resume` falls back to re-running the
deterministic stream and skipping the first N already-written candidates via
the sidecar.

**`--wordlist`** (the default generator, `--append-only` and `--prepend-only`) saves one
position for the whole run: the level it is on and which parts of that level
are finished. `--resume` continues from there. It re-generates the unfinished
parts of that level and leaves out every line already written, so nothing is
repeated and a `--strict` run continues byte for byte. To know which lines are
written, it reads `--output` back (cut to the position saved with the
checkpoint), or, when the run writes to stdout, a file of 8 bytes per written
line kept next to the checkpoint (`<checkpoint>.seen`, removed when the run
finishes). Writing to stdout, the lines printed after the last checkpoint come
out again, as in standard mode. Strict runs resume without `--output` too.
`--append-only-legacy` and `--rare-first` resume as described above.

