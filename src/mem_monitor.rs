/// Background memory monitor that aborts tokenov before it wedges the host.
///
/// Two independent triggers:
///   1. Own RSS exceeds `max_rss_kb` (default: 75% of MemTotal, floored at 1 GiB —
///      see `default_rss_cap_kb`).
///   2. System-wide MemAvailable / MemTotal drops below `pressure_threshold` (default 10 %).
///
/// VmSwap > 0 emits a warning but does not abort by itself; swap onset is a
/// strong signal that trigger 2 is imminent.
///
/// The monitor runs on a detached thread. On breach it sets a shared AtomicBool
/// `abort` flag and exits; the main thread polls the flag at each chunk boundary
/// (every few thousand candidates) and returns an error if it is set.
///
/// On clean exit the caller should read `peak_rss_kb()` and log it.
///
/// # Testing
///
/// `MemReader` is a trait so tests can inject a `FakeMemReader` that returns
/// controlled sequences of snapshots without touching /proc.

use std::fs;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc,
};
use std::thread;
use std::time::Duration;

// ── Snapshot ─────────────────────────────────────────────────────────────────

/// One sample of process + system memory.
#[derive(Debug, Default, Clone)]
pub struct MemSnapshot {
    pub rss_kb:   u64, // VmRSS  — resident set size of this process
    pub swap_kb:  u64, // VmSwap — swap used by this process
    pub avail_kb: u64, // MemAvailable — system-wide free+reclaimable
    pub total_kb: u64, // MemTotal     — system-wide installed RAM
}

// ── MemReader trait ───────────────────────────────────────────────────────────

pub trait MemReader: Send + 'static {
    fn snapshot(&self) -> anyhow::Result<MemSnapshot>;
}

// ── Real /proc reader ─────────────────────────────────────────────────────────

pub struct ProcMemReader;

impl MemReader for ProcMemReader {
    fn snapshot(&self) -> anyhow::Result<MemSnapshot> {
        let mut s = MemSnapshot::default();
        parse_proc_self_status(&mut s)?;
        parse_proc_meminfo(&mut s)?;
        Ok(s)
    }
}

fn parse_proc_self_status(s: &mut MemSnapshot) -> anyhow::Result<()> {
    let text = fs::read_to_string("/proc/self/status")?;
    for line in text.lines() {
        let Some((key, val)) = line.split_once(':') else { continue };
        let kb: u64 = val.split_whitespace().next()
            .and_then(|v| v.parse().ok()).unwrap_or(0);
        match key.trim() {
            "VmRSS"  => s.rss_kb  = kb,
            "VmSwap" => s.swap_kb = kb,
            _ => {}
        }
    }
    Ok(())
}

fn parse_proc_meminfo(s: &mut MemSnapshot) -> anyhow::Result<()> {
    let text = fs::read_to_string("/proc/meminfo")?;
    for line in text.lines() {
        let Some((key, val)) = line.split_once(':') else { continue };
        let kb: u64 = val.split_whitespace().next()
            .and_then(|v| v.parse().ok()).unwrap_or(0);
        match key.trim() {
            "MemAvailable" => s.avail_kb = kb,
            "MemTotal"     => s.total_kb = kb,
            _ => {}
        }
        if s.avail_kb > 0 && s.total_kb > 0 { break; } // both found — stop scanning
    }
    Ok(())
}

// ── Config ────────────────────────────────────────────────────────────────────

pub struct MemMonitorConfig {
    /// Abort if process RSS exceeds this (kB). 0 = disabled.
    pub max_rss_kb: u64,
    /// Retire one worker thread when RSS exceeds this (kB). 0 = disabled.
    /// Soft action: decrements `active_target` (passed to `start`) by 1, with a
    /// 30-interval cooldown between decrements to avoid rapid de-threading.
    /// Typical value: 60% of max_rss_kb.
    pub soft_rss_kb: u64,
    /// Abort if MemAvailable/MemTotal < this fraction. 0.0 = disabled.
    pub pressure_threshold: f64,
    /// How often to sample.
    pub interval: Duration,
}

impl MemMonitorConfig {
    /// Compute the default RSS cap: 75% of MemTotal.
    ///
    /// Uses MemTotal (total installed RAM) rather than MemAvailable so the cap
    /// is stable and hardware-derived. MemAvailable fluctuates with page cache
    /// and transient activity — a warm system can report 11 GiB available on a
    /// 31 GiB machine, producing a uselessly small default cap and forcing users
    /// to pass --max-rss-gb. 75% of MemTotal leaves a consistent 25% cushion for
    /// the OS, other processes, and kernel buffers regardless of cache state.
    ///
    /// Falls back to 4 GiB if /proc/meminfo is unreadable.
    pub fn default_rss_cap_kb() -> u64 {
        const GIB: u64 = 1024 * 1024;
        let total = Self::read_total_kb().unwrap_or(4 * GIB);
        let cap = (total as f64 * 0.75) as u64;
        cap.max(GIB) // floor at 1 GiB on tiny systems
    }

    fn read_total_kb() -> anyhow::Result<u64> {
        let text = fs::read_to_string("/proc/meminfo")?;
        for line in text.lines() {
            if let Some((key, val)) = line.split_once(':') {
                if key.trim() == "MemTotal" {
                    let kb = val.split_whitespace().next()
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(0);
                    return Ok(kb);
                }
            }
        }
        anyhow::bail!("MemTotal not found in /proc/meminfo")
    }
}

// ── Monitor ───────────────────────────────────────────────────────────────────

pub struct MemMonitor {
    abort:       Arc<AtomicBool>,
    peak_rss_kb: Arc<AtomicU64>,
}

impl MemMonitor {
    /// Spawn the monitor thread. Returns immediately; monitoring runs in background.
    ///
    /// `active_target` — an atomic holding the desired number of active worker
    /// threads. When the soft RSS threshold fires the monitor decrements it (floor
    /// `min_threads`). Workers check this atomic before pulling the next domain and
    /// retire if their thread-id >= the current value.
    ///
    /// Pass `Arc::new(AtomicUsize::new(usize::MAX))` and `0` if soft-retirement
    /// is not needed (e.g., calibration runs).
    pub fn start<R: MemReader>(
        config:        MemMonitorConfig,
        reader:        R,
        active_target: Arc<AtomicUsize>,
        min_threads:   usize,
    ) -> Self {
        let abort       = Arc::new(AtomicBool::new(false));
        let peak_rss_kb = Arc::new(AtomicU64::new(0));

        let abort_t  = Arc::clone(&abort);
        let peak_t   = Arc::clone(&peak_rss_kb);

        thread::spawn(move || monitor_loop(config, reader, abort_t, peak_t, active_target, min_threads));

        MemMonitor { abort, peak_rss_kb }
    }

    /// The flag the main thread should poll at chunk boundaries.
    pub fn abort_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.abort)
    }

    /// Peak RSS seen so far (kB). Safe to read at any time.
    pub fn peak_rss_kb(&self) -> u64 {
        self.peak_rss_kb.load(Ordering::Relaxed)
    }
}

fn monitor_loop<R: MemReader>(
    config:        MemMonitorConfig,
    reader:        R,
    abort:         Arc<AtomicBool>,
    peak:          Arc<AtomicU64>,
    active_target: Arc<AtomicUsize>,
    min_threads:   usize,
) {
    let mut warned_swap    = false;
    let mut soft_cooldown  = 0u32; // ticks remaining before next soft-retirement

    loop {
        thread::sleep(config.interval);

        // Stop if main thread already set abort (e.g., we're exiting cleanly).
        if abort.load(Ordering::Relaxed) { return; }

        let snap = match reader.snapshot() {
            Ok(s)  => s,
            Err(e) => {
                eprintln!("[mem_monitor] cannot read memory stats: {e}");
                continue;
            }
        };

        peak.fetch_max(snap.rss_kb, Ordering::Relaxed);
        if soft_cooldown > 0 { soft_cooldown -= 1; }

        // Early warning: swap onset
        if snap.swap_kb > 0 && !warned_swap {
            eprintln!(
                "[mem_monitor] WARNING: process swapping {:.1} MB — \
                 memory pressure building; system may wedge if swap fills.",
                snap.swap_kb as f64 / 1024.0
            );
            warned_swap = true;
        }

        // Soft trigger: gracefully retire one worker thread to ease memory pressure.
        // Fires when RSS > soft_rss_kb, a cooldown has elapsed, and there are more
        // than min_threads workers left.
        if config.soft_rss_kb > 0 && snap.rss_kb > config.soft_rss_kb && soft_cooldown == 0 {
            let cur = active_target.load(Ordering::Relaxed);
            if cur > min_threads {
                let new_val = cur - 1;
                // CAS: only write if still == cur (another thread might have decremented).
                if active_target.compare_exchange(cur, new_val, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                    eprintln!(
                        "[mem_monitor] soft-retire: RSS {:.2} GiB > soft cap {:.2} GiB; \
                         active_target {} → {} (min {}). \
                         Thread will retire after its current domain.",
                        snap.rss_kb as f64 / (1024.0 * 1024.0),
                        config.soft_rss_kb as f64 / (1024.0 * 1024.0),
                        cur, new_val, min_threads,
                    );
                    soft_cooldown = 30; // wait 30 intervals before next retirement
                }
            }
        }

        // Hard trigger 1: own RSS — abort immediately
        if config.max_rss_kb > 0 && snap.rss_kb > config.max_rss_kb {
            eprintln!(
                "[mem_monitor] ABORT (rss_cap): process RSS {:.2} GiB > cap {:.2} GiB. \
                 Swap in use: {:.1} MB. \
                 Aborting before system wedges. \
                 Hint: restart with --resume; use --max-rss-gb or TOKENOV_MAX_RSS_GB \
                 to raise the cap, or reduce --threads.",
                snap.rss_kb as f64 / (1024.0 * 1024.0),
                config.max_rss_kb as f64 / (1024.0 * 1024.0),
                snap.swap_kb as f64 / 1024.0,
            );
            abort.store(true, Ordering::SeqCst);
            return;
        }

        // Hard trigger 2: global memory pressure — abort immediately
        if config.pressure_threshold > 0.0 && snap.total_kb > 0 {
            let avail_frac = snap.avail_kb as f64 / snap.total_kb as f64;
            if avail_frac < config.pressure_threshold {
                eprintln!(
                    "[mem_monitor] ABORT (mem_pressure): system memory critically low: \
                     {:.1}% available ({:.2} GiB of {:.2} GiB total). \
                     Process RSS {:.2} GiB, swap {:.1} MB. \
                     Aborting before sshd/UI become unresponsive. \
                     Hint: restart with --resume after freeing memory.",
                    avail_frac * 100.0,
                    snap.avail_kb as f64 / (1024.0 * 1024.0),
                    snap.total_kb as f64 / (1024.0 * 1024.0),
                    snap.rss_kb as f64 / (1024.0 * 1024.0),
                    snap.swap_kb as f64 / 1024.0,
                );
                abort.store(true, Ordering::SeqCst);
                return;
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Fake reader that returns snapshots from a queue.
    struct FakeMemReader {
        queue: Mutex<std::collections::VecDeque<MemSnapshot>>,
        default: MemSnapshot,
    }

    impl FakeMemReader {
        fn new(snapshots: Vec<MemSnapshot>) -> Self {
            FakeMemReader {
                queue:   Mutex::new(snapshots.into()),
                default: MemSnapshot::default(),
            }
        }
    }

    impl MemReader for FakeMemReader {
        fn snapshot(&self) -> anyhow::Result<MemSnapshot> {
            let mut q = self.queue.lock().unwrap();
            Ok(q.pop_front().unwrap_or_else(|| self.default.clone()))
        }
    }

    fn dummy_active() -> Arc<AtomicUsize> { Arc::new(AtomicUsize::new(usize::MAX)) }

    fn start_with_fake(cfg: MemMonitorConfig, snaps: Vec<MemSnapshot>) -> MemMonitor {
        MemMonitor::start(cfg, FakeMemReader::new(snaps), dummy_active(), 0)
    }

    fn default_cfg() -> MemMonitorConfig {
        MemMonitorConfig {
            max_rss_kb:         20 * 1024 * 1024, // 20 GiB
            soft_rss_kb:        0,                 // disabled for most tests
            pressure_threshold: 0.10,
            interval:           Duration::from_millis(10), // fast for tests
        }
    }

    #[test]
    fn no_abort_when_under_threshold() {
        let snaps = vec![
            MemSnapshot { rss_kb: 1_000_000, swap_kb: 0, avail_kb: 10_000_000, total_kb: 32_000_000 },
            MemSnapshot { rss_kb: 2_000_000, swap_kb: 0, avail_kb:  9_000_000, total_kb: 32_000_000 },
        ];
        let mon = start_with_fake(default_cfg(), snaps);
        std::thread::sleep(Duration::from_millis(50));
        assert!(!mon.abort_flag().load(Ordering::Relaxed), "should not abort when under thresholds");
    }

    #[test]
    fn aborts_on_rss_breach() {
        let snaps = vec![
            MemSnapshot { rss_kb: 25 * 1024 * 1024, swap_kb: 0, avail_kb: 8_000_000, total_kb: 32_000_000 },
        ];
        let cfg = MemMonitorConfig {
            max_rss_kb:         20 * 1024 * 1024, // cap at 20 GiB
            soft_rss_kb:        0,
            pressure_threshold: 0.10,
            interval:           Duration::from_millis(10),
        };
        let mon = start_with_fake(cfg, snaps);
        std::thread::sleep(Duration::from_millis(100));
        assert!(mon.abort_flag().load(Ordering::Relaxed), "should abort when RSS > cap");
    }

    #[test]
    fn aborts_on_global_pressure() {
        let snaps = vec![
            // 5% available — below 10% threshold
            MemSnapshot { rss_kb: 4_000_000, swap_kb: 0, avail_kb: 1_600_000, total_kb: 32_000_000 },
        ];
        let mon = start_with_fake(default_cfg(), snaps);
        std::thread::sleep(Duration::from_millis(100));
        assert!(mon.abort_flag().load(Ordering::Relaxed), "should abort when global memory < 10%");
    }

    #[test]
    fn rss_trigger_disabled_when_cap_zero() {
        let snaps = vec![
            // RSS over any sane cap — but cap is 0 (disabled)
            MemSnapshot { rss_kb: 100 * 1024 * 1024, swap_kb: 0, avail_kb: 20_000_000, total_kb: 32_000_000 },
        ];
        let cfg = MemMonitorConfig {
            max_rss_kb:         0,   // disabled
            soft_rss_kb:        0,
            pressure_threshold: 0.10,
            interval:           Duration::from_millis(10),
        };
        let mon = start_with_fake(cfg, snaps);
        std::thread::sleep(Duration::from_millis(100));
        assert!(!mon.abort_flag().load(Ordering::Relaxed), "rss trigger should be disabled when max_rss_kb=0");
    }

    #[test]
    fn pressure_trigger_disabled_when_threshold_zero() {
        let snaps = vec![
            // 1% available — below any real threshold, but threshold is 0.0 (disabled)
            MemSnapshot { rss_kb: 1_000_000, swap_kb: 0, avail_kb: 320_000, total_kb: 32_000_000 },
        ];
        let cfg = MemMonitorConfig {
            max_rss_kb:         20 * 1024 * 1024,
            soft_rss_kb:        0,
            pressure_threshold: 0.0, // disabled
            interval:           Duration::from_millis(10),
        };
        let mon = start_with_fake(cfg, snaps);
        std::thread::sleep(Duration::from_millis(100));
        assert!(!mon.abort_flag().load(Ordering::Relaxed), "pressure trigger should be disabled when threshold=0");
    }

    #[test]
    fn peak_rss_tracked() {
        let snaps = vec![
            MemSnapshot { rss_kb: 1_000_000, ..Default::default() },
            MemSnapshot { rss_kb: 3_000_000, ..Default::default() },
            MemSnapshot { rss_kb: 2_000_000, ..Default::default() },
        ];
        let cfg = MemMonitorConfig {
            max_rss_kb:         0,   // disabled so we don't abort
            soft_rss_kb:        0,
            pressure_threshold: 0.0,
            interval:           Duration::from_millis(10),
        };
        let mon = start_with_fake(cfg, snaps);
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(mon.peak_rss_kb(), 3_000_000, "peak RSS should be the highest seen");
    }

    #[test]
    fn soft_threshold_decrements_active_target() {
        // RSS above soft cap → active_target should be decremented by 1
        let snaps = vec![
            MemSnapshot { rss_kb: 14_000_000, swap_kb: 0, avail_kb: 10_000_000, total_kb: 32_000_000 },
            MemSnapshot { rss_kb: 14_000_000, swap_kb: 0, avail_kb: 10_000_000, total_kb: 32_000_000 },
            MemSnapshot { rss_kb: 14_000_000, swap_kb: 0, avail_kb: 10_000_000, total_kb: 32_000_000 },
        ];
        let cfg = MemMonitorConfig {
            max_rss_kb:         20 * 1024 * 1024, // hard cap above snapshot value
            soft_rss_kb:        12 * 1024 * 1024, // soft cap below snapshot value (12 GiB)
            pressure_threshold: 0.0,
            interval:           Duration::from_millis(10),
        };
        let active_target = Arc::new(AtomicUsize::new(8));
        let mon = MemMonitor::start(cfg, FakeMemReader::new(snaps), Arc::clone(&active_target), 1);
        std::thread::sleep(Duration::from_millis(80));
        assert!(!mon.abort_flag().load(Ordering::Relaxed), "soft trigger should not abort");
        // Should have decremented by 1 (cooldown prevents further drops in this window)
        assert_eq!(active_target.load(Ordering::Relaxed), 7, "should retire one thread");
    }

    #[test]
    fn soft_threshold_respects_min_threads() {
        // active_target == min_threads — soft trigger should not decrement further
        let snaps = vec![
            MemSnapshot { rss_kb: 14_000_000, swap_kb: 0, avail_kb: 10_000_000, total_kb: 32_000_000 },
        ];
        let cfg = MemMonitorConfig {
            max_rss_kb:         20 * 1024 * 1024,
            soft_rss_kb:        12 * 1024 * 1024,
            pressure_threshold: 0.0,
            interval:           Duration::from_millis(10),
        };
        let active_target = Arc::new(AtomicUsize::new(1)); // already at min
        let mon = MemMonitor::start(cfg, FakeMemReader::new(snaps), Arc::clone(&active_target), 1);
        std::thread::sleep(Duration::from_millis(80));
        assert!(!mon.abort_flag().load(Ordering::Relaxed));
        assert_eq!(active_target.load(Ordering::Relaxed), 1, "should not go below min_threads");
    }
}
