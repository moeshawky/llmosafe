//! llmosafe_body — Physical resource guard (the autonomic nervous system).
//!
//! Monitors RSS memory, maps to CognitiveEntropy, triggers KernelError
//! when physical resource thresholds are crossed.
//!
//! This module requires `std` (reads `/proc` on Linux, Win32 API on Windows).
//! Disabled in `no_std` builds.

use crate::llmosafe_kernel::{KernelError, Synapse};
use std::fs;
use std::io::{BufRead, BufReader};
use std::thread;
use std::time::Duration;

/// EnvironmentalVitals tracks system-level metabolic signals.
#[derive(Debug, Clone, Default)]
pub struct EnvironmentalVitals {
    pub iowait: u64,
    pub load_avg: f64,
    pub vitals_available: bool,
}

impl EnvironmentalVitals {
    /// Captures current system vitals.
    pub fn capture() -> Self {
        let iowait = Self::read_iowait();
        let load_avg = Self::read_loadavg();
        let vitals_available = iowait.is_some() && load_avg.is_some();
        Self {
            iowait: iowait.unwrap_or(0),
            load_avg: load_avg.unwrap_or(0.0),
            vitals_available,
        }
    }

    #[cfg(target_os = "linux")]
    fn read_iowait() -> Option<u64> {
        if let Ok(file) = fs::File::open("/proc/stat") {
            if let Some(Ok(line)) = BufReader::new(file).lines().next() {
                if let Some(iowait_str) = line.split_whitespace().nth(5) {
                    return Some(iowait_str.parse().unwrap_or(0));
                }
            }
        }
        None
    }

    #[cfg(not(target_os = "linux"))]
    fn read_iowait() -> Option<u64> {
        None
    }

    #[cfg(target_os = "linux")]
    fn read_loadavg() -> Option<f64> {
        if let Ok(file) = fs::File::open("/proc/loadavg") {
            if let Some(Ok(line)) = BufReader::new(file).lines().next() {
                if let Some(first_part) = line.split_whitespace().next() {
                    return Some(first_part.parse().unwrap_or(0.0));
                }
            }
        }
        None
    }

    #[cfg(not(target_os = "linux"))]
    fn read_loadavg() -> Option<f64> {
        None
    }
}

/// ResourceGuard monitors physical resource consumption and triggers safety halts.
/// Maps physical metrics (RAM, CPU) to the CognitiveEntropy/Synapse system.
#[derive(Debug, Clone)]
pub struct ResourceGuard {
    memory_ceiling_bytes: usize,
}

impl ResourceGuard {
    /// Creates a new ResourceGuard with a specified memory ceiling.
    ///
    /// # Arguments
    /// * `memory_ceiling_bytes` - Maximum allowed RSS memory in bytes
    pub fn new(memory_ceiling_bytes: usize) -> Self {
        Self {
            memory_ceiling_bytes,
        }
    }

    /// Returns a weighted metabolic entropy score (0-1000).
    /// Weighted by: RSS (50%), IO Wait (25%), Load Average (25%).
    /// IO Wait uses delta-based measurement on Linux for responsiveness.
    ///
    /// Returns a value in [0, 1000]. The Halt threshold in EscalationPolicy
    /// uses strict greater-than (> self.halt_entropy), so resource entropy at
    /// the cap (1000) triggers Escalate, not Halt. Use Halt for entropy values
    /// > 1000 from composite/synthetic sources outside the resource body.
    pub fn raw_entropy(&self) -> u16 {
        let current_rss = Self::try_current_rss_bytes().unwrap_or(self.memory_ceiling_bytes);
        let rss_ratio = if self.memory_ceiling_bytes > 0 {
            (current_rss as f64 / self.memory_ceiling_bytes as f64).min(1.0)
        } else {
            1.0
        };

        let vitals = EnvironmentalVitals::capture();

        // Fail-closed: if /proc is unavailable, assume worst-case load and IO pressure.
        let load_ratio = if vitals.vitals_available {
            (vitals.load_avg / 10.0).min(1.0)
        } else {
            1.0
        };

        // IO Wait: use delta-based measurement on Linux, fallback to 0 elsewhere
        #[cfg(target_os = "linux")]
        let iowait_ratio = Self::delta_iowait_ratio();
        #[cfg(not(target_os = "linux"))]
        let iowait_ratio = 0.0_f64;

        let weighted_score = (rss_ratio * 500.0) + (iowait_ratio * 250.0) + (load_ratio * 250.0);
        weighted_score.min(1000.0) as u16
    }

    /// Returns the current resource pressure as a percentage of the ceiling (0-100).
    pub fn pressure(&self) -> u8 {
        if self.memory_ceiling_bytes == 0 {
            return 100;
        }
        let current_rss = Self::try_current_rss_bytes().unwrap_or(self.memory_ceiling_bytes);
        let ratio = current_rss as f64 / self.memory_ceiling_bytes as f64;
        (ratio * 100.0).min(100.0) as u8
    }

    /// Checks current resource usage and returns a Synapse with mapped entropy.
    ///
    /// ⚠ Reads `/proc/stat` twice with a 100ms sleep between reads to compute
    /// delta-based CPU/IO metrics. Do NOT call in async contexts without
    /// spawning to a blocking thread.
    #[must_use = "ignoring the safety check defeats the purpose of the guard"]
    pub fn check(&self) -> Result<Synapse, KernelError> {
        if self.memory_ceiling_bytes == 0 {
            return Err(KernelError::ResourceExhaustion);
        }
        let current_rss = match Self::try_current_rss_bytes() {
            Some(rss) => rss,
            None => return Err(KernelError::ResourceExhaustion),
        };
        let ratio = current_rss as f64 / self.memory_ceiling_bytes as f64;

        if ratio >= 1.0 {
            return Err(KernelError::ResourceExhaustion);
        }

        let entropy = self.raw_entropy();

        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(entropy);
        synapse.set_raw_surprise(0);
        synapse.set_has_bias(false);
        synapse.set_anchor_hash(0);

        Ok(synapse)
    }

    /// Like `check()` but reuses a previously-measured entropy value.
    ///
    /// Prevents TOCTOU when entropy is measured for a policy decision
    /// and then recomputed inside `check()`, potentially returning a
    /// different entropy than what was approved.
    #[must_use = "ignoring the safety check defeats the purpose of the guard"]
    pub fn check_with_entropy(&self, entropy: u16) -> Result<Synapse, KernelError> {
        if self.memory_ceiling_bytes == 0 {
            return Err(KernelError::ResourceExhaustion);
        }
        let current_rss = match Self::try_current_rss_bytes() {
            Some(rss) => rss,
            None => return Err(KernelError::ResourceExhaustion),
        };
        let ratio = current_rss as f64 / self.memory_ceiling_bytes as f64;

        if ratio >= 1.0 {
            return Err(KernelError::ResourceExhaustion);
        }

        let mut synapse = Synapse::new();
        synapse.set_raw_entropy(entropy);
        synapse.set_raw_surprise(0);
        synapse.set_has_bias(false);
        synapse.set_anchor_hash(0);

        Ok(synapse)
    }

    /// Blocks until resources are safe, automatically honoring Escalate/Halt cooldowns.
    ///
    /// Returns an error after `max_retries` consecutive non-Proceed decisions
    /// to prevent infinite spinning under sustained pressure. Default: 3 retries.
    ///
    /// ⚠ BLOCKING: Reads /proc/stat multiple times with sleeps.
    /// Do NOT call in async contexts without spawn_blocking.
    #[cfg(feature = "std")]
    pub fn check_blocking(&self) -> Result<Synapse, KernelError> {
        self.check_blocking_with_max_retries(3)
    }

    /// Same as check_blocking() but with configurable max retries.
    #[cfg(feature = "std")]
    pub fn check_blocking_with_max_retries(
        &self,
        max_retries: u32,
    ) -> Result<Synapse, KernelError> {
        use crate::llmosafe_integration::{EscalationPolicy, SafetyDecision};

        let policy = EscalationPolicy::default();
        let mut retries = 0u32;
        loop {
            if retries >= max_retries {
                return Err(KernelError::DeadlineExceeded);
            }
            let entropy = self.raw_entropy();
            let decision = policy.decide(entropy, 0, false);
            match decision {
                SafetyDecision::Proceed | SafetyDecision::Warn(_) => {
                    return self.check_with_entropy(entropy);
                }
                SafetyDecision::Escalate { cooldown_ms, .. } => {
                    retries += 1;
                    thread::sleep(Duration::from_millis(cooldown_ms.max(1) as u64));
                }
                SafetyDecision::Halt(_, cooldown_ms) => {
                    retries += 1;
                    thread::sleep(Duration::from_millis(cooldown_ms.max(1) as u64));
                }
                SafetyDecision::Exit(err) => {
                    return Err(err);
                }
            }
        }
    }

    /// Same as check_blocking() but with deadline.
    #[cfg(feature = "std")]
    pub fn check_with_deadline(
        &self,
        deadline: std::time::Instant,
    ) -> Result<Synapse, KernelError> {
        use crate::llmosafe_integration::{EscalationPolicy, SafetyDecision};

        let policy = EscalationPolicy::default();
        loop {
            if std::time::Instant::now() >= deadline {
                return Err(KernelError::DeadlineExceeded);
            }
            let entropy = self.raw_entropy();
            let decision = policy.decide(entropy, 0, false);
            match decision {
                SafetyDecision::Proceed | SafetyDecision::Warn(_) => {
                    return self.check_with_entropy(entropy);
                }
                SafetyDecision::Escalate { cooldown_ms, .. } => {
                    thread::sleep(Duration::from_millis(cooldown_ms.max(1) as u64));
                }
                SafetyDecision::Halt(_, cooldown_ms) => {
                    thread::sleep(Duration::from_millis(cooldown_ms.max(1) as u64));
                }
                SafetyDecision::Exit(err) => {
                    return Err(err);
                }
            }
        }
    }

    /// Returns current RSS memory usage in bytes.
    #[cfg(unix)]
    pub fn current_rss_bytes() -> usize {
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };

        if ret == 0 {
            // ru_maxrss is in KB on Linux and BSDs, bytes on macOS/iOS.
            #[cfg(any(
                target_os = "linux",
                target_os = "freebsd",
                target_os = "openbsd",
                target_os = "netbsd",
                target_os = "dragonfly",
            ))]
            {
                (usage.ru_maxrss as usize).saturating_mul(1024)
            }
            #[cfg(not(any(
                target_os = "linux",
                target_os = "freebsd",
                target_os = "openbsd",
                target_os = "netbsd",
                target_os = "dragonfly",
            )))]
            {
                usage.ru_maxrss as usize
            }
        } else {
            Self::read_rss_from_proc().unwrap_or(0)
        }
    }

    #[cfg(windows)]
    pub fn current_rss_bytes() -> usize {
        use windows_sys::Win32::Foundation::HANDLE;
        use windows_sys::Win32::System::ProcessStatus::{
            GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
        };

        let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
        let handle: HANDLE = unsafe { windows_sys::Win32::System::Threading::GetCurrentProcess() };
        let ret = unsafe {
            GetProcessMemoryInfo(
                handle,
                &mut counters,
                std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            )
        };
        if ret != 0 {
            counters.WorkingSetSize as usize
        } else {
            0
        }
    }

    #[cfg(not(any(unix, windows)))]
    pub fn current_rss_bytes() -> usize {
        0
    }

    /// Like current_rss_bytes() but returns None when RSS measurement is
    /// unavailable. Callers should fail-closed (return ResourceExhaustion
    /// or max pressure) when None is returned.
    #[cfg(target_os = "linux")]
    fn try_current_rss_bytes() -> Option<usize> {
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        if ret == 0 {
            Some((usage.ru_maxrss as usize).saturating_mul(1024))
        } else {
            Self::read_rss_from_proc()
        }
    }

    #[cfg(unix)]
    #[cfg(not(target_os = "linux"))]
    fn try_current_rss_bytes() -> Option<usize> {
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        if ret == 0 {
            Some(usage.ru_maxrss as usize)
        } else {
            None
        }
    }

    #[cfg(windows)]
    fn try_current_rss_bytes() -> Option<usize> {
        use windows_sys::Win32::Foundation::HANDLE;
        use windows_sys::Win32::System::ProcessStatus::{
            GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
        };
        let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
        let handle: HANDLE = unsafe { windows_sys::Win32::System::Threading::GetCurrentProcess() };
        let ret = unsafe {
            GetProcessMemoryInfo(
                handle,
                &mut counters,
                std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            )
        };
        if ret != 0 {
            Some(counters.WorkingSetSize as usize)
        } else {
            None
        }
    }

    #[cfg(not(any(unix, windows)))]
    fn try_current_rss_bytes() -> Option<usize> {
        None
    }

    /// Helper: parse RSS from /proc/self/status (Linux fallback)
    #[cfg(target_os = "linux")]
    fn read_rss_from_proc() -> Option<usize> {
        if let Ok(file) = fs::File::open("/proc/self/status") {
            for line in BufReader::new(file).lines().map_while(Result::ok) {
                if line.starts_with("VmRSS:") {
                    if let Some(size_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = size_str.parse::<usize>() {
                            return Some(kb.saturating_mul(1024));
                        }
                    }
                }
            }
        }
        None
    }

    #[cfg(not(target_os = "linux"))]
    fn read_rss_from_proc() -> Option<usize> {
        None
    }

    /// Returns system memory in bytes.
    #[cfg(target_os = "linux")]
    pub fn system_memory_bytes() -> usize {
        if let Ok(file) = fs::File::open("/proc/meminfo") {
            for line in BufReader::new(file).lines().map_while(Result::ok) {
                if line.starts_with("MemTotal:") {
                    if let Some(size_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = size_str.parse::<usize>() {
                            return kb.saturating_mul(1024);
                        }
                    }
                }
            }
        }
        0
    }

    #[cfg(not(target_os = "linux"))]
    pub fn system_memory_bytes() -> usize {
        0
    }

    /// Creates a ResourceGuard with ceiling = system_memory * fraction.
    pub fn auto(fraction: f64) -> Self {
        let sys_mem = Self::system_memory_bytes();
        let ceiling = if sys_mem > 0 {
            (sys_mem as f64 * fraction) as usize
        } else {
            0
        };
        Self::new(ceiling)
    }

    /// Parses the first "cpu" line from /proc/stat and returns (active, total).
    /// active = user + nice + system, total = active + idle + iowait.
    #[cfg(target_os = "linux")]
    fn parse_proc_stat() -> Option<(u64, u64)> {
        let file = fs::File::open("/proc/stat").ok()?;
        let line = BufReader::new(file).lines().next()?.ok()?;
        let mut parts = line.split_whitespace().skip(1);
        let user: u64 = parts.next()?.parse().unwrap_or(0);
        let nice: u64 = parts.next()?.parse().unwrap_or(0);
        let system: u64 = parts.next()?.parse().unwrap_or(0);
        let idle: u64 = parts.next()?.parse().unwrap_or(0);
        let iowait: u64 = parts.next()?.parse().unwrap_or(0);
        let active = user + nice + system;
        let total = active + idle + iowait;
        Some((active, total))
    }

    /// Parses the iowait field from /proc/stat and returns (iowait, total).
    #[cfg(target_os = "linux")]
    fn parse_proc_stat_iowait() -> Option<(u64, u64)> {
        let file = fs::File::open("/proc/stat").ok()?;
        let line = BufReader::new(file).lines().next()?.ok()?;
        let mut parts = line.split_whitespace().skip(1);
        let user: u64 = parts.next()?.parse().unwrap_or(0);
        let nice: u64 = parts.next()?.parse().unwrap_or(0);
        let system: u64 = parts.next()?.parse().unwrap_or(0);
        let idle: u64 = parts.next()?.parse().unwrap_or(0);
        let iowait: u64 = parts.next()?.parse().unwrap_or(0);
        let active = user + nice + system;
        let total = active + idle + iowait;
        Some((iowait, total))
    }

    /// Returns the current CPU load percentage (0-100) using delta measurement.
    /// Reads /proc/stat twice with a 100ms sleep to compute instantaneous load.
    pub fn system_cpu_load() -> u8 {
        #[cfg(target_os = "linux")]
        {
            if let Some((active1, total1)) = Self::parse_proc_stat() {
                thread::sleep(Duration::from_millis(100));
                if let Some((active2, total2)) = Self::parse_proc_stat() {
                    let d_active = active2.saturating_sub(active1);
                    let d_total = total2.saturating_sub(total1);
                    if d_total == 0 {
                        return 0;
                    }
                    return ((d_active as f64 / d_total as f64) * 100.0) as u8;
                }
            }
            0
        }
        #[cfg(not(target_os = "linux"))]
        {
            0
        }
    }

    /// Returns a delta-based iowait ratio (0.0-1.0) over a 100ms window.
    #[cfg(target_os = "linux")]
    fn delta_iowait_ratio() -> f64 {
        if let Some((iowait1, total1)) = Self::parse_proc_stat_iowait() {
            thread::sleep(Duration::from_millis(100));
            if let Some((iowait2, total2)) = Self::parse_proc_stat_iowait() {
                let d_iowait = iowait2.saturating_sub(iowait1);
                let d_total = total2.saturating_sub(total1);
                if d_total == 0 {
                    return 0.0;
                }
                return (d_iowait as f64 / d_total as f64).min(1.0);
            }
        }
        0.0
    }
}

/// C-ABI entry point for environmental entropy.
/// Returns 0-1000 weighted metabolic entropy score.
/// On Linux: uses actual system memory from /proc/meminfo.
/// On non-Linux (or when /proc unreadable): ceiling is 0 (fail-closed),
/// which causes raw_entropy() to return a high score since rss_ratio defaults to 1.0.
/// Callers should not treat a high return value as a definitive exhaustion signal
/// without also checking platform availability.
#[no_mangle]
pub extern "C" fn llmosafe_get_environmental_entropy() -> u16 {
    // Uses a default 50% system RAM ceiling for the global signal
    let guard = ResourceGuard::auto(0.5);
    guard.raw_entropy()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_cpu_load_returns_bounded_value() {
        // Delta-based CPU load should return a value in [0, 100]
        let load = ResourceGuard::system_cpu_load();
        assert!(load <= 100, "CPU load {} should be <= 100", load);
    }

    #[test]
    fn test_system_cpu_load_two_calls_consistent() {
        // Two consecutive calls should both return valid bounded values
        let load1 = ResourceGuard::system_cpu_load();
        let load2 = ResourceGuard::system_cpu_load();
        assert!(load1 <= 100);
        assert!(load2 <= 100);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_proc_stat_returns_some() {
        // On Linux, /proc/stat should be readable
        let result = ResourceGuard::parse_proc_stat();
        assert!(result.is_some(), "/proc/stat should be parseable on Linux");
        let (active, total) = result.expect("checked above");
        assert!(total >= active, "total must be >= active");
        assert!(total > 0, "total should be positive on a running system");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_delta_iowait_ratio_bounded() {
        let ratio = ResourceGuard::delta_iowait_ratio();
        assert!(
            (0.0..=1.0).contains(&ratio),
            "iowait ratio {} should be in [0.0, 1.0]",
            ratio
        );
    }
}
