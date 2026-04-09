//! llmosafe_body — Physical resource guard (the autonomic nervous system).
//!
//! Monitors RSS memory, maps to CognitiveEntropy, triggers KernelError
//! when physical resource thresholds are crossed.
//!
//! This module requires `std` (reads `/proc` on Linux, Win32 API on Windows).
//! Disabled in `no_std` builds.

use crate::llmosafe_kernel::{KernelError, Synapse};
use std::fs;
use std::thread;
use std::time::Duration;

/// EnvironmentalVitals tracks system-level metabolic signals.
#[derive(Debug, Clone, Default)]
pub struct EnvironmentalVitals {
    pub iowait: u64,
    pub load_avg: f64,
}

impl EnvironmentalVitals {
    /// Captures current system vitals.
    pub fn capture() -> Self {
        Self {
            iowait: Self::read_iowait(),
            load_avg: Self::read_loadavg(),
        }
    }

    #[cfg(target_os = "linux")]
    fn read_iowait() -> u64 {
        if let Ok(stat) = fs::read_to_string("/proc/stat") {
            if let Some(line) = stat.lines().next() {
                let parts: Vec<&str> = line.split_whitespace().skip(1).collect();
                if parts.len() >= 5 {
                    return parts[4].parse().unwrap_or(0);
                }
            }
        }
        0
    }

    #[cfg(not(target_os = "linux"))]
    fn read_iowait() -> u64 {
        0
    }

    #[cfg(target_os = "linux")]
    fn read_loadavg() -> f64 {
        if let Ok(loadavg) = fs::read_to_string("/proc/loadavg") {
            if let Some(first_part) = loadavg.split_whitespace().next() {
                return first_part.parse().unwrap_or(0.0);
            }
        }
        0.0
    }

    #[cfg(not(target_os = "linux"))]
    fn read_loadavg() -> f64 {
        0.0
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
    pub fn raw_entropy(&self) -> u16 {
        let current_rss = Self::current_rss_bytes();
        let rss_ratio = if self.memory_ceiling_bytes > 0 {
            (current_rss as f64 / self.memory_ceiling_bytes as f64).min(1.0)
        } else {
            1.0
        };

        let vitals = EnvironmentalVitals::capture();

        // Map Load Average to 0-1.0 (capped at 10.0 for scaling)
        let load_ratio = (vitals.load_avg / 10.0).min(1.0);

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
        let current_rss = Self::current_rss_bytes();
        if self.memory_ceiling_bytes == 0 {
            return 100;
        }
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
        let current_rss = Self::current_rss_bytes();
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

    /// Blocks until resources are safe, automatically honoring Escalate/Halt cooldowns.
    ///
    /// ⚠ BLOCKING: Reads /proc/stat multiple times with sleeps.
    /// Do NOT call in async contexts without spawn_blocking.
    #[cfg(feature = "std")]
    pub fn check_blocking(&self) -> Result<Synapse, KernelError> {
        use crate::llmosafe_integration::{EscalationPolicy, SafetyDecision};
        use std::thread;
        use std::time::Duration;

        let policy = EscalationPolicy::default();
        loop {
            let entropy = self.raw_entropy();
            let decision = policy.decide(entropy, 0, false);
            match decision {
                SafetyDecision::Proceed | SafetyDecision::Warn(_) => {
                    return self.check();
                }
                SafetyDecision::Escalate { cooldown_ms, .. } => {
                    thread::sleep(Duration::from_millis(cooldown_ms as u64));
                }
                SafetyDecision::Halt(_, cooldown_ms) => {
                    thread::sleep(Duration::from_millis(cooldown_ms as u64));
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
        use std::thread;
        use std::time::Duration;

        let policy = EscalationPolicy::default();
        loop {
            if std::time::Instant::now() >= deadline {
                return Err(KernelError::DeadlineExceeded);
            }
            let entropy = self.raw_entropy();
            let decision = policy.decide(entropy, 0, false);
            match decision {
                SafetyDecision::Proceed | SafetyDecision::Warn(_) => {
                    return self.check();
                }
                SafetyDecision::Escalate { cooldown_ms, .. } => {
                    thread::sleep(Duration::from_millis(cooldown_ms as u64));
                }
                SafetyDecision::Halt(_, cooldown_ms) => {
                    thread::sleep(Duration::from_millis(cooldown_ms as u64));
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
            (usage.ru_maxrss as usize) * 1024
        } else {
            Self::read_rss_from_proc()
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

    /// Helper: parse RSS from /proc/self/status (Linux fallback)
    #[cfg(target_os = "linux")]
    fn read_rss_from_proc() -> usize {
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(size_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = size_str.parse::<usize>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
        0
    }

    #[cfg(not(target_os = "linux"))]
    fn read_rss_from_proc() -> usize {
        0
    }

    /// Returns system memory in bytes.
    #[cfg(target_os = "linux")]
    pub fn system_memory_bytes() -> usize {
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(size_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = size_str.parse::<usize>() {
                            return kb * 1024;
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
            usize::MAX / 2
        };
        Self::new(ceiling)
    }

    /// Parses the first "cpu" line from /proc/stat and returns (active, total).
    /// active = user + nice + system, total = active + idle + iowait.
    #[cfg(target_os = "linux")]
    fn parse_proc_stat() -> Option<(u64, u64)> {
        let stat = fs::read_to_string("/proc/stat").ok()?;
        let line = stat.lines().next()?;
        let parts: Vec<&str> = line.split_whitespace().skip(1).collect();
        if parts.len() >= 5 {
            let user: u64 = parts[0].parse().unwrap_or(0);
            let nice: u64 = parts[1].parse().unwrap_or(0);
            let system: u64 = parts[2].parse().unwrap_or(0);
            let idle: u64 = parts[3].parse().unwrap_or(0);
            let iowait: u64 = parts[4].parse().unwrap_or(0);
            let active = user + nice + system;
            let total = active + idle + iowait;
            Some((active, total))
        } else {
            None
        }
    }

    /// Parses the iowait field from /proc/stat and returns (iowait, total).
    #[cfg(target_os = "linux")]
    fn parse_proc_stat_iowait() -> Option<(u64, u64)> {
        let stat = fs::read_to_string("/proc/stat").ok()?;
        let line = stat.lines().next()?;
        let parts: Vec<&str> = line.split_whitespace().skip(1).collect();
        if parts.len() >= 5 {
            let user: u64 = parts[0].parse().unwrap_or(0);
            let nice: u64 = parts[1].parse().unwrap_or(0);
            let system: u64 = parts[2].parse().unwrap_or(0);
            let idle: u64 = parts[3].parse().unwrap_or(0);
            let iowait: u64 = parts[4].parse().unwrap_or(0);
            let active = user + nice + system;
            let total = active + idle + iowait;
            Some((iowait, total))
        } else {
            None
        }
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
        let (active, total) = result.unwrap();
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
