use crate::crypto::{ADDRESS_LEN, Backend, MAX_DEVICE_PREFIX, MatchType};

/// Probability of a single keypair matching the pattern.
///
/// `Anywhere` is backend-dependent: the GPU on-device matcher only scans the
/// pubkey region (first `MAX_DEVICE_PREFIX` chars), so it has fewer start
/// positions than the CPU / write-all paths, which scan the full address. Using
/// the full address for a GPU anywhere search would make the ETA optimistic.
pub fn match_probability(pattern_len: usize, match_type: &MatchType, backend: &Backend) -> f64 {
    if pattern_len == 0 {
        return 1.0;
    }
    let p_single = 1.0_f64 / 32.0_f64.powi(pattern_len as i32);
    match match_type {
        MatchType::Prefix | MatchType::Suffix => p_single,
        MatchType::Anywhere => {
            // Chars actually scanned. GPU on-device matcher is capped at
            // MAX_DEVICE_PREFIX; longer patterns there fall back to write-all
            // (full address), as does the CPU backend.
            let window = match backend {
                Backend::Gpu if pattern_len <= MAX_DEVICE_PREFIX => MAX_DEVICE_PREFIX,
                _ => ADDRESS_LEN,
            };
            let positions = window.saturating_sub(pattern_len) + 1;
            // union-bound approximation (good for small p)
            (positions as f64 * p_single).min(1.0)
        }
    }
}

/// Expected keypairs needed (mean of geometric distribution).
pub fn expected_attempts(p: f64) -> f64 {
    if p <= 0.0 { f64::INFINITY } else { 1.0 / p }
}

/// Quantile of geometric distribution: smallest n such that CDF(n) >= q.
/// CDF(n) = 1 - (1-p)^n  →  n = ceil(ln(1-q) / ln(1-p))
///
/// For very small p (< machine epsilon ~2.2e-16), (1-p) rounds to 1.0 in f64
/// and ln(1.0) = 0, causing division by zero. Use ln(1-p) ≈ -p instead, which
/// is accurate to O(p²) and exact in the limit.
pub fn quantile_attempts(p: f64, q: f64) -> f64 {
    if p <= 0.0 || q <= 0.0 {
        return f64::INFINITY;
    }
    if p >= 1.0 || q >= 1.0 {
        return 1.0;
    }
    let ln_1mp = if p < 1e-10 { -p } else { (1.0_f64 - p).ln() };
    let n = (1.0_f64 - q).ln() / ln_1mp;
    n.ceil()
}

/// Probabilistic progress: CDF(n) = 1 - (1-p)^n = 1 - exp(n·ln(1-p)).
/// Uses exp/ln form to avoid powi's i32 exponent cap.
pub fn cdf(attempts: u64, p: f64) -> f64 {
    if p >= 1.0 {
        return 1.0;
    }
    let ln_1mp = if p < 1e-10 { -p } else { (1.0_f64 - p).ln() };
    1.0 - (ln_1mp * attempts as f64).exp()
}

pub fn format_duration(secs: f64) -> String {
    if !secs.is_finite() {
        return "∞".into();
    }
    if secs < 1.0 {
        return format!("{:.0}ms", secs * 1000.0);
    }
    let secs_u = secs as u64;
    if secs_u < 60 {
        return format!("{}s", secs_u);
    }
    if secs_u < 3600 {
        return format!("{}m {}s", secs_u / 60, secs_u % 60);
    }
    if secs_u < 86400 {
        return format!("{}h {}m", secs_u / 3600, (secs_u % 3600) / 60);
    }
    if secs_u < 86400 * 365 {
        return format!("{}d {}h", secs_u / 86400, (secs_u % 86400) / 3600);
    }
    let years = secs / 86400.0 / 365.25;
    if years < 1e4 {
        return format!("{:.1}yr", years);
    }
    format!("{:.2e}yr", years)
}

pub fn format_count(n: f64) -> String {
    if n >= 1e18 {
        return format!("{:.2e}", n);
    }
    if n >= 1e15 {
        return format!("{:.1}P", n / 1e15);
    }
    if n >= 1e12 {
        return format!("{:.1}T", n / 1e12);
    }
    if n >= 1e9 {
        return format!("{:.1}B", n / 1e9);
    }
    if n >= 1e6 {
        return format!("{:.1}M", n / 1e6);
    }
    if n >= 1e3 {
        return format!("{:.1}K", n / 1e3);
    }
    format!("{:.0}", n)
}

pub fn format_rate(kps: f64) -> String {
    format!("{}/s", format_count(kps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::{ADDRESS_LEN, Backend, MAX_DEVICE_PREFIX, MatchType};

    #[test]
    fn quantile_no_underflow_for_long_patterns() {
        // 11-char prefix: p = 1/32^11 = 1/2^55 ≈ 2.78e-17 < f64 epsilon → was broken
        let p = match_probability(11, &MatchType::Prefix, &Backend::Cpu);
        let q50 = quantile_attempts(p, 0.50);
        assert!(
            q50.is_finite() && q50 > 1e15,
            "p50 for 11-char pattern must be finite and huge, got {}",
            q50
        );

        // 20-char prefix: p = 1/32^20, astronomically small
        let p20 = match_probability(20, &MatchType::Prefix, &Backend::Cpu);
        let q01 = quantile_attempts(p20, 0.01);
        assert!(
            q01.is_finite(),
            "quantile must be finite even for 20-char pattern"
        );
        assert!(q01 > 0.0, "quantile must be positive");
    }

    #[test]
    fn cdf_correct_for_large_attempts() {
        // CDF at the median should be ~0.5 (within rounding)
        let p = 1.0 / 1_000_000.0;
        let n = quantile_attempts(p, 0.50) as u64;
        let c = cdf(n, p);
        assert!(
            (c - 0.5).abs() < 0.01,
            "CDF at median should be ~0.5, got {}",
            c
        );
    }

    #[test]
    fn format_duration_branches() {
        assert_eq!(format_duration(0.123), "123ms");
        assert_eq!(format_duration(5.0), "5s");
        assert_eq!(format_duration(125.0), "2m 5s");
        assert_eq!(format_duration(3700.0), "1h 1m");
        assert_eq!(format_duration(90_000.0), "1d 1h");
        let years = format_duration(86400.0 * 365.25 * 2.5);
        assert!(years.ends_with("yr"), "expected yr suffix, got {years}");
        let huge = format_duration(86400.0 * 365.25 * 1e6);
        assert!(
            huge.contains('e'),
            "expected scientific notation, got {huge}"
        );
        assert_eq!(format_duration(f64::INFINITY), "\u{221e}");
    }

    #[test]
    fn format_count_branches() {
        assert_eq!(format_count(7.0), "7");
        assert_eq!(format_count(1_500.0), "1.5K");
        assert_eq!(format_count(2_500_000.0), "2.5M");
        assert_eq!(format_count(1.2e9), "1.2B");
        assert_eq!(format_count(3.4e12), "3.4T");
        assert_eq!(format_count(5.6e15), "5.6P");
        assert!(format_count(1.0e20).contains('e'));
    }

    #[test]
    fn expected_attempts_edge_cases() {
        assert_eq!(expected_attempts(0.0), f64::INFINITY);
        assert_eq!(expected_attempts(-1.0), f64::INFINITY);
        assert!((expected_attempts(0.25) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn match_probability_scales_with_length() {
        let p3 = match_probability(3, &MatchType::Prefix, &Backend::Cpu);
        let p4 = match_probability(4, &MatchType::Prefix, &Backend::Cpu);
        // each char divides probability by 32
        assert!((p3 / p4 - 32.0).abs() < 1e-12);
        assert_eq!(match_probability(0, &MatchType::Prefix, &Backend::Cpu), 1.0);
    }

    #[test]
    fn match_probability_anywhere_uses_positions() {
        // CPU anywhere scans the full address: positions = ADDRESS_LEN - len + 1.
        let p_single = match_probability(3, &MatchType::Prefix, &Backend::Cpu);
        let p_any = match_probability(3, &MatchType::Anywhere, &Backend::Cpu);
        let positions = ADDRESS_LEN - 3 + 1;
        assert!((p_any - (positions as f64 * p_single)).abs() < 1e-15);
        // very short pattern saturates to 1.0
        assert_eq!(
            match_probability(0, &MatchType::Anywhere, &Backend::Cpu),
            1.0
        );
    }

    #[test]
    fn match_probability_anywhere_gpu_window_is_smaller() {
        // GPU on-device matcher scans only MAX_DEVICE_PREFIX chars, so it has
        // fewer start positions than CPU -> lower probability, higher ETA.
        let p_single = match_probability(3, &MatchType::Prefix, &Backend::Gpu);
        let p_gpu = match_probability(3, &MatchType::Anywhere, &Backend::Gpu);
        let positions = MAX_DEVICE_PREFIX - 3 + 1;
        assert!((p_gpu - (positions as f64 * p_single)).abs() < 1e-15);
        // GPU anywhere must be strictly less likely than CPU anywhere (8 fewer chars)
        let p_cpu = match_probability(3, &MatchType::Anywhere, &Backend::Cpu);
        assert!(
            p_gpu < p_cpu,
            "GPU window must yield lower p ({p_gpu} vs {p_cpu})"
        );
        // A pattern longer than the device window falls back to full-address scan.
        let p_long_gpu = match_probability(50, &MatchType::Anywhere, &Backend::Gpu);
        let p_long_cpu = match_probability(50, &MatchType::Anywhere, &Backend::Cpu);
        assert_eq!(
            p_long_gpu, p_long_cpu,
            "len > MAX_DEVICE_PREFIX uses full address"
        );
    }

    #[test]
    fn quantile_monotone_in_q() {
        let p = 1.0 / 1024.0;
        let q50 = quantile_attempts(p, 0.50);
        let q95 = quantile_attempts(p, 0.95);
        assert!(q95 > q50, "p95 must be larger than p50 ({q50} vs {q95})");
    }

    #[test]
    fn format_rate_has_per_second() {
        assert!(format_rate(1.5e6).ends_with("/s"));
    }
}
