use crate::crypto::{ADDRESS_LEN, MatchType};

/// Probability of a single keypair matching the pattern.
pub fn match_probability(pattern_len: usize, match_type: &MatchType) -> f64 {
    if pattern_len == 0 {
        return 1.0;
    }
    let p_single = 1.0_f64 / 32.0_f64.powi(pattern_len as i32);
    match match_type {
        MatchType::Prefix | MatchType::Suffix => p_single,
        MatchType::Anywhere => {
            // number of positions the pattern can start at
            let positions = ADDRESS_LEN.saturating_sub(pattern_len) + 1;
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
    use crate::crypto::MatchType;

    #[test]
    fn quantile_no_underflow_for_long_patterns() {
        // 11-char prefix: p = 1/32^11 = 1/2^55 ≈ 2.78e-17 < f64 epsilon → was broken
        let p = match_probability(11, &MatchType::Prefix);
        let q50 = quantile_attempts(p, 0.50);
        assert!(
            q50.is_finite() && q50 > 1e15,
            "p50 for 11-char pattern must be finite and huge, got {}",
            q50
        );

        // 20-char prefix: p = 1/32^20, astronomically small
        let p20 = match_probability(20, &MatchType::Prefix);
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
}
