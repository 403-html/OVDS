//! Manual throughput benchmarks (CPU and GPU). Permanent harness so measuring
//! keys/s does not require churning other modules. Ignored by default (needs a
//! GPU and several seconds); run explicitly:
//!
//!   cargo test -- bench::keygen_throughput --ignored --nocapture
//!
//! Reports a per-backend median; take it as the figure. Numbers are noisy when
//! the GPU is shared (e.g. a live search running). To A/B a shader change, edit
//! the shader, run this, then revert and run again.

#[cfg(test)]
mod tests {
    use crate::gpu::{GpuContext, bench_dispatch_rate};

    /// CPU keygen throughput (keys/s) over ~`secs`, mirroring the generate path's
    /// prefix fast-path: ed25519 keygen + check_prefix_fast on all Rayon threads.
    fn cpu_keygen_rate(secs: f64) -> f64 {
        use crate::crypto::check_prefix_fast;
        use ed25519_dalek::SigningKey;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        use std::time::Instant;

        let attempts = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let start = Instant::now();
        rayon::scope(|s| {
            for _ in 0..rayon::current_num_threads() {
                let attempts = Arc::clone(&attempts);
                let stop = Arc::clone(&stop);
                s.spawn(move |_| {
                    let pattern: &[u8] = b"ovdsbench42";
                    let mut rng = rand::thread_rng();
                    let mut local = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        let key = SigningKey::generate(&mut rng);
                        std::hint::black_box(check_prefix_fast(
                            &key.verifying_key().to_bytes(),
                            pattern,
                        ));
                        local += 1;
                        if local.is_multiple_of(4096) {
                            attempts.fetch_add(4096, Ordering::Relaxed);
                            if start.elapsed().as_secs_f64() >= secs {
                                stop.store(true, Ordering::Relaxed);
                            }
                        }
                    }
                    attempts.fetch_add(local % 4096, Ordering::Relaxed);
                });
            }
        });
        attempts.load(Ordering::Relaxed) as f64 / start.elapsed().as_secs_f64()
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    #[test]
    #[ignore]
    fn keygen_throughput() {
        let cpu = cpu_keygen_rate(5.0);
        eprintln!("BENCH cpu = {:.3}M keys/s", cpu / 1e6);

        match GpuContext::init() {
            Ok(ctx) => {
                let samples = bench_dispatch_rate(&ctx, 5, 40).expect("bench");
                for s in &samples {
                    eprintln!("BENCH gpu sample = {:.3}M keys/s", s / 1e6);
                }
                eprintln!("BENCH gpu = {:.3}M keys/s (median)", median(samples) / 1e6);
            }
            Err(_) => eprintln!("BENCH gpu = n/a (no adapter)"),
        }
    }
}
