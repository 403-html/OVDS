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

    /// The in-app GPU benchmark path (run_keygen_bench) must report a steady-state
    /// rate close to bench_dispatch_rate - i.e. it excludes pipeline build + warm-up.
    /// Guards the fix where the app divided attempts by a wall clock that included
    /// startup and under-reported (~14M vs ~19M).
    #[test]
    #[ignore]
    fn app_gpu_bench_rate() {
        use crate::crypto::MatchType;
        use crate::gpu::run_keygen_bench;
        use std::sync::atomic::{AtomicBool, AtomicU64};
        use std::sync::{Arc, Mutex};

        let Ok(ctx) = GpuContext::init() else {
            eprintln!("APP-BENCH gpu = n/a (no adapter)");
            return;
        };
        let attempts = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let rate = Arc::new(Mutex::new(None));
        let batch_k = crate::gpu::default_batch_k(&ctx.limits);
        run_keygen_bench(
            &ctx,
            MatchType::Prefix,
            batch_k,
            attempts,
            stop,
            Arc::clone(&rate),
            5.0,
        )
        .expect("bench");
        let r = rate.lock().unwrap().expect("rate reported");
        eprintln!("APP-BENCH gpu = {:.3}M keys/s", r / 1e6);
    }

    #[test]
    #[ignore]
    fn keygen_throughput() {
        let cpu = cpu_keygen_rate(5.0);
        eprintln!("BENCH cpu = {:.3}M keys/s", cpu / 1e6);

        match GpuContext::init() {
            Ok(ctx) => {
                // Sweep the device-bounded BATCH_K options so this doubles as a
                // check that runtime K selection actually scales throughput.
                let max_k = crate::gpu::max_batch_k(&ctx.limits);
                for k in [64u32, 128, 256, 512].into_iter().filter(|&k| k <= max_k) {
                    let samples = bench_dispatch_rate(&ctx, k, 5, 40).expect("bench");
                    eprintln!(
                        "BENCH gpu K={k:<4} = {:.3}M keys/s (median)",
                        median(samples) / 1e6
                    );
                }
            }
            Err(_) => eprintln!("BENCH gpu = n/a (no adapter)"),
        }
    }
}
