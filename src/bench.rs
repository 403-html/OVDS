//! Manual GPU throughput benchmark. Permanent harness so measuring keys/s does
//! not require churning gpu.rs. Ignored by default (needs a GPU and several
//! seconds); run explicitly:
//!
//!   cargo test --  bench::keygen_throughput --ignored --nocapture
//!
//! Reports one keys/s sample per round; take the median. Numbers are noisy when
//! the GPU is shared (e.g. a live search running). To A/B a shader change, edit
//! the shader, run this, then revert and run again.

#[cfg(test)]
mod tests {
    use crate::gpu::{bench_dispatch_rate, GpuContext};

    #[test]
    #[ignore]
    fn keygen_throughput() {
        let Ok(ctx) = GpuContext::init() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let mut samples = bench_dispatch_rate(&ctx, 5, 40).expect("bench");
        for s in &samples {
            eprintln!("BENCH keys/s = {:.3}M", s / 1e6);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "BENCH median = {:.3}M keys/s",
            samples[samples.len() / 2] / 1e6
        );
    }
}
