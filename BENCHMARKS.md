# Benchmarks

Keygen throughput (keys per second) for every released version of OVDS, measured
on one machine with one method so the numbers are directly comparable.

## Environment

| | |
|---|---|
| Machine | Apple M3 Pro (11 logical cores reported to Rayon) |
| GPU backend | Metal (wgpu 22) |
| Build | `--release` profile (opt-level 3, LTO, codegen-units 1) |
| Toolchain | rustc 1.94.1 |
| Date measured | 2026-06-17 |
| GPU state | idle (no concurrent search) |

## What "keys/s" means here

The figure is raw keypair generation throughput: how many ed25519 keypairs the
backend produces per second. It is the denominator behind every ETA the app
shows. It is not the same as solved addresses per second, which also depends on
pattern length.

Both backends are measured doing the same job: generate keys in prefix mode
against a never-matching 11 character pattern (`ovdsbench42`), so the timer
captures pure generation and on-device prefix screening with no match-readback
or host bookkeeping skewing it.

- CPU: `SigningKey::generate` plus public-key serialization, run on every Rayon
  thread. In prefix mode the CPU search path skips SHA3-256 (the prefix encodes
  straight from pubkey bytes), so raw keygen is also the realistic CPU prefix
  search rate.
- GPU: one `keygen_dispatch` in prefix mode produces `threads * BATCH_K` =
  16384 * 64 = 1,048,576 candidates and screens them on the device, returning
  only the (here, zero) hits.

## How we measure

The harness is deliberately tiny and identical across versions. For historical
versions it was injected into a throwaway `git worktree` for that tag, built in
release, and run; nothing was committed to the tagged source.

CPU probe (runs for 5 s on all threads, reports keys/s):

```rust
let key = SigningKey::generate(&mut rng);
std::hint::black_box(key.verifying_key().to_bytes());
// counted across rayon::current_num_threads() threads for 5 s
```

GPU probe (median of 7 rounds of 40 dispatches; warm-up dispatch excluded so the
comb-table build and pipeline compile are not in the timer):

```rust
let per = pipe.threads as u64 * BATCH_K as u64;     // 1,048,576
keygen_dispatch(&ctx, &pipe, &seed, 0, MODE_PREFIX, &syms)?; // warm up, untimed
// then time n=40 dispatches per round, keys/s = per * n / elapsed
```

Why these choices:

- Median of several rounds, not the mean, because GPU samples are right-skewed
  (the occasional slow round from scheduler noise should not drag the figure).
- Idle GPU. A live search roughly halves the GPU rate, so all GPU numbers below
  are with nothing else on the device.
- Never-matching pattern, so no round does extra work compacting hits.

### Reproducing the current version

The permanent harness lives in the tree and needs no injection:

```sh
cargo test --release -- bench::keygen_throughput --ignored --nocapture
```

It prints a per-backend median; take that as the figure.

## Results

Measured 2026-06-17 on the machine above, idle, release, median of 7 GPU rounds.

| Version | CPU keys/s | GPU keys/s | Key change |
|---------|-----------|------------|------------|
| v0.1.0  | 0.70M | n/a | CPU-only search, no GPU backend |
| v0.2.0  | 0.70M | n/a | GPU device init + wgpu pipeline, but keygen still on CPU (the GPU pipeline only ran an iterated SHA-256 benchmark) |
| v0.3.0  | 0.67M | 11.96M | real ed25519 keygen in WGSL: incremental walk + batched Montgomery inversion (BATCH_K=64), on-device prefix screening |
| v0.4.0  | 0.70M | 11.91M | on-device anywhere matching + fast suffix path (correctness/feature work; prefix throughput unchanged) |
| v0.5.0  | 0.69M | 12.86M | mixed-base addition for the walk step, coalesced scratch, dedicated `fe_sq` (~7-8% over v0.4.0) |

CPU keygen is effectively flat at ~0.70M keys/s across every version (the 0.67M
at v0.3.0 is within run-to-run noise); the keygen primitive never changed, only
the search and backend code around it.

## Per-version notes

- v0.1.0 / v0.2.0: no GPU keygen exists, so GPU keys/s is n/a. v0.2.0 shipped the
  wgpu device init and a working compute pipeline, but it powered only an
  iterated SHA-256 benchmark used for the time-estimate columns; pressing
  generate on the GPU backend fell back to CPU keygen.
- v0.3.0: the whole GPU keygen arc landed inside this release. Early development
  builds went roughly 45K -> 510K -> ~11.9M keys/s as the kernel moved from a
  fixed-base comb scalar-mult to the incremental walk with batched Montgomery
  inversion (BATCH_K=64). The tagged v0.3.0 already includes the batched walk, so
  it measures ~11.96M, not the intermediate numbers. Treat the 45K and 510K
  figures as historical context for the in-progress kernel, not as released
  versions.
- v0.4.0: added on-device anywhere matching and a fast suffix path. These are
  correctness and feature changes; prefix-mode generation throughput is
  unchanged from v0.3.0 (11.91M vs 11.96M is noise).
- v0.5.0: the field and group arithmetic work (mixed-base addition for the walk
  step, coalesced scratch buffers, a dedicated `fe_sq` squaring routine) gives a
  real, repeatable ~7-8% lift to 12.86M. The earlier "needs an idle re-measure"
  caveat is resolved by this run; the previously observed ~7.1M was contended by
  a live search.

## Caveats

- These are single-machine numbers (Apple M3 Pro / Metal). Absolute throughput
  will differ on other GPUs; the version-over-version shape is the point.
- The GPU figure counts candidates generated and screened on-device in prefix
  mode. Suffix mode reads back every candidate for a host scan, so its
  user-visible rate is lower; that is a separate measurement, not this table.
- GPU samples are noisy when the device is shared. Re-measure idle before
  quoting a new number.
