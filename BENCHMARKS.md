# Benchmarks

Keygen throughput (keys/s) for every released OVDS version, measured on one
machine with one method so the numbers are directly comparable.

## Environment

| | |
|---|---|
| Machine | Apple M3 Pro (11 logical cores) |
| GPU | Metal (wgpu 22) |
| Build | `--release` (opt-level 3, LTO, codegen-units 1), rustc 1.94.1 |
| Date / state | 2026-06-18, GPU idle |

## Method

"keys/s" is raw ed25519 keypairs generated per second (the denominator behind the
app's ETAs), not solved addresses. Both backends run the same job: prefix mode
against a never-matching 11-char pattern, so the timer is pure generation plus
on-device prefix screening.

- CPU: `SigningKey::generate` + pubkey serialization on all Rayon threads for 5 s.
  The prefix search path skips SHA3-256, so this is also the realistic CPU prefix
  rate.
- GPU: median of 5 rounds of 40 `keygen_dispatch` calls (warm-up excluded). Each
  dispatch generates `threads * BATCH_K` = 16384 * BATCH_K candidates and screens
  them on-device. From v0.6.0 BATCH_K is selectable at runtime (bounded by the
  device's allocation limits), so GPU numbers now state the batch size. The
  per-version table below holds BATCH_K = 64 fixed so the version-over-version
  comparison stays apples-to-apples; the batch-size sweep is in its own table.

GPU numbers require an idle device (a live search roughly halves the rate).
Historical versions were measured by injecting this harness into a throwaway
`git worktree` per tag; nothing was committed to tagged source. For the current
version the harness is in-tree:

```sh
cargo test --release -- bench::keygen_throughput --ignored --nocapture
```

## Results

| Version | CPU keys/s | GPU keys/s (BATCH_K=64) | Key change |
|---------|-----------|------------|------------|
| v0.1.0  | 0.70M | n/a | CPU-only search |
| v0.2.0  | 0.70M | n/a | wgpu init + pipeline, but keygen still on CPU (GPU ran only a SHA-256 benchmark) |
| v0.3.0  | 0.67M | 11.96M | real ed25519 keygen in WGSL: incremental walk + batched Montgomery inversion (BATCH_K=64) |
| v0.4.0  | 0.70M | 11.91M | on-device anywhere + fast suffix path (features; prefix throughput unchanged) |
| v0.5.0  | 0.69M | 12.86M | mixed-base walk add, coalesced scratch, dedicated `fe_sq` (~7-8% over v0.4.0) |
| v0.6.0  | 0.68M | 19.65M | y-only walk for prefix/anywhere (dual-addition + left-fold division); 1.53x over v0.5.0 |

- CPU is flat at ~0.70M every version (the primitive never changed; 0.67M at
  v0.3.0 is run-to-run noise).
- The 45K -> 510K -> ~11.9M GPU arc all happened *within* v0.3.0 development; the
  tag itself already ships the batched walk and measures ~11.96M.
- v0.5.0's lift came from arithmetic/memory work informed by gECC (see
  References); the earlier contended ~7.1M is superseded by this idle run.
- v0.6.0 computes only the candidate y-coordinate (all a Tor prefix needs):
  ~5 field muls/candidate vs ~13, and half the scratch. End-to-end 1.53x rather
  than the ~1.9x mul ratio because the per-thread comb (P0 = s0*B) is shared by
  both paths. Suffix is unchanged (write-all still needs the x-sign bit). From
  AlexanderYastrebov's onion-vanity-address (see References).
- Single machine (M3 Pro / Metal); the version-over-version shape is the point.
  Suffix mode reads back all candidates for a host scan, so its rate is lower and
  not in this table.

## Batch size (v0.6.0+)

BATCH_K is the number of candidates each GPU thread walks per dispatch. Larger
batches amortise the per-thread comb (P0 = s0*B) and the one inversion over more
candidates, raising throughput, at the cost of linearly larger GPU buffers. The
in-app picker only offers powers of two the device can allocate; the default is
256. GPU prefix (y-only), M3 Pro, idle:

| BATCH_K | GPU keys/s | vs 64 | ~GPU memory |
|---------|-----------|-------|-------------|
| 64      | 19.64M | 1.00x | 0.4 GB |
| 128     | 28.34M | 1.44x | 0.8 GB |
| 256 (default) | 36.42M | 1.85x | 1.6 GB |
| 512     | 42.46M | 2.16x | 3.2 GB |

- 512 is the device max on this machine (scratch hits the 4 GiB storage-binding
  limit). Bigger GPUs may allow more; the picker caps at what fits.
- Diminishing returns past 256: each doubling adds less because the fixed
  per-thread comb is increasingly amortised and memory bandwidth starts to bind.
- Memory is dominated by the write-all scratch buffer (threads * BATCH_K * 256 B);
  output and staging add ~half that again.

## References

- Q. Xiong, W. Ma, X. Shi, Y. Zhou, H. Jin, K. Huang, H. Wang, and Z. Wang, "gECC: A GPU-based high-throughput framework for Elliptic Curve Cryptography," 2024, arXiv:2501.03245.
- cathugger, "mkp224o," https://github.com/cathugger/mkp224o. Reference CPU Tor
  v3 generator; same additive-walk + batched-inversion algorithm OVDS runs on the
  GPU.
- A. Yastrebov, "onion-vanity-address," https://github.com/AlexanderYastrebov/onion-vanity-address.
  Source of the y-only walk (dual addition formula, eprint 2008/522; simultaneous
  field division, eprint 2008/199) that OVDS v0.6.0 ports to the GPU.
