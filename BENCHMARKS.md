# Benchmarks

CPU vs GPU keygen throughput by version, on the dev machine (**Apple M3 Pro**),
prefix mode, keys/s. Measure with:

```sh
cargo test -- bench::keygen_throughput --ignored --nocapture
```

GPU numbers should be taken with the GPU **idle** (a concurrent search roughly
halves the rate) as the median of >=5 rounds. The CPU path (ed25519 keygen +
prefix fast-path on all Rayon cores) is unchanged across versions, so it is a
flat baseline; all GPU gains are the moving figure.

| Version | CPU keys/s | GPU keys/s | Key change |
|---------|------------|------------|------------|
| v0.1.0  | ~55K       | n/a        | CPU-only search |
| v0.2.0  | ~55K       | n/a        | - |
| v0.3.0  | ~55K       | ~45K -> ~510K | GPU keygen; fixed-base comb scalar-mult |
| v0.3.0+ | ~55K       | ~11.8M     | incremental walk + batched Montgomery inversion (BATCH_K=64) |
| v0.4.0  | ~55K       | ~11.8M     | on-device anywhere match + fast suffix path (throughput unchanged) |
| (unreleased) | ~55K  | TODO: re-measure idle | mixed-base add for the walk step; coalesced scratch; dedicated fe_sq |

Notes:

- CPU ~55K and the unreleased GPU rows were measured 2026-06-17 with a live search
  running. CPU is on-device-independent so it is reliable; the unreleased GPU
  number was contended (~7.1M observed) and needs an idle re-measure before release.
- The v0.3.0 GPU arc (~45K -> ~510K -> ~11.8M) is from earlier idle runs on this
  machine; treat as approximate historical context.
