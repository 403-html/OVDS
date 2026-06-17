# Benchmarks

GPU keygen throughput by version, measured on the dev machine (**Apple M3 Pro**)
with the GPU **idle** (a concurrent search roughly halves the rate). Median of
>=5 rounds via:

```sh
cargo test -- bench::keygen_throughput --ignored --nocapture
```

| Version | keys/s (M3 Pro, idle) | Key change |
|---------|-----------------------|------------|
| v0.1.0  | n/a (CPU only)        | initial CPU search |
| v0.2.0  | n/a (CPU only)        | - |
| v0.3.0  | ~45K -> ~510K         | GPU keygen; fixed-base comb scalar-mult |
| v0.3.0+ | ~11.8M                | incremental walk + batched Montgomery inversion (BATCH_K=64) |
| v0.4.0  | ~11.8M                | on-device anywhere match + fast suffix path (throughput unchanged) |
| (unreleased) | TODO: re-measure idle | mixed-base add for the walk step; coalesced scratch; dedicated fe_sq |
