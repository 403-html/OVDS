# OVDS

Generate a custom `.onion` address for your Tor hidden service. Type a word, pick prefix/suffix/anywhere, and `ovds` searches all CPU cores until it finds a matching Ed25519 keypair.

```
 OVDS  onion vanity domain search                                      v0.6.0
 mode  ›  estimate
┌ SEARCH ───────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  string  ›  fireside                                                          │
│  chars   ›  ✓ 8 chars                                                         │
│  match   ›  [Prefix]  Suffix  Anywhere                                        │
│  backend ›  CPU  [GPU]   Metal · Apple M3 Pro                                 │
│▸ batch   ›  64  128  [256]  512   ~1.6 GB GPU   ← →                           │
│  example ›  firesideabcdefghijklmnop234567abcdefghij.onion                    │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Features

- Prefix, suffix, or anywhere matching
- CPU backend: all cores via Rayon, prefix fast-path skips SHA3-256 (~2x faster)
- GPU backend (v0.3.0): real ed25519 keygen on the device via wgpu compute, cross-platform (Metal on macOS, Vulkan on Linux, DX12 on Windows)
- GPU prefix and anywhere matching run on-device at full keygen rate; suffix uses the same incremental kernel with a parallel host scan (v0.4.0)
- GPU prefix/anywhere use a y-only additive walk (v0.6.0): a Tor prefix needs only the y-coordinate, so each candidate is ~5 field muls instead of ~13 (~1.5x more keys/s)
- Selectable GPU batch size (v0.6.0): a larger batch amortises per-thread setup for more keys/s (up to ~2.2x on an M3 Pro); the picker is bounded by the device's memory so it cannot overshoot. See [BENCHMARKS.md](BENCHMARKS.md)
- Live throughput sparkline, ETA at p50/p95, probabilistic progress gauge
- Side-by-side CPU vs GPU benchmark columns in the time estimates panel
- Saves in Tor's native format, ready to drop into `HiddenServiceDir`

## Backends

In the SEARCH panel, move the field cursor with `↑ ↓` and change the selected field with `← →`: this is how you switch between CPU and GPU, pick the match type, and (on GPU) set the batch size. Run `[b]` to benchmark the active backend; both rates are remembered and shown side-by-side in the time estimates table.

| Backend | Status | Notes |
|---------|--------|-------|
| CPU     | full   | ed25519 keygen + SHA3 + base32 on all cores |
| GPU     | full   | ed25519 keygen on the device (Metal / Vulkan / DX12); prefix and anywhere matched on-device, suffix scanned on the host |

The GPU computes `scalar * B` on-device with a complete twisted-Edwards formula. Prefix and anywhere patterns are matched on the device, which compacts only the hits so the readback stays tiny; suffix patterns touch the address tail (which the device does not hash), so they run the same kernel in write-all mode and are scanned on the host with Rayon. Every match is re-verified against the full address on the host. For prefix and anywhere search the kernel computes only the candidate y-coordinate (rebuilding the matched pubkey from its scalar on the host). The WGSL arithmetic is verified against curve25519-dalek (`src/fe16_ref.rs`, `gpu_keygen_matches_dalek`), the batched-inversion and memory-coalescing techniques draw on gECC (Xiong et al., 2024, [arXiv:2501.03245](https://arxiv.org/abs/2501.03245)), and the y-only walk follows AlexanderYastrebov's [onion-vanity-address](https://github.com/AlexanderYastrebov/onion-vanity-address).

GPU keys are stored in Tor's expanded secret-key form (the clamped scalar plus a random signing-nonce prefix), so they drop into `HiddenServiceDir` exactly like CPU-generated keys.

## Install

```sh
cargo build --release
./target/release/ovds
```

Requires Rust 1.85+.

## Keys

Each character multiplies the expected search time by 32 (base32 alphabet size):

| Pattern length | Mean attempts | ~Time at 500K keys/s |
|---------------|---------------|----------------------|
| 3 chars       | 32K           | < 1s                 |
| 4 chars       | 1M            | ~2s                  |
| 5 chars       | 33M           | ~1 min               |
| 6 chars       | 1B            | ~30 min              |
| 7 chars       | 34B           | ~19 hr               |

Run `[b]` inside the app to benchmark your hardware first. See [BENCHMARKS.md](BENCHMARKS.md) for GPU throughput across versions and how to measure it.

## Keybinds

| Key | Action |
|-----|--------|
| `a-z` `2-7` | Type pattern (base32 alphabet; string field) |
| `Backspace` | Delete last character |
| `↑ ↓` | Move field cursor (string / match / backend / batch) |
| `← →` | Change the selected field (match type, backend, or GPU batch size) |
| `Tab` | Switch panel |
| `b` | Benchmark active backend |
| `g` | Start search |
| `s` | Stop search |
| `n` | New search after a find |
| `q` / `Esc` | Quit |

Valid characters are `a-z` and `2-7`. Invalid characters (`0`, `1`, `8`, `9`, etc.) are highlighted in red as you type.

## Output

On a match, a directory is written to your working directory:

```
secretabcdefghijkl.onion/
├── hostname               # full .onion address
├── hs_ed25519_public_key  # Tor v3 public key (64 bytes)
└── hs_ed25519_secret_key  # Tor v3 secret key (96 bytes)
```

Point Tor's `HiddenServiceDir` at this directory and restart Tor. Keep the secret key safe - it is your onion identity.

## How it works

Tor v3 addresses are derived from an Ed25519 public key:

```
payload  = pubkey[32] || checksum[2] || version[1]
checksum = SHA3-256(".onion checksum" || pubkey || version)[0:2]
address  = base32(payload).lower()
```

For prefix patterns up to 51 characters, the prefix encodes entirely from pubkey bytes - the checksum bytes don't appear until position 52+. `OVDS` exploits this by encoding only the necessary pubkey prefix bytes and skipping the SHA3-256 hash, computing the full address only once a match is confirmed.

Suffix and anywhere matching compute the full address on every attempt.

## License

MIT
