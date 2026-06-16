# OVDS

Generate a custom `.onion` address for your Tor hidden service. Type a word, pick prefix/suffix/anywhere, and `ovds` searches all CPU cores until it finds a matching Ed25519 keypair.

```
 OVDS  onion vanity domain search                                      v0.3.0
 mode  ›  estimate
┌ SEARCH ───────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  string  ›  fireside█                                                         │
│  chars   ›  ✓ 8 chars                                                         │
│  match   ›  [Prefix]  Suffix  Anywhere    ← →                                 │
│  backend ›  [CPU]  GPU   8 threads        ↑ ↓                                 │
│  example ›  firesideabcdefghijklmnop234567abcdefghij.onion                    │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Features

- Prefix, suffix, or anywhere matching
- CPU backend: all cores via Rayon, prefix fast-path skips SHA3-256 (~2x faster)
- GPU backend (v0.3.0): real ed25519 keygen on the device via wgpu compute, cross-platform (Metal on macOS, Vulkan on Linux, DX12 on Windows)
- Live throughput sparkline, ETA at p50/p95, probabilistic progress gauge
- Side-by-side CPU vs GPU benchmark columns in the time estimates panel
- Saves in Tor's native format, ready to drop into `HiddenServiceDir`

## Backends

Toggle between CPU and GPU from the SEARCH panel with `↑ ↓`. Run `[b]` to benchmark the active backend; both rates are remembered and shown side-by-side in the time estimates table.

| Backend | Status | Notes |
|---------|--------|-------|
| CPU     | full   | ed25519 keygen + SHA3 + base32 on all cores |
| GPU     | full   | ed25519 scalar multiplication on the device (Metal / Vulkan / DX12); host scans the resulting pubkeys against the pattern |

The GPU backend computes `scalar * B` (compressed Edwards pubkeys) directly on the device with a complete twisted-Edwards point formula, then the host matches the pubkeys with the same prefix fast-path used by the CPU backend. The WGSL field and group arithmetic is verified against curve25519-dalek (see `src/fe16_ref.rs` and the `gpu_keygen_matches_dalek` test).

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

Run `[b]` inside the app to benchmark your hardware first.

## Keybinds

| Key | Action |
|-----|--------|
| `a-z` `2-7` | Type pattern (base32 alphabet) |
| `Backspace` | Delete last character |
| `← →` | Cycle match type |
| `↑ ↓` | Toggle backend (CPU ↔ GPU) |
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
