# OVDS

Generate a custom `.onion` address for your Tor hidden service. Type a word, pick prefix/suffix/anywhere, and `ovds` searches all CPU cores until it finds a matching Ed25519 keypair.

```
 OVDS  onion vanity domain search                                      v0.1.0
 mode  ›  estimate
┌ PATTERN ──────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  string  ›  fireside█                                                         │
│                                                                               │
│  chars   ›  ✓ 8 chars                                                         │
│  match   ›  [Prefix]  Suffix  Anywhere  <- ->                                 │
│  example ›  firesideabcdefghijklmnop234567abcdefghij.onion                   │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Features

- Prefix, suffix, or anywhere matching
- All CPU cores used automatically via Rayon
- Prefix fast-path: skips SHA3-256 entirely for prefix patterns (~2x faster)
- Live throughput sparkline, ETA at p50/p95, probabilistic progress gauge
- Built-in benchmark to measure actual keys/s on your hardware
- Saves in Tor's native format, ready to drop into `HiddenServiceDir`

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
| `Tab` | Switch panel |
| `b` | Benchmark key generation speed |
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
