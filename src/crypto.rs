use data_encoding::BASE32_NOPAD;
use ed25519_dalek::SigningKey;
use sha2::Sha512;
use sha3::{Digest, Sha3_256};
use std::fs;
use std::path::PathBuf;

pub fn derive_address(key: &SigningKey) -> String {
    let mut buf = [0u8; 56];
    derive_address_into(key, &mut buf);
    String::from_utf8(buf.to_vec()).unwrap()
}

/// Fill a pre-allocated 56-byte buffer - no heap allocation.
pub fn derive_address_into(key: &SigningKey, buf: &mut [u8; 56]) {
    let pubkey = key.verifying_key().to_bytes();
    build_payload_and_encode(&pubkey, buf);
}

/// Build a full 56-char Tor v3 address from a raw 32-byte compressed pubkey.
/// Used by the GPU keygen path where we have a pubkey but no SigningKey.
pub fn address_from_pubkey(pubkey: &[u8; 32]) -> [u8; 56] {
    let mut buf = [0u8; 56];
    build_payload_and_encode(pubkey, &mut buf);
    buf
}

fn build_payload_and_encode(pubkey: &[u8; 32], buf: &mut [u8; 56]) {
    let version: u8 = 0x03;
    let mut hasher = Sha3_256::new();
    hasher.update(b".onion checksum");
    hasher.update(pubkey);
    hasher.update([version]);
    let hash = hasher.finalize();

    let mut payload = [0u8; 35];
    payload[..32].copy_from_slice(pubkey);
    payload[32] = hash[0];
    payload[33] = hash[1];
    payload[34] = version;

    BASE32_NOPAD.encode_mut(&payload, buf);
    for b in buf.iter_mut() {
        *b = b.to_ascii_lowercase();
    }
}

/// Prefix fast-path: for pattern_len ≤ 51 the prefix chars come entirely from
/// pubkey bytes, so we can skip the SHA3-256 checksum computation entirely.
/// Returns false (forcing full derivation) for patterns longer than 51 chars.
pub fn check_prefix_fast(pubkey: &[u8; 32], pattern: &[u8]) -> bool {
    let n = pattern.len();
    if n == 0 {
        return true;
    }
    // ceil(n*5/8) payload bytes needed; for n≤51 these are all pubkey bytes.
    // For n>51 the 52nd+ chars involve checksum bytes - fall back to full path.
    let bytes_needed = (n * 5).div_ceil(8);
    if bytes_needed > 32 {
        return false;
    }
    let chars_encoded = BASE32_NOPAD.encode_len(bytes_needed);
    // 32 pubkey bytes encode to at most 52 base32 chars
    let mut enc = [0u8; 56];
    BASE32_NOPAD.encode_mut(&pubkey[..bytes_needed], &mut enc[..chars_encoded]);
    for b in enc[..chars_encoded].iter_mut() {
        *b = b.to_ascii_lowercase();
    }
    // Only compare the first n chars; the last encoded char may have zero-padded
    // bits that differ from the real address, but those bits are never in the slice.
    enc[..n] == *pattern
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Backend {
    Cpu,
    Gpu,
}

impl Backend {
    pub fn label(&self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            Backend::Gpu => "GPU",
        }
    }

    pub fn toggle(&self) -> Self {
        match self {
            Backend::Cpu => Backend::Gpu,
            Backend::Gpu => Backend::Cpu,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum MatchType {
    Prefix,
    Suffix,
    Anywhere,
}

impl MatchType {
    pub fn label(&self) -> &str {
        match self {
            MatchType::Prefix => "Prefix",
            MatchType::Suffix => "Suffix",
            MatchType::Anywhere => "Anywhere",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            MatchType::Prefix => MatchType::Suffix,
            MatchType::Suffix => MatchType::Anywhere,
            MatchType::Anywhere => MatchType::Prefix,
        }
    }

    pub fn prev(&self) -> Self {
        match self {
            MatchType::Prefix => MatchType::Anywhere,
            MatchType::Suffix => MatchType::Prefix,
            MatchType::Anywhere => MatchType::Suffix,
        }
    }
}

pub const VALID_CHARS: &str = "abcdefghijklmnopqrstuvwxyz234567";
pub const ADDRESS_LEN: usize = 56;

pub fn validate_pattern(s: &str) -> (bool, Vec<usize>) {
    let invalid: Vec<usize> = s
        .chars()
        .enumerate()
        .filter(|(_, c)| !VALID_CHARS.contains(*c))
        .map(|(i, _)| i)
        .collect();
    (invalid.is_empty(), invalid)
}

pub fn save_keys(key: &SigningKey, address: &str) -> std::io::Result<PathBuf> {
    let dir_name = format!("{}.onion", &address[..16.min(address.len())]);
    let dir = PathBuf::from(&dir_name);
    fs::create_dir_all(&dir)?;

    // hostname file
    fs::write(dir.join("hostname"), format!("{}.onion\n", address))?;

    // public key: 32-byte header + 32-byte pubkey
    let pub_header = b"== ed25519v1-public: type0 ==\x00\x00\x00";
    let mut pub_bytes = Vec::with_capacity(64);
    pub_bytes.extend_from_slice(pub_header);
    pub_bytes.extend_from_slice(&key.verifying_key().to_bytes());
    fs::write(dir.join("hs_ed25519_public_key"), pub_bytes)?;

    // secret key: 32-byte header + 64-byte expanded key
    let sec_header = b"== ed25519v1-secret: type0 ==\x00\x00\x00";
    let seed = key.to_bytes();
    let mut expanded = sha512_expand(&seed);
    expanded[0] &= 248;
    expanded[31] &= 63;
    expanded[31] |= 64;
    let mut sec_bytes = Vec::with_capacity(96);
    sec_bytes.extend_from_slice(sec_header);
    sec_bytes.extend_from_slice(&expanded);
    fs::write(dir.join("hs_ed25519_secret_key"), sec_bytes)?;

    Ok(dir)
}

/// Save a GPU-generated keypair, which we have as a raw compressed pubkey plus
/// the clamped ed25519 scalar (not a seed). Tor v3 stores the secret key in
/// expanded form (32-byte scalar followed by a 32-byte nonce prefix), so no seed
/// is required: we keep the scalar verbatim and fill the prefix with random bytes
/// (it only seeds signing nonces and any value is valid as long as it stays secret).
pub fn save_keys_expanded(
    pubkey: &[u8; 32],
    scalar: &[u8; 32],
    address: &str,
) -> std::io::Result<PathBuf> {
    use rand::RngCore;
    let dir_name = format!("{}.onion", &address[..16.min(address.len())]);
    let dir = PathBuf::from(&dir_name);
    fs::create_dir_all(&dir)?;

    fs::write(dir.join("hostname"), format!("{}.onion\n", address))?;

    let pub_header = b"== ed25519v1-public: type0 ==\x00\x00\x00";
    let mut pub_bytes = Vec::with_capacity(64);
    pub_bytes.extend_from_slice(pub_header);
    pub_bytes.extend_from_slice(pubkey);
    fs::write(dir.join("hs_ed25519_public_key"), pub_bytes)?;

    let sec_header = b"== ed25519v1-secret: type0 ==\x00\x00\x00";
    let mut prefix = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut prefix);
    let mut sec_bytes = Vec::with_capacity(96);
    sec_bytes.extend_from_slice(sec_header);
    sec_bytes.extend_from_slice(scalar);
    sec_bytes.extend_from_slice(&prefix);
    fs::write(dir.join("hs_ed25519_secret_key"), sec_bytes)?;

    Ok(dir)
}

fn sha512_expand(seed: &[u8; 32]) -> [u8; 64] {
    use sha2::Digest as _;
    let mut hasher = Sha512::new();
    hasher.update(seed);
    let result = hasher.finalize();
    let mut out = [0u8; 64];
    out.copy_from_slice(&result);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_is_56_chars() {
        let mut rng = rand::thread_rng();
        let key = SigningKey::generate(&mut rng);
        let addr = derive_address(&key);
        assert_eq!(
            addr.len(),
            56,
            "Tor v3 address must be 56 chars, got: {}",
            addr
        );
        assert!(
            addr.chars()
                .all(|c| "abcdefghijklmnopqrstuvwxyz234567".contains(c))
        );
    }

    #[test]
    fn prefix_fast_check_works() {
        let mut rng = rand::thread_rng();
        // Generate a known keypair and verify fast-prefix agrees with full address
        for _ in 0..50 {
            let key = ed25519_dalek::SigningKey::generate(&mut rng);
            let pubkey = key.verifying_key().to_bytes();
            let full = derive_address(&key);
            let pat = full.as_bytes()[..4].to_vec();
            assert!(
                check_prefix_fast(&pubkey, &pat),
                "fast prefix must match own address"
            );
            // pattern that shouldn't match
            let wrong: Vec<u8> = pat
                .iter()
                .map(|&b| if b == b'a' { b'b' } else { b'a' })
                .collect();
            let full_starts = full.as_bytes().starts_with(&wrong);
            assert_eq!(check_prefix_fast(&pubkey, &wrong), full_starts);
        }
    }

    #[test]
    fn validate_rejects_invalid_chars() {
        let (ok, _) = validate_pattern("abc234");
        assert!(ok);
        let (ok, bad) = validate_pattern("abc89z");
        assert!(!ok);
        assert!(bad.contains(&3) || bad.contains(&4)); // '8' and '9' are at index 3,4
    }

    #[test]
    fn validate_extra_cases() {
        // empty is "valid" (no invalid chars), even though pattern_valid in app rejects it
        let (ok, bad) = validate_pattern("");
        assert!(ok);
        assert!(bad.is_empty());
        // uppercase is rejected (base32 lowercase only)
        let (ok, _) = validate_pattern("ABC");
        assert!(!ok);
        // 0, 1, 8, 9 are not in base32 alphabet
        let (ok, bad) = validate_pattern("a0b1c8d9e");
        assert!(!ok);
        assert_eq!(bad, vec![1, 3, 5, 7]);
    }

    /// Regression test: for a fixed seed the derived address must never change.
    /// This guards against any silent break of the Tor v3 derivation algorithm
    /// (SHA3-256 checksum, base32 encoding, version byte, byte ordering).
    #[test]
    fn address_is_deterministic_for_known_seed() {
        let seed = [0u8; 32];
        let key = ed25519_dalek::SigningKey::from_bytes(&seed);
        let a = derive_address(&key);
        let b = derive_address(&key);
        assert_eq!(a, b);
        assert_eq!(a.len(), 56);
        // Independent re-derivation using only sha3 + base32 directly (no shared
        // production helpers besides the inputs) - catches refactoring drift.
        let pubkey = key.verifying_key().to_bytes();
        let mut hasher = sha3::Sha3_256::new();
        hasher.update(b".onion checksum");
        hasher.update(pubkey);
        hasher.update([0x03u8]);
        let h = hasher.finalize();
        let mut payload = [0u8; 35];
        payload[..32].copy_from_slice(&pubkey);
        payload[32] = h[0];
        payload[33] = h[1];
        payload[34] = 0x03;
        let mut expected = [0u8; 56];
        BASE32_NOPAD.encode_mut(&payload, &mut expected);
        for c in expected.iter_mut() {
            *c = c.to_ascii_lowercase();
        }
        assert_eq!(a.as_bytes(), &expected);
    }

    /// For every prefix length the fast-path supports, fast-path on the address's
    /// own prefix must agree with the truth (true), and fast-path on a deliberately
    /// mutated prefix must agree with the slow-path starts_with check.
    #[test]
    fn prefix_fast_sweep_matches_full() {
        let mut rng = rand::thread_rng();
        for _ in 0..16 {
            let key = ed25519_dalek::SigningKey::generate(&mut rng);
            let pubkey = key.verifying_key().to_bytes();
            let full = derive_address(&key);
            let full_bytes = full.as_bytes();
            // Fast path is valid for n where ceil(n*5/8) <= 32, i.e. n <= 51.
            for n in 1..=51 {
                let pat = &full_bytes[..n];
                assert!(
                    check_prefix_fast(&pubkey, pat),
                    "self prefix len={n} must match"
                );
                // mutate one char and confirm fast-path mirrors full path
                let mut wrong = pat.to_vec();
                let last = wrong[n - 1];
                wrong[n - 1] = if last == b'a' { b'b' } else { b'a' };
                let truth = full_bytes.starts_with(&wrong);
                assert_eq!(
                    check_prefix_fast(&pubkey, &wrong),
                    truth,
                    "fast vs full disagree at n={n}"
                );
            }
            // Patterns past 51 chars need checksum bytes; fast path must bail.
            for n in [52, 53, 56] {
                if n <= full_bytes.len() {
                    assert!(
                        !check_prefix_fast(&pubkey, &full_bytes[..n]),
                        "fast path should bail for n>{}",
                        51
                    );
                }
            }
        }
    }

    #[test]
    fn check_prefix_fast_empty_is_true() {
        let pubkey = [0u8; 32];
        assert!(check_prefix_fast(&pubkey, b""));
    }

    #[test]
    fn match_type_cycle() {
        let p = MatchType::Prefix;
        assert_eq!(p.next(), MatchType::Suffix);
        assert_eq!(p.next().next(), MatchType::Anywhere);
        assert_eq!(p.next().next().next(), MatchType::Prefix);
        assert_eq!(p.prev(), MatchType::Anywhere);
        assert_eq!(MatchType::Suffix.prev(), MatchType::Prefix);
        assert_eq!(MatchType::Prefix.label(), "Prefix");
    }

    #[test]
    fn backend_toggle() {
        assert_eq!(Backend::Cpu.toggle(), Backend::Gpu);
        assert_eq!(Backend::Gpu.toggle(), Backend::Cpu);
        assert_eq!(Backend::Cpu.label(), "CPU");
        assert_eq!(Backend::Gpu.label(), "GPU");
    }
}
