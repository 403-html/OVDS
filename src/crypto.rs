use data_encoding::BASE32_NOPAD;
use ed25519_dalek::SigningKey;
use rand::rngs::ThreadRng;
use sha2::Sha512;
use sha3::{Digest, Sha3_256};
use std::fs;
use std::path::PathBuf;

pub fn generate_keypair(rng: &mut ThreadRng) -> (SigningKey, String) {
    let key = SigningKey::generate(rng);
    let address = derive_address(&key);
    (key, address)
}

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

#[derive(Clone, Copy, PartialEq, Debug)]
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

#[derive(Clone, PartialEq, Debug)]
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
        let (key, addr) = generate_keypair(&mut rng);
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
        let _ = key;
    }

    #[test]
    fn prefix_fast_check_works() {
        let mut rng = rand::thread_rng();
        // Generate a known keypair and verify fast-prefix agrees with full address
        for _ in 0..50 {
            let key = ed25519_dalek::SigningKey::generate(&mut rng);
            let pubkey = key.verifying_key().to_bytes();
            let full = derive_address(&key);
            let pat = full[..4].as_bytes().to_vec();
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
}
