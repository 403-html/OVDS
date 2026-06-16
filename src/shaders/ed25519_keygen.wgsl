// Ed25519 keygen kernels for OVDS GPU vanity search.
//
// Two scalar-mult strategies share the same field/group math:
//   * Fixed-base comb (comb_scalarmult): one scalar * B with a precomputed
//     table of multiples (build_table), no doublings.
//   * Incremental batch (prefix mode of main): each thread computes one start
//     point P0 = s0*B via the comb, then walks P0, P0+B, P0+2B, ... emitting
//     BATCH_K candidates. Each step is a single projective add and all the
//     per-candidate inversions are batched into one (Montgomery's trick), so
//     the amortized cost per candidate is ~one point add. Matching prefixes are
//     compacted on-device; the scalar for candidate j is just s0 + j.
//
// Field representation: fe25519 = array<u32, 16>, each limb holds 16 bits.
// All arithmetic is unsigned. Subtraction uses an additive form (a - b = a + 2p - b)
// so limbs never go negative.

// Params layout (storage buffer, std430 packing - each u32 takes 4 bytes):
//   [0..8]    base_seed (8 LE u32)
//   [8]       batch_id
//   [9]       threads
//   [10]      mode (0 = on-device prefix match + compaction, 1 = write-all by index)
//   [11]      pattern_len (base32 chars, prefix mode only)
//   [12..28]  bX (16 limbs)
//   [28..44]  bY
//   [44..60]  bZ
//   [60..76]  bT
//   [76..124] pattern (one base32 5-bit value per char, prefix mode only)
@group(0) @binding(0) var<storage, read> params: array<u32, 128>;
// Comb table: COMB_WINDOWS * COMB_PTS_PER_WIN points, each 64 u32 (X|Y|Z|T,
// 16 limbs each). Built once by build_table, read by the main keygen kernel.
@group(0) @binding(1) var<storage, read_write> table: array<u32>;
// Output buffer (atomic so prefix mode can compact matches via atomicAdd):
//   mode 0 (prefix): [0] = match count, records start at OUT_HEADER, 16 u32 each
//   mode 1 (write-all): record i at i*16, 8 u32 pubkey + 8 u32 scalar
@group(0) @binding(2) var<storage, read_write> output: array<atomic<u32>>;
// Scratch for incremental batch generation: per candidate 4 field elements
// (X, Y, Z, running-prefix-product), 16 limbs each = 64 u32.
@group(0) @binding(3) var<storage, read_write> scratch: array<u32>;

// Fixed-base comb parameters. W must divide 32 so a digit never straddles two
// scalar limbs. 32 windows * 8 bits = 256 bits covers any clamped scalar.
const COMB_W: u32 = 8u;
const COMB_WINDOWS: u32 = 32u;
const COMB_PTS_PER_WIN: u32 = 256u; // 1 << COMB_W
const GE_WORDS: u32 = 64u;         // 4 field elements * 16 limbs
const OUT_HEADER: u32 = 4u;
const MAX_MATCHES: u32 = 256u;

// Incremental batch: each thread computes P0 = s0*B once (comb), then walks
// P0, P0+B, P0+2B, ... emitting BATCH_K candidates. Each step is one projective
// point add; the per-candidate inversions are batched (one inversion/thread).
const BATCH_K: u32 = 64u;
const SCRATCH_WORDS_PER: u32 = 64u; // 4 field elements * 16 limbs

// --- u64 emulation as vec2<u32> ---

fn u64_zero() -> vec2<u32> { return vec2<u32>(0u, 0u); }

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = u32(lo < a.x);
    return vec2<u32>(lo, a.y + b.y + carry);
}

fn u64_add_u32(a: vec2<u32>, b: u32) -> vec2<u32> {
    let lo = a.x + b;
    let carry = u32(lo < a.x);
    return vec2<u32>(lo, a.y + carry);
}

// Multiply two u32s, returning a full 64-bit product as vec2<u32>.
fn u64_mul(a: u32, b: u32) -> vec2<u32> {
    let al = a & 0xFFFFu;
    let ah = a >> 16u;
    let bl = b & 0xFFFFu;
    let bh = b >> 16u;
    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;
    let mid = lh + hl;
    let mid_carry = u32(mid < lh);
    let lo_part = ll + (mid << 16u);
    let lo_carry = u32(lo_part < ll);
    let hi_part = hh + (mid >> 16u) + (mid_carry << 16u) + lo_carry;
    return vec2<u32>(lo_part, hi_part);
}

// --- fe25519 ops ---
// Note: we use array<u32, 16> directly rather than an alias, because naga emits
// `alias Fe = array<u32, 16>` as a distinct struct in Metal that doesn't unify
// with the underlying array type at the call site (breaks e.g. TWO_D init).

fn fe_zero() -> array<u32, 16> {
    var r: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { r[i] = 0u; }
    return r;
}

fn fe_one() -> array<u32, 16> {
    var r = fe_zero();
    r[0] = 1u;
    return r;
}

fn fe_add(a: array<u32, 16>, b: array<u32, 16>) -> array<u32, 16> {
    var aa = a; var bb = b;
    var r: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        r[i] = aa[i] + bb[i];
    }
    return r;
}

// Subtraction: r = a + 2p - b. The per-limb +0x10000 buffer avoids u32
// underflow; 2p = 2^256 - 38 = [0xFFDA, 0xFFFF*15]. If b's representation
// exceeds 2p a final borrow leaks an uncancelled +2^256 (= +38 mod p), so we
// detect that final borrow and fold the 38 back out. (Verified against the
// fe16_ref Rust mirror.)
fn fe_sub(a: array<u32, 16>, b: array<u32, 16>) -> array<u32, 16> {
    var aa = a; var bb = b;
    var r: array<u32, 16>;
    var borrow: u32 = 0u;
    var d: u32 = (0xFFDAu + 0x10000u) - bb[0] - borrow;
    r[0] = aa[0] + (d & 0xFFFFu);
    borrow = u32(d < 0x10000u);
    for (var i = 1u; i < 16u; i = i + 1u) {
        d = (0xFFFFu + 0x10000u) - bb[i] - borrow;
        r[i] = aa[i] + (d & 0xFFFFu);
        borrow = u32(d < 0x10000u);
    }
    // Subtract borrow*38 (the leaked 2^256) with borrow propagation.
    var owe: u32 = borrow * 38u;
    for (var i = 0u; i < 16u; i = i + 1u) {
        if (r[i] >= owe) {
            r[i] = r[i] - owe;
            owe = 0u;
        } else {
            r[i] = (r[i] + 0x10000u) - owe;
            owe = 1u;
        }
    }
    return r;
}

// Reduce limbs into canonical 16-bit form (mostly). Limb 15 keeps full 16
// bits since we use 2^256 ≡ 38 folding (not 2^255 ≡ 19).
fn fe_carry(a: array<u32, 16>) -> array<u32, 16> {
    var r = a;
    // First pass: propagate carries across limbs 0..14
    for (var i = 0u; i < 15u; i = i + 1u) {
        let v = r[i];
        r[i] = v & 0xFFFFu;
        r[i + 1u] = r[i + 1u] + (v >> 16u);
    }
    // Limb 15: take low 16, fold high to limb 0 via *38 (2^256 ≡ 38)
    let v15 = r[15];
    r[15] = v15 & 0xFFFFu;
    r[0] = r[0] + 38u * (v15 >> 16u);
    // Second pass to absorb cascade
    for (var i = 0u; i < 15u; i = i + 1u) {
        let v = r[i];
        r[i] = v & 0xFFFFu;
        r[i + 1u] = r[i + 1u] + (v >> 16u);
    }
    let v15b = r[15];
    r[15] = v15b & 0xFFFFu;
    r[0] = r[0] + 38u * (v15b >> 16u);
    return r;
}

fn fe_mul(a: array<u32, 16>, b: array<u32, 16>) -> array<u32, 16> {
    var aa = a; var bb = b;
    // Accumulate partial products into 31 vec2<u32> column sums.
    var t: array<vec2<u32>, 31>;
    for (var i = 0u; i < 31u; i = i + 1u) { t[i] = u64_zero(); }

    for (var i = 0u; i < 16u; i = i + 1u) {
        for (var j = 0u; j < 16u; j = j + 1u) {
            let p = u64_mul(aa[i], bb[j]);
            t[i + j] = u64_add(t[i + j], p);
        }
    }

    // Fold high half (t[16..30]) into low half via *38 (since 2^256 ≡ 38 mod p).
    for (var i = 0u; i < 15u; i = i + 1u) {
        let hi = t[i + 16u];
        let lo_part = u64_mul(38u, hi.x);
        let hi_shift = vec2<u32>(0u, 38u * hi.y);
        t[i] = u64_add(t[i], u64_add(lo_part, hi_shift));
    }

    // Pack into 16-limb Fe with carry propagation.
    var r: array<u32, 16>;
    var carry_lo: u32 = 0u;
    var carry_hi: u32 = 0u;
    for (var i = 0u; i < 16u; i = i + 1u) {
        var v = u64_add_u32(t[i], carry_lo);
        v = vec2<u32>(v.x, v.y + carry_hi);
        r[i] = v.x & 0xFFFFu;
        carry_lo = (v.x >> 16u) | (v.y << 16u);
        carry_hi = v.y >> 16u;
    }
    // Carry beyond limb 15 folds to limb 0 via *38 (since 2^256 ≡ 38 mod p).
    // When input limbs aren't reduced (e.g. coming from fe_add or fe_sq of a sum),
    // carry_lo can exceed 2^26, making 38*carry_lo overflow u32. Use the full
    // 64-bit product and spill its high part into limb 1.
    let prod = u64_mul(38u, carry_lo);
    let r0_new = r[0] + prod.x;
    let r0_carry = u32(r0_new < r[0]);
    r[0] = r0_new;
    r[1] = r[1] + prod.y + r0_carry;
    // carry_hi at this point is essentially 0 for in-range inputs; ignore.
    return fe_carry(r);
}

fn fe_sq(a: array<u32, 16>) -> array<u32, 16> {
    return fe_mul(a, a);
}

// Inversion via Fermat: a^(p-2) where p-2 = 2^255 - 21.
// Addition chain (~265 muls including squarings).
fn fe_invert(z: array<u32, 16>) -> array<u32, 16> {
    var t0 = fe_sq(z);
    var t1 = fe_sq(t0);
    t1 = fe_sq(t1);
    t1 = fe_mul(z, t1);
    t0 = fe_mul(t0, t1);
    var t2 = fe_sq(t0);
    t1 = fe_mul(t1, t2);
    t2 = fe_sq(t1);
    for (var i = 0u; i < 4u; i = i + 1u) { t2 = fe_sq(t2); }
    t1 = fe_mul(t1, t2);
    t2 = fe_sq(t1);
    for (var i = 0u; i < 9u; i = i + 1u) { t2 = fe_sq(t2); }
    t2 = fe_mul(t2, t1);
    var t3 = fe_sq(t2);
    for (var i = 0u; i < 19u; i = i + 1u) { t3 = fe_sq(t3); }
    t2 = fe_mul(t3, t2);
    t2 = fe_sq(t2);
    for (var i = 0u; i < 9u; i = i + 1u) { t2 = fe_sq(t2); }
    t1 = fe_mul(t2, t1);
    t2 = fe_sq(t1);
    for (var i = 0u; i < 49u; i = i + 1u) { t2 = fe_sq(t2); }
    t2 = fe_mul(t2, t1);
    t3 = fe_sq(t2);
    for (var i = 0u; i < 99u; i = i + 1u) { t3 = fe_sq(t3); }
    t2 = fe_mul(t3, t2);
    t2 = fe_sq(t2);
    for (var i = 0u; i < 49u; i = i + 1u) { t2 = fe_sq(t2); }
    t1 = fe_mul(t2, t1);
    t1 = fe_sq(t1);
    t1 = fe_sq(t1);
    t1 = fe_sq(t1);
    t1 = fe_sq(t1);
    t1 = fe_sq(t1);
    return fe_mul(t1, t0);
}

// Final reduction to canonical form: ensure value is in [0, p).
fn fe_freeze(a: array<u32, 16>) -> array<u32, 16> {
    var r = fe_carry(a);
    r = fe_carry(r);
    // After two carry passes, limbs 0..14 are ≤ 0xFFFF, limb 15 may be ≤ 0xFFFF + small.
    // Now conditionally subtract p twice.
    for (var iter = 0u; iter < 2u; iter = iter + 1u) {
        // Try to subtract p. p limbs: l[0]=0xFFED, l[1..14]=0xFFFF, l[15]=0x7FFF.
        var t: array<u32, 16>;
        var borrow: u32 = 0u;
        // limb 0
        let d0 = r[0] - 0xFFEDu - borrow;
        t[0] = d0 & 0xFFFFu;
        borrow = u32(r[0] < 0xFFEDu + borrow);
        for (var i = 1u; i < 15u; i = i + 1u) {
            let d = r[i] - 0xFFFFu - borrow;
            t[i] = d & 0xFFFFu;
            borrow = u32(r[i] < 0xFFFFu + borrow);
        }
        let d15 = r[15] - 0x7FFFu - borrow;
        t[15] = d15 & 0xFFFFu;
        let final_borrow = u32(r[15] < 0x7FFFu + borrow);
        // If no final borrow, r >= p, so use t. Otherwise keep r.
        let mask = (final_borrow ^ 1u) * 0xFFFFFFFFu;
        for (var i = 0u; i < 16u; i = i + 1u) {
            r[i] = (t[i] & mask) | (r[i] & ~mask);
        }
    }
    return r;
}

// Test parity of the LSB of a (after freeze).
fn fe_isnegative(a: array<u32, 16>) -> u32 {
    var f = fe_freeze(a);
    return f[0] & 1u;
}

// Pack fe25519 into 32 bytes (8 u32 LE).
fn fe_pack(a: array<u32, 16>) -> array<u32, 8> {
    var f = fe_freeze(a);
    var r: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        r[i] = f[2u * i] | (f[2u * i + 1u] << 16u);
    }
    return r;
}

// --- ge25519 (extended Edwards coords) ---
//
// Point = (X, Y, Z, T) with x = X/Z, y = Y/Z, T = XY/Z.
// Curve: -x^2 + y^2 = 1 + d*x^2*y^2, d = -121665/121666.

struct Ge {
    X: array<u32, 16>,
    Y: array<u32, 16>,
    Z: array<u32, 16>,
    T: array<u32, 16>,
}

fn ge_identity() -> Ge {
    var p: Ge;
    p.X = fe_zero();
    p.Y = fe_one();
    p.Z = fe_one();
    p.T = fe_zero();
    return p;
}

// d (the curve constant, not 2d) in fe25519 limb form. d = -121665/121666 mod p.
// BE hex 0x52036cee2b6ffe738cc740797779e89800700a4d4141d8ab75eb4dca135978a3.
// var<private> (not const) avoids a Metal type-check issue with naga const arrays.
var<private> D_CONST: array<u32, 16> = array<u32, 16>(
    0x78a3u, 0x1359u, 0x4dcau, 0x75ebu, 0xd8abu, 0x4141u, 0x0a4du, 0x0070u,
    0xe898u, 0x7779u, 0x4079u, 0x8cc7u, 0xfe73u, 0x2b6fu, 0x6ceeu, 0x5203u,
);

// Complete unified twisted-Edwards addition for a=-1 (HWCD 2008, the
// complete/unified variant). Complete for Ed25519 since a=-1 is a square and d
// is a non-square, so it serves as doubling (P+P) and handles the identity. The
// fast "hwcd-3" doubling is NOT complete and produces off-curve garbage on the
// neutral element, so we use this single formula everywhere. (Verified against
// the fe16_ref Rust mirror and curve25519-dalek.)
fn ge_add(p: Ge, q: Ge) -> Ge {
    let A = fe_mul(p.X, q.X);
    let B = fe_mul(p.Y, q.Y);
    let C = fe_mul(fe_mul(p.T, q.T), D_CONST);
    let D = fe_mul(p.Z, q.Z);
    let E = fe_sub(fe_sub(fe_mul(fe_add(p.X, p.Y), fe_add(q.X, q.Y)), A), B);
    let F = fe_sub(D, C);
    let G = fe_add(D, C);
    let H = fe_add(B, A); // H = B - a*A = B + A  (a = -1)
    var r: Ge;
    r.X = fe_mul(E, F);
    r.Y = fe_mul(G, H);
    r.Z = fe_mul(F, G);
    r.T = fe_mul(E, H);
    return r;
}

fn ge_double(p: Ge) -> Ge {
    return ge_add(p, p);
}

// --- comb table storage ---

fn store_ge(slot: u32, g: Ge) {
    let o = slot * GE_WORDS;
    var gx = g.X; var gy = g.Y; var gz = g.Z; var gt = g.T;
    for (var i = 0u; i < 16u; i = i + 1u) {
        table[o + i] = gx[i];
        table[o + 16u + i] = gy[i];
        table[o + 32u + i] = gz[i];
        table[o + 48u + i] = gt[i];
    }
}

fn load_ge(slot: u32) -> Ge {
    let o = slot * GE_WORDS;
    var g: Ge;
    for (var i = 0u; i < 16u; i = i + 1u) {
        g.X[i] = table[o + i];
        g.Y[i] = table[o + 16u + i];
        g.Z[i] = table[o + 32u + i];
        g.T[i] = table[o + 48u + i];
    }
    return g;
}

// Fixed-base comb (windowed) scalar mult: Q = scalar * B. No doublings - each
// window contributes one precomputed multiple table[window][digit]. The
// complete addition handles the identity, so zero digits are simply skipped.
fn comb_scalarmult(scalar: array<u32, 8>) -> Ge {
    var ss = scalar;
    var Q = ge_identity();
    for (var i = 0u; i < COMB_WINDOWS; i = i + 1u) {
        let bitpos = i * COMB_W;
        let limb = ss[bitpos >> 5u];
        let digit = (limb >> (bitpos & 31u)) & (COMB_PTS_PER_WIN - 1u);
        if (digit != 0u) {
            Q = ge_add(Q, load_ge(i * COMB_PTS_PER_WIN + digit));
        }
    }
    return Q;
}

// Compress: pubkey = pack(Y/Z) with high bit of byte 31 = parity(X/Z).
fn ge_compress(p: Ge) -> array<u32, 8> {
    let zInv = fe_invert(p.Z);
    let x_aff = fe_mul(p.X, zInv);
    let y_aff = fe_mul(p.Y, zInv);
    var bytes = fe_pack(y_aff);
    let parity = fe_isnegative(x_aff);
    // Set top bit of byte 31 (= top bit of limb 7's high 16 bits)
    bytes[7] = bytes[7] | (parity << 31u);
    return bytes;
}

// --- main kernel ---

// Cheap per-thread scalar derivation. Mix base_seed with thread_idx + batch_id.
// (Quality of randomness isn't critical: vanity search just needs distinct
// scalars; the host re-seeds each batch.)
fn derive_scalar(base_seed: array<u32, 8>, batch_id: u32, idx: u32) -> array<u32, 8> {
    var bs = base_seed;
    var s: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        s[i] = bs[i] ^ (idx * (0x9E3779B9u + i * 0x6F4A7855u) + batch_id * 0xBB67AE85u);
    }
    // Avalanche pass so consecutive idx aren't trivially related per-limb.
    var a = s[0]; var b = s[7];
    s[0] = a ^ (b << 13u) ^ (b >> 19u);
    s[7] = b ^ (a << 7u)  ^ (a >> 25u);
    // Ed25519 clamp: clear low 3 bits of byte 0, clear high bit of byte 31, set bit 254.
    s[0] = s[0] & 0xFFFFFFF8u;
    s[7] = (s[7] & 0x7FFFFFFFu) | 0x40000000u;
    return s;
}

fn load_base_seed() -> array<u32, 8> {
    var s: array<u32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) { s[i] = params[i]; }
    return s;
}

fn load_basepoint() -> Ge {
    var B: Ge;
    for (var i = 0u; i < 16u; i = i + 1u) {
        B.X[i] = params[12u + i];
        B.Y[i] = params[28u + i];
        B.Z[i] = params[44u + i];
        B.T[i] = params[60u + i];
    }
    return B;
}

// --- on-device base32 prefix match ---

// Byte `idx` (0..31) of the compressed pubkey (8 LE u32 = 32 bytes).
fn pk_byte(pk: array<u32, 8>, idx: u32) -> u32 {
    var p = pk;
    return (p[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
}

// 5-bit base32 symbol value at char `j`. The byte stream is read MSB-first
// (RFC 4648), so char j covers bits [5j, 5j+5). Caller guarantees the 5 bits
// fall within bytes 0..31 (pattern_len capped at 48 host-side, max bit 235).
fn base32_sym(pk: array<u32, 8>, j: u32) -> u32 {
    let bit = j * 5u;
    let byte_idx = bit >> 3u;
    let bit_in_byte = bit & 7u;
    let win = (pk_byte(pk, byte_idx) << 8u) | pk_byte(pk, byte_idx + 1u);
    return (win >> (11u - bit_in_byte)) & 0x1Fu;
}

fn matches_prefix(pk: array<u32, 8>, plen: u32) -> bool {
    for (var j = 0u; j < plen; j = j + 1u) {
        if (base32_sym(pk, j) != params[76u + j]) { return false; }
    }
    return true;
}

fn write_record(o: u32, pubkey: array<u32, 8>, scalar: array<u32, 8>) {
    var pk = pubkey;
    var sc = scalar;
    for (var i = 0u; i < 8u; i = i + 1u) {
        atomicStore(&output[o + i], pk[i]);
        atomicStore(&output[o + 8u + i], sc[i]);
    }
}

// --- incremental batch helpers ---

fn scratch_store_fe(off: u32, v: array<u32, 16>) {
    var vv = v;
    for (var i = 0u; i < 16u; i = i + 1u) { scratch[off + i] = vv[i]; }
}

fn scratch_load_fe(off: u32) -> array<u32, 16> {
    var r: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { r[i] = scratch[off + i]; }
    return r;
}

// Compress a point given a precomputed 1/Z (from the batched inversion).
fn compress_zinv(X: array<u32, 16>, Y: array<u32, 16>, zinv: array<u32, 16>) -> array<u32, 8> {
    let x_aff = fe_mul(X, zinv);
    let y_aff = fe_mul(Y, zinv);
    var bytes = fe_pack(y_aff);
    let parity = fe_isnegative(x_aff);
    bytes[7] = bytes[7] | (parity << 31u);
    return bytes;
}

// scalar (8 LE u32) + small integer j, with carry propagation.
fn scalar_add_u32(s: array<u32, 8>, j: u32) -> array<u32, 8> {
    var r = s;
    var carry = j;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let prev = r[i];
        let lo = prev + carry;
        r[i] = lo;
        carry = select(0u, 1u, lo < prev);
    }
    return r;
}

// --- comb table builder (run once at pipeline init) ---

@compute @workgroup_size(1)
fn build_table(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    var base = load_basepoint(); // window 0 base = B
    for (var i = 0u; i < COMB_WINDOWS; i = i + 1u) {
        let win = i * COMB_PTS_PER_WIN;
        store_ge(win, ge_identity());
        var acc = base;
        store_ge(win + 1u, acc);
        for (var k = 2u; k < COMB_PTS_PER_WIN; k = k + 1u) {
            acc = ge_add(acc, base);
            store_ge(win + k, acc);
        }
        // Next window base = base * 2^COMB_W.
        for (var d = 0u; d < COMB_W; d = d + 1u) { base = ge_double(base); }
    }
}

// --- main keygen kernel ---

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let threads = params[9u];
    if (idx >= threads) { return; }
    let base_seed = load_base_seed();
    let batch_id = params[8u];
    let mode = params[10u];
    var s0 = derive_scalar(base_seed, batch_id, idx);

    // Write-all mode: one full keypair per thread (the dalek regression path).
    if (mode == 1u) {
        let Q = comb_scalarmult(s0);
        let pubkey = ge_compress(Q);
        write_record(idx * 16u, pubkey, s0);
        return;
    }

    // Prefix mode: incremental generation + batched inversion.
    let plen = params[11u];
    let B = load_basepoint();
    var P = comb_scalarmult(s0); // P0 = s0 * B (projective)
    let sbase = idx * BATCH_K * SCRATCH_WORDS_PER;

    // Forward pass: emit P0, P0+B, ... storing (X,Y,Z) and the running product
    // of Z values so each 1/Z_j can be recovered after a single inversion.
    var acc = fe_one();
    for (var j = 0u; j < BATCH_K; j = j + 1u) {
        let o = sbase + j * SCRATCH_WORDS_PER;
        scratch_store_fe(o, P.X);
        scratch_store_fe(o + 16u, P.Y);
        scratch_store_fe(o + 32u, P.Z);
        scratch_store_fe(o + 48u, acc); // prefix product before folding in Z_j
        acc = fe_mul(acc, P.Z);
        P = ge_add(P, B);
    }

    // One inversion for the whole batch.
    var inv = fe_invert(acc);

    // Backward pass: 1/Z_j = inv * prefix_j, then strip Z_j out of inv.
    for (var jj = 0u; jj < BATCH_K; jj = jj + 1u) {
        let j = BATCH_K - 1u - jj;
        let o = sbase + j * SCRATCH_WORDS_PER;
        let zj = scratch_load_fe(o + 32u);
        let pre = scratch_load_fe(o + 48u);
        let zinv = fe_mul(inv, pre);
        inv = fe_mul(inv, zj);
        let pubkey = compress_zinv(scratch_load_fe(o), scratch_load_fe(o + 16u), zinv);
        if (matches_prefix(pubkey, plen)) {
            let slot = atomicAdd(&output[0u], 1u);
            if (slot < MAX_MATCHES) {
                write_record(OUT_HEADER + slot * 16u, pubkey, scalar_add_u32(s0, j));
            }
        }
    }
}
