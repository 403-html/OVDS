//! Test-only Rust mirror of the WGSL fe25519 limb arithmetic.
//!
//! The GPU shader is hard to debug (no prints, flaky Metal compiler), so we
//! reproduce its exact 16-limb (16-bit radix) field algorithm here and check it
//! against a straightforward bignum reference. A bug found here is a bug in the
//! algorithm, which the WGSL shares line-for-line.

#![cfg(test)]

// ---- the mirrored limb ops (must match shaders/ed25519_keygen.wgsl) ----

pub type Fe = [u32; 16];

pub fn fe_zero() -> Fe {
    [0u32; 16]
}

pub fn fe_one() -> Fe {
    let mut r = [0u32; 16];
    r[0] = 1;
    r
}

pub fn fe_add(a: &Fe, b: &Fe) -> Fe {
    let mut r = [0u32; 16];
    for i in 0..16 {
        r[i] = a[i].wrapping_add(b[i]);
    }
    r
}

pub fn fe_sub(a: &Fe, b: &Fe) -> Fe {
    // r = a + 2p - b. The per-limb +0x10000 buffer avoids underflow, but if b's
    // representation exceeds 2p a final borrow leaks an uncancelled +2^256 (= +38
    // mod p). Detect that final borrow and fold the 38 back out.
    let mut r = [0u32; 16];
    let mut borrow: u32 = 0;
    let mut d: u32 = (0xFFDA + 0x10000u32)
        .wrapping_sub(b[0])
        .wrapping_sub(borrow);
    r[0] = a[0].wrapping_add(d & 0xFFFF);
    borrow = (d < 0x10000) as u32;
    for i in 1..16 {
        d = (0xFFFF + 0x10000u32)
            .wrapping_sub(b[i])
            .wrapping_sub(borrow);
        r[i] = a[i].wrapping_add(d & 0xFFFF);
        borrow = (d < 0x10000) as u32;
    }
    // Subtract borrow*38 (the leaked 2^256) with borrow propagation.
    let mut owe = borrow * 38;
    for limb in r.iter_mut() {
        if *limb >= owe {
            *limb -= owe;
            owe = 0;
        } else {
            *limb = (*limb + 0x10000) - owe;
            owe = 1;
        }
    }
    r
}

pub fn fe_carry(a: &Fe) -> Fe {
    let mut r = *a;
    for i in 0..15 {
        let v = r[i];
        r[i] = v & 0xFFFF;
        r[i + 1] = r[i + 1].wrapping_add(v >> 16);
    }
    let v15 = r[15];
    r[15] = v15 & 0xFFFF;
    r[0] = r[0].wrapping_add(38 * (v15 >> 16));
    for i in 0..15 {
        let v = r[i];
        r[i] = v & 0xFFFF;
        r[i + 1] = r[i + 1].wrapping_add(v >> 16);
    }
    let v15b = r[15];
    r[15] = v15b & 0xFFFF;
    r[0] = r[0].wrapping_add(38 * (v15b >> 16));
    r
}

pub fn fe_mul(a: &Fe, b: &Fe) -> Fe {
    let mut t = [0u64; 31];
    for i in 0..16 {
        for j in 0..16 {
            t[i + j] += (a[i] as u64) * (b[j] as u64);
        }
    }
    for i in 0..15 {
        t[i] += 38 * t[i + 16];
    }
    let mut r = [0u32; 16];
    let mut carry: u64 = 0;
    for i in 0..16 {
        let v = t[i] + carry;
        r[i] = (v & 0xFFFF) as u32;
        carry = v >> 16;
    }
    // fold carry beyond limb 15 via *38
    let prod = 38u64 * carry;
    let r0_new = r[0] as u64 + (prod & 0xFFFFFFFF);
    r[0] = (r0_new & 0xFFFFFFFF) as u32;
    let r0_carry = (r0_new >> 32) as u32;
    r[1] = r[1]
        .wrapping_add((prod >> 32) as u32)
        .wrapping_add(r0_carry);
    fe_carry(&r)
}

pub fn fe_sq(a: &Fe) -> Fe {
    fe_mul(a, a)
}

pub fn fe_invert(z: &Fe) -> Fe {
    let mut t0 = fe_sq(z);
    let mut t1 = fe_sq(&t0);
    t1 = fe_sq(&t1);
    t1 = fe_mul(z, &t1);
    t0 = fe_mul(&t0, &t1);
    let mut t2 = fe_sq(&t0);
    t1 = fe_mul(&t1, &t2);
    t2 = fe_sq(&t1);
    for _ in 0..4 {
        t2 = fe_sq(&t2);
    }
    t1 = fe_mul(&t1, &t2);
    t2 = fe_sq(&t1);
    for _ in 0..9 {
        t2 = fe_sq(&t2);
    }
    t2 = fe_mul(&t2, &t1);
    let mut t3 = fe_sq(&t2);
    for _ in 0..19 {
        t3 = fe_sq(&t3);
    }
    t2 = fe_mul(&t3, &t2);
    t2 = fe_sq(&t2);
    for _ in 0..9 {
        t2 = fe_sq(&t2);
    }
    t1 = fe_mul(&t2, &t1);
    t2 = fe_sq(&t1);
    for _ in 0..49 {
        t2 = fe_sq(&t2);
    }
    t2 = fe_mul(&t2, &t1);
    t3 = fe_sq(&t2);
    for _ in 0..99 {
        t3 = fe_sq(&t3);
    }
    t2 = fe_mul(&t3, &t2);
    t2 = fe_sq(&t2);
    for _ in 0..49 {
        t2 = fe_sq(&t2);
    }
    t1 = fe_mul(&t2, &t1);
    t1 = fe_sq(&t1);
    t1 = fe_sq(&t1);
    t1 = fe_sq(&t1);
    t1 = fe_sq(&t1);
    t1 = fe_sq(&t1);
    fe_mul(&t1, &t0)
}

pub fn fe_freeze(a: &Fe) -> Fe {
    let mut r = fe_carry(a);
    r = fe_carry(&r);
    for _ in 0..2 {
        let mut t = [0u32; 16];
        let mut borrow: u32 = 0;
        let d0 = r[0].wrapping_sub(0xFFED).wrapping_sub(borrow);
        t[0] = d0 & 0xFFFF;
        borrow = (r[0] < 0xFFED + borrow) as u32;
        for i in 1..15 {
            let d = r[i].wrapping_sub(0xFFFF).wrapping_sub(borrow);
            t[i] = d & 0xFFFF;
            borrow = (r[i] < 0xFFFF + borrow) as u32;
        }
        let d15 = r[15].wrapping_sub(0x7FFF).wrapping_sub(borrow);
        t[15] = d15 & 0xFFFF;
        let final_borrow = (r[15] < 0x7FFF + borrow) as u32;
        let mask = (final_borrow ^ 1).wrapping_mul(0xFFFFFFFF);
        for i in 0..16 {
            r[i] = (t[i] & mask) | (r[i] & !mask);
        }
    }
    r
}

pub fn fe_pack(a: &Fe) -> [u8; 32] {
    let f = fe_freeze(a);
    let mut out = [0u8; 32];
    for i in 0..16 {
        out[2 * i] = (f[i] & 0xFF) as u8;
        out[2 * i + 1] = ((f[i] >> 8) & 0xFF) as u8;
    }
    out
}

pub fn fe_from_bytes(b: &[u8; 32]) -> Fe {
    let mut r = [0u32; 16];
    for i in 0..16 {
        r[i] = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as u32;
    }
    r
}

// ---- bignum reference (canonical [u8;32] < p) ----

const P_BYTES: [u8; 32] = {
    let mut p = [0xFFu8; 32];
    p[0] = 0xED;
    p[31] = 0x7F;
    p
};

fn ref_add(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut carry = 0u16;
    for i in 0..32 {
        let v = a[i] as u16 + b[i] as u16 + carry;
        out[i] = (v & 0xFF) as u8;
        carry = v >> 8;
    }
    // result may be >= p (or even >= 2^256 via carry); reduce.
    reduce_once(&mut out, carry);
    out
}

fn reduce_once(out: &mut [u8; 32], extra_carry: u16) {
    // if extra_carry (value >= 2^256) add 38 (since 2^256 = 38 mod p), then
    // conditionally subtract p up to twice.
    if extra_carry > 0 {
        let mut c = 38u16 * extra_carry;
        for byte in out.iter_mut() {
            let v = *byte as u16 + (c & 0xFF);
            *byte = (v & 0xFF) as u8;
            c = (c >> 8) + (v >> 8);
        }
    }
    for _ in 0..2 {
        if geq_p(out) {
            sub_p(out);
        }
    }
}

fn geq_p(a: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if a[i] != P_BYTES[i] {
            return a[i] > P_BYTES[i];
        }
    }
    true
}

fn sub_p(a: &mut [u8; 32]) {
    let mut borrow = 0i16;
    for i in 0..32 {
        let v = a[i] as i16 - P_BYTES[i] as i16 - borrow;
        if v < 0 {
            a[i] = (v + 256) as u8;
            borrow = 1;
        } else {
            a[i] = v as u8;
            borrow = 0;
        }
    }
}

/// Reference value of a 16-limb (possibly unreduced) field element, as canonical
/// bytes mod p. Uses Horner: acc = sum limb[i]*2^(16i).
pub fn limbs16_to_ref(a: &Fe) -> [u8; 32] {
    let two16: [u8; 32] = {
        let mut b = [0u8; 32];
        b[2] = 1; // 2^16
        b
    };
    let mut acc = [0u8; 32];
    for i in (0..16).rev() {
        acc = super::gpu::field_mul_p(&acc, &two16);
        let mut limb = [0u8; 32];
        let l = a[i];
        limb[0] = (l & 0xFF) as u8;
        limb[1] = ((l >> 8) & 0xFF) as u8;
        limb[2] = ((l >> 16) & 0xFF) as u8;
        limb[3] = ((l >> 24) & 0xFF) as u8;
        // limb may itself be >= p in pathological cases; reduce it first.
        let mut limb_r = limb;
        reduce_once(&mut limb_r, 0);
        acc = ref_add(&acc, &limb_r);
    }
    acc
}

pub fn ref_mul(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    super::gpu::field_mul_p(a, b)
}

pub fn ref_pow(base: &[u8; 32], exp: &[u8; 32]) -> [u8; 32] {
    // square-and-multiply, MSB first
    let mut result = {
        let mut o = [0u8; 32];
        o[0] = 1;
        o
    };
    for byte_i in (0..32).rev() {
        for bit in (0..8).rev() {
            result = ref_mul(&result, &result);
            if (exp[byte_i] >> bit) & 1 == 1 {
                result = ref_mul(&result, base);
            }
        }
    }
    result
}

// ---- ge25519 mirror (extended coords) ----

#[derive(Clone)]
pub struct Ge {
    pub x: Fe,
    pub y: Fe,
    pub z: Fe,
    pub t: Fe,
}

pub fn ge_identity() -> Ge {
    Ge {
        x: fe_zero(),
        y: fe_one(),
        z: fe_one(),
        t: fe_zero(),
    }
}

// d (the curve constant) in limb form (matches WGSL D_CONST).
pub fn d_const() -> Fe {
    [
        0x78a3, 0x1359, 0x4dca, 0x75eb, 0xd8ab, 0x4141, 0x0a4d, 0x0070, 0xe898, 0x7779, 0x4079,
        0x8cc7, 0xfe73, 0x2b6f, 0x6cee, 0x5203,
    ]
}

/// Complete unified twisted-Edwards addition for a=-1 (Hisil-Wong-Carter-Dawson
/// 2008, the unified/complete variant). Complete for Ed25519 since a=-1 is a
/// square and d is a non-square, so it also serves as doubling (P+P) and handles
/// the neutral element. The fast "hwcd-3" doubling is NOT complete and produces
/// off-curve garbage on the identity, so we use this single formula everywhere.
pub fn ge_add(p: &Ge, q: &Ge) -> Ge {
    let a = fe_mul(&p.x, &q.x);
    let b = fe_mul(&p.y, &q.y);
    let c = fe_mul(&fe_mul(&p.t, &q.t), &d_const());
    let d = fe_mul(&p.z, &q.z);
    let e = fe_sub(
        &fe_sub(&fe_mul(&fe_add(&p.x, &p.y), &fe_add(&q.x, &q.y)), &a),
        &b,
    );
    let f = fe_sub(&d, &c);
    let g = fe_add(&d, &c);
    let h = fe_add(&b, &a); // H = B - a*A = B + A  (a = -1)
    Ge {
        x: fe_mul(&e, &f),
        y: fe_mul(&g, &h),
        z: fe_mul(&f, &g),
        t: fe_mul(&e, &h),
    }
}

pub fn ge_double(p: &Ge) -> Ge {
    ge_add(p, p)
}

pub fn ge_scalarmult_base(b: &Ge, scalar: &[u8; 32]) -> Ge {
    // scalar as 8 LE u32
    let s: [u32; 8] =
        std::array::from_fn(|i| u32::from_le_bytes(scalar[4 * i..4 * i + 4].try_into().unwrap()));
    let mut q = ge_identity();
    for bit_i in 0..256u32 {
        let i = 255 - bit_i;
        q = ge_double(&q);
        let limb = s[(i >> 5) as usize];
        let bit = (limb >> (i & 31)) & 1;
        if bit == 1 {
            q = ge_add(&q, b);
        }
    }
    q
}

pub fn ge_compress(p: &Ge) -> [u8; 32] {
    let z_inv = fe_invert(&p.z);
    let x_aff = fe_mul(&p.x, &z_inv);
    let y_aff = fe_mul(&p.y, &z_inv);
    let mut bytes = fe_pack(&y_aff);
    let parity = (fe_freeze(&x_aff)[0] & 1) as u8;
    bytes[31] |= parity << 7;
    bytes
}

pub fn basepoint(bx: &[u8; 32], by: &[u8; 32]) -> Ge {
    let x = fe_from_bytes(bx);
    let y = fe_from_bytes(by);
    let z = fe_one();
    let t = fe_from_bytes(&super::gpu::field_mul_p(bx, by));
    Ge { x, y, z, t }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_reduced(seed: u64) -> ([u8; 32], Fe) {
        // simple LCG to fill bytes, then reduce mod p
        let mut s = seed;
        let mut bytes = [0u8; 32];
        for b in bytes.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *b = (s >> 33) as u8;
        }
        bytes[31] &= 0x7F; // keep < 2^255, likely < p
        let mut r = bytes;
        for _ in 0..2 {
            if geq_p(&r) {
                sub_p(&mut r);
            }
        }
        (r, fe_from_bytes(&r))
    }

    #[test]
    fn pack_roundtrip() {
        for seed in 0..50 {
            let (bytes, fe) = rand_reduced(seed);
            assert_eq!(fe_pack(&fe), bytes, "pack roundtrip seed {seed}");
        }
    }

    #[test]
    fn limbs_to_ref_matches_pack() {
        for seed in 0..50 {
            let (bytes, fe) = rand_reduced(seed);
            assert_eq!(limbs16_to_ref(&fe), bytes, "limbs_to_ref seed {seed}");
        }
    }

    #[test]
    fn mul_matches_ref() {
        for seed in 0..100 {
            let (ab, a) = rand_reduced(seed);
            let (bb, b) = rand_reduced(seed.wrapping_add(777));
            let got = limbs16_to_ref(&fe_mul(&a, &b));
            let want = ref_mul(&ab, &bb);
            assert_eq!(got, want, "fe_mul seed {seed}\n a={ab:02x?}\n b={bb:02x?}");
        }
    }

    #[test]
    fn add_matches_ref() {
        for seed in 0..100 {
            let (ab, a) = rand_reduced(seed);
            let (bb, b) = rand_reduced(seed.wrapping_add(13));
            let got = limbs16_to_ref(&fe_add(&a, &b));
            let want = {
                let mut o = ab;
                let mut carry = 0u16;
                for i in 0..32 {
                    let v = o[i] as u16 + bb[i] as u16 + carry;
                    o[i] = (v & 0xFF) as u8;
                    carry = v >> 8;
                }
                reduce_once(&mut o, carry);
                o
            };
            assert_eq!(got, want, "fe_add seed {seed}");
        }
    }

    #[test]
    fn sub_matches_ref() {
        for seed in 0..100 {
            let (ab, a) = rand_reduced(seed);
            let (bb, b) = rand_reduced(seed.wrapping_add(99));
            let got = limbs16_to_ref(&fe_sub(&a, &b));
            // ref: (a - b) mod p = (a + p - b) mod p
            let want = ref_mul(&ref_sub(&ab, &bb), &{
                let mut o = [0u8; 32];
                o[0] = 1;
                o
            });
            assert_eq!(got, want, "fe_sub seed {seed}");
        }
    }

    fn ref_sub(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
        // a - b mod p
        let mut o = [0u8; 32];
        let mut borrow = 0i16;
        for i in 0..32 {
            let v = a[i] as i16 - b[i] as i16 - borrow;
            if v < 0 {
                o[i] = (v + 256) as u8;
                borrow = 1;
            } else {
                o[i] = v as u8;
                borrow = 0;
            }
        }
        if borrow == 1 {
            // add p
            let mut carry = 0u16;
            for i in 0..32 {
                let v = o[i] as u16 + P_BYTES[i] as u16 + carry;
                o[i] = (v & 0xFF) as u8;
                carry = v >> 8;
            }
        }
        o
    }

    #[test]
    fn invert_matches_ref() {
        // p - 2
        let mut pm2 = P_BYTES;
        pm2[0] = pm2[0].wrapping_sub(2); // 0xED - 2 = 0xEB
        for seed in 1..20 {
            let (ab, a) = rand_reduced(seed);
            let got = limbs16_to_ref(&fe_invert(&a));
            let want = ref_pow(&ab, &pm2);
            assert_eq!(got, want, "fe_invert seed {seed}");
            // sanity: a * inv(a) == 1
            let prod = ref_mul(&ab, &got);
            let mut one = [0u8; 32];
            one[0] = 1;
            assert_eq!(prod, one, "a*inv(a) seed {seed}");
        }
    }

    use crate::gpu::{BX_LE, BY_LE};
    use curve25519_dalek::edwards::EdwardsPoint;
    use curve25519_dalek::scalar::Scalar;

    fn canonical_basepoint() -> [u8; 32] {
        // Ed25519 basepoint compression: LE 0x58 then 0x66 in all other bytes.
        let mut b = [0x66u8; 32];
        b[0] = 0x58;
        b
    }

    #[test]
    fn basepoint_compresses_correctly() {
        let bp = basepoint(&BX_LE, &BY_LE);
        assert_eq!(ge_compress(&bp), canonical_basepoint());
    }

    #[test]
    fn basepoint_t_is_xy() {
        // T must equal X*Y mod p (since Z=1). Independent check via field_mul_p.
        let bp = basepoint(&BX_LE, &BY_LE);
        let xy = crate::gpu::field_mul_p(&BX_LE, &BY_LE);
        assert_eq!(fe_pack(&bp.t), xy);
    }

    #[test]
    fn ge_double_identity_is_identity() {
        let id = ge_identity();
        let dd = ge_double(&id);
        assert_eq!(ge_compress(&dd), ge_compress(&id), "2*identity != identity");
    }

    #[test]
    fn ge_add_identity_is_noop() {
        let bp = basepoint(&BX_LE, &BY_LE);
        let id = ge_identity();
        let sum = ge_add(&id, &bp);
        assert_eq!(ge_compress(&sum), canonical_basepoint(), "id + B != B");
    }

    #[test]
    fn ge_add_bb_matches_dalek() {
        // 2B via addition only (uses T).
        let bp = basepoint(&BX_LE, &BY_LE);
        let two_b = ge_add(&bp, &bp);
        let expected = EdwardsPoint::mul_base(&Scalar::from(2u64)).compress().0;
        assert_eq!(ge_compress(&two_b), expected, "B+B mismatch");
    }

    #[test]
    fn ge_double_matches_dalek() {
        let bp = basepoint(&BX_LE, &BY_LE);
        let two_b = ge_double(&bp);
        let two = Scalar::from(2u64);
        let expected = EdwardsPoint::mul_base(&two).compress().0;
        assert_eq!(ge_compress(&two_b), expected, "2B mismatch");
    }

    #[test]
    fn ge_add_matches_dalek() {
        let bp = basepoint(&BX_LE, &BY_LE);
        // 3B = 2B + B
        let three_b = ge_add(&ge_double(&bp), &bp);
        let expected = EdwardsPoint::mul_base(&Scalar::from(3u64)).compress().0;
        assert_eq!(ge_compress(&three_b), expected, "3B mismatch");
    }

    #[test]
    fn add_to_doubled_identity() {
        let bp = basepoint(&BX_LE, &BY_LE);
        let dbl_id = ge_double(&ge_identity());
        // (0,-1,-1,0) + B should still be B.
        let sum = ge_add(&dbl_id, &bp);
        assert_eq!(ge_compress(&sum), canonical_basepoint(), "dbl_id + B != B");
        // doubling the non-canonical identity should still compress to identity
        let dd = ge_double(&dbl_id);
        assert_eq!(
            ge_compress(&dd),
            ge_compress(&ge_identity()),
            "double(dbl_id) != identity"
        );
    }

    #[test]
    fn sq_of_sum_of_2p_reps() {
        // X = 2p, Y = 2p+1.  X+Y = 4p+1 ≡ 1.  (X+Y)^2 must be 1.
        let mut x = [0xFFFFu32; 16];
        x[0] = 0xFFDA;
        let mut y = [0xFFFFu32; 16];
        y[0] = 0xFFDB;
        let s = fe_add(&x, &y); // limbs ~0x1FFFE
        let sq = fe_mul(&s, &s);
        let mut one = [0u8; 32];
        one[0] = 1;
        assert_eq!(limbs16_to_ref(&sq), one, "(4p+1)^2 should be 1");
    }

    #[test]
    fn mul_of_2p_inputs() {
        let mut two_p = [0xFFFFu32; 16];
        two_p[0] = 0xFFDA;
        // 2p * 2p === 0
        assert_eq!(limbs16_to_ref(&fe_mul(&two_p, &two_p)), [0u8; 32], "2p*2p");
        // 2p * (2p+1) === 0
        let mut two_p1 = two_p;
        two_p1[0] = 0xFFDB;
        assert_eq!(
            limbs16_to_ref(&fe_mul(&two_p, &two_p1)),
            [0u8; 32],
            "2p*(2p+1)"
        );
        // freezing first should also give 0
        let fz = fe_freeze(&two_p);
        assert_eq!(limbs16_to_ref(&fe_mul(&fz, &fz)), [0u8; 32], "freeze(2p)^2");
    }

    #[test]
    fn freeze_reduces_multiples_of_p() {
        // 2p = [0xFFDA, 0xFFFF*15] must freeze to 0.
        let mut two_p = [0xFFFFu32; 16];
        two_p[0] = 0xFFDA;
        assert_eq!(fe_pack(&two_p), [0u8; 32], "2p should pack to 0");
        // p itself = [0xFFED, 0xFFFF*14, 0x7FFF]
        let mut p = [0xFFFFu32; 16];
        p[0] = 0xFFED;
        p[15] = 0x7FFF;
        assert_eq!(fe_pack(&p), [0u8; 32], "p should pack to 0");
    }

    #[test]
    fn mul_of_zerorep_is_zero() {
        // E = (1 - 0) - 1 represented unreduced, times 1, must be 0.
        let one = fe_one();
        let zero = fe_zero();
        let inner = fe_sub(&one, &zero); // ~1, unreduced
        let e = fe_sub(&inner, &one); // ~0, unreduced (= 2^257-76)
        assert_eq!(limbs16_to_ref(&e), [0u8; 32], "e value should be 0 mod p");
        let prod = fe_mul(&e, &one);
        assert_eq!(limbs16_to_ref(&prod), [0u8; 32], "0 * 1 should be 0");
        assert_eq!(fe_pack(&prod), [0u8; 32], "0*1 packs to 0");
    }

    #[test]
    fn manual_256_doublings_then_add() {
        // Regression: the incomplete "hwcd-3" doubling drifted the neutral
        // element off-curve after a few doublings. With the complete formula,
        // doubling the identity 256 times must stay the identity, and adding B
        // then yields B (this is the leading-zero-bits path of scalarmult).
        let bp = basepoint(&BX_LE, &BY_LE);
        let id_c = ge_compress(&ge_identity());
        let mut q = ge_identity();
        for n in 0..256 {
            q = ge_double(&q);
            assert_eq!(ge_compress(&q), id_c, "after {} doublings Q drifted", n + 1);
        }
        let q = ge_add(&q, &bp);
        assert_eq!(
            ge_compress(&q),
            canonical_basepoint(),
            "256x double then +B != B"
        );
    }

    #[test]
    fn scalarmult_small_scalars() {
        let bp = basepoint(&BX_LE, &BY_LE);
        for k in 1u64..6 {
            let mut sc = [0u8; 32];
            sc[0] = k as u8;
            let q = ge_scalarmult_base(&bp, &sc);
            let expected = EdwardsPoint::mul_base(&Scalar::from(k)).compress().0;
            assert_eq!(ge_compress(&q), expected, "scalarmult by {k}");
        }
    }

    #[test]
    fn scalarmult_matches_dalek() {
        let bp = basepoint(&BX_LE, &BY_LE);
        for seed in 1u64..8 {
            let mut sc = [0u8; 32];
            sc[0] = (seed * 37) as u8;
            sc[1] = (seed * 11) as u8;
            sc[8] = seed as u8;
            let q = ge_scalarmult_base(&bp, &sc);
            let expected = EdwardsPoint::mul_base(&Scalar::from_bytes_mod_order(sc))
                .compress()
                .0;
            assert_eq!(ge_compress(&q), expected, "scalarmult seed {seed}");
        }
    }

    // Canonical byte-based field ops for a reference point double.
    fn fr_sub(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
        let mut o = [0u8; 32];
        let mut borrow = 0i16;
        for i in 0..32 {
            let v = a[i] as i16 - b[i] as i16 - borrow;
            if v < 0 {
                o[i] = (v + 256) as u8;
                borrow = 1;
            } else {
                o[i] = v as u8;
                borrow = 0;
            }
        }
        if borrow == 1 {
            let mut carry = 0u16;
            for i in 0..32 {
                let v = o[i] as u16 + P_BYTES[i] as u16 + carry;
                o[i] = (v & 0xFF) as u8;
                carry = v >> 8;
            }
        }
        o
    }
    fn fr_add(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
        let mut o = [0u8; 32];
        let mut carry = 0u16;
        for i in 0..32 {
            let v = a[i] as u16 + b[i] as u16 + carry;
            o[i] = (v & 0xFF) as u8;
            carry = v >> 8;
        }
        reduce_once(&mut o, carry);
        o
    }
    fn fr_neg(a: &[u8; 32]) -> [u8; 32] {
        fr_sub(&[0u8; 32], a)
    }

    #[test]
    fn reference_double_matches_dalek() {
        // Compute 2B using only canonical byte ops + the verified field_mul_p.
        // This decides whether the HWCD formula itself is correct.
        let m = |a: &[u8; 32], b: &[u8; 32]| crate::gpu::field_mul_p(a, b);
        let x = BX_LE;
        let y = BY_LE;
        let z = {
            let mut o = [0u8; 32];
            o[0] = 1;
            o
        };
        let two = {
            let mut o = [0u8; 32];
            o[0] = 2;
            o
        };
        let a = m(&x, &x);
        let b = m(&y, &y);
        let c = m(&m(&z, &z), &two);
        let d = fr_neg(&a);
        let xy = fr_add(&x, &y);
        let e = fr_sub(&fr_sub(&m(&xy, &xy), &a), &b);
        let g = fr_add(&d, &b);
        let f = fr_sub(&g, &c);
        let h = fr_sub(&d, &b);
        let x3 = m(&e, &f);
        let y3 = m(&g, &h);
        let z3 = m(&f, &g);
        let z_inv = ref_pow(&z3, &{
            let mut pm2 = P_BYTES;
            pm2[0] = pm2[0].wrapping_sub(2);
            pm2
        });
        let x_aff = m(&x3, &z_inv);
        let y_aff = m(&y3, &z_inv);
        let mut comp = y_aff;
        comp[31] |= (x_aff[0] & 1) << 7;
        let expected = EdwardsPoint::mul_base(&Scalar::from(2u64)).compress().0;
        assert_eq!(comp, expected, "reference HWCD double != dalek 2B");
    }

    #[test]
    fn mul_at_2_18() {
        // fe_mul with limbs around 2^18 (the regime ge_double's E,F,H reach).
        let big: Fe = [0x3FFFF; 16];
        let got = limbs16_to_ref(&fe_mul(&big, &big));
        let r = limbs16_to_ref(&big);
        let want = ref_mul(&r, &r);
        assert_eq!(got, want, "fe_mul at 2^18 limbs");
    }

    #[test]
    fn mul_handles_unreduced_inputs() {
        // feed sums (unreduced, limbs up to ~2^17) into fe_mul
        for seed in 0..50 {
            let (ab, a) = rand_reduced(seed);
            let (bb, b) = rand_reduced(seed.wrapping_add(5));
            let asum = fe_add(&a, &a); // 2a, unreduced
            let bsum = fe_add(&b, &b); // 2b
            let got = limbs16_to_ref(&fe_mul(&asum, &bsum));
            let two = {
                let mut o = [0u8; 32];
                o[0] = 2;
                o
            };
            let want = ref_mul(&ref_mul(&ab, &two), &ref_mul(&bb, &two));
            assert_eq!(got, want, "fe_mul unreduced seed {seed}");
        }
    }
}
