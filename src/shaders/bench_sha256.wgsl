// Iterated SHA-256 GPU benchmark.
// Each invocation hashes its own seed `iterations` times and writes the final
// 256-bit digest. The point isn't the digests; it's measuring real compute
// throughput on whatever backend (Metal/Vulkan/DX12) wgpu picked.

struct Params {
    iterations: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

var<private> K: array<u32, 64> = array<u32, 64>(
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
);

fn rotr(x: u32, n: u32) -> u32 {
    return (x >> n) | (x << (32u - n));
}

// SHA-256 of a single 64-byte block. `state` is initial hash, mutated in place.
fn sha256_block(state: ptr<function, array<u32, 8>>, block: ptr<function, array<u32, 16>>) {
    var w: array<u32, 64>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        w[i] = (*block)[i];
    }
    for (var i = 16u; i < 64u; i = i + 1u) {
        let x = w[i - 15u];
        let y = w[i - 2u];
        let s0 = rotr(x, 7u) ^ rotr(x, 18u) ^ (x >> 3u);
        let s1 = rotr(y, 17u) ^ rotr(y, 19u) ^ (y >> 10u);
        w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
    }

    var a = (*state)[0];
    var b = (*state)[1];
    var c = (*state)[2];
    var d = (*state)[3];
    var e = (*state)[4];
    var f = (*state)[5];
    var g = (*state)[6];
    var h = (*state)[7];

    for (var i = 0u; i < 64u; i = i + 1u) {
        let S1 = rotr(e, 6u) ^ rotr(e, 11u) ^ rotr(e, 25u);
        let ch = (e & f) ^ ((~e) & g);
        let t1 = h + S1 + ch + K[i] + w[i];
        let S0 = rotr(a, 2u) ^ rotr(a, 13u) ^ rotr(a, 22u);
        let mj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = S0 + mj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    (*state)[0] = (*state)[0] + a;
    (*state)[1] = (*state)[1] + b;
    (*state)[2] = (*state)[2] + c;
    (*state)[3] = (*state)[3] + d;
    (*state)[4] = (*state)[4] + e;
    (*state)[5] = (*state)[5] + f;
    (*state)[6] = (*state)[6] + g;
    (*state)[7] = (*state)[7] + h;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    var state: array<u32, 8> = array<u32, 8>(
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    );

    // 64-byte padded block: 8 bytes message (thread id + iter) + 0x80 + zeros + length
    var block: array<u32, 16>;
    block[0] = idx;
    block[1] = 0u;
    block[2] = 0x80000000u;
    for (var i = 3u; i < 15u; i = i + 1u) {
        block[i] = 0u;
    }
    block[15] = 64u; // 8 bytes = 64 bits

    for (var iter = 0u; iter < params.iterations; iter = iter + 1u) {
        block[0] = state[0] ^ idx;
        block[1] = state[1] ^ iter;
        sha256_block(&state, &block);
    }

    let base = idx * 8u;
    output[base + 0u] = state[0];
    output[base + 1u] = state[1];
    output[base + 2u] = state[2];
    output[base + 3u] = state[3];
    output[base + 4u] = state[4];
    output[base + 5u] = state[5];
    output[base + 6u] = state[6];
    output[base + 7u] = state[7];
}
