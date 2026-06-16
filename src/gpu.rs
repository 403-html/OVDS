//! GPU compute backend.
//!
//! v0.3.0 ships both the SHA-256 benchmark pipeline (run_bench) and a real
//! ed25519 scalar-mult kernel (KeygenPipeline) that produces clamped scalars
//! and compressed pubkeys on the device. The host then scans pubkeys against
//! the requested pattern using the existing CPU prefix fast-path.
//!
//! Targets:
//!   - Metal on macOS
//!   - Vulkan on Linux (including multi-GPU via wgpu adapter selection)
//!   - DX12 on Windows

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Result, anyhow};
use wgpu::util::DeviceExt;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend: wgpu::Backend,
}

impl GpuContext {
    /// Try to initialise a wgpu adapter. Returns Err if no compatible GPU.
    pub fn init() -> Result<Self> {
        pollster::block_on(Self::init_async())
    }

    async fn init_async() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow!("no compatible GPU adapter found"))?;

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend = info.backend;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ovds-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| anyhow!("device request failed: {e}"))?;

        Ok(Self {
            device,
            queue,
            adapter_name,
            backend,
        })
    }

    pub fn backend_label(&self) -> &'static str {
        match self.backend {
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Dx12 => "DX12",
            wgpu::Backend::Gl => "GL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            wgpu::Backend::Empty => "none",
        }
    }
}

/// Iterated SHA-256 benchmark shader. Each thread loops `iterations` SHA-256
/// compressions over its own seed and writes the final digest. Total hash ops
/// = threads * iterations.
const BENCH_SHADER: &str = include_str!("shaders/bench_sha256.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BenchParams {
    iterations: u32,
    _pad: [u32; 3],
}

/// Run iterated SHA-256 on GPU for ~`target_secs` seconds, updating
/// `total_hashes` as each dispatch completes. Stops early if `stop` is set.
pub fn run_bench(
    ctx: &GpuContext,
    total_hashes: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    target_secs: f64,
) -> Result<()> {
    let threads: u32 = 65_536;
    let workgroup_size: u32 = 64;
    let workgroups = threads / workgroup_size;
    let iters_per_dispatch: u32 = 1024;

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bench-sha256"),
            source: wgpu::ShaderSource::Wgsl(BENCH_SHADER.into()),
        });

    let params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&BenchParams {
                iterations: iters_per_dispatch,
                _pad: [0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let output_size = (threads as u64) * 8 * 4; // 8 u32 per thread
    let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bench-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bench-bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    let pl = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bench-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bench-pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: "main",
            cache: None,
            compilation_options: Default::default(),
        });

    let start = Instant::now();
    while !stop.load(Ordering::Relaxed) && start.elapsed().as_secs_f64() < target_secs {
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bench-enc"),
            });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bench-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        // Block until GPU finishes this batch so timing & counter stay honest.
        ctx.device.poll(wgpu::Maintain::Wait);
        let done = (threads as u64) * (iters_per_dispatch as u64);
        total_hashes.fetch_add(done, Ordering::Relaxed);
    }
    Ok(())
}

// ============================================================================
// Ed25519 GPU keygen (v0.3.0)
// ============================================================================

const KEYGEN_SHADER: &str = include_str!("shaders/ed25519_keygen.wgsl");

/// Standard Ed25519 basepoint B affine coordinates, little-endian 32-byte form.
/// B.x = 15112221349535400772501151409588531511454012693041857206046113283949847762202
pub const BX_LE: [u8; 32] = [
    0x1A, 0xD5, 0x25, 0x8F, 0x60, 0x2D, 0x56, 0xC9, 0xB2, 0xA7, 0x25, 0x95, 0x60, 0xC7, 0x2C, 0x69,
    0x5C, 0xDC, 0xD6, 0xFD, 0x31, 0xE2, 0xA4, 0xC0, 0xFE, 0x53, 0x6E, 0xCD, 0xD3, 0x36, 0x69, 0x21,
];
/// B.y = 4/5 mod p. Big-endian value 0x6666...6658; little-endian encoding is
/// 0x58 followed by 0x66 in every remaining byte (byte 31 included, high bit 0).
pub const BY_LE: [u8; 32] = [
    0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
];

/// Modular multiplication a * b mod (2^255 - 19), 32-byte LE in/out.
/// Used at startup to compute basepoint T = X*Y; never runs in the hot loop.
pub fn field_mul_p(a_bytes: &[u8; 32], b_bytes: &[u8; 32]) -> [u8; 32] {
    // 8 x 32-bit limbs. Column sums stay under 8 * 2^64 = 2^67, well within u128.
    let mut a = [0u64; 8];
    let mut b = [0u64; 8];
    for i in 0..8 {
        a[i] = u32::from_le_bytes(a_bytes[4 * i..4 * i + 4].try_into().unwrap()) as u64;
        b[i] = u32::from_le_bytes(b_bytes[4 * i..4 * i + 4].try_into().unwrap()) as u64;
    }
    let mut t = [0u128; 15];
    for i in 0..8 {
        for j in 0..8 {
            t[i + j] += (a[i] * b[j]) as u128;
        }
    }
    // Fold columns 8..14 (weight 2^256+) into 0..6 via *38.
    for i in 0..7 {
        t[i] += 38 * t[i + 8];
    }
    // Carry-propagate into 32-bit limbs.
    let mut r = [0u64; 8];
    let mut carry: u128 = 0;
    for i in 0..8 {
        let v = t[i] + carry;
        r[i] = (v & 0xFFFFFFFF) as u64;
        carry = v >> 32;
    }
    // Remaining carry folds to limb 0 via *38.
    let mut c = 38 * carry;
    for limb in r.iter_mut() {
        let v = *limb as u128 + (c & 0xFFFFFFFF);
        *limb = (v & 0xFFFFFFFF) as u64;
        c = (c >> 32) + (v >> 32);
    }
    // Pack to bytes, then conditional subtract p up to twice.
    let mut out = [0u8; 32];
    for i in 0..8 {
        out[4 * i..4 * i + 4].copy_from_slice(&(r[i] as u32).to_le_bytes());
    }
    let p: [u8; 32] = {
        let mut pp = [0xFFu8; 32];
        pp[0] = 0xED;
        pp[31] = 0x7F;
        pp
    };
    for _ in 0..2 {
        let mut ge = false;
        for i in (0..32).rev() {
            if out[i] != p[i] {
                ge = out[i] > p[i];
                break;
            }
            if i == 0 {
                ge = true;
            }
        }
        if ge {
            let mut borrow = 0i16;
            for i in 0..32 {
                let d = out[i] as i16 - p[i] as i16 - borrow;
                if d < 0 {
                    out[i] = (d + 256) as u8;
                    borrow = 1;
                } else {
                    out[i] = d as u8;
                    borrow = 0;
                }
            }
        }
    }
    out
}

/// Convert 32-byte LE field element to 16-limb form (16 bits per limb, u32 storage).
fn bytes_to_fe16(b: &[u8; 32]) -> [u32; 16] {
    let mut r = [0u32; 16];
    for i in 0..16 {
        r[i] = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as u32;
    }
    r
}

pub struct KeygenPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    pub threads: u32,
    workgroups: u32,
    bx: [u32; 16],
    by: [u32; 16],
    bz: [u32; 16],
    bt: [u32; 16],
}

const KEYGEN_THREADS: u32 = 8192;
const KEYGEN_WORKGROUP_SIZE: u32 = 64;

impl KeygenPipeline {
    pub fn new(ctx: &GpuContext) -> Result<Self> {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ed25519-keygen"),
                source: wgpu::ShaderSource::Wgsl(KEYGEN_SHADER.into()),
            });
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("keygen-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("keygen-pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("keygen-pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });
        let bx = bytes_to_fe16(&BX_LE);
        let by = bytes_to_fe16(&BY_LE);
        let mut bz = [0u32; 16];
        bz[0] = 1;
        let bt_bytes = field_mul_p(&BX_LE, &BY_LE);
        let bt = bytes_to_fe16(&bt_bytes);
        Ok(Self {
            pipeline,
            bgl,
            threads: KEYGEN_THREADS,
            workgroups: KEYGEN_THREADS / KEYGEN_WORKGROUP_SIZE,
            bx,
            by,
            bz,
            bt,
        })
    }

    fn make_params(&self, base_seed: &[u8; 32], batch_id: u32) -> [u32; 76] {
        let mut p = [0u32; 76];
        for i in 0..8 {
            p[i] = u32::from_le_bytes(base_seed[4 * i..4 * i + 4].try_into().unwrap());
        }
        p[8] = batch_id;
        p[9] = self.threads;
        p[12..28].copy_from_slice(&self.bx);
        p[28..44].copy_from_slice(&self.by);
        p[44..60].copy_from_slice(&self.bz);
        p[60..76].copy_from_slice(&self.bt);
        p
    }
}

#[derive(Clone, Copy)]
pub struct GpuKeyPair {
    pub pubkey: [u8; 32],
    pub scalar: [u8; 32],
}

/// One dispatch: produces `pipe.threads` keypairs from (base_seed, batch_id) and
/// returns them parsed as (pubkey, scalar) pairs.
pub fn keygen_dispatch(
    ctx: &GpuContext,
    pipe: &KeygenPipeline,
    base_seed: &[u8; 32],
    batch_id: u32,
) -> Result<Vec<GpuKeyPair>> {
    let params = pipe.make_params(base_seed, batch_id);
    let params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("keygen-params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
    let output_size = (pipe.threads as u64) * 16 * 4;
    let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("keygen-output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("keygen-staging"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("keygen-bg"),
        layout: &pipe.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("keygen-enc"),
        });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("keygen-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipe.pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(pipe.workgroups, 1, 1);
    }
    enc.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
    ctx.queue.submit(Some(enc.finish()));

    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| anyhow!("map_async recv: {e}"))?
        .map_err(|e| anyhow!("map_async: {e:?}"))?;
    let raw = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&raw);
    let mut pairs = Vec::with_capacity(pipe.threads as usize);
    for i in 0..pipe.threads as usize {
        let off = i * 16;
        let mut pubkey = [0u8; 32];
        let mut scalar = [0u8; 32];
        for j in 0..8 {
            pubkey[4 * j..4 * j + 4].copy_from_slice(&words[off + j].to_le_bytes());
            scalar[4 * j..4 * j + 4].copy_from_slice(&words[off + 8 + j].to_le_bytes());
        }
        pairs.push(GpuKeyPair { pubkey, scalar });
    }
    drop(raw);
    staging_buf.unmap();
    Ok(pairs)
}

/// GPU search loop: dispatch batches until match or stop.
pub fn run_keygen(
    ctx: &GpuContext,
    pipe: &KeygenPipeline,
    pattern: Vec<u8>,
    match_type: crate::crypto::MatchType,
    attempts: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    result: Arc<Mutex<Option<GpuKeyPair>>>,
) -> Result<()> {
    use crate::crypto::{MatchType, address_from_pubkey, check_prefix_fast};
    use rand::RngCore;
    let mut rng = rand::thread_rng();
    let mut batch_id: u32 = 0;
    let mut base_seed = [0u8; 32];
    while !stop.load(Ordering::Relaxed) {
        rng.fill_bytes(&mut base_seed);
        let pairs = keygen_dispatch(ctx, pipe, &base_seed, batch_id)?;
        attempts.fetch_add(pipe.threads as u64, Ordering::Relaxed);
        for kp in &pairs {
            let matched = match match_type {
                MatchType::Prefix => check_prefix_fast(&kp.pubkey, &pattern),
                MatchType::Suffix => {
                    let addr = address_from_pubkey(&kp.pubkey);
                    addr.ends_with(pattern.as_slice())
                }
                MatchType::Anywhere => {
                    let addr = address_from_pubkey(&kp.pubkey);
                    addr.windows(pattern.len()).any(|w| w == pattern.as_slice())
                }
            };
            if matched {
                *result.lock().unwrap() = Some(*kp);
                stop.store(true, Ordering::Relaxed);
                return Ok(());
            }
        }
        batch_id = batch_id.wrapping_add(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirror of the WGSL `derive_scalar` so we can reproduce the GPU's clamped
    /// scalar bytes on the host.
    fn derive_scalar_host(base_seed: &[u8; 32], batch_id: u32, idx: u32) -> [u8; 32] {
        let bs: [u32; 8] = std::array::from_fn(|i| {
            u32::from_le_bytes(base_seed[4 * i..4 * i + 4].try_into().unwrap())
        });
        let mut s = [0u32; 8];
        for i in 0..8u32 {
            let inner = 0x9E3779B9u32.wrapping_add(i.wrapping_mul(0x6F4A7855u32));
            let scaled = idx.wrapping_mul(inner);
            let with_batch = scaled.wrapping_add(batch_id.wrapping_mul(0xBB67AE85u32));
            s[i as usize] = bs[i as usize] ^ with_batch;
        }
        let a = s[0];
        let b = s[7];
        s[0] = a ^ (b << 13) ^ (b >> 19);
        s[7] = b ^ (a << 7) ^ (a >> 25);
        s[0] &= 0xFFFFFFF8;
        s[7] = (s[7] & 0x7FFFFFFF) | 0x40000000;
        let mut out = [0u8; 32];
        for i in 0..8 {
            out[4 * i..4 * i + 4].copy_from_slice(&s[i].to_le_bytes());
        }
        out
    }

    /// Full-pipeline validation: for each GPU thread, the device-computed pubkey
    /// must equal curve25519-dalek's scalar*B for the GPU's own clamped scalar.
    /// This is the regression guard for the entire WGSL ed25519 implementation.
    #[test]
    fn gpu_keygen_matches_dalek() {
        use curve25519_dalek::edwards::EdwardsPoint;
        use curve25519_dalek::scalar::Scalar;

        let Ok(ctx) = GpuContext::init() else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };
        let pipe = KeygenPipeline::new(&ctx).expect("pipeline init");
        // Exercise a couple of batches/seeds so we cover varied scalars.
        for (base_seed, batch_id) in [([0u8; 32], 0u32), ([0x5Au8; 32], 7u32)] {
            let pairs = keygen_dispatch(&ctx, &pipe, &base_seed, batch_id).expect("dispatch");
            for (i, kp) in pairs.iter().enumerate().step_by(101) {
                let expected_scalar = derive_scalar_host(&base_seed, batch_id, i as u32);
                assert_eq!(kp.scalar, expected_scalar, "thread {i}: scalar mismatch");
                let s = Scalar::from_bytes_mod_order(expected_scalar);
                let expected_pub = EdwardsPoint::mul_base(&s).compress().0;
                assert_eq!(
                    kp.pubkey, expected_pub,
                    "thread {i}: GPU pubkey != dalek scalar*B"
                );
            }
        }
    }

    /// End-to-end: the GPU search loop must find a short prefix and the matched
    /// pubkey's full Tor address must actually start with the pattern.
    #[test]
    fn gpu_search_finds_prefix() {
        use crate::crypto::{MatchType, address_from_pubkey};

        let Ok(ctx) = GpuContext::init() else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };
        let pipe = KeygenPipeline::new(&ctx).expect("pipeline init");
        let pattern = b"a".to_vec(); // 1 char: ~1/32, found within the first batch
        let attempts = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let result = Arc::new(Mutex::new(None));
        run_keygen(
            &ctx,
            &pipe,
            pattern.clone(),
            MatchType::Prefix,
            attempts,
            stop,
            Arc::clone(&result),
        )
        .expect("keygen run");
        let kp = result.lock().unwrap().take().expect("should find a match");
        let addr = address_from_pubkey(&kp.pubkey);
        assert!(
            addr.starts_with(pattern.as_slice()),
            "matched address {:?} does not start with pattern",
            std::str::from_utf8(&addr).unwrap()
        );
    }
}
