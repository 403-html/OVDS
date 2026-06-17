//! GPU compute backend.
//!
//! Ships the ed25519 keygen pipeline (KeygenPipeline) plus a benchmark
//! (run_keygen_bench) that measures real key-generation throughput on the same
//! kernel. Every mode uses incremental generation with batched inversion (one
//! point add per candidate, amortized). Prefix and anywhere patterns match
//! on-device and return only the matches (tiny readback). Suffix (and patterns
//! too long for the on-device matcher) use the same incremental kernel in
//! write-all mode and are scanned on the host in parallel with Rayon.
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

        // The incremental keygen needs a large scratch buffer (one binding holds
        // threads * BATCH_K points), well past downlevel defaults, so request the
        // adapter's real limits.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ovds-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
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

// ============================================================================
// Ed25519 GPU keygen
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

/// Dispatch modes (must match the `mode` param in ed25519_keygen.wgsl).
const MODE_PREFIX: u32 = 0; // on-device prefix match + atomic compaction
const MODE_WRITE_ALL: u32 = 1; // incremental, every candidate written by flat index
const MODE_ANYWHERE: u32 = 2; // on-device anywhere match + atomic compaction

const KEYGEN_THREADS: u32 = 16_384;
const KEYGEN_WORKGROUP_SIZE: u32 = 64;
/// Candidates generated per thread in prefix mode (mirrors BATCH_K in the shader).
const BATCH_K: u32 = 64;
const SCRATCH_WORDS_PER: u64 = 64; // 4 field elements * 16 limbs
// Comb table geometry (mirrors COMB_* in the shader).
const COMB_WINDOWS: u32 = 32;
const COMB_PTS_PER_WIN: u32 = 256;
const GE_WORDS: u32 = 64;
const TABLE_WORDS: u64 = (COMB_WINDOWS * COMB_PTS_PER_WIN * GE_WORDS) as u64;
const OUT_HEADER: usize = 4;
const MAX_MATCHES: usize = 256;
const PARAMS_WORDS: usize = 128;
/// Longest prefix the on-device base32 match supports: char j reads bytes
/// `5j/8` and `5j/8 + 1`, so j <= 47 keeps both bytes within the 32-byte pubkey.
use crate::crypto::MAX_DEVICE_PREFIX;

pub struct KeygenPipeline {
    main_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    pub threads: u32,
    workgroups: u32,
    /// Params with the basepoint (and threads) pre-filled; per-dispatch fields
    /// (seed, batch, mode, pattern) are overwritten before each submit.
    base_params: [u32; PARAMS_WORDS],
    output_size: u64,
}

impl KeygenPipeline {
    pub fn new(ctx: &GpuContext) -> Result<Self> {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ed25519-keygen"),
                source: wgpu::ShaderSource::Wgsl(KEYGEN_SHADER.into()),
            });
        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("keygen-bgl"),
                entries: &[
                    storage_entry(0, true),  // params
                    storage_entry(1, false), // comb table (built then read)
                    storage_entry(2, false), // output (atomic)
                    storage_entry(3, false), // incremental scratch
                ],
            });
        let pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("keygen-pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let make_pipeline = |entry: &'static str| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: entry,
                    cache: None,
                    compilation_options: Default::default(),
                })
        };
        let build_pipeline = make_pipeline("build_table");
        let main_pipeline = make_pipeline("main");

        // Basepoint extended coords (Z=1, T=X*Y) for the table builder.
        let bx = bytes_to_fe16(&BX_LE);
        let by = bytes_to_fe16(&BY_LE);
        let mut bz = [0u32; 16];
        bz[0] = 1;
        let bt = bytes_to_fe16(&field_mul_p(&BX_LE, &BY_LE));

        let mut base_params = [0u32; PARAMS_WORDS];
        base_params[9] = KEYGEN_THREADS;
        base_params[12..28].copy_from_slice(&bx);
        base_params[28..44].copy_from_slice(&by);
        base_params[44..60].copy_from_slice(&bz);
        base_params[60..76].copy_from_slice(&bt);

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("keygen-params"),
                contents: bytemuck::cast_slice(&base_params),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let table_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("keygen-table"),
            size: TABLE_WORDS * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // Sized for write-all mode (the largest layout): every candidate
        // (threads * BATCH_K) writes a 16-u32 record. Compaction modes use only
        // the small header + matches region at the front of the same buffer.
        let output_size = (KEYGEN_THREADS as u64) * (BATCH_K as u64) * 16 * 4;
        let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("keygen-output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("keygen-staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Per-candidate scratch (prefix mode). threads * BATCH_K * 4 field elems.
        let scratch_size = (KEYGEN_THREADS as u64) * (BATCH_K as u64) * SCRATCH_WORDS_PER * 4;
        let scratch_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("keygen-scratch"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("keygen-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scratch_buf.as_entire_binding(),
                },
            ],
        });

        // Build the comb table once. It stays resident in table_buf and is read
        // by every subsequent main dispatch.
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("table-build-enc"),
            });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("table-build-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&build_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);

        Ok(Self {
            main_pipeline,
            bind_group,
            params_buf,
            output_buf,
            staging_buf,
            threads: KEYGEN_THREADS,
            workgroups: KEYGEN_THREADS / KEYGEN_WORKGROUP_SIZE,
            base_params,
            output_size,
        })
    }
}

#[derive(Clone, Copy)]
pub struct GpuKeyPair {
    pub pubkey: [u8; 32],
    pub scalar: [u8; 32],
}

fn parse_pair(words: &[u32], off: usize) -> GpuKeyPair {
    let mut pubkey = [0u8; 32];
    let mut scalar = [0u8; 32];
    for j in 0..8 {
        pubkey[4 * j..4 * j + 4].copy_from_slice(&words[off + j].to_le_bytes());
        scalar[4 * j..4 * j + 4].copy_from_slice(&words[off + 8 + j].to_le_bytes());
    }
    GpuKeyPair { pubkey, scalar }
}

/// Map a base32 pattern (lowercase alphabet) to its 5-bit symbol values for the
/// on-device matcher. Returns None if any char is outside the base32 alphabet.
fn pattern_to_symbols(pattern: &[u8]) -> Option<Vec<u32>> {
    let alphabet = crate::crypto::VALID_CHARS.as_bytes();
    pattern
        .iter()
        .map(|c| alphabet.iter().position(|a| a == c).map(|p| p as u32))
        .collect()
}

/// One dispatch. In prefix mode the device compacts matches and we read back
/// only the small header + matches region; in write-all mode every thread's
/// keypair is read back by index.
fn keygen_dispatch(
    ctx: &GpuContext,
    pipe: &KeygenPipeline,
    base_seed: &[u8; 32],
    batch_id: u32,
    mode: u32,
    pattern_syms: &[u32],
) -> Result<Vec<GpuKeyPair>> {
    let mut params = pipe.base_params;
    for i in 0..8 {
        params[i] = u32::from_le_bytes(base_seed[4 * i..4 * i + 4].try_into().unwrap());
    }
    params[8] = batch_id;
    params[10] = mode;
    let compaction = mode != MODE_WRITE_ALL;
    if compaction {
        params[11] = pattern_syms.len() as u32;
        params[76..76 + pattern_syms.len()].copy_from_slice(pattern_syms);
    }
    ctx.queue
        .write_buffer(&pipe.params_buf, 0, bytemuck::cast_slice(&params));
    // Reset the match counter before each compaction batch.
    if compaction {
        ctx.queue.write_buffer(&pipe.output_buf, 0, &[0u8; 4]);
    }

    // Only read back what each mode produces.
    let copy_size: u64 = if compaction {
        ((OUT_HEADER + MAX_MATCHES * 16) * 4) as u64
    } else {
        pipe.output_size
    };

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
        cpass.set_pipeline(&pipe.main_pipeline);
        cpass.set_bind_group(0, &pipe.bind_group, &[]);
        cpass.dispatch_workgroups(pipe.workgroups, 1, 1);
    }
    enc.copy_buffer_to_buffer(&pipe.output_buf, 0, &pipe.staging_buf, 0, copy_size);
    ctx.queue.submit(Some(enc.finish()));

    let slice = pipe.staging_buf.slice(..copy_size);
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

    let pairs = if compaction {
        let count = (words[0] as usize).min(MAX_MATCHES);
        (0..count)
            .map(|i| parse_pair(words, OUT_HEADER + i * 16))
            .collect()
    } else {
        (0..pipe.threads as usize * BATCH_K as usize)
            .map(|i| parse_pair(words, i * 16))
            .collect()
    };
    drop(raw);
    pipe.staging_buf.unmap();
    Ok(pairs)
}

/// Pick the dispatch mode and on-device pattern symbols for a match type. Prefix
/// and anywhere patterns short enough for the on-device matcher compact their hits
/// on the device (tiny readback); everything else (suffix, or over-long patterns)
/// uses the same incremental kernel in write-all mode and is scanned on the host.
fn pick_mode(match_type: &crate::crypto::MatchType, pattern: &[u8]) -> (u32, Vec<u32>) {
    use crate::crypto::MatchType;
    let device_ok = pattern.len() <= MAX_DEVICE_PREFIX && pattern_to_symbols(pattern).is_some();
    match match_type {
        MatchType::Prefix if device_ok => (MODE_PREFIX, pattern_to_symbols(pattern).unwrap()),
        MatchType::Anywhere if device_ok => (MODE_ANYWHERE, pattern_to_symbols(pattern).unwrap()),
        _ => (MODE_WRITE_ALL, Vec::new()),
    }
}

/// Host-side re-verification of a candidate against the full address. Device-matched
/// candidates are double-checked here; write-all candidates are scanned here.
fn host_match(match_type: &crate::crypto::MatchType, pattern: &[u8], kp: &GpuKeyPair) -> bool {
    use crate::crypto::{MatchType, address_from_pubkey, check_prefix_fast};
    match match_type {
        MatchType::Prefix => check_prefix_fast(&kp.pubkey, pattern),
        MatchType::Suffix => address_from_pubkey(&kp.pubkey).ends_with(pattern),
        MatchType::Anywhere => {
            let addr = address_from_pubkey(&kp.pubkey);
            addr.windows(pattern.len()).any(|w| w == pattern)
        }
    }
}

/// GPU search loop: dispatch batches until match or stop. Prefix patterns short
/// enough for the on-device matcher run in compaction mode (tiny readback);
/// suffix/anywhere (and over-long prefixes) fall back to write-all + host scan.
pub fn run_keygen(
    ctx: &GpuContext,
    pipe: &KeygenPipeline,
    pattern: Vec<u8>,
    match_type: crate::crypto::MatchType,
    attempts: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    result: Arc<Mutex<Option<GpuKeyPair>>>,
) -> Result<()> {
    use rand::RngCore;
    use rayon::prelude::*;

    let (mode, pattern_syms) = pick_mode(&match_type, &pattern);
    // Every mode walks BATCH_K candidates per thread.
    let per_dispatch = pipe.threads as u64 * BATCH_K as u64;

    let mut rng = rand::thread_rng();
    let mut batch_id: u32 = 0;
    let mut base_seed = [0u8; 32];
    while !stop.load(Ordering::Relaxed) {
        rng.fill_bytes(&mut base_seed);
        let pairs = keygen_dispatch(ctx, pipe, &base_seed, batch_id, mode, &pattern_syms)?;
        attempts.fetch_add(per_dispatch, Ordering::Relaxed);
        // Write-all returns ~1M candidates to scan (full SHA3 per key), so fan it
        // out across cores; compaction modes return only a handful of hits.
        let hit = if mode == MODE_WRITE_ALL {
            pairs
                .par_iter()
                .find_any(|kp| host_match(&match_type, &pattern, kp))
                .copied()
        } else {
            pairs
                .iter()
                .find(|kp| host_match(&match_type, &pattern, kp))
                .copied()
        };
        if let Some(kp) = hit {
            *result.lock().unwrap() = Some(kp);
            stop.store(true, Ordering::Relaxed);
            return Ok(());
        }
        batch_id = batch_id.wrapping_add(1);
    }
    Ok(())
}

/// Benchmark real ed25519 keygen throughput for ~`target_secs` in the same
/// dispatch mode a search of `match_type` would use, so the reported keys/s
/// reflects the actual generate path (write-all suffix includes its parallel host
/// scan; prefix/anywhere include the on-device match). A long representative
/// pattern keeps the search from short-circuiting during the window; matches are
/// ignored - this measures generation rate, not search time.
pub fn run_keygen_bench(
    ctx: &GpuContext,
    match_type: crate::crypto::MatchType,
    attempts: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    target_secs: f64,
) -> Result<()> {
    use rand::RngCore;
    use rayon::prelude::*;
    let pipe = KeygenPipeline::new(ctx)?;
    let pattern = b"ovdsbench42".to_vec(); // 11 valid base32 chars; never matches
    let (mode, pattern_syms) = pick_mode(&match_type, &pattern);
    let per_dispatch = pipe.threads as u64 * BATCH_K as u64;
    let mut rng = rand::thread_rng();
    let mut base_seed = [0u8; 32];
    let mut batch_id: u32 = 0;
    let start = Instant::now();
    while !stop.load(Ordering::Relaxed) && start.elapsed().as_secs_f64() < target_secs {
        rng.fill_bytes(&mut base_seed);
        let pairs = keygen_dispatch(ctx, &pipe, &base_seed, batch_id, mode, &pattern_syms)?;
        // Write-all measures generation plus the parallel host scan it really does.
        if mode == MODE_WRITE_ALL {
            let _ = std::hint::black_box(
                pairs
                    .par_iter()
                    .any(|kp| host_match(&match_type, &pattern, kp)),
            );
        }
        attempts.fetch_add(per_dispatch, Ordering::Relaxed);
        batch_id = batch_id.wrapping_add(1);
    }
    Ok(())
}

/// Raw keygen-dispatch throughput (keys/s) over `runs` rounds of `n` dispatches
/// each, on an already-built pipeline so the comb-table build and PSO compile are
/// excluded from timing. Returns one keys/s sample per round (take the median;
/// expect noise when the GPU is shared). Prefix mode + a never-matching pattern,
/// so it measures pure generation. Drives the ignored `bench` throughput test;
/// not on the UI path. Lives here to reach the crate-private dispatch internals.
#[cfg(test)]
pub(crate) fn bench_dispatch_rate(ctx: &GpuContext, runs: u32, n: u32) -> Result<Vec<f64>> {
    let pipe = KeygenPipeline::new(ctx)?;
    let syms = pattern_to_symbols(b"ovdsbench42").unwrap();
    let per = pipe.threads as u64 * BATCH_K as u64;
    keygen_dispatch(ctx, &pipe, &[0u8; 32], 0, MODE_PREFIX, &syms)?; // warm up
    let mut samples = Vec::with_capacity(runs as usize);
    for run in 0..runs {
        let start = Instant::now();
        for b in 0..n {
            let seed = [(run as u8).wrapping_add(b as u8); 32];
            keygen_dispatch(ctx, &pipe, &seed, b, MODE_PREFIX, &syms)?;
        }
        let secs = start.elapsed().as_secs_f64();
        samples.push((per * n as u64) as f64 / secs);
    }
    Ok(samples)
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

    /// Mirror of the WGSL `scalar_add_u32`: add a small integer to a 256-bit LE
    /// scalar with carry propagation.
    fn scalar_add_host(mut s: [u8; 32], j: u32) -> [u8; 32] {
        let mut carry = j as u64;
        for byte in s.iter_mut() {
            let v = *byte as u64 + (carry & 0xFF);
            *byte = v as u8;
            carry = (carry >> 8) + (v >> 8);
        }
        s
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
        // Exercise a couple of batches/seeds so we cover varied scalars. Write-all
        // now returns every incremental candidate (threads * BATCH_K): flat index
        // i maps to thread = i / BATCH_K, step j = i % BATCH_K, and the candidate
        // scalar is s0(thread) + j. Sampling across the flat range covers both the
        // comb start point (j == 0) and the incremental adds (j > 0).
        for (base_seed, batch_id) in [([0u8; 32], 0u32), ([0x5Au8; 32], 7u32)] {
            let pairs = keygen_dispatch(&ctx, &pipe, &base_seed, batch_id, MODE_WRITE_ALL, &[])
                .expect("dispatch");
            for (i, kp) in pairs.iter().enumerate().step_by(1009) {
                let thread = i as u32 / BATCH_K;
                let j = i as u32 % BATCH_K;
                let s0 = derive_scalar_host(&base_seed, batch_id, thread);
                let expected_scalar = scalar_add_host(s0, j);
                assert_eq!(kp.scalar, expected_scalar, "candidate {i}: scalar mismatch");
                let s = Scalar::from_bytes_mod_order(expected_scalar);
                let expected_pub = EdwardsPoint::mul_base(&s).compress().0;
                assert_eq!(
                    kp.pubkey, expected_pub,
                    "candidate {i}: GPU pubkey != dalek scalar*B"
                );
            }
        }
    }

    /// End-to-end: the GPU search loop must find a short anywhere pattern via the
    /// on-device anywhere matcher, and the matched address must actually contain
    /// the pattern (and the recovered scalar must reproduce the pubkey).
    #[test]
    fn gpu_search_finds_anywhere() {
        use crate::crypto::{MatchType, address_from_pubkey};
        use curve25519_dalek::{edwards::EdwardsPoint, scalar::Scalar};

        let Ok(ctx) = GpuContext::init() else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };
        let pipe = KeygenPipeline::new(&ctx).expect("pipeline init");
        let pattern = b"aa".to_vec(); // 2 chars: found within the first batch
        let attempts = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let result = Arc::new(Mutex::new(None));
        run_keygen(
            &ctx,
            &pipe,
            pattern.clone(),
            MatchType::Anywhere,
            attempts,
            stop,
            Arc::clone(&result),
        )
        .expect("keygen run");
        let kp = result.lock().unwrap().take().expect("should find a match");
        let addr = address_from_pubkey(&kp.pubkey);
        assert!(
            addr.windows(pattern.len()).any(|w| w == pattern.as_slice()),
            "matched address {:?} does not contain pattern",
            std::str::from_utf8(&addr).unwrap()
        );
        let s = Scalar::from_bytes_mod_order(kp.scalar);
        assert_eq!(
            EdwardsPoint::mul_base(&s).compress().0,
            kp.pubkey,
            "recovered scalar*B != matched pubkey"
        );
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
        // The incremental path returns scalar = s0 + j; verify it actually
        // produces the matched pubkey (so the saved key is valid).
        use curve25519_dalek::{edwards::EdwardsPoint, scalar::Scalar};
        let s = Scalar::from_bytes_mod_order(kp.scalar);
        assert_eq!(
            EdwardsPoint::mul_base(&s).compress().0,
            kp.pubkey,
            "recovered scalar*B != matched pubkey"
        );
    }
}
