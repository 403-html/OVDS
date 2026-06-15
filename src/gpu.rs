//! GPU compute backend.
//!
//! v0.2.0 ships the device init + a real wgpu compute pipeline used for the
//! GPU benchmark (iterated SHA-256). The full ed25519-on-GPU keygen kernel is
//! v0.3.0 work; `gpu_generate` currently delegates to CPU with a clear status.
//!
//! Targets:
//!   - Metal on macOS
//!   - Vulkan on Linux (including multi-GPU via wgpu adapter selection)
//!   - DX12 on Windows

use std::sync::Arc;
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
