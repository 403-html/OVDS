use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::crypto::{Backend, MatchType};
use crate::gpu::GpuContext;

#[derive(Clone)]
pub struct WorkerState {
    pub attempts: Arc<AtomicU64>,
    pub stop: Arc<AtomicBool>,
    pub result: Arc<Mutex<Option<FoundResult>>>,
    pub error: Arc<Mutex<Option<String>>>,
}

impl WorkerState {
    pub fn new() -> Self {
        Self {
            attempts: Arc::new(AtomicU64::new(0)),
            stop: Arc::new(AtomicBool::new(false)),
            result: Arc::new(Mutex::new(None)),
            error: Arc::new(Mutex::new(None)),
        }
    }

    pub fn attempts(&self) -> u64 {
        self.attempts.load(Ordering::Relaxed)
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

#[derive(Clone)]
pub struct FoundResult {
    pub address: String,
    pub key_path: PathBuf,
}

pub enum Mode {
    Idle,
    Benchmarking {
        started: Instant,
        worker: WorkerState,
        backend: Backend,
    },
    Generating {
        started: Instant,
        worker: WorkerState,
        rate_tracker: Arc<Mutex<RateTracker>>,
    },
    Found {
        result: FoundResult,
        attempts: u64,
        elapsed: Duration,
    },
    #[allow(dead_code)]
    Error(String),
}

pub struct RateTracker {
    pub samples: Vec<(Instant, u64)>,
    pub last_rate: f64,
    pub history: VecDeque<u64>, // one sample per second, up to 80s
    last_sampled_at: Option<Instant>,
}

impl RateTracker {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            last_rate: 0.0,
            history: VecDeque::with_capacity(80),
            last_sampled_at: None,
        }
    }

    pub fn update(&mut self, now: Instant, total: u64) {
        self.samples.push((now, total));
        // keep a 2-second window for instantaneous rate
        let cutoff = now - Duration::from_secs(2);
        self.samples.retain(|(t, _)| *t >= cutoff);
        if self.samples.len() >= 2 {
            let (t0, n0) = self.samples.first().unwrap();
            let (t1, n1) = self.samples.last().unwrap();
            let dt = t1.duration_since(*t0).as_secs_f64();
            if dt > 0.01 {
                self.last_rate = (*n1 - *n0) as f64 / dt;
            }
        }
        // snapshot to history once per second
        let should_sample = self
            .last_sampled_at
            .map(|t| now.duration_since(t) >= Duration::from_secs(1))
            .unwrap_or(true);
        if should_sample && self.last_rate > 0.0 {
            self.history.push_back(self.last_rate as u64);
            if self.history.len() > 80 {
                self.history.pop_front();
            }
            self.last_sampled_at = Some(now);
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum FocusedPanel {
    Pattern,
    Actions,
}

impl FocusedPanel {
    pub fn toggle(&self) -> Self {
        match self {
            Self::Pattern => Self::Actions,
            Self::Actions => Self::Pattern,
        }
    }
}

pub struct App {
    pub pattern: String,
    pub match_type: MatchType,
    pub mode: Mode,
    pub cpu_benchmark_rate: Option<f64>,
    pub gpu_benchmark_rate: Option<f64>,
    pub threads: usize,
    pub backend: Backend,
    pub gpu: Option<Arc<GpuContext>>,
    pub gpu_init_error: Option<String>,
    pub status_msg: String,
    pub focused_panel: FocusedPanel,
    pub quit: bool,
}

impl App {
    pub fn new() -> Self {
        let (gpu, gpu_init_error) = match GpuContext::init() {
            Ok(ctx) => (Some(Arc::new(ctx)), None),
            Err(e) => (None, Some(e.to_string())),
        };
        Self {
            pattern: String::new(),
            match_type: MatchType::Prefix,
            mode: Mode::Idle,
            cpu_benchmark_rate: None,
            gpu_benchmark_rate: None,
            threads: rayon::current_num_threads(),
            backend: Backend::Cpu,
            gpu,
            gpu_init_error,
            status_msg: String::new(),
            focused_panel: FocusedPanel::Pattern,
            quit: false,
        }
    }

    pub fn toggle_backend(&mut self) {
        self.backend = self.backend.toggle();
        self.status_msg = match (self.backend, &self.gpu) {
            (Backend::Gpu, None) => format!(
                "GPU unavailable: {}",
                self.gpu_init_error
                    .clone()
                    .unwrap_or_else(|| "no adapter".into())
            ),
            (Backend::Gpu, Some(ctx)) => format!(
                "Backend: GPU ({} · {})",
                ctx.backend_label(),
                ctx.adapter_name
            ),
            (Backend::Cpu, _) => format!("Backend: CPU ({} threads)", self.threads),
        };
    }

    pub fn cycle_panel(&mut self) {
        self.focused_panel = self.focused_panel.toggle();
    }

    pub fn pattern_valid(&self) -> bool {
        !self.pattern.is_empty()
            && self
                .pattern
                .chars()
                .all(|c| "abcdefghijklmnopqrstuvwxyz234567".contains(c))
    }

    pub fn type_char(&mut self, c: char) {
        if self.pattern.len() < crate::crypto::ADDRESS_LEN
            && "abcdefghijklmnopqrstuvwxyz234567".contains(c)
        {
            self.pattern.push(c);
        }
    }

    pub fn backspace(&mut self) {
        self.pattern.pop();
    }

    pub fn cycle_match_type(&mut self, forward: bool) {
        self.match_type = if forward {
            self.match_type.next()
        } else {
            self.match_type.prev()
        };
    }

    pub fn start_benchmark(&mut self) {
        match self.backend {
            Backend::Cpu => self.start_benchmark_cpu(),
            Backend::Gpu => self.start_benchmark_gpu(),
        }
    }

    fn start_benchmark_cpu(&mut self) {
        let worker = WorkerState::new();
        let w = worker.clone();

        std::thread::spawn(move || {
            use crate::crypto::generate_keypair;
            rayon::scope(|_| {
                (0..rayon::current_num_threads()).for_each(|_| {
                    let attempts = Arc::clone(&w.attempts);
                    let stop = Arc::clone(&w.stop);
                    std::thread::spawn(move || {
                        let mut rng = rand::thread_rng();
                        while !stop.load(Ordering::Relaxed) {
                            generate_keypair(&mut rng);
                            attempts.fetch_add(1, Ordering::Relaxed);
                        }
                    });
                });
                std::thread::sleep(Duration::from_secs(5));
                w.stop();
            });
        });

        self.mode = Mode::Benchmarking {
            started: Instant::now(),
            worker,
            backend: Backend::Cpu,
        };
        self.status_msg = "Benchmarking CPU key generation...".into();
    }

    fn start_benchmark_gpu(&mut self) {
        let Some(gpu) = self.gpu.clone() else {
            self.status_msg = format!(
                "GPU unavailable: {}",
                self.gpu_init_error
                    .clone()
                    .unwrap_or_else(|| "no adapter".into())
            );
            return;
        };

        let status_label = format!(
            "Benchmarking GPU ({} · {}) - iterated SHA-256...",
            gpu.backend_label(),
            gpu.adapter_name
        );

        let worker = WorkerState::new();
        let w = worker.clone();
        let gpu_thread = Arc::clone(&gpu);

        std::thread::spawn(move || {
            let attempts = Arc::clone(&w.attempts);
            let stop = Arc::clone(&w.stop);
            let error = Arc::clone(&w.error);
            // Isolate any wgpu panic (e.g. shader validation) from the TUI.
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::gpu::run_bench(&gpu_thread, attempts, stop, 5.0)
            }));
            match res {
                Ok(Err(e)) => *error.lock().unwrap() = Some(format!("GPU error: {e}")),
                Err(_) => *error.lock().unwrap() = Some("GPU panicked (see ovds-panic.log)".into()),
                Ok(Ok(())) => {}
            }
            w.stop();
        });

        self.mode = Mode::Benchmarking {
            started: Instant::now(),
            worker,
            backend: Backend::Gpu,
        };
        self.status_msg = status_label;
    }

    pub fn start_generate(&mut self) {
        if !self.pattern_valid() {
            self.status_msg = "Invalid pattern - use a-z and 2-7 only".into();
            return;
        }

        match self.backend {
            Backend::Gpu if self.gpu.is_some() => {
                self.start_generate_gpu();
                return;
            }
            Backend::Gpu => {
                self.status_msg = format!(
                    "GPU unavailable: {}; using CPU",
                    self.gpu_init_error
                        .clone()
                        .unwrap_or_else(|| "no adapter".into())
                );
            }
            Backend::Cpu => {}
        }

        let worker = WorkerState::new();
        let rate_tracker = Arc::new(Mutex::new(RateTracker::new()));
        let w = worker.clone();
        let pattern = self.pattern.clone();
        let match_type = self.match_type.clone();

        std::thread::spawn(move || {
            use crate::crypto::{
                MatchType, check_prefix_fast, derive_address, derive_address_into, save_keys,
            };
            use ed25519_dalek::SigningKey;
            let nthreads = rayon::current_num_threads();
            let handles: Vec<_> = (0..nthreads)
                .map(|_| {
                    let attempts = Arc::clone(&w.attempts);
                    let stop = Arc::clone(&w.stop);
                    let result = Arc::clone(&w.result);
                    let pattern_bytes = pattern.as_bytes().to_vec();
                    let match_type = match_type.clone();
                    std::thread::spawn(move || {
                        let mut rng = rand::thread_rng();
                        let mut iter = 0u64;
                        if matches!(match_type, MatchType::Prefix) {
                            // Hot path: skip SHA3 - prefix chars come from pubkey bytes only
                            loop {
                                let key = SigningKey::generate(&mut rng);
                                let pubkey = key.verifying_key().to_bytes();
                                attempts.fetch_add(1, Ordering::Relaxed);
                                if check_prefix_fast(&pubkey, &pattern_bytes) {
                                    let address = derive_address(&key);
                                    let mut guard = result.lock().unwrap();
                                    if guard.is_none() {
                                        let key_path = save_keys(&key, &address)
                                            .unwrap_or_else(|_| PathBuf::from("."));
                                        *guard = Some(FoundResult { address, key_path });
                                        stop.store(true, Ordering::Relaxed);
                                    }
                                    break;
                                }
                                iter += 1;
                                if iter & 63 == 0 && stop.load(Ordering::Relaxed) {
                                    break;
                                }
                            }
                        } else {
                            let mut addr_buf = [0u8; 56];
                            loop {
                                let key = SigningKey::generate(&mut rng);
                                derive_address_into(&key, &mut addr_buf);
                                attempts.fetch_add(1, Ordering::Relaxed);
                                let matched = match &match_type {
                                    MatchType::Suffix => addr_buf.ends_with(&pattern_bytes),
                                    MatchType::Anywhere => addr_buf
                                        .windows(pattern_bytes.len())
                                        .any(|w| w == pattern_bytes.as_slice()),
                                    MatchType::Prefix => unreachable!(),
                                };
                                if matched {
                                    let address = String::from_utf8(addr_buf.to_vec()).unwrap();
                                    let mut guard = result.lock().unwrap();
                                    if guard.is_none() {
                                        let key_path = save_keys(&key, &address)
                                            .unwrap_or_else(|_| PathBuf::from("."));
                                        *guard = Some(FoundResult { address, key_path });
                                        stop.store(true, Ordering::Relaxed);
                                    }
                                    break;
                                }
                                iter += 1;
                                if iter & 63 == 0 && stop.load(Ordering::Relaxed) {
                                    break;
                                }
                            }
                        }
                    })
                })
                .collect();
            for h in handles {
                let _ = h.join();
            }
        });

        self.mode = Mode::Generating {
            started: Instant::now(),
            worker,
            rate_tracker,
        };
        self.status_msg = "Searching...".into();
    }

    fn start_generate_gpu(&mut self) {
        let Some(gpu) = self.gpu.clone() else {
            self.status_msg = "GPU unavailable".into();
            return;
        };

        let worker = WorkerState::new();
        let rate_tracker = Arc::new(Mutex::new(RateTracker::new()));
        let w = worker.clone();
        let pattern = self.pattern.clone().into_bytes();
        let match_type = self.match_type.clone();

        std::thread::spawn(move || {
            use crate::crypto::{address_from_pubkey, save_keys_expanded};
            let attempts = Arc::clone(&w.attempts);
            let stop = Arc::clone(&w.stop);
            let error = Arc::clone(&w.error);
            let result = Arc::clone(&w.result);

            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let pipe = crate::gpu::KeygenPipeline::new(&gpu)?;
                let found = Arc::new(Mutex::new(None));
                crate::gpu::run_keygen(
                    &gpu,
                    &pipe,
                    pattern,
                    match_type,
                    attempts,
                    stop,
                    Arc::clone(&found),
                )?;
                let kp = found.lock().unwrap().take();
                anyhow::Ok(kp)
            }));

            match res {
                Ok(Ok(Some(kp))) => {
                    let addr_bytes = address_from_pubkey(&kp.pubkey);
                    let address = String::from_utf8(addr_bytes.to_vec()).unwrap();
                    let key_path = save_keys_expanded(&kp.pubkey, &kp.scalar, &address)
                        .unwrap_or_else(|_| PathBuf::from("."));
                    *result.lock().unwrap() = Some(FoundResult { address, key_path });
                }
                Ok(Ok(None)) => {} // stopped without a match
                Ok(Err(e)) => *error.lock().unwrap() = Some(format!("GPU error: {e}")),
                Err(_) => *error.lock().unwrap() = Some("GPU panicked (see ovds-panic.log)".into()),
            }
            w.stop();
        });

        self.mode = Mode::Generating {
            started: Instant::now(),
            worker,
            rate_tracker,
        };
        self.status_msg = "Searching on GPU...".into();
    }

    pub fn stop(&mut self) {
        match &self.mode {
            Mode::Benchmarking { worker, .. } => worker.stop(),
            Mode::Generating { worker, .. } => worker.stop(),
            _ => {}
        }
        self.mode = Mode::Idle;
        self.status_msg = "Stopped.".into();
    }

    /// Poll worker state - call each tick.
    pub fn tick(&mut self) {
        match &mut self.mode {
            Mode::Benchmarking {
                started,
                worker,
                backend,
            } => {
                let elapsed = started.elapsed();
                if worker.stop.load(Ordering::Relaxed) || elapsed >= Duration::from_secs(6) {
                    let attempts = worker.attempts();
                    let secs = elapsed.as_secs_f64().max(0.1);
                    let rate = attempts as f64 / secs;
                    let err = worker.error.lock().unwrap().clone();
                    match backend {
                        Backend::Cpu => {
                            self.cpu_benchmark_rate = Some(rate);
                            self.status_msg = format!(
                                "CPU benchmark done: {:.0} keys/s on {} threads",
                                rate, self.threads
                            );
                        }
                        Backend::Gpu => {
                            if let Some(e) = err {
                                self.status_msg = e;
                            } else {
                                self.gpu_benchmark_rate = Some(rate);
                                self.status_msg =
                                    format!("GPU benchmark done: {:.0} SHA-256/s", rate);
                            }
                        }
                    }
                    self.mode = Mode::Idle;
                }
            }
            Mode::Generating {
                started,
                worker,
                rate_tracker,
            } => {
                let attempts = worker.attempts();
                let now = Instant::now();
                rate_tracker.lock().unwrap().update(now, attempts);

                // Snapshot result/error and drop the MutexGuards before mutating self.mode.
                let found = worker.result.lock().unwrap().clone();
                let err = worker.error.lock().unwrap().clone();
                if let Some(found) = found {
                    let elapsed = started.elapsed();
                    self.mode = Mode::Found {
                        result: found,
                        attempts,
                        elapsed,
                    };
                    self.status_msg = "Found!".into();
                } else if let Some(err) = err {
                    // GPU generate thread failed (e.g. shader/device error).
                    self.status_msg = err;
                    self.mode = Mode::Idle;
                }
            }
            _ => {}
        }
    }
}
