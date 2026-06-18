use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span, Text},
    widgets::{Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Wrap},
};

use crate::app::{App, FocusedPanel, Mode, SearchField};
use crate::crypto::{ADDRESS_LEN, Backend, MatchType, validate_pattern};
use crate::stats::{
    cdf, expected_attempts, format_count, format_duration, format_rate, match_probability,
    quantile_attempts,
};

const ACCENT: Color = Color::Cyan;
const DIM: Color = Color::DarkGray;
const OK: Color = Color::Green;
const WARN: Color = Color::Yellow;
const ERR: Color = Color::Red;
const BRIGHT: Color = Color::White;

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();

    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // header (title + mode)
            Constraint::Min(0),    // body
            Constraint::Length(1), // footer
        ])
        .split(area);

    draw_header(f, app, root[0]);
    draw_body(f, app, root[1]);
    draw_footer(f, app, root[2]);
}

fn mode_label(app: &App) -> (&'static str, Color) {
    match &app.mode {
        Mode::Idle => ("estimate", BRIGHT),
        Mode::Benchmarking { .. } => ("benchmark", ACCENT),
        Mode::Generating { .. } => ("search & save keypair", ACCENT),
        Mode::Found { .. } => ("keypair saved", OK),
        Mode::Error(_) => ("error", ERR),
    }
}

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(area);

    let title_line = Line::from(vec![
        Span::styled(
            " OVDS",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  onion vanity domain search", Style::default().fg(DIM)),
        Span::styled(
            format!(
                "{:>width$}",
                concat!("v", env!("CARGO_PKG_VERSION"), " "),
                width = area.width as usize - 32
            ),
            Style::default().fg(DIM),
        ),
    ]);
    f.render_widget(
        Paragraph::new(title_line).style(Style::default().bg(Color::Black)),
        rows[0],
    );

    let (label, color) = mode_label(app);
    let mode_line = Line::from(vec![
        Span::styled(" mode  ", Style::default().fg(DIM)),
        Span::styled("›  ", Style::default().fg(DIM)),
        Span::styled(
            label,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
    ]);
    f.render_widget(
        Paragraph::new(mode_line).style(Style::default().bg(Color::Black)),
        rows[1],
    );
}

fn draw_body(f: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(9),  // search (full width)
            Constraint::Min(0),     // probability + time estimates
            Constraint::Length(12), // status / actions
        ])
        .split(area);

    let middle = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(rows[1]);

    draw_pattern_panel(f, app, rows[0]);
    draw_prob_panel(f, app, middle[0]);
    draw_time_panel(f, app, middle[1]);
    draw_status_panel(f, app, rows[2]);
}

// ── Pattern ──────────────────────────────────────────────────────────────────

fn draw_pattern_panel(f: &mut Frame, app: &App, area: Rect) {
    let is_active = matches!(app.mode, Mode::Idle | Mode::Error(_));
    let focused = is_active && matches!(app.focused_panel, FocusedPanel::Pattern);
    let border_col = if focused { ACCENT } else { DIM };
    let title_col = if focused { ACCENT } else { DIM };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_col))
        .title(Span::styled(
            " SEARCH ",
            Style::default().fg(title_col).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let (valid, invalid_indices) = validate_pattern(&app.pattern);

    // Field cursor: a row is "selected" only while the panel is focused. The label
    // gets a ▸ marker + accent so Up/Down selection is visible; Left/Right then
    // change that row's value.
    let sel = |fld: SearchField| focused && app.search_field == fld;
    let field_label = |name: &str, fld: SearchField| {
        let (mark, col) = if sel(fld) {
            ("▸ ", ACCENT)
        } else {
            ("  ", DIM)
        };
        Span::styled(format!("{}{:<8}", mark, name), Style::default().fg(col))
    };
    let arrows = |fld: SearchField| {
        if sel(fld) {
            Span::styled("← →", Style::default().fg(DIM))
        } else {
            Span::raw("")
        }
    };

    // input line
    let mut input_spans = vec![
        field_label("string", SearchField::Pattern),
        Span::styled("›  ", Style::default().fg(DIM)),
    ];
    if app.pattern.is_empty() {
        if sel(SearchField::Pattern) {
            input_spans.push(Span::styled("█", Style::default().fg(ACCENT)));
        } else {
            input_spans.push(Span::styled(
                "type here…",
                Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
            ));
        }
    } else {
        for (i, c) in app.pattern.chars().enumerate() {
            let style = if invalid_indices.contains(&i) {
                Style::default().fg(ERR).add_modifier(Modifier::UNDERLINED)
            } else {
                Style::default().fg(BRIGHT).add_modifier(Modifier::BOLD)
            };
            input_spans.push(Span::styled(c.to_string(), style));
        }
        if sel(SearchField::Pattern) {
            input_spans.push(Span::styled("█", Style::default().fg(ACCENT)));
        }
    }

    let validity_line = if app.pattern.is_empty() {
        Line::from(vec![
            Span::styled("  chars   ", Style::default().fg(DIM)),
            Span::styled("›  ", Style::default().fg(DIM)),
            Span::styled("a–z  2–7  (base32)", Style::default().fg(DIM)),
        ])
    } else if valid {
        Line::from(vec![
            Span::styled("  chars   ", Style::default().fg(DIM)),
            Span::styled("›  ", Style::default().fg(DIM)),
            Span::styled("✓ ", Style::default().fg(OK)),
            Span::styled(
                format!(
                    "{} char{}",
                    app.pattern.len(),
                    if app.pattern.len() == 1 { "" } else { "s" }
                ),
                Style::default().fg(DIM),
            ),
        ])
    } else {
        Line::from(vec![
            Span::styled("  chars   ", Style::default().fg(DIM)),
            Span::styled("›  ", Style::default().fg(DIM)),
            Span::styled("✗ invalid - use a-z 2-7", Style::default().fg(ERR)),
        ])
    };

    let match_line = {
        let options = ["Prefix", "Suffix", "Anywhere"];
        let cur = app.match_type.label();
        let mut spans = vec![
            field_label("match", SearchField::Match),
            Span::styled("›  ", Style::default().fg(DIM)),
        ];
        for opt in options {
            if opt == cur {
                spans.push(Span::styled(
                    format!("[{}]", opt),
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                ));
            } else {
                spans.push(Span::styled(format!(" {} ", opt), Style::default().fg(DIM)));
            }
            spans.push(Span::raw(" "));
        }
        spans.push(arrows(SearchField::Match));
        Line::from(spans)
    };

    let backend_line = {
        let options = [Backend::Cpu, Backend::Gpu];
        let cur = app.backend;
        let gpu_available = app.gpu.is_some();
        let mut spans = vec![
            field_label("backend", SearchField::Backend),
            Span::styled("›  ", Style::default().fg(DIM)),
        ];
        for opt in options {
            let unavailable = matches!(opt, Backend::Gpu) && !gpu_available;
            if opt == cur {
                let col = if unavailable { ERR } else { ACCENT };
                spans.push(Span::styled(
                    format!("[{}]", opt.label()),
                    Style::default().fg(col).add_modifier(Modifier::BOLD),
                ));
            } else {
                let col = if unavailable { ERR } else { DIM };
                spans.push(Span::styled(
                    format!(" {} ", opt.label()),
                    Style::default().fg(col),
                ));
            }
            spans.push(Span::raw(" "));
        }
        let detail = match (cur, &app.gpu) {
            (Backend::Cpu, _) => format!("{} threads", app.threads),
            (Backend::Gpu, Some(ctx)) => format!("{} · {}", ctx.backend_label(), ctx.adapter_name),
            (Backend::Gpu, None) => "unavailable".into(),
        };
        spans.push(Span::styled(detail, Style::default().fg(DIM)));
        spans.push(Span::raw("  "));
        spans.push(arrows(SearchField::Backend));
        Line::from(spans)
    };

    // GPU BATCH_K picker: only meaningful on the GPU backend. Options are bounded
    // by the device so the footprint can never exceed what the GPU can allocate.
    let show_batch = app.backend == Backend::Gpu && app.gpu.is_some();
    let batch_line = if show_batch {
        let mut spans = vec![
            field_label("batch", SearchField::Batch),
            Span::styled("›  ", Style::default().fg(DIM)),
        ];
        for &k in &app.batch_k_options {
            if k == app.batch_k {
                spans.push(Span::styled(
                    format!("[{}]", k),
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                ));
            } else {
                spans.push(Span::styled(format!(" {} ", k), Style::default().fg(DIM)));
            }
            spans.push(Span::raw(" "));
        }
        let footprint = crate::gpu::est_footprint_bytes(app.batch_k);
        spans.push(Span::styled(
            format!("~{:.1} GB GPU  ", footprint as f64 / 1e9),
            Style::default().fg(DIM),
        ));
        spans.push(arrows(SearchField::Batch));
        Some(Line::from(spans))
    } else {
        None
    };

    let preview_line = if !app.pattern.is_empty() && valid {
        let pat = &app.pattern;
        let remaining = ADDRESS_LEN.saturating_sub(pat.len());
        let filler: String = "abcdefghijklmnop234567abcdefghijklmnop234567"
            .chars()
            .take(remaining)
            .collect();
        let preview_addr = match app.match_type {
            crate::crypto::MatchType::Prefix => format!("{}{}.onion", pat, filler),
            crate::crypto::MatchType::Suffix => format!("{}{}.onion", filler, pat),
            crate::crypto::MatchType::Anywhere => {
                let half = remaining / 2;
                let pre: String = filler.chars().take(half).collect();
                let suf: String = filler.chars().skip(half).take(remaining - half).collect();
                format!("{}{}{}.onion", pre, pat, suf)
            }
        };
        let (before, matched, after) = split_preview(&preview_addr, pat, &app.match_type);
        Line::from(vec![
            Span::styled("  example ", Style::default().fg(DIM)),
            Span::styled("›  ", Style::default().fg(DIM)),
            Span::styled(before, Style::default().fg(DIM)),
            Span::styled(
                matched,
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(after, Style::default().fg(DIM)),
        ])
    } else {
        Line::from(Span::styled(
            "  example  ›  enter a pattern above",
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        ))
    };

    let mut lines = vec![
        Line::raw(""),
        Line::from(input_spans),
        validity_line,
        match_line,
        backend_line,
    ];
    if let Some(batch_line) = batch_line {
        lines.push(batch_line);
    }
    lines.push(preview_line);
    f.render_widget(Paragraph::new(Text::from(lines)), inner);
}

fn split_preview(
    addr: &str,
    pattern: &str,
    match_type: &crate::crypto::MatchType,
) -> (String, String, String) {
    let n = pattern.len();
    let total = addr.len();
    let (s, e) = match match_type {
        crate::crypto::MatchType::Prefix => (0, n.min(total)),
        crate::crypto::MatchType::Suffix => {
            let start = total.saturating_sub(n + ".onion".len());
            (start, (start + n).min(total))
        }
        crate::crypto::MatchType::Anywhere => {
            let half = (ADDRESS_LEN - n) / 2;
            (half, (half + n).min(total))
        }
    };
    (
        addr[..s].to_string(),
        addr[s..e].to_string(),
        addr[e..].to_string(),
    )
}

// ── Probability ───────────────────────────────────────────────────────────────

fn draw_prob_panel(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(
            " PROBABILITY ",
            Style::default().fg(DIM).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let (valid, _) = validate_pattern(&app.pattern);
    if app.pattern.is_empty() || !valid {
        f.render_widget(
            Paragraph::new(Text::from(vec![
                Line::raw(""),
                Line::from(Span::styled(
                    "  enter a valid pattern",
                    Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
                )),
            ])),
            inner,
        );
        return;
    }

    let p = match_probability(app.pattern.len(), &app.match_type, &app.backend);
    let exp = expected_attempts(p);

    let lines = vec![
        Line::raw(""),
        Line::from(vec![
            Span::styled("  1 in ", Style::default().fg(DIM)),
            Span::styled(
                format_count(exp),
                Style::default().fg(BRIGHT).add_modifier(Modifier::BOLD),
            ),
            Span::styled("  (mean)", Style::default().fg(DIM)),
        ]),
        Line::from(vec![
            Span::styled("  p1  ", Style::default().fg(DIM)),
            Span::styled(
                format!("{:<8}", format_count(quantile_attempts(p, 0.01))),
                Style::default().fg(BRIGHT),
            ),
            Span::styled("p10  ", Style::default().fg(DIM)),
            Span::styled(
                format!("{:<8}", format_count(quantile_attempts(p, 0.10))),
                Style::default().fg(BRIGHT),
            ),
            Span::styled("p25  ", Style::default().fg(DIM)),
            Span::styled(
                format_count(quantile_attempts(p, 0.25)),
                Style::default().fg(BRIGHT),
            ),
        ]),
        Line::from(vec![
            Span::styled("  p50 ", Style::default().fg(DIM)),
            Span::styled(
                format!("{:<8}", format_count(quantile_attempts(p, 0.50))),
                Style::default().fg(BRIGHT),
            ),
            Span::styled("p75  ", Style::default().fg(DIM)),
            Span::styled(
                format!("{:<8}", format_count(quantile_attempts(p, 0.75))),
                Style::default().fg(BRIGHT),
            ),
            Span::styled("p95  ", Style::default().fg(DIM)),
            Span::styled(
                format_count(quantile_attempts(p, 0.95)),
                Style::default().fg(BRIGHT),
            ),
        ]),
        Line::from(vec![
            Span::styled("  p99 ", Style::default().fg(DIM)),
            Span::styled(
                format_count(quantile_attempts(p, 0.99)),
                Style::default().fg(BRIGHT),
            ),
        ]),
        Line::raw(""),
        Line::from(vec![
            Span::styled("  bits  ", Style::default().fg(DIM)),
            Span::styled(
                format!("{:.1}", (1.0_f64 / p).log2()),
                Style::default().fg(BRIGHT),
            ),
            Span::styled("   p  ", Style::default().fg(DIM)),
            Span::styled(format!("{:.3e}", p), Style::default().fg(BRIGHT)),
        ]),
    ];
    f.render_widget(Paragraph::new(Text::from(lines)), inner);
}

// ── Time estimates ────────────────────────────────────────────────────────────

fn draw_time_panel(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM))
        .title(Span::styled(
            " TIME ESTIMATES ",
            Style::default().fg(DIM).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let (valid, _) = validate_pattern(&app.pattern);
    if app.pattern.is_empty() || !valid {
        f.render_widget(
            Paragraph::new(Text::from(vec![
                Line::raw(""),
                Line::from(Span::styled(
                    "  enter a valid pattern",
                    Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
                )),
            ])),
            inner,
        );
        return;
    }

    // One block per match type, each listing CPU and GPU p50/p95. Every block
    // uses its own probability (anywhere is likelier than prefix/suffix) and the
    // per-(backend, match type) rate cached this session, so all measured combos
    // are visible at once, not just the current mode.
    let mut lines: Vec<Line> = vec![Line::from(Span::styled(
        format!(
            "  {} · {} char{}",
            app.pattern,
            app.pattern.len(),
            if app.pattern.len() == 1 { "" } else { "s" }
        ),
        Style::default().fg(DIM),
    ))];

    let gpu_available = app.gpu.is_some();
    for mt in [MatchType::Prefix, MatchType::Suffix, MatchType::Anywhere] {
        let is_current_mt = mt == app.match_type;

        let name_col = format!(
            "{}{}",
            mt.label().to_uppercase(),
            if is_current_mt { " ▸" } else { "" }
        );
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<15}", name_col),
                Style::default()
                    .fg(if is_current_mt { ACCENT } else { BRIGHT })
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("{:<10}", "p50"), Style::default().fg(DIM)),
            Span::styled("p95", Style::default().fg(DIM)),
        ]));

        for backend in [Backend::Cpu, Backend::Gpu] {
            let label = backend.label();
            let active = is_current_mt && backend == app.backend;
            let col = if active { OK } else { DIM };
            match app.benchmarks.get(&(backend, mt.clone())).copied() {
                Some(rate) => {
                    let p = match_probability(app.pattern.len(), &mt, &backend);
                    let p50 = format_duration(quantile_attempts(p, 0.50) / rate);
                    let p95 = format_duration(quantile_attempts(p, 0.95) / rate);
                    lines.push(Line::from(vec![
                        Span::styled(
                            format!("    {:<4}{:<9}", label, format_rate(rate)),
                            Style::default().fg(col),
                        ),
                        Span::styled(format!("{:<10}", p50), Style::default().fg(col)),
                        Span::styled(p95, Style::default().fg(col)),
                    ]));
                }
                None => {
                    let note = if backend == Backend::Gpu && !gpu_available {
                        "n/a"
                    } else {
                        "[b]"
                    };
                    lines.push(Line::from(Span::styled(
                        format!("    {:<4}{}", label, note),
                        Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
                    )));
                }
            }
        }
    }

    f.render_widget(Paragraph::new(Text::from(lines)), inner);
}

// ── Status / Actions ──────────────────────────────────────────────────────────

fn draw_status_panel(f: &mut Frame, app: &App, area: Rect) {
    match &app.mode {
        Mode::Idle => draw_idle_status(f, app, area),
        Mode::Error(msg) => draw_error(f, area, msg),
        Mode::Benchmarking {
            started,
            worker,
            backend,
            ..
        } => {
            let steady = *worker.bench_rate.lock().unwrap();
            draw_benchmark_status(f, area, *started, worker.attempts(), steady, *backend)
        }
        Mode::Generating {
            started,
            worker,
            rate_tracker,
            backend,
        } => {
            let attempts = worker.attempts();
            let tracker = rate_tracker.lock().unwrap();
            let rate = tracker.last_rate;
            let history: Vec<u64> = tracker.history.iter().copied().collect();
            drop(tracker);
            draw_generate_status(f, app, area, *started, attempts, rate, &history, *backend)
        }
        Mode::Found {
            result,
            attempts,
            elapsed,
        } => draw_found(f, area, result, *attempts, *elapsed),
    }
}

fn panel_focused(app: &App) -> bool {
    matches!(app.focused_panel, FocusedPanel::Actions)
        && matches!(app.mode, Mode::Idle | Mode::Error(_))
}

fn draw_idle_status(f: &mut Frame, app: &App, area: Rect) {
    let focused = panel_focused(app);
    let border_col = if focused { ACCENT } else { DIM };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_col))
        .title(Span::styled(
            " ACTIONS ",
            Style::default()
                .fg(if focused { ACCENT } else { DIM })
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let key_style = Style::default()
        .fg(if focused { ACCENT } else { DIM })
        .add_modifier(Modifier::BOLD);

    let mut lines = vec![
        Line::raw(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("[g]", key_style),
            Span::raw("  "),
            Span::styled("generate   ", Style::default().fg(BRIGHT)),
            Span::styled(
                "search for a matching .onion keypair",
                Style::default().fg(DIM),
            ),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("[b]", key_style),
            Span::raw("  "),
            Span::styled("benchmark  ", Style::default().fg(BRIGHT)),
            Span::styled(
                format!("measure {} throughput", app.backend.label()),
                Style::default().fg(DIM),
            ),
        ]),
    ];

    if !app.status_msg.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            format!("  {}", app.status_msg),
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        )));
    }

    if !focused {
        lines.push(Line::from(Span::styled(
            "  [Tab] to focus",
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        )));
    }

    f.render_widget(Paragraph::new(Text::from(lines)), inner);
}

fn draw_benchmark_status(
    f: &mut Frame,
    area: Rect,
    started: std::time::Instant,
    attempts: u64,
    steady_rate: Option<f64>,
    backend: Backend,
) {
    let title = match backend {
        Backend::Cpu => " BENCHMARKING · CPU ",
        Backend::Gpu => " BENCHMARKING · GPU ",
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(Span::styled(
            title,
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let elapsed = started.elapsed().as_secs_f64();
    let total = 5.0_f64;
    let progress = (elapsed / total).min(1.0);
    // Prefer the steady-state rate the bench publishes (excludes pipeline build +
    // warm-up); fall back to the cold average only before the first productive
    // dispatch reports. The cold average alone under-reads high-K GPU benches.
    let rate = steady_rate.unwrap_or(if elapsed > 0.0 {
        attempts as f64 / elapsed
    } else {
        0.0
    });

    let gauge_area = Rect {
        x: inner.x + 2,
        y: inner.y + 2,
        width: inner.width.saturating_sub(4),
        height: 1,
    };

    let unit = match backend {
        Backend::Cpu => "keys/s",
        Backend::Gpu => "keys/s",
    };
    let label = format!(
        " {:.0}s / {:.0}s  |  {} {} ",
        elapsed.min(total),
        total,
        format_count(rate),
        unit
    );

    f.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(ACCENT).bg(Color::Black))
            .ratio(progress)
            .label(label),
        gauge_area,
    );

    let work_label = match backend {
        Backend::Cpu => "keypairs generated",
        Backend::Gpu => "keys generated",
    };
    f.render_widget(
        Paragraph::new(Text::from(vec![
            Line::raw(""),
            Line::raw(""),
            Line::raw(""),
            Line::from(Span::styled(
                format!("  {} {}", format_count(attempts as f64), work_label),
                Style::default().fg(DIM),
            )),
        ])),
        inner,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_generate_status(
    f: &mut Frame,
    app: &App,
    area: Rect,
    started: std::time::Instant,
    attempts: u64,
    rate: f64,
    history: &[u64],
    backend: Backend,
) {
    let title = match backend {
        Backend::Cpu => " GENERATING · CPU ",
        Backend::Gpu => " GENERATING · GPU ",
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(Span::styled(
            title,
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let backend_detail = match (backend, &app.gpu) {
        (Backend::Cpu, _) => format!("CPU · {} threads", app.threads),
        (Backend::Gpu, Some(ctx)) => {
            format!("GPU · {} · {}", ctx.backend_label(), ctx.adapter_name)
        }
        (Backend::Gpu, None) => "GPU".into(),
    };

    let elapsed = started.elapsed();
    let p = match_probability(app.pattern.len(), &app.match_type, &backend);
    let progress = cdf(attempts, p);

    let (eta_p50, eta_p95) = if rate > 0.0 {
        let rem_p50 = (quantile_attempts(p, 0.50) - attempts as f64).max(0.0) / rate;
        let rem_p95 = (quantile_attempts(p, 0.95) - attempts as f64).max(0.0) / rate;
        (format_duration(rem_p50), format_duration(rem_p95))
    } else {
        ("…".into(), "…".into())
    };

    let peak = history.iter().copied().max().unwrap_or(0) as f64;

    // Chart + rail on top, then the gauge and the stop hint. A 1-col margin plus
    // blank rows between sections give the panel breathing room, and the gauge row
    // is split so the percentage has its own column instead of camping on the bar.
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .horizontal_margin(1)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Min(20),
            Constraint::Length(2),
            Constraint::Length(24),
        ])
        .split(rows[0]);
    let chart_area = top[0];
    let rail_area = top[2];
    let gauge_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(16),
            Constraint::Length(1),
            Constraint::Min(10),
        ])
        .split(rows[2]);
    let pct_area = gauge_cols[0];
    let bar_area = gauge_cols[2];
    let stop_area = rows[4];

    // The line is scaled to the plot rect, so it fills the full width regardless
    // of how many samples exist (this is what fixes the sparkline under-fill).
    let data: Vec<(f64, f64)> = history
        .iter()
        .enumerate()
        .map(|(i, &r)| (i as f64, r as f64))
        .collect();
    let x_max = history.len().saturating_sub(1).max(1) as f64;
    // Zoom Y to the data range (not 0..peak) so a stable rate centers in the plot
    // and small fluctuations stay visible instead of a flat line pinned to the top.
    let lo = history.iter().copied().min().unwrap_or(0) as f64;
    let (y_min, y_max) = if peak > 0.0 {
        ((lo * 0.9).max(0.0), peak * 1.1)
    } else {
        (0.0, 1.0)
    };
    let datasets = vec![
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(ACCENT))
            .data(&data),
    ];
    let chart = Chart::new(datasets)
        .x_axis(
            Axis::default()
                .style(Style::default().fg(DIM))
                .bounds([0.0, x_max])
                .labels(vec![
                    Span::styled(format!("-{}s", history.len()), Style::default().fg(DIM)),
                    Span::styled("now", Style::default().fg(DIM)),
                ]),
        )
        .y_axis(
            Axis::default()
                .style(Style::default().fg(DIM))
                .bounds([y_min, y_max])
                .labels(vec![
                    Span::styled(
                        format!("{}/s", format_count(y_min)),
                        Style::default().fg(DIM),
                    ),
                    Span::styled(
                        format!("{}/s", format_count(y_max)),
                        Style::default().fg(DIM),
                    ),
                ]),
        );
    f.render_widget(chart, chart_area);

    let label = |s: &str| Span::styled(format!("  {:<8}", s), Style::default().fg(DIM));
    let rail = Paragraph::new(Text::from(vec![
        Line::from(Span::styled(
            backend_detail,
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(vec![
            label("tries"),
            Span::styled(
                format_count(attempts as f64),
                Style::default().fg(BRIGHT).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            label("rate"),
            Span::styled(format_rate(rate), Style::default().fg(BRIGHT)),
        ]),
        Line::from(vec![
            label("elapsed"),
            Span::styled(
                format_duration(elapsed.as_secs_f64()),
                Style::default().fg(BRIGHT),
            ),
        ]),
        Line::from(vec![
            label("ETA"),
            Span::styled(eta_p50, Style::default().fg(OK)),
            Span::styled(" · ", Style::default().fg(DIM)),
            Span::styled(eta_p95, Style::default().fg(WARN)),
        ]),
    ]));
    f.render_widget(rail, rail_area);

    // Percentage in its own column; the bar is a separate rect so nothing overlaps.
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("covered ", Style::default().fg(DIM)),
            Span::styled(
                format!("{:.2}%", progress * 100.0),
                Style::default().fg(BRIGHT).add_modifier(Modifier::BOLD),
            ),
        ])),
        pct_area,
    );
    f.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(ACCENT).bg(Color::Black))
            .ratio(progress)
            .label(""),
        bar_area,
    );
    // p50 / p95 markers at fixed points of the covered axis: once the fill passes
    // the green tick you are past the median run, past the yellow one you are in
    // the unlucky 5%.
    for (frac, col) in [(0.50_f64, OK), (0.95_f64, WARN)] {
        let tx = bar_area.x + (bar_area.width as f64 * frac) as u16;
        if tx < bar_area.x + bar_area.width {
            f.render_widget(
                Paragraph::new(Span::styled("┊", Style::default().fg(col))),
                Rect {
                    x: tx,
                    y: bar_area.y,
                    width: 1,
                    height: 1,
                },
            );
        }
    }

    f.render_widget(
        Paragraph::new(Span::styled("[s] stop", Style::default().fg(DIM))),
        stop_area,
    );
}

fn draw_found(
    f: &mut Frame,
    area: Rect,
    result: &crate::app::FoundResult,
    attempts: u64,
    elapsed: std::time::Duration,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(OK))
        .title(Span::styled(
            " FOUND ",
            Style::default().fg(OK).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    f.render_widget(
        Paragraph::new(Text::from(vec![
            Line::raw(""),
            Line::from(vec![
                Span::styled("  ✓ ", Style::default().fg(OK).add_modifier(Modifier::BOLD)),
                Span::styled(
                    format!("{}.onion", result.address),
                    Style::default().fg(BRIGHT).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::raw(""),
            Line::from(vec![
                Span::styled("  tries  ", Style::default().fg(DIM)),
                Span::styled(format_count(attempts as f64), Style::default().fg(BRIGHT)),
                Span::styled("    time  ", Style::default().fg(DIM)),
                Span::styled(
                    format_duration(elapsed.as_secs_f64()),
                    Style::default().fg(BRIGHT),
                ),
            ]),
            Line::from(vec![
                Span::styled("  saved  ", Style::default().fg(DIM)),
                Span::styled("→  ", Style::default().fg(DIM)),
                Span::styled(
                    format!("./{}/", result.key_path.display()),
                    Style::default().fg(ACCENT),
                ),
            ]),
            Line::from(Span::styled(
                "  ⚠  secret key saved - protect this directory",
                Style::default().fg(WARN),
            )),
            Line::raw(""),
            Line::from(vec![
                Span::styled("  [n] ", Style::default().fg(ACCENT)),
                Span::styled("new search    ", Style::default().fg(BRIGHT)),
                Span::styled("[q] ", Style::default().fg(ACCENT)),
                Span::styled("quit", Style::default().fg(BRIGHT)),
            ]),
        ])),
        inner,
    );
}

fn draw_error(f: &mut Frame, area: Rect, msg: &str) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ERR))
        .title(Span::styled(
            " ERROR ",
            Style::default().fg(ERR).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    f.render_widget(
        Paragraph::new(format!("\n  {}", msg))
            .style(Style::default().fg(ERR))
            .wrap(Wrap { trim: true }),
        inner,
    );
}

// ── Footer ────────────────────────────────────────────────────────────────────

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let hint = match &app.mode {
        Mode::Benchmarking { .. } => "  benchmarking… please wait  │  [q] quit".to_string(),
        Mode::Generating { .. } => "  [s] stop search  │  [q] quit".to_string(),
        Mode::Found { .. } => "  [n] new search  │  [q] quit".to_string(),
        Mode::Error(_) => "  [Esc] dismiss  │  [q] quit".to_string(),
        Mode::Idle => match &app.focused_panel {
            FocusedPanel::Pattern => {
                "  [↑ ↓] select field  │  [← →] change  │  type a–z 2–7  │  [Tab] → actions  │  [q] quit"
                    .to_string()
            }
            FocusedPanel::Actions => {
                "  [g] generate  │  [b] benchmark  │  [Tab] → search  │  [q] quit".to_string()
            }
        },
    };

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(hint, Style::default().fg(DIM))))
            .style(Style::default().bg(Color::Black)),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    fn render(app: &App) -> String {
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| draw(f, app)).unwrap();
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect()
    }

    /// The SEARCH panel must render the field rows without panicking (guards the
    /// fixed panel height vs the number of rows, which grows with the batch row).
    #[test]
    fn search_panel_renders_fields() {
        let app = App::new();
        let text = render(&app);
        assert!(text.contains("string"), "missing string field");
        assert!(text.contains("match"), "missing match field");
        assert!(text.contains("backend"), "missing backend field");
        // Field-cursor marker is present on the focused panel's selected row.
        assert!(text.contains('▸'), "missing field cursor marker");
    }

    /// The batch row appears only on the GPU backend, and renders with the cursor
    /// there too (no overflow panic with all rows visible).
    #[test]
    fn batch_row_visible_only_on_gpu() {
        let mut app = App::new();
        if app.gpu.is_none() {
            return; // no GPU in this environment; nothing to assert
        }
        // CPU backend: no batch row.
        assert!(!render(&app).contains("batch"), "batch row shown on CPU");
        // Switch to GPU and select the batch field.
        app.backend = Backend::Gpu;
        app.search_field = SearchField::Batch;
        let text = render(&app);
        assert!(text.contains("batch"), "batch row missing on GPU");
    }
}
