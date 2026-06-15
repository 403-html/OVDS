use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Gauge, Paragraph, Sparkline, Wrap},
};

use crate::app::{App, FocusedPanel, Mode};
use crate::crypto::{ADDRESS_LEN, Backend, validate_pattern};
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
            format!("{:>width$}", "v0.2.0 ", width = area.width as usize - 32),
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
            Constraint::Length(10), // status / actions
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

    // input line
    let mut input_spans = vec![
        Span::styled("  string  ", Style::default().fg(DIM)),
        Span::styled("›  ", Style::default().fg(DIM)),
    ];
    if app.pattern.is_empty() {
        if focused {
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
        if focused {
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
            Span::styled("  match   ", Style::default().fg(DIM)),
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
        if focused {
            spans.push(Span::styled("← →", Style::default().fg(DIM)));
        }
        Line::from(spans)
    };

    let backend_line = {
        let options = [Backend::Cpu, Backend::Gpu];
        let cur = app.backend;
        let gpu_available = app.gpu.is_some();
        let mut spans = vec![
            Span::styled("  backend ", Style::default().fg(DIM)),
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
        if focused {
            spans.push(Span::styled("   ↑ ↓", Style::default().fg(DIM)));
        }
        Line::from(spans)
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

    let text = Text::from(vec![
        Line::raw(""),
        Line::from(input_spans),
        validity_line,
        match_line,
        backend_line,
        preview_line,
    ]);
    f.render_widget(Paragraph::new(text), inner);
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

    let p = match_probability(app.pattern.len(), &app.match_type);
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

    let p = match_probability(app.pattern.len(), &app.match_type);

    // Transposed table: rows = quantiles, columns = rates
    let quantiles: &[(f64, &str)] = &[
        (0.01, "p1"),
        (0.10, "p10"),
        (0.25, "p25"),
        (0.50, "p50"),
        (0.75, "p75"),
        (0.95, "p95"),
        (0.99, "p99"),
    ];

    let ref_rates: &[(&str, f64)] = &[("100K/s", 100_000.0), ("1M/s", 1_000_000.0)];

    // Header: blank label col + ref columns + CPU + GPU
    let sep = "  ─────────────────────────────────────────────────────────";

    let mut header_spans = vec![Span::styled(
        format!("  {:<5}", ""),
        Style::default().fg(DIM),
    )];
    for (label, _) in ref_rates {
        header_spans.push(Span::styled(
            format!("{:<11}", label),
            Style::default().fg(DIM),
        ));
    }
    // CPU column
    let cpu_active = matches!(app.backend, Backend::Cpu);
    if let Some(rate) = app.cpu_benchmark_rate {
        header_spans.push(Span::styled(
            format!("CPU {:<7}", format_rate(rate)),
            Style::default()
                .fg(if cpu_active { OK } else { DIM })
                .add_modifier(Modifier::BOLD),
        ));
    } else {
        header_spans.push(Span::styled(
            "CPU [b]    ",
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        ));
    }
    // GPU column
    let gpu_active = matches!(app.backend, Backend::Gpu);
    if let Some(rate) = app.gpu_benchmark_rate {
        header_spans.push(Span::styled(
            format!("GPU {}", format_rate(rate)),
            Style::default()
                .fg(if gpu_active { OK } else { DIM })
                .add_modifier(Modifier::BOLD),
        ));
    } else if app.gpu.is_some() {
        header_spans.push(Span::styled(
            "GPU [c,b]",
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        ));
    } else {
        header_spans.push(Span::styled(
            "GPU n/a",
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        ));
    }

    let mut lines = vec![
        Line::raw(""),
        Line::from(header_spans),
        Line::from(Span::styled(sep, Style::default().fg(DIM))),
    ];

    for (q, label) in quantiles {
        let mut spans = vec![Span::styled(
            format!("  {:<5}", label),
            Style::default().fg(DIM),
        )];
        for (_, rate) in ref_rates {
            let secs = quantile_attempts(p, *q) / rate;
            spans.push(Span::styled(
                format!("{:<11}", format_duration(secs)),
                Style::default().fg(DIM),
            ));
        }
        // CPU column
        if let Some(rate) = app.cpu_benchmark_rate {
            let secs = quantile_attempts(p, *q) / rate;
            spans.push(Span::styled(
                format!("{:<11}", format_duration(secs)),
                Style::default().fg(if cpu_active { OK } else { DIM }),
            ));
        } else {
            spans.push(Span::styled(
                format!("{:<11}", "-"),
                Style::default().fg(DIM),
            ));
        }
        // GPU column
        if let Some(rate) = app.gpu_benchmark_rate {
            let secs = quantile_attempts(p, *q) / rate;
            spans.push(Span::styled(
                format_duration(secs),
                Style::default().fg(if gpu_active { OK } else { DIM }),
            ));
        } else {
            spans.push(Span::styled("-", Style::default().fg(DIM)));
        }
        lines.push(Line::from(spans));
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
        } => draw_benchmark_status(f, area, *started, worker.attempts(), *backend),
        Mode::Generating {
            started,
            worker,
            rate_tracker,
        } => {
            let attempts = worker.attempts();
            let tracker = rate_tracker.lock().unwrap();
            let rate = tracker.last_rate;
            let history: Vec<u64> = tracker.history.iter().copied().collect();
            drop(tracker);
            draw_generate_status(f, app, area, *started, attempts, rate, &history)
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
    let rate = if elapsed > 0.0 {
        attempts as f64 / elapsed
    } else {
        0.0
    };

    let gauge_area = Rect {
        x: inner.x + 2,
        y: inner.y + 2,
        width: inner.width.saturating_sub(4),
        height: 1,
    };

    let unit = match backend {
        Backend::Cpu => "keys/s",
        Backend::Gpu => "SHA-256/s",
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
        Backend::Gpu => "SHA-256 ops dispatched",
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

fn draw_generate_status(
    f: &mut Frame,
    app: &App,
    area: Rect,
    started: std::time::Instant,
    attempts: u64,
    rate: f64,
    history: &[u64],
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(Span::styled(
            " GENERATING ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let elapsed = started.elapsed();
    let p = match_probability(app.pattern.len(), &app.match_type);
    let progress = cdf(attempts, p);

    let (eta_p50, eta_p95) = if rate > 0.0 {
        let rem_p50 = (quantile_attempts(p, 0.50) - attempts as f64).max(0.0) / rate;
        let rem_p95 = (quantile_attempts(p, 0.95) - attempts as f64).max(0.0) / rate;
        (format_duration(rem_p50), format_duration(rem_p95))
    } else {
        ("…".into(), "…".into())
    };

    // layout (8 inner lines):
    //  0  empty
    //  1  tries / rate
    //  2  elapsed / ETA p50 / p95
    //  3  "throughput history" label
    //  4  sparkline
    //  5  progress gauge
    //  6  [s] stop
    //  7  empty

    let spark_area = Rect {
        x: inner.x + 2,
        y: inner.y + 4,
        width: inner.width.saturating_sub(4),
        height: 1,
    };
    let gauge_area = Rect {
        x: inner.x + 2,
        y: inner.y + 5,
        width: inner.width.saturating_sub(4),
        height: 1,
    };

    f.render_widget(
        Paragraph::new(Text::from(vec![
            Line::raw(""),
            Line::from(vec![
                Span::styled("  tries    ", Style::default().fg(DIM)),
                Span::styled(
                    format_count(attempts as f64),
                    Style::default().fg(BRIGHT).add_modifier(Modifier::BOLD),
                ),
                Span::styled("    rate  ", Style::default().fg(DIM)),
                Span::styled(format_rate(rate), Style::default().fg(BRIGHT)),
            ]),
            Line::from(vec![
                Span::styled("  elapsed  ", Style::default().fg(DIM)),
                Span::styled(
                    format_duration(elapsed.as_secs_f64()),
                    Style::default().fg(BRIGHT),
                ),
                Span::styled("    ETA p50  ", Style::default().fg(DIM)),
                Span::styled(eta_p50, Style::default().fg(OK)),
                Span::styled("    p95  ", Style::default().fg(DIM)),
                Span::styled(eta_p95, Style::default().fg(WARN)),
            ]),
            Line::from(Span::styled(
                "  throughput history",
                Style::default().fg(DIM),
            )),
            Line::raw(""), // sparkline
            Line::raw(""), // gauge
            Line::from(Span::styled("  [s] stop", Style::default().fg(DIM))),
            Line::raw(""),
        ])),
        inner,
    );

    if !history.is_empty() {
        f.render_widget(
            Sparkline::default()
                .data(history)
                .style(Style::default().fg(ACCENT)),
            spark_area,
        );
    }

    f.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(ACCENT).bg(Color::Black))
            .ratio(progress)
            .label(format!(
                " {:.1}% probability mass covered ",
                progress * 100.0
            )),
        gauge_area,
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
                "  type a–z 2–7  │  [← →] match  │  [↑ ↓] backend  │  [Tab] → actions  │  [q] quit"
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
