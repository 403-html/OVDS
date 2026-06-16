mod app;
mod crypto;
#[cfg(test)]
mod fe16_ref;
mod gpu;
mod stats;
mod ui;

use app::{App, FocusedPanel, Mode};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io;
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    // Redirect panic output to a log file so wgpu/driver panics in worker
    // threads don't clobber the ratatui alternate screen.
    std::panic::set_hook(Box::new(|info| {
        use std::io::Write as _;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("ovds-panic.log")
        {
            let _ = writeln!(f, "{info}");
        }
    }));

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let result = run(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(e) = result {
        eprintln!("Error: {}", e);
    }
    Ok(())
}

fn run<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
) -> anyhow::Result<()> {
    loop {
        terminal.draw(|f| ui::draw(f, app))?;
        app.tick();

        if event::poll(Duration::from_millis(80))?
            && let Event::Key(key) = event::read()?
        {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            handle_key(app, key.code);
        }

        if app.quit {
            return Ok(());
        }
    }
}

fn handle_key(app: &mut App, code: KeyCode) {
    // q quits from anywhere
    if matches!(code, KeyCode::Char('q') | KeyCode::Char('Q')) {
        app.stop();
        app.quit = true;
        return;
    }

    // Tab cycles panels only when editable
    if code == KeyCode::Tab {
        if matches!(app.mode, Mode::Idle | Mode::Error(_)) {
            app.cycle_panel();
        }
        return;
    }

    match &app.mode {
        Mode::Error(_) => {
            if code == KeyCode::Esc {
                app.mode = Mode::Idle;
            }
        }
        Mode::Found { .. } => {
            if matches!(code, KeyCode::Char('n') | KeyCode::Char('N')) {
                app.mode = Mode::Idle;
                app.status_msg.clear();
            }
        }
        Mode::Benchmarking { .. } => {}
        Mode::Generating { .. } => {
            if matches!(code, KeyCode::Char('s') | KeyCode::Char('S')) {
                app.stop();
            }
        }
        Mode::Idle => match &app.focused_panel {
            FocusedPanel::Pattern => match code {
                KeyCode::Esc => app.quit = true,
                KeyCode::Backspace | KeyCode::Delete => app.backspace(),
                KeyCode::Left => app.cycle_match_type(false),
                KeyCode::Right => app.cycle_match_type(true),
                KeyCode::Up | KeyCode::Down => app.toggle_backend(),
                KeyCode::Char(c) => app.type_char(c.to_ascii_lowercase()),
                _ => {}
            },
            FocusedPanel::Actions => match code {
                KeyCode::Esc => app.quit = true,
                KeyCode::Char('b') | KeyCode::Char('B') => app.start_benchmark(),
                KeyCode::Char('g') | KeyCode::Char('G') => app.start_generate(),
                _ => {}
            },
        },
    }
}
