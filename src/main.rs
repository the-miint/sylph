use clap::{CommandFactory, Parser};
use std::io;
use sylph::cmdline::*;
use sylph::contain;
use sylph::inspect;
use sylph::sketch;
use sylph::twostage_db;
use termcolor::{BufferWriter, Color, ColorChoice, ColorSpec, WriteColor};
//use std::panic::set_hook;

/// `sketch`/`profile` are the primary day-to-day commands; `query`/`inspect`/
/// `convert-db-two-screen` are secondary/advanced. clap 3.2's subcommand
/// listing only supports one flat heading (no per-group headings, derive or
/// builder), so this reworks it into two by taking clap's own fully-rendered
/// help text -- inheriting its exact wrapping/column alignment -- and
/// re-splitting just the "SUBCOMMANDS:" section by name, rather than
/// hand-duplicating the preamble/usage/options or the per-command about text.
const PRIMARY_COMMANDS: [&str; 2] = ["sketch", "profile"];

/// Build the grouped help as plain text first. `Command::write_help` is
/// intentionally color-unaware in clap 3; clap only applies its styles in the
/// terminal-printing path, which we cannot use because the subcommand section
/// needs to be rearranged before it is printed.
fn grouped_help_text() -> (String, Vec<String>) {
    let mut cmd = Cli::command();
    let mut buf: Vec<u8> = Vec::new();
    cmd.write_help(&mut buf)
        .expect("writing help to an in-memory buffer cannot fail");
    let full = String::from_utf8(buf).expect("clap help output must be valid UTF-8");
    let marker = "SUBCOMMANDS:\n";
    let Some(marker_pos) = full.find(marker) else {
        // clap's heading text changed out from under us -- fall back to the
        // normal (ungrouped) rendering rather than showing something broken.
        return (full, Vec::new());
    };
    let preamble = &full[..marker_pos];
    let listing = &full[marker_pos + marker.len()..];

    let names_in_order: Vec<String> = cmd
        .get_subcommands()
        .map(|s| s.get_name().to_string())
        .collect();

    // Group `listing`'s lines back into per-subcommand entries: a line at the
    // base (4-space) indent whose first token is a known subcommand name
    // starts a new entry; anything else (deeper-indented wrapped
    // continuation lines) belongs to the entry above it.
    let mut entries: Vec<(String, Vec<&str>)> = Vec::new();
    for line in listing.lines() {
        let trimmed = line.trim_start();
        let indent = line.len() - trimmed.len();
        let first_token = trimmed.split_whitespace().next();
        let starts_new = indent == 4
            && names_in_order
                .iter()
                .any(|n| first_token == Some(n.as_str()));
        if starts_new {
            let name = names_in_order
                .iter()
                .find(|n| first_token == Some(n.as_str()))
                .unwrap()
                .clone();
            entries.push((name, vec![line]));
        } else if let Some(last) = entries.last_mut() {
            last.1.push(line);
        }
    }

    let render_group = |names: &[String]| -> String {
        entries
            .iter()
            .filter(|(name, _)| names.contains(name))
            .map(|(_, lines)| lines.join("\n"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let primary_names: Vec<String> = PRIMARY_COMMANDS.iter().map(|s| s.to_string()).collect();
    let advanced_names: Vec<String> = names_in_order
        .iter()
        .filter(|n| !primary_names.contains(n))
        .cloned()
        .collect();

    (
        format!(
            "{preamble}COMMANDS:\n{}\n\nADVANCED COMMANDS:\n{}\n",
            render_group(&primary_names),
            render_group(&advanced_names),
        ),
        names_in_order,
    )
}

fn write_colored<W: WriteColor>(writer: &mut W, text: &str, color: Color) -> io::Result<()> {
    writer.set_color(ColorSpec::new().set_fg(Some(color)))?;
    writer.write_all(text.as_bytes())?;
    writer.reset()
}

/// Reapply clap 3's own help palette after grouping the plain rendered text:
/// yellow headings and green program/option/primary-command names. Advanced
/// command names deliberately retain the terminal's default foreground.
fn write_grouped_help<W: WriteColor>(writer: &mut W) -> io::Result<()> {
    let (help, subcommand_names) = grouped_help_text();

    for (line_number, line) in help.lines().enumerate() {
        let trimmed = line.trim_start();
        let indent = line.len() - trimmed.len();

        if line_number == 0 {
            // clap's default template colors only the display name, not the
            // version which follows it on the same line.
            if let Some((name, rest)) = line.split_once(' ') {
                write_colored(writer, name, Color::Green)?;
                write!(writer, " {rest}")?;
            } else {
                write_colored(writer, line, Color::Green)?;
            }
        } else if matches!(
            trimmed,
            "USAGE:" | "OPTIONS:" | "COMMANDS:" | "ADVANCED COMMANDS:"
        ) {
            // This covers clap's USAGE/OPTIONS headings and both headings
            // introduced by the grouping refactor.
            writer.write_all(&line.as_bytes()[..indent])?;
            write_colored(writer, trimmed, Color::Yellow)?;
        } else if indent == 4 && trimmed.starts_with('-') {
            // The option spelling ends where clap's alignment padding begins.
            let option_end = trimmed.find("  ").unwrap_or(trimmed.len());
            let option = &trimmed[..option_end];
            writer.write_all(&line.as_bytes()[..indent])?;
            if let Some((short, long)) = option.split_once(", ") {
                // clap styles each flag but leaves the separator unstyled.
                write_colored(writer, short, Color::Green)?;
                writer.write_all(b", ")?;
                write_colored(writer, long, Color::Green)?;
            } else {
                write_colored(writer, option, Color::Green)?;
            }
            writer.write_all(trimmed[option_end..].as_bytes())?;
        } else if indent == 4 {
            let token_end = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
            let name = &trimmed[..token_end];
            if subcommand_names.iter().any(|candidate| candidate == name) {
                writer.write_all(&line.as_bytes()[..indent])?;
                let is_advanced = !PRIMARY_COMMANDS.contains(&name);
                if is_advanced {
                    writer.write_all(name.as_bytes())?;
                } else {
                    write_colored(writer, name, Color::Green)?;
                }
                writer.write_all(trimmed[token_end..].as_bytes())?;
            } else {
                writer.write_all(line.as_bytes())?;
            }
        } else {
            writer.write_all(line.as_bytes())?;
        }
        writer.write_all(b"\n")?;
    }

    Ok(())
}

enum HelpStream {
    Stdout,
    Stderr,
}

fn print_grouped_help(stream: HelpStream) -> io::Result<()> {
    // Auto matches clap 3: ANSI colors on an interactive terminal, plain text
    // when help is redirected or piped.
    let buffer_writer = match stream {
        HelpStream::Stdout => BufferWriter::stdout(ColorChoice::Auto),
        HelpStream::Stderr => BufferWriter::stderr(ColorChoice::Auto),
    };
    let mut buffer = buffer_writer.buffer();
    write_grouped_help(&mut buffer)?;
    buffer_writer.print(&buffer)
}

fn main() {
    //    set_hook(Box::new(|info| {
    //        if let Some(s) = info.payload().downcast_ref::<String>() {
    //            log::error!("{}", s);
    //        }
    //    }));
    let raw_first_arg = std::env::args().nth(1);
    match raw_first_arg.as_deref() {
        // Bare `sylph` (no args): matches clap's own arg_required_else_help
        // behavior exactly -- stderr, exit code 2 -- since scripts may check
        // `sylph; echo $?`.
        None => {
            print_grouped_help(HelpStream::Stderr).expect("failed to write help to stderr");
            std::process::exit(2);
        }
        // Explicit `-h`/`--help` as the very first token: stdout, exit 0,
        // matching clap's own handling of top-level help.
        Some("-h") | Some("--help") => {
            print_grouped_help(HelpStream::Stdout).expect("failed to write help to stdout");
            std::process::exit(0);
        }
        // Everything else (a real subcommand, `sylph <subcommand> --help`,
        // `-V`/`--version`, or genuinely invalid usage) is untouched --
        // handled entirely by clap's normal parsing/error/help machinery.
        _ => {
            let cli = Cli::parse();
            match cli.mode {
                Mode::Sketch(sketch_args) => sketch::sketch(sketch_args),
                Mode::Query(contain_args) => contain::contain(contain_args, false),
                Mode::Profile(contain_args) => contain::contain(contain_args, true),
                Mode::Inspect(inspect_args) => inspect::inspect(inspect_args),
                Mode::ConvertDbTwoScreen(args) => twostage_db::run_db_convert(args),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use termcolor::Buffer;

    #[test]
    fn grouped_help_uses_clap_three_colors() {
        let mut buffer = Buffer::ansi();
        write_grouped_help(&mut buffer).unwrap();
        let help = String::from_utf8(buffer.as_slice().to_vec()).unwrap();

        assert!(help.contains("\x1b[32msylph\x1b[0m "));
        assert!(help.contains("\x1b[33mCOMMANDS:\x1b[0m"));
        assert!(help.contains("\x1b[33mADVANCED COMMANDS:\x1b[0m"));
        assert!(help.contains("\x1b[32m--help\x1b[0m"));
        assert!(help.contains("\x1b[32msketch\x1b[0m"));
        assert!(!help.contains("\x1b[32mquery\x1b[0m"));
        assert!(!help.contains("\x1b[38;5;10mquery\x1b[0m"));
        assert!(help.contains("    query                    Coverage-adjusted"));
    }

    #[test]
    fn grouped_help_can_be_rendered_without_ansi() {
        let (expected, _) = grouped_help_text();
        let mut buffer = Buffer::no_color();
        write_grouped_help(&mut buffer).unwrap();

        assert_eq!(buffer.as_slice(), expected.as_bytes());
    }
}
