//! The `laya` command line interface.
//!
//!     laya route --state "I was charged twice"          # which checkpoint? (no weights)
//!     laya ask --preset triage --state "..." --model ~/laya_models/laya
//!     laya info ~/laya_models/laya                      # what is in a checkpoint
//!     laya presets                                      # built-in question sets

use std::io::Read;
use std::process::ExitCode;

use laya_rs::agent::Agent;
use laya_rs::criteria::Questions;
use laya_rs::presets;
use laya_rs::router::{Router, RouterOptions};

const HELP: &str = "\
laya — fast, non-autoregressive System 1 decision engine

USAGE:
    laya <COMMAND> [OPTIONS]

COMMANDS:
    route      Decide which checkpoint should handle a state (no weights needed)
    ask        Run a question preset through a checkpoint
    info       Show what a checkpoint contains
    presets    List the built-in question presets

OPTIONS:
    -h, --help       Print this help
    -V, --version    Print the version

EXAMPLES:
    laya route --state \"I was charged twice, please refund\"
    laya route --state \"मुझे दो बार शुल्क लिया गया\" --json
    laya ask --preset triage --state \"...\" --model ~/laya_models/laya
    laya info ~/laya_models/laya
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        print!("{}", HELP);
        return ExitCode::SUCCESS;
    };
    let rest = &args[1..];
    let result = match cmd.as_str() {
        "-h" | "--help" | "help" => {
            print!("{}", HELP);
            return ExitCode::SUCCESS;
        }
        "-V" | "--version" | "version" => {
            println!("laya {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        "route" => route(rest),
        "ask" => ask(rest),
        "info" => info(rest),
        "presets" => presets_cmd(rest),
        other => Err(format!("unknown command {:?}\n\nTry `laya --help`.", other)),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("laya: {}", e);
            ExitCode::FAILURE
        }
    }
}

/// Pull `--flag value` / `--flag=value` out of the args, returning the value and the
/// remaining positional arguments.
fn flag(args: &[String], name: &str) -> (Option<String>, Vec<String>) {
    let mut value = None;
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == name {
            if let Some(v) = args.get(i + 1) {
                value = Some(v.clone());
                i += 2;
            } else {
                rest.push(a.clone());
                i += 1;
            }
        } else if let Some(v) = a.strip_prefix(&format!("{}=", name)) {
            value = Some(v.to_string());
            i += 1;
        } else {
            rest.push(a.clone());
            i += 1;
        }
    }
    (value, rest)
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

/// Read the state from `--state`, `--state-file`, or stdin (in that order).
fn read_state(args: &[String]) -> Result<String, String> {
    if let (Some(text), _) = flag(args, "--state") {
        return Ok(text);
    }
    if let (Some(path), _) = flag(args, "--state-file") {
        return std::fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {}", path, e));
    }
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .map_err(|e| format!("cannot read stdin: {}", e))?;
    Ok(buf)
}

fn preset_by_name(name: &str) -> Result<Questions, String> {
    Ok(match name {
        "triage" => presets::triage_questions(),
        "email" => presets::email_questions(None),
        "guard" => presets::guard_questions(),
        "moderation" => presets::moderation_questions(),
        "router" => presets::router_questions(),
        other => {
            return Err(format!(
                "unknown preset {:?}; choose one of triage, email, guard, moderation, router",
                other
            ))
        }
    })
}

fn route(args: &[String]) -> Result<(), String> {
    if has_flag(args, "-h") || has_flag(args, "--help") {
        println!(
            "Decide which checkpoint should handle a state. No weights are loaded.\n\n\
             USAGE: laya route [--state TEXT] [--state-file PATH] [--preset NAME]\n\
             \x20                  [--model NAME] [--task NAME] [--lang CODE] [--json]"
        );
        return Ok(());
    }
    let (preset_name, args) = flag(args, "--preset");
    let (model, args) = (flag(&args, "--model").0, args);
    let (task, args) = (flag(&args, "--task").0, args);
    let (lang, args) = (flag(&args, "--lang").0, args);
    let as_json = has_flag(&args, "--json");
    let questions = preset_by_name(preset_name.as_deref().unwrap_or("triage"))?;
    let state_text = read_state(&args)?;
    let state = serde_json::json!({"message": state_text.trim()});

    let router = Router::new(RouterOptions::default()).map_err(|e| e.to_string())?;
    let decision = router
        .route(
            &state,
            &questions,
            model.as_deref(),
            task.as_deref(),
            lang.as_deref(),
        )
        .map_err(|e| e.to_string())?;

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&decision.to_value()).unwrap()
        );
    } else {
        println!("model   {}", decision.model);
        println!("repo    {}", decision.repo.id());
        println!("reason  {}", decision.reason);
        if let Some(wf) = &decision.workflow {
            println!("workflow {}", wf);
        }
    }
    Ok(())
}

fn ask(args: &[String]) -> Result<(), String> {
    if has_flag(args, "-h") || has_flag(args, "--help") {
        println!(
            "Run a question preset through a checkpoint.\n\n\
             USAGE: laya ask [--state TEXT] [--state-file PATH] [--preset NAME]\n\
             \x20               [--model DIR] [--device cpu|metal|cuda] [--json]"
        );
        return Ok(());
    }
    let (preset_name, args) = flag(args, "--preset");
    let (model_dir, args) = (flag(&args, "--model").0, args);
    let (device, args) = (flag(&args, "--device").0, args);
    let as_json = has_flag(&args, "--json");
    let questions = preset_by_name(preset_name.as_deref().unwrap_or("triage"))?;
    let state_text = read_state(&args)?;
    let state = serde_json::json!({"message": state_text.trim()});

    let model_dir = model_dir.unwrap_or_else(|| {
        format!(
            "{}/laya_models/laya",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let agent =
        Agent::load(&model_dir, device.as_deref(), None, None).map_err(|e| e.to_string())?;
    let out = agent
        .system_one(&state, &questions)
        .map_err(|e| e.to_string())?;

    if as_json {
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
        return Ok(());
    }
    let answers = out.get("answers").and_then(|a| a.as_object());
    let Some(answers) = answers else {
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
        return Ok(());
    };
    for (qid, a) in answers {
        let line = match a.get("type").and_then(|t| t.as_str()) {
            Some("choice") => format!(
                "{} -> {} (p={:.3})",
                qid,
                a.get("choice").and_then(|c| c.as_str()).unwrap_or("?"),
                a.get("probabilities")
                    .and_then(|p| p.get(a.get("choice").and_then(|c| c.as_str()).unwrap_or("")))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::NAN)
            ),
            Some("score") => format!(
                "{} -> {:.3} (conf={:.3})",
                qid,
                a.get("score").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                a.get("confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::NAN)
            ),
            Some("noul") => format!(
                "{} -> {} (p={:.3})",
                qid,
                if a.get("noul").and_then(|v| v.as_f64()).unwrap_or(0.0) > 0.5 {
                    "true"
                } else {
                    "false"
                },
                a.get("noul").and_then(|v| v.as_f64()).unwrap_or(f64::NAN)
            ),
            _ => format!("{} -> {}", qid, a),
        };
        println!("{}", line);
    }
    if let Some(u) = out.get("usage") {
        println!(
            "usage   {} tokens in, {} out",
            u.get("input_tokens").and_then(|v| v.as_i64()).unwrap_or(0),
            u.get("output_tokens").and_then(|v| v.as_i64()).unwrap_or(0)
        );
    }
    Ok(())
}

fn info(args: &[String]) -> Result<(), String> {
    if has_flag(args, "-h") || has_flag(args, "--help") {
        println!("Show what a checkpoint contains.\n\nUSAGE: laya info [MODEL_DIR]");
        return Ok(());
    }
    let (_, positional) = flag(args, "--model");
    let dir = positional.first().cloned().unwrap_or_else(|| {
        format!(
            "{}/laya_models/laya",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let agent = Agent::load(&dir, Some("cpu"), None, None).map_err(|e| e.to_string())?;
    // The encoder geometry lives in encoder/config.json, not in the agent config (whose
    // `encoder` key is just a repo name).
    let enc = agent.encoder_config();
    let hidden = enc.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(0);
    let layers = enc
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let vocab = enc.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(0);
    let intermediate = enc
        .get("intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let n_act = agent
        .config()
        .get("act_costs")
        .and_then(|c| c.as_object())
        .map(|m| m.len() as u64 + 1)
        .unwrap_or(1);

    println!("checkpoint   {}", dir);
    println!("device       {}", agent.device());
    println!("max_len      {}", agent.max_len());
    println!("head_max_len {}", agent.head_max_len());
    println!(
        "encoder      {} layers x {} hidden x {} intermediate",
        layers, hidden, intermediate
    );
    println!("vocab        {}", vocab);
    println!("actions      {}", n_act);
    Ok(())
}

fn presets_cmd(args: &[String]) -> Result<(), String> {
    if has_flag(args, "-h") || has_flag(args, "--help") {
        println!("List the built-in question presets.\n\nUSAGE: laya presets");
        return Ok(());
    }
    for (name, q) in [
        ("triage", presets::triage_questions()),
        ("email", presets::email_questions(None)),
        ("guard", presets::guard_questions()),
        ("moderation", presets::moderation_questions()),
        ("router", presets::router_questions()),
    ] {
        println!("{} ({} questions)", name, q.ids().len());
        for id in q.ids() {
            println!("    {}", id);
        }
    }
    Ok(())
}
