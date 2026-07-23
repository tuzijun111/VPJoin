//! DP lane proving cost for the three circuits that materialize an
//! intermediate under a released capacity: Q5, GQ3 and GQ4.
//!
//! Every row proves the query circuit at its PINNED Revealing-Join-Size degree
//! with the DP release hosted in lanes, so the released capacity shows up as
//! extra lanes rather than as a bigger domain. The proving and verifying keys
//! are built ONCE per row, outside the timed region; then `reps` proofs are
//! generated and EVERY one of them is verified. The table reports the mean
//! prove time and the min..max spread.
//!
//! This binary replaces
//!
//!   cargo test --release <module>::tests::test_dp_lanes -- --ignored --nocapture
//!
//! for the paper's runs: it drives the same `run_dp_lanes` entry point those
//! tests call, and it makes the PROFILE explicit at the call site. The paper's
//! DP lane numbers are the DEBUG-profile ones, so run it as
//!
//!   cargo run --bin dp_lane_bench -- ...
//!
//! with no `--release`. Nothing is written to disk; the table is printed.
//!
//! Usage:
//!   cargo run --bin dp_lane_bench                    # q5 + gq3 + gq4, all datasets
//!   cargo run --bin dp_lane_bench -- q5              # one query
//!   cargo run --bin dp_lane_bench -- gq3:lastfm      # one graph query, one dataset
//!   cargo run --bin dp_lane_bench -- reps=5 gq4      # 5 repetitions instead of 3
//!   cargo run --bin dp_lane_bench -- --help          # the selector syntax
//!
//! `q5` always runs on tpch-60K; a `:dataset` suffix on it is ignored. `gq3`
//! and `gq4` expand over lastfm, facebook and wiki unless a suffix pins one.
//! `reps=N` may appear anywhere in the argument list and overrides VPJOIN_REPS;
//! the default is 3.
//!
//! Environment:
//!   VPJOIN_PRIVACY  dp (default) | rjs | legacy -- the regime that sizes the
//!                   materialized bags. `rjs` is the Revealing-Join-Size
//!                   baseline (no pad, exact cardinality leaked) and `legacy`
//!                   replays the hand-picked constants, so the same binary
//!                   produces the comparison rows.
//!   VPJOIN_EPS      total epsilon of the DP release (default 0.1)
//!   VPJOIN_DELTA    delta of the DP release (default 1e-5)
//!   VPJOIN_DP_SEED  seed of `bench_queries::dp_rng`, so a re-run reproduces
//!                   the same released capacities (default 20260721)
//!   VPJOIN_REPS     repetitions, overridden by `reps=N`
//!   VPJOIN_PLAN_ONLY=1  print the planned degree, lane counts, released
//!                   capacities and pads, then exit BEFORE any keygen or
//!                   proving. Cheap; use it to check a selection first.

use halo2_experiments::bench_queries::{Privacy, GRAPH_DATASETS};
use halo2_experiments::dp_lane::{DpLanePlan, DpLaneRun};

const ALL: &[&str] = &["q5", "gq3", "gq4"];

/// Q5's dataset is fixed: the harness proves it over the full TPC-H tables.
const TPCH: &str = "tpch-60K";

fn print_header() {
    println!(
        "{:<6}{:<10}{:>4}{:>8}{:>16}{:>16}{:>11}{:>18}{:>10}",
        "query",
        "dataset",
        "k",
        "lanes",
        "capacity",
        "pad",
        "prove(s)",
        "prove min..max",
        "verify(s)"
    );
    // 6 + 10 + 4 + 8 + 16 + 16 + 11 + 18 + 10, the widths above.
    println!("{}", "-".repeat(99));
}

/// Printed as each row completes, so a long run shows progress and an
/// interrupted one still yields data. `run` is `None` in plan-only mode.
fn print_row(plan: &DpLanePlan, run: Option<&DpLaneRun>) {
    use std::io::Write;
    let (mean, span, verify) = match run {
        Some(r) => (
            format!("{:.2}", r.prove_mean()),
            format!("{:.2}..{:.2}", r.prove_min(), r.prove_max()),
            format!("{:.2}", r.verify_s),
        ),
        None => ("-".to_string(), "-".to_string(), "-".to_string()),
    };
    println!(
        "{:<6}{:<10}{:>4}{:>8}{:>16}{:>16}{:>11}{:>18}{:>10}",
        plan.query,
        plan.dataset,
        plan.k,
        plan.lanes_str(),
        plan.capacity_str(),
        plan.pads_str(),
        mean,
        span,
        verify
    );
    let _ = std::io::stdout().flush();
}

fn usage() -> String {
    format!(
        "usage: dp_lane_bench [reps=N] [query[:dataset] ...]\n  \
         queries:  {}   (default: all three)\n  \
         datasets: {}   (gq3/gq4 only; q5 is always {})\n  \
         reps=N may appear anywhere and overrides VPJOIN_REPS (default 3).\n  \
         Results go to stdout, per-row progress to stderr, nothing to disk.\n  \
         VPJOIN_PLAN_ONLY=1 prints the geometry and exits before any keygen.\n  \
         VPJOIN_PRIVACY=dp|rjs|legacy, VPJOIN_EPS, VPJOIN_DELTA, VPJOIN_DP_SEED.",
        ALL.join(" "),
        GRAPH_DATASETS.join(" "),
        TPCH
    )
}

/// Clean error on stderr and a non-zero exit, matching `vpjoin_bench` and
/// `pone_graph_bench`; a bad selector is a usage mistake, not a bug.
fn die(msg: String) -> ! {
    eprintln!("{}", msg);
    eprintln!("{}", usage());
    std::process::exit(2)
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", usage());
        return;
    }

    // `reps=N` may appear anywhere in the argument list; everything else is a
    // query selector.
    let mut reps_arg: Option<usize> = None;
    let mut args: Vec<String> = Vec::new();
    for a in raw {
        if let Some(v) = a.strip_prefix("reps=") {
            match v.parse::<usize>() {
                Ok(n) if n >= 1 => reps_arg = Some(n),
                _ => die(format!("reps= expects a positive integer, got `{}`", v)),
            }
        } else {
            args.push(a);
        }
    }

    // "gq3" runs all three datasets; "gq3:wiki" runs just that one. "q5" is
    // tpch-60K whatever suffix it carries.
    let want: Vec<(String, Option<String>)> = if args.is_empty() {
        ALL.iter().map(|s| (s.to_string(), None)).collect()
    } else {
        args.iter()
            .map(|a| {
                let (q, ds) = match a.split_once(':') {
                    Some((q, d)) => (q.to_string(), Some(d.to_string())),
                    None => (a.clone(), None),
                };
                if !ALL.contains(&q.as_str()) {
                    die(format!(
                        "unknown query `{}` (expected one of {})",
                        q,
                        ALL.join(" ")
                    ));
                }
                if let Some(d) = &ds {
                    if q.starts_with("gq") && !GRAPH_DATASETS.contains(&d.as_str()) {
                        die(format!(
                            "unknown dataset `{}` (expected one of {})",
                            d,
                            GRAPH_DATASETS.join(" ")
                        ));
                    }
                }
                (q, ds)
            })
            .collect()
    };

    let privacy = match std::env::var("VPJOIN_PRIVACY").as_deref().unwrap_or("dp") {
        "rjs" => Privacy::Rjs,
        "legacy" => Privacy::Legacy,
        "dp" => Privacy::Dp {
            epsilon: std::env::var("VPJOIN_EPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.1),
            delta: std::env::var("VPJOIN_DELTA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1e-5),
        },
        other => panic!(
            "unknown VPJOIN_PRIVACY {} (expected dp | rjs | legacy)",
            other
        ),
    };

    // Precedence: reps=N on the command line, else VPJOIN_REPS, else 3.
    let reps: usize = reps_arg
        .or_else(|| {
            std::env::var("VPJOIN_REPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|n| *n >= 1)
        })
        .unwrap_or(3);

    let plan_only = std::env::var("VPJOIN_PLAN_ONLY")
        .map(|v| v == "1")
        .unwrap_or(false);

    println!(
        "DP lane proving cost, {} build. privacy={} seed={} reps={} per row; keys are\n\
         built once outside the timed region and every proof is verified.\n",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        privacy.label(),
        std::env::var("VPJOIN_DP_SEED").unwrap_or_else(|_| "default".into()),
        reps
    );

    if plan_only {
        println!("VPJOIN_PLAN_ONLY=1 -- geometry only, nothing is proved.");
    }

    // Expand the selectors into concrete (query, dataset) rows.
    let mut rows: Vec<(String, String)> = Vec::new();
    for (q, only_ds) in &want {
        if q == "q5" {
            // Q5's dataset is fixed; any `:suffix` on it is ignored.
            rows.push((q.clone(), TPCH.to_string()));
        } else {
            for d in GRAPH_DATASETS
                .iter()
                .filter(|d| only_ds.as_deref().map_or(true, |o| o == **d))
            {
                rows.push((q.clone(), d.to_string()));
            }
        }
    }

    let plan_for = |q: &str, ds: &str| -> DpLanePlan {
        match q {
            "q5" => halo2_experiments::sql::q5_obj_dp::plan_dp_lanes(privacy),
            "gq3" => halo2_experiments::graph_sql::g_sql3_obj_dp::plan_dp_lanes(ds, privacy),
            _ => halo2_experiments::graph_sql::g_sql4_obj_dp::plan_dp_lanes(ds, privacy),
        }
    };

    if plan_only {
        // Release the capacities FIRST: the DP mechanism narrates each release
        // on stdout, and letting that run before the header keeps the table
        // contiguous.
        let plans: Vec<DpLanePlan> = rows.iter().map(|(q, ds)| plan_for(q, ds)).collect();
        println!();
        print_header();
        for p in &plans {
            print_row(p, None);
        }
        print_legend();
        println!("\nlane geometry per row:");
        for p in &plans {
            println!(
                "  {:<4} {:<9} k={} lane_rows={} lanes={} true={} capacity={} effective={}",
                p.query,
                p.dataset,
                p.k,
                p.lane_rows,
                p.lanes_str(),
                p.true_size_str(),
                p.capacity_str(),
                p.lanes
                    .iter()
                    .map(|c| (c * p.lane_rows).to_string())
                    .collect::<Vec<_>>()
                    .join("/")
            );
        }
        println!("\nVPJOIN_PLAN_ONLY=1 -- stopping before any keygen or proving.");
        return;
    }

    print_header();
    for (q, ds) in &rows {
        eprint!("  {} on {} ... ", q, ds);
        // `proof_path = None`: this binary writes nothing to disk.
        let run: DpLaneRun = match q.as_str() {
            "q5" => halo2_experiments::sql::q5_obj_dp::run_dp_lanes(privacy, reps, None),
            "gq3" => {
                halo2_experiments::graph_sql::g_sql3_obj_dp::run_dp_lanes(ds, privacy, reps, None)
            }
            _ => halo2_experiments::graph_sql::g_sql4_obj_dp::run_dp_lanes(ds, privacy, reps, None),
        };
        eprintln!(
            "ok (k={}, lanes={}, keygen {:.2}s, prove {:.2}..{:.2}s, proof {} B)",
            run.plan.k,
            run.plan.lanes_str(),
            run.keygen_s,
            run.prove_min(),
            run.prove_max(),
            run.proof_bytes
        );
        print_row(&run.plan, Some(&run));
    }

    print_legend();
}

fn print_legend() {
    println!(
        "\ncapacity = released bag capacity (true size + DP pad), one per laned bag\n\
         pad      = pad_extra per materialized intermediate, in the circuit's own order\n\
         lanes    = lane count per laned bag; GQ4 lanes two bags, so it probes lanes[0]*lanes[1] times"
    );
}
