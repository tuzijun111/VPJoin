//! DP lane proving cost for the three circuits that materialize an
//! intermediate under a released capacity: Q5, GQ3 and GQ4.
//!
//! Every row proves the query circuit at a PINNED degree with the DP release
//! hosted in lanes, so the released capacity shows up as extra lanes rather
//! than as a bigger domain. GQ3 and GQ4 pin the Revealing-Join-Size degree of
//! their dataset; Q5 pins k = 17, the degree its single-column circuit takes at
//! the sweep's reference budget eps = 0.1, so at that budget a lane row and the
//! baseline row it is compared against share a domain size (at smaller eps the
//! single-column circuit escalates to 18/19/20 and the lane circuit does not).
//! The proving and verifying keys
//! are built ONCE per row, outside the timed region; then `reps` proofs are
//! generated and EVERY one of them is verified. The table reports the mean
//! prove time and the min..max spread.
//!
//! This binary replaces
//!
//!   cargo test --release <module>::tests::test_dp_lanes -- --ignored --nocapture
//!
//! for the paper's runs: it drives the same `run_dp_lanes` entry point those
//! tests call, and it makes the PROFILE explicit at the call site.
//!
//! RELEASE is not something the caller has to remember: `[profile.dev]` in
//! `Cargo.toml` carries the release settings, so a plain `cargo run` is
//! optimized and no invocation of this binary can quietly produce an
//! unoptimized number. That matters here because a lane row is only meaningful
//! next to the single-column row it is compared against, which `cargo vpjoin`
//! produces optimized. The header line of every run records the profile.
//! Nothing is written to disk; the table is printed.
//!
//! Usage:
//!   cargo run --bin dp_lane_bench                    # q5 + gq3 + gq4, all datasets
//!   cargo run --bin dp_lane_bench -- q5              # one query
//!   cargo run --bin dp_lane_bench -- gq3:lastfm      # one graph query, one dataset
//!   cargo run --bin dp_lane_bench -- reps=5 gq4      # 5 repetitions instead of 3
//!   cargo run --bin dp_lane_bench -- --help          # the selector syntax
//!
//! `cargo dp-lane ...` is an alias for the same thing, one word shorter.
//!
//! `q5` always runs on tpch-60K; a `:dataset` suffix on it is ignored. `gq3`
//! and `gq4` expand over lastfm, facebook and wiki unless a suffix pins one.
//! `reps=N` may appear anywhere in the argument list and overrides VPJOIN_REPS;
//! the default is 3.
//!
//! SWEEPS. `VPJOIN_EPS` takes a comma-separated list, so one invocation covers
//! a whole privacy budget curve:
//!
//!   VPJOIN_EPS=0.01,0.02,0.05,0.1 cargo run --bin dp_lane_bench -- gq3 gq4
//!
//! The epsilons are visited in the order given and the selected rows within
//! each, so the table reads top to bottom as the sweep progresses. A row whose
//! released capacity needs more lanes than the circuit can host, or whose
//! sections do not fit the pinned degree, is reported as SKIPPED with the
//! reason and the sweep continues; skipped rows do not change the exit status.
//! Each dataset is parsed once and reused across epsilons. Keys are NOT reused:
//! the lane count changes the circuit shape, so keygen belongs to the
//! configuration. (Two configurations can land on the same shape; caching keys
//! across them is left alone here.)
//!
//! Environment:
//!   VPJOIN_PRIVACY  dp (default) | rjs | legacy -- the regime that sizes the
//!                   materialized bags. `rjs` is the Revealing-Join-Size
//!                   baseline (no pad, exact cardinality leaked) and `legacy`
//!                   replays the hand-picked constants, so the same binary
//!                   produces the comparison rows.
//!   VPJOIN_EPS      total epsilon of the DP release, or a comma-separated list
//!                   of them to sweep (default 0.1)
//!   VPJOIN_DELTA    delta of the DP release (default 1e-5)
//!   VPJOIN_DP_SEED  seed of `bench_queries::dp_rng`, so a re-run reproduces
//!                   the same released capacities (default 20260721). The draw
//!                   depends on the query, the dataset and this seed only, so a
//!                   row of a sweep is identical to the same row run alone.
//!   VPJOIN_REPS     repetitions, overridden by `reps=N`
//!   VPJOIN_PLAN_ONLY=1  print the planned degree, lane counts, released
//!                   capacities and pads, then exit BEFORE any keygen or
//!                   proving. Cheap; use it to check a selection first.
//!   VPJOIN_FULL_LANES=1  fill EVERY lane to `lane_rows` instead of stopping
//!                   the last one at the released capacity. Off by default, in
//!                   which case the `dp` and `rjs` regimes stop short (their
//!                   capacity is public, so the verifying key may pin it) and
//!                   `legacy` fills, since its pad is a public constant and a
//!                   capacity in the key would pin the true bag size. Setting
//!                   it forces the full layout everywhere, which is the A/B
//!                   arm for measuring what the short last lane is worth: same
//!                   binary, same witness, same lane count.

use halo2_experiments::bench_queries::{Privacy, GRAPH_DATASETS};
use halo2_experiments::dp_lane::{DpLanePlan, DpLaneRun};

const ALL: &[&str] = &["q5", "gq3", "gq4"];

/// Q5's dataset is fixed: the harness proves it over the full TPC-H tables.
const TPCH: &str = "tpch-60K";

/// One cell of the sweep: the privacy regime to size the bags with, plus the
/// epsilon to label it by (`None` outside the dp regime, where no budget is
/// spent and the column is a dash).
#[derive(Clone, Copy)]
struct Budget {
    eps: Option<f64>,
    privacy: Privacy,
}

impl Budget {
    fn eps_str(&self) -> String {
        match self.eps {
            Some(e) => format!("{}", e),
            None => "-".to_string(),
        }
    }
}

fn print_header() {
    println!(
        "{:<6}{:<10}{:>7}{:>4}{:>8}{:>20}{:>20}{:>11}{:>18}{:>10}",
        "query",
        "dataset",
        "eps",
        "k",
        "lanes",
        "capacity",
        "pad",
        "prove(s)",
        "prove min..max",
        "verify(s)"
    );
    // 6 + 10 + 7 + 4 + 8 + 20 + 20 + 11 + 18 + 10, the widths above.
    println!("{}", "-".repeat(114));
}

/// Printed as each row completes, so a long run shows progress and an
/// interrupted one still yields data. `run` is `None` in plan-only mode.
fn print_row(plan: &DpLanePlan, eps: &str, run: Option<&DpLaneRun>) {
    let (mean, span, verify) = match run {
        Some(r) => (
            format!("{:.2}", r.prove_mean()),
            format!("{:.2}..{:.2}", r.prove_min(), r.prove_max()),
            format!("{:.2}", r.verify_s),
        ),
        None => ("-".to_string(), "-".to_string(), "-".to_string()),
    };
    println!(
        "{:<6}{:<10}{:>7}{:>4}{:>8}{:>20}{:>20}{:>11}{:>18}{:>10}",
        plan.query,
        plan.dataset,
        eps,
        plan.k,
        plan.lanes_str(),
        plan.capacity_str(),
        plan.pads_str(),
        mean,
        span,
        verify
    );
    flush();
}

/// A configuration that cannot be built keeps its identifying columns, so the
/// sweep's shape stays readable, and carries the reason inline.
fn print_skip(query: &str, dataset: &str, eps: &str, reason: &str) {
    println!(
        "{:<6}{:<10}{:>7}{:>4}{:>8}{:>20}{:>20}  SKIPPED  {}",
        query, dataset, eps, "-", "-", "-", "-", reason
    );
    flush();
}

fn flush() {
    use std::io::Write;
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
         VPJOIN_PRIVACY=dp|rjs|legacy, VPJOIN_DELTA, VPJOIN_DP_SEED.\n  \
         VPJOIN_EPS takes one budget or a comma-separated list to sweep,\n  \
         e.g. VPJOIN_EPS=0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10.",
        ALL.join(" "),
        GRAPH_DATASETS.join(" "),
        TPCH
    )
}

/// Clean error on stderr and a non-zero exit, matching `vpjoin_bench` and
/// `pone_graph_bench`; a bad selector is a usage mistake, not a bug. A row that
/// merely cannot be built is NOT one of these: it is reported and skipped.
fn die(msg: String) -> ! {
    eprintln!("{}", msg);
    eprintln!("{}", usage());
    std::process::exit(2)
}

/// `VPJOIN_EPS` as a list. One value is the single-budget case; several sweep,
/// in the order given.
fn epsilons() -> Vec<f64> {
    let raw = std::env::var("VPJOIN_EPS").unwrap_or_else(|_| "0.1".to_string());
    let mut out = Vec::new();
    for tok in raw.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            die(format!(
                "VPJOIN_EPS=`{}` has an empty entry; expected one budget or a \
                 comma-separated list of positive numbers",
                raw
            ));
        }
        match t.parse::<f64>() {
            Ok(v) if v.is_finite() && v > 0.0 => out.push(v),
            Ok(v) => die(format!(
                "VPJOIN_EPS entry `{}` must be a positive finite number, got {}",
                t, v
            )),
            Err(e) => die(format!("VPJOIN_EPS entry `{}` is not a number: {}", t, e)),
        }
    }
    out
}

fn delta() -> f64 {
    match std::env::var("VPJOIN_DELTA") {
        Err(_) => 1e-5,
        Ok(raw) => match raw.trim().parse::<f64>() {
            Ok(v) if v > 0.0 && v < 0.5 => v,
            Ok(v) => die(format!("VPJOIN_DELTA must lie in (0, 1/2), got {}", v)),
            Err(e) => die(format!("VPJOIN_DELTA=`{}` is not a number: {}", raw, e)),
        },
    }
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

    // Outside the dp regime no budget is spent, so VPJOIN_EPS is not read at
    // all and the eps column is a dash.
    let (budgets, regime_label) = match std::env::var("VPJOIN_PRIVACY").as_deref().unwrap_or("dp") {
        "rjs" => (
            vec![Budget {
                eps: None,
                privacy: Privacy::Rjs,
            }],
            Privacy::Rjs.label(),
        ),
        "legacy" => (
            vec![Budget {
                eps: None,
                privacy: Privacy::Legacy,
            }],
            Privacy::Legacy.label(),
        ),
        "dp" => {
            let del = delta();
            let eps = epsilons();
            let label = format!(
                "dp(eps={} del={})",
                eps.iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                del
            );
            (
                eps.iter()
                    .map(|e| Budget {
                        eps: Some(*e),
                        privacy: Privacy::Dp {
                            epsilon: *e,
                            delta: del,
                        },
                    })
                    .collect(),
                label,
            )
        }
        other => die(format!(
            "unknown VPJOIN_PRIVACY `{}` (expected dp | rjs | legacy)",
            other
        )),
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
         built once outside the timed region and every proof is verified.",
        halo2_experiments::bench_queries::build_profile(),
        regime_label,
        std::env::var("VPJOIN_DP_SEED").unwrap_or_else(|_| "default".into()),
        reps
    );
    // A declared degree cap changes every capacity a graph row is proved at, so
    // it belongs in the banner and not only in the per-row `[dp]` line: two
    // sweeps run under different caps are not comparable, and the difference is
    // otherwise invisible in the table.
    {
        let caps: Vec<String> = ["lastfm", "facebook", "wiki"]
            .iter()
            .filter_map(|d| {
                halo2_experiments::bench_queries::declared_degree_cap(d)
                    .map(|t| format!("{}={}", d, t))
            })
            .collect();
        let overridden = std::env::var("VPJOIN_TAU_PUB").ok();
        match (overridden, caps.is_empty()) {
            (Some(v), _) => println!(
                "declared degree cap: VPJOIN_TAU_PUB={} for every graph row (overrides the \
                 built-in caps)",
                v
            ),
            (None, false) => println!(
                "declared degree cap: {} (built in; VPJOIN_TAU_PUB overrides, =0 restores the \
                 released-tau mechanism)",
                caps.join(", ")
            ),
            (None, true) => {}
        }
    }
    // Which layout the last lane gets, since it changes both the prover's work
    // and what the verifying key discloses. Two runs under different settings
    // are two different circuits, so it belongs in the banner next to the
    // degree cap rather than only in the source.
    println!(
        "last lane: {}",
        if std::env::var("VPJOIN_FULL_LANES").as_deref() == Ok("1") {
            "FULL (VPJOIN_FULL_LANES=1 forces every lane to lane_rows in every regime; \
             the vk pins only the lane count)"
        } else {
            "stops at the released capacity under dp/rjs, where that capacity is public \
             and the vk may pin it; full under legacy, whose pad is a public constant"
        }
    );
    println!();

    // `cargo run` is optimized (see `[profile.dev]` in Cargo.toml), so this
    // only fires under the `test` profile or a hand-rolled one that leaves the
    // assertions on. Those timings are not comparable with the ones the sweep
    // reports, so say it rather than let the one-word label carry it.
    if !plan_only && cfg!(debug_assertions) {
        println!(
            "WARNING: this build has debug assertions ON, so it is NOT the profile the\n\
             single-column rows are measured under and the timings below are not comparable\n\
             with them. Run it as `cargo run --bin dp_lane_bench -- ...` (or `cargo dp-lane`).\n"
        );
    }

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

    // The CHECKED planning path: an infeasible release is an `Err` carrying the
    // reason, never a panic, so one bad cell cannot take the sweep down.
    let plan_for = |q: &str, ds: &str, privacy: Privacy| -> Result<DpLanePlan, String> {
        match q {
            "q5" => halo2_experiments::sql::q5_obj_dp::try_plan_dp_lanes(privacy),
            "gq3" => halo2_experiments::graph_sql::g_sql3_obj_dp::try_plan_dp_lanes(ds, privacy),
            _ => halo2_experiments::graph_sql::g_sql4_obj_dp::try_plan_dp_lanes(ds, privacy),
        }
    };

    let mut skipped = 0usize;

    if plan_only {
        // Release the capacities FIRST: the DP mechanism narrates each release
        // on stderr, and letting that run before the header keeps the table
        // contiguous. Planning is cheap, so nothing is lost by batching it.
        let planned: Vec<(&Budget, &(String, String), Result<DpLanePlan, String>)> = budgets
            .iter()
            .flat_map(|b| {
                rows.iter()
                    .map(move |r| (b, r, plan_for(&r.0, &r.1, b.privacy)))
            })
            .collect();
        println!();
        print_header();
        for (b, (q, ds), plan) in &planned {
            match plan {
                Ok(p) => print_row(p, &b.eps_str(), None),
                Err(why) => {
                    skipped += 1;
                    print_skip(q, ds, &b.eps_str(), why);
                }
            }
        }
        print_legend();
        println!("\nlane geometry per row:");
        for (b, _, plan) in &planned {
            let Ok(p) = plan else { continue };
            println!(
                "  {:<4} {:<9} eps={:<6} k={} lane_rows={} lanes={} true={} capacity={} \
                 effective={}",
                p.query,
                p.dataset,
                b.eps_str(),
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
        print_skipped(skipped);
        println!("\nVPJOIN_PLAN_ONLY=1 -- stopping before any keygen or proving.");
        return;
    }

    print_header();
    for b in &budgets {
        for (q, ds) in &rows {
            let eps = b.eps_str();
            eprint!("  {} on {} (eps={}) ... ", q, ds, eps);
            // Plan first: a release that cannot be hosted is reported here,
            // before any SRS is read or any key is built. `run_dp_lanes`
            // re-derives the same geometry internally -- the release is a
            // deterministic function of the seed, the query and the dataset --
            // so a feasible row pays one extra planning pass, which is cheap
            // next to keygen.
            if let Err(why) = plan_for(q, ds, b.privacy) {
                eprintln!("SKIPPED ({})", why);
                skipped += 1;
                print_skip(q, ds, &eps, &why);
                continue;
            }
            // `proof_path = None`: this binary writes nothing to disk. Keygen
            // happens inside, per configuration: the lane count is part of the
            // circuit shape, so keys cannot be hoisted out of this loop.
            let run: DpLaneRun = match q.as_str() {
                "q5" => halo2_experiments::sql::q5_obj_dp::run_dp_lanes(b.privacy, reps, None),
                "gq3" => halo2_experiments::graph_sql::g_sql3_obj_dp::run_dp_lanes(
                    ds, b.privacy, reps, None,
                ),
                _ => halo2_experiments::graph_sql::g_sql4_obj_dp::run_dp_lanes(
                    ds, b.privacy, reps, None,
                ),
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
            print_row(&run.plan, &eps, Some(&run));
        }
    }

    print_legend();
    print_skipped(skipped);
}

/// A skip is information, not a failure: it is counted and reported, and the
/// exit status stays 0.
fn print_skipped(skipped: usize) {
    if skipped == 0 {
        return;
    }
    println!(
        "\n{} row{} skipped as infeasible.",
        skipped,
        if skipped == 1 { "" } else { "s" }
    );
    flush();
}

fn print_legend() {
    println!(
        "\ncapacity = released bag capacity (true size + DP pad), one per laned bag\n\
         pad      = pad_extra per materialized intermediate, in the circuit's own order\n\
         lanes    = lane count per laned bag; GQ4's two column roles share ONE laned relation, so it probes lanes[0]^2 times\n\
         eps      = the VPJOIN_EPS entry this row was released under (dash outside the dp regime)"
    );
}
