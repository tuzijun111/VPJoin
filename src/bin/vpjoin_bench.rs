//! Single-command benchmark driver for every VPJoin query (revision).
//!
//! Runs the 5 TPC-H queries and the 4 graph queries on each of the 3 SNAP
//! datasets (17 rows), in one of two modes, and writes one CSV.
//!
//!   simplification -- the pure query circuit, i.e. exactly what
//!               `cargo test ... sql::qN_obj::tests::test_1` proves.
//!   full     -- the same circuits plus the complete commitment layer:
//!               a published per-column Pedersen commitment to the dataset,
//!               the in-circuit check that the query's witness columns equal
//!               the committed data, and the column openings.
//!   commit   -- ONLY the commitment layer, in the corresponding query
//!               circuit's own domain (same k). The query proof is not re-run,
//!               so these numbers add to the matching `simplification` row. Much
//!               cheaper than `full`, which re-proves the query.
//!
//! The difference between the two CSVs is the cost of making the proof bind to
//! a committed database instead of to an unauthenticated input.
//!
//! Each circuit is proved at the fixed degree given by
//! `bench_queries::degree_for`; degrees are never searched for, so no prover
//! time is spent on a `k` that is then discarded.
//!
//! Usage:
//!   cargo vpjoin <simplification|full|commit> [query[:dataset] ...]
//!   cargo vpjoin <simplification|full|commit> [out.csv] [query[:dataset] ...]
//!
//!   Every result is printed to stdout, and the run ends with an aligned
//!   summary table of all completed rows, so the terminal output alone is a
//!   complete result.  An argument ending in `.csv` is taken as an output file
//!   and additionally writes the same rows there; with no such argument no
//!   file is written at all.
//!
//!   queries:  q3 q5 q8 q9 q18 gq1 gq2 gq3 gq4     (default: all)
//!             A graph query runs all three networks; `gq4:lastfm` pins one
//!             row, the same spelling `dp_lane_bench` takes.  Worth using: the
//!             three networks differ by up to 16x in domain size, so gq3/gq4
//!             on lastfm is k=18 against k=22 on facebook and wiki.
//!   env:      VPJOIN_DATA   root holding data/ graph_data/ proof/
//!                           (default: <crate>/src)
//!             VPJOIN_RESUME =1 to keep the rows already in <out.csv> and only
//!                           run the ones that are missing or previously
//!                           FAILED, appending results in place.  Requires an
//!                           <out.csv> argument; ignored without one.
//!             VPJOIN_PRIVACY  how the queries that MATERIALIZE intermediates
//!                           (Q5, GQ3, GQ4) size their bags:
//!                             dp (default) | rjs | oblivious | legacy
//!                           `oblivious` is TDJ's fully oblivious default: each
//!                           materialized bag is padded to its WORST-CASE bound,
//!                           a function of the public input length alone, so no
//!                           DP budget is spent and the bag size is genuinely
//!                           hidden (`rjs` reveals it exactly, and `legacy`'s
//!                           hand-picked constants sit just above the observed
//!                           sizes).  It is normally far too large to prove --
//!                           GQ3/GQ4 land at k=28/31/32 -- so pair it with
//!                           VPJOIN_PLAN_ONLY=1 to read off the degree the fully
//!                           oblivious configuration needs.  See
//!                           `bench_queries::oblivious_wedge_bound` for the
//!                           derivation.
//!                           Q5 supports all three; under dp BOTH bags get a
//!                           release and k grows with the capacities (k=20
//!                           at eps=0.1, k=16 at eps>=1; see q5_pads).
//!                           The other queries materialize nothing and ignore it.
//!             VPJOIN_EPS / VPJOIN_DELTA   total DP budget (default 0.1 / 1e-5)
//!             VPJOIN_DP_SEED  fixes the capacity-release randomness so a
//!                           re-run reproduces the same circuit sizes
//!             VPJOIN_PLAN_ONLY =1 to print the planned degrees and exit
//!                           without proving anything
//!             VPJOIN_MAX_EDGES  cap the edge list of every graph dataset, so a
//!                           graph query runs at a smaller degree.  `k` is NOT
//!                           a free parameter -- it is the height of the
//!                           materialized bag -- so this, not VPJOIN_K, is how
//!                           to move it: GQ3 on all 27,806 lastfm edges has a
//!                           232,943-row wedge (k=18), while the first 21,403
//!                           edges give 131,001 rows, the largest prefix that
//!                           fits k=17.  Bag sizes, DP capacities, the public
//!                           count and the degree are all recomputed from the
//!                           capped list, so the run is self-consistent; it
//!                           just measures a smaller instance than the paper's.
//!             VPJOIN_K      prove at this degree instead of the derived one.
//!                           A LARGER k proves the same circuit in a bigger
//!                           domain; a SMALLER one is rejected by keygen as
//!                           soon as the layout overflows, so it cannot be used
//!                           to shrink a circuit -- use VPJOIN_MAX_EDGES.
//!
//! Drop `--release` to measure under the unoptimized profile, which is the
//! same profile `cargo test` uses.

use halo2_experiments::bench_queries::{
    build_profile, data_root, degree_for, plan_specs, run_one, tpch_label, Mode, Privacy, Row,
    ALL_QUERIES, GRAPH_DATASETS,
};
use std::collections::HashMap;
use std::io::Write;

/// Existing rows in `path`, keyed by "query/dataset".  Rows whose status is a
/// failure are dropped so a resumed run retries them.
fn load_completed(path: &str) -> HashMap<String, String> {
    let mut done = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return done;
    };
    for line in text.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 2 {
            continue;
        }
        let status = f.last().copied().unwrap_or("");
        if status.starts_with("FAILED") {
            continue;
        }
        done.insert(format!("{}/{}", f[0], f[1]), line.to_string());
    }
    done
}

fn usage() -> String {
    format!(
        "usage: vpjoin_bench <simplification|full|commit> [out.csv] [query[:dataset] ...]\n  \
         queries:  {}\n  \
         datasets: {} (graph queries only; default all three)\n  \
         results are always printed to stdout; an argument ending in .csv also\n  \
         writes them to that file (and enables VPJOIN_RESUME=1).",
        ALL_QUERIES.join(" "),
        GRAPH_DATASETS.join(" ")
    )
}

/// Aligned stdout table of every completed row, built from the same CSV fields
/// the file carries so the terminal output is self-sufficient.
fn print_summary(jobs: &[(String, String)], results: &HashMap<String, String>, mode: Mode) {
    let names: Vec<&str> = Row::header().split(',').collect();
    // query,dataset,k + the timing columns + status.  The commitment-layer
    // timings are all zero in baseline mode, so they are only shown when the
    // mode actually measures them.  Columns are selected BY NAME and resolved
    // against `Row::header()`, so reordering or inserting a CSV field can
    // never silently shift the table.
    const HEAD: &[&str] = &[
        "query", "dataset", "k", "load_s", "vk_s", "pk_s", "keygen_s", "prove_s", "verify_s",
        "proof_bytes",
    ];
    const COMMIT: &[&str] = &[
        "bind_prove_s",
        "open_prove_s",
        "total_prove_s",
        "total_proof_bytes",
    ];
    const TAIL: &[&str] = &["wall_s", "config", "status"];

    let mut wanted: Vec<&str> = HEAD.to_vec();
    if mode != Mode::Simplification {
        wanted.extend_from_slice(COMMIT);
    }
    wanted.extend_from_slice(TAIL);
    let cols: Vec<usize> = wanted
        .iter()
        .map(|w| {
            names
                .iter()
                .position(|n| n == w)
                .unwrap_or_else(|| panic!("`{}` is not a column of Row::header()", w))
        })
        .collect();

    let mut table: Vec<Vec<String>> = vec![cols.iter().map(|&c| names[c].to_string()).collect()];
    for (q, d) in jobs {
        let Some(line) = results.get(&format!("{}/{}", q, d)) else {
            continue;
        };
        let f: Vec<&str> = line.split(',').collect();
        table.push(
            cols.iter()
                .map(|&c| f.get(c).copied().unwrap_or("").to_string())
                .collect(),
        );
    }
    if table.len() < 2 {
        return;
    }

    let widths: Vec<usize> = (0..cols.len())
        .map(|i| table.iter().map(|r| r[i].len()).max().unwrap_or(0))
        .collect();
    println!("\nsummary ({} row(s), all timings in seconds):", table.len() - 1);
    for (r, row) in table.iter().enumerate() {
        let cells: Vec<String> = row
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                // identity and text columns left, numbers right
                if i < 2 || i + 2 >= cols.len() {
                    format!("{:<w$}", cell, w = widths[i])
                } else {
                    format!("{:>w$}", cell, w = widths[i])
                }
            })
            .collect();
        println!("{}", cells.join("  ").trim_end());
        if r == 0 {
            let rule: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
            println!("{}", rule.join("  "));
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", usage());
        return;
    }
    if args.is_empty() {
        eprintln!("{}", usage());
        std::process::exit(2);
    }

    let mode = match args[0].as_str() {
        // `baseline` is kept as a silent alias so the commands in older notes,
        // scripts and result filenames keep working.
        "simplification" | "baseline" => Mode::Simplification,
        "full" => Mode::Full,
        "commit" => Mode::Commit,
        other => {
                eprintln!(
                "unknown mode `{}` (expected `simplification`, `full` or `commit`)",
                other
            );
            std::process::exit(2);
        }
    };
    // The output file is optional and recognised by its `.csv` suffix; every
    // other argument after the mode is a query, optionally pinned to one
    // dataset with a `:dataset` suffix (`gq4:lastfm`), the same spelling
    // `dp_lane_bench` takes.  Without a suffix a graph query runs all three
    // networks.
    let mut out_path: Option<String> = None;
    let mut specs: Vec<(String, Option<String>)> = Vec::new();
    for a in &args[1..] {
        if a.ends_with(".csv") {
            if out_path.is_some() {
                eprintln!("at most one <out.csv> argument (got `{}` twice over)", a);
                std::process::exit(2);
            }
            out_path = Some(a.clone());
            continue;
        }
        let (q, ds) = match a.split_once(':') {
            Some((q, d)) => (q.to_string(), Some(d.to_string())),
            None => (a.clone(), None),
        };
        if !ALL_QUERIES.contains(&q.as_str()) {
            eprintln!(
                "unknown argument `{}` (expected a query {}, optionally `query:dataset`, \
                 or a path ending in .csv)",
                a,
                ALL_QUERIES.join(" ")
            );
            std::process::exit(2);
        }
        if let Some(d) = &ds {
            if q.starts_with("gq") {
                if !GRAPH_DATASETS.contains(&d.as_str()) {
                    eprintln!(
                        "unknown dataset `{}` in `{}` (expected one of {})",
                        d,
                        a,
                        GRAPH_DATASETS.join(" ")
                    );
                    std::process::exit(2);
                }
            } else if *d != tpch_label() {
                // The TPC-H queries have exactly one dataset per run, chosen by
                // VPJOIN_DATA and named by VPJOIN_LABEL, so a suffix naming
                // anything else is a mistake worth reporting rather than
                // dropping on the floor.
                eprintln!(
                    "`{}` has one dataset per run, currently `{}`: select the tables with \
                     VPJOIN_TABLES / VPJOIN_DATA and name them with VPJOIN_LABEL, not with \
                     a `:{}` suffix",
                    q,
                    tpch_label(),
                    d
                );
                std::process::exit(2);
            }
        }
        specs.push((q, ds));
    }
    if specs.is_empty() {
        specs = ALL_QUERIES.iter().map(|s| (s.to_string(), None)).collect();
    }

    // Privacy regime for the queries that materialize intermediates (Q5, GQ3,
    // GQ4). The others materialize nothing and are unaffected.
    let privacy = match std::env::var("VPJOIN_PRIVACY").as_deref().unwrap_or("dp") {
        "rjs" => Privacy::Rjs,
        "legacy" => Privacy::Legacy,
        "oblivious" => Privacy::Oblivious,
        "dp" => {
            let epsilon: f64 = std::env::var("VPJOIN_EPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.1);
            let delta: f64 = std::env::var("VPJOIN_DELTA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1e-5);
            Privacy::Dp { epsilon, delta }
        }
        other => {
            eprintln!(
                "unknown VPJOIN_PRIVACY `{}` (expected rjs, oblivious, legacy or dp)",
                other
            );
            std::process::exit(2);
        }
    };

    let mut resume = std::env::var("VPJOIN_RESUME").map(|v| v == "1").unwrap_or(false);
    if resume && out_path.is_none() {
        println!(
            "note: VPJOIN_RESUME=1 needs an <out.csv> argument to resume from; \
             continuing without resume."
        );
        resume = false;
    }
    let existing = match (&out_path, resume) {
        (Some(p), true) => load_completed(p),
        _ => HashMap::new(),
    };

    let all_jobs = plan_specs(&specs);
    let jobs: Vec<_> = all_jobs
        .iter()
        .filter(|(q, d)| !existing.contains_key(&format!("{}/{}", q, d)))
        .cloned()
        .collect();

    println!(
        "mode={}  profile={}  privacy={} (Q5, GQ3, GQ4 -- the queries that materialize bags)",
        args[0],
        build_profile(),
        privacy.label()
    );
    println!(
        "prover: REAL Halo2 pipeline (keygen_vk / keygen_pk / create_proof / verify_proof, \
         IPA over Pasta) -- MockProver is NOT used anywhere in this harness"
    );
    println!(
        "params: {}/proof/param{{k}} (every load is logged per row on stderr; proofs are \
         written to {}/proof/bench/ as auditable artifacts, independently of any .csv)",
        data_root().display(),
        data_root().display()
    );
    if cfg!(debug_assertions) {
        println!(
            "build profile: DEBUG ASSERTIONS ON. This is the `test` profile, which keeps the \
             host-side debug_assert!s; it is not a measurement profile -- do not mix its \
             timings with the ones below."
        );
    } else {
        println!(
            "build profile: RELEASE (optimized). `[profile.dev]` in Cargo.toml carries the \
             release settings, so `cargo run` is optimized with or without --release."
        );
    }
    if resume {
        println!(
            "resume: keeping {} completed row(s) from {}; running {} of {}",
            existing.len(),
            out_path.as_deref().unwrap_or(""),
            jobs.len(),
            all_jobs.len()
        );
    }
    // Resolve every degree up front and check its params file exists, so a
    // missing param never surfaces hours into a run.
    println!("planned degrees:");
    let mut degrees = Vec::new();
    let mut missing: Vec<u32> = Vec::new();
    for (q, d) in &jobs {
        let k = degree_for(q, d, privacy);
        let p = halo2_experiments::paths::param_file(k);
        let have = std::path::Path::new(&p).exists();
        println!(
            "    {:<4} {:<10} k={:<3} {}",
            q,
            d,
            k,
            if have { "params ok" } else { "PARAMS MISSING" }
        );
        if !have && !missing.contains(&k) {
            missing.push(k);
        }
        degrees.push(k);
    }
    if !missing.is_empty() {
        missing.sort();
        let list: Vec<String> = missing.iter().map(|k| k.to_string()).collect();
        eprintln!(
            "\nrefusing to start: missing params for k = {}.\nGenerate them first:\n  \
             cargo run --release --bin gen_params -- {}",
            list.join(", "),
            list.join(" ")
        );
        std::process::exit(2);
    }
    if std::env::var("VPJOIN_PLAN_ONLY").map(|v| v == "1").unwrap_or(false) {
        println!("\nVPJOIN_PLAN_ONLY=1 -- stopping before any proving.");
        return;
    }
    // Every completed row, keyed "query/dataset", printed as a table at the end
    // whether or not a CSV is being written.
    let mut results: HashMap<String, String> = existing.clone();

    if jobs.is_empty() {
        println!("nothing to do -- every requested row is already present and ok.");
        print_summary(&all_jobs, &results, mode);
        return;
    }

    // Rewrite the file with the preserved rows first, then append as we go, so
    // an interrupted resume never loses earlier measurements.
    let mut out: Option<std::fs::File> = match &out_path {
        Some(p) => {
            if let Some(dir) = std::path::Path::new(p).parent() {
                if !dir.as_os_str().is_empty() {
                    let _ = std::fs::create_dir_all(dir);
                }
            }
            let mut f = std::fs::File::create(p).expect("cannot create output csv");
            writeln!(f, "{}", Row::header()).unwrap();
            let mut kept: Vec<_> = existing.values().collect();
            kept.sort();
            for line in kept {
                writeln!(f, "{}", line).unwrap();
            }
            f.flush().unwrap();
            Some(f)
        }
        None => None,
    };

    let total = jobs.len();
    let mut failed: Vec<String> = Vec::new();
    for (i, (q, d)) in jobs.iter().enumerate() {
        println!("[{}/{}] {} on {} (k={}) ...", i + 1, total, q, d, degrees[i]);
        let row = run_one(q, d, mode, privacy);
        if row.status.starts_with("FAILED") {
            failed.push(format!("{} on {} -- {}", q, d, row.status));
        }
        println!(
            "        k={} load={:.2}s vk={:.2}s pk={:.2}s prove={:.2}s verify={:.3}s \
             proof={}B wall={:.2}s [{}]{}",
            row.k,
            row.load_s,
            row.vk_s,
            row.pk_s,
            row.prove_s,
            row.verify_s,
            row.proof_bytes,
            row.wall_s,
            row.profile,
            if mode == Mode::Full || mode == Mode::Commit {
                format!(
                    "  |  +commit: bind={:.2}s open={:.2}s  total_prove={:.2}s",
                    row.bind_prove_s, row.open_prove_s, row.total_prove_s
                )
            } else {
                String::new()
            }
        );
        let csv = row.to_csv();
        if let Some(f) = out.as_mut() {
            writeln!(f, "{}", csv).unwrap();
            f.flush().unwrap();
        }
        results.insert(format!("{}/{}", q, d), csv);
    }

    print_summary(&all_jobs, &results, mode);

    if let Some(p) = &out_path {
        println!("\nwrote {}", p);
    }
    if failed.is_empty() {
        println!("all {} row(s) ok", total);
    } else {
        eprintln!(
            "{}/{} rows FAILED (the measured rows are still in the summary above):",
            failed.len(),
            total
        );
        for x in &failed {
            eprintln!("  {}", x);
        }
        std::process::exit(1);
    }
}
