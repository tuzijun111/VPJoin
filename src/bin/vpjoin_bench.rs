//! Single-command benchmark driver for every VPJoin query (revision).
//!
//! Runs the 5 TPC-H queries and the 4 graph queries on each of the 3 SNAP
//! datasets (17 rows), in one of two modes, and writes one CSV.
//!
//!   baseline -- the pure query circuit, i.e. exactly what
//!               `cargo test ... sql::qN_obj::tests::test_1` proves.
//!   full     -- the same circuits plus the complete commitment layer:
//!               a published per-column Pedersen commitment to the dataset,
//!               the in-circuit check that the query's witness columns equal
//!               the committed data, and the column openings.
//!   commit   -- ONLY the commitment layer, in the corresponding query
//!               circuit's own domain (same k). The query proof is not re-run,
//!               so these numbers add to the matching `baseline` row. Much
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
//!   cargo run --release --bin vpjoin_bench -- <baseline|full|commit> <out.csv> [query ...]
//!
//!   queries:  q3 q5 q8 q9 q18 gq1 gq2 gq3 gq4     (default: all)
//!   env:      VPJOIN_DATA   root holding data/ graph_data/ proof/
//!                           (default: <crate>/src)
//!             VPJOIN_RESUME =1 to keep the rows already in <out.csv> and only
//!                           run the ones that are missing or previously
//!                           FAILED, appending results in place.
//!             VPJOIN_PRIVACY  how GQ3/GQ4 size their materialized bags:
//!                           dp (default) | rjs | legacy
//!             VPJOIN_EPS / VPJOIN_DELTA   total DP budget (default 0.1 / 1e-5)
//!             VPJOIN_DP_SEED  fixes the capacity-release randomness so a
//!                           re-run reproduces the same circuit sizes
//!             VPJOIN_PLAN_ONLY =1 to print the planned degrees and exit
//!                           without proving anything
//!
//! Drop `--release` to measure under the unoptimized profile, which is the
//! same profile `cargo test` uses.

use halo2_experiments::bench_queries::{
    build_profile, data_root, degree_for, plan, run_one, Mode, Privacy, Row, ALL_QUERIES,
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!(
            "usage: vpjoin_bench <baseline|full|commit> <out.csv> [query ...]\n\
             queries: {}",
            ALL_QUERIES.join(" ")
        );
        std::process::exit(2);
    }

    let mode = match args[0].as_str() {
        "baseline" => Mode::Baseline,
        "full" => Mode::Full,
        "commit" => Mode::Commit,
        other => {
            eprintln!("unknown mode `{}` (expected `baseline`, `full` or `commit`)", other);
            std::process::exit(2);
        }
    };
    let out_path = args[1].clone();

    let queries: Vec<String> = if args.len() > 2 {
        args[2..].to_vec()
    } else {
        ALL_QUERIES.iter().map(|s| s.to_string()).collect()
    };
    for q in &queries {
        if !ALL_QUERIES.contains(&q.as_str()) {
            eprintln!("unknown query `{}` (expected one of {})", q, ALL_QUERIES.join(" "));
            std::process::exit(2);
        }
    }

    // Privacy regime for the cyclic queries (GQ3/GQ4). The acyclic queries
    // materialize no intermediate and are unaffected.
    let privacy = match std::env::var("VPJOIN_PRIVACY").as_deref().unwrap_or("dp") {
        "rjs" => Privacy::Rjs,
        "legacy" => Privacy::Legacy,
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
            eprintln!("unknown VPJOIN_PRIVACY `{}` (expected rjs, legacy or dp)", other);
            std::process::exit(2);
        }
    };

    let resume = std::env::var("VPJOIN_RESUME").map(|v| v == "1").unwrap_or(false);
    let existing = if resume {
        load_completed(&out_path)
    } else {
        HashMap::new()
    };

    let all_jobs = plan(&queries);
    let jobs: Vec<_> = all_jobs
        .iter()
        .filter(|(q, d)| !existing.contains_key(&format!("{}/{}", q, d)))
        .cloned()
        .collect();

    println!(
        "mode={}  profile={}  privacy={} (cyclic queries GQ3/GQ4 only)",
        args[0],
        build_profile(),
        privacy.label()
    );
    println!(
        "prover: REAL Halo2 pipeline (keygen_vk / keygen_pk / create_proof / verify_proof, \
         IPA over Pasta) -- MockProver is NOT used anywhere in this harness"
    );
    println!(
        "params: {}/proof/param{{k}} (every load is logged per row; proofs are written to \
         {}/proof/bench/)",
        data_root().display(),
        data_root().display()
    );
    if cfg!(debug_assertions) {
        println!(
            "build profile: DEBUG (unoptimized, opt-level=0 -- including halo2 itself).\n\
             This is the SAME profile as `cargo test ... qN_obj::tests::test_1` without \
             --release."
        );
    } else {
        println!(
            "build profile: RELEASE (optimized). `cargo test` without --release builds DEBUG \
             and is several times slower -- do not mix the two in one comparison."
        );
    }
    if resume {
        println!(
            "resume: keeping {} completed row(s) from {}; running {} of {}",
            existing.len(),
            out_path,
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
    if jobs.is_empty() {
        println!("nothing to do -- every requested row is already present and ok.");
        return;
    }

    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        if !dir.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(dir);
        }
    }

    // Rewrite the file with the preserved rows first, then append as we go, so
    // an interrupted resume never loses earlier measurements.
    let mut f = std::fs::File::create(&out_path).expect("cannot create output csv");
    writeln!(f, "{}", Row::header()).unwrap();
    let mut kept: Vec<_> = existing.values().collect();
    kept.sort();
    for line in kept {
        writeln!(f, "{}", line).unwrap();
    }
    f.flush().unwrap();

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
        writeln!(f, "{}", row.to_csv()).unwrap();
        f.flush().unwrap();
    }

    println!("\nwrote {}", out_path);
    if failed.is_empty() {
        println!("all {} row(s) ok", total);
    } else {
        eprintln!("{}/{} rows FAILED (measured rows are still in the CSV):", failed.len(), total);
        for x in &failed {
            eprintln!("  {}", x);
        }
        std::process::exit(1);
    }
}
