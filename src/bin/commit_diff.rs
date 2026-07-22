//! Additional cost of the data commitment + the in-circuit witness check,
//! measured the way Appendix A describes it: the check lives INSIDE the query
//! circuit.
//!
//! For each query this proves the same witness in two variants at the same
//! degree, repeated VPJOIN_REPS times (default 3) with the order ALTERNATING:
//!
//!   base  -- the query circuit exactly as the submission runs it
//!   bound -- the SAME query circuit whose constraint system additionally
//!            carries the binding columns and Horner gates, so one proof
//!            establishes both the query result and that the witnessed input
//!            columns equal the published per-column commitments
//!
//! and reports the MEDIAN of the paired `bound - base` differences, plus the
//! one-off commitment publication and the
//! IPA openings the verifier checks against the circuit's exposed evaluations.
//! Nothing is written to disk; the table is printed.
//!
//! Usage:
//!   cargo commit-diff              # all 5 TPC-H + 4 graph queries x 3 datasets
//!   cargo commit-diff q3 q18       # a subset
//!   cargo commit-diff gq1 gq2      # graph queries expand over all 3 datasets
//!   cargo commit-diff gq2:wiki     # one graph query on one dataset
//!   cargo commit-diff reps=5 q18   # 5 repetitions instead of the default 3
//!   cargo commit-diff reps=1 gq3   # single quick pass (noisier)
//!
//! `reps=N` may appear anywhere in the argument list. It overrides the
//! VPJOIN_REPS environment variable; the default is 3.
//!
//! The binding cost is a small fraction of the query proof, so a single pair is
//! dominated by run-to-run variance; repeating and alternating the order also
//! removes the downward bias from always proving the base circuit first.
//!
//! Graph rows are proved in the query circuit's own domain, which for GQ3/GQ4
//! is set by their materialized bags (k up to 23) rather than by the 2-column
//! Edge input -- those rows are correspondingly expensive.

use halo2_experiments::bench_queries::{
    count_gq1, count_gq2, count_gq3, count_gq4, graph_pads, load_graph, tpch_inputs, Privacy,
    GRAPH_DATASETS,
};
use halo2_experiments::column_commit::{
    binding_challenge, commit_column_vectors, min_k_rows, open_column_vectors,
    verify_column_openings,
};
use halo2_experiments::inline_bind::{
    bind_instance, graph_paired, prove_graph_base, prove_plain, tpch_paired,
};
use halo2_experiments::paths;
use halo2_proofs::poly::{commitment::Params, ipa::commitment::ParamsIPA};
use halo2curves::pasta::vesta;
use rand::rngs::OsRng;
use std::time::Instant;

const K: u32 = 16;

struct RowOut {
    q: String,
    dataset: String,
    cols: usize,
    /// bound - base: the prover time added by the binding gates inside the
    /// query circuit.
    incirc_s: f64,
    commit_s: f64,
    open_s: f64,
    /// incirc + commitment publication + column openings.
    extra_s: f64,
}

fn row_out(
    q: &str,
    dataset: &str,
    cols: usize,
    incirc_s: f64,
    commit_s: f64,
    open_s: f64,
) -> RowOut {
    RowOut {
        q: q.to_string(),
        dataset: dataset.to_string(),
        cols,
        incirc_s,
        commit_s,
        open_s,
        extra_s: incirc_s + commit_s + open_s,
    }
}

/// Median of paired (bound - base) differences.
///
/// Paired rather than median(bound) - median(base): each pair is measured back
/// to back, so shared drift cancels within the pair.
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Summarise raw per-repetition `(base_s, bound_s)` timings.
///
/// Returns `(median_diff, min_diff, max_diff)`. Paired rather than
/// median(bound) - median(base): each pair is measured back to back, so shared
/// drift cancels within the pair.
fn summarise(runs: &[(f64, f64)]) -> (f64, f64, f64) {
    for (i, (b, d)) in runs.iter().enumerate() {
        eprintln!(
            "\n      rep {}: base {:>8.2}s  bound {:>8.2}s  diff {:>+8.2}s",
            i + 1,
            b,
            d,
            d - b
        );
    }
    let diffs: Vec<f64> = runs.iter().map(|(b, d)| d - b).collect();
    let lo = diffs.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = diffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (median(diffs), lo, hi)
}

fn print_header() {
    println!(
        "{:<5}{:<10}{:>6}{:>11}{:>11}{:>10}{:>11}",
        "query", "dataset", "cols", "in-circ(s)", "commit(s)", "open(s)", "extra(s)"
    );
    println!("{}", "-".repeat(64));
}

/// Printed as each row completes, so an interrupted run still yields data.
fn print_row(r: &RowOut) {
    use std::io::Write;
    println!(
        "{:<5}{:<10}{:>6}{:>11.2}{:>11.3}{:>10.2}{:>11.2}",
        r.q, r.dataset, r.cols, r.incirc_s, r.commit_s, r.open_s, r.extra_s
    );
    let _ = std::io::stdout().flush();
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();

    // `reps=N` may appear anywhere in the argument list; everything else is a
    // query selector.
    let mut reps_arg: Option<usize> = None;
    let mut args: Vec<String> = Vec::new();
    for a in raw {
        if let Some(v) = a.strip_prefix("reps=") {
            let n: usize = v
                .parse()
                .unwrap_or_else(|_| panic!("reps= expects a positive integer, got `{}`", v));
            assert!(n >= 1, "reps must be at least 1");
            reps_arg = Some(n);
        } else {
            args.push(a);
        }
    }
    const ALL: &[&str] = &["q3", "q5", "q8", "q9", "q18", "gq1", "gq2", "gq3", "gq4"];
    // "gq2" runs all three datasets; "gq2:wiki" runs just that one.
    let want: Vec<(String, Option<String>)> = if args.is_empty() {
        ALL.iter().map(|s| (s.to_string(), None)).collect()
    } else {
        args.iter()
            .map(|a| {
                let (q, ds) = match a.split_once(':') {
                    Some((q, d)) => (q.to_string(), Some(d.to_string())),
                    None => (a.clone(), None),
                };
                assert!(ALL.contains(&q.as_str()), "unknown query {} (expected one of {:?})", q, ALL);
                if let Some(d) = &ds {
                    assert!(
                        q.starts_with("gq"),
                        "{}: only graph queries have datasets",
                        a
                    );
                    assert!(
                        GRAPH_DATASETS.contains(&d.as_str()),
                        "unknown dataset {} (expected one of {:?})",
                        d,
                        GRAPH_DATASETS
                    );
                }
                (q, ds)
            })
            .collect()
    };

    if cfg!(debug_assertions) {
        eprintln!("WARNING: unoptimized build -- use `cargo commit-diff`, which builds release.");
    }

    // Privacy regime for the cyclic queries, same knobs as vpjoin_bench.
    let privacy = match std::env::var("VPJOIN_PRIVACY").as_deref().unwrap_or("dp") {
        "rjs" => Privacy::Rjs,
        "legacy" => Privacy::Legacy,
        _ => Privacy::Dp {
            epsilon: std::env::var("VPJOIN_EPS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.1),
            delta: std::env::var("VPJOIN_DELTA").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-5),
        },
    };

    let load_params = |k: u32| {
        let p = paths::param_file(k);
        let mut fd = std::fs::File::open(&p).unwrap_or_else(|e| {
            panic!("open {}: {} -- generate it with `cargo run --release --bin gen_params -- {}`", p, e, k)
        });
        ParamsIPA::<vesta::Affine>::read(&mut fd).expect("read params")
    };

    println!(
        "In-circuit binding cost, {} build. The witness-equality check is INLINED\n\
         into the query circuit: one proof covers both the query result and the\n\
         binding (Appendix A). Each row is proved twice at the same degree.\n",
        if cfg!(debug_assertions) { "debug" } else { "release" }
    );

    // Precedence: reps=N on the command line, else VPJOIN_REPS, else 3.
    let reps: usize = reps_arg
        .or_else(|| {
            std::env::var("VPJOIN_REPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|n| *n >= 1)
        })
        .unwrap_or(3);
    println!(
        "reps={} per row. Both proving keys are built once outside the timed\n\
         region, one proof of each circuit is run and DISCARDED as warm-up, then\n\
         base/bound alternate; the median of the paired differences is reported.\n\
         Set VPJOIN_REPS=1 for a single quick pass.\n",
        reps
    );

    let mut rows: Vec<RowOut> = Vec::new();
    print_header();

    for (q, only_ds) in &want {
        if q.starts_with("gq") {
            for ds in GRAPH_DATASETS
                .iter()
                .filter(|d| only_ds.as_deref().map_or(true, |o| o == **d))
            {
                eprint!("  {} on {} ... ", q, ds);
                let edges = load_graph(ds);
                let pads = if matches!(q.as_str(), "gq3" | "gq4") {
                    graph_pads(q, ds, &edges, privacy)
                } else {
                    (0, 0)
                };
                let k = match q.as_str() {
                    "gq1" | "gq2" => 17,
                    _ => halo2_experiments::bench_queries::ceil_log2(
                        halo2_experiments::bench_queries::graph_rows(&edges, q, pads.0, pads.1) + 64,
                    ),
                };
                let params = load_params(k);
                let cnt = match q.as_str() {
                    "gq1" => count_gq1(&edges),
                    "gq2" => count_gq2(&edges),
                    "gq3" => count_gq3(&edges),
                    _ => count_gq4(&edges),
                };
                let cols: Vec<Vec<u64>> = vec![
                    edges.iter().map(|e| e.src).collect(),
                    edges.iter().map(|e| e.dst).collect(),
                ];

                // Commit and open in a domain sized by the DATA (2 columns of
                // |E| edges), not by the query circuit's degree. For GQ3/GQ4
                // the latter is inflated to 2^22-2^23 by their materialized
                // bags, and charging the commitment layer for that would
                // measure the intermediate blow-up rather than the cost of
                // binding the input. Sound because IPA generators are
                // position-indexed, so the degree-ck SRS is exactly the prefix
                // of the degree-k one and the commitment point is unchanged.
                let ck = min_k_rows(cols.iter().map(|c| c.len()).max().unwrap_or(0));
                assert!(ck <= k, "input needs k={} but query circuit is k={}", ck, k);
                let cparams = load_params(ck);
                let t = Instant::now();
                let commitments = commit_column_vectors(&cparams, &cols, ck, OsRng);
                let commit_s = t.elapsed().as_secs_f64();
                let _published = commitments.published_bytes();

                // One canonical base proof establishes the challenge, bound to
                // it as the protocol requires. Its timing is NOT reused: it is
                // the first proof in the process and would be warm-up biased.
                let first = prove_graph_base(&params, q, &edges, pads, cnt);
                let x = binding_challenge(&commitments, &first.proof);
                let bind_pub = bind_instance(&cols, x);
                drop(first);

                let runs = graph_paired(&params, q, &edges, pads, cnt, x, &bind_pub, reps);
                let (incirc, lo, hi) = summarise(&runs);

                let t = Instant::now();
                let openings = open_column_vectors(&cparams, &commitments, &cols, x, OsRng);
                let open_s = t.elapsed().as_secs_f64();
                let _open_bytes: usize = openings.iter().map(|o| o.len()).sum();

                let t = Instant::now();
                assert!(
                    verify_column_openings(&cparams, &commitments.points, x, &bind_pub[1..], &openings),
                    "column openings failed to verify"
                );
                let _open_verify_s = t.elapsed().as_secs_f64();

                eprintln!(
                    "ok (query k={}, commit k={}, reps={}, in-circ {:.2}..{:.2}s)",
                    k, ck, reps, lo, hi
                );
                let r = row_out(q, ds, cols.len(), incirc, commit_s, open_s);
                print_row(&r);
                rows.push(r);
            }
            continue;
        }

        eprint!("  {} ... ", q);
        let params = load_params(K);
        let input = tpch_inputs(q);
        input.require_loaded();
        let cols = input.columns();

        let t = Instant::now();
        let commitments = commit_column_vectors(&params, &cols, K, OsRng);
        let commit_s = t.elapsed().as_secs_f64();
        let _published = commitments.published_bytes();

        let first = prove_plain(&params, &input);
        let x = binding_challenge(&commitments, &first.proof);
        let bind_pub = bind_instance(&cols, x);
        drop(first);

        let runs = tpch_paired(&params, &input, x, &bind_pub, reps);
        let (incirc, lo, hi) = summarise(&runs);

        let t = Instant::now();
        let openings = open_column_vectors(&params, &commitments, &cols, x, OsRng);
        let open_s = t.elapsed().as_secs_f64();
        let _open_bytes: usize = openings.iter().map(|o| o.len()).sum();

        let t = Instant::now();
        assert!(
            verify_column_openings(&params, &commitments.points, x, &bind_pub[1..], &openings),
            "column openings failed to verify"
        );
        let _open_verify_s = t.elapsed().as_secs_f64();

        eprintln!("ok (reps={}, in-circ {:.2}..{:.2}s)", reps, lo, hi);
        let r = row_out(q, "tpch-60K", cols.len(), incirc, commit_s, open_s);
        print_row(&r);
        rows.push(r);
    }

    let _ = rows;
    println!(
        "\nin-circ = prover time added by the binding gates inside the query circuit\n\
         extra   = in-circ + commitment publication + column openings\n\
         every proof above was generated AND verified."
    );
}
