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
//! and reports an ORDER-BALANCED estimate of the paired `bound - base`
//! differences, plus the one-off commitment publication and the IPA openings
//! the verifier checks against the circuit's exposed evaluations. Nothing is
//! written to disk; the table is printed.
//!
//! Usage:
//!   cargo commit-diff              # all 5 TPC-H + 4 graph queries x 3 datasets
//!   cargo commit-diff reps=3 q3 q5 q8 q9 q18 gq1 gq2 gq3 gq4   # all nine, one run
//!   cargo commit-diff q3 q18       # a subset
//!   cargo commit-diff gq1 gq2      # graph queries expand over all 3 datasets
//!   cargo commit-diff gq2:wiki     # one graph query on one dataset
//!   cargo commit-diff reps=5 q18   # 5 repetitions instead of the default 3
//!   cargo commit-diff reps=1 gq3   # single quick pass (noisier)
//!
//! VPJOIN_PRIVACY (dp | rjs | legacy) sizes the materialized bags of Q5, GQ3
//! and GQ4; every other query materializes nothing and ignores it. Left unset,
//! the cyclic three default to `rjs` HERE, so one invocation covers all nine.
//! That default is forced, not a preference: this binary fixes TPC-H at k=16
//! and Q5's dp capacities need k>=17 below eps=1, so dp cannot be measured in
//! this harness at all. Setting VPJOIN_PRIVACY applies it to every query, and
//! the regime in force is printed at the top of each run. Quote the result for
//! the regime the paper reports the query in, since `in-circ` scales with 2^k.
//!
//! `reps=N` may appear anywhere in the argument list. It overrides the
//! VPJOIN_REPS environment variable; the default is 3.
//!
//! MEASUREMENT FLOOR, computed per row from the run itself -- no second command
//! is needed. `in-circ` is a difference of two large proof times, and which of
//! the two multi-GB proving keys is built first is worth seconds, several times
//! the effect being measured. `paired_runs` therefore alternates the order, and
//! `summarise` averages within each ordering before averaging the two groups,
//! so both orderings carry equal weight whatever their counts.
//!
//! `floor` is the LARGER of two error estimates: half the gap between the two
//! orderings, and half the spread across repetitions. Both are needed. The
//! ordering gap catches a bias that differs between the key-build orders; the
//! spread catches drift that hits both orders alike, which the gap cannot see
//! at all. gq1:facebook was measured twice, once at +1.42s and once at -4.18s,
//! with the orderings agreeing to 0.14s and the repetitions spanning 5.3s: an
//! ordering-only floor called the second one resolved.
//!
//! A row is reported as measured only if in-circ is POSITIVE and exceeds that
//! floor. The sign is a hard check, not a heuristic: adding constraints cannot
//! make a proof faster, so a negative estimate is a row dominated by something
//! other than the effect, whatever its floor says. Otherwise the row shows
//! `<floor`, and its `extra` omits in-circ rather than adding -- or, worse,
//! subtracting -- a value that is not separable from zero.
//!
//! This matters most where the effect is smallest: GQ1 and GQ2 bind the same
//! two Edge columns at the same degree, so their true `in-circ` is the same,
//! and any table showing them with opposite signs is reporting bias.
//!
//! `VPJOIN_SELFTEST=1` still exists and proves the BASE circuit against ITSELF
//! (true difference zero) as an independent check on the floor. Other knobs:
//! VPJOIN_BURNIN (discarded measurements before timing, default 2),
//! VPJOIN_SWAP_KEYGEN=1 (diagnostic: reverses key-build order), VPJOIN_VERBOSE=1
//! (print each repetition's raw base/bound time and which circuit ran first).
//!
//! The binding cost is a small fraction of the query proof, so a single pair is
//! dominated by run-to-run variance; repeating and alternating the order also
//! removes the downward bias from always proving the base circuit first.
//!
//! Graph rows are proved in the query circuit's own domain, which for GQ3/GQ4
//! is set by their materialized bags (k up to 23) rather than by the 2-column
//! Edge input -- those rows are correspondingly expensive.

use halo2_experiments::bench_queries::{
    column_indices, count_gq1, count_gq2, count_gq3, count_gq4, graph_pads, load_graph,
    published_db, tpch_label,
    tpch_database_columns, tpch_inputs, Privacy,
    GRAPH_DATASETS,
};
use halo2_experiments::column_commit::{
    binding_challenge_db, min_k_rows, open_column_vectors,
    verify_column_openings,
};
use halo2_experiments::inline_bind::{
    advice_overhead_ratio, bind_instance, graph_paired, graph_selftest, prove_graph_base,
    prove_plain, tpch_paired, tpch_selftest,
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
    /// Half the ordering-bias gap; `None` when only one ordering ran.
    floor_s: Option<f64>,
    /// `incirc_s` exceeds the bias that could have produced it.
    resolved: bool,
    commit_s: f64,
    open_s: f64,
    /// incirc + commitment publication + column openings.
    extra_s: f64,
}

fn row_out(
    q: &str,
    dataset: &str,
    cols: usize,
    est: &Estimate,
    commit_s: f64,
    open_s: f64,
) -> RowOut {
    RowOut {
        q: q.to_string(),
        dataset: dataset.to_string(),
        cols,
        incirc_s: est.value,
        floor_s: est.floor,
        resolved: est.resolved(),
        commit_s,
        open_s,
        extra_s: est.value + commit_s + open_s,
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

/// An order-balanced estimate of `bound - base`, with the floor below which it
/// is not distinguishable from the harness's own bias.
struct Estimate {
    /// Order-balanced (bound - base), in seconds.
    value: f64,
    /// Half the gap between the two orderings' means: the systematic error this
    /// harness cannot see past on this row. `None` when only one ordering was
    /// measured (`reps=1`), where no floor can be derived from the data.
    floor: Option<f64>,
    lo: f64,
    hi: f64,
}

impl Estimate {
    /// How the row should be read: a number when the effect clears its floor,
    /// `<floor` when it does not.
    fn describe(&self) -> String {
        if self.resolved() {
            format!("{:.2}s", self.value)
        } else if self.value < 0.0 {
            // Reported separately from an ordinary unresolved row: a negative
            // here is not a small effect lost in noise, it is a measurement
            // known to be invalid, and saying so is more useful than `<floor`.
            format!("<{:.2}s (unresolved; measured {:.2}s, impossible)", self.floor_or_nan(), self.value)
        } else {
            match self.floor {
                Some(f) => format!("<{:.2}s (unresolved)", f),
                None => format!("{:.2}s (no floor; reps=1)", self.value),
            }
        }
    }

    fn floor_or_nan(&self) -> f64 {
        self.floor.unwrap_or(f64::NAN)
    }

    /// Is this a usable measurement of the binding cost?
    ///
    /// Two independent conditions, and BOTH must hold.
    ///
    /// First, the sign. Adding constraints to a circuit cannot make its proof
    /// faster, so a negative estimate is not a small effect measured with the
    /// wrong sign -- it is proof that this row is dominated by something other
    /// than the effect. No floor comparison can rescue it, because the floor
    /// only bounds the error sources it was derived from.
    ///
    /// Second, the magnitude, against a floor that combines the two error
    /// sources this run can see: the gap between the key-build orderings, and
    /// the spread across repetitions. Taking the LARGER of the two matters --
    /// gq1:facebook was measured with the orderings agreeing to 0.14s while
    /// its repetitions spanned 5.3s, so an ordering-only floor certified a
    /// physically impossible -4.18s as resolved.
    fn resolved(&self) -> bool {
        if self.value <= 0.0 {
            return false;
        }
        match self.floor {
            Some(f) => self.value > f,
            None => false,
        }
    }
}

/// Summarise raw per-repetition `(base_s, bound_s)` timings.
///
/// `paired_runs` alternates which circuit is built first, because whichever
/// proving key is allocated second gets better page placement -- worth seconds
/// on a multi-GB key, several times the effect being measured. Alternation only
/// cancels that if the two orderings carry EQUAL WEIGHT, and neither a mean nor
/// a median over all repetitions gives them equal weight when their counts
/// differ. An odd `reps` guarantees they differ: `reps=3` runs base-first,
/// bound-first, base-first, so a median over the three lands on a base-first
/// sample and reports the bias essentially undiluted. That is what produces a
/// negative `in-circ` for a circuit to which gates were ADDED.
///
/// So: average within each ordering, then average the two group means, giving
/// each ordering equal weight whatever the counts. The gap between those means
/// is the ordering bias itself; half of it is the floor reported alongside.
fn summarise(runs: &[(f64, f64)]) -> Estimate {
    // Raw per-repetition base/bound times are diagnostic only (they were added
    // to track down a keygen-ordering bias); opt in with VPJOIN_VERBOSE=1.
    if std::env::var("VPJOIN_VERBOSE").map(|v| v == "1").unwrap_or(false) {
        for (i, (b, d)) in runs.iter().enumerate() {
            eprintln!(
                "\n      rep {}: base {:>8.2}s  bound {:>8.2}s  diff {:>+8.2}s  ({} first)",
                i + 1,
                b,
                d,
                d - b,
                if i % 2 == 0 { "base" } else { "bound" }
            );
        }
    }
    let diffs: Vec<f64> = runs.iter().map(|(b, d)| d - b).collect();
    let lo = diffs.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = diffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // `paired_runs` builds the base circuit first on even repetitions and the
    // bound circuit first on odd ones, so the index parity IS the ordering.
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let base_first: Vec<f64> = diffs.iter().step_by(2).cloned().collect();
    let bound_first: Vec<f64> = diffs.iter().skip(1).step_by(2).cloned().collect();

    let (value, floor) = if base_first.is_empty() || bound_first.is_empty() {
        // Only one ordering was measured, so the bias is entirely inside the
        // estimate and nothing in the data bounds it. Fall back to the median
        // and report no floor rather than a floor of zero, which would claim a
        // precision this run cannot support.
        (median(diffs.clone()), None)
    } else {
        let (a, b) = (mean(&base_first), mean(&bound_first));
        // Two error sources, and the floor is the larger. The ordering gap
        // catches a bias that differs between the two key-build orders; the
        // repetition spread catches drift that hits both orders alike, which
        // an ordering gap cannot see at all and which is what actually
        // dominates the graph rows.
        let ordering_gap = 0.5 * (a - b).abs();
        let spread = 0.5 * (hi - lo);
        (0.5 * (a + b), Some(ordering_gap.max(spread)))
    };
    Estimate {
        value,
        floor,
        lo,
        hi,
    }
}


/// Ambient machine load, read before a row starts so a polluted measurement is
/// attributable. On this shared 256-core server a concurrent prover inflates
/// individual ~200s proofs by tens of seconds; the pairing cannot cancel that
/// at k>=18, where the two halves of a pair are separated by minutes of
/// untimed keygen and warm-up. A 1-minute load far above the idle baseline
/// (~15-20 here) at ROW START means the in-circ column of that row is suspect
/// no matter what the floor says.
fn ambient_load() -> String {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    match std::fs::read_to_string("/proc/loadavg") {
        Ok(l) => {
            let f: Vec<&str> = l.split_whitespace().collect();
            format!("[load {} (1m) {} (15m) / {} cores] ", f[0], f.get(2).unwrap_or(&"?"), cores)
        }
        Err(_) => String::new(),
    }
}


/// For a row whose in-circ could not be resolved by time-differencing, print
/// the DERIVED estimate: the structural advice-overhead ratio times the mean
/// measured base proof time. Labelled as derived so it is never mistaken for a
/// measurement; the k=13 anchor puts its accuracy at about a factor of two.
fn print_derived(q: &str, runs: &[(f64, f64)], resolved: bool) {
    if resolved || runs.is_empty() {
        return;
    }
    let mean_base = runs.iter().map(|(b, _)| b).sum::<f64>() / runs.len() as f64;
    let ratio = advice_overhead_ratio(q);
    eprintln!(
        "      derived in-circ ~ +{:.1}% x base {:.1}s = ~{:.1}s (structural advice ratio, \
NOT a measurement; below this row's differencing noise floor)",
        100.0 * ratio,
        mean_base,
        ratio * mean_base
    );
}

fn print_header() {
    println!(
        "{:<5}{:<10}{:>6}{:>11}{:>9}{:>11}{:>10}{:>11}",
        "query", "dataset", "cols", "in-circ(s)", "floor(s)", "commit(s)", "open(s)", "extra(s)"
    );
    println!("{}", "-".repeat(73));
}

/// Printed as each row completes, so an interrupted run still yields data.
fn print_row(r: &RowOut) {
    use std::io::Write;
    // An in-circ at or below its floor is not separable from the ordering bias
    // that could have produced it, so it is shown as `<floor` rather than as a
    // number -- and a NEGATIVE such value is that bias, not a speed-up from
    // adding gates. `extra` then omits it instead of subtracting it away.
    let incirc = if r.resolved {
        format!("{:>11.2}", r.incirc_s)
    } else {
        format!("{:>11}", format!("<{:.2}", r.floor_s.unwrap_or(f64::NAN)))
    };
    let extra = if r.resolved {
        r.extra_s
    } else {
        r.commit_s + r.open_s
    };
    println!(
        "{:<5}{:<10}{:>6}{}{:>9.2}{:>11.3}{:>10.2}{:>11.2}",
        r.q,
        r.dataset,
        r.cols,
        incirc,
        r.floor_s.unwrap_or(f64::NAN),
        r.commit_s,
        r.open_s,
        extra
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

    // Privacy regime, same knobs as vpjoin_bench. Only Q5, GQ3 and GQ4 read it:
    // they materialize intermediates whose capacity it sets. The other six
    // materialize nothing and are identical under every regime.
    //
    // When VPJOIN_PRIVACY is unset the three cyclic queries default to `rjs`
    // rather than `dp`, so ONE invocation can cover all nine. That is not a
    // convenience: this binary fixes TPC-H at k=16, and Q5's dp capacities need
    // k>=17 below eps=1, so dp is not measurable here at all -- which is why the
    // cyclic queries previously needed a separate `VPJOIN_PRIVACY=rjs` run. An
    // explicit VPJOIN_PRIVACY still applies to every query, unchanged.
    let explicit = std::env::var("VPJOIN_PRIVACY").ok();
    let parse_regime = |v: &str| match v {
        "rjs" => Privacy::Rjs,
        "legacy" => Privacy::Legacy,
        _ => Privacy::Dp {
            epsilon: std::env::var("VPJOIN_EPS").ok().and_then(|p| p.parse().ok()).unwrap_or(0.1),
            delta: std::env::var("VPJOIN_DELTA").ok().and_then(|p| p.parse().ok()).unwrap_or(1e-5),
        },
    };
    // Takes the query so the per-query intent is explicit at every call site,
    // even though the resolved regime is currently uniform: the six acyclic
    // queries ignore it, and the three cyclic ones all default to rjs here.
    let privacy_for = |_q: &str| -> Privacy {
        match &explicit {
            Some(v) => parse_regime(v),
            None => Privacy::Rjs,
        }
    };
    if explicit.is_none() {
        println!(
            "regime: rjs (default here. Q5/GQ3/GQ4 materialize bags; dp needs k>=17, which \
this binary's fixed k=16 cannot hold. Set VPJOIN_PRIVACY to override.)"
        );
    } else {
        println!("regime: {} (VPJOIN_PRIVACY)", explicit.as_deref().unwrap_or(""));
    }

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

    // Control mode: prove the BASE circuit against itself. Reports harness
    // bias, which should be ~0.
    let selftest = std::env::var("VPJOIN_SELFTEST").map(|v| v == "1").unwrap_or(false);

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
         base/bound alternate; the order-balanced difference is reported, with the\n\
         per-row bias floor beneath which it is not resolved.\n\
         Set VPJOIN_REPS=1 for a single quick pass.\n",
        reps
    );

    if selftest {
        println!(
            "*** VPJOIN_SELFTEST=1: proving the BASE circuit against ITSELF.\n\
             *** in-circ should be ~0; anything else is harness bias.\n"
        );
    }

    let mut rows: Vec<RowOut> = Vec::new();
    print_header();

    for (q, only_ds) in &want {
        if q.starts_with("gq") {
            for ds in GRAPH_DATASETS
                .iter()
                .filter(|d| only_ds.as_deref().map_or(true, |o| o == **d))
            {
                eprint!("  {} on {} ... {}", q, ds, ambient_load());
                let edges = load_graph(ds);
                let pads = if matches!(q.as_str(), "gq3" | "gq4") {
                    graph_pads(q, ds, &edges, privacy_for(q))
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
                // Setup: Commit(D) over the whole graph database, which IS its
                // two Edge columns. Published ONCE per dataset and cached, so
                // every graph query in this run opens the same two points under
                // the same blinders; publishing per query would leave each proof
                // bound only to a commitment made for it.
                let published = published_db(ds, &cols, ck);
                let db = &published.0;
                let commit_s = published.1;
                let idx: Vec<usize> = (0..cols.len()).collect();
                let commitments = db.view(&idx);
                let _published_bytes = db.published_bytes();

                // One canonical base proof establishes the challenge, bound to
                // it as the protocol requires. Its timing is NOT reused: it is
                // the first proof in the process and would be warm-up biased.
                let first = prove_graph_base(&params, q, &edges, pads, cnt);
                let x = binding_challenge_db(&db, &idx, &first.proof);
                let bind_pub = bind_instance(&cols, x);
                drop(first);

                let runs = if selftest {
                    // Control: base vs base. True difference is zero, so any
                    // non-zero result here is harness bias.
                    graph_selftest(&params, q, &edges, pads, cnt, reps)
                } else {
                    graph_paired(&params, q, &edges, pads, cnt, x, &bind_pub, reps)
                };
                let est = summarise(&runs);

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
                    "ok (query k={}, commit k={}, reps={}, in-circ {}, raw spread {:.2}..{:.2}s)",
                    k, ck, reps, est.describe(), est.lo, est.hi
                );
                print_derived(q, &runs, est.resolved());
                let r = row_out(q, ds, cols.len(), &est, commit_s, open_s);
                print_row(&r);
                rows.push(r);
            }
            continue;
        }

        eprint!("  {} ... {}", q, ambient_load());
        let params = load_params(K);
        let input = tpch_inputs(q, privacy_for(q));
        input.require_loaded();
        let cols = input.columns();

        // Setup: Commit(D) over every column of the TPC-H database the workload
        // reads, published ONCE for the run and cached, with this query opening
        // the subset its circuit witnesses. Two queries sharing a column open
        // the same published point under the same blinder, which is what binds
        // them to one database.
        let db_cols = tpch_database_columns(privacy_for(q));
        let published = published_db(&tpch_label(), &db_cols, K);
        let db = &published.0;
        let commit_s = published.1;
        let idx = column_indices(&db_cols, &cols);
        let commitments = db.view(&idx);
        let _published_bytes = db.published_bytes();

        let first = prove_plain(&params, &input);
        let x = binding_challenge_db(&db, &idx, &first.proof);
        let bind_pub = bind_instance(&cols, x);
        drop(first);

        let runs = if selftest {
            // Control: base vs base. True difference is zero.
            tpch_selftest(&params, &input, reps)
        } else {
            tpch_paired(&params, &input, x, &bind_pub, reps)
        };
        let est = summarise(&runs);

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

        eprintln!(
            "ok (reps={}, in-circ {}, raw spread {:.2}..{:.2}s)",
            reps,
            est.describe(),
            est.lo,
            est.hi
        );
        print_derived(q, &runs, est.resolved());
        let r = row_out(q, "tpch-60K", cols.len(), &est, commit_s, open_s);
        print_row(&r);
        rows.push(r);
    }

    let _ = rows;
    println!(
        "\nin-circ = prover time added by the binding gates inside the query circuit,\n\
                   averaged within each key-build order and then across the two, so\n\
                   the ordering bias cancels whatever the repetition count\n\
         floor   = half the gap between those two orders: the bias this run cannot\n\
                   see past. `<x` means in-circ did not exceed it, so the effect is\n\
                   not separable from the bias and its sign carries no meaning\n\
         extra   = commitment publication + column openings, plus in-circ when it is\n\
                   resolved; an unresolved in-circ is omitted rather than added\n\
         every proof above was generated AND verified."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reported failure. `reps=3` alternates base-first, bound-first,
    /// base-first, so two of the three pairs carry the ordering bias in the
    /// same direction. A median picks a middle sample and therefore reports
    /// that bias almost undiluted -- which is how adding gates to a circuit
    /// came out as a 4-second SPEED-UP on gq2:facebook.
    #[test]
    fn odd_reps_defeat_a_median_but_not_the_group_estimate() {
        // true effect +1.0s, ordering bias +/-5.0s depending on which key is
        // built first: base-first pairs read -4.0, bound-first reads +6.0.
        let runs = vec![(60.0, 56.0), (60.0, 66.0), (60.0, 56.0)];
        let diffs: Vec<f64> = runs.iter().map(|(b, d)| d - b).collect();
        assert_eq!(median(diffs), -4.0, "the median lands on the biased sample");

        let est = summarise(&runs);
        assert!(
            (est.value - 1.0).abs() < 1e-9,
            "group-balanced estimate must recover the true effect, got {}",
            est.value
        );
        assert!((est.floor.unwrap() - 5.0).abs() < 1e-9, "floor is the bias");
        assert!(!est.resolved(), "a 1s effect under a 5s bias is not resolved");
    }


    /// gq1:facebook, run 2. All three repetitions negative, and the two
    /// key-build orderings agreeing to 0.14s -- so an ordering-only floor
    /// certified a physically impossible -4.18s as a resolved measurement and
    /// carried it into `extra`, which went negative. Adding constraints cannot
    /// speed a proof up, so the sign alone disqualifies the row.
    #[test]
    fn a_negative_estimate_is_never_resolved() {
        let runs = vec![(60.0, 53.05), (60.0, 58.33), (60.0, 55.10)];
        let est = summarise(&runs);
        assert!(est.value < 0.0, "reproduces the negative, got {}", est.value);
        assert!(!est.resolved(), "a negative in-circ must never be reported as measured");
        assert!(est.describe().contains("impossible"));
    }

    /// The same row's other failure: the orderings agreed closely while the
    /// repetitions spanned seconds, so the floor has to come from the spread.
    #[test]
    fn the_floor_tracks_the_spread_when_orderings_agree() {
        // orderings agree (both means ~ +1.0) but repetitions span 8s
        let runs = vec![(60.0, 65.0), (60.0, 61.0), (60.0, 57.0)];
        let est = summarise(&runs);
        let f = est.floor.unwrap();
        assert!(f >= 4.0, "floor must reflect the 8s spread, got {}", f);
        assert!(!est.resolved(), "a ~1s effect under an 8s spread is not resolved");
    }

    /// gq1:wiki, run 1: a small positive estimate buried in a 16s spread was
    /// previously printed as a measured 1.48s.
    #[test]
    fn a_small_effect_in_a_wide_spread_is_unresolved() {
        let runs = vec![(60.0, 58.01), (60.0, 74.07), (60.0, 61.48)];
        let est = summarise(&runs);
        assert!(!est.resolved(), "1.5s inside a 16s spread cannot be resolved");
    }

    /// An effect well clear of the bias is reported as measured.
    #[test]
    fn a_large_effect_is_resolved() {
        let runs = vec![(60.0, 80.0), (60.0, 82.0), (60.0, 80.0)];
        let est = summarise(&runs);
        assert!(est.resolved());
        assert!(est.value > 20.0 && est.value < 21.5, "got {}", est.value);
    }

    /// With one ordering there is nothing in the data to bound the bias, so no
    /// floor is claimed and the row is never reported as resolved.
    #[test]
    fn a_single_ordering_claims_no_floor() {
        let est = summarise(&[(60.0, 56.0)]);
        assert!(est.floor.is_none());
        assert!(!est.resolved(), "reps=1 cannot resolve anything");
    }
}
