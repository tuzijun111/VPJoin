//! Anchor + extrapolation harness for the PoneglyphDB-style graph baselines
//! (estimation methodology of Section 8.1).
//!
//! For each (query, dataset) it runs the binary-join-chain baseline circuit
//! with capacities set to the TRUE intermediate sizes -- the measured anchor
//! (N_0, G_{C,0}, T_0) -- then reports the fully padded worst-case circuit
//! size (N, G_C, from the public-size bound |P_t| <= |E|^t) and the
//! extrapolated proving time T_est = T_0 * G_C / G_{C,0}.
//!
//! Two things make the anchor sound:
//!
//!   * The worst case is ALWAYS derived from the full dataset, even when the
//!     anchor itself runs on a subsample.  The anchor's only job is to
//!     measure seconds per domain row for this circuit shape; the padded
//!     circuit it is scaled to is the one over the real graph.
//!   * Anchor and padded circuit use the same row-count formula
//!     (`pone_baseline::circuit_rows`) and the same column count -- the
//!     configure step always allocates all MAX_LEVELS-1 level groups -- so
//!     T_0 / G_{C,0} really is a per-domain-row rate for an unchanged shape.
//!
//! Subsampling is required: |P_3| is 79M rows on facebook and 202M on wiki,
//! so the unpadded execution does not fit at full scale.  The anchor takes a
//! prefix of the edge list large enough to fill a 2^PONE_K0 domain.
//! Intermediate size grows superlinearly in the edge count, so a fraction of
//! the edges already fills the domain: at K0=17, 4% of facebook, 4% of wiki,
//! but 55% of lastfm.  (On facebook and wiki the prefix is also far denser in
//! paths than the whole graph, so very few edges suffice; on lastfm it is
//! path-sparser and needs about 22% more edges than a uniform subsample
//! would.)  This changes which subsample is used, not the measured rate.
//!
//! Datasets and edge parsing match `bench_queries::load_graph` exactly, so
//! the baseline sees the same edge multiset as VPJoin: no symmetrization, and
//! (verified) no duplicate edges in any of the three files.
//!
//! Usage:
//!   cargo run --bin pone_graph_bench -- <out.csv> [query ...] [dataset ...]
//!
//!   queries:   gq1 gq2 gq3 gq4          (default: all)
//!   datasets:  lastfm facebook wiki     (default: all)
//!   env:       PONE_K0     target anchor domain exponent (default 17)
//!              PONE_EDGES  force the anchor subsample size, skipping the
//!                          automatic sizing
//!              PONE_PLAN_ONLY=1  print the plan and exit without proving
//!              VPJOIN_DATA root holding graph_data/ and proof/
//!
//! Add `--release` to measure under the optimized profile.  Without it this
//! builds DEBUG, the same profile as `cargo test ... qN_obj::tests::test_1`,
//! which is the profile the paper's other proving times use -- do not mix the
//! two in one comparison.

use std::time::Instant;

use halo2_experiments::bench_queries::params_for;
use halo2_experiments::graph_sql::pone_baseline::{
    circuit_rows, enumerate_paths, level_sizes, PoneBaselineCircuit,
};
use halo2_proofs::{
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof},
    poly::{
        ipa::{commitment::IPACommitmentScheme, multiopen::ProverIPA, strategy::SingleStrategy},
        VerificationStrategy,
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use halo2curves::pasta::{EqAffine, Fp};
use rand::rngs::OsRng;
use std::io::Write;

const QUERIES: &[&str] = &["gq1", "gq2", "gq3", "gq4"];
const DATASETS: &[&str] = &["lastfm", "facebook", "wiki"];

/// (levels, cyclic) for each graph query: GQ1 is a 3-path, GQ2 a 4-path,
/// GQ3 a triangle, GQ4 a 4-cycle.
fn shape(query: &str) -> (usize, bool) {
    match query {
        "gq1" => (3, false),
        "gq2" => (4, false),
        "gq3" => (3, true),
        "gq4" => (4, true),
        q => panic!("unknown query {}", q),
    }
}

/// The same edge multiset `bench_queries::load_graph` hands to VPJoin.
fn load_edges(dataset: &str) -> Vec<(u64, u64)> {
    halo2_experiments::bench_queries::load_graph(dataset)
        .into_iter()
        .map(|e| (e.src, e.dst))
        .collect()
}

/// Smallest degree whose domain holds `rows` plus room for blinding and the
/// lookup arguments' unusable rows.
fn min_k(rows: u128) -> u32 {
    let mut k = 8u32;
    while (1u128 << k) < rows + 128 {
        k += 1;
    }
    k
}

/// Rows and degree of the anchor over the first `a` edges.
fn anchor_shape(edges: &[(u64, u64)], a: usize, levels: usize) -> (u128, u32) {
    let sizes = level_sizes(&edges[..a], levels);
    let n0 = circuit_rows(a as u128, &sizes[1..]);
    (n0, min_k(n0))
}

/// Largest prefix of `edges` whose anchor still fits a 2^`target` domain.
///
/// Monotone in the prefix length (adding an edge can only add paths), so a
/// binary search is exact.  Falls back to the smallest prefix that produces a
/// usable circuit when even that overshoots the target.
fn size_anchor(edges: &[(u64, u64)], levels: usize, target: u32) -> usize {
    let (mut lo, mut hi) = (levels, edges.len());
    if anchor_shape(edges, hi, levels).1 <= target {
        return hi;
    }
    while lo < hi {
        let mid = lo + (hi - lo + 1) / 2;
        if anchor_shape(edges, mid, levels).1 <= target {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo.max(levels)
}

/// Row count of the fully padded, fully oblivious circuit over `m` edges,
/// using the public-size bound |P_t| <= m^t that PoneglyphDB must assume when
/// no intermediate cardinality may leak.
///
/// This is the bound the paper reports, and in the evaluated setting it is
/// tight, not merely valid: the graph queries are SQL queries (bag
/// semantics) with no uniqueness constraint on the edge relation, and an
/// instance concentrating its multiplicity on a single t-edge path attains
/// (m/t)^t = Theta(m^t).  `worst_case_rows_agm` gives the set-semantics
/// bound, reported for the paper's robustness comparison.
fn worst_case_rows(m: u128, levels: usize) -> u128 {
    let mut caps = Vec::new();
    let mut c: u128 = m;
    for _ in 0..(levels - 1) {
        c = c.saturating_mul(m);
        caps.push(c);
    }
    circuit_rows(m, &caps)
}

fn pow_sat(m: u128, e: u32) -> u128 {
    (0..e).fold(1u128, |a, _| a.saturating_mul(m))
}

/// The same padded circuit sized by the AGM bound instead.
///
/// A t-edge path has t+1 vertices and fractional edge cover number
/// rho* = ceil((t+1)/2), so under SET semantics (every edge distinct)
/// |P_t| <= m^ceil((t+1)/2) for every graph with m edges, tighter than m^t
/// by a factor of m at every level past P_2.
///
/// This bound does NOT govern the paper's evaluated setting: the queries use
/// SQL bag semantics with no public uniqueness constraint on the edges, and
/// with duplicates allowed m^t is tight (see `worst_case_rows`), so padding
/// to AGM there would under-provision on duplicate-heavy instances.  It is
/// computed because the paper's robustness claim quotes it: even granting
/// PoneglyphDB set semantics, the padded domains stay far beyond runnable
/// (2^30 to 2^50 vs 2^23) and the speedups stay at or above three orders of
/// magnitude.
fn worst_case_rows_agm(m: u128, levels: usize) -> u128 {
    let caps: Vec<u128> = (2..=levels)
        .map(|t| pow_sat(m, ((t + 1) / 2 + (t + 1) % 2) as u32))
        .collect();
    circuit_rows(m, &caps)
}

/// ceil(log2(n)).
fn ceil_log2(n: u128) -> u32 {
    128 - n.leading_zeros() - u32::from(n.is_power_of_two())
}

struct Row {
    query: String,
    dataset: String,
    edges_full: usize,
    anchor_edges: usize,
    anchor_levels: String,
    output_count: u64,
    n0: u128,
    k0: u32,
    keygen_s: f64,
    t0_s: f64,
    verify_s: f64,
    proof_bytes: usize,
    n_worst_log2: f64,
    gc_log2: u32,
    t_est_s: f64,
    t_est_years: f64,
    gc_agm_log2: u32,
    t_est_agm_s: f64,
    t_est_agm_years: f64,
    status: String,
}

impl Row {
    fn header() -> &'static str {
        "query,dataset,edges_full,anchor_edges,anchor_level_sizes,output_count,N0,k0,\
         keygen_s,T0_prove_s,verify_s,proof_bytes,N_worst_log2,GC_log2,T_est_s,T_est_years,\
         GC_agm_log2,T_est_agm_s,T_est_agm_years,profile,status"
    }
    fn to_csv(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{:.3},{:.3},{:.4},{},{:.2},{},{:.6e},{:.3e},\
             {},{:.6e},{:.3e},{},{}",
            self.query,
            self.dataset,
            self.edges_full,
            self.anchor_edges,
            self.anchor_levels,
            self.output_count,
            self.n0,
            self.k0,
            self.keygen_s,
            self.t0_s,
            self.verify_s,
            self.proof_bytes,
            self.n_worst_log2,
            self.gc_log2,
            self.t_est_s,
            self.t_est_years,
            self.gc_agm_log2,
            self.t_est_agm_s,
            self.t_est_agm_years,
            if cfg!(debug_assertions) { "debug" } else { "release" },
            self.status
        )
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!(
            "usage: pone_graph_bench <out.csv> [query ...] [dataset ...]\n  \
             queries:  {}\n  datasets: {}",
            QUERIES.join(" "),
            DATASETS.join(" ")
        );
        std::process::exit(2);
    }
    let out_path = args[0].clone();
    let mut queries: Vec<String> = Vec::new();
    let mut datasets: Vec<String> = Vec::new();
    for a in &args[1..] {
        if QUERIES.contains(&a.as_str()) {
            queries.push(a.clone());
        } else if DATASETS.contains(&a.as_str()) {
            datasets.push(a.clone());
        } else {
            eprintln!(
                "unknown argument `{}` (queries: {}; datasets: {})",
                a,
                QUERIES.join(" "),
                DATASETS.join(" ")
            );
            std::process::exit(2);
        }
    }
    if queries.is_empty() {
        queries = QUERIES.iter().map(|s| s.to_string()).collect();
    }
    if datasets.is_empty() {
        datasets = DATASETS.iter().map(|s| s.to_string()).collect();
    }

    let target_k: u32 = std::env::var("PONE_K0")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(17);
    let forced_edges: Option<usize> = std::env::var("PONE_EDGES")
        .ok()
        .and_then(|v| v.parse().ok());
    let plan_only = std::env::var("PONE_PLAN_ONLY").map(|v| v == "1").unwrap_or(false);

    println!(
        "PoneglyphDB-style graph baseline: anchor + extrapolation (Section 8.1)\n\
         profile={}  target anchor domain=2^{}\n\
         prover: REAL Halo2 pipeline (keygen_vk / keygen_pk / create_proof / verify_proof, \
         IPA over Pasta) -- MockProver is NOT used here",
        if cfg!(debug_assertions) { "DEBUG (unoptimized)" } else { "RELEASE (optimized)" },
        target_k
    );

    // Plan every job first: loading and counting is cheap next to proving, and
    // it means a bad dataset or a missing params file surfaces immediately.
    struct Job {
        query: String,
        dataset: String,
        edges: Vec<(u64, u64)>,
        anchor_edges: usize,
        n0: u128,
        k0: u32,
        gc_log2: u32,
        n_worst: u128,
        gc_agm_log2: u32,
    }
    let mut jobs: Vec<Job> = Vec::new();
    println!("\nplan:");
    println!(
        "    {:<4} {:<9} {:>8}  {:>8} {:>5}   {:>7} {:>7}  {}",
        "qry", "dataset", "edges", "anchor", "k0", "GC", "GC(agm)", "worst-case rows N"
    );
    for d in &datasets {
        let edges = load_edges(d);
        for q in &queries {
            let (levels, _) = shape(q);
            let a = forced_edges
                .map(|f| f.min(edges.len()).max(levels))
                .unwrap_or_else(|| size_anchor(&edges, levels, target_k));
            let (n0, k0) = anchor_shape(&edges, a, levels);
            let n_worst = worst_case_rows(edges.len() as u128, levels);
            let gc_log2 = ceil_log2(n_worst);
            let gc_agm_log2 = ceil_log2(worst_case_rows_agm(edges.len() as u128, levels));
            println!(
                "    {:<4} {:<9} {:>8}  {:>8} {:>5}   {:>7} {:>7}  {} (~2^{:.1})",
                q,
                d,
                edges.len(),
                a,
                k0,
                format!("2^{}", gc_log2),
                format!("2^{}", gc_agm_log2),
                n_worst,
                (n_worst as f64).log2()
            );
            jobs.push(Job {
                query: q.clone(),
                dataset: d.clone(),
                edges: edges.clone(),
                anchor_edges: a,
                n0,
                k0,
                gc_log2,
                n_worst,
                gc_agm_log2,
            });
        }
    }
    println!(
        "\nGC uses |P_t| <= m^t, the bound the paper reports; it is TIGHT in the paper's\n\
         setting (SQL bag semantics, no public uniqueness constraint on the edges).\n\
         GC(agm) is the set-semantics AGM bound |P_t| <= m^ceil((t+1)/2), which would apply\n\
         only if edge distinctness were publicly assumed; it is reported for the paper's\n\
         robustness claim: even under AGM the padded domains stay unrunnable and the\n\
         speedups stay at or above three orders of magnitude.  Both T_est values are in the CSV."
    );
    if plan_only {
        println!("\nPONE_PLAN_ONLY=1 -- stopping before any proving.");
        return;
    }

    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        if !dir.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(dir);
        }
    }
    let mut f = std::fs::File::create(&out_path).expect("cannot create output csv");
    writeln!(f, "{}", Row::header()).unwrap();
    f.flush().unwrap();

    let total = jobs.len();
    let mut rows: Vec<Row> = Vec::new();
    for (i, job) in jobs.iter().enumerate() {
        println!(
            "\n[{}/{}] {} on {}: anchor over {} of {} edges (k0={})",
            i + 1,
            total,
            job.query,
            job.dataset,
            job.anchor_edges,
            job.edges.len(),
            job.k0
        );
        let (levels, cyclic) = shape(&job.query);
        let anchor: Vec<(u64, u64)> = job.edges[..job.anchor_edges].to_vec();
        let (lvls, count) = enumerate_paths(&anchor, levels, cyclic);
        let capacities: Vec<usize> = lvls.iter().map(|l| l.len().max(1)).collect();
        let level_str = capacities
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        println!("        true intermediates: [{}]  output count = {}", level_str, count);

        let circuit = PoneBaselineCircuit {
            edges: anchor,
            levels,
            cyclic,
            capacities,
            ..Default::default()
        };
        let public_input = vec![Fp::from(count)];

        let params = params_for(job.k0);
        let t = Instant::now();
        let vk = keygen_vk(&params, &circuit).expect("keygen_vk");
        let pk = keygen_pk(&params, vk, &circuit).expect("keygen_pk");
        let keygen_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut transcript = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);
        create_proof::<IPACommitmentScheme<_>, ProverIPA<_>, _, _, _, _>(
            &params,
            &pk,
            &[circuit],
            &[&[&public_input]],
            OsRng,
            &mut transcript,
        )
        .expect("proof generation");
        let proof = transcript.finalize();
        let t0_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let strategy = SingleStrategy::new(&params);
        let mut rt = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        let ok = verify_proof(&params, pk.get_vk(), strategy, &[&[&public_input]], &mut rt).is_ok();
        let verify_s = t.elapsed().as_secs_f64();

        // T_est = T_0 * G_C / G_{C,0}; both domains are powers of two, so the
        // ratio is an exact shift and stays in f64 range as an exponent.
        let year_s = 365.25 * 24.0 * 3600.0;
        let scale = 2f64.powi(job.gc_log2 as i32 - job.k0 as i32);
        let t_est_s = t0_s * scale;
        let t_est_years = t_est_s / year_s;
        let t_est_agm_s = t0_s * 2f64.powi(job.gc_agm_log2 as i32 - job.k0 as i32);
        let t_est_agm_years = t_est_agm_s / year_s;
        println!(
            "        keygen={:.2}s  T_0={:.2}s  verify={:.3}s  proof={}B  verified={}",
            keygen_s,
            t0_s,
            verify_s,
            proof.len(),
            ok
        );
        println!(
            "        T_est = T_0 * 2^({} - {}) = {:.3e} s  (~{:.2e} years)   [bound m^t]",
            job.gc_log2, job.k0, t_est_s, t_est_years
        );
        println!(
            "        T_est = T_0 * 2^({} - {}) = {:.3e} s  (~{:.2e} years)   [bound AGM]",
            job.gc_agm_log2, job.k0, t_est_agm_s, t_est_agm_years
        );

        let row = Row {
            query: job.query.clone(),
            dataset: job.dataset.clone(),
            edges_full: job.edges.len(),
            anchor_edges: job.anchor_edges,
            anchor_levels: level_str,
            output_count: count,
            n0: job.n0,
            k0: job.k0,
            keygen_s,
            t0_s,
            verify_s,
            proof_bytes: proof.len(),
            n_worst_log2: (job.n_worst as f64).log2(),
            gc_log2: job.gc_log2,
            t_est_s,
            t_est_years,
            gc_agm_log2: job.gc_agm_log2,
            t_est_agm_s,
            t_est_agm_years,
            status: if ok { "ok".into() } else { "FAILED_VERIFY".into() },
        };
        writeln!(f, "{}", row.to_csv()).unwrap();
        f.flush().unwrap();
        rows.push(row);
    }

    println!("\nwrote {}\n", out_path);
    println!(
        "{:<5} {:<9} {:>9} {:>8} {:>13} {:>8} {:>13}",
        "query", "dataset", "T_0 (s)", "GC", "T_est (yr)", "GC(agm)", "T_est agm (yr)"
    );
    for r in &rows {
        println!(
            "{:<5} {:<9} {:>9.2} {:>8} {:>13.2e} {:>8} {:>13.2e}",
            r.query,
            r.dataset,
            r.t0_s,
            format!("2^{}", r.gc_log2),
            r.t_est_years,
            format!("2^{}", r.gc_agm_log2),
            r.t_est_agm_years
        );
    }
    let bad = rows.iter().filter(|r| r.status != "ok").count();
    if bad > 0 {
        eprintln!("\n{}/{} rows FAILED verification", bad, rows.len());
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_log2_is_exact_on_and_around_powers_of_two() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(1 << 60), 60);
        assert_eq!(ceil_log2((1 << 60) + 1), 61);
    }

    #[test]
    fn min_k_leaves_room_for_blinding_and_lookups() {
        assert_eq!(min_k(1), 8);
        // A domain must never be chosen that the rows plus headroom overflow.
        for rows in [100u128, 1000, 60000, 70000, 130000, 200000] {
            let k = min_k(rows);
            assert!((1u128 << k) >= rows + 128, "2^{} too small for {}", k, rows);
            assert!(k == 8 || (1u128 << (k - 1)) < rows + 128, "2^{} not minimal", k);
        }
    }

    /// The naive bound is m^levels and the AGM bound m^ceil((levels+1)/2);
    /// for the shapes the harness supports they differ by exactly one factor
    /// of m, which is the whole reason both are reported.
    #[test]
    fn agm_bound_is_one_factor_of_m_tighter() {
        for &m in &[27_806u128, 88_234, 103_689] {
            for &levels in &[3usize, 4] {
                let naive = worst_case_rows(m, levels);
                let agm = worst_case_rows_agm(m, levels);
                assert_eq!(naive, pow_sat(m, levels as u32) + 1);
                assert_eq!(agm, pow_sat(m, levels as u32 - 1) + 1);
                assert!(agm < naive);
            }
        }
    }

    #[test]
    fn pow_sat_saturates_instead_of_wrapping() {
        assert_eq!(pow_sat(2, 0), 1);
        assert_eq!(pow_sat(3, 4), 81);
        assert_eq!(pow_sat(u128::MAX, 2), u128::MAX);
    }

    /// `size_anchor`'s binary search is only exact if the predicate is
    /// monotone in the prefix length: adding an edge can only add paths.
    #[test]
    fn anchor_degree_is_monotone_in_prefix_length() {
        let edges: Vec<(u64, u64)> = (0u64..400).map(|i| (i % 40, (i * 7) % 40 + 40)).collect();
        for levels in [3usize, 4] {
            let mut prev = 0u32;
            for a in levels..=edges.len() {
                let k = anchor_shape(&edges, a, levels).1;
                assert!(k >= prev, "degree fell from 2^{} to 2^{} at a={}", prev, k, a);
                prev = k;
            }
            // and the search returns the largest prefix under the target
            let target = anchor_shape(&edges, edges.len(), levels).1;
            assert_eq!(size_anchor(&edges, levels, target), edges.len());
            if target > 10 {
                let a = size_anchor(&edges, levels, target - 1);
                assert!(anchor_shape(&edges, a, levels).1 <= target - 1);
                assert!(a == edges.len() || anchor_shape(&edges, a + 1, levels).1 > target - 1);
            }
        }
    }
}
