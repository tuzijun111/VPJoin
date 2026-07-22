//! Per-QUERY cost of the database-commitment layer (Appendix A), written to
//! CSV: for each VPJoin query, the additional cost of binding exactly the
//! columns that query reads, measured against a baseline circuit consuming
//! the same columns without binding.
//!
//! Column inventories are taken from the actual query circuits (which
//! attributes enter the witness), and each query runs in its circuit's own
//! domain: k = 16 for all TPC-H queries (param16), k = 17 for GQ1/GQ2
//! (param17) and k = 18 for GQ3/GQ4 (param18), matching the degrees the real
//! proofs use.  Graph queries read Edge(src,dst) -- 2 columns -- and are
//! measured on each of the three SNAP datasets.
//!
//! Usage:
//!   cargo run --release --bin query_commit_bench -- <out.csv> [query ...]
//!   queries: q3 q5 q8 q9 q18 gq1 gq2 gq3 gq4   (default: all)
//! e.g.
//!   cargo run --release --bin query_commit_bench -- results/query_commit_cost.csv
//!   cargo run --release --bin query_commit_bench -- results/tpch.csv q3 q5 q8 q9 q18
//!
//! CSV columns:
//!   query, dataset, k, n_columns,
//!   setup_commit_s, published_commit_bytes,
//!   baseline_keygen_s, baseline_prove_s, baseline_verify_s, baseline_proof_bytes,
//!   bound_keygen_s, bound_prove_s, bound_verify_s, bound_proof_bytes,
//!   open_prove_s, open_verify_s, open_proof_bytes,
//!   delta_prove_s, delta_verify_s, delta_proof_bytes, total_extra_prove_s

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::Instant;

use ff::Field;
use halo2_experiments::column_commit::{
    binding_challenge, commit_column_vectors, open_column_vectors, vector_evaluations,
    verify_column_openings, BoundColumnsCircuit, PlainColumnsCircuit,
};
use halo2_proofs::{
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit},
    poly::{
        commitment::ParamsProver,
        ipa::{
            commitment::{IPACommitmentScheme, ParamsIPA},
            multiopen::ProverIPA,
            strategy::SingleStrategy,
        },
        VerificationStrategy,
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use halo2curves::pasta::{vesta, Fp};
use rand::rngs::OsRng;

// ---------------------------------------------------------------------------
// Query specs: exactly the source attributes each VPJoin circuit witnesses,
// as (.tbl file, column indices in standard TPC-H column order).
// ---------------------------------------------------------------------------

struct Spec {
    query: &'static str,
    k: u32,
    /// (table file base name, column indices)
    tables: &'static [(&'static str, &'static [usize])],
}

// .tbl column orders (standard TPC-H):
// customer: 0 custkey 1 name 2 address 3 nationkey 4 phone 5 acctbal 6 mktsegment
// orders:   0 orderkey 1 custkey 2 status 3 totalprice 4 orderdate 5 prio 6 clerk 7 shipprio
// lineitem: 0 orderkey 1 partkey 2 suppkey 3 lineno 4 qty 5 extprice 6 discount ... 10 shipdate
// supplier: 0 suppkey 1 name 2 address 3 nationkey
// nation:   0 nationkey 1 name 2 regionkey
// region:   0 regionkey 1 name
// part:     0 partkey 1 name 2 mfgr 3 brand 4 type
// partsupp: 0 partkey 1 suppkey 2 availqty 3 supplycost
const TPCH_SPECS: &[Spec] = &[
    Spec {
        query: "q3",
        k: 16,
        tables: &[
            ("customer", &[6, 0]),          // c_mktsegment, c_custkey
            ("orders", &[4, 7, 1, 0]),      // o_orderdate, o_shippriority, o_custkey, o_orderkey
            ("lineitem", &[0, 5, 6, 10]),   // l_orderkey, l_extendedprice, l_discount, l_shipdate
        ],
    },
    Spec {
        query: "q5",
        k: 16,
        tables: &[
            ("customer", &[0, 3]),          // c_custkey, c_nationkey
            ("orders", &[4, 1, 0]),         // o_orderdate, o_custkey, o_orderkey
            ("lineitem", &[0, 2, 5, 6]),    // l_orderkey, l_suppkey, l_extendedprice, l_discount
            ("supplier", &[0, 3]),          // s_suppkey, s_nationkey
            ("nation", &[0, 1, 2]),         // n_nationkey, n_name, n_regionkey
            ("region", &[0, 1]),            // r_regionkey, r_name
        ],
    },
    Spec {
        query: "q8",
        k: 16,
        tables: &[
            ("region", &[0, 1]),            // r_regionkey, r_name
            ("nation", &[0, 2, 1]),         // n_nationkey, n_regionkey, n_name
            ("customer", &[0, 3]),          // c_custkey, c_nationkey
            ("orders", &[0, 1, 4]),         // o_orderkey, o_custkey, o_orderdate
            ("part", &[0, 4]),              // p_partkey, p_type
            ("supplier", &[0, 3]),          // s_suppkey, s_nationkey
            ("lineitem", &[0, 1, 2, 5, 6]), // l_orderkey, l_partkey, l_suppkey, l_extprice, l_discount
        ],
    },
    Spec {
        query: "q9",
        k: 16,
        tables: &[
            ("part", &[0, 1]),              // p_partkey, p_name
            ("supplier", &[0, 3]),          // s_suppkey, s_nationkey
            ("nation", &[0, 1]),            // n_nationkey, n_name
            ("orders", &[0, 4]),            // o_orderkey, o_orderdate
            ("partsupp", &[0, 1, 3]),       // ps_partkey, ps_suppkey, ps_supplycost
            ("lineitem", &[0, 1, 2, 4, 5, 6]), // orderkey, partkey, suppkey, qty, extprice, discount
        ],
    },
    Spec {
        query: "q18",
        k: 16,
        tables: &[
            ("customer", &[1, 0]),          // c_name, c_custkey
            ("orders", &[0, 1, 4, 3]),      // o_orderkey, o_custkey, o_orderdate, o_totalprice
            ("lineitem", &[0, 4]),          // l_orderkey, l_quantity
        ],
    },
];

/// Graph queries: Edge(src,dst) = 2 columns; k as used by the real proofs.
const GRAPH_SPECS: &[(&str, u32)] = &[("gq1", 17), ("gq2", 17), ("gq3", 18), ("gq4", 18)];
const GRAPH_DATASETS: &[(&str, &str)] = &[
    ("lastfm", "src/graph_data/last/lastfm_asia_edges.csv"),
    ("facebook", "src/graph_data/facebook/facebook_combined.txt"),
    ("wiki", "src/graph_data/wiki/wiki_Vote.txt"),
];

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn parse_field(s: &str) -> u64 {
    let t = s.trim();
    if let Ok(v) = t.parse::<u64>() {
        return v;
    }
    if let Ok(v) = t.parse::<f64>() {
        assert!(v.is_finite(), "unsupported numeric field {}", t);
        let i = (v * 100.0).round() as i64;
        return ((i << 1) ^ (i >> 63)) as u64;
    }
    let digits: String = t.chars().filter(|c| c.is_ascii_digit()).collect();
    if !digits.is_empty() && t.chars().all(|c| c.is_ascii_digit() || c == '-') {
        if let Ok(v) = digits.parse::<u64>() {
            return v;
        }
    }
    fnv1a(t)
}

/// Load selected columns of a pipe-separated TPC-H table ('.tbl', with a
/// '.cvs' fallback for the repo's region file).
fn load_columns(table: &str, idxs: &[usize]) -> Vec<Vec<u64>> {
    let paths = [
        format!("src/data/{}.tbl", table),
        format!("src/data/{}.cvs", table),
    ];
    let path = paths
        .iter()
        .find(|p| std::path::Path::new(p.as_str()).exists())
        .unwrap_or_else(|| panic!("no data file for table {}", table));
    let mut cols: Vec<Vec<u64>> = vec![Vec::new(); idxs.len()];
    for line in std::fs::read_to_string(path).unwrap().lines() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.trim_end_matches('|').split('|').collect();
        for (c, &j) in idxs.iter().enumerate() {
            cols[c].push(parse_field(fields.get(j).copied().unwrap_or("0")));
        }
    }
    cols
}

/// Load the two Edge columns (src, dst) of a SNAP edge list.
fn load_edge_columns(path: &str) -> Vec<Vec<u64>> {
    let mut src = Vec::new();
    let mut dst = Vec::new();
    for line in std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e))
        .lines()
    {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let mut it = t
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|x| !x.is_empty());
        if let (Some(a), Some(b)) = (it.next(), it.next()) {
            if let (Ok(a), Ok(b)) = (a.parse(), b.parse()) {
                src.push(a);
                dst.push(b);
            }
        }
    }
    vec![src, dst]
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

struct Timings {
    keygen_s: f64,
    prove_s: f64,
    verify_s: f64,
    proof_bytes: usize,
}

fn run_circuit<C: Circuit<Fp> + Clone>(
    params: &ParamsIPA<vesta::Affine>,
    circuit: C,
    instances: &[&[Fp]],
) -> Timings {
    let t = Instant::now();
    let vk = keygen_vk(params, &circuit).expect("keygen_vk");
    let pk = keygen_pk(params, vk, &circuit).expect("keygen_pk");
    let keygen_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    create_proof::<IPACommitmentScheme<_>, ProverIPA<_>, _, _, _, _>(
        params,
        &pk,
        &[circuit],
        &[instances],
        OsRng,
        &mut transcript,
    )
    .expect("proof generation");
    let proof = transcript.finalize();
    let prove_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let strategy = SingleStrategy::new(params);
    let mut rt = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    assert!(
        verify_proof(params, pk.get_vk(), strategy, &[instances], &mut rt).is_ok(),
        "verification failed"
    );
    let verify_s = t.elapsed().as_secs_f64();

    Timings {
        keygen_s,
        prove_s,
        verify_s,
        proof_bytes: proof.len(),
    }
}

/// Measure one query at const column count NC.
fn measure<const NC: usize>(
    params: &ParamsIPA<vesta::Affine>,
    k: u32,
    columns: Vec<Vec<u64>>,
) -> (f64, usize, Timings, Timings, f64, f64, usize, Vec<Fp>) {
    assert_eq!(columns.len(), NC);

    // setup: publish per-column commitments (hiding)
    let t = Instant::now();
    let commitments = commit_column_vectors(params, &columns, k, OsRng);
    let setup_commit_s = t.elapsed().as_secs_f64();
    let published = commitments.published_bytes();

    // baseline: the query's inputs without binding
    let base = run_circuit(
        params,
        PlainColumnsCircuit::<NC> {
            columns: columns.clone(),
        },
        &[],
    );

    // bound: inputs + in-circuit evaluation at the Fiat-Shamir challenge
    let x = binding_challenge(&commitments, &[0u8; 32]);
    let evals = vector_evaluations(&columns, k, x);
    let mut instance = vec![x];
    instance.extend_from_slice(&evals);
    let bound = run_circuit(
        params,
        BoundColumnsCircuit::<NC> {
            columns: columns.clone(),
            x,
        },
        &[&instance[..]],
    );

    // per-query openings of exactly these columns
    let t = Instant::now();
    let proofs = open_column_vectors(params, &commitments, &columns, x, OsRng);
    let open_prove_s = t.elapsed().as_secs_f64();
    let open_bytes: usize = proofs.iter().map(|p| p.len()).sum();
    let t = Instant::now();
    assert!(
        verify_column_openings(params, &commitments.points, x, &evals, &proofs),
        "openings failed"
    );
    let open_verify_s = t.elapsed().as_secs_f64();

    (
        setup_commit_s,
        published,
        base,
        bound,
        open_prove_s,
        open_verify_s,
        open_bytes,
        evals,
    )
}

fn dispatch(
    n: usize,
    params: &ParamsIPA<vesta::Affine>,
    k: u32,
    columns: Vec<Vec<u64>>,
) -> (f64, usize, Timings, Timings, f64, f64, usize, Vec<Fp>) {
    match n {
        2 => measure::<2>(params, k, columns),
        8 => measure::<8>(params, k, columns),
        10 => measure::<10>(params, k, columns),
        16 => measure::<16>(params, k, columns),
        17 => measure::<17>(params, k, columns),
        19 => measure::<19>(params, k, columns),
        n => panic!("unsupported column count {}: add a dispatch arm", n),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: query_commit_bench <out.csv> [q3|q5|q8|q9|q18|gq1|gq2|gq3|gq4 ...]");
        std::process::exit(1);
    }
    let out_csv = &args[1];
    let selected: Vec<String> = if args.len() > 2 {
        args[2..].iter().map(|s| s.to_lowercase()).collect()
    } else {
        ["q3", "q5", "q8", "q9", "q18", "gq1", "gq2", "gq3", "gq4"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    };

    let mut params_cache: HashMap<u32, ParamsIPA<vesta::Affine>> = HashMap::new();
    let mut get_params = |k: u32| {
        if !params_cache.contains_key(&k) {
            let t = Instant::now();
            params_cache.insert(k, ParamsIPA::<vesta::Affine>::new(k));
            println!("[params] generated k = {} in {:.1} s (one-time, shared)", k, t.elapsed().as_secs_f64());
        }
        params_cache.get(&k).unwrap().clone()
    };

    let mut csv = String::new();
    writeln!(
        csv,
        "query,dataset,k,n_columns,setup_commit_s,published_commit_bytes,\
         baseline_keygen_s,baseline_prove_s,baseline_verify_s,baseline_proof_bytes,\
         bound_keygen_s,bound_prove_s,bound_verify_s,bound_proof_bytes,\
         open_prove_s,open_verify_s,open_proof_bytes,\
         delta_prove_s,delta_verify_s,delta_proof_bytes,total_extra_prove_s"
    )
    .unwrap();

    let mut emit = |query: &str,
                    dataset: &str,
                    k: u32,
                    n: usize,
                    r: (f64, usize, Timings, Timings, f64, f64, usize, Vec<Fp>)| {
        let (setup_s, pub_b, base, bound, open_p, open_v, open_b, _evals) = r;
        let dp = bound.prove_s - base.prove_s;
        let dv = bound.verify_s - base.verify_s;
        let db = bound.proof_bytes as i64 - base.proof_bytes as i64;
        let total = dp + open_p;
        println!(
            "== {query} [{dataset}] k={k} cols={n}: baseline prove {:.3}s | bound {:.3}s | \
             delta +{:.3}s | openings {:.3}s (verify {:.3}s) | TOTAL extra prove {:.3}s",
            base.prove_s, bound.prove_s, dp, open_p, open_v, total
        );
        writeln!(
            csv,
            "{query},{dataset},{k},{n},{setup_s:.6},{pub_b},\
             {:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{},{open_p:.6},{open_v:.6},{open_b},\
             {dp:.6},{dv:.6},{db},{total:.6}",
            base.keygen_s,
            base.prove_s,
            base.verify_s,
            base.proof_bytes,
            bound.keygen_s,
            bound.prove_s,
            bound.verify_s,
            bound.proof_bytes,
        )
        .unwrap();
    };

    for spec in TPCH_SPECS {
        if !selected.contains(&spec.query.to_string()) {
            continue;
        }
        let mut columns = Vec::new();
        for (table, idxs) in spec.tables {
            columns.extend(load_columns(table, idxs));
        }
        let n = columns.len();
        let params = get_params(spec.k);
        println!("-- {} : {} bound columns from {} tables --", spec.query, n, spec.tables.len());
        let r = dispatch(n, &params, spec.k, columns);
        emit(spec.query, "tpch-60K", spec.k, n, r);
    }

    for (query, k) in GRAPH_SPECS {
        if !selected.contains(&query.to_string()) {
            continue;
        }
        for (dataset, path) in GRAPH_DATASETS {
            let columns = load_edge_columns(path);
            let n = columns.len();
            assert!(
                columns[0].len() + 8 <= (1usize << *k),
                "{}: {} edges exceed 2^{}",
                dataset,
                columns[0].len(),
                k
            );
            let params = get_params(*k);
            println!("-- {} [{}] : {} edges --", query, dataset, columns[0].len());
            let r = dispatch(n, &params, *k, columns);
            emit(query, dataset, *k, n, r);
        }
    }

    if let Some(parent) = std::path::Path::new(out_csv).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    std::fs::write(out_csv, &csv).unwrap_or_else(|e| panic!("cannot write {}: {}", out_csv, e));
    println!("\nresults written to {}", out_csv);
}
