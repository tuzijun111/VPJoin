//! Cost-measurement harness for the database-commitment layer (Appendix A).
//!
//! Measures, without touching any query circuit:
//!   1. one-time setup: IPA parameter generation, canonicalization, and the
//!      published Pedersen commitment Commit(D) (one length-2^k MSM);
//!   2. per-query binding: the IPA opening proof that ties a query proof to
//!      Commit(D), and its verification.
//!
//! The reported numbers are the *additional* cost of the commitment layer:
//! add (2) to any query's proving/verification time, and (1) once at setup.
//!
//! Usage:
//!   cargo run --release --bin commitment_bench                 # synthetic DB
//!   cargo run --release --bin commitment_bench <dir> <t1> ...  # TPC-H .tbl
//!
//! With a directory argument, each named table is loaded from `<dir>/<t>.tbl`
//! ('|'-separated; numeric fields parsed as u64, dates as YYYYMMDD, other
//! strings FNV-hashed to u64), reproducing the canonical fixed layout shared
//! across queries.

use std::time::Instant;

use halo2_experiments::commitment::{
    bind_query_proof, canonicalize, commit_db, min_k, verify_binding,
};
use halo2_proofs::poly::commitment::ParamsProver;
use halo2_proofs::poly::ipa::commitment::ParamsIPA;
use halo2curves::pasta::{vesta, Fp};
use ff::Field;
use rand::rngs::OsRng;

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
        // signed fixed-point with 2 decimals, zigzag-encoded (injective)
        let i = (v * 100.0).round() as i64;
        return ((i << 1) ^ (i >> 63)) as u64;
    }
    // dates like 1996-01-02 -> 19960102
    let digits: String = t.chars().filter(|c| c.is_ascii_digit()).collect();
    if !digits.is_empty() && t.chars().all(|c| c.is_ascii_digit() || c == '-') {
        if let Ok(v) = digits.parse::<u64>() {
            return v;
        }
    }
    fnv1a(t)
}

fn load_tbl(path: &str) -> Vec<Vec<u64>> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            line.trim_end_matches('|')
                .split('|')
                .map(parse_field)
                .collect()
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let relations: Vec<(String, Vec<Vec<u64>>)> = if args.len() >= 3 {
        let dir = &args[1];
        args[2..]
            .iter()
            .map(|t| (t.clone(), load_tbl(&format!("{}/{}.tbl", dir, t))))
            .collect()
    } else {
        println!("no data directory given; using a synthetic 60K x 16 fact table + dimensions");
        let mut rels = vec![(
            "lineitem".to_string(),
            (0..60_000u64)
                .map(|i| (0..16).map(|j| i.wrapping_mul(31).wrapping_add(j)).collect())
                .collect::<Vec<Vec<u64>>>(),
        )];
        for (name, rows, cols) in [
            ("orders", 15_000u64, 9usize),
            ("customer", 1_500, 8),
            ("supplier", 100, 7),
            ("nation", 25, 4),
            ("region", 5, 3),
        ] {
            rels.push((
                name.to_string(),
                (0..rows)
                    .map(|i| (0..cols as u64).map(|j| i * 7 + j).collect())
                    .collect(),
            ));
        }
        rels
    };

    let total_cells: usize = relations
        .iter()
        .map(|(_, rows)| rows.iter().map(|r| r.len()).sum::<usize>())
        .sum();
    let k = min_k(total_cells);
    println!(
        "database: {} relations, {} cells -> committed vector length 2^{}",
        relations.len(),
        total_cells,
        k
    );

    // ---- one-time setup costs -------------------------------------------
    let t = Instant::now();
    let params = ParamsIPA::<vesta::Affine>::new(k);
    println!("[setup] IPA params (k = {:2}):        {:?}", k, t.elapsed());

    let t = Instant::now();
    let db = canonicalize(&relations, k);
    println!("[setup] canonicalization:            {:?}", t.elapsed());

    let blind = Fp::random(OsRng);
    let t = Instant::now();
    let commitment = commit_db(&params, &db, blind);
    println!("[setup] Commit(D) (one MSM):         {:?}", t.elapsed());

    // ---- per-query binding costs ----------------------------------------
    // A stand-in for a query proof; only its bytes enter the Fiat--Shamir
    // binding, so its content does not affect the measured cost.
    let query_proof = vec![0x5au8; 30_000];

    let t = Instant::now();
    let opening = bind_query_proof(&params, &db, blind, &commitment, &query_proof, OsRng);
    println!("[per-query] binding opening proof:   {:?}", t.elapsed());
    println!("[per-query] opening proof size:      {} bytes", opening.proof.len());

    let t = Instant::now();
    let ok = verify_binding(&params, &commitment, &db.layout, &query_proof, &opening);
    println!("[per-query] binding verification:    {:?}", t.elapsed());
    assert!(ok, "binding verification failed");
    println!("binding verified against the published Commit(D)");
}
