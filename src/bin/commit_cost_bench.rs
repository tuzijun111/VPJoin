//! Unified cost measurement for the database-commitment layer (Appendix A),
//! reported as the **additional** cost over a pure query circuit and written
//! to CSV.
//!
//! Every measurement uses ONE domain `k`, derived from the largest table in
//! the set (the dominant one -- `lineitem` at 60,175 rows gives `k = 16`,
//! matching the degree the TPC-H circuits actually run at).  This is the
//! realistic setting: a query circuit that reads `lineitem` is a `k = 16`
//! circuit, so the small dimension tables it also reads live in that same
//! `2^16` domain, and their column openings cost the same as `lineitem`'s.
//! The IPA generators are position-indexed and `k`-independent, so a column
//! commitment is the same point at any `k` that fits it; only the cost of
//! opening it changes.  Parameters are generated once and shared, as the
//! query circuits share `param16`.
//!
//! For each table it measures, in that shared domain:
//!
//!   * `baseline`  -- a circuit that only consumes the table as witness
//!                    columns (a stand-in for the pure query circuit);
//!   * `bound`     -- the same inputs plus the in-circuit evaluation that
//!                    binds those witness columns to the published per-column
//!                    commitments.
//!
//! The `bound - baseline` delta is the additional in-circuit cost.  On top of
//! that it reports the one-time setup (publishing the per-column commitments)
//! and the per-query opening/verification of those commitments.  Each phase
//! is timed once (no repetition).
//!
//! Usage:
//!   cargo run --release --bin commit_cost_bench -- <dir> <out.csv> <t1> [t2 ...]
//!   e.g.
//!   cargo run --release --bin commit_cost_bench -- src/data results/commit_cost.csv \
//!       nation supplier customer orders lineitem
//!
//! CSV columns:
//!   table, rows, cols, k,
//!   setup_params_s, setup_commit_s, published_commit_bytes,
//!   baseline_keygen_s, baseline_prove_s, baseline_verify_s, baseline_proof_bytes,
//!   bound_keygen_s, bound_prove_s, bound_verify_s, bound_proof_bytes,
//!   open_prove_s, open_verify_s, open_proof_bytes,
//!   delta_prove_s, delta_verify_s, delta_proof_bytes, total_extra_prove_s

use std::fmt::Write as _;
use std::time::Instant;

use halo2_experiments::column_commit::{
    binding_challenge, column_evaluations, commit_columns, min_k_rows, open_columns,
    verify_column_openings, BoundTableCircuit, PlainTableCircuit, MAX_COLS,
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
use ff::Field;
use halo2curves::pasta::{vesta, Fp};
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

fn load_tbl(path: &str) -> Vec<Vec<u64>> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            line.trim_end_matches('|')
                .split('|')
                .take(MAX_COLS)
                .map(parse_field)
                .collect()
        })
        .collect()
}

/// Prove + verify one circuit, returning (keygen_s, prove_s, verify_s, bytes).
fn run_circuit<C: Circuit<Fp> + Clone>(
    params: &ParamsIPA<vesta::Affine>,
    circuit: C,
    instances: &[&[Fp]],
) -> (f64, f64, f64, usize) {
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

    (keygen_s, prove_s, verify_s, proof.len())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: commit_cost_bench <dir> <out.csv> <table> [table ...]\n\
             e.g.   commit_cost_bench src/data results/commit_cost.csv nation customer orders lineitem"
        );
        std::process::exit(1);
    }
    let (dir, out_csv) = (&args[1], &args[2]);

    // Load every table first so the domain can be set by the dominant one.
    let tables: Vec<(String, Vec<Vec<u64>>)> = args[3..]
        .iter()
        .map(|t| (t.clone(), load_tbl(&format!("{}/{}.tbl", dir, t))))
        .collect();
    let k = tables
        .iter()
        .map(|(_, rows)| min_k_rows(rows.len()))
        .max()
        .expect("at least one table");
    let dominant = tables
        .iter()
        .max_by_key(|(_, rows)| rows.len())
        .map(|(n, _)| n.clone())
        .unwrap_or_default();
    println!(
        "query-circuit domain: k = {} (set by the dominant table `{}`); \
         all tables share it, as they share one query circuit\n",
        k, dominant
    );

    // Parameters are generated once and shared by every table, exactly as the
    // TPC-H circuits share `param16`.
    let t = Instant::now();
    let params = ParamsIPA::<vesta::Affine>::new(k);
    let setup_params_s = t.elapsed().as_secs_f64();
    println!("[setup]    params (k={}) once:      {:.3} s\n", k, setup_params_s);

    let mut csv = String::new();
    writeln!(
        csv,
        "table,rows,cols,k,setup_params_s,setup_commit_s,published_commit_bytes,\
         baseline_keygen_s,baseline_prove_s,baseline_verify_s,baseline_proof_bytes,\
         bound_keygen_s,bound_prove_s,bound_verify_s,bound_proof_bytes,\
         open_prove_s,open_verify_s,open_proof_bytes,\
         delta_prove_s,delta_verify_s,delta_proof_bytes,total_extra_prove_s"
    )
    .unwrap();

    for (table_name, table) in &tables {
        let table = table.clone();
        let rows = table.len();
        let cols = table.first().map(|r| r.len()).unwrap_or(0);
        println!("== {}: {} rows x {} cols, k = {} ==", table_name, rows, cols, k);

        // ---- setup: publish per-column commitments (hiding) --------------
        let t = Instant::now();
        let commitments = commit_columns(&params, &table, k, OsRng);
        let setup_commit_s = t.elapsed().as_secs_f64();
        let published_bytes = commitments.published_bytes();
        println!(
            "[setup]    commit {} columns:      {:.3} s ({} bytes published)",
            commitments.cols, setup_commit_s, published_bytes
        );

        // ---- baseline: pure query circuit consuming the inputs -----------
        let plain = PlainTableCircuit {
            table: table.clone(),
            cols,
        };
        let (b_keygen, b_prove, b_verify, b_bytes) = run_circuit(&params, plain, &[]);
        println!(
            "[baseline] keygen {:.3} s | prove {:.3} s | verify {:.3} s | {} bytes",
            b_keygen, b_prove, b_verify, b_bytes
        );

        // ---- bound: same inputs + in-circuit binding ---------------------
        // COST ONLY. The transcript below is a constant stand-in, not a query
        // proof, so this `x` binds nothing: it is a public constant, exactly the
        // case `binding_challenge` rejects when the transcript is empty rather
        // than merely fixed. The measurement is unaffected -- the circuit does
        // the same work at any `x` -- but do not read this line as the protocol.
        let x = binding_challenge(&commitments, &[0u8; 32]);
        let evals = column_evaluations(&table, k, x);
        let mut instance = vec![x];
        instance.extend_from_slice(&evals);
        instance.resize(1 + MAX_COLS, Fp::ZERO);

        let bound = BoundTableCircuit {
            table: table.clone(),
            cols,
            x,
        };
        let (c_keygen, c_prove, c_verify, c_bytes) = run_circuit(&params, bound, &[&instance[..]]);
        println!(
            "[bound]    keygen {:.3} s | prove {:.3} s | verify {:.3} s | {} bytes",
            c_keygen, c_prove, c_verify, c_bytes
        );

        // ---- per-query: open the published commitments at x --------------
        let t = Instant::now();
        let proofs = open_columns(&params, &commitments, &table, x, OsRng);
        let open_prove_s = t.elapsed().as_secs_f64();
        let open_bytes: usize = proofs.iter().map(|p| p.len()).sum();

        let t = Instant::now();
        let ok = verify_column_openings(&params, &commitments.points, x, &evals, &proofs);
        let open_verify_s = t.elapsed().as_secs_f64();
        assert!(ok, "column openings failed to verify");
        println!(
            "[openings] prove {:.3} s | verify {:.3} s | {} bytes (all {} columns)",
            open_prove_s, open_verify_s, open_bytes, commitments.cols
        );

        // ---- deltas: the additional cost of the commitment layer ---------
        let delta_prove = c_prove - b_prove;
        let delta_verify = c_verify - b_verify;
        let delta_bytes = c_bytes as i64 - b_bytes as i64;
        let total_extra_prove = delta_prove + open_prove_s;
        println!(
            "[DELTA]    in-circuit prove +{:.3} s | verify +{:.3} s | {:+} bytes",
            delta_prove, delta_verify, delta_bytes
        );
        println!(
            "[DELTA]    total extra prover time (in-circuit + openings): {:.3} s\n",
            total_extra_prove
        );

        writeln!(
            csv,
            "{},{},{},{},{:.6},{:.6},{},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{},{:.6},{:.6},{},{:.6},{:.6},{},{:.6}",
            table_name,
            rows,
            cols,
            k,
            setup_params_s,
            setup_commit_s,
            published_bytes,
            b_keygen,
            b_prove,
            b_verify,
            b_bytes,
            c_keygen,
            c_prove,
            c_verify,
            c_bytes,
            open_prove_s,
            open_verify_s,
            open_bytes,
            delta_prove,
            delta_verify,
            delta_bytes,
            total_extra_prove
        )
        .unwrap();
    }

    if let Some(parent) = std::path::Path::new(out_csv).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    std::fs::write(out_csv, &csv).unwrap_or_else(|e| panic!("cannot write {}: {}", out_csv, e));
    println!("results written to {}", out_csv);
}
