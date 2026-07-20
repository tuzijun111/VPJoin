//! Cost-measurement harness for the IN-CIRCUIT input-binding step
//! (Appendix A): proving, inside a circuit, that the input columns a query
//! consumes equal the committed database tables, whose per-column
//! commitments are published at setup and reproduced in the verifying key.
//!
//! Reported per relation: keygen time (produces the VK whose fixed
//! commitments ARE the published ones), the equality that the published
//! standalone commitments match the VK, binding proof time, size, and
//! verification time.  These are the *additional* in-circuit costs of
//! binding a query's inputs to Commit(D)-style commitments; add them to a
//! query's cost (composing the same constraints into the query circuit
//! enforces them at the same per-cell price).
//!
//! Usage:
//!   cargo run --release --bin input_binding_bench -- <dir> <t1> [t2 ...]
//!   e.g.  cargo run --release --bin input_binding_bench -- src/data nation customer orders lineitem

use std::time::Instant;

use halo2_experiments::input_binding::{
    column_commitments, min_k_rows, InputBindingCircuit, MAX_COLS,
};
use halo2_proofs::{
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof},
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
use halo2curves::pasta::vesta;
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: input_binding_bench <dir> <table> [table ...]");
        std::process::exit(1);
    }
    for table in &args[2..] {
        let rows = load_tbl(&format!("{}/{}.tbl", args[1], table));
        let cols = rows.first().map(|r| r.len()).unwrap_or(0);
        let k = min_k_rows(rows.len());
        println!(
            "== {}: {} rows x {} cols, k = {} ==",
            table,
            rows.len(),
            cols,
            k
        );

        let circuit = InputBindingCircuit {
            rows: rows.clone(),
            cols,
        };
        let t = Instant::now();
        let params = ParamsIPA::<vesta::Affine>::new(k);
        println!("[setup] params:               {:?}", t.elapsed());

        let t = Instant::now();
        let vk = keygen_vk(&params, &circuit).expect("keygen_vk");
        let published = column_commitments(&params, &rows, k);
        assert_eq!(
            &vk.fixed_commitments()[..MAX_COLS],
            &published[..],
            "published column commitments must equal the VK's fixed commitments"
        );
        println!("[setup] keygen_vk + publish:  {:?} (commitments match VK)", t.elapsed());
        let t = Instant::now();
        let pk = keygen_pk(&params, vk, &circuit).expect("keygen_pk");
        println!("[setup] keygen_pk:            {:?}", t.elapsed());

        let t = Instant::now();
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        create_proof::<IPACommitmentScheme<_>, ProverIPA<_>, _, _, _, _>(
            &params,
            &pk,
            &[circuit],
            &[&[]],
            OsRng,
            &mut transcript,
        )
        .expect("proof generation");
        let proof = transcript.finalize();
        println!("[per-query] binding proof:    {:?}", t.elapsed());
        println!("[per-query] proof size:       {} bytes", proof.len());

        let t = Instant::now();
        let strategy = SingleStrategy::new(&params);
        let mut rt = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        assert!(
            verify_proof(&params, pk.get_vk(), strategy, &[&[]], &mut rt).is_ok(),
            "verification failed"
        );
        println!("[per-query] verification:     {:?}", t.elapsed());
    }
}
