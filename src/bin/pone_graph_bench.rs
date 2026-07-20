//! Anchor + extrapolation harness for the PoneglyphDB-style graph baselines
//! (estimation methodology of Section 8.1).
//!
//! Runs the binary-join-chain baseline circuit with capacities set to the
//! TRUE intermediate sizes (the measured anchor: N_0, G_{C,0}, T_0), then
//! reports the fully padded worst-case circuit size (N, G_C, using the
//! public-size bound |P_t| <= |E|^t) and the extrapolated proving time
//! T_est = T_0 * G_C / G_{C,0}.
//!
//! Usage:
//!   cargo run --release --bin pone_graph_bench -- <query> <edge_file> [max_edges] [sym]
//! where <query> is one of gq1 (3-path), gq2 (4-path), gq3 (triangle),
//! gq4 (4-cycle), and <edge_file> is a whitespace-separated edge list
//! (e.g. dp/dataset/facebook_combined.txt).  [max_edges] optionally
//! subsamples the first N edges to keep the anchor runnable; the literal
//! argument `sym` additionally inserts every edge in both directions
//! (SNAP files list each undirected edge once).

use std::time::Instant;

use halo2_experiments::graph_sql::pone_baseline::{
    enumerate_paths, PoneBaselineCircuit,
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
use halo2curves::pasta::{vesta, EqAffine, Fp};
use rand::rngs::OsRng;

fn load_edges(path: &str, max_edges: Option<usize>) -> Vec<(u64, u64)> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    // Accepts whitespace- or comma-separated pairs; '#' comments and header
    // lines (non-numeric) are skipped, covering SNAP .txt and .csv exports.
    let mut edges: Vec<(u64, u64)> = content
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .filter_map(|l| {
            let mut it = l
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter(|t| !t.is_empty());
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })
        .collect();
    edges.sort_unstable();
    edges.dedup();
    if let Some(m) = max_edges {
        edges.truncate(m);
    }
    edges
}

fn min_k(rows: usize) -> u32 {
    let mut k = 8u32;
    while (1usize << k) < rows + 16 {
        k += 1;
    }
    k
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: pone_graph_bench <gq1|gq2|gq3|gq4> <edge_file> [max_edges] [sym]");
        std::process::exit(1);
    }
    let (levels, cyclic) = match args[1].as_str() {
        "gq1" => (3usize, false),
        "gq2" => (4, false),
        "gq3" => (3, true),
        "gq4" => (4, true),
        q => panic!("unknown query {}", q),
    };
    let symmetrize = args[3..].iter().any(|a| a == "sym");
    let max_edges = args[3..]
        .iter()
        .find_map(|a| a.parse::<usize>().ok());
    let mut edges = load_edges(&args[2], max_edges);
    if symmetrize {
        let mut both: Vec<(u64, u64)> = edges
            .iter()
            .flat_map(|&(a, b)| [(a, b), (b, a)])
            .collect();
        both.sort_unstable();
        both.dedup();
        edges = both;
    }
    let m = edges.len();
    println!("query {} over {} edges", args[1], m);

    // True intermediate sizes -> anchor capacities and N_0.
    let (lvls, count) = enumerate_paths(&edges, levels, cyclic);
    let capacities: Vec<usize> = lvls.iter().map(|l| l.len().max(1)).collect();
    let n0: usize = m + 1 + capacities.iter().map(|c| c + 1).sum::<usize>();
    let k0 = min_k(n0.max(m + 1));
    println!(
        "true intermediate sizes: {:?}; output count = {}",
        capacities, count
    );
    println!("anchor gate rows N_0 = {}, padded domain G_C0 = 2^{}", n0, k0);

    // Worst-case (fully oblivious, public-size bound |P_t| <= |E|^t).
    let mut worst: u128 = (m as u128) + 1;
    let mut cap_w: u128 = m as u128;
    for _ in 0..(levels - 1) {
        cap_w = cap_w.saturating_mul(m as u128);
        worst += cap_w + 1;
    }
    // ceil(log2(worst)): position of the highest bit, +1 unless a power of two
    let gc_log2 = 128 - worst.leading_zeros() - u32::from(worst.is_power_of_two());
    println!(
        "worst-case gate rows N = {} (~2^{:.1}), padded domain G_C = 2^{}",
        worst,
        (worst as f64).log2(),
        gc_log2
    );

    // ---- run the anchor ----
    let circuit = PoneBaselineCircuit {
        edges,
        levels,
        cyclic,
        capacities,
        ..Default::default()
    };
    let public_input = vec![Fp::from(count)];

    let t = Instant::now();
    let params = ParamsIPA::<vesta::Affine>::new(k0);
    println!("[anchor] params (k = {}):    {:?}", k0, t.elapsed());
    let t = Instant::now();
    let vk = keygen_vk(&params, &circuit).expect("keygen_vk");
    let pk = keygen_pk(&params, vk, &circuit).expect("keygen_pk");
    println!("[anchor] keygen:             {:?}", t.elapsed());

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
    let t0 = t.elapsed();
    println!("[anchor] proving time T_0:   {:?}", t0);
    println!("[anchor] proof size:         {} bytes", proof.len());

    let t = Instant::now();
    let strategy = SingleStrategy::new(&params);
    let mut rt = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    assert!(
        verify_proof(&params, pk.get_vk(), strategy, &[&[&public_input]], &mut rt).is_ok(),
        "verification failed"
    );
    println!("[anchor] verification:       {:?}", t.elapsed());

    // ---- extrapolation ----
    let t_est_secs = t0.as_secs_f64() * 2f64.powi(gc_log2 as i32 - k0 as i32);
    println!(
        "extrapolated fully padded proving time T_est = T_0 * 2^({} - {}) = {:.3e} s (~{:.1e} years)",
        gc_log2,
        k0,
        t_est_secs,
        t_est_secs / (365.25 * 24.0 * 3600.0)
    );
}
