use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use halo2_proofs::{
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, Error as PlonkError},
    poly::{
        commitment::{Params, ParamsProver},
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

use crate::column_commit::{
    binding_challenge_db, open_column_vectors, vector_evaluations, verify_column_openings,
    BoundColumnsCircuit,
};
use crate::data::graph_data_processing::{read_edges, read_edges_csv, Edge};

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// Root under which `data/`, `graph_data/` and `proof/` live.  Override with
/// `VPJOIN_DATA=/path/to/src`.  Shared with the query tests via
/// [`crate::paths`], so both resolve identically.
pub fn data_root() -> PathBuf {
    crate::paths::src_root()
}

fn tbl(name: &str) -> String {
    crate::paths::data_file(name)
}

pub fn params_for(k: u32) -> ParamsIPA<vesta::Affine> {
    let path = PathBuf::from(crate::paths::param_file(k));
    if path.exists() {
        let mut fd = std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("cannot open {}: {}", path.display(), e));
        let p = ParamsIPA::<vesta::Affine>::read(&mut fd).unwrap_or_else(|e| {
            panic!(
                "{} exists but is not a valid IPA params file: {} -- refusing to \
                 silently regenerate; delete the file if it is corrupt",
                path.display(),
                e
            )
        });
        // The degree is encoded in the file header; make sure the file really
        // is the degree its name claims, so a mislabeled file can never be
        // silently used.
        assert_eq!(
            p.k(),
            k,
            "{} claims degree 2^{} in its header, expected 2^{}",
            path.display(),
            p.k(),
            k
        );
        // Diagnostics go to stderr: stdout carries results only, so a harness
        // can print a table without the SRS loads interleaving into it.
        eprintln!("  [params] loaded {} (degree 2^{})", path.display(), k);
        return p;
    }
    eprintln!(
        "  [params] {} NOT FOUND -- generating 2^{} params once and persisting them there",
        path.display(),
        k
    );
    let p: ParamsIPA<vesta::Affine> = ParamsIPA::new(k);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(mut fd) = std::fs::File::create(&path) {
        let _ = p.write(&mut fd);
    }
    p
}

// ---------------------------------------------------------------------------
// Result row
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
pub struct Row {
    pub query: String,
    pub dataset: String,
    pub k: u32,
    pub input_rows: usize,
    pub n_columns: usize,
    pub public_output: u64,

    /// "release" or "debug".  Recorded because the same circuit proves several
    /// times slower under the unoptimized profile, so timings are only
    /// comparable within one profile.
    pub profile: &'static str,
    /// Reading and parsing the input tables (outside the prover).
    pub load_s: f64,

    // simplification: the pure query circuit, as in the submission.
    // vk_s / pk_s are reported separately so they line up 1:1 with the
    // "Time to generate vk / pk" lines the per-query tests print.
    pub vk_s: f64,
    pub pk_s: f64,
    pub keygen_s: f64,
    pub prove_s: f64,
    pub verify_s: f64,
    pub proof_bytes: usize,

    // commitment layer (zero in simplification mode)
    pub commit_k: u32,
    pub commit_setup_s: f64,
    pub published_bytes: usize,
    pub bind_keygen_s: f64,
    pub bind_prove_s: f64,
    pub bind_verify_s: f64,
    pub bind_proof_bytes: usize,
    pub open_prove_s: f64,
    pub open_verify_s: f64,
    pub open_bytes: usize,

    pub total_prove_s: f64,
    pub total_proof_bytes: usize,
    /// End-to-end wall clock for this row (load + keygen + prove + verify +
    /// commitment layer).  Directly comparable to the "finished in ..." line
    /// that `cargo test` prints for the corresponding per-query test.
    pub wall_s: f64,

    /// Privacy regime the bags were sized under ("rjs" / "legacy-dp" /
    /// "dp(...)"), or "oblivious" for queries that materialize nothing.
    pub config: String,
    /// "ok", or a reason if this (query, dataset) pair could not be measured.
    pub status: String,
}

impl Row {
    pub fn header() -> &'static str {
        "query,dataset,k,input_rows,n_columns,public_output,profile,load_s,\
vk_s,pk_s,keygen_s,prove_s,verify_s,proof_bytes,\
commit_k,commit_setup_s,published_bytes,\
bind_keygen_s,bind_prove_s,bind_verify_s,bind_proof_bytes,\
open_prove_s,open_verify_s,open_bytes,\
total_prove_s,total_proof_bytes,wall_s,config,status"
    }

    pub fn to_csv(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{:.3},{},{:.3},{:.3},{:.3},{},{:.3},{:.3},{},{:.3},{},{:.3},{},{}",
            self.query,
            self.dataset,
            self.k,
            self.input_rows,
            self.n_columns,
            self.public_output,
            self.profile,
            self.load_s,
            self.vk_s,
            self.pk_s,
            self.keygen_s,
            self.prove_s,
            self.verify_s,
            self.proof_bytes,
            self.commit_k,
            self.commit_setup_s,
            self.published_bytes,
            self.bind_keygen_s,
            self.bind_prove_s,
            self.bind_verify_s,
            self.bind_proof_bytes,
            self.open_prove_s,
            self.open_verify_s,
            self.open_bytes,
            self.total_prove_s,
            self.total_proof_bytes,
            self.wall_s,
            if self.config.is_empty() { "n-a" } else { &self.config },
            if self.status.is_empty() { "ok" } else { &self.status },
        )
    }
}

pub const fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Pure query circuit (what the submission reports). Spelled
    /// `simplification` on the command line.
    Simplification,
    /// Query circuit + published per-column commitment + in-circuit
    /// witness-equality check + column openings.
    Full,
    /// ONLY the commitment layer, measured in the corresponding query
    /// circuit's own domain (same `k`).  The query proof itself is not re-run;
    /// add these numbers to the matching `simplification` row to get the full cost.
    /// The Fiat-Shamir challenge is bound to the query proof saved by the
    /// simplification run (`src/proof/bench/<query>_<dataset>.proof`) when present.
    Commit,
}

fn forced_k(derived: u32) -> u32 {
    let Ok(v) = std::env::var("VPJOIN_K") else {
        return derived;
    };
    let k: u32 = v
        .parse()
        .unwrap_or_else(|_| panic!("VPJOIN_K=`{}` is not a degree", v));
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "  [k] VPJOIN_K={} overrides every derived degree (first one derived here: {})",
            k, derived
        );
    });
    k
}

pub fn degree_for(query: &str, dataset: &str, privacy: Privacy) -> u32 {
    let derived = match (query, dataset) {
        ("q5", _) => {
            let (_, co_pad, ls_pad) = q5_pads(privacy);
            let TpchInput::Q5 {
                orders, lineitem, ..
            } = q5_raw_cached()
            else {
                unreachable!("q5_raw_cached() returns Q5")
            };
            let rows = lineitem
                .len()
                .max(orders.len() + co_pad)
                .max(q5_ls_true() + ls_pad) as u64;
            ceil_log2(rows + 64)
        }

        ("q3" | "q8" | "q9" | "q18", _) => ceil_log2(lineitem_rows() as u64 + 64),

        // Path queries: k = 17 on all three graphs (measured).
        ("gq1" | "gq2", _) => 17,

        ("gq3" | "gq4", _) => graph_degree(query, dataset, privacy),

        (q, d) => panic!("no degree tabulated for ({}, {})", q, d),
    };
    forced_k(derived)
}

// ---------------------------------------------------------------------------
// Generic prove/verify at a fixed degree
// ---------------------------------------------------------------------------

struct Timed {
    k: u32,
    vk_s: f64,
    pk_s: f64,
    keygen_s: f64,
    prove_s: f64,
    verify_s: f64,
    proof_bytes: usize,
    /// The proof itself, kept so the commitment layer's Fiat-Shamir challenge
    /// can be bound to it (as Appendix A specifies).
    proof: Vec<u8>,
}

/// The query proof written by an earlier `simplification` run, which the
/// commit-only measurement binds its Fiat--Shamir challenge to.
///
/// Missing proof is a hard error, not a degraded run. The challenge has to be
/// derived from a proof that already commits the witness; falling back to an
/// empty transcript makes it a public constant, and every number the run then
/// reports is the cost of a check that binds nothing.
fn saved_proof(label: &str) -> Vec<u8> {
    let p = PathBuf::from(crate::paths::proof_file("bench")).join(format!("{}.proof", label));
    match std::fs::read(&p) {
        Ok(b) if !b.is_empty() => {
            println!(
                "  [{}] binding challenge to saved query proof ({} bytes)",
                label,
                b.len()
            );
            b
        }
        Ok(_) => panic!(
            "[{}] the saved query proof at {} is empty. The binding challenge \
             must come from a proof that commits the witness; delete the file and \
             re-run `cargo vpjoin simplification {}` to regenerate it.",
            label,
            p.display(),
            label.split('_').next().unwrap_or(label)
        ),
        Err(e) => panic!(
            "[{}] no saved query proof at {} ({}). The commit-only mode binds its \
             challenge to the query proof, so that proof must exist first: run \
             `cargo vpjoin simplification {}` and then repeat this command. \
             (`cargo vpjoin full` proves the query in the same run and does not \
             need one.)",
            label,
            p.display(),
            e,
            label.split('_').next().unwrap_or(label)
        ),
    }
}

impl Timed {
    /// A placeholder for `Mode::Commit`, where the query proof is deliberately
    /// not re-run.  All timings are zero and the CSV records the degree only.
    fn skipped(k: u32) -> Self {
        Timed {
            k,
            vk_s: 0.0,
            pk_s: 0.0,
            keygen_s: 0.0,
            prove_s: 0.0,
            verify_s: 0.0,
            proof_bytes: 0,
            proof: Vec::new(),
        }
    }
}

/// Prove the query circuit unless we are only measuring the commitment layer.
fn maybe_run<C: Circuit<Fp>>(
    mode: Mode,
    label: &str,
    circuit: &C,
    instance: &[Fp],
    k: u32,
) -> Timed {
    if mode == Mode::Commit {
        println!(
            "  [{}] commit-only mode: query proof not re-run (k={})",
            label, k
        );
        return Timed::skipped(k);
    }
    if mode == Mode::Full {
        // The TIED proof carries the query AND the binding, so it IS the query
        // proof; proving the base circuit as well would prove the query twice.
        // It used to be done only to seed the Fiat--Shamir challenge, which it
        // could not soundly do: nothing constrained its witness to equal the
        // tied proof's. `binding_challenge_public` derives the challenge without
        // it, at the same security and half the cost.
        println!(
            "  [{}] full mode: the tied proof is the query proof (k={})",
            label, k
        );
        return Timed::skipped(k);
    }
    run_at(label, circuit, instance, k)
}

/// Diagnostic pass: check every gate, lookup, shuffle and copy constraint with
/// `MockProver` before spending a real proof.
///
/// The real prover reports only "verification failed"; `MockProver` names the
/// failing constraint, its region and its row, which is the difference between
/// a diagnosis and a guess. Enabled with `VPJOIN_MOCK=1`. It runs INSTEAD of the
/// real proof (the timings it would produce are meaningless), so a mock run
/// answers "is this circuit satisfiable at these parameters" and nothing else.
fn mock_check<C: Circuit<Fp>>(label: &str, circuit: &C, instance: &[Fp], k: u32) -> Timed {
    use halo2_proofs::dev::MockProver;
    println!("  [{}] VPJOIN_MOCK=1: MockProver at k={} (no real proof)", label, k);
    let t = Instant::now();
    let prover = MockProver::run(k, circuit, vec![instance.to_vec()])
        .unwrap_or_else(|e| panic!("[{}] MockProver could not synthesize at k={}: {:?}", label, k, e));
    match prover.verify() {
        Ok(()) => {
            println!(
                "  [{}] MockProver: SATISFIED in {:.2}s",
                label,
                t.elapsed().as_secs_f64()
            );
        }
        Err(failures) => {
            println!("  [{}] MockProver: {} FAILURE(S)", label, failures.len());
            // Every failure, not just the first: a padding or encoding mistake
            // typically breaks many rows at once and the pattern is the clue.
            for f in failures.iter().take(20) {
                println!("      {:?}", f);
            }
            if failures.len() > 20 {
                println!("      ... {} more", failures.len() - 20);
            }
            panic!("[{}] MockProver rejected the circuit at k={}", label, k);
        }
    }
    Timed::skipped(k)
}

fn run_at<C: Circuit<Fp>>(label: &str, circuit: &C, instance: &[Fp], k: u32) -> Timed {
    if std::env::var("VPJOIN_MOCK").map(|v| v == "1").unwrap_or(false) {
        return mock_check(label, circuit, instance, k);
    }
    let too_small = |stage: &str| -> ! {
        panic!(
            "{} does not fit at k={} (during {}). The degree table in \
             bench_queries::degree_for is wrong for this (query, dataset) -- \
             raise it and make sure src/proof/param{} exists.",
            label,
            k,
            stage,
            k + 1
        )
    };

    {
        let params = params_for(k);

        let t_vk = Instant::now();
        let vk = match keygen_vk(&params, circuit) {
            Ok(vk) => vk,
            Err(PlonkError::NotEnoughRowsAvailable { .. }) => too_small("keygen_vk"),
            Err(e) => panic!("keygen_vk failed at k={}: {:?}", k, e),
        };
        let vk_s = t_vk.elapsed().as_secs_f64();
        println!("  [{}] time to generate vk: {:.2}s (k={})", label, vk_s, k);

        let t_pk = Instant::now();
        let pk = match keygen_pk(&params, vk, circuit) {
            Ok(pk) => pk,
            Err(PlonkError::NotEnoughRowsAvailable { .. }) => too_small("keygen_pk"),
            Err(e) => panic!("keygen_pk failed at k={}: {:?}", k, e),
        };
        let pk_s = t_pk.elapsed().as_secs_f64();
        println!("  [{}] time to generate pk: {:.2}s", label, pk_s);

        let t = Instant::now();
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        create_proof::<IPACommitmentScheme<_>, ProverIPA<_>, _, _, _, _>(
            &params,
            &pk,
            std::slice::from_ref(circuit),
            &[&[instance]],
            OsRng,
            &mut transcript,
        )
        .unwrap_or_else(|e| panic!("create_proof failed at k={}: {:?}", k, e));
        let proof = transcript.finalize();
        let prove_s = t.elapsed().as_secs_f64();
        println!(
            "  [{}] REAL proof generated by create_proof in {:.2}s ({} bytes)",
            label,
            prove_s,
            proof.len()
        );

        // Persist the proof bytes as an auditable artifact, like the tests do.
        let dir = PathBuf::from(crate::paths::proof_file("bench"));
        let _ = std::fs::create_dir_all(&dir);
        let ppath = dir.join(format!("{}.proof", label));
        match std::fs::write(&ppath, &proof) {
            Ok(()) => println!("  [{}] proof written to {}", label, ppath.display()),
            Err(e) => eprintln!("  [{}] could not write proof file: {}", label, e),
        }

        let t = Instant::now();
        let strategy = SingleStrategy::new(&params);
        let mut rt = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        let ok = verify_proof(&params, pk.get_vk(), strategy, &[&[instance]], &mut rt).is_ok();
        let verify_s = t.elapsed().as_secs_f64();
        assert!(ok, "proof verification failed at k={}", k);
        println!("  [{}] verify_proof OK in {:.3}s", label, verify_s);

        return Timed {
            k,
            vk_s,
            pk_s,
            keygen_s: vk_s + pk_s,
            prove_s,
            verify_s,
            proof_bytes: proof.len(),
            proof,
        };
    }
}

// ---------------------------------------------------------------------------
// Commitment layer, applied to the columns a given query actually reads
// ---------------------------------------------------------------------------

/// Dispatch on column count -> const-generic bound circuit.
macro_rules! bind_dispatch {
    ($pub:expr, $idx:expr, $cols:expr, $k:expr, $proof:expr, $row:expr, $tied:expr, $($n:literal),+) => {
        match $cols.len() {
            $($n => bind_columns::<$n>($pub, $idx, $cols, $k, $proof, $row, $tied),)+
            other => panic!(
                "no BoundColumnsCircuit instantiation for {} columns; add it to bind_dispatch!",
                other
            ),
        }
    };
}


/// What `full` mode needs in order to prove the TIED bound circuit: the circuit
/// that carries the query AND the binding, with the binding copy-constrained to
/// the query's own witness cells.
///
/// This is the architecture Appendix A describes. The alternative the layer used
/// to prove -- `BoundColumnsCircuit`, a standalone circuit over a private copy
/// of the columns -- is a separate proof with no link to the query proof, so it
/// establishes only that SOME columns match `Commit(D)`.
pub enum Tied<'a> {
    Graph {
        query: &'a str,
        edges: &'a [Edge],
        pads: (usize, usize),
        cnt: u64,
    },
    Tpch(&'a TpchInput),
    /// Commit-only measurement: no tied proof, just the additive layer.
    None,
}

fn bind_columns<const NC: usize>(
    published: &Published,
    idx: &[usize],
    cols: &[Vec<u64>],
    k: u32,
    query_proof: &[u8],
    row: &mut Row,
    tied: Tied<'_>,
) {
    let params = params_for(k);
    let (db, setup_s) = (&published.0, published.1);

    // 1. Setup already published Commit(D) over every column of the database.
    //    A query SELECTS the columns it reads; it does not commit anything, so
    //    two queries reading a column open the same point under the same
    //    blinder. `commit_setup_s` and `published_bytes` are therefore the
    //    one-time cost of that publication, not a per-query cost.
    let commitments = db.view(idx);
    row.commit_setup_s = setup_s;
    row.published_bytes = db.published_bytes();

    // 2. In-circuit check that the witness columns equal the committed data, at
    //    a Fiat-Shamir point bound to the WHOLE publication, the columns this
    //    query opens, and the query proof itself (as Appendix A specifies).
    let x = match tied {
        // Additive layer measured on its own: bind to the saved query proof.
        Tied::None => binding_challenge_db(db, idx, query_proof),
        // Tied architecture: one proof, so there is no prior proof to bind to
        // and none is needed -- see `binding_challenge_public`.
        _ => crate::column_commit::binding_challenge_public(
            db,
            idx,
            format!("{}/{}", row.query, row.dataset).as_bytes(),
        ),
    };
    let evals = vector_evaluations(cols, k, x);
    let mut instance = vec![x];
    instance.extend_from_slice(&evals);

    let label = format!("{}_{}_bind", row.query, row.dataset);
    // `full` proves the TIED circuit: the query and the binding in ONE proof,
    // with the binding copy-constrained to the query's own witness cells. That
    // is what makes the evaluation an evaluation of THIS proof's witness, and
    // it is why the binding is not a separate proof over a private copy.
    let t = match tied {
        Tied::Graph {
            query,
            edges,
            pads,
            cnt,
        } => {
            let params_q = params_for(row.k);
            let p = crate::inline_bind::prove_graph_bound(
                &params_q, query, edges, pads, cnt, x, &instance,
            );
            println!(
                "  [{}] TIED proof (query + binding, one proof): {:.2}s, {} bytes",
                label, p.prove_s, p.bytes
            );
            Timed {
                k: row.k,
                prove_s: p.prove_s,
                verify_s: p.verify_s,
                proof_bytes: p.bytes,
                proof: p.proof,
                ..Timed::skipped(row.k)
            }
        }
        Tied::Tpch(input) => {
            let params_q = params_for(row.k);
            let p = crate::inline_bind::prove_bound(&params_q, input, x, &instance);
            println!(
                "  [{}] TIED proof (query + binding, one proof): {:.2}s, {} bytes",
                label, p.prove_s, p.bytes
            );
            Timed {
                k: row.k,
                prove_s: p.prove_s,
                verify_s: p.verify_s,
                proof_bytes: p.bytes,
                proof: p.proof,
                ..Timed::skipped(row.k)
            }
        }
        Tied::None => {
            let circuit = BoundColumnsCircuit::<NC> {
                columns: cols.to_vec(),
                x,
            };
            run_at(&label, &circuit, &instance, k)
        }
    };
    if t.k != k {
        // Record it if the bound circuit needed a larger degree than the
        // commitment domain (sound -- the evaluation depends only on the
        // data -- but the CSV should say so).
        row.status = format!("ok bind_k={}", t.k);
    }
    row.bind_keygen_s = t.keygen_s;
    row.bind_prove_s = t.prove_s;
    row.bind_verify_s = t.verify_s;
    row.bind_proof_bytes = t.proof_bytes;

    // 3. Open each committed column at x and verify the openings.
    let t = Instant::now();
    let openings = open_column_vectors(&params, &commitments, cols, x, OsRng);
    row.open_prove_s = t.elapsed().as_secs_f64();
    row.open_bytes = openings.iter().map(|o| o.len()).sum();

    let t = Instant::now();
    let ok = verify_column_openings(&params, &commitments.points, x, &evals, &openings);
    row.open_verify_s = t.elapsed().as_secs_f64();
    assert!(ok, "column openings failed to verify");
}

/// One published `Commit(D)` per dataset, built once and reused by every query.
pub type Published = std::sync::Arc<(crate::column_commit::DatabaseCommitment, f64)>;

fn publication_cache() -> &'static std::sync::Mutex<HashMap<String, Published>> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<String, Published>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// The canonical column layout of `D`: every distinct column any query of the
/// workload witnesses, deduplicated BY CONTENT, in first-seen order.
///
/// Built from the projections the circuits actually use rather than from a
/// hand-written schema, so the layout cannot drift from them. Two queries that
/// derive the same values are reading the same column of `D` and end up sharing
/// one published commitment, which is the whole point. Two queries that derive
/// DIFFERENT values from the same attribute -- Q3 encodes `o_orderdate` as a
/// timestamp where Q8 takes its year, and `n_name` is hashed with
/// `string_to_u64` for Q5 but `string_to_u64_trim` for Q8 -- are reading
/// different derived columns, and each is published separately, since a proof
/// binds to the column it actually reads.
pub fn tpch_database_columns(privacy: Privacy) -> Vec<Vec<u64>> {
    let mut db: Vec<Vec<u64>> = Vec::new();
    for q in ["q3", "q5", "q8", "q9", "q18"] {
        for col in tpch_inputs(q, privacy).columns() {
            if !db.contains(&col) {
                db.push(col);
            }
        }
    }
    db
}

/// Where each of `cols` sits in the published layout.
///
/// Every column a query witnesses must already be published; a miss means the
/// layout was built from a different workload than the one being proved, and
/// silently re-committing would be exactly the per-query behaviour this
/// replaces.
pub fn column_indices(db: &[Vec<u64>], cols: &[Vec<u64>]) -> Vec<usize> {
    cols.iter()
        .map(|c| {
            db.iter().position(|d| d == c).unwrap_or_else(|| {
                panic!(
                    "a witnessed column of length {} is not in the published Commit(D) \
                     ({} columns): the publication does not cover this query",
                    c.len(),
                    db.len()
                )
            })
        })
        .collect()
}

/// `Commit(D)` for `key`, published on first use and reused afterwards.
/// Returns the commitment and the one-time Setup cost that produced it.
pub fn published_db(key: &str, db_cols: &[Vec<u64>], k: u32) -> Published {
    let mut cache = publication_cache().lock().expect("publication cache poisoned");
    cache
        .entry(format!("{}/k{}", key, k))
        .or_insert_with(|| {
            let params = params_for(k);
            let t = Instant::now();
            let db = crate::column_commit::commit_database(&params, db_cols, k, OsRng);
            let setup_s = t.elapsed().as_secs_f64();
            println!(
                "  [setup] published Commit(D) for {}: {} columns, {} bytes, {:.3}s (once)",
                key,
                db.cols(),
                db.published_bytes(),
                setup_s
            );
            std::sync::Arc::new((db, setup_s))
        })
        .clone()
}

fn commit_k(cols: &[Vec<u64>], query_k: u32) -> u32 {
    let rows = cols.iter().map(|c| c.len()).max().unwrap_or(0);
    let need = crate::column_commit::min_k_rows(rows);
    assert!(
        need <= query_k,
        "columns need k={} but the query circuit is k={}",
        need,
        query_k
    );

    need
}

/// Bind one query's proof to the database commitment published at Setup.
///
/// `db_cols` is the canonical layout of `D` for this dataset; `cols` are the
/// columns this query's circuit witnesses, which must be a subset of it. The
/// commitment domain is sized for the WHOLE database, not for this query, so
/// that a column has one commitment whichever query reads it.
fn apply_commitment_layer(
    key: &str,
    db_cols: &[Vec<u64>],
    cols: &[Vec<u64>],
    query_k: u32,
    query_proof: &[u8],
    row: &mut Row,
    tied: Tied<'_>,
) {
    row.n_columns = cols.len();
    let k = commit_k(db_cols, query_k);
    row.commit_k = k;
    let published = published_db(key, db_cols, k);
    let idx = column_indices(db_cols, cols);
    bind_dispatch!(
        &published,
        &idx,
        cols,
        k,
        query_proof,
        row,
        tied,
        2,
        4,
        5,
        6,
        8,
        10,
        12,
        14,
        16,
        17,
        18,
        19,
        20
    );
}

// ---------------------------------------------------------------------------
// Graph queries
// ---------------------------------------------------------------------------

pub const GRAPH_DATASETS: &[&str] = &["lastfm", "facebook", "wiki"];

fn read_graph_file(dataset: &str) -> Vec<Edge> {
    match dataset {
        "lastfm" => read_edges_csv(&crate::paths::graph_file("last/lastfm_asia_edges.csv"))
            .expect("lastfm_asia_edges.csv"),
        "facebook" => read_edges(&crate::paths::graph_file("facebook/facebook_combined.txt"))
            .expect("facebook_combined.txt"),
        "wiki" => {
            read_edges(&crate::paths::graph_file("wiki/wiki_Vote.txt")).expect("wiki_Vote.txt")
        }
        other => panic!("unknown graph dataset {}", other),
    }
}

fn graph_cache() -> &'static std::sync::Mutex<HashMap<String, std::sync::Arc<Vec<Edge>>>> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<HashMap<String, std::sync::Arc<Vec<Edge>>>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

fn cap_edges(mut edges: Vec<Edge>, dataset: &str) -> Vec<Edge> {
    let Ok(v) = std::env::var("VPJOIN_MAX_EDGES") else {
        return edges;
    };
    let cap: usize = v
        .parse()
        .unwrap_or_else(|_| panic!("VPJOIN_MAX_EDGES=`{}` is not an edge count", v));
    if edges.len() > cap {
        eprintln!(
            "  [graph] VPJOIN_MAX_EDGES={} truncates {} from {} edges -- this is a SMALLER \
             instance than the paper's, not the published one",
            cap,
            dataset,
            edges.len()
        );
        edges.truncate(cap);
    }
    edges
}

pub fn load_graph(dataset: &str) -> Vec<Edge> {
    let mut cache = graph_cache().lock().expect("graph cache poisoned");
    let edges = cache
        .entry(dataset.to_string())
        // The cap is deterministic, so caching the capped list is
        // indistinguishable from capping every clone -- and it keeps the
        // stderr note to one line per dataset.
        .or_insert_with(|| std::sync::Arc::new(cap_edges(read_graph_file(dataset), dataset)));
    (**edges).clone()
}

/// The graph database's canonical published layout: BOTH encodings its queries
/// witness, as distinct columns.
///
/// There is no single encoding that serves every graph query. GQ1, GQ3 and GQ4
/// shift node ids by one so 0 stays free as the lookup gadgets' dummy row, and
/// witness `src + 1`; GQ2 witnesses the raw ids (its own SHIFT_ID appears only
/// inside masked Pairwise-Consistency keys, never in the base relation). A
/// commitment to one encoding cannot be tied to a witness holding the other --
/// publishing raw ids alone made GQ1/GQ3/GQ4 fail the tie, and shifting them
/// alone made GQ2 fail it.
///
/// So the publication carries both, exactly as the TPC-H layout carries the
/// distinct derived columns its queries witness (`tpch_database_columns`). Two
/// queries sharing an encoding share the published commitment, which is what
/// keeps the binding cross-query; a query opens only the pair it reads, via
/// [`edge_columns_for`].
pub fn edge_columns(edges: &[Edge]) -> Vec<Vec<u64>> {
    let mut cols = edge_columns_for("gq2", edges); // raw
    cols.extend(edge_columns_for("gq1", edges)); // shifted
    cols
}

/// The two committed columns `query` actually witnesses, in row order.
pub fn edge_columns_for(query: &str, edges: &[Edge]) -> Vec<Vec<u64>> {
    // GQ2 alone keeps the base relation unshifted; see `edge_columns`.
    let shift: u64 = if query == "gq2" { 0 } else { 1 };
    vec![
        edges.iter().map(|e| e.src + shift).collect(),
        edges.iter().map(|e| e.dst + shift).collect(),
    ]
}


/// GQ1 expected count: 3-path a->b->c->d with a<b, b<c (multiplicity aware).
/// Mirrors `g_sql1_obj::tests::dp_expected`.
pub fn count_gq1(edges: &[Edge]) -> u64 {
    let mut outdeg: HashMap<u64, u64> = HashMap::new();
    for e in edges {
        *outdeg.entry(e.src).or_insert(0) += 1;
    }
    let mut t2: HashMap<u64, u128> = HashMap::new();
    for e in edges {
        if e.src < e.dst {
            let v = *outdeg.get(&e.dst).unwrap_or(&0) as u128;
            *t2.entry(e.src).or_insert(0) += v;
        }
    }
    let mut ans: u128 = 0;
    for e in edges {
        if e.src < e.dst {
            ans += *t2.get(&e.dst).unwrap_or(&0);
        }
    }
    ans as u64
}

/// GQ2 expected count: 4-path with a<b<c<d ordering constraints.
/// Mirrors `g_sql2_obj::tests::dp_count_path4_order`.
pub fn count_gq2(edges: &[Edge]) -> u64 {
    let mut t4: HashMap<u64, u64> = HashMap::new();
    for e in edges {
        *t4.entry(e.src).or_insert(0) += 1;
    }
    let mut t3: HashMap<u64, u64> = HashMap::new();
    for e in edges {
        if e.src < e.dst {
            let v = *t4.get(&e.dst).unwrap_or(&0);
            *t3.entry(e.src).or_insert(0) += v;
        }
    }
    let mut t2: HashMap<u64, u64> = HashMap::new();
    for e in edges {
        if e.src < e.dst {
            let v = *t3.get(&e.dst).unwrap_or(&0);
            *t2.entry(e.src).or_insert(0) += v;
        }
    }
    let mut ans: u128 = 0;
    for e in edges {
        if e.src < e.dst {
            ans += *t2.get(&e.dst).unwrap_or(&0) as u128;
        }
    }
    ans as u64
}

fn edge_multiplicity(edges: &[Edge]) -> (HashMap<(u64, u64), u64>, HashMap<u64, Vec<(u64, u64)>>) {
    let mut cnt: HashMap<(u64, u64), u64> = HashMap::new();
    for e in edges {
        *cnt.entry((e.src, e.dst)).or_default() += 1;
    }
    let mut out: HashMap<u64, Vec<(u64, u64)>> = HashMap::new();
    for (&(s, d), &c) in cnt.iter() {
        out.entry(s).or_default().push((d, c));
    }
    (cnt, out)
}

/// GQ3 expected count: triangles a<b<c.  Mirrors `g_sql3_obj::tests::expected_cnt`.
pub fn count_gq3(edges: &[Edge]) -> u64 {
    let (cnt, out) = edge_multiplicity(edges);
    let mut total: u128 = 0;
    for (&a, outs_ab) in out.iter() {
        for &(b, cab) in outs_ab.iter() {
            if a >= b {
                continue;
            }
            if let Some(outs_bc) = out.get(&b) {
                for &(c, cbc) in outs_bc.iter() {
                    if b >= c {
                        continue;
                    }
                    if let Some(&cca) = cnt.get(&(c, a)) {
                        total += (cab as u128) * (cbc as u128) * (cca as u128);
                    }
                }
            }
        }
    }
    total as u64
}

/// GQ4 expected count: 4-cycles a<b<c<d.  Mirrors `g_sql4_obj::tests::expected_cnt`.
pub fn count_gq4(edges: &[Edge]) -> u64 {
    let (cnt, out) = edge_multiplicity(edges);
    let mut total: u128 = 0;
    for (&a, outs_ab) in out.iter() {
        for &(b, cab) in outs_ab.iter() {
            if a >= b {
                continue;
            }
            let Some(outs_bc) = out.get(&b) else { continue };
            for &(c, cbc) in outs_bc.iter() {
                if b >= c {
                    continue;
                }
                let Some(outs_cd) = out.get(&c) else { continue };
                for &(d, ccd) in outs_cd.iter() {
                    if c >= d {
                        continue;
                    }
                    if let Some(&cda) = cnt.get(&(d, a)) {
                        total += (cab as u128) * (cbc as u128) * (ccd as u128) * (cda as u128);
                    }
                }
            }
        }
    }
    total as u64
}

pub fn declared_degree_cap(dataset: &str) -> Option<u64> {
    match dataset {
        "lastfm" => Some(384),

        "facebook" | "wiki" => Some(2048),
        _ => None,
    }
}

fn cyclic_pad_extra(dataset: &str) -> usize {
    match dataset {
        "lastfm" => 18_067,
        "facebook" => 92_827,
        "wiki" => 79_427,
        _ => 0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Privacy {
    /// Capacity = the true bag size.  Leaks the exact intermediate
    /// cardinality; this is the paper's "Revealing Join Size" lower bound.
    Rjs,
    /// The hand-picked constants used for the submitted results
    /// (`dp/legacy_capacities.md`).  GQ3 and GQ4 share them.
    Legacy,

    Oblivious,

    Dp {
        epsilon: f64,
        delta: f64,
    },
}

pub fn oblivious_wedge_bound(_dataset: &str, n_edges: usize) -> u64 {
    let m = n_edges as u64;
    let quadratic = (m / 2) * (m - m / 2);

    let tau_pub: Option<u64> = std::env::var("VPJOIN_TAU_PUB")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&t| t > 0);
    match tau_pub {
        Some(tau) => quadratic.min(tau.saturating_mul(m)),
        None => quadratic,
    }
}

impl Privacy {
    pub fn label(&self) -> String {
        match self {
            Privacy::Rjs => "rjs".to_string(),
            Privacy::Oblivious => "oblivious".to_string(),
            Privacy::Legacy => "legacy-dp".to_string(),
            Privacy::Dp { epsilon, delta } => format!("dp(eps={} del={})", epsilon, delta),
        }
    }
}

pub fn q5_pads(privacy: Privacy) -> (usize, usize, usize) {
    match privacy {
        Privacy::Rjs => (0, 0, 0),
        // Values used for the DP results reported in the paper; see
        // `dp/legacy_capacities.md` and the note in `q5_obj.rs`.
        Privacy::Legacy => (0, 2848, 59452),
        // Fully oblivious: each materialized bag padded to the product of the
        // public cardinalities of the relations feeding it, which is the same
        // worst-case rule `letter.tex` applies to the PoneglyphDB estimate.
        // The NR bag is the fixed TPC-H nation/region catalogue (25 x 5),
        // public dimension data, so it needs no pad.
        Privacy::Oblivious => {
            let TpchInput::Q5 {
                customer,
                orders,
                lineitem,
                supplier,
                ..
            } = q5_raw_cached()
            else {
                unreachable!("q5_raw_cached() returns Q5")
            };
            let co_bound = (customer.len() as u64).saturating_mul(orders.len() as u64);
            let ls_bound = (lineitem.len() as u64).saturating_mul(supplier.len() as u64);
            (
                0,
                co_bound.saturating_sub(orders.len() as u64) as usize,
                ls_bound.saturating_sub(q5_ls_true() as u64) as usize,
            )
        }
        Privacy::Dp { epsilon, delta } => {
            let (nr, co, ls) = match q5_raw_cached() {
                TpchInput::Q5 {
                    customer,
                    orders,
                    lineitem,
                    start_ts,
                    end_ts,
                    ..
                } => {
                    // Bag sizes come from the circuit's OWN derivation, so the
                    // released capacity is calibrated to the size the circuit
                    // will actually materialize.
                    let ls_true = q5_ls_true() as u64;

                    let max_key_freq = |rows: &[Vec<u64>], idx: usize| -> u64 {
                        let mut m = std::collections::HashMap::new();
                        for r in rows.iter() {
                            *m.entry(r[idx]).or_insert(0u64) += 1;
                        }
                        m.values().copied().max().unwrap_or(0)
                    };
                    let tau_c = max_key_freq(orders, 1); // max orders per custkey
                    let tau_s = max_key_freq(lineitem, 1); // max lineitems per suppkey

                    // Cross-channel bound: max, over custkeys, of the
                    // number of lineitem rows whose order lies in the date
                    // window and belongs to that custkey.  Orders and
                    // lineitem are both unprotected, so this too is exact.
                    let mut okey_cust = std::collections::HashMap::new();
                    for o in orders.iter() {
                        if o[0] >= *start_ts && o[0] < *end_ts {
                            okey_cust.insert(o[2], o[1]);
                        }
                    }
                    let mut cust_lines = std::collections::HashMap::new();
                    for l in lineitem.iter() {
                        if let Some(c) = okey_cust.get(&l[0]) {
                            *cust_lines.entry(*c).or_insert(0u64) += 1;
                        }
                    }
                    let delta_ls_cust = cust_lines.values().copied().max().unwrap_or(0);

                    // Bag 1 {O,C} is laid out positionally, one slot per
                    // orders row, so its release provisions the rows BEYOND
                    // that base: matches past the first for one order.
                    // Their true count is measured under bag semantics
                    // (0 on keyed TPC-H data; nothing assumes it).
                    let mut cust_count = std::collections::HashMap::new();
                    for c in customer.iter() {
                        *cust_count.entry(c[0]).or_insert(0u64) += 1;
                    }
                    let co_extra_true: u64 = orders
                        .iter()
                        .filter(|o| o[0] >= *start_ts && o[0] < *end_ts)
                        .map(|o| {
                            cust_count
                                .get(&o[1])
                                .copied()
                                .unwrap_or(0)
                                .saturating_sub(1)
                        })
                        .sum();

                    // NOTE (benchmarking artifact): dp_rng is publicly
                    // seeded for reproducibility; a deployment must draw
                    // this noise from secret entropy.
                    let mut rng = dp_rng("q5", "tpch-60K");
                    let ls_sens = tau_s.max(delta_ls_cust);
                    let eps_ls = epsilon;
                    let eps_co = epsilon * (1.0 - delta_ls_cust as f64 / ls_sens as f64);
                    let co_cap = crate::dp_noise::dp_join_capacity_unprotected_tau(
                        co_extra_true,
                        tau_c,
                        eps_co,
                        delta / 2.0,
                        &mut rng,
                    );
                    let co_pad = co_cap.capacity as usize;

                    let ls_cap = crate::dp_noise::dp_join_capacity_unprotected_tau(
                        ls_true,
                        ls_sens,
                        eps_ls,
                        delta / 2.0,
                        &mut rng,
                    );
                    // No clamp: the release is valid as-is.  The DP lane
                    // circuit hosts the capacity in
                    // ceil(cap / q5_obj_dp::LANE_ROWS) parallel column-group
                    // lanes at fixed k = 17 and asserts its own structural
                    // maximum; the single-column circuit instead takes the
                    // degree bump computed in `degree_for`.
                    let ls_pad = ls_cap.capacity.saturating_sub(ls_true) as usize;
                    // stderr, like the `[params]` load lines: the released
                    // capacities also appear as columns in every harness's
                    // result table, so keeping the narration off stdout lets
                    // that table stay contiguous when it is piped or pasted.
                    eprintln!(
                        "  [dp] q5 (P = {{customer, supplier}}, row-level): \
                         tau_c={} tau_s={} delta_ls_cust={} (exact, 0-sensitive \
                         w.r.t. P); bag1 {{O,C}} extra true {} cap +{} on |O|={} \
                         (eps_co={:.4}), bag2 {{L,S}} true {} cap {} (+{} pad) \
                         (eps_ls={}, sens {}); delta/2 each",
                        tau_c,
                        tau_s,
                        delta_ls_cust,
                        co_extra_true,
                        co_pad,
                        orders.len(),
                        eps_co,
                        ls_true,
                        ls_cap.capacity,
                        ls_pad,
                        eps_ls,
                        ls_sens,
                    );
                    (0, co_pad, ls_pad)
                }
                _ => unreachable!("tpch_inputs_raw(\"q5\") returns Q5"),
            };
            (nr, co, ls)
        }
    }
}

pub struct BagStats {
    pub bag1_size: u64,
    bag1_mf_a: u64,
    bag1_mf_b: u64,
    pub bag2_size: u64,
    bag2_mf_a: u64,
    bag2_mf_b: u64,
    /// bag2 is a base relation whose size is public (GQ3), so no DP is owed.
    bag2_is_public: bool,
    /// The two bags are the same relation read with different column roles
    /// (GQ4), so their cardinalities are ONE statistic: one release covers
    /// both, and both get the same capacity.
    bags_are_one_statistic: bool,
}

pub fn bag_stats(query: &str, edges: &[Edge]) -> BagStats {
    let mut indeg: HashMap<u64, u64> = HashMap::new();
    let mut outdeg: HashMap<u64, u64> = HashMap::new();
    // in-degree counting only a<b edges, which is the filter GQ3 applies.
    let mut indeg_lt: HashMap<u64, u64> = HashMap::new();
    for e in edges {
        *indeg.entry(e.dst).or_insert(0) += 1;
        *outdeg.entry(e.src).or_insert(0) += 1;
        if e.src < e.dst {
            *indeg_lt.entry(e.dst).or_insert(0) += 1;
        }
    }
    let max_of = |m: &HashMap<u64, u64>| m.values().copied().max().unwrap_or(0);
    let wedge = |ind: &HashMap<u64, u64>| -> u64 {
        ind.iter()
            .map(|(v, &i)| i * outdeg.get(v).copied().unwrap_or(0))
            .sum()
    };

    match query {
        "gq3" => BagStats {
            bag1_size: wedge(&indeg_lt),
            bag1_mf_a: max_of(&indeg_lt),
            bag1_mf_b: max_of(&outdeg),
            bag2_size: edges.len() as u64,
            bag2_mf_a: 1,
            bag2_mf_b: 1,
            bag2_is_public: true,
            bags_are_one_statistic: false,
        },

        "gq4" => {
            let n = wedge(&indeg_lt);
            BagStats {
                bag1_size: n,
                bag1_mf_a: max_of(&indeg_lt),
                bag1_mf_b: max_of(&outdeg),
                bag2_size: n,
                bag2_mf_a: max_of(&indeg_lt),
                bag2_mf_b: max_of(&outdeg),
                bag2_is_public: false,
                // same relation, different column roles: see the budget note
                // in `graph_pads`
                bags_are_one_statistic: true,
            }
        }
        other => panic!("no bag model for {}", other),
    }
}

/// Deterministic RNG for capacity release, so a benchmark re-run reproduces
/// the same circuit sizes.  Override the stream with `VPJOIN_DP_SEED`.
///
/// A real deployment MUST draw fresh randomness per release; a fixed seed is
/// only appropriate for reproducing a measurement.
fn dp_rng(query: &str, dataset: &str) -> rand_xorshift::XorShiftRng {
    use rand::SeedableRng;
    let base: u64 = std::env::var("VPJOIN_DP_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_260_721);
    let mut h = base;
    for b in query.bytes().chain(dataset.bytes()) {
        h = h.wrapping_mul(0x100000001b3) ^ (b as u64);
    }
    let mut seed = [0u8; 16];
    seed[..8].copy_from_slice(&h.to_le_bytes());
    seed[8..].copy_from_slice(&h.rotate_left(17).to_le_bytes());
    rand_xorshift::XorShiftRng::from_seed(seed)
}

/// `(bag1_pad_extra, bag2_pad_extra)` for a cyclic graph query.
///
/// The pair shape is kept because GQ3, Q5, `commit_diff.rs`,
/// `tests/freq_noising_cost.rs` and `tests/graph_lane_plan.rs` all consume two
/// pads.  For GQ4 the two entries are always equal (one materialized relation,
/// one released capacity) and its consumers read `.0`.
pub fn graph_pads(query: &str, dataset: &str, edges: &[Edge], privacy: Privacy) -> (usize, usize) {
    match privacy {
        Privacy::Rjs => (0, 0),
        Privacy::Legacy => {
            // Same constants for GQ3 and GQ4, applied to both bags.
            let p = cyclic_pad_extra(dataset);
            (p, p)
        }

        Privacy::Oblivious => {
            let s = bag_stats(query, edges);
            let bound = oblivious_wedge_bound(dataset, edges.len());
            let pad_of = |true_size: u64| bound.saturating_sub(true_size) as usize;
            let p1 = pad_of(s.bag1_size);
            let p2 = if s.bag2_is_public {
                0
            } else {
                pad_of(s.bag2_size)
            };
            (p1, p2)
        }
        Privacy::Dp { epsilon, delta } => {
            let s = bag_stats(query, edges);
            let mut rng = dp_rng(query, dataset);

            let n_size = if s.bag2_is_public || s.bags_are_one_statistic {
                1
            } else {
                2
            };

            let tau_pub: Option<u64> = match std::env::var("VPJOIN_TAU_PUB") {
                Ok(v) => v.parse().ok().filter(|&t| t > 0),
                Err(_) => declared_degree_cap(dataset),
            };
            let n_release = n_size + usize::from(tau_pub.is_none());
            let (eps, del) = (epsilon / n_release as f64, delta / n_release as f64);

            // The bound must cover BOTH degree statistics the sensitivity is
            // built from, so it is taken over their maximum.
            let max_deg = s
                .bag1_mf_a
                .max(s.bag1_mf_b)
                .max(s.bag2_mf_a)
                .max(s.bag2_mf_b);
            let (tau, tau_shown, released) =
                match tau_pub {
                    Some(t) => {
                        assert!(
                        t >= max_deg,
                        "VPJOIN_TAU_PUB={} is below the data's maximum degree {} on {}/{}: the \
                         noise would be calibrated to a sensitivity smaller than the true one and \
                         the reported epsilon would be void. A deployment must TRUNCATE the \
                         relation to the declared cap, which changes the answer; this harness \
                         refuses instead of reporting a guarantee it does not provide.",
                        t, max_deg, query, dataset
                    );
                        (2 * t, t as f64, false)
                    }
                    None => {
                        let tt = crate::dp_noise::dp_frequency_bound(max_deg, eps, del, &mut rng);
                        ((2.0 * tt).ceil() as u64, tt, true)
                    }
                };

            let c1 =
                crate::dp_noise::dp_join_capacity_public_tau(s.bag1_size, tau, eps, del, &mut rng);
            let pad1 = c1.capacity.saturating_sub(s.bag1_size) as usize;

            let pad2 = if s.bag2_is_public {
                0
            } else if s.bags_are_one_statistic {
                // one statistic, one release, one capacity
                pad1
            } else {
                let c2 = crate::dp_noise::dp_join_capacity_public_tau(
                    s.bag2_size,
                    tau,
                    eps,
                    del,
                    &mut rng,
                );
                c2.capacity.saturating_sub(s.bag2_size) as usize
            };
            // stderr, for the reason given in `q5_pads`.
            eprintln!(
                "  [dp] {} on {}: max_deg {} -> {} tau {:.0} (sens {}, {} release{}), \
                 bag1 {} (+{} pad), bag2 {} (+{} pad{})",
                query,
                dataset,
                max_deg,
                if released {
                    "released"
                } else {
                    "PUBLIC (declared, no release spent)"
                },
                tau_shown,
                tau,
                n_release,
                if n_release == 1 { "" } else { "s" },
                s.bag1_size,
                pad1,
                s.bag2_size,
                pad2,
                if s.bag2_is_public {
                    ", public size"
                } else {
                    ""
                }
            );
            (pad1, pad2)
        }
    }
}

pub fn graph_rows(edges: &[Edge], query: &str, pad1: usize, pad2: usize) -> u64 {
    let s = bag_stats(query, edges);
    let n_base = edges.len() as u64 + 1;
    let n2 = s.bag2_size + pad2 as u64;
    let n1 = s.bag1_size + pad1 as u64;
    (n_base + 1).max(n2 + 1).max(n1)
}

pub fn ceil_log2(n: u64) -> u32 {
    let mut k = 1u32;
    while (1u64 << k) < n {
        k += 1;
    }
    k
}

/// Degree needed by a cyclic query under a given privacy regime.  Computed
/// from the actual bag sizes -- arithmetic, not a search.
pub fn graph_degree(query: &str, dataset: &str, privacy: Privacy) -> u32 {
    let edges = load_graph(dataset);
    let (p1, p2) = graph_pads(query, dataset, &edges, privacy);
    // Halo2 reserves a handful of unusable rows for blinding.
    ceil_log2(graph_rows(&edges, query, p1, p2) + 64)
}

fn run_graph(query: &str, dataset: &str, mode: Mode, privacy: Privacy) -> Row {
    use crate::graph_sql::{g_sql1_obj, g_sql2_obj, g_sql3_obj, g_sql4_obj};
    use std::marker::PhantomData;

    let t_all = Instant::now();
    let t_load = Instant::now();
    let edges = load_graph(dataset);
    let load_s = t_load.elapsed().as_secs_f64();
    let mut row = Row {
        query: query.to_string(),
        dataset: dataset.to_string(),
        input_rows: edges.len(),
        profile: build_profile(),
        load_s,
        ..Default::default()
    };

    let cols = edge_columns_for(query, &edges);
    // Built before `edges` is moved into the circuit below.
    let db_cols = edge_columns(&edges);
    let edges_for_tie = edges.clone();
    let label = format!("{}_{}", query, dataset);
    // Bag capacities and hence the degree follow from the privacy regime.
    let (pad1, pad2) = if matches!(query, "gq3" | "gq4") {
        graph_pads(query, dataset, &edges, privacy)
    } else {
        (0, 0)
    };
    let k = match query {
        // Computed here rather than via `degree_for` so the pads are not
        // derived twice; `forced_k` is applied on this branch only, since
        // `degree_for` already applies it on the other.
        "gq3" | "gq4" => forced_k(ceil_log2(graph_rows(&edges, query, pad1, pad2) + 64)),
        _ => degree_for(query, dataset, privacy),
    };
    row.config = if matches!(query, "gq3" | "gq4") {
        privacy.label()
    } else {
        "oblivious".to_string()
    };

    let (cnt, timed) = match query {
        "gq1" => {
            let cnt = count_gq1(&edges);
            let ins = [Fp::from(cnt)];
            let c = g_sql1_obj::Path3OrdCircuit::<Fp> {
                edges,
                _marker: PhantomData,
            };
            let timed = maybe_run(mode, &label, &c, &ins, k);
            (cnt, timed)
        }
        "gq2" => {
            let cnt = count_gq2(&edges);
            let ins = [Fp::from(cnt)];
            let c = g_sql2_obj::GraphPath4OrderCircuit::<Fp> {
                edges,
                _marker: PhantomData,
            };
            let timed = maybe_run(mode, &label, &c, &ins, k);
            (cnt, timed)
        }
        "gq3" => {
            let cnt = count_gq3(&edges);
            let ins = [Fp::from(cnt)];
            let c = g_sql3_obj::MyCircuit::<Fp> {
                edges,
                bag1_pad_extra: pad1,
                bag2_pad_extra: pad2,
                _marker: PhantomData,
            };
            let timed = maybe_run(mode, &label, &c, &ins, k);
            (cnt, timed)
        }
        "gq4" => {
            let cnt = count_gq4(&edges);
            let ins = [Fp::from(cnt)];
            // ONE bag, read in two column roles, so ONE pad knob. `graph_pads`
            // returns (pad1, pad1) for gq4 and the consumer reads .0 only.
            let c = g_sql4_obj::MyCircuit::<Fp> {
                edges,
                pad_extra: pad1,
                _marker: PhantomData,
            };
            let timed = maybe_run(mode, &label, &c, &ins, k);
            (cnt, timed)
        }
        other => panic!("unknown graph query {}", other),
    };

    row.public_output = cnt;
    row.k = timed.k;
    row.vk_s = timed.vk_s;
    row.pk_s = timed.pk_s;
    row.keygen_s = timed.keygen_s;
    row.prove_s = timed.prove_s;
    row.verify_s = timed.verify_s;
    row.proof_bytes = timed.proof_bytes;
    row.n_columns = cols.len();

    if mode == Mode::Full || mode == Mode::Commit {
        // Only the commit-only measurement binds to a prior proof; `full`
        // derives its challenge without one.
        let bound_to = if mode == Mode::Commit {
            saved_proof(&label)
        } else {
            Vec::new()
        };
        // The publication carries BOTH encodings the graph queries witness;
        // this query opens the pair it actually reads. In `full` the binding is
        // proved TIED to this query's witness; `commit` measures the additive
        // layer alone and deliberately does not re-prove the query.
        let tied = if mode == Mode::Full {
            Tied::Graph {
                query,
                edges: &edges_for_tie,
                pads: (pad1, pad2),
                cnt,
            }
        } else {
            Tied::None
        };
        apply_commitment_layer(dataset, &db_cols, &cols, timed.k, &bound_to, &mut row, tied);
    }
    finish(&mut row, t_all.elapsed().as_secs_f64());
    row
}

fn finish(row: &mut Row, wall_s: f64) {
    row.total_prove_s = row.prove_s + row.bind_prove_s + row.open_prove_s;
    row.total_proof_bytes = row.proof_bytes + row.bind_proof_bytes + row.open_bytes;
    row.wall_s = wall_s;
}

// ---------------------------------------------------------------------------
// TPC-H queries
// ---------------------------------------------------------------------------

pub fn string_to_u64(s: &str) -> u64 {
    let mut result = 0u64;
    for (i, c) in s.chars().enumerate() {
        result += (i as u64 + 1) * (c as u64);
    }
    result
}

pub fn scale_by_1000(x: f64) -> u64 {
    (1000.0 * x) as u64
}

pub fn date_to_timestamp(date_str: &str) -> u64 {
    use chrono::{DateTime, NaiveDate, Utc};
    match NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
        Ok(date) => {
            let datetime: DateTime<Utc> = DateTime::<Utc>::from_utc(date.and_hms(0, 0, 0), Utc);
            datetime.timestamp() as u64
        }
        Err(_) => 0,
    }
}

/// q8 trims before hashing; q5/q9/q18 do not.  The two are NOT interchangeable
/// (they disagree on whitespace-padded fields), so both are kept.
pub fn string_to_u64_trim(s: &str) -> u64 {
    string_to_u64(s.trim())
}

pub fn year_from_date(date_str: &str) -> u64 {
    use chrono::{Datelike, NaiveDate};
    match NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
        Ok(d) => d.year() as u64,
        Err(_) => 0,
    }
}

/// Mirrors the private `q9_obj::PS_SHIFT`.
const PS_SHIFT: u64 = 1u64 << 20;

/// Row-major table -> one vector per attribute, which is exactly the set of
/// source columns that circuit witnesses.  Deriving the commitment layer's
/// columns this way (rather than from a separate hand-written index list)
/// guarantees the committed columns are the ones the query actually reads.
fn transpose(tables: &[&Vec<Vec<u64>>]) -> Vec<Vec<u64>> {
    let mut cols: Vec<Vec<u64>> = Vec::new();
    for t in tables {
        let width = t.first().map(|r| r.len()).unwrap_or(0);
        for j in 0..width {
            cols.push(t.iter().map(|r| r[j]).collect());
        }
    }
    cols
}

fn require(name: &str, t: &[Vec<u64>]) {
    assert!(
        !t.is_empty(),
        "table `{}` loaded as EMPTY -- check VPJOIN_DATA (currently {}). \
         The per-query tests swallow read errors with `if let Ok(..)`, which \
         silently proves over an empty database; this harness refuses to.",
        name,
        data_root().display()
    );
}

fn run_tpch(query: &str, dataset: &str, mode: Mode, privacy: Privacy) -> Row {
    let k = degree_for(query, dataset, privacy);
    use crate::data::data_processing as dp;
    use crate::sql::{q18_obj, q3_obj, q5_obj, q8_obj, q9_obj};
    use std::marker::PhantomData;

    let t_all = Instant::now();
    let mut row = Row {
        query: query.to_string(),
        dataset: dataset.to_string(),
        public_output: 1,
        profile: build_profile(),

        config: if query == "q5" {
            privacy.label()
        } else if is_graph(query) {
            match (privacy, declared_degree_cap(dataset)) {
                (Privacy::Dp { .. }, Some(t)) => format!("{}+tau{}", privacy.label(), t),
                _ => privacy.label(),
            }
        } else {
            "oblivious".to_string()
        },
        ..Default::default()
    };
    let one = [Fp::from(1u64)];
    let label = format!("{}_{}", query, dataset);
    // Set inside each arm once that query's tables are loaded and projected.
    let mut load_s = 0.0f64;

    // Shared loaders (each query selects only the attributes it witnesses).
    let customer = |proj: &dyn Fn(&dp::Customer) -> Vec<u64>| -> Vec<Vec<u64>> {
        dp::customer_read_records_from_file(&tbl("customer.tbl"))
            .map(|rs| rs.iter().map(proj).collect())
            .unwrap_or_default()
    };
    let orders = |proj: &dyn Fn(&dp::Orders) -> Vec<u64>| -> Vec<Vec<u64>> {
        dp::orders_read_records_from_file(&tbl("orders.tbl"))
            .map(|rs| rs.iter().map(proj).collect())
            .unwrap_or_default()
    };
    let lineitem = |proj: &dyn Fn(&dp::Lineitem) -> Vec<u64>| -> Vec<Vec<u64>> {
        dp::lineitem_read_records_from_file(&tbl("lineitem.tbl"))
            .map(|rs| rs.iter().map(proj).collect())
            .unwrap_or_default()
    };
    let supplier = |proj: &dyn Fn(&dp::Supplier) -> Vec<u64>| -> Vec<Vec<u64>> {
        dp::supplier_read_records_from_file(&tbl("supplier.tbl"))
            .map(|rs| rs.iter().map(proj).collect())
            .unwrap_or_default()
    };
    let nation = |proj: &dyn Fn(&dp::Nation) -> Vec<u64>| -> Vec<Vec<u64>> {
        dp::nation_read_records_from_file(&tbl("nation.tbl"))
            .map(|rs| rs.iter().map(proj).collect())
            .unwrap_or_default()
    };
    let region = |proj: &dyn Fn(&dp::Region) -> Vec<u64>| -> Vec<Vec<u64>> {
        dp::region_read_records_from_cvs(&tbl("region.cvs"))
            .map(|rs| rs.iter().map(proj).collect())
            .unwrap_or_default()
    };
    let part = |proj: &dyn Fn(&dp::Part) -> Vec<u64>| -> Vec<Vec<u64>> {
        dp::part_read_records_from_file(&tbl("part.tbl"))
            .map(|rs| rs.iter().map(proj).collect())
            .unwrap_or_default()
    };
    let partsupp = |proj: &dyn Fn(&dp::Partsupp) -> Vec<u64>| -> Vec<Vec<u64>> {
        dp::partsupp_read_records_from_file(&tbl("partsupp.tbl"))
            .map(|rs| rs.iter().map(proj).collect())
            .unwrap_or_default()
    };

    let (cols, timed) = match query {
        // ---------------- Q3 ----------------
        "q3" => {
            let c = customer(&|r| vec![string_to_u64(&r.c_mktsegment), r.c_custkey]);
            let o = orders(&|r| {
                vec![
                    date_to_timestamp(&r.o_orderdate),
                    r.o_shippriority,
                    r.o_custkey,
                    r.o_orderkey,
                ]
            });
            let l = lineitem(&|r| {
                vec![
                    r.l_orderkey,
                    scale_by_1000(r.l_extendedprice),
                    scale_by_1000(r.l_discount),
                    date_to_timestamp(&r.l_shipdate),
                ]
            });
            require("customer", &c);
            require("orders", &o);
            require("lineitem", &l);
            row.input_rows = c.len() + o.len() + l.len();
            let cols = transpose(&[&c, &o, &l]);
            (cols, {
                load_s = t_all.elapsed().as_secs_f64();
                let condition = [string_to_u64("HOUSEHOLD"), date_to_timestamp("1995-03-25")];
                // The One-Pass OBJ of `q3_obj.rs` certifies the split as one
                // selector bit per committed row instead of a materialized
                // partition.
                let circuit = q3_obj::MyCircuit::<Fp> {
                    customer: c,
                    orders: o,
                    lineitem: l,
                    condition,
                    _marker: PhantomData,
                };
                maybe_run(mode, &label, &circuit, &one, k)
            })
        }

        // ---------------- Q5 ----------------
        "q5" => {
            let c = customer(&|r| vec![r.c_custkey, r.c_nationkey]);
            let o = orders(&|r| vec![date_to_timestamp(&r.o_orderdate), r.o_custkey, r.o_orderkey]);
            let l = lineitem(&|r| {
                vec![
                    r.l_orderkey,
                    r.l_suppkey,
                    scale_by_1000(r.l_extendedprice),
                    scale_by_1000(r.l_discount),
                ]
            });
            let s = supplier(&|r| vec![r.s_suppkey, r.s_nationkey]);
            let n = nation(&|r| vec![r.n_nationkey, string_to_u64(&r.n_name), r.n_regionkey]);
            let rg = region(&|r| vec![r.r_regionkey, string_to_u64(&r.r_name)]);
            require("customer", &c);
            require("orders", &o);
            require("lineitem", &l);
            require("supplier", &s);
            require("nation", &n);
            require("region", &rg);
            row.input_rows = c.len() + o.len() + l.len() + s.len() + n.len() + rg.len();
            // The publication must carry the encoding the circuit witnesses:
            // Q5 shifts its five key columns by one (see the Q5 arm of
            // `TpchInput::columns`). Derive `cols` from `columns()` instead of
            // restating a raw transpose here -- publishing RAW keys made the
            // tied proof's public evaluations disagree with the binding on
            // exactly those five columns, and `column_indices` could not catch
            // it because Q8/Q9 publish the same raw keys, so the lookups
            // silently resolved to THEIR columns.
            let input = TpchInput::Q5 {
                customer: c,
                orders: o,
                lineitem: l,
                supplier: s,
                nation: n,
                region: rg,
                europe_hash: string_to_u64("EUROPE"),
                start_ts: date_to_timestamp("1997-01-01"),
                end_ts: date_to_timestamp("1998-01-01"),
                // Same source of truth as commit_diff: see `q5_pads`.
                nr_pad_extra: q5_pads(privacy).0,
                co_pad_extra: q5_pads(privacy).1,
                ls_pad_extra: q5_pads(privacy).2,
            };
            let cols = input.columns();
            (cols, {
                load_s = t_all.elapsed().as_secs_f64();
                let TpchInput::Q5 {
                    customer,
                    orders,
                    lineitem,
                    supplier,
                    nation,
                    region,
                    europe_hash,
                    start_ts,
                    end_ts,
                    nr_pad_extra,
                    co_pad_extra,
                    ls_pad_extra,
                } = input
                else {
                    unreachable!()
                };
                // The One-Pass gate keeps every capacity `q5_pads` returns, so
                // the padding layer is unchanged by the realization.
                let circuit = q5_obj::MyCircuit::<Fp> {
                    customer,
                    orders,
                    lineitem,
                    supplier,
                    nation,
                    region,
                    europe_hash,
                    start_ts,
                    end_ts,
                    nr_pad_extra,
                    co_pad_extra,
                    ls_pad_extra,
                    _marker: PhantomData,
                };
                maybe_run(mode, &label, &circuit, &one, k)
            })
        }

        // ---------------- Q8 ----------------
        "q8" => {
            let rg = region(&|r| vec![r.r_regionkey, string_to_u64_trim(&r.r_name)]);
            let n = nation(&|r| vec![r.n_nationkey, r.n_regionkey, string_to_u64_trim(&r.n_name)]);
            let c = customer(&|r| vec![r.c_custkey, r.c_nationkey]);
            let o = orders(&|r| vec![r.o_orderkey, r.o_custkey, year_from_date(&r.o_orderdate)]);
            let p = part(&|r| vec![r.p_partkey, string_to_u64_trim(&r.p_type)]);
            let s = supplier(&|r| vec![r.s_suppkey, r.s_nationkey]);
            let l = lineitem(&|r| {
                vec![
                    r.l_orderkey,
                    r.l_partkey,
                    r.l_suppkey,
                    scale_by_1000(r.l_extendedprice),
                    scale_by_1000(r.l_discount),
                ]
            });
            require("region", &rg);
            require("nation", &n);
            require("customer", &c);
            require("orders", &o);
            require("part", &p);
            require("supplier", &s);
            require("lineitem", &l);
            row.input_rows = rg.len() + n.len() + c.len() + o.len() + p.len() + s.len() + l.len();
            let cols = transpose(&[&rg, &n, &c, &o, &p, &s, &l]);
            (cols, {
                load_s = t_all.elapsed().as_secs_f64();
                let circuit = q8_obj::MyCircuit::<Fp> {
                    region: rg,
                    nation: n,
                    customer: c,
                    orders: o,
                    part: p,
                    supplier: s,
                    lineitem: l,
                    cond_nation_hash: string_to_u64_trim("EGYPT"),
                    const_region_name_hash: string_to_u64_trim("MIDDLE EAST"),
                    const_part_type_hash: string_to_u64_trim("PROMO BRUSHED COPPER"),
                    _marker: PhantomData,
                };
                maybe_run(mode, &label, &circuit, &one, k)
            })
        }

        // ---------------- Q9 ----------------
        "q9" => {
            let p = part(&|r| vec![r.p_partkey, string_to_u64(&r.p_name)]);
            let s = supplier(&|r| vec![r.s_suppkey, r.s_nationkey]);
            let n = nation(&|r| vec![r.n_nationkey, string_to_u64(&r.n_name)]);
            let o = orders(&|r| vec![r.o_orderkey, year_from_date(&r.o_orderdate)]);
            let ps = partsupp(&|r| {
                vec![
                    r.ps_partkey * PS_SHIFT + r.ps_suppkey,
                    scale_by_1000(r.ps_supplycost),
                ]
            });
            let l = lineitem(&|r| {
                vec![
                    r.l_orderkey,
                    r.l_partkey,
                    r.l_suppkey,
                    r.l_quantity,
                    scale_by_1000(r.l_extendedprice),
                    scale_by_1000(r.l_discount),
                ]
            });
            require("part", &p);
            require("supplier", &s);
            require("nation", &n);
            require("orders", &o);
            require("partsupp", &ps);
            require("lineitem", &l);
            row.input_rows = p.len() + s.len() + n.len() + o.len() + ps.len() + l.len();
            let cols = transpose(&[&p, &s, &n, &o, &ps, &l]);
            (cols, {
                load_s = t_all.elapsed().as_secs_f64();
                let circuit = q9_obj::MyCircuit::<Fp> {
                    part: p,
                    supplier: s,
                    nation: n,
                    orders: o,
                    partsupp: ps,
                    lineitem: l,
                    cond_hash: string_to_u64("green"),
                    _marker: PhantomData,
                };
                maybe_run(mode, &label, &circuit, &one, k)
            })
        }

        // ---------------- Q18 ----------------
        "q18" => {
            let c = customer(&|r| vec![string_to_u64(&r.c_name), r.c_custkey]);
            let o = orders(&|r| {
                vec![
                    r.o_orderkey,
                    r.o_custkey,
                    date_to_timestamp(&r.o_orderdate),
                    scale_by_1000(r.o_totalprice),
                ]
            });
            let l = lineitem(&|r| vec![r.l_orderkey, r.l_quantity]);
            require("customer", &c);
            require("orders", &o);
            require("lineitem", &l);
            row.input_rows = c.len() + o.len() + l.len();
            let cols = transpose(&[&c, &o, &l]);
            (cols, {
                load_s = t_all.elapsed().as_secs_f64();
                let circuit = q18_obj::MyCircuit::<Fp> {
                    customer: c,
                    orders: o,
                    lineitem: l,
                    threshold: 300,
                    _marker: PhantomData,
                };
                maybe_run(mode, &label, &circuit, &one, k)
            })
        }

        other => panic!("unknown TPC-H query {}", other),
    };

    row.load_s = load_s;
    row.k = timed.k;
    row.vk_s = timed.vk_s;
    row.pk_s = timed.pk_s;
    row.keygen_s = timed.keygen_s;
    row.prove_s = timed.prove_s;
    row.verify_s = timed.verify_s;
    row.proof_bytes = timed.proof_bytes;
    row.n_columns = cols.len();

    if mode == Mode::Full || mode == Mode::Commit {
        // Only the commit-only measurement binds to a prior proof; `full`
        // derives its challenge without one.
        let bound_to = if mode == Mode::Commit {
            saved_proof(&label)
        } else {
            Vec::new()
        };
        // Commit(D) covers every column of the TPC-H database that the
        // workload reads; this query opens the subset its circuit witnesses.
        let db_cols = tpch_database_columns(privacy);
        let tied_input;
        let tied = if mode == Mode::Full {
            tied_input = tpch_inputs(query, privacy);
            Tied::Tpch(&tied_input)
        } else {
            Tied::None
        };
        apply_commitment_layer(
            &tpch_label(),
            &db_cols,
            &cols,
            timed.k,
            &bound_to,
            &mut row,
            tied,
        );
    }
    finish(&mut row, t_all.elapsed().as_secs_f64());
    row
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub const ALL_QUERIES: &[&str] = &["q3", "q5", "q8", "q9", "q18", "gq1", "gq2", "gq3", "gq4"];

fn is_graph(q: &str) -> bool {
    q.starts_with("gq")
}

/// Run one (query, dataset) pair.  `dataset` is ignored for TPC-H queries.
///
/// A failure of one pair is recorded in the row's `status` rather than
/// aborting the whole sweep, so a long run still yields every other
/// measurement.
pub fn run_one(query: &str, dataset: &str, mode: Mode, privacy: Privacy) -> Row {
    let q = query.to_string();
    let d = dataset.to_string();
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if is_graph(query) {
            run_graph(query, dataset, mode, privacy)
        } else {
            run_tpch(query, dataset, mode, privacy)
        }
    }));
    match res {
        Ok(row) => row,
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic".to_string());
            let msg = msg.lines().next().unwrap_or("panic").replace(',', ";");
            eprintln!("  [skip] {} on {}: {}", q, d, msg);
            Row {
                query: q,
                dataset: d,
                status: format!("FAILED: {}", msg),
                ..Default::default()
            }
        }
    }
}

/// Expand a query list into the (query, dataset) pairs to run.
pub fn plan(queries: &[String]) -> Vec<(String, String)> {
    let specs: Vec<(String, Option<String>)> = queries.iter().map(|q| (q.clone(), None)).collect();
    plan_specs(&specs)
}

pub fn plan_specs(specs: &[(String, Option<String>)]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for (q, ds) in specs {
        let datasets: Vec<String> = match ds {
            Some(d) => vec![d.clone()],
            None if is_graph(q) => GRAPH_DATASETS.iter().map(|d| d.to_string()).collect(),
            None => vec![tpch_label()],
        };
        for d in datasets {
            let pair = (q.clone(), d);
            if !out.contains(&pair) {
                out.push(pair);
            }
        }
    }
    out
}

pub fn tpch_label() -> String {
    std::env::var("VPJOIN_LABEL").unwrap_or_else(|_| "tpch-60K".to_string())
}

#[derive(Clone)]
pub enum TpchInput {
    Q3 {
        customer: Vec<Vec<u64>>,
        orders: Vec<Vec<u64>>,
        lineitem: Vec<Vec<u64>>,
        condition: [u64; 2],
    },
    Q5 {
        customer: Vec<Vec<u64>>,
        orders: Vec<Vec<u64>>,
        lineitem: Vec<Vec<u64>>,
        supplier: Vec<Vec<u64>>,
        nation: Vec<Vec<u64>>,
        region: Vec<Vec<u64>>,
        europe_hash: u64,
        start_ts: u64,
        end_ts: u64,
        nr_pad_extra: usize,
        co_pad_extra: usize,
        ls_pad_extra: usize,
    },
    Q8 {
        region: Vec<Vec<u64>>,
        nation: Vec<Vec<u64>>,
        customer: Vec<Vec<u64>>,
        orders: Vec<Vec<u64>>,
        part: Vec<Vec<u64>>,
        supplier: Vec<Vec<u64>>,
        lineitem: Vec<Vec<u64>>,
        cond_nation_hash: u64,
        const_region_name_hash: u64,
        const_part_type_hash: u64,
    },
    Q9 {
        part: Vec<Vec<u64>>,
        supplier: Vec<Vec<u64>>,
        nation: Vec<Vec<u64>>,
        orders: Vec<Vec<u64>>,
        partsupp: Vec<Vec<u64>>,
        lineitem: Vec<Vec<u64>>,
        cond_hash: u64,
    },
    Q18 {
        customer: Vec<Vec<u64>>,
        orders: Vec<Vec<u64>>,
        lineitem: Vec<Vec<u64>>,
        threshold: u64,
    },
}

/// Load one TPC-H query's inputs, with the same projections the query tests use.
///
/// `privacy` only affects Q5, the sole TPC-H query that materializes
/// intermediates (see [`q5_pads`]); the others ignore it.
pub fn tpch_inputs(query: &str, privacy: Privacy) -> TpchInput {
    let (nr_pad_extra, co_pad_extra, ls_pad_extra) = if query == "q5" {
        q5_pads(privacy)
    } else {
        (0, 0, 0)
    };
    tpch_inputs_padded(query, nr_pad_extra, co_pad_extra, ls_pad_extra)
}

/// Q5's inputs with zero padding, memoized for the life of the process.
///
/// The projections do not depend on the pads at all (the circuit stores them as
/// separate fields and pads internally), so one parse serves every privacy
/// regime and every epsilon of a sweep.  Used by the DP path to read the true
/// bag sizes without recursing through [`q5_pads`].
fn q5_raw_cached() -> &'static TpchInput {
    static CACHE: std::sync::OnceLock<TpchInput> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| tpch_inputs_uncached("q5", 0, 0, 0))
}

/// Row count of the active `lineitem.tbl` (whatever `VPJOIN_DATA` points at),
/// memoized.  Drives the degree of the acyclic TPC-H queries, whose circuit
/// height is dominated by the lineitem section.
fn lineitem_rows() -> usize {
    use std::io::BufRead;
    static CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::fs::File::open(tbl("lineitem.tbl"))
            .map(|f| std::io::BufReader::new(f).lines().count())
            .unwrap_or(0)
    })
}

/// Size of Q5's LS join at zero padding, memoized: it is derived from the
/// tables alone, and both `q5_pads` and the DP lane plan need it.
pub fn q5_ls_true() -> usize {
    static CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        let TpchInput::Q5 {
            customer,
            orders,
            lineitem,
            supplier,
            nation,
            region,
            europe_hash,
            start_ts,
            end_ts,
            ..
        } = q5_raw_cached()
        else {
            unreachable!("q5_raw_cached() returns Q5")
        };
        crate::sql::q5_obj::q5_derive(
            customer,
            orders,
            lineitem,
            supplier,
            nation,
            region,
            *europe_hash,
            *start_ts,
            *end_ts,
            0,
            0,
        )
        .ls_join_u64
        .len()
    })
}

fn tpch_inputs_padded(
    query: &str,
    nr_pad_extra: usize,
    co_pad_extra: usize,
    ls_pad_extra: usize,
) -> TpchInput {
    // Q5 is the query a sweep re-requests per epsilon; serve it from the cache
    // and just stamp the pads, which are plain fields of the returned struct.
    if query == "q5" {
        let mut out = q5_raw_cached().clone();
        if let TpchInput::Q5 {
            nr_pad_extra: nr,
            co_pad_extra: co,
            ls_pad_extra: ls,
            ..
        } = &mut out
        {
            *nr = nr_pad_extra;
            *co = co_pad_extra;
            *ls = ls_pad_extra;
        }
        return out;
    }
    tpch_inputs_uncached(query, nr_pad_extra, co_pad_extra, ls_pad_extra)
}

fn tpch_inputs_uncached(
    query: &str,
    nr_pad_extra: usize,
    co_pad_extra: usize,
    ls_pad_extra: usize,
) -> TpchInput {
    use crate::data::data_processing as dp;
    let load = |name: &str| tbl(name);

    macro_rules! proj {
        ($reader:path, $file:expr, $f:expr) => {
            $reader(&load($file))
                .map(|rs| rs.iter().map($f).collect::<Vec<Vec<u64>>>())
                .unwrap_or_default()
        };
    }

    match query {
        "q3" => TpchInput::Q3 {
            customer: proj!(
                dp::customer_read_records_from_file,
                "customer.tbl",
                |r| vec![string_to_u64(&r.c_mktsegment), r.c_custkey]
            ),
            orders: proj!(dp::orders_read_records_from_file, "orders.tbl", |r| vec![
                date_to_timestamp(&r.o_orderdate),
                r.o_shippriority,
                r.o_custkey,
                r.o_orderkey
            ]),
            lineitem: proj!(
                dp::lineitem_read_records_from_file,
                "lineitem.tbl",
                |r| vec![
                    r.l_orderkey,
                    scale_by_1000(r.l_extendedprice),
                    scale_by_1000(r.l_discount),
                    date_to_timestamp(&r.l_shipdate)
                ]
            ),
            condition: [string_to_u64("HOUSEHOLD"), date_to_timestamp("1995-03-25")],
        },
        "q5" => TpchInput::Q5 {
            customer: proj!(
                dp::customer_read_records_from_file,
                "customer.tbl",
                |r| vec![r.c_custkey, r.c_nationkey]
            ),
            orders: proj!(dp::orders_read_records_from_file, "orders.tbl", |r| vec![
                date_to_timestamp(&r.o_orderdate),
                r.o_custkey,
                r.o_orderkey
            ]),
            lineitem: proj!(
                dp::lineitem_read_records_from_file,
                "lineitem.tbl",
                |r| vec![
                    r.l_orderkey,
                    r.l_suppkey,
                    scale_by_1000(r.l_extendedprice),
                    scale_by_1000(r.l_discount)
                ]
            ),
            supplier: proj!(
                dp::supplier_read_records_from_file,
                "supplier.tbl",
                |r| vec![r.s_suppkey, r.s_nationkey]
            ),
            nation: proj!(dp::nation_read_records_from_file, "nation.tbl", |r| vec![
                r.n_nationkey,
                string_to_u64(&r.n_name),
                r.n_regionkey
            ]),
            region: proj!(dp::region_read_records_from_cvs, "region.cvs", |r| vec![
                r.r_regionkey,
                string_to_u64(&r.r_name)
            ]),
            europe_hash: string_to_u64("EUROPE"),
            start_ts: date_to_timestamp("1997-01-01"),
            end_ts: date_to_timestamp("1998-01-01"),
            nr_pad_extra,
            co_pad_extra,
            ls_pad_extra,
        },
        "q8" => TpchInput::Q8 {
            region: proj!(dp::region_read_records_from_cvs, "region.cvs", |r| vec![
                r.r_regionkey,
                string_to_u64_trim(&r.r_name)
            ]),
            nation: proj!(dp::nation_read_records_from_file, "nation.tbl", |r| vec![
                r.n_nationkey,
                r.n_regionkey,
                string_to_u64_trim(&r.n_name)
            ]),
            customer: proj!(
                dp::customer_read_records_from_file,
                "customer.tbl",
                |r| vec![r.c_custkey, r.c_nationkey]
            ),
            orders: proj!(dp::orders_read_records_from_file, "orders.tbl", |r| vec![
                r.o_orderkey,
                r.o_custkey,
                year_from_date(&r.o_orderdate)
            ]),
            part: proj!(dp::part_read_records_from_file, "part.tbl", |r| vec![
                r.p_partkey,
                string_to_u64_trim(&r.p_type)
            ]),
            supplier: proj!(
                dp::supplier_read_records_from_file,
                "supplier.tbl",
                |r| vec![r.s_suppkey, r.s_nationkey]
            ),
            lineitem: proj!(
                dp::lineitem_read_records_from_file,
                "lineitem.tbl",
                |r| vec![
                    r.l_orderkey,
                    r.l_partkey,
                    r.l_suppkey,
                    scale_by_1000(r.l_extendedprice),
                    scale_by_1000(r.l_discount)
                ]
            ),
            cond_nation_hash: string_to_u64_trim("EGYPT"),
            const_region_name_hash: string_to_u64_trim("MIDDLE EAST"),
            const_part_type_hash: string_to_u64_trim("PROMO BRUSHED COPPER"),
        },
        "q9" => TpchInput::Q9 {
            part: proj!(dp::part_read_records_from_file, "part.tbl", |r| vec![
                r.p_partkey,
                string_to_u64(&r.p_name)
            ]),
            supplier: proj!(
                dp::supplier_read_records_from_file,
                "supplier.tbl",
                |r| vec![r.s_suppkey, r.s_nationkey]
            ),
            nation: proj!(dp::nation_read_records_from_file, "nation.tbl", |r| vec![
                r.n_nationkey,
                string_to_u64(&r.n_name)
            ]),
            orders: proj!(dp::orders_read_records_from_file, "orders.tbl", |r| vec![
                r.o_orderkey,
                year_from_date(&r.o_orderdate)
            ]),
            partsupp: proj!(
                dp::partsupp_read_records_from_file,
                "partsupp.tbl",
                |r| vec![
                    r.ps_partkey * PS_SHIFT + r.ps_suppkey,
                    scale_by_1000(r.ps_supplycost)
                ]
            ),
            lineitem: proj!(
                dp::lineitem_read_records_from_file,
                "lineitem.tbl",
                |r| vec![
                    r.l_orderkey,
                    r.l_partkey,
                    r.l_suppkey,
                    r.l_quantity,
                    scale_by_1000(r.l_extendedprice),
                    scale_by_1000(r.l_discount)
                ]
            ),
            cond_hash: string_to_u64("green"),
        },
        "q18" => TpchInput::Q18 {
            customer: proj!(
                dp::customer_read_records_from_file,
                "customer.tbl",
                |r| vec![string_to_u64(&r.c_name), r.c_custkey]
            ),
            orders: proj!(dp::orders_read_records_from_file, "orders.tbl", |r| vec![
                r.o_orderkey,
                r.o_custkey,
                date_to_timestamp(&r.o_orderdate),
                scale_by_1000(r.o_totalprice)
            ]),
            lineitem: proj!(
                dp::lineitem_read_records_from_file,
                "lineitem.tbl",
                |r| vec![r.l_orderkey, r.l_quantity]
            ),
            threshold: 300,
        },
        other => panic!("unknown TPC-H query {}", other),
    }
}

impl TpchInput {
    /// The query's input columns, in witness order -- exactly the columns the
    /// commitment layer binds.
    pub fn columns(&self) -> Vec<Vec<u64>> {
        match self {
            TpchInput::Q3 {
                customer,
                orders,
                lineitem,
                ..
            } => transpose(&[customer, orders, lineitem]),
            TpchInput::Q5 {
                customer,
                orders,
                lineitem,
                supplier,
                nation,
                region,
                ..
            } => {
                // Q5 witnesses its five KEY columns shifted by one, because
                // `q5_derive` uses 0 as the "no match" sentinel for an unmatched
                // customer/supplier/nation while TPC-H has a real nationkey 0
                // (ALGERIA) and regionkey 0 (AFRICA), so 0 must stay free. The
                // publication must carry the encoding the circuit witnesses, or
                // the binding cannot be tied to it -- the same rule the graph
                // layout follows in `edge_columns_for`.
                //
                // These become their own published columns: Q8 witnesses
                // c_nationkey RAW, so its column and Q5's are different derived
                // data and are committed separately, which is exactly how
                // `tpch_database_columns` already treats differing encodings.
                let mut cols =
                    transpose(&[customer, orders, lineitem, supplier, nation, region]);
                // c_nationkey, s_nationkey, n_nationkey, n_regionkey, r_regionkey
                for j in [1usize, 10, 11, 13, 14] {
                    for v in cols[j].iter_mut() {
                        *v += 1;
                    }
                }
                cols
            }
            TpchInput::Q8 {
                region,
                nation,
                customer,
                orders,
                part,
                supplier,
                lineitem,
                ..
            } => transpose(&[region, nation, customer, orders, part, supplier, lineitem]),
            TpchInput::Q9 {
                part,
                supplier,
                nation,
                orders,
                partsupp,
                lineitem,
                ..
            } => transpose(&[part, supplier, nation, orders, partsupp, lineitem]),
            TpchInput::Q18 {
                customer,
                orders,
                lineitem,
                ..
            } => transpose(&[customer, orders, lineitem]),
        }
    }

    pub fn require_loaded(&self) {
        for (i, c) in self.columns().iter().enumerate() {
            assert!(
                !c.is_empty(),
                "input column {} loaded EMPTY -- check VPJOIN_DATA",
                i
            );
        }
    }
}

#[cfg(test)]
mod publication_tests {
    use super::*;

    fn cols() -> Vec<Vec<u64>> {
        vec![
            (0..40u64).collect(),
            (0..40u64).map(|i| i * 7 + 1).collect(),
            (0..25u64).map(|i| i * i).collect(),
        ]
    }

    /// The property the whole rewiring exists for: two queries reading the same
    /// column of one dataset must open the SAME published point. `published_db`
    /// caches per (dataset, k), so the second caller gets the publication the
    /// first one made instead of committing again under a fresh blinder.
    #[test]
    fn queries_on_one_dataset_share_one_publication() {
        let db_cols = cols();
        let k = crate::column_commit::min_k_rows(40);

        // Query A reads columns {0, 2}; query B reads {1, 2}. Column 2 is shared.
        let a = published_db("unit-test-ds", &db_cols, k);
        let b = published_db("unit-test-ds", &db_cols, k);
        assert!(
            std::sync::Arc::ptr_eq(&a, &b),
            "the second query must reuse the publication, not make a new one"
        );

        let va = a.0.view(&column_indices(&db_cols, &[db_cols[0].clone(), db_cols[2].clone()]));
        let vb = b.0.view(&column_indices(&db_cols, &[db_cols[1].clone(), db_cols[2].clone()]));
        assert_eq!(
            va.points[1], vb.points[1],
            "the shared column must be the same published point for both queries"
        );
        assert_ne!(va.points[0], vb.points[0], "distinct columns stay distinct");

        // A different dataset is a different database and must not collide.
        let other = published_db("unit-test-ds-2", &db_cols, k);
        assert!(!std::sync::Arc::ptr_eq(&a, &other));
    }

    /// `column_indices` locates a query's columns in the published layout, and
    /// refuses rather than silently re-committing when one is absent.
    #[test]
    fn column_indices_maps_and_refuses_misses() {
        let db_cols = cols();
        assert_eq!(
            column_indices(&db_cols, &[db_cols[2].clone(), db_cols[0].clone()]),
            vec![2, 0]
        );
        let absent = vec![vec![999u64; 4]];
        assert!(
            std::panic::catch_unwind(|| column_indices(&db_cols, &absent)).is_err(),
            "a column outside the publication must be refused"
        );
    }
}

#[cfg(test)]
mod oblivious_bound_tests {
    use super::*;

    #[test]
    fn oblivious_wedge_bound_is_tight_and_tau_gated() {
        std::env::remove_var("VPJOIN_TAU_PUB");
        for m in [2u64, 3, 7, 8, 100, 27_806] {
            // one hub: split m incident edges into a incoming, b = m - a outgoing
            let one_hub = (0..=m).map(|a| a * (m - a)).max().unwrap();
            // h hubs, each with m/h edges split evenly, is the spread-out case
            let spread = (2..=8u64)
                .map(|h| {
                    let per = m / h;
                    h * (per / 2) * (per - per / 2)
                })
                .max()
                .unwrap_or(0);
            let bound = oblivious_wedge_bound("facebook", m as usize);
            assert_eq!(bound, one_hub, "m = {}", m);
            assert!(bound >= spread, "spreading beat the bound at m = {}", m);
        }
        // no fallback to declared_degree_cap: "facebook" above must have used
        // the quadratic bound even though a cap is tabulated for it
        let m = 88_234usize;
        let quadratic = oblivious_wedge_bound("facebook", m);
        assert_eq!(quadratic, 44_117 * 44_117);
        std::env::set_var("VPJOIN_TAU_PUB", "1043");
        let with_tau = oblivious_wedge_bound("facebook", m);
        std::env::remove_var("VPJOIN_TAU_PUB");
        assert_eq!(with_tau, 1043 * m as u64);
        assert!(with_tau < quadratic);
    }
}
