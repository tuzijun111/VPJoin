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
    binding_challenge, commit_column_vectors, open_column_vectors, vector_evaluations,
    verify_column_openings, BoundColumnsCircuit,
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

/// IPA parameters, loaded from the persisted `src/proof/param{k}` files (the
/// same files the per-query tests read).  Every load is logged so a run can
/// be audited.  If the file for a degree exists but cannot be parsed we FAIL
/// rather than silently regenerating, so there is no way to end up proving
/// under different parameters than the ones in `src/proof/`.  A degree that
/// is genuinely absent (the repo ships param15..param19; GQ3 on the larger
/// graphs needs more) is generated once, persisted there, and loudly logged.
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

    // baseline: the pure query circuit, as in the submission.
    // vk_s / pk_s are reported separately so they line up 1:1 with the
    // "Time to generate vk / pk" lines the per-query tests print.
    pub vk_s: f64,
    pub pk_s: f64,
    pub keygen_s: f64,
    pub prove_s: f64,
    pub verify_s: f64,
    pub proof_bytes: usize,

    // commitment layer (zero in baseline mode)
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

/// The profile this binary was compiled with.  `cargo test` (without
/// `--release`) and `cargo run` (without `--release`) both build the
/// unoptimized profile -- including halo2 itself at `opt-level = 0` -- under
/// which the same circuit proves several times slower.  Timings are only
/// comparable within one profile, so every row records which one produced it.
pub const fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Pure query circuit (what the submission reports).
    Baseline,
    /// Query circuit + published per-column commitment + in-circuit
    /// witness-equality check + column openings.
    Full,
    /// ONLY the commitment layer, measured in the corresponding query
    /// circuit's own domain (same `k`).  The query proof itself is not re-run;
    /// add these numbers to the matching `baseline` row to get the full cost.
    /// The Fiat-Shamir challenge is bound to the query proof saved by the
    /// baseline run (`src/proof/bench/<query>_<dataset>.proof`) when present.
    Commit,
}

// ---------------------------------------------------------------------------
// Explicit circuit degrees
// ---------------------------------------------------------------------------

/// The degree `k` each (query, dataset) is proved at.
///
/// These are fixed rather than searched for: escalating `k` on failure would
/// burn a full keygen at every rejected degree, which on the large graph
/// circuits costs many minutes each.
///
/// Provenance: TPC-H uses `param16` exactly as the per-query tests do; GQ1/GQ2
/// use `param17` and GQ3/GQ4 `param18`, matching the degrees the tests' real
/// proofs load.  All of these are confirmed by completed runs in
/// `results/vpjoin_baseline_debug.csv`.  GQ3/GQ4 on the two larger graphs
/// overflow `k=21` (also measured) and are tabulated separately.
pub fn degree_for(query: &str, dataset: &str, privacy: Privacy) -> u32 {
    match (query, dataset) {
        // TPC-H: every query includes lineitem (60,175 rows at the base
        // scale), so k = 16 there.
        //
        // Q5 materializes three intermediates (NR, CO, and the LS join).  The
        // circuit lays them in disjoint column groups, so the circuit height
        // is their MAXIMUM, not their sum.  Size the degree from the actual
        // (padded) section heights under whatever privacy regime is in force,
        // so the same query scales with the data instead of being pinned to
        // one hand-tabulated value:
        //   * rjs shrinks every bag to its true size (pads = 0,0,0);
        //   * legacy adds the fixed 2,848 / 59,452 constants on top of
        //     |orders| and the LS join;
        //   * dp releases per-bag capacities that can far exceed the base at
        //     small epsilon.
        // The `lineitem` section is the raw fact table, so it dominates once
        // lineitem grows past the padded CO/LS bags.  At the base 60,175-row
        // lineitem this returns 16 for every regime (the max of lineitem
        // 60,175 / CO 17,848 / LS 59,515 confirms max, not sum, and is
        // unchanged from the previous hand-tabulated 16); at 120K/240K rows it
        // grows to 17/18.  `run_at` still fails loudly if a bag ever outgrows
        // the returned degree.
        ("q5", _) => {
            let (_, co_pad, ls_pad) = q5_pads(privacy);
            let TpchInput::Q5 { orders, lineitem, .. } = q5_raw_cached()
            else {
                unreachable!("q5_raw_cached() returns Q5")
            };
            let rows = lineitem
                .len()
                .max(orders.len() + co_pad)
                .max(q5_ls_true() + ls_pad) as u64;
            ceil_log2(rows + 64)
        }
        // The acyclic queries materialize no join intermediate: they lay each
        // input table in its own disjoint column group and do lookups, so the
        // circuit height is the tallest single section, which is always the raw
        // `lineitem` fact table (every other table is smaller at every scale,
        // and the filtered result section is a subset of lineitem).  Size the
        // degree from the live lineitem row count so these scale with the data
        // too: 60,175 -> 16, 120K -> 17, 240K -> 18.  (Was pinned to 16, which
        // only fits <=65,536 rows.)
        ("q3" | "q8" | "q9" | "q18", _) => ceil_log2(lineitem_rows() as u64 + 64),

        // Path queries: k = 17 on all three graphs (measured).
        ("gq1" | "gq2", _) => 17,

        // Cyclic queries: both lay out one region whose height is the
        // materialized bag size, so the degree follows the bag directly.
        //
        //   Both bags of both queries are the SAME filtered wedge join
        //   sum_b (#in-edges a->b with a<b) * (#out-edges b->c), since GQ4
        //   applies the same early ordering filter as GQ3 (see the EARLY
        //   FILTER comments in g_sql4_obj::synthesize).  With the submitted
        //   `cyclic_pad_extra` constants:
        //     lastfm     232,943 +  18,067 =   251,010  -> 2^18
        //     facebook 2,690,019 +  92,827 = 2,782,846  -> 2^22
        //     wiki     2,255,867 +  79,427 = 2,335,294  -> 2^22
        //
        // facebook and lastfm store every edge with src < dst, so the a<b
        // filter removes nothing there; wiki is genuinely directed, which is
        // the only place the filter bites (it halves wiki's wedge, and is why
        // GQ4 on wiki fits 2^22 rather than needing 2^23).
        // Cyclic queries depend on the privacy regime, so their degree is
        // computed from the actual bag sizes by `graph_degree`.
        ("gq3" | "gq4", _) => graph_degree(query, dataset, privacy),

        (q, d) => panic!("no degree tabulated for ({}, {})", q, d),
    }
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

/// The query proof written by an earlier `baseline` run, if present, so a
/// commit-only measurement still binds its Fiat-Shamir challenge to the real
/// query proof rather than to nothing.
fn saved_proof(label: &str) -> Vec<u8> {
    let p = PathBuf::from(crate::paths::proof_file("bench")).join(format!("{}.proof", label));
    match std::fs::read(&p) {
        Ok(b) => {
            println!("  [{}] binding challenge to saved query proof ({} bytes)", label, b.len());
            b
        }
        Err(_) => {
            println!(
                "  [{}] NOTE: no saved query proof at {} -- run `baseline` first to bind the \
                 challenge to a real proof; using an empty transcript for now",
                label,
                p.display()
            );
            Vec::new()
        }
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
/// `VPJOIN_OBJ=test` selects the `*_obj_test` circuits, whose One-Pass OBJ
/// certifies condition (4) with the Cardinality Preservation Check, instead of
/// the shipped `*_obj` ones, whose residual-side condition argued over the
/// residual relations. Everything else about the run is identical, so the two
/// modes are directly comparable at the same degree and on the same data.
pub fn obj_variant_is_test() -> bool {
    std::env::var("VPJOIN_OBJ")
        .map(|v| v.eq_ignore_ascii_case("test"))
        .unwrap_or(false)
}

/// Suffix appended to the reported config, so a results file keeps the two
/// modes apart.
fn obj_variant_tag() -> &'static str {
    if obj_variant_is_test() {
        "+cp"
    } else {
        ""
    }
}

fn maybe_run<C: Circuit<Fp>>(
    mode: Mode,
    label: &str,
    circuit: &C,
    instance: &[Fp],
    k: u32,
) -> Timed {
    if mode == Mode::Commit {
        println!("  [{}] commit-only mode: query proof not re-run (k={})", label, k);
        return Timed::skipped(k);
    }
    run_at(label, circuit, instance, k)
}

/// Run one circuit with the REAL prover -- keygen_vk / keygen_pk /
/// create_proof / verify_proof over IPA on the Pasta curves, identical to the
/// `generate_and_verify_proof` helpers in the per-query test modules.
/// MockProver is never used anywhere in this harness.
///
/// `k` is taken from the explicit per-(query, dataset) table in [`degree_for`]
/// and used as-is: no trial-and-error escalation, so no prover time is ever
/// spent on a degree that is then thrown away.  A circuit that does not fit at
/// its tabulated degree is a hard error naming the row to correct.
fn run_at<C: Circuit<Fp>>(label: &str, circuit: &C, instance: &[Fp], k: u32) -> Timed {
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
    ($cols:expr, $k:expr, $proof:expr, $row:expr, $($n:literal),+) => {
        match $cols.len() {
            $($n => bind_columns::<$n>($cols, $k, $proof, $row),)+
            other => panic!(
                "no BoundColumnsCircuit instantiation for {} columns; add it to bind_dispatch!",
                other
            ),
        }
    };
}

fn bind_columns<const NC: usize>(cols: &[Vec<u64>], k: u32, query_proof: &[u8], row: &mut Row) {
    let params = params_for(k);
    let rng = OsRng;

    // 1. Publish one hiding Pedersen commitment per column (setup, once).
    let t = Instant::now();
    let commitments = commit_column_vectors(&params, cols, k, rng);
    row.commit_setup_s = t.elapsed().as_secs_f64();
    row.published_bytes = commitments.published_bytes();

    // 2. In-circuit check that the witness columns equal the committed data,
    //    at a Fiat-Shamir point bound to the published commitments AND the
    //    query proof itself (as Appendix A specifies).
    let x = binding_challenge(&commitments, query_proof);
    let evals = vector_evaluations(cols, k, x);
    let mut instance = vec![x];
    instance.extend_from_slice(&evals);

    let circuit = BoundColumnsCircuit::<NC> {
        columns: cols.to_vec(),
        x,
    };
    let label = format!("{}_{}_bind", row.query, row.dataset);
    let t = run_at(&label, &circuit, &instance, k);
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
    let openings = open_column_vectors(&params, &commitments, cols, x, rng);
    row.open_prove_s = t.elapsed().as_secs_f64();
    row.open_bytes = openings.iter().map(|o| o.len()).sum();

    let t = Instant::now();
    let ok = verify_column_openings(&params, &commitments.points, x, &evals, &openings);
    row.open_verify_s = t.elapsed().as_secs_f64();
    assert!(ok, "column openings failed to verify");
}

/// Degree of the commitment/binding domain.
///
/// This is sized by the *data* (the longest committed column), not by the
/// query circuit's degree.  It is always <= the query circuit's `k`, so the
/// layer still reuses a prefix of the very same IPA parameters -- the IPA
/// generators are derived by position, so the prefix of a degree-`k_query`
/// SRS *is* the degree-`k_commit` SRS.  Sizing by the query's `k` instead
/// would, for example, build a 2^22 commitment circuit for GQ3's ~100K-row
/// edge table purely because its wedge intermediate is large.
///
/// For every TPC-H query this yields k = 16, since lineitem (60,175 rows)
/// dominates and appears in all of them.
fn commit_k(cols: &[Vec<u64>], query_k: u32) -> u32 {
    let rows = cols.iter().map(|c| c.len()).max().unwrap_or(0);
    let need = crate::column_commit::min_k_rows(rows);
    assert!(
        need <= query_k,
        "columns need k={} but the query circuit is k={}",
        need,
        query_k
    );
    // Size the domain by the DATA BEING COMMITTED, not by the query circuit.
    //
    // The layer commits the query's *input* columns, so the work is
    // proportional to the input length.  For the TPC-H queries this is the
    // query's own degree anyway (lineitem's 60,175 rows are what forces
    // k = 16).  For the cyclic graph queries it is not: GQ3/GQ4 need k = 22-23
    // only because of their materialized intermediate bags, while the Edge
    // relation they commit is 27K-104K rows (k = 15-17).  Charging the
    // commitment layer for a 2^23 domain would measure the intermediate
    // blow-up rather than the cost of binding the input.
    //
    // This still reuses the query's parameters: the IPA generators are derived
    // by position, so the degree-`need` SRS is exactly the prefix of the
    // degree-`query_k` one (see column_commit's
    // `commitment_point_is_independent_of_k` test).
    need
}

fn apply_commitment_layer(cols: &[Vec<u64>], query_k: u32, query_proof: &[u8], row: &mut Row) {
    row.n_columns = cols.len();
    let k = commit_k(cols, query_k);
    row.commit_k = k;
    bind_dispatch!(cols, k, query_proof, row, 2, 4, 5, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20);
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

/// Parsed edge lists, memoized per dataset for the life of the process.
///
/// An epsilon sweep asks for the same dataset once per epsilon, and parsing the
/// file dominates the cost of planning a row, so the file is read once and the
/// list is cloned out of the cache.  The parse is deterministic, so a cached
/// list is indistinguishable from a fresh one.
fn graph_cache() -> &'static std::sync::Mutex<HashMap<String, std::sync::Arc<Vec<Edge>>>> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<HashMap<String, std::sync::Arc<Vec<Edge>>>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

pub fn load_graph(dataset: &str) -> Vec<Edge> {
    let mut cache = graph_cache().lock().expect("graph cache poisoned");
    let edges = cache
        .entry(dataset.to_string())
        .or_insert_with(|| std::sync::Arc::new(read_graph_file(dataset)));
    (**edges).clone()
}

fn edge_columns(edges: &[Edge]) -> Vec<Vec<u64>> {
    vec![
        edges.iter().map(|e| e.src).collect(),
        edges.iter().map(|e| e.dst).collect(),
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
                        total +=
                            (cab as u128) * (cbc as u128) * (ccd as u128) * (cda as u128);
                    }
                }
            }
        }
    }
    total as u64
}

/// Cyclic-query bag capacities for the submitted runs.  Hand-picked constants
/// (`dp/legacy_capacities.md`), dataset specific, so keyed by dataset rather
/// than hard-coded to the lastfm value.  GQ3 and GQ4 share them: both queries
/// materialize two bags through the same pad knobs, and their Bag 1 is the
/// same filtered wedge join of identical size.
fn cyclic_pad_extra(dataset: &str) -> usize {
    match dataset {
        "lastfm" => 18_067,
        "facebook" => 92_827,
        "wiki" => 79_427,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Privacy regime for the cyclic (TDJ) queries
// ---------------------------------------------------------------------------

/// How the materialized bags of a cyclic query are sized.
///
/// Only GQ3/GQ4 (and Q5) materialize intermediates, so only they have a
/// choice here; the acyclic queries never materialize one and are oblivious by
/// construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Privacy {
    /// Capacity = the true bag size.  Leaks the exact intermediate
    /// cardinality; this is the paper's "Revealing Join Size" lower bound.
    Rjs,
    /// The hand-picked constants used for the submitted results
    /// (`dp/legacy_capacities.md`).  GQ3 and GQ4 share them.
    Legacy,
  
    Dp { epsilon: f64, delta: f64 },
}

impl Privacy {
    pub fn label(&self) -> String {
        match self {
            Privacy::Rjs => "rjs".to_string(),
            Privacy::Legacy => "legacy-dp".to_string(),
            Privacy::Dp { epsilon, delta } => format!("dp(eps={} del={})", epsilon, delta),
        }
    }
}

/// Padding knobs for Q5's three materialized intermediates, under a given
/// privacy regime.
///
/// Returns `(nr_pad_extra, co_pad_extra, ls_pad_extra)`. The circuit sizes each
/// intermediate as `base + pad_extra`, where `base` is the true size it
/// computes internally (`nation.len()`, `orders.len()`, and the filtered
/// lineitem-supplier join length respectively, see `q5_obj.rs`).
///
/// `Rjs` and `Legacy` need no size model: the former is all-zero by definition,
/// the latter replays the constants used for the submitted results.  `Dp`
/// releases both bag capacities under the row-level policy
/// P = {customer, supplier}: the per-key fan-outs bounding each release's
/// sensitivity live in the unprotected relations (orders, lineitem), so the
/// frequency bounds are exact and each capacity release spends the full
/// budget (see the `Privacy::Dp` arm below).
pub fn q5_pads(privacy: Privacy) -> (usize, usize, usize) {
    match privacy {
        Privacy::Rjs => (0, 0, 0),
        // Values used for the DP results reported in the paper; see
        // `dp/legacy_capacities.md` and the note in `q5_obj.rs`.
        Privacy::Legacy => (0, 2848, 59452),
        Privacy::Dp { epsilon, delta } => {
            let (nr, co, ls) = match q5_raw_cached() {
                TpchInput::Q5 {
                    customer, orders, lineitem, start_ts, end_ts, ..
                } => {
                    // Bag sizes come from the circuit's OWN derivation, so the
                    // released capacity is calibrated to the size the circuit
                    // will actually materialize.
                    let ls_true = q5_ls_true() as u64;

                    // DP POLICY: row-level neighbors, P = {customer,
                    // supplier}.  The two entity relations are private;
                    // orders, lineitem, nation and region are not.
                    // Neighboring instances differ in ONE customer row or
                    // ONE supplier row.  The N|x|R dimension filter stays
                    // unpadded: nation and region are the fixed TPC-H
                    // catalogs (25 and 5 rows), public dimension data.
                    // Input cardinalities (|C|, |S| included) are declared
                    // public metadata, as is standard in the DP-join
                    // literature; strictly, add/remove neighbors reveal
                    // them through the positional layout, so a deployment
                    // wanting them hidden must switch to replacement
                    // neighbors (doubling the taus below) or pad the base
                    // sections to released capacities.
                    //
                    // SENSITIVITY.  One customer row with custkey x moves
                    // Bag 1 {O,C} by at most freq_orders(x) <= tau_c rows;
                    // one supplier row with suppkey y moves Bag 2 {L,S} by
                    // at most freq_lineitem(y) <= tau_s rows.  Both
                    // fan-outs live in UNPROTECTED relations, identical
                    // across neighbors, so tau_c and tau_s are 0-sensitive
                    // statistics of the instance: the frequency stage of
                    // the two-stage mechanism degenerates to an EXACT
                    // release (no noise, no budget), and Delta <= tau holds
                    // on every neighbor with no truncation and no declared
                    // cap (see `dp_join_capacity_unprotected_tau`).
                    //
                    // COMPOSITION.  A supplier row touches only the Bag 2
                    // release (suppkey links only L-S), so the supplier
                    // axis carries eps_ls alone.  A customer row touches
                    // Bag 1 through custkey AND Bag 2 through the
                    // c_nationkey = s_nationkey channel: the ls_join gate
                    // goes through co_set, which encodes the customer's
                    // nationkey, so removing one customer removes up to
                    // delta_ls_cust rows of ls_join (its window lineitems;
                    // measured exactly below, again from unprotected
                    // relations).  The customer axis therefore pays
                    //
                    //   eps_co + eps_ls * delta_ls_cust / ls_sens <= eps
                    //
                    // (scale argument: Bag 2's noise has scale
                    // ls_sens/eps_ls, so a shift of delta_ls_cust costs
                    // eps_ls * delta_ls_cust/ls_sens of budget).  Bag 2
                    // spends the FULL eps; Bag 1 gets the remainder, which
                    // is ~0.9 eps since delta_ls_cust << tau_s.  delta
                    // splits in half per release.
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
                    let delta_ls_cust =
                        cust_lines.values().copied().max().unwrap_or(0);

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
                            cust_count.get(&o[1]).copied().unwrap_or(0).saturating_sub(1)
                        })
                        .sum();

                    // NOTE (benchmarking artifact): dp_rng is publicly
                    // seeded for reproducibility; a deployment must draw
                    // this noise from secret entropy.
                    let mut rng = dp_rng("q5", "tpch-60K");
                    let ls_sens = tau_s.max(delta_ls_cust);
                    let eps_ls = epsilon;
                    let eps_co =
                        epsilon * (1.0 - delta_ls_cust as f64 / ls_sens as f64);
                    let co_cap = crate::dp_noise::dp_join_capacity_unprotected_tau(
                        co_extra_true, tau_c, eps_co, delta / 2.0, &mut rng,
                    );
                    let co_pad = co_cap.capacity as usize;

                    let ls_cap = crate::dp_noise::dp_join_capacity_unprotected_tau(
                        ls_true, ls_sens, eps_ls, delta / 2.0, &mut rng,
                    );
                    // No clamp: the release is valid as-is.  The DP lane
                    // circuit hosts the capacity in ceil(cap / 2^16)
                    // parallel column-group lanes at fixed k = 16 and
                    // asserts its own structural maximum; the single-column
                    // circuit instead takes the degree bump computed in
                    // `degree_for`.
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

/// True sizes and join-key frequencies of a cyclic query's two bags.
///
/// GQ3: bag1 is the wedge join `in(a->b, a<b) |x|_b out(b->c)`; bag2 is `t3`,
/// the plain edge relation (its size is the public |E|, so it needs no DP).
/// GQ4: both bags are the unfiltered wedge join `in |x|_v out`, of equal size.
// pub(crate) so the lane-plan harness can report true bag sizes alongside
// the released pads without re-deriving them.
pub struct BagStats {
    pub bag1_size: u64,
    bag1_mf_a: u64,
    bag1_mf_b: u64,
    pub bag2_size: u64,
    bag2_mf_a: u64,
    bag2_mf_b: u64,
    /// bag2 is a base relation whose size is public (GQ3), so no DP is owed.
    bag2_is_public: bool,
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
        },
        // Both GQ4 bags are the SAME wedge shape (Bag1 is A->B->C joined on B,
        // Bag2 is C->D->A joined on D), and both apply an early ordering
        // filter on the incoming edge (A<B and C<D respectively), so both
        // materialize `wedge(indeg_lt)` rows -- see the EARLY FILTER comments
        // in `g_sql4_obj::synthesize`.  Modelling them with the unfiltered
        // `indeg` overstated wiki by 2x and pushed it to k=23.
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
pub fn graph_pads(query: &str, dataset: &str, edges: &[Edge], privacy: Privacy) -> (usize, usize) {
    match privacy {
        Privacy::Rjs => (0, 0),
        Privacy::Legacy => {
            // Same constants for GQ3 and GQ4, applied to both bags.
            let p = cyclic_pad_extra(dataset);
            (p, p)
        }
        Privacy::Dp { epsilon, delta } => {
            let s = bag_stats(query, edges);
            let mut rng = dp_rng(query, dataset);
            // NO PUBLIC DEGREE CAP.  The join-size sensitivity depends on
            // the graph's maximum degree, which is private, so the bound is
            // itself RELEASED under DP rather than declared: one one-sided
            // release of the max degree (global sensitivity 1 -- one edge
            // moves any degree, hence the maximum, by at most 1), then the
            // capacity releases calibrated to it.
            //
            // SELF-JOIN SENSITIVITY IS 2*tau, NOT tau.  The bags are
            // `wedge = sum_v indeg_lt(v) * outdeg(v)` over ONE Edge relation,
            // so inserting (u, v) with u < v moves TWO terms: indeg_lt(v) += 1
            // adds outdeg(v), and outdeg(u) += 1 adds indeg_lt(u).  Hence
            // d(wedge) = outdeg(v) + indeg_lt(u), and both summands reach the
            // bound independently.  Calibrating to tau alone would give only
            // (2 eps, delta e^eps)-DP.  (This is where the "each aliased
            // instance is its own relation" reading goes wrong: a single
            // stored edge really does appear in both instances.)
            //
            // Budget: basic composition over 1 degree release + one capacity
            // release per bag that needs one.  The single degree release
            // serves every bag, since all of them join the same relation.
            let n_size = if s.bag2_is_public { 1 } else { 2 };
            let n_release = 1 + n_size;
            let (eps, del) = (epsilon / n_release as f64, delta / n_release as f64);

            // The released bound must cover BOTH degree statistics the
            // sensitivity is built from, so it is taken over their maximum.
            let max_deg = s
                .bag1_mf_a
                .max(s.bag1_mf_b)
                .max(s.bag2_mf_a)
                .max(s.bag2_mf_b);
            let tau_tilde = crate::dp_noise::dp_frequency_bound(max_deg, eps, del, &mut rng);
            let tau = (2.0 * tau_tilde).ceil() as u64;

            let c1 = crate::dp_noise::dp_join_capacity_public_tau(
                s.bag1_size,
                tau,
                eps,
                del,
                &mut rng,
            );
            let pad1 = c1.capacity.saturating_sub(s.bag1_size) as usize;

            let pad2 = if s.bag2_is_public {
                0
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
                "  [dp] {} on {}: max_deg {} -> released tau {:.0} (sens {}), \
                 bag1 {} (+{} pad), bag2 {} (+{} pad{})",
                query,
                dataset,
                max_deg,
                tau_tilde,
                tau,
                s.bag1_size,
                pad1,
                s.bag2_size,
                pad2,
                if s.bag2_is_public { ", public size" } else { "" }
            );
            (pad1, pad2)
        }
    }
}

/// Rows the cyclic circuit lays out: one region whose height is the largest
/// of the base table, bag2 (+its aggregation row) and bag1.
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
    use crate::graph_sql::{
        g_sql1_obj, g_sql1_obj_test, g_sql2_obj, g_sql2_obj_test, g_sql3_obj, g_sql3_obj_test,
        g_sql4_obj, g_sql4_obj_test,
    };
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

    let cols = edge_columns(&edges);
    let label = format!("{}_{}", query, dataset);
    // Bag capacities and hence the degree follow from the privacy regime.
    let (pad1, pad2) = if matches!(query, "gq3" | "gq4") {
        graph_pads(query, dataset, &edges, privacy)
    } else {
        (0, 0)
    };
    let k = match query {
        "gq3" | "gq4" => ceil_log2(graph_rows(&edges, query, pad1, pad2) + 64),
        _ => degree_for(query, dataset, privacy),
    };
    row.config = if matches!(query, "gq3" | "gq4") {
        format!("{}{}", privacy.label(), obj_variant_tag())
    } else {
        format!("oblivious{}", obj_variant_tag())
    };

    let (cnt, timed) = match query {
        "gq1" => {
            let cnt = count_gq1(&edges);
            let ins = [Fp::from(cnt)];
            let timed = if obj_variant_is_test() {
                let c = g_sql1_obj_test::Path3OrdCircuit::<Fp> {
                    edges,
                    _marker: PhantomData,
                };
                maybe_run(mode, &label, &c, &ins, k)
            } else {
                let c = g_sql1_obj::Path3OrdCircuit::<Fp> {
                    edges,
                    _marker: PhantomData,
                };
                maybe_run(mode, &label, &c, &ins, k)
            };
            (cnt, timed)
        }
        "gq2" => {
            let cnt = count_gq2(&edges);
            let ins = [Fp::from(cnt)];
            let timed = if obj_variant_is_test() {
                let c = g_sql2_obj_test::GraphPath4OrderCircuit::<Fp> {
                    edges,
                    _marker: PhantomData,
                };
                maybe_run(mode, &label, &c, &ins, k)
            } else {
                let c = g_sql2_obj::GraphPath4OrderCircuit::<Fp> {
                    edges,
                    _marker: PhantomData,
                };
                maybe_run(mode, &label, &c, &ins, k)
            };
            (cnt, timed)
        }
        "gq3" => {
            let cnt = count_gq3(&edges);
            let ins = [Fp::from(cnt)];
            let timed = if obj_variant_is_test() {
                let c = g_sql3_obj_test::MyCircuit::<Fp> {
                    edges,
                    bag1_pad_extra: pad1,
                    bag2_pad_extra: pad2,
                    _marker: PhantomData,
                };
                maybe_run(mode, &label, &c, &ins, k)
            } else {
                let c = g_sql3_obj::MyCircuit::<Fp> {
                    edges,
                    bag1_pad_extra: pad1,
                    bag2_pad_extra: pad2,
                    _marker: PhantomData,
                };
                maybe_run(mode, &label, &c, &ins, k)
            };
            (cnt, timed)
        }
        "gq4" => {
            let cnt = count_gq4(&edges);
            let ins = [Fp::from(cnt)];
            let timed = if obj_variant_is_test() {
                let c = g_sql4_obj_test::MyCircuit::<Fp> {
                    edges,
                    bag1_pad_extra: pad1,
                    bag2_pad_extra: pad2,
                    _marker: PhantomData,
                };
                maybe_run(mode, &label, &c, &ins, k)
            } else {
                let c = g_sql4_obj::MyCircuit::<Fp> {
                    edges,
                    bag1_pad_extra: pad1,
                    bag2_pad_extra: pad2,
                    _marker: PhantomData,
                };
                maybe_run(mode, &label, &c, &ins, k)
            };
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
        let bound_to = if timed.proof.is_empty() {
            saved_proof(&label)
        } else {
            timed.proof.clone()
        };
        apply_commitment_layer(&cols, timed.k, &bound_to, &mut row);
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
    use crate::sql::{
        q18_obj, q18_obj_test, q3_obj, q3_obj_test, q5_obj, q5_obj_test, q8_obj, q8_obj_test,
        q9_obj, q9_obj_test,
    };
    use std::marker::PhantomData;

    let t_all = Instant::now();
    let mut row = Row {
        query: query.to_string(),
        dataset: dataset.to_string(),
        public_output: 1,
        profile: build_profile(),
        config: format!("oblivious{}", obj_variant_tag()),
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
                if obj_variant_is_test() {
                    let circuit = q3_obj_test::MyCircuit::<Fp> {
                        customer: c,
                        orders: o,
                        lineitem: l,
                        condition: [
                            string_to_u64("HOUSEHOLD"),
                            date_to_timestamp("1995-03-25"),
                        ],
                        _marker: PhantomData,
                    };
                    maybe_run(mode, &label, &circuit, &one, k)
                } else {
                    let circuit = q3_obj::MyCircuit::<Fp> {
                        customer: c,
                        orders: o,
                        lineitem: l,
                        condition: [
                            string_to_u64("HOUSEHOLD"),
                            date_to_timestamp("1995-03-25"),
                        ],
                        _marker: PhantomData,
                    };
                    maybe_run(mode, &label, &circuit, &one, k)
                }
            })
        }

        // ---------------- Q5 ----------------
        "q5" => {
            let c = customer(&|r| vec![r.c_custkey, r.c_nationkey]);
            let o = orders(&|r| {
                vec![
                    date_to_timestamp(&r.o_orderdate),
                    r.o_custkey,
                    r.o_orderkey,
                ]
            });
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
            let cols = transpose(&[&c, &o, &l, &s, &n, &rg]);
            (cols, {
                load_s = t_all.elapsed().as_secs_f64();
                if obj_variant_is_test() {
                    let circuit = q5_obj_test::MyCircuit::<Fp> {
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
                        _marker: PhantomData,
                    };
                    maybe_run(mode, &label, &circuit, &one, k)
                } else {
                    let circuit = q5_obj::MyCircuit::<Fp> {
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
                        _marker: PhantomData,
                    };
                    maybe_run(mode, &label, &circuit, &one, k)
                }
            })
        }

        // ---------------- Q8 ----------------
        "q8" => {
            let rg = region(&|r| vec![r.r_regionkey, string_to_u64_trim(&r.r_name)]);
            let n = nation(&|r| {
                vec![
                    r.n_nationkey,
                    r.n_regionkey,
                    string_to_u64_trim(&r.n_name),
                ]
            });
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
            row.input_rows =
                rg.len() + n.len() + c.len() + o.len() + p.len() + s.len() + l.len();
            let cols = transpose(&[&rg, &n, &c, &o, &p, &s, &l]);
            (cols, {
                load_s = t_all.elapsed().as_secs_f64();
                if obj_variant_is_test() {
                    let circuit = q8_obj_test::MyCircuit::<Fp> {
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
                } else {
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
                }
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
                if obj_variant_is_test() {
                    let circuit = q9_obj_test::MyCircuit::<Fp> {
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
                } else {
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
                }
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
                if obj_variant_is_test() {
                    let circuit = q18_obj_test::MyCircuit::<Fp> {
                        customer: c,
                        orders: o,
                        lineitem: l,
                        threshold: 300,
                        _marker: PhantomData,
                    };
                    maybe_run(mode, &label, &circuit, &one, k)
                } else {
                    let circuit = q18_obj::MyCircuit::<Fp> {
                        customer: c,
                        orders: o,
                        lineitem: l,
                        threshold: 300,
                        _marker: PhantomData,
                    };
                    maybe_run(mode, &label, &circuit, &one, k)
                }
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
        let bound_to = if timed.proof.is_empty() {
            saved_proof(&label)
        } else {
            timed.proof.clone()
        };
        apply_commitment_layer(&cols, timed.k, &bound_to, &mut row);
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
    let mut out = Vec::new();
    for q in queries {
        if is_graph(q) {
            for d in GRAPH_DATASETS {
                out.push((q.clone(), d.to_string()));
            }
        } else {
            out.push((q.clone(), tpch_label()));
        }
    }
    out
}

/// Label reported for the TPC-H dataset.  `VPJOIN_DATA` selects WHICH tables are
/// read; this only *names* them in the results (the CSV `dataset` column, the
/// resume key, the log lines), so a sweep over several data roots stays
/// distinguishable in one place.  Defaults to the historical `tpch-60K`.
pub fn tpch_label() -> String {
    std::env::var("VPJOIN_LABEL").unwrap_or_else(|_| "tpch-60K".to_string())
}

// ---------------------------------------------------------------------------
// TPC-H inputs, shared by the sweep and the in-circuit binding diff
// ---------------------------------------------------------------------------

/// One TPC-H query's loaded input tables plus its query constants.
///
/// This is the single definition of each query's attribute projections; both
/// the sweep and `commit_diff` build their circuits from it, so the two can
/// never drift apart.
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
            customer: proj!(dp::customer_read_records_from_file, "customer.tbl", |r| vec![
                string_to_u64(&r.c_mktsegment),
                r.c_custkey
            ]),
            orders: proj!(dp::orders_read_records_from_file, "orders.tbl", |r| vec![
                date_to_timestamp(&r.o_orderdate),
                r.o_shippriority,
                r.o_custkey,
                r.o_orderkey
            ]),
            lineitem: proj!(dp::lineitem_read_records_from_file, "lineitem.tbl", |r| vec![
                r.l_orderkey,
                scale_by_1000(r.l_extendedprice),
                scale_by_1000(r.l_discount),
                date_to_timestamp(&r.l_shipdate)
            ]),
            condition: [
                string_to_u64("HOUSEHOLD"),
                date_to_timestamp("1995-03-25"),
            ],
        },
        "q5" => TpchInput::Q5 {
            customer: proj!(dp::customer_read_records_from_file, "customer.tbl", |r| vec![
                r.c_custkey,
                r.c_nationkey
            ]),
            orders: proj!(dp::orders_read_records_from_file, "orders.tbl", |r| vec![
                date_to_timestamp(&r.o_orderdate),
                r.o_custkey,
                r.o_orderkey
            ]),
            lineitem: proj!(dp::lineitem_read_records_from_file, "lineitem.tbl", |r| vec![
                r.l_orderkey,
                r.l_suppkey,
                scale_by_1000(r.l_extendedprice),
                scale_by_1000(r.l_discount)
            ]),
            supplier: proj!(dp::supplier_read_records_from_file, "supplier.tbl", |r| vec![
                r.s_suppkey,
                r.s_nationkey
            ]),
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
            customer: proj!(dp::customer_read_records_from_file, "customer.tbl", |r| vec![
                r.c_custkey,
                r.c_nationkey
            ]),
            orders: proj!(dp::orders_read_records_from_file, "orders.tbl", |r| vec![
                r.o_orderkey,
                r.o_custkey,
                year_from_date(&r.o_orderdate)
            ]),
            part: proj!(dp::part_read_records_from_file, "part.tbl", |r| vec![
                r.p_partkey,
                string_to_u64_trim(&r.p_type)
            ]),
            supplier: proj!(dp::supplier_read_records_from_file, "supplier.tbl", |r| vec![
                r.s_suppkey,
                r.s_nationkey
            ]),
            lineitem: proj!(dp::lineitem_read_records_from_file, "lineitem.tbl", |r| vec![
                r.l_orderkey,
                r.l_partkey,
                r.l_suppkey,
                scale_by_1000(r.l_extendedprice),
                scale_by_1000(r.l_discount)
            ]),
            cond_nation_hash: string_to_u64_trim("EGYPT"),
            const_region_name_hash: string_to_u64_trim("MIDDLE EAST"),
            const_part_type_hash: string_to_u64_trim("PROMO BRUSHED COPPER"),
        },
        "q9" => TpchInput::Q9 {
            part: proj!(dp::part_read_records_from_file, "part.tbl", |r| vec![
                r.p_partkey,
                string_to_u64(&r.p_name)
            ]),
            supplier: proj!(dp::supplier_read_records_from_file, "supplier.tbl", |r| vec![
                r.s_suppkey,
                r.s_nationkey
            ]),
            nation: proj!(dp::nation_read_records_from_file, "nation.tbl", |r| vec![
                r.n_nationkey,
                string_to_u64(&r.n_name)
            ]),
            orders: proj!(dp::orders_read_records_from_file, "orders.tbl", |r| vec![
                r.o_orderkey,
                year_from_date(&r.o_orderdate)
            ]),
            partsupp: proj!(dp::partsupp_read_records_from_file, "partsupp.tbl", |r| vec![
                r.ps_partkey * PS_SHIFT + r.ps_suppkey,
                scale_by_1000(r.ps_supplycost)
            ]),
            lineitem: proj!(dp::lineitem_read_records_from_file, "lineitem.tbl", |r| vec![
                r.l_orderkey,
                r.l_partkey,
                r.l_suppkey,
                r.l_quantity,
                scale_by_1000(r.l_extendedprice),
                scale_by_1000(r.l_discount)
            ]),
            cond_hash: string_to_u64("green"),
        },
        "q18" => TpchInput::Q18 {
            customer: proj!(dp::customer_read_records_from_file, "customer.tbl", |r| vec![
                string_to_u64(&r.c_name),
                r.c_custkey
            ]),
            orders: proj!(dp::orders_read_records_from_file, "orders.tbl", |r| vec![
                r.o_orderkey,
                r.o_custkey,
                date_to_timestamp(&r.o_orderdate),
                scale_by_1000(r.o_totalprice)
            ]),
            lineitem: proj!(dp::lineitem_read_records_from_file, "lineitem.tbl", |r| vec![
                r.l_orderkey,
                r.l_quantity
            ]),
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
            TpchInput::Q3 { customer, orders, lineitem, .. } => {
                transpose(&[customer, orders, lineitem])
            }
            TpchInput::Q5 { customer, orders, lineitem, supplier, nation, region, .. } => {
                transpose(&[customer, orders, lineitem, supplier, nation, region])
            }
            TpchInput::Q8 {
                region, nation, customer, orders, part, supplier, lineitem, ..
            } => transpose(&[region, nation, customer, orders, part, supplier, lineitem]),
            TpchInput::Q9 { part, supplier, nation, orders, partsupp, lineitem, .. } => {
                transpose(&[part, supplier, nation, orders, partsupp, lineitem])
            }
            TpchInput::Q18 { customer, orders, lineitem, .. } => {
                transpose(&[customer, orders, lineitem])
            }
        }
    }

    pub fn require_loaded(&self) {
        for (i, c) in self.columns().iter().enumerate() {
            assert!(!c.is_empty(), "input column {} loaded EMPTY -- check VPJOIN_DATA", i);
        }
    }
}

