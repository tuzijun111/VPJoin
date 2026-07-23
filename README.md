# VPJoin



We implement **VPJoin**, a zero-knowledge proof framework for verifiable multi-way SQL join processing. Built on the [Halo2](https://github.com/zcash/halo2) proving system with PLONKish arithmetization, VPJoin enables a data holder to prove the correct execution of complex SQL queries over private data without revealing the underlying records.

## Problem

Existing ZK database systems decompose multi-way joins into sequential binary joins, materializing each intermediate result inside the proof circuit. Because circuit dimensions are public, intermediate sizes leak cardinality information, and padding them to worst-case sizes leads to **O(IN^k) circuit blowup** for a k-way join. For TPC-H Q8 (an 8-way join), this translates to an estimated proving time exceeding 10^19 seconds.

VPJoin eliminates this bottleneck via **witness-guided verification**: the prover performs all join work offline and supplies a witness marking which input tuples participate in the result. The circuit only *verifies* the witness through lightweight structural checks, never materializing intermediate results.

## Key Contributions

### Oblivious Join Gate (OBJ) for Acyclic Joins

OBJ verifies multi-way joins via semijoin-based structural checks along a join tree, achieving **worst-case O(IN + OUT) circuit complexity**. Since the circuit never creates intermediate results, its layout depends only on input table sizes, so **obliviousness is achieved entirely for free** with zero padding overhead. The gate enforces four properties:

- **Conservation** -- every original tuple appears in exactly one group (via Permutation Argument)
- **Disjointness** -- clean and residual groups share no tuples (via Non-Membership Check)
- **Pairwise Consistency** -- clean tuples in neighboring relations agree on join keys (via Lookup Arguments)
- **Completeness** -- no valid join result hides among the residuals (via semijoin pass)

### Aggregation Without Join Materialization

OBJ computes join-aggregates directly over compact clean relations via **tuple multiplicities**, keeping proof cost at O(IN + OUT) without ever constructing the full O(IN^k)-sized join result.

### Tree-Decomposed Join Gate (TDJ) for Cyclic Joins

TDJ extends OBJ to cyclic queries via **tree decomposition**: intra-cluster joins are materialized at fixed capacities and the acyclic inter-cluster structure is verified by OBJ. In its default mode the capacities are worst-case bounds, so cyclic queries remain **fully oblivious**. As an *optional* relaxation, **DP-guided padding** sets the capacities using a one-sided DP noise mechanism that always exceeds the true size (preserving correctness) while providing formal (epsilon, delta)-differential privacy for the intermediate cardinalities. (TDJ was called "DP-Guided Join Gate (DPJ)" in the original submission; see `dp/legacy_capacities.md` for the padding constants used in the originally reported DP runs.)

## Project Structure

```
src/
  chips/          # Custom Halo2 gate implementations (hash, lookup, comparison, etc.)
  circuits/       # Reusable circuit gadgets (inclusion checks, permutations, Merkle trees)
  sql/            # TPC-H query circuits (Q3, Q5, Q8, Q9, Q18) using OBJ/TDJ gates
  graph_sql/      # Graph pattern query circuits (GQ1--GQ4) for network datasets
  data/           # TPC-H dataset files and data processing utilities
  graph_data/     # Network dataset files and graph data processing
  proof/          # Proof generation and verification infrastructure
  commitment.rs   # [revision] database-commitment layer: canonical layout, Commit(D), per-proof binding
  input_binding.rs# [revision] in-circuit binding of query inputs to the published commitments
  dp_noise.rs     # [revision] DP capacity generation (Rust twin of dp/noise_generator.py)
  dp_lane.rs      # [revision] shared plan/run records for the DP lane circuits
  graph_sql/pone_baseline.rs  # [revision] PoneglyphDB-style binary-join-chain baselines for GQ1-GQ4
  sql/q5_obj_dp.rs            # [revision] multi-lane DP variant of Q5 (fixed k, capacity in c lanes)
  graph_sql/g_sql3_obj_dp.rs  # [revision] multi-lane DP variant of GQ3
  graph_sql/g_sql4_obj_dp.rs  # [revision] multi-lane DP variant of GQ4 (see the caveat in section 5)
  bin/            # [revision] benchmark harnesses (see "Revision Experiments" below)
```

## Benchmarks

### Supported Queries

| Query | Type | Join Arity | Cyclic? | Description |
|-------|------|-----------|---------|-------------|
| Q3    | TPC-H | 3 | No  | Revenue of shipping-priority orders |
| Q5    | TPC-H | 6 | Yes | Local supplier volume |
| Q8    | TPC-H | 8 | No  | National market share |
| Q9    | TPC-H | 6 | No  | Product type profit measure |
| Q18   | TPC-H | 3 | No  | Large volume customer |
| GQ1   | Graph | 3 | No  | 3-way path query |
| GQ2   | Graph | 4 | No  | 4-way path query |
| GQ3   | Graph | 3 | Yes | Triangle (3-cycle) query |
| GQ4   | Graph | 4 | Yes | 4-cycle query |

### Datasets

- **TPC-H**: Standard analytical benchmark (60K / 120K / 240K rows in lineitem)
- **LastFM**: Social network, 27,806 edges
- **Facebook**: Ego-network, 88,234 edges
- **Wikipedia Vote**: Voting network, 103,689 edges


## Getting Started

### Prerequisites

- **Rust** (stable toolchain)
- Sufficient stack size for Halo2 circuit proving:

```bash
export RUST_MIN_STACK=33554432
```

### Build

```bash
cargo build --release
```

### Reproducing the Paper (the whole command set)

Every command below prints its results to stdout; none of them writes a results file.
(`vpjoin_bench` does still persist each proof to `src/proof/bench/` as an auditable
artifact — `commit-diff`'s commit mode reads those back — but no measurement leaves
the terminal.)  `cargo run` without `--release` is the **debug** profile, which is the
profile the paper's proving times use — do not mix the two in one comparison.

```bash
# 1-2. VPJoin proving time, all queries x datasets, without and with the commitment layer
cargo run --bin vpjoin_bench -- baseline
cargo run --bin vpjoin_bench -- full

# 3-4. Additional in-circuit cost of binding to a committed database
cargo commit-diff reps=3 q3 q8 q9 q18 gq1 gq2
VPJOIN_PRIVACY=rjs cargo commit-diff reps=3 q5 gq3 gq4

# 5-6. DP-guided padding for the three cyclic queries, and its no-privacy lower bound
cargo run --bin dp_lane_bench -- reps=3 q5 gq3 gq4
VPJOIN_PRIVACY=rjs cargo run --bin dp_lane_bench -- reps=3 q5 gq3 gq4

# 7. PoneglyphDB-style graph baselines (measured anchors; extrapolation is derived)
PONE_K0=17 cargo run --bin pone_graph_bench
```

One point of the epsilon sweep behind Figure "various epsilon" (vary `VPJOIN_DP_SEED=1..10`
for the paper's 10 rounds, and repeat per epsilon):

```bash
VPJOIN_EPS=0.01 VPJOIN_DP_SEED=1 cargo run --bin dp_lane_bench -- reps=3 q5 gq3 gq4
```

Preview any run's geometry — degrees, lane counts, released capacities, pads — without
proving anything:

```bash
VPJOIN_PLAN_ONLY=1 cargo run --bin dp_lane_bench
VPJOIN_PLAN_ONLY=1 cargo run --bin vpjoin_bench -- baseline
PONE_PLAN_ONLY=1 cargo run --bin pone_graph_bench
```

Each command is documented in full below: `vpjoin_bench` under
[Running Everything](#running-everything-two-commands), `commit-diff` under
[In-circuit binding cost](#in-circuit-binding-cost-cargo-commit-diff), `dp_lane_bench`
under [DP-guided padding](#5-dp-guided-padding-mechanism-lane-circuits-and-the-epsilon-sweep),
and `pone_graph_bench` under
[PoneglyphDB-style graph baselines](#4-poneglyphdb-style-graph-baselines-section-81-estimation-methodology).

**Not yet covered by any command.**  The fully oblivious (worst-case padding) bars for Q5,
GQ3 and GQ4 have no `Privacy` arm, and the 120K/240K scalability figure has no scale knob
in the query loaders — the scaled tables (`src/data/lineitem_120K.tbl`, `lineitem_240K.tbl`)
exist and are reachable from `commitment_bench`, but `bench_queries` always reads
`lineitem.tbl`.

### Running TPC-H Query Proofs

```bash
# Query 3
cargo test --package halo2-experiments --lib -- sql::q3_obj::tests::test_1 --exact --nocapture

# Query 5
cargo test --package halo2-experiments --lib -- sql::q5_obj::tests::test_1 --exact --nocapture

# Query 8
cargo test --package halo2-experiments --lib -- sql::q8_obj::tests::test_1 --exact --nocapture

# Query 9
cargo test --package halo2-experiments --lib -- sql::q9_obj::tests::test_1 --exact --nocapture

# Query 18
cargo test --package halo2-experiments --lib -- sql::q18_obj::tests::test_1 --exact --nocapture
```

### Running Graph Query Proofs

```bash
# GQ1 -- 3-way path query
cargo test --package halo2-experiments --lib -- graph_sql::g_sql1_obj::tests::test --exact --nocapture

# GQ2 -- 4-way path query
cargo test --package halo2-experiments --lib -- graph_sql::g_sql2_obj::tests::test --exact --nocapture

# GQ3 -- Triangle query
cargo test --package halo2-experiments --lib -- graph_sql::g_sql3_obj::tests::test --exact --nocapture

# GQ4 -- 4-cycle query
cargo test --package halo2-experiments --lib -- graph_sql::g_sql4_obj::tests::test --exact --nocapture
```

The per-test commands above prove **one** query on **one** dataset, and the graph tests
select their dataset by editing a commented-out line inside the test body (each of
`g_sqlN_obj::tests::test` has the three `read_edges(...)` calls with two commented out).
All paths are resolved relative to the crate root by `src/paths.rs`, so the tests work in
any checkout with no editing. (They previously hard-coded absolute paths into a sibling
directory and panicked with `NotFound` on `File::open(...).unwrap()` in a fresh clone.)
Set `VPJOIN_DATA` to relocate the `data/` + `graph_data/` + `proof/` tree.

One caveat remains in the tests themselves: the TPC-H data loaders swallow read errors
with `if let Ok(records)`, so a missing *table* leaves that relation **empty** and the test
still "passes" instead of failing. `cargo test --lib -- paths::` checks up front that every
table, graph dataset, and params file is present. The sweep commands below refuse to run on
an empty table.

### Running Everything (two commands)

`vpjoin_bench` rebuilds exactly the circuit instances the per-query tests build, but
drives all of them from one command: the 5 TPC-H queries plus the 4 graph queries on
**each** of the 3 SNAP datasets (17 rows).  Results are printed as an aligned summary
table on stdout; nothing is written to disk unless you ask for it.

```bash
# (A) BASELINE -- the pure query circuits, i.e. what the submission reports.
cargo run --bin vpjoin_bench -- baseline

# (B) FULL -- the same circuits plus the complete commitment layer: published
#     per-column Pedersen commitments to the dataset, the in-circuit check that the
#     query's witness columns equal the committed data, and the column openings.
cargo run --bin vpjoin_bench -- full
```

Omitting `--release` builds the **debug** profile, which is the profile the paper's
proving times use (see the profile table below).  Append an argument ending in `.csv`
to additionally write the full per-row schema to a file; without one, no file is
created and `VPJOIN_RESUME` has nothing to resume from (the binary says so and
continues).

Both modes use the **real proving pipeline** — `keygen_vk` / `keygen_pk` / `create_proof` /
`verify_proof` over IPA on the Pasta curves, exactly like the `generate_and_verify_proof`
helpers inside the per-query tests. **`MockProver` is never used.** Every proof is verified,
and the proof bytes are written to `src/proof/bench/<query>_<dataset>.proof` as auditable
artifacts. Public parameters are the **persisted `src/proof/param{k}` files** (param15–19
ship with the repo); every load is logged per row, a params file that exists but cannot be
parsed is a hard error (never silently regenerated), and only a genuinely absent degree
(k ≥ 20, needed by GQ3 on the larger graphs) is generated once and persisted there.

**Build profile — read this before comparing any two numbers.** The per-query commands
above (`cargo test` **without** `--release`) build the *debug* profile, which compiles the
halo2 library itself at `opt-level = 0` (this repo has no `[profile]` overrides). The same
circuit therefore proves several times slower there than under `cargo run --release`.
Measured on Q3, 60K, k=16 — the *same* circuit both times, confirmed by both runs emitting a
byte-identical 28,448-byte proof:

| | `cargo test` (debug) | `vpjoin_bench` (release) |
|---|---|---|
| vk + pk | 25.34s (7.09 + 18.25) | 4.59s |
| prove | ~117s (by subtraction) | 24.25s |
| end-to-end | 142.1s (`finished in`) | 28.9s + load |

Neither is "wrong" — they are the same computation under two compilers. Every row
records a `profile` column (`debug`/`release`) and the binary prints the profile at startup,
so a mixed comparison is visible rather than silent. **Use one profile for both commands.**
The commands above already omit `--release`, matching the `cargo test` timings; add it only
when you want the optimized numbers, and then add it to *both*.

The table reports `vk_s` and `pk_s` separately, which line up 1:1 with the
`Time to generate vk` / `Time to generate pk` lines the tests print, plus `load_s` (data
parsing) and `wall_s` (end-to-end, comparable to the test's `finished in ...` line).

The difference between the two modes is exactly the cost of making a proof bind to a
committed database instead of to an unauthenticated input. Both carry the same schema
(`query,dataset,k,input_rows,n_columns,public_output,keygen_s,prove_s,verify_s,proof_bytes,
commit_setup_s,published_bytes,bind_*,open_*,total_prove_s,total_proof_bytes,status`);
in `baseline` the commitment columns are zero.

Useful variations:

```bash
# a subset of queries (graph queries still expand over all 3 datasets)
cargo run --bin vpjoin_bench -- baseline q3 q5 q8 q9 q18
cargo run --bin vpjoin_bench -- full     gq1 gq2 gq3 gq4

# also write the full per-row schema to a file (any argument ending in .csv)
cargo run --bin vpjoin_bench -- baseline results/vpjoin_baseline.csv

# point at a different data root (must contain data/ graph_data/ proof/)
VPJOIN_DATA=/path/to/src cargo run --bin vpjoin_bench -- baseline

# cap the degree the k-fitter may try (default 21)
VPJOIN_MAX_K=19 cargo run --bin vpjoin_bench -- baseline

# print the planned degrees and DP capacities, then stop before any proving
VPJOIN_PLAN_ONLY=1 cargo run --bin vpjoin_bench -- baseline
```

Notes:

* **`k` is fitted, not assumed.** Each circuit starts at the degree its test uses
  (16 for TPC-H, 17 for GQ1/GQ2, 18 for GQ3/GQ4) and `k` is raised until the circuit
  fits. This is necessary because the graph circuits' intermediate size depends on the
  dataset, so no single hard-coded `k` serves all three. Missing `param{k}` files are
  generated and cached under `src/proof/`.
* **GQ3 is much larger on `facebook` and `wiki`** than on `lastfm` (its wedge
  intermediate is ~2.7M and ~2.3M rows vs ~233K), so those two rows need `k = 22` and are
  by far the most expensive in the sweep. With the default `VPJOIN_MAX_K=21` they are
  recorded as `FAILED: ...` in the `status` column and the sweep continues; raise
  `VPJOIN_MAX_K=22` to actually measure them (expect a long run and high memory).
* A failure in any single row is recorded in `status` rather than aborting the sweep,
  and rows are flushed as they complete, so an interrupted run is still usable.
* The padding of the three bag-materializing queries (Q5, GQ3, GQ4) comes from
  `bench_queries::q5_pads` / `graph_pads`, selected by `VPJOIN_PRIVACY`
  (`rjs` | `legacy` | `dp`, default `dp` with `VPJOIN_EPS=0.1 VPJOIN_DELTA=1e-5`).
  Nothing is hard-coded in a test any more; `legacy` replays the constants in
  `dp/legacy_capacities.md`.  Under `dp` the released capacity can exceed `2^k`, so
  `degree_for` raises `k` accordingly — use `dp_lane_bench` (below) to hold `k` fixed
  and absorb the capacity in parallel lanes instead.

### Baseline vs bound circuit files (where to look when debugging)

Every query has TWO circuit files sitting next to each other; the bound file is
the baseline **plus** the inlined witness-binding check of Appendix A, and must
differ from it **only** by the binding columns/gates:

| baseline (pure query) | bound (query + in-circuit binding check) |
|---|---|
| `src/sql/q3_obj.rs` | `src/sql/q3_bound.rs` |
| `src/sql/q5_obj.rs` | `src/sql/q5_bound.rs` |
| `src/sql/q8_obj.rs` | `src/sql/q8_bound.rs` |
| `src/sql/q9_obj.rs` | `src/sql/q9_bound.rs` |
| `src/sql/q18_obj.rs` | `src/sql/q18_bound.rs` |
| `src/graph_sql/g_sql1_obj.rs` | `src/graph_sql/g_sql1_bound.rs` |
| `src/graph_sql/g_sql2_obj.rs` | `src/graph_sql/g_sql2_bound.rs` |
| `src/graph_sql/g_sql3_obj.rs` | `src/graph_sql/g_sql3_bound.rs` |
| `src/graph_sql/g_sql4_obj.rs` | `src/graph_sql/g_sql4_bound.rs` |

The pieces of the commitment layer that live **outside** the circuit are shared:
`src/column_commit.rs` (per-column Pedersen commitments, Fiat-Shamir challenge,
IPA openings) and `src/inline_bind.rs` (the binding gate library
`configure_bind`/`assign_bind`, plus the measurement helpers). The structural
guard `cargo test --lib -- inline_bind` asserts every bound circuit is a strict
superset of its baseline (same lookups, +2 gates, +6/+2NC advice columns), so a
drift between the pair fails the test suite instead of silently skewing
measurements.

### In-circuit binding cost (`cargo commit-diff`)

The additional cost of the commitment layer measured the way Appendix A
describes it -- the witness-equality check INLINED into the query circuit, one
proof covering both. Each row is proved twice (base and bound) at the same
degree; `in-circ` is the median of the paired differences.

The two commands for the full test:

```bash
# queries that materialize no intermediates (privacy setting irrelevant)
cargo commit-diff reps=3 q3 q8 q9 q18 gq1 gq2

# queries that materialize bags (Q5, GQ3, GQ4): measure at Revealing-Join-Size
VPJOIN_PRIVACY=rjs cargo commit-diff reps=3 q5 gq3 gq4
```

Notes:

* **`in-circ` is a difference of two large proof times**, so pin to one NUMA
  node for publishable numbers (floor drops from +/-1.5 s to +/-0.2 s on this
  4-node server). `numactl` must wrap the binary, not cargo:

  ```bash
  numactl --cpunodebind=0 --membind=0 ./target/release/commit_diff reps=3 q3 q8 q9 q18 gq1 gq2
  VPJOIN_PRIVACY=rjs numactl --cpunodebind=0 --membind=0 ./target/release/commit_diff reps=3 q5 gq3 gq4
  ```
* Measure the noise floor first with the control experiment (proves the base
  circuit against itself; the true answer is zero, so whatever it reports is
  bias): `VPJOIN_SELFTEST=1 cargo commit-diff reps=2 q18`.
* **The gq3/gq4 rows on facebook/wiki run at k=22-23 and cost hours each**
  (the inlined check inherits the query circuit's bag-inflated domain even
  though it binds only 2 Edge columns). For those rows the separate-circuit
  measurement (`vpjoin_bench commit`, ~1 s at k=15-17) is both far cheaper and
  the more meaningful number; if you do run them inlined, say which framing the
  reported number uses.
* `reps=N` on the command line overrides `VPJOIN_REPS`; `gq2:wiki` selects one
  dataset; `VPJOIN_VERBOSE=1` prints each repetition's raw base/bound times.

### Public Parameter Selection (k)

Select appropriate Halo2 public parameter k depending on dataset size and SQL queries.

Note: The necessary parameters are already persisted as param15 through param22. Please be aware that manually configuring the degree (k) within the source code is only supported when using the MockProver.



## Revision Experiments (SIGMOD revision artifacts)

All revision additions are **additive**: no pre-existing query circuit is modified, so each
layer's cost is measured separately and added on top of a query's cost.

### 1. Database-commitment layer (Appendix A)

Canonicalizes the database into a fixed layout shared across queries, publishes a single
Pedersen vector commitment `Commit(D)`, and binds every query proof to it via an IPA opening
at a Fiat-Shamir challenge derived from `(Commit(D), layout, proof)`.

```bash
# correctness tests
cargo test --release --lib commitment::tests -- --nocapture

# cost measurement on the real 60K TPC-H data (setup + per-query binding, median of 5)
cargo run --release --bin commitment_bench -- src/data lineitem orders customer supplier nation part partsupp

# 120K / 240K scales (only lineitem scales, per the inherited PoneglyphDB setup)
cargo run --release --bin commitment_bench -- src/data lineitem_120K orders customer supplier nation part partsupp
cargo run --release --bin commitment_bench -- src/data lineitem_240K orders customer supplier nation part partsupp
```

### 2. In-circuit input binding + per-column commitments (Appendix A)

`src/column_commit.rs` commits **each attribute column separately, over the same
evaluation domain the query circuits already use** (`k = 16` for the TPC-H circuits, which
holds the 60K-row `lineitem`).  This reuses the circuits' existing `ParamsIPA` verbatim (no
separate large setup), gives one published commitment per circuit witness column, and uses a
fresh random blinder per column so the published commitments are **hiding**.  (Halo2's
*fixed*-column commitments use the constant `Blind(1)` and are therefore reproducible from
the data -- not suitable for a private database.)

Binding is a random-point check: a Fiat-Shamir challenge `x` is derived from the published
commitments, the layout and the query proof; the prover opens each committed column at `x`
(IPA opening, revealing `v_j`), and the circuit evaluates its own witness column at `x` by a
Horner accumulation and exposes `v_j` as a public output.  Matching values certify that the
**query circuit's witness columns equal the committed data**.

```bash
cargo test --release --lib column_commit

# PRIMARY command -- per-QUERY additional cost: for each VPJoin query, binds exactly
# the columns that query's circuit witnesses (inventoried from the circuit code), in
# that circuit's own domain (k=16 TPC-H / k=17 GQ1-GQ2 / k=18 GQ3-GQ4), and measures
# baseline vs bound plus openings, into a CSV.  Graph queries run on all 3 SNAP datasets.
cargo run --release --bin query_commit_bench -- results/query_commit_cost.csv
# or a subset:
cargo run --release --bin query_commit_bench -- results/tpch.csv q3 q5 q8 q9 q18

# per-TABLE variant (all tables in one shared k):
cargo run --release --bin commit_cost_bench -- src/data results/commit_cost.csv \
    nation supplier customer orders lineitem
```

Per-query bound-column counts (from the circuits' witness code): Q3 10, Q5 16, Q8 19,
Q9 17, Q18 8; GQ1-GQ4 bind Edge(src,dst) = 2 columns per dataset.

The CSV (`results/commit_cost.csv`) has one row per table with:
`rows, cols, k, setup_params_s, setup_commit_s, published_commit_bytes,`
`baseline_{keygen,prove,verify}_s, baseline_proof_bytes,`
`bound_{keygen,prove,verify}_s, bound_proof_bytes,`
`open_{prove,verify}_s, open_proof_bytes,`
`delta_prove_s, delta_verify_s, delta_proof_bytes, total_extra_prove_s`.

All tables share ONE domain `k`, derived from the largest table in the set: `lineitem`
(60,175 rows) gives **k = 16**, the degree the TPC-H circuits actually run at.  This is the
realistic setting -- a query that reads `lineitem` *is* a k=16 circuit, so the dimension
tables it also reads live in the same `2^16` domain and their openings cost the same as
`lineitem`'s.  (The IPA generators are position-indexed and k-independent, so a column
commits to the same point at any k that holds it -- see the
`commitment_point_is_independent_of_k` test; only the opening cost changes.)  Parameters are
generated once and shared, as the circuits share `param16`.

The `delta_*` columns are the additional in-circuit cost of the commitment layer over the
pure query circuit; `total_extra_prove_s` adds the column openings.  Measured at k=16:

| table | cols | commit setup | published | baseline prove | bound prove | Δ prove | openings |
|---|---|---|---|---|---|---|---|
| nation | 4 | 0.017 s | 128 B | 0.77 s | 2.37 s | +1.60 s | 1.00 s |
| supplier | 7 | 0.019 s | 224 B | 0.75 s | 2.39 s | +1.64 s | 1.74 s |
| customer | 8 | 0.030 s | 256 B | 0.77 s | 2.54 s | +1.77 s | 1.98 s |
| orders | 9 | 0.028 s | 288 B | 0.79 s | 2.43 s | +1.64 s | 2.23 s |
| lineitem | 16 | 0.075 s | 512 B | 0.87 s | 2.74 s | +1.86 s | 3.96 s |

Note how to read these: at a fixed k the circuit cost is dominated by the domain, not by the
table's row count, so the in-circuit delta is ~1.6--1.9 s for every table; the **openings**
are what scale with the number of columns (~0.25 s per column at k=16).  The
`delta_prove_s` values must **not** be summed across tables -- a query uses a *single* k=16
circuit holding all the columns it reads, so its in-circuit binding is charged once, growing
with the total column count (the `lineitem` row, 16 bound columns, is the practical upper
bound).  The openings *do* add up, but only over the columns a query actually reads.

The earlier monolithic single-vector layer (`src/commitment.rs`, one 2^21 commitment for the
whole database) is retained for reference and benchmarked by `commitment_bench`; it needs a
separate large setup and cannot be matched to circuit columns without reindexing, so
`column_commit` supersedes it.

### 3. DP capacity generation (Appendix G.2)

Two-stage one-sided mechanism: noisy truncation thresholds for the max join-key
frequencies (sensitivity 1 each), then the capacity calibrated to the released thresholds
(their *sum* for self-join bags such as Edge|x|Edge).  Every release is a single draw;
total budget by basic composition.  This binary is the standalone reference calculator; the
benches no longer need it, since `bench_queries::q5_pads` / `graph_pads` release the
capacities from the circuits' own bag derivations at witness-generation time (section 5),
and the legacy hard-coded `*_pad_extra` constants it was written to replace are gone from
the harnesses (they survive only as `VPJOIN_PRIVACY=legacy`, recorded in
`dp/legacy_capacities.md`).  Note that the two-stage form below is what the *graph* queries
use; Q5's fan-out bounds come from unprotected relations and so are released exactly, with
no threshold stage — see section 5.

```bash
python3 dp/noise_generator.py          # reference implementation + self-check
cargo test --release --lib dp_noise    # Rust twin tests

# per-bag capacities for a single epsilon (total budget (0.1, 1e-5), 2 self-join bags):
cargo run --release --bin dp_capacity_gen -- 0.1 1e-5 2  120000 7 4 1  118000 7 4 1

# The FULL epsilon sweep the paper reports (Figures for "various epsilon"),
# delta = 1e-5, in one run -- the first argument is a comma-separated list:
cargo run --release --bin dp_capacity_gen -- \
    0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10 1e-5 2  120000 7 4 1  118000 7 4 1
```

Each epsilon prints, per bag, `capacity` and `pad_extra = capacity - true_size`.  The paper's
default is epsilon = 0.1; the swept values are `0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10`.

### 4. PoneglyphDB-style graph baselines (Section 8.1 estimation methodology)

The released PoneglyphDB artifact covers only TPC-H, so `graph_sql/pone_baseline.rs`
implements its binary-join-chain strategy for GQ1-GQ4 (every intermediate materialized at a
fixed capacity, verified by the same lookup-argument machinery).  The bench runs the
*anchor* execution at true intermediate sizes (measured N_0, G_C0, T_0), reports the
worst-case padded size (N, G_C), and prints the extrapolated T_est = T_0 * G_C / G_C0.

All 4 graph queries on all 3 datasets, one command.  **Omit `--release`** so the anchor is
measured under the same unoptimized profile as the paper's other proving times
(`cargo test ... qN_obj::tests::test_1`); mixing the two profiles in one comparison inflates
the ratio by roughly 5x:

```bash
PONE_K0=17 cargo run --bin pone_graph_bench
```

The default output is deliberately minimal — one header line plus one line per
(query, dataset) with the measured anchor proving time:

```
query dataset    anchor_prove_s
gq1   lastfm             105.32
```

Everything else (banner, plan table, per-level true intermediate sizes, the
extrapolation arithmetic and the trailing summary) is behind `PONE_VERBOSE=1`.
Check the plan without proving anything (instant), restrict to a subset, or add a
`.csv` argument to also write the full row (anchor shape, worst-case bounds, both
extrapolations) to a file:

```bash
PONE_PLAN_ONLY=1 cargo run --bin pone_graph_bench
```

```bash
PONE_VERBOSE=1 cargo run --bin pone_graph_bench -- gq3 gq4 wiki
```

```bash
cargo run --bin pone_graph_bench -- gq3 wiki results/pone_gq3.csv
```

The anchor **must** subsample: `|P_3|` is 79M rows on Facebook and 202M on Wiki, so the
unpadded execution does not fit at full scale.  The harness therefore takes the largest
prefix of the edge list that still fits a `2^PONE_K0` domain (default `k0=17`, giving
anchors of 2.6K-15K edges), and — this is the part that must not be got wrong — derives the
worst-case `N`, `G_C` from the **full** dataset regardless.  The anchor's only job is to
measure seconds per domain row for this circuit shape; the padded circuit it is scaled to is
the one over the real graph.  Raise `PONE_K0` for a tighter anchor (a larger anchor amortizes
fixed overheads better, which is *more* conservative in PoneglyphDB's favor); force a
specific subsample with `PONE_EDGES=n`.

Anchor and padded circuit share the same row-count formula (`pone_baseline::circuit_rows`)
and the same column count, so `T_0 / G_{C,0}` is a per-domain-row rate for an unchanged
shape.  Note that the levels occupy **disjoint column groups of one region**, so the row
count is the max of the per-level heights, not their sum.

**Two worst-case bounds are reported.**

| bound | applies when | GQ1/GQ3 lastfm | source |
|---|---|---|---|
| `m^t` (the paper's) | bag semantics, no key constraint (the evaluated setting) | `G_C = 2^45` | `worst_case_rows` |
| AGM `m^ceil((t+1)/2)` | set semantics (publicly assumed edge distinctness) | `G_C = 2^30` | `worst_case_rows_agm` |

`m^t` is not merely valid but **tight** in the paper's setting: the graph queries are SQL
queries (bag semantics) with no uniqueness constraint on the edge relation, and an instance
that concentrates its multiplicity on a single t-edge path attains `(m/t)^t = Theta(m^t)`.
The AGM bound (fractional edge cover `ceil((t+1)/2)` of a t-edge path) applies only if a key
constraint on the edges is publicly assumed, which is exactly the class of schema assumptions
the paper's evaluation excludes (Parameter Setting paragraph of the experiments section);
padding to it without that constraint would under-provision on duplicate-heavy instances.
The harness reports both because the paper's robustness claim quotes them: even under AGM
the padded domains span 2^30 to 2^50, all beyond the largest runnable domain (2^23), and the
speedups stay at or above three orders of magnitude.

The baseline computes the **same query** as VPJoin with the **same per-row join machinery**:
every materialized row is bound to one specific (parent, edge) occurrence pair through
`g_sql4_obj::IndexedViewChip`, the identical PermAny-bound sorted-view chip VPJoin's own bag
materializations use, and the ascending-vertex predicates (`a<b<c`, plus `c<d` for the
4-edge queries) are enforced with the same 8-byte `LtChip` comparisons.
`tests::counts_match_the_vpjoin_ground_truth` pins the baseline's counts to
`bench_queries::count_gq1..4`, and two adversarial tests pin the gates: unordered rows and
mis-indexed rows are both rejected.  This matters for the anchor: an under-constrained
baseline is cheaper per row than PoneglyphDB really is, and the extrapolation multiplies
that error by `G_C / G_{C,0}`.  The shapes are genuinely comparable now -- 131 advice / 70
lookups against gq1's 141 / 84, and a measured debug `T_0` of ~195 s at `2^17` against
VPJoin's ~251 s.  That 0.78 ratio matches the released PoneglyphDB artifact's own TPC-H
per-row cost (about 0.75x of VPJoin's); the residual gap is PoneglyphDB's genuine advantage
(simpler gates, constraint degree 5 vs 7), not missing constraints.

Edge parsing matches `bench_queries::load_graph` exactly, so the baseline sees the same edge
multiset as VPJoin: no symmetrization, and no duplicate edges in any of the three files.
The ordering predicates are a no-op on LastFM and Facebook, which store every edge as
`src < dst`, so only Wiki's anchor sizing changes (its intermediates roughly halve).
LastFM and Facebook store each undirected edge once with `src < dst`, so no directed cycle
exists and GQ3/GQ4 report a count of 0 on them — matching VPJoin's own `count_gq3`/`count_gq4`.
That does not affect the cost anchor: every `|P_t|` row is still materialized and constrained,
and the closure gate only marks the non-closed ones as dummies.

```bash
cargo test --release --lib pone_baseline
```

### 5. DP-guided padding: mechanism, lane circuits, and the epsilon sweep

The originally reported DP runs used hand-chosen constants (recorded in
`dp/legacy_capacities.md`).  Both the release mechanism and the circuits that host the
released capacity have since been rebuilt; `dp_lane_bench` drives all of it from one
command, in the same shape as `cargo commit-diff`:

```bash
# the default budget (eps = 0.1, delta = 1e-5), all three bag-materializing queries
cargo run --bin dp_lane_bench -- reps=3 q5 gq3 gq4

# the no-privacy lower bound, same binary and table
VPJOIN_PRIVACY=rjs cargo run --bin dp_lane_bench -- reps=3 q5 gq3 gq4

# one point of the epsilon sweep; vary VPJOIN_DP_SEED=1..10 for the paper's 10 rounds
VPJOIN_EPS=0.01 VPJOIN_DP_SEED=1 cargo run --bin dp_lane_bench -- reps=3 q5 gq3 gq4

# geometry only (degrees, lane counts, released capacities, pads) -- nothing is proved
VPJOIN_PLAN_ONLY=1 cargo run --bin dp_lane_bench
```

`reps=N` may appear anywhere in the argument list; keys are built once outside the timed
region, every proof is verified, and nothing is written to disk.  Selectors take an
optional dataset (`gq3:lastfm`); `q5` always runs on tpch-60K.

**The privacy policy is now declared per workload** (`Privacy::Dp`, `src/bench_queries.rs`).
Q5 uses row-level neighbors with `P = {customer, supplier}`: the fan-out bounds that
govern its sensitivity (`tau_C = 32` orders per custkey, `tau_S = 668` lineitems per
suppkey) live in the *unprotected* relations, so they are identical on every neighboring
instance and are released exactly at no budget cost.  Because a customer row reaches the
LS bag through `c_nationkey = s_nationkey` as well as through custkey, the LS release
takes the full epsilon and the CO release takes `eps * (1 - 43/668)`; delta splits in half.
The graph queries protect one edge tuple, and there the maximum degree *is* a statistic of
the protected relation, so it must itself be released under the one-sided mechanism before
it can calibrate the capacity release — the two-stage form the appendix describes.  That
asymmetry is the whole rule: noise the frequency bound only when it depends on protected
data.  `cargo test --test freq_noising_cost -- --nocapture` prints what the second stage
costs (2--4.5x on the graphs).

**Lane circuits keep `k` fixed.**  A released capacity larger than the current domain would
otherwise force the next power of two, quantizing proving time into 2x jumps.  The `_dp`
circuits instead host the capacity in `c = ceil(capacity / lane_rows)` parallel column-group
lanes at the Revealing-Join-Size degree:

| file | query | lanes | growth |
|---|---|---|---|
| `src/sql/q5_obj_dp.rs` | Q5 | LS pipeline | linear, ~8% per lane |
| `src/graph_sql/g_sql3_obj_dp.rs` | GQ3 | Bag1 only (Bag2 is the Edge relation, public size) | linear, +51 advice per lane |
| `src/graph_sql/g_sql4_obj_dp.rs` | GQ4 | both bags | **quadratic**, see below |

Every lane is a full structural replica with identical gates and lookups, and the lane
count is a function of the *released* capacity only — never of the true bag size.  A
cheaper padding-only overflow lane would leak the true size through the circuit shape and
is deliberately not implemented.  `cargo test --test graph_lane_plan -- --nocapture` prints
the plan per (query, dataset, epsilon) and `cargo test --test lane_cost_probe -- --nocapture`
measures how the shape actually grows.

**GQ4 is a known negative result.**  Resolving a private key against private-size tables
needs one lookup argument per candidate table, so `c2` lane-local message maps force
`c1 * c2` probe replicas.  At the LastFM eps=0.1 cell (`c1=4, c2=3`) the laned circuit is
580 advice columns at `2^18` = 152.0M advice cells, against 141 columns at `2^20` = 147.8M
unlaned: a wash.  `g_sql4_obj_dp.rs` is correct and adversarially reviewed, but it is not
evidence of linear GQ4 cost, and should not be cited as such.  The linear route is a
sort-merge redesign that drops the message map (both GQ4 bags have identical row shape);
that is a different circuit and has not been built.

At eps = 0.1 the released pads fit the existing headroom on Facebook and Wikipedia for both
graph queries (one lane, no degree change), so DP-guided padding is free there; only LastFM
pays, at 2 lanes for GQ3 and 4/3 for GQ4.

NOTE — the paper's DP figures were produced from the legacy hand-tuned constants, not from
this mechanism.  `2848 = 32 * 89` and `59452 = 668 * 89` decompose exactly into Q5's
sensitivities times a noise multiplier of 89, whereas `(eps, delta) = (0.1, 1e-5)` mandates
109.2; the legacy pair is therefore a low-tail draw of the current distribution, not its
mean.  Regenerate every DP curve before comparing, and re-check the "2--10% overhead"
text: with the corrected pads the graph overhead at eps = 0.1 is 0% on Facebook and
Wikipedia but substantially larger on LastFM.  `cargo test --test q5_appendix_mechanism --
--nocapture` shows what the appendix's verbatim two-stage mechanism would cost Q5 instead
(1.08M rows at eps = 0.1, which exceeds its own worst-case bound below eps ~ 0.03).

### 6. End-to-end comparison with the commitment layers

For the reviewer-requested "effect of adding the commitments" comparison, per query and
scale: (a) run the query proof as usual; (b) add the per-query line of `commitment_bench`
(the across-proof binding to the published Commit(D)); (c) add the per-relation lines of
`input_binding_bench` for the tables the query reads (the in-circuit input check).  The
setup lines of both benches are one-time costs amortized over all queries.

### Paper build

`latex/zk.tex` is the revised paper (compile with the `review` option for margin line
numbers); `latex/letter.tex` is the revision letter.  Reviewer colors: blue = R1,
violet = R2, teal = R3.
