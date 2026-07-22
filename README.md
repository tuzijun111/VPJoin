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
  graph_sql/pone_baseline.rs  # [revision] PoneglyphDB-style binary-join-chain baselines for GQ1-GQ4
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
**each** of the 3 SNAP datasets (17 rows), writing one CSV.

```bash
# (A) BASELINE -- the pure query circuits, i.e. what the submission reports.
cargo run --release --bin vpjoin_bench -- baseline results/vpjoin_baseline.csv

# (B) FULL -- the same circuits plus the complete commitment layer: published
#     per-column Pedersen commitments to the dataset, the in-circuit check that the
#     query's witness columns equal the committed data, and the column openings.
cargo run --release --bin vpjoin_bench -- full results/vpjoin_full.csv
```

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

Neither is "wrong" — they are the same computation under two compilers. Every CSV row
records a `profile` column (`debug`/`release`) and the binary prints the profile at startup,
so a mixed comparison is visible rather than silent. **Use one profile for both commands.**
To reproduce the `cargo test` timings with this harness, just drop `--release`:

```bash
cargo run --bin vpjoin_bench -- baseline results/vpjoin_baseline_debug.csv q3
```

The CSV also reports `vk_s` and `pk_s` separately, which line up 1:1 with the
`Time to generate vk` / `Time to generate pk` lines the tests print, plus `load_s` (data
parsing) and `wall_s` (end-to-end, comparable to the test's `finished in ...` line).

The difference between the two CSVs is exactly the cost of making a proof bind to a
committed database instead of to an unauthenticated input. Both write the same schema
(`query,dataset,k,input_rows,n_columns,public_output,keygen_s,prove_s,verify_s,proof_bytes,
commit_setup_s,published_bytes,bind_*,open_*,total_prove_s,total_proof_bytes,status`);
in `baseline` the commitment columns are zero.

Useful variations:

```bash
# a subset of queries (graph queries still expand over all 3 datasets)
cargo run --release --bin vpjoin_bench -- baseline results/tpch.csv q3 q5 q8 q9 q18
cargo run --release --bin vpjoin_bench -- full    results/graph.csv gq1 gq2 gq3 gq4

# point at a different data root (must contain data/ graph_data/ proof/)
VPJOIN_DATA=/path/to/src cargo run --release --bin vpjoin_bench -- baseline out.csv

# cap the degree the k-fitter may try (default 21)
VPJOIN_MAX_K=19 cargo run --release --bin vpjoin_bench -- baseline out.csv
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
* Q5 keeps the DP padding knobs at the values currently set in its test
  (`nr/co/ls_pad_extra = 0 / 32*89 / 668*89`, see `dp/legacy_capacities.md`), and GQ3's
  bag padding is applied per dataset rather than fixed at the lastfm constant.

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
total budget by basic composition.  Replace the legacy hard-coded `*_pad_extra` constants
(recorded in `dp/legacy_capacities.md`) by these outputs, or call
`halo2_experiments::dp_noise::dp_join_capacity` directly at witness-generation time.

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

```bash
cargo test --release --lib pone_baseline
# quick smoke run on a subsample ('sym' inserts both edge directions; the count subsamples edges)
cargo run --release --bin pone_graph_bench -- gq3 src/graph_data/facebook/facebook_combined.txt 2000 sym
```

Full-scale anchors for every query/dataset pair of Figure 3 (the three SNAP datasets ship in
`src/graph_data/`; the loader accepts whitespace- or comma-separated edge lists and skips
headers/comments).  Each run prints the anchor `N_0`, `G_C0`, `T_0`, the worst-case `N`,
`G_C`, and the extrapolated `T_est` — these are the values for the Section 8.1 anchor table:

```bash
for q in gq1 gq2 gq3 gq4; do
  cargo run --release --bin pone_graph_bench -- $q src/graph_data/last/lastfm_asia_edges.csv sym
  cargo run --release --bin pone_graph_bench -- $q src/graph_data/facebook/facebook_combined.txt sym
  cargo run --release --bin pone_graph_bench -- $q src/graph_data/wiki/wiki_Vote.txt sym
done
```

Warning: anchor cost scales with the true intermediate sizes; on the dense Facebook graph
the higher levels (gq2/gq4) can be very large.  Use a `[max_edges]` subsample first to gauge
the size (the true intermediate sizes are printed before proving starts).

### 5. Regenerating the DP experiments with the corrected mechanism

The originally reported DP runs used hand-chosen constants (recorded in
`dp/legacy_capacities.md`).  To regenerate with the rigorous mechanism:

1. Obtain each bag's true join size and per-side maximum join-key frequencies (printed by the
   witness-generation code, or computed offline from the data).
2. Release the capacities for the whole epsilon sweep at once, dividing the query's total
   budget across its bags (`self_join = 1` for the Edge-Edge bags of GQ3/GQ4, `0` for Q5's
   distinct-relation bags):

```bash
# GQ3 / GQ4 (two self-join bags), full sweep:
cargo run --release --bin dp_capacity_gen -- \
    0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10 1e-5 2 \
    <bag1_size> <mfA> <mfB> 1  <bag2_size> <mfA> <mfB> 1
# Q5 (two bags over distinct relations), full sweep:
cargo run --release --bin dp_capacity_gen -- \
    0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10 1e-5 2 \
    <co_size> <mfO> <mfC> 0  <ls_size> <mfL> <mfS> 0
```

3. For each epsilon, plug that epsilon's `pad_extra` values into the circuit padding knobs:
   - GQ3: `bag1_pad_extra` / `bag2_pad_extra` in the test harness of `src/graph_sql/g_sql3_obj.rs`
   - GQ4: `bag1_pad_extra` / `bag2_pad_extra` in the test harness of `src/graph_sql/g_sql4_obj.rs`
   - Q5:  `co_pad_extra` / `ls_pad_extra` in the test harness of `src/sql/q5_obj.rs`
   (or call `halo2_experiments::dp_noise::dp_join_capacity` directly at witness-generation
   time instead of hard-coding).
4. Re-run the corresponding query proofs (commands under "Running TPC-H/Graph Query Proofs")
   once per epsilon value; the proving time scales with the padded circuit size, so one run
   per epsilon regenerates the privacy--efficiency curve.

NOTE — the paper's originally reported DP figures were produced from the legacy hand-tuned
constants (`dp/legacy_capacities.md`), not from this mechanism.  The mechanism here is a
rigorous (epsilon, delta)-DP release: the join-size sensitivity is `max(F_A, F_B)` (with the
DP-released frequency upper bounds), and the one-sided noise has scale ~ sensitivity *
ln(1/delta)/epsilon.  The resulting overhead therefore depends on (i) the real ratio of the
sensitivity to the true bag size and (ii) how the total budget is split across the query's
bags and the three per-bag releases (thresholds + size); the equal split used by the CLI is
one choice, not the only one.  Regenerate the curves with the real bag sizes and max
frequencies before comparing to the submitted figures, and re-check the "2--10% overhead"
text against the regenerated numbers.  If the overhead at small epsilon is larger than
desired, options include a less aggressive budget split, a larger delta, the Gaussian
mechanism, or reporting a larger operating epsilon.

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
