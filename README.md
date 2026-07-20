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

### 2. In-circuit input binding (Appendix A)

Proves *inside a circuit* that the input columns a query consumes equal the committed
tables: the committed table lives in fixed columns (whose per-column commitments are
published at setup and reproduced verbatim in the verifying key), and an equality gate binds
the advice inputs to them.  Composing the same constraints into a query circuit enforces the
binding at the same per-cell cost, so the measured numbers are the additional cost of
checking the input commitments before query processing.

```bash
cargo test --release --lib input_binding
cargo run --release --bin input_binding_bench -- src/data nation customer orders lineitem
```

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
# per-bag capacities: total budget (0.1, 1e-5) split over 2 self-join bags
cargo run --release --bin dp_capacity_gen -- 0.1 1e-5 2  120000 7 4 1  118000 7 4 1
```

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
2. Release the capacities, dividing the query's total budget across its bags
   (`self_join = 1` for the Edge-Edge bags of GQ3/GQ4):

```bash
# GQ3 / GQ4 (two self-join bags):
cargo run --release --bin dp_capacity_gen -- 0.1 1e-5 2  <bag1_size> <mfA> <mfB> 1  <bag2_size> <mfA> <mfB> 1
# Q5 (two bags over distinct relations):
cargo run --release --bin dp_capacity_gen -- 0.1 1e-5 2  <co_size> <mfO> <mfC> 0  <ls_size> <mfL> <mfS> 0
```

3. Plug the released capacities into the circuit padding knobs as
   `pad_extra = capacity - true_size`:
   - GQ3: `bag1_pad_extra` / `bag2_pad_extra` in the test harness of `src/graph_sql/g_sql3_obj.rs`
   - GQ4: `bag1_pad_extra` / `bag2_pad_extra` in the test harness of `src/graph_sql/g_sql4_obj.rs`
   - Q5:  `co_pad_extra` / `ls_pad_extra` in the test harness of `src/sql/q5_obj.rs`
   (or call `halo2_experiments::dp_noise::dp_join_capacity` directly at witness-generation
   time instead of hard-coding).
4. Re-run the corresponding query proofs (commands under "Running TPC-H/Graph Query Proofs")
   once per epsilon value to regenerate the privacy--efficiency curves.

Note: at small total epsilon (e.g. 0.1) the rigorous mechanism produces much larger
capacities than the legacy constants; expect the small-epsilon end of the DP curves to rise.

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
