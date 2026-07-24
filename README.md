# VPJoin

**VPJoin** is a zero-knowledge proof framework for verifiable multi-way SQL join processing.
Built on the [Halo2](https://github.com/zcash/halo2) proving system with PLONKish
arithmetization, it lets a data holder prove the correct execution of complex SQL queries
over private data without revealing the underlying records.

## Problem

Existing ZK database systems decompose multi-way joins into sequential binary joins,
materializing each intermediate result inside the proof circuit. Because circuit dimensions
are public, intermediate sizes leak cardinality information, and padding them to worst-case
sizes leads to **O(IN^k) circuit blowup** for a k-way join. For TPC-H Q8 (an 8-way join),
that translates to an estimated proving time exceeding 10^19 seconds.

VPJoin eliminates this bottleneck via **witness-guided verification**: the prover performs
all join work offline and supplies a witness marking which input tuples participate in the
result. The circuit only *verifies* the witness through lightweight structural checks, never
materializing intermediate results.

## Key Contributions

### Oblivious Join Gate (OBJ) for Acyclic Joins

OBJ verifies multi-way joins via semijoin-based structural checks along a join tree,
achieving **worst-case O(IN + OUT) circuit complexity**. Since the circuit never creates
intermediate results, its layout depends only on input table sizes, so **obliviousness is
achieved entirely for free** with zero padding overhead. The gate enforces four properties:

- **Conservation** -- every original tuple appears in exactly one group (permutation argument)
- **Disjointness** -- clean and residual groups share no tuples (non-membership check)
- **Pairwise Consistency** -- clean tuples in neighboring relations agree on join keys (lookup arguments)
- **Completeness** -- no valid join result hides among the residuals (semijoin pass)

### Aggregation Without Join Materialization

OBJ computes join-aggregates directly over compact clean relations via **tuple
multiplicities**, keeping proof cost at O(IN + OUT) without ever constructing the full
O(IN^k)-sized join result.

### Tree-Decomposed Join Gate (TDJ) for Cyclic Joins

TDJ extends OBJ to cyclic queries via **tree decomposition**: intra-cluster joins are
materialized at fixed capacities and the acyclic inter-cluster structure is verified by OBJ.
In its default mode the capacities are worst-case bounds, so cyclic queries remain **fully
oblivious**. As an *optional* relaxation, **DP-guided padding** sets each capacity with a
one-sided noise mechanism that always exceeds the true size (preserving correctness) while
giving formal (epsilon, delta)-differential privacy for the intermediate cardinalities.

## Project Structure

```
src/
  chips/          # Custom Halo2 gate implementations (hash, lookup, comparison)
  circuits/       # Reusable circuit gadgets (inclusion checks, permutations, Merkle trees)
  sql/            # TPC-H query circuits (Q3, Q5, Q8, Q9, Q18)
  graph_sql/      # Graph pattern query circuits (GQ1--GQ4)
  data/           # TPC-H dataset files and parsing utilities
  graph_data/     # Network dataset files and parsing
  proof/          # Persisted public parameters (param15..param23) and proof artifacts
  bench_queries.rs  # Shared query/dataset/privacy plumbing for every harness
  dp_noise.rs       # DP capacity release (one-sided noise mechanism)
  dp_lane.rs        # Plan/run records shared by the DP lane circuits
  commitment.rs     # Database-commitment layer: canonical layout, Commit(D), per-proof binding
  input_binding.rs  # In-circuit binding of query inputs to the published commitments
  bin/              # Benchmark harnesses (see Running the Benchmarks)
```

Each cyclic query has three circuit files: the baseline (`q5_obj.rs`, `g_sql3_obj.rs`,
`g_sql4_obj.rs`), a `_bound` variant with the witness-binding check inlined, and a `_dp`
variant that holds the degree fixed and hosts the DP-released capacity in parallel lanes.

## Benchmarks

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

Datasets: **TPC-H** (60K rows in lineitem), **LastFM** (27,806 edges), **Facebook**
(88,234 edges), **Wikipedia Vote** (103,689 edges). Graph queries run on all three networks.

## Getting Started

### Prerequisites

- **Rust**, stable toolchain
- A large stack for Halo2 proving:

  ```bash
  export RUST_MIN_STACK=33554432
  ```

- Public parameters: `param15` through `param19` ship in `src/proof/`. Parameters for degrees outside this range are generated and cached on first use because the corresponding files may be too large to upload to GitHub.
- Memory: the graph queries at `k = 22` need tens of GB. Pin to one NUMA node for stable
  timings (`numactl --cpunodebind=0 --membind=0 ./target/release/<bin> ...`).

### Build

```bash
cargo build --release
```

## Running the Benchmarks


**1-2. VPJoin proving time**, 5 TPC-H queries plus 4 graph queries on 3 datasets, without
and with the database-commitment layer:

```bash
cargo run --bin vpjoin_bench -- baseline
```

```bash
cargo run --bin vpjoin_bench -- full
```

**3-4. Additional in-circuit cost of binding a proof to a committed database.** Each row is
proved twice at the same degree and the reported cost is the median paired difference:

```bash
cargo commit-diff reps=3 q3 q8 q9 q18 gq1 gq2
```

```bash
VPJOIN_PRIVACY=rjs cargo commit-diff reps=3 q5 gq3 gq4
```

**5-6. DP-guided padding** for the three cyclic queries, and its no-privacy lower bound
(capacities set to the true bag sizes):

```bash
cargo run --bin dp_lane_bench -- reps=3 q5 gq3 gq4
```

```bash
VPJOIN_PRIVACY=rjs cargo run --bin dp_lane_bench -- reps=3 q5 gq3 gq4
```

**7. The privacy budget sweep.** `VPJOIN_EPS` takes a comma-separated list, so one
invocation covers every point; vary `VPJOIN_DP_SEED` for independent rounds. A budget
whose released capacity needs more lanes than the circuit can host is reported as
SKIPPED with the reason and the sweep continues:

```bash
VPJOIN_EPS=0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10 VPJOIN_DP_SEED=1 cargo run --bin dp_lane_bench -- reps=3 q5 gq3 gq4
```

**8. PoneglyphDB-style graph baselines.** Runs the binary-join-chain baseline at true
intermediate sizes (the measured anchor) and reports its proving time; the worst-case
extrapolation is derived from it:

```bash
PONE_K0=17 cargo run --bin pone_graph_bench
```

