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

- Public parameters: `param15` through `param23` ship in `src/proof/`. A degree outside that
  range is generated and cached on first use.
- Memory: the graph queries at `k = 22` need tens of GB. Pin to one NUMA node for stable
  timings (`numactl --cpunodebind=0 --membind=0 ./target/release/<bin> ...`).

### Build

```bash
cargo build --release
```

## Running the Benchmarks

All harnesses print their results to stdout as an aligned table; none writes a results file.
`cargo run` **without** `--release` builds the debug profile, which is several times slower
but is the profile all reported proving times use. Do not mix the two in one comparison.

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

**7. One point of the privacy budget sweep.** Vary `VPJOIN_EPS` per point and
`VPJOIN_DP_SEED` for independent rounds:

```bash
VPJOIN_EPS=0.01 VPJOIN_DP_SEED=1 cargo run --bin dp_lane_bench -- reps=3 q5 gq3 gq4
```

**8. PoneglyphDB-style graph baselines.** Runs the binary-join-chain baseline at true
intermediate sizes (the measured anchor) and reports its proving time; the worst-case
extrapolation is derived from it:

```bash
PONE_K0=17 cargo run --bin pone_graph_bench
```

### Options

Preview any run's geometry -- degrees, lane counts, released capacities, pads -- without
proving anything:

```bash
VPJOIN_PLAN_ONLY=1 cargo run --bin dp_lane_bench
```

| Variable | Applies to | Meaning |
|---|---|---|
| `VPJOIN_PRIVACY` | all | `dp` (default), `rjs` (true sizes), `legacy` (fixed constants) |
| `VPJOIN_EPS` / `VPJOIN_DELTA` | DP runs | privacy budget, default `0.1` / `1e-5` |
| `VPJOIN_DP_SEED` | DP runs | noise seed; vary it for independent rounds |
| `VPJOIN_PLAN_ONLY` | `vpjoin_bench`, `dp_lane_bench` | print the plan and exit |
| `VPJOIN_DATA` | all | data root (must contain `data/`, `graph_data/`, `proof/`) |
| `VPJOIN_MAX_K` | `vpjoin_bench` | cap on the degree the k-fitter may try (default 21) |
| `PONE_K0` / `PONE_EDGES` | `pone_graph_bench` | anchor domain exponent / forced subsample |
| `PONE_VERBOSE` / `PONE_PLAN_ONLY` | `pone_graph_bench` | full diagnostics / plan only |

Every harness accepts a subset of queries, and graph queries accept a dataset suffix:

```bash
cargo run --bin dp_lane_bench -- reps=3 gq3:lastfm gq4:lastfm
```

Appending an argument ending in `.csv` to `vpjoin_bench` or `pone_graph_bench` additionally
writes the full per-row schema to that file.

## DP-Guided Padding

The privacy policy is declared per workload in `bench_queries::q5_pads` and
`bench_queries::graph_pads`, and the released capacity feeds the circuit as an input.

**Q5** uses row-level neighbors with `P = {customer, supplier}`. The fan-out bounds that
govern its sensitivity (32 orders per custkey, 668 lineitems per suppkey) live in the
*unprotected* relations, so they are identical on every neighboring instance and are
released exactly at no budget cost. Because a customer row also reaches the second bag
through `c_nationkey = s_nationkey`, that bag's release takes the full epsilon and the first
bag's takes the remainder; delta splits in half.

**GQ3 and GQ4** protect one edge tuple. There the maximum degree *is* a statistic of the
protected relation, so it must itself be released under the one-sided mechanism before it can
calibrate the capacity release. That is the rule: noise the frequency bound only when it
depends on protected data.

**Lane circuits keep the degree fixed.** A released capacity larger than the current domain
would otherwise force the next power of two, quantizing proving time into 2x jumps. The
`_dp` circuits instead host the capacity in `c = ceil(capacity / lane_rows)` parallel
column-group lanes. Every lane is a full structural replica with identical gates and
lookups, and the lane count depends only on the *released* capacity, never on the true bag
size -- a cheaper padding-only overflow lane would leak the true size through the circuit
shape and is deliberately not implemented.

Cost behaviour differs by query. Q5 and GQ3 grow linearly in the lane count (about 8% and
51 advice columns per lane). **GQ4 does not**: resolving a private key against private-size
tables needs one lookup argument per candidate table, so its two laned bags force `c1 * c2`
probe replicas. At `c1=4, c2=3` the laned circuit is 580 advice columns against 141 unlaned
at twice the domain, which is a wash. `g_sql4_obj_dp.rs` is correct and reviewed, but it is
not evidence of linear cost for GQ4.

## Analysis Harnesses

Fast, proof-free checks that print the numbers behind the configuration:

```bash
cargo test --test graph_lane_plan -- --nocapture
```

| Test | Reports |
|---|---|
| `graph_lane_plan` | released pads, degrees and lane counts per query/dataset/epsilon |
| `q5_dp_pads` | the same for Q5 across the epsilon sweep |
| `lane_cost_probe` | how advice columns and lookup arguments grow with the lane count |
| `freq_noising_cost` | what the frequency-noising stage costs on the graph queries |
| `q5_appendix_mechanism` | what the two-stage mechanism would cost Q5 instead |

Correctness tests, none of which run a full proof:

```bash
cargo test --release --lib q5_obj_dp
cargo test --release --lib g_sql3_obj_dp
cargo test --release --lib g_sql4_obj_dp
cargo test --release --lib pone_baseline
cargo test --release --lib dp_noise
```

The structural guard `cargo test --lib -- inline_bind` asserts every `_bound` circuit is a
strict superset of its baseline, so drift between a pair fails the suite instead of silently
skewing measurements.

## Notes

- **`k` is fitted, not assumed.** Each circuit starts at its natural degree (16 for TPC-H,
  17 for GQ1/GQ2, 18 for GQ3/GQ4) and `k` rises until the circuit fits, because the graph
  circuits' intermediate size depends on the dataset. Manually setting `k` in source is
  supported only under `MockProver`.
- **GQ3 and GQ4 are much larger on Facebook and Wikipedia** than on LastFM (~2.7M and ~2.3M
  wedge rows vs ~233K), so those rows need `k = 22` and dominate any sweep. With the default
  `VPJOIN_MAX_K=21` they are recorded as failures and the sweep continues; raise it to
  actually measure them.
- **Every proof is verified**, and `vpjoin_bench` persists proof bytes to
  `src/proof/bench/` as auditable artifacts. `MockProver` is never used in the benchmark
  harnesses, only in the correctness tests above.
- A failure in one row is recorded in its `status` column rather than aborting the sweep.
