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

OBJ verifies multi-way joins via structural checks along a join tree, achieving
**worst-case O(IN + OUT) circuit complexity**. Since the circuit never creates intermediate
results, its layout depends only on input table sizes, so **obliviousness is achieved entirely
for free** with zero padding overhead.

In the **One-Pass** gate the prover computes the semijoin reduction offline and supplies the
resulting clean/residual split directly as witness; the circuit only certifies it. The split is
stated over the *indexed* relation, in which every row carries its committed position `l` and
the indicator `c(l)` marking the part it went to:

```
R^_i = { (l, t_l, c(l)) : l in [|R_i|] }
```

Three conditions certify it as the fully reduced instance:

- **Conservation** -- `R^_i == R^_i^c U+ R^_i^r`, one permutation argument per relation between
  the indexed relation and the concatenation of its two parts. Because the indices are distinct,
  `R^_i` is a *set* even when `R_i` is a bag, so this single permutation already places every
  occurrence on exactly one side: none fabricated, lost, duplicated, or counted in both. This is
  where the gate saves against a value-level split, which needs a separate non-membership
  argument to keep two equal tuples apart. Carrying `c` inside the conserved entry ties the
  partition to the indicator column the other two conditions read.
- **Pairwise Consistency** -- clean tuples in neighboring relations project to the same key set
  on every tree edge, as two mutual membership checks gated by the indicator on both sides.
- **Cardinality Preservation** -- the clean join and the predicate-filtered input join have the
  same size, so no valid join result hides among the residuals.

Cardinality preservation replaced an earlier residual-side condition, which asked only that a
semijoin reduction over the residual relations empty at the root. That detects a join witness
lying entirely on the residual side but not one mixing clean and residual tuples, since a tuple
wrongly moved to the residual whose join partners stay clean leaves no trace there at all. The
current condition counts instead: the clean join is always contained in the input join, so equal
cardinality forces the two to be the same multiset. Both cardinalities come from one traversal
of the join tree carrying two multiplicities per tuple, anchored at the predicate bit on the
input channel and at the indicator on the clean one, and a single equality constraint compares
the two root sums.

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
  circuits/card_preserve.rs  # Cardinality Preservation Check
  circuits/conserve_idx.rs   # indexed Conservation Check (revised One-Pass OBJ)
  data/           # TPC-H dataset files and parsing utilities
  graph_data/     # Network dataset files and parsing
  proof/          # Persisted public parameters (param15..param19) and some proof artifacts
  bench_queries.rs  # Shared query/dataset/privacy plumbing for every harness
  dp_noise.rs       # DP capacity release 
  dp_lane.rs        # Plan/run records shared by the DP lane circuits
  commitment.rs     # Database-commitment layer: canonical layout, Commit(D), per-proof binding
  input_binding.rs  # In-circuit binding of query inputs to the published commitments
  bin/              # Benchmark harnesses 
```



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


### Build

```bash
cargo build --release
```

## Running the Benchmarks


`VPJOIN_OBJ=new` selects the three-condition One-Pass OBJ above
(`src/sql/*_obj_new.rs`, `src/graph_sql/*_obj_new.rs`), which is what the
commands below measure; rows proved with it carry `+new` in the `config` column.
Dropping the variable falls back to the earlier four-condition realization
(`*_obj.rs`), which is kept runnable so the two can be compared at the same `k`
on the same inputs.

**1. VPJoin proving time**, 5 TPC-H queries plus 4 graph queries on 3 datasets without
the database-commitment layer:

```bash
VPJOIN_OBJ=new cargo vpjoin simplification
```

Cyclic queries by revealing the true join results size.
```bash
VPJOIN_OBJ=new VPJOIN_PRIVACY=rjs cargo vpjoin simplification q5 gq3 gq4
```

The same harness proves the **full** system, query circuit plus the complete
database-commitment layer:

```bash
VPJOIN_OBJ=new cargo vpjoin full
```

Prefix any of these with `VPJOIN_PLAN_ONLY=1` to print the planned degrees and
check the parameter files exist without proving anything.


**2. Additional in-circuit cost of binding a proof to a committed database.** The reported cost is the median paired difference:

```bash
cargo commit-diff reps=3 q3 q8 q9 q18 gq1 gq2
```

```bash
VPJOIN_PRIVACY=rjs cargo commit-diff reps=3 q5 gq3 gq4
```

These two are the exception: `commit_diff` builds its own paired circuits
(`inline_bind::tpch_paired` / `graph_paired`) rather than going through the
shared query dispatch, so it measures the binding cost against the `*_obj.rs`
circuits and ignores `VPJOIN_OBJ`. Moving it onto the revised gate needs a bound
variant per query, not just the switch.

**3. DP-guided padding** for the three cyclic queries. The lane geometry and every
released capacity are the same under either realization, so the switch isolates the
gate rather than the padding:

```bash
VPJOIN_OBJ=new VPJOIN_EPS=0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10 VPJOIN_DP_SEED=1 cargo run --bin dp_lane_bench -- reps=3 q5 gq3 gq4
```

GQ4 runs the same circuit either way: `g_sql4_obj_dp.rs` never materialized a
partition, so it already realizes the three conditions and has no `_new` sibling.


**4. Scaling every table, not only `lineitem`.** `src/new_data/` holds two dataset families:
`all_scaled` grows every table by the same 2x and 4x factors, while `lineitem_scaled` grows
only `lineitem` and holds the dimension tables at the base size, so the pair isolates what
the dimension tables cost:

```bash
VPJOIN_OBJ=new VPJOIN_TABLES=$PWD/src/new_data/all_scaled/60K/data       VPJOIN_LABEL=all-60K       VPJOIN_PRIVACY=rjs cargo vpjoin simplification q3 q5 q8 q9 q18
```

```bash
VPJOIN_OBJ=new VPJOIN_TABLES=$PWD/src/new_data/all_scaled/120K/data      VPJOIN_LABEL=all-120K      VPJOIN_PRIVACY=rjs cargo vpjoin simplification q3 q5 q8 q9 q18
```

```bash
VPJOIN_OBJ=new VPJOIN_TABLES=$PWD/src/new_data/all_scaled/240K/data      VPJOIN_LABEL=all-240K      VPJOIN_PRIVACY=rjs cargo vpjoin simplification q3 q5 q8 q9 q18
```

```bash
VPJOIN_OBJ=new VPJOIN_TABLES=$PWD/src/new_data/lineitem_scaled/60K/data  VPJOIN_LABEL=lineitem-60K  VPJOIN_PRIVACY=rjs cargo vpjoin simplification q3 q5 q8 q9 q18
```

```bash
VPJOIN_OBJ=new VPJOIN_TABLES=$PWD/src/new_data/lineitem_scaled/120K/data VPJOIN_LABEL=lineitem-120K VPJOIN_PRIVACY=rjs cargo vpjoin simplification q3 q5 q8 q9 q18
```

```bash
VPJOIN_OBJ=new VPJOIN_TABLES=$PWD/src/new_data/lineitem_scaled/240K/data VPJOIN_LABEL=lineitem-240K VPJOIN_PRIVACY=rjs cargo vpjoin simplification q3 q5 q8 q9 q18
```

