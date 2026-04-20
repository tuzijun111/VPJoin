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

### DP-Guided Join Gate (DPJ) for Cyclic Joins

DPJ extends OBJ to cyclic queries via **tree decomposition** with **differentially private capacity bounds**. A one-sided DP noise mechanism sets circuit capacities that always exceed the true size (preserving correctness) while providing formal (epsilon, delta)-differential privacy for intermediate cardinalities. At epsilon = 0.1 and delta = 10^-5, this adds only **2--10% overhead** versus the non-private baseline.

## Project Structure

```
src/
  chips/          # Custom Halo2 gate implementations (hash, lookup, comparison, etc.)
  circuits/       # Reusable circuit gadgets (inclusion checks, permutations, Merkle trees)
  sql/            # TPC-H query circuits (Q3, Q5, Q8, Q9, Q18) using OBJ/DPJ gates
  graph_sql/      # Graph pattern query circuits (GQ1--GQ4) for network datasets
  data/           # TPC-H dataset files and data processing utilities
  graph_data/     # Network dataset files and graph data processing
  proof/          # Proof generation and verification infrastructure
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


