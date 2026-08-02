# VPJoin

**VPJoin** is a zero-knowledge proof framework for verifiable multi-way SQL join processing.
Built on the [Halo2](https://github.com/zcash/halo2) proving system with PLONKish
arithmetization, it lets a data holder prove the correct execution of complex SQL queries
over private data without revealing the underlying records.


## Project Structure

```
src/
  chips/          # Custom Halo2 gate implementations (hash, lookup, comparison)
  circuits/       # Reusable circuit gadgets (inclusion checks, permutations, Merkle trees)
  sql/            # TPC-H query circuits (Q3, Q5, Q8, Q9, Q18)
  graph_sql/      # Graph pattern query circuits (GQ1--GQ4)
  circuits/card_preserve.rs  # Cardinality Preservation Check
  circuits/conserve_idx.rs   # indexed Conservation Check (One-Pass OBJ)
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


The three-condition One-Pass OBJ is the only realization each query ships.
`src/sql/*_obj.rs` and `src/graph_sql/*_obj.rs`, and the DP-padded
`*_obj_dp.rs` variants, all certify (7) Conservation, (9) Pairwise Consistency
and (10) Cardinality Preservation as selector bits on the committed rows, and
every command below proves that. The earlier four-condition circuits that
materialized a partition are gone, so no environment variable selects between
realizations and the `config` column no longer carries a `+new` tag.

**1. VPJoin proving time**, 5 TPC-H queries plus 4 graph queries on 3 datasets without
the database-commitment layer:

```bash
cargo vpjoin simplification
```

The same harness proves the **full** system, query circuit plus the complete
database-commitment layer:

```bash
cargo vpjoin full
```




**2. Additional in-circuit cost of binding a proof to a committed database.**

```bash
cargo commit-diff reps=5 q3 q5 q8 q9 q18 gq1 gq2 gq3 gq4
```



`commit_diff` builds its own paired circuits (`inline_bind::tpch_paired` /
`graph_paired`) rather than going through the shared query dispatch. The bound
wrappers in `*_bound.rs` delegate to the `*_obj.rs` chips, so each pair is the
One-Pass circuit with and without the binding gates, over the same circuit the
commands above prove.

**3. DP-guided padding** for the three cyclic queries.

```bash
VPJOIN_EPS=0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10 VPJOIN_DP_SEED=1 cargo run --bin dp_lane_bench -- reps=3 q5 gq3 gq4
```


**4. Scaling every table, not only `lineitem`.** `src/new_data/` holds two dataset families:
`all_scaled` grows every table by the same 2x and 4x factors, while `lineitem_scaled` grows
only `lineitem` and holds the dimension tables at the base size, so the pair isolates what
the dimension tables cost:

```bash
VPJOIN_TABLES=$PWD/src/new_data/all_scaled/60K/data       VPJOIN_LABEL=all-60K       VPJOIN_PRIVACY=rjs cargo vpjoin simplification q3 q5 q8 q9 q18
```

```bash
VPJOIN_TABLES=$PWD/src/new_data/all_scaled/120K/data      VPJOIN_LABEL=all-120K      VPJOIN_PRIVACY=rjs cargo vpjoin simplification q3 q5 q8 q9 q18
```

```bash
VPJOIN_TABLES=$PWD/src/new_data/all_scaled/240K/data      VPJOIN_LABEL=all-240K      VPJOIN_PRIVACY=rjs cargo vpjoin simplification q3 q5 q8 q9 q18
```

```bash
VPJOIN_TABLES=$PWD/src/new_data/lineitem_scaled/60K/data  VPJOIN_LABEL=lineitem-60K  VPJOIN_PRIVACY=rjs cargo vpjoin simplification q3 q5 q8 q9 q18
```

```bash
VPJOIN_TABLES=$PWD/src/new_data/lineitem_scaled/120K/data VPJOIN_LABEL=lineitem-120K VPJOIN_PRIVACY=rjs cargo vpjoin simplification q3 q5 q8 q9 q18
```

```bash
VPJOIN_TABLES=$PWD/src/new_data/lineitem_scaled/240K/data VPJOIN_LABEL=lineitem-240K VPJOIN_PRIVACY=rjs cargo vpjoin simplification q3 q5 q8 q9 q18
```


**5. Estimation anchors.** The PoneglyphDB times and VPJoin's fully oblivious
cyclic times cannot be run to completion, so both are extrapolated from a measured
anchor: each is that anchor's per-domain-row rate, `T_0 / 2^k`, charged for every
row of the padded domain. These two commands produce the four graph anchors.


```bash
VPJOIN_PRIVACY=rjs VPJOIN_MAX_EDGES=65520 VPJOIN_K=16 cargo vpjoin simplification gq1:wiki gq2:wiki
```

```bash
VPJOIN_PRIVACY=rjs VPJOIN_MAX_EDGES=21403 cargo vpjoin simplification gq3:lastfm gq4:lastfm
```
