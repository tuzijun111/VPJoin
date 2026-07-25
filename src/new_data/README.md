# `new_data/` — scaled TPC-H datasets for VPJoin scaling experiments

Two families of TPC-H datasets, each at three scales keyed by the `lineitem` row
count (60K / 120K / 240K). They exist to answer two different scaling questions:

| family | what grows | what stays fixed | question it answers |
|--------|-----------|------------------|---------------------|
| `all_scaled/`      | **every** table (2x / 4x) | nothing (nation/region are spec-fixed) | how does cost scale when the whole database grows? |
| `lineitem_scaled/` | only `lineitem` (2x / 4x) | all dimension tables (held at the 60K base) | how does cost scale with just the fact table? |

At 60K the two families are identical (nothing to scale yet), so they share the base
point.

## Layout

```
new_data/
  all_scaled/{60K,120K,240K}/data/       <- experiment (2): full-schema scaling
  lineitem_scaled/{60K,120K,240K}/data/  <- experiment (1): fact-table-only scaling
  verify_integrity.sh
  README.md
```

Each `<family>/<scale>/data/` holds the eight tables under their **canonical** names
(`lineitem.tbl`, `customer.tbl`, `part.tbl`, ... plus `region.cvs`). Same format as
`src/data/`: pipe-delimited, no trailing pipe, UTF-8. `region` is stored as `region.cvs`
(pipe-delimited despite the extension) to match `region_read_records_from_cvs` in
`src/data/data_processing.rs`.

## Row counts

`all_scaled/` — every table grows 1x / 2x / 4x (`nation` and `region` are TPC-H reference
tables the spec fixes at 25 and 5):

| dir | SF | lineitem | orders | customer | partsupp | part | supplier | nation | region |
|-----|----|---------:|-------:|---------:|---------:|-----:|---------:|-------:|-------:|
| `60K/`  | 0.01 |  60,175 | 15,000 | 1,500 |  8,000 | 2,000 | 100 | 25 | 5 |
| `120K/` | 0.02 | 120,515 | 30,000 | 3,000 | 16,000 | 4,000 | 200 | 25 | 5 |
| `240K/` | 0.04 | 240,292 | 60,000 | 6,000 | 32,000 | 8,000 | 400 | 25 | 5 |

`lineitem_scaled/` — only `lineitem` grows; every dimension table stays at the 60K base:

| dir | lineitem | orders | customer | partsupp | part | supplier | nation | region |
|-----|---------:|-------:|---------:|---------:|-----:|---------:|-------:|-------:|
| `60K/`  |  60,175 | 15,000 | 1,500 | 8,000 | 2,000 | 100 | 25 | 5 |
| `120K/` | 120,350 | 15,000 | 1,500 | 8,000 | 2,000 | 100 | 25 | 5 |
| `240K/` | 240,700 | 15,000 | 1,500 | 8,000 | 2,000 | 100 | 25 | 5 |

## Provenance

The base tables and the `all_scaled/` family were generated with the **official TPC-H
toolkit `dbgen`** (version 2.14.0, Transaction Processing Performance Council) at scale
factors 0.01 / 0.02 / 0.04, then the trailing `|` on every line was stripped to match the
repo's `.tbl` format. Because a given scale factor is deterministic, the generated
`lineitem` at each scale is **byte-for-byte identical** to the repo's existing
`src/data/lineitem.tbl`, `lineitem_120K.tbl`, and `lineitem_240K.tbl`, and the 60K
dimension tables are byte-identical to `src/data/`. So this is the same TPC-H data VPJoin
already used, now completed with matching dimension tables.

The larger `lineitem_scaled/` fact tables are **not** a larger `dbgen` scale factor —
`dbgen`'s SF-0.02/0.04 `lineitem` references dimension keys (`partkey` up to 4,000,
`suppkey` up to 200, `orderkey` up to 120,000) that the fixed 60K dimensions do not
contain, so ~half of those rows would dangle. Instead each `lineitem_scaled/` fact table
is the base `dbgen` `lineitem` with every row **replicated in place** 2x / 4x. Replicated
rows reuse base-range foreign keys, so referential integrity against the fixed 60K
dimensions is exact (see below). Aggregates therefore scale linearly and `l_linenumber`
repeats across copies; that is fine for a size-scaling benchmark, and Q5 (the intended
target) does not read `l_linenumber`.

## Validation — `verify_integrity.sh`

```bash
./verify_integrity.sh                              # every data dir, both families
./verify_integrity.sh lineitem_scaled/240K/data    # one dir
```

All 10 foreign keys resolve inside every one of the six `data/` directories, 0 dangling:
`lineitem -> {orders, part, supplier, partsupp}`, `orders -> customer`,
`{customer, supplier} -> nation`, `partsupp -> {part, supplier}`, `nation -> region`.
This is the integrity the older "swap in `lineitem_120K.tbl` over the base dimensions"
workflow broke: there ~60,000 of the 120,515 rows had a dangling part/supplier/order key,
and Q5's inner join (`q5_derive`) silently dropped them, collapsing the "120K" run back
toward the base size.

## Using a dataset

Point the harness at a group's tables with **`VPJOIN_TABLES`** (the directory holding the
`.tbl`/`.cvs` files). This swaps only the tables; the proof params (`src/proof/param{k}`)
and graph datasets stay under the crate, so nothing else has to move:

```bash
# experiment (2): whole schema 4x
VPJOIN_TABLES=$PWD/src/new_data/all_scaled/240K/data      \
  cargo run --release --bin vpjoin_bench -- full q5

# experiment (1): fact table 4x, dimensions fixed
VPJOIN_TABLES=$PWD/src/new_data/lineitem_scaled/240K/data \
  cargo run --release --bin vpjoin_bench -- full q5
```

(`VPJOIN_DATA` still exists and overrides the *whole* root — data + `proof/` + `graph_data/`
— but a scale dir has no `proof/`, so for these sweeps use `VPJOIN_TABLES`, which leaves the
params in place.)

### Q5 privacy mode

For a clean scaling curve use `VPJOIN_PRIVACY=rjs`: `q5_pads(Rjs)` is `(0,0,0)`, so there
is zero padding and no dependence on the DP mechanism or `VPJOIN_DP_SEED` — runs are
deterministic and the only variable across scales is the data size. (`dp` mode releases
per-bag capacities that vary with epsilon/delta and would confound a pure size sweep;
`rjs` reveals the true join size, i.e. no privacy, which is the right trade for a
performance sweep.)

### Circuit degree at scale

All five TPC-H queries now choose their circuit degree `k` from the live `lineitem` row
count in `degree_for` (`src/bench_queries.rs`), so they auto-grow with the fact table:

| lineitem | k | 2^k |
|----------|---|-----|
| 60K  (60,175)  | 16 |  65,536 |
| 120K (120,3-120,5K) | 17 | 131,072 |
| 240K (240,3-240,7K) | 18 | 262,144 |

Q5 sizes from `max(lineitem, orders+co_pad, ls_join+ls_pad)`; `q3/q8/q9/q18` materialize no
intermediate and are lineitem-dominated, so they size from `lineitem` directly. Both were
previously pinned to `k=16` (fits only <=65,536 rows and would trip the `run_at` assertion
at 120K/240K); both still return 16 at the 60K base, so existing results are unchanged. The
params for k=16/17/18 already ship in `src/proof/`.

## Running the six-group sweep

`run_sweep.sh` runs `q3 q5 q8 q9 q18` on all six groups, one `vpjoin_bench` process per
group, tagging each with `VPJOIN_LABEL` and writing `results/scaling/scaling_<label>.csv`
plus a stitched `scaling_ALL.csv`:

```bash
./run_sweep.sh                     # mode=full (prove + commitment layer)
./run_sweep.sh baseline            # prove/verify only
VPJOIN_PLAN_ONLY=1 ./run_sweep.sh  # dry run: print the planned degrees and exit
```

Or drive one group by hand (build once, then run the binary):

```bash
cargo build --release --bin vpjoin_bench
VPJOIN_TABLES=$PWD/src/new_data/all_scaled/120K/data VPJOIN_LABEL=all-120K \
  VPJOIN_PRIVACY=rjs ./target/release/vpjoin_bench full results/scaling/all-120K.csv q3 q5 q8 q9 q18
```

### The six groups, debug profile, no files written

`baseline` measures prove/verify only, with no commitment/binding layer. Any argument
ending in `.csv` is the output file, so omitting it writes no file at all: results go to
stdout, ending in an aligned summary table. Omitting it also disables resume, so every
query genuinely re-runs instead of being skipped as already done. Dropping `--release`
selects the debug (unoptimized) profile, the same one `cargo test` uses.

Run these from the repo root, so `$PWD` resolves:

```bash
cargo build --bin vpjoin_bench
```

```bash
VPJOIN_TABLES=$PWD/src/new_data/all_scaled/60K/data       VPJOIN_LABEL=all-60K       VPJOIN_PRIVACY=rjs ./target/debug/vpjoin_bench baseline q3 q5 q8 q9 q18
```

```bash
VPJOIN_TABLES=$PWD/src/new_data/all_scaled/120K/data      VPJOIN_LABEL=all-120K      VPJOIN_PRIVACY=rjs ./target/debug/vpjoin_bench baseline q3 q5 q8 q9 q18
```

```bash
VPJOIN_TABLES=$PWD/src/new_data/all_scaled/240K/data      VPJOIN_LABEL=all-240K      VPJOIN_PRIVACY=rjs ./target/debug/vpjoin_bench baseline q3 q5 q8 q9 q18
```

```bash
VPJOIN_TABLES=$PWD/src/new_data/lineitem_scaled/60K/data  VPJOIN_LABEL=lineitem-60K  VPJOIN_PRIVACY=rjs ./target/debug/vpjoin_bench baseline q3 q5 q8 q9 q18
```

```bash
VPJOIN_TABLES=$PWD/src/new_data/lineitem_scaled/120K/data VPJOIN_LABEL=lineitem-120K VPJOIN_PRIVACY=rjs ./target/debug/vpjoin_bench baseline q3 q5 q8 q9 q18
```

```bash
VPJOIN_TABLES=$PWD/src/new_data/lineitem_scaled/240K/data VPJOIN_LABEL=lineitem-240K VPJOIN_PRIVACY=rjs ./target/debug/vpjoin_bench baseline q3 q5 q8 q9 q18
```

Debug proving is several times slower than release, so do not mix debug and release
timings in one comparison.

One thing the missing `.csv` does not suppress: each proof is still written to
`src/proof/bench/<query>_<dataset>.proof` as an auditable artifact, about 29 KB per run at
`k=18`. `VPJOIN_LABEL` keeps those filenames distinct per group, so no group overwrites
another. Discard them with `rm -rf src/proof/bench` (the directory is gitignored).

## Notes

The dataset label reported in every row is `VPJOIN_LABEL` (defaults to `tpch-60K`), so the
six groups stay distinct even in a combined CSV. `VPJOIN_PLAN_ONLY=1` verified the
degrees: `all-60K`/`lineitem-60K` -> k=16, `*-120K` -> k=17, `*-240K` -> k=18, for all five
queries.
