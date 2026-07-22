//! 3-way path COUNT(*) with ordering (a<b<c) in ONE table Edge (self-join).
//!
//! Query:
//!   SELECT COUNT(*) AS cnt
//!   FROM Edge r1
//!   JOIN Edge r2 ON r1.dst = r2.src
//!   JOIN Edge r3 ON r2.dst = r3.src
//!   WHERE r1.src < r2.src AND r2.src < r3.src;
//!
//! Variables:
//!   a = r1.src
//!   b = r2.src = r1.dst
//!   c = r3.src = r2.dst
//!
//! Ordering becomes local constraints:
//!   a < b  <=>  r1.src < r1.dst
//!   b < c  <=>  r2.src < r2.dst
//!
//! DP plan (acyclic, like 5-way path):
//!   T3[c] = outdegree(c)  (count edges in r3 grouped by src=c)
//!   T2[b] = Σ_{(b->c) in r2} [b<c] * T3[c]
//!   ans   = Σ_{(a->b) in r1} [a<b] * T2[b]
//!
//! Join is proven with membership+gap proof pattern using:
//!   - map table with dummy row (0,0)
//!   - lookup_any for (gap low/high) and (map key/val)
//!
//! NOTE (real dataset safety):
//! - We shift all node IDs by +1 (SHIFT_ID) inside the circuit so that key=0
//!   is reserved for the dummy map row (0,0). This prevents collisions when
//!   real datasets contain node id 0.
//! - Shifting preserves equality joins and < ordering.
//!
//! Depends on your existing chips:
//!   crate::chips::is_zero::{IsZeroChip, IsZeroConfig}
//!   crate::chips::less_than::{LtChip, LtConfig, LtInstruction}
//!   crate::chips::permutation_any::{PermAnyChip, PermAnyConfig}

use halo2_proofs::halo2curves::ff::PrimeField;
use halo2_proofs::plonk::Expression;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};

// IMPORTANT: use the real dataset Edge type
use crate::data::graph_data_processing::Edge;

use std::collections::HashMap;
use std::marker::PhantomData;

// Use 8 bytes for robustness on real datasets; adjust if you know your ID width.
const NUM_BYTES: usize = 8;

// Reserve key=0 for the dummy map row
const SHIFT_ID: u64 = 1;

// Sentinel/pad key goes last in ASC order
const PAD_KEY: u64 = u64::MAX;
const PAD_VAL: u64 = 0;

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

/// ---------- Aggregator config (same pattern as GraphJoin5) ----------
#[derive(Clone, Debug)]
struct AggConfig<F: Field + Ord> {
    // Input triple: (src, dst, val)
    val_in: Column<Advice>,

    // Sorted triple columns
    sorted: [Column<Advice>; 3],

    // perm: (src, dst, val_in) <-> sorted
    perm_sort: PermAnyConfig,

    // sortedness check on sorted[0] (src)
    q_sort: Selector,
    lt_src_cur_next: LtConfig<F, NUM_BYTES>,
    iz_src_eq: IsZeroConfig<F>,

    // group helpers on sorted src
    q_first: Selector,
    q_accu: Selector,
    q_emit: Selector,
    run_sum: Column<Advice>,
    iz_same_prev: IsZeroConfig<F>,
    iz_same_next: IsZeroConfig<F>,

    // emitted padded group pairs (key,val) over rows of sorted
    emit_pair: [Column<Advice>; 2],

    // table (key,val) padded, permuted from emit_pair
    tbl_pair: [Column<Advice>; 2],
    tbl_key_next: Column<Advice>,
    perm_tbl: PermAnyConfig,

    // prove tbl_pair keys are sorted and tbl_key_next is the "next" key
    q_tbl_sort: Selector,
    lt_tbl_key_cur_next: LtConfig<F, NUM_BYTES>,
    iz_tbl_key_eq: IsZeroConfig<F>,

    q_tbl_shift: Selector, // enforce tbl_key_next[i] == tbl_key[i+1]
    q_tbl_last: Selector,  // enforce tbl_key_next[last] == PAD_KEY

    // map table used by joins (has dummy row 0)
    map_pair: [Column<Advice>; 2], // (key,val) with row0=(0,0), rows 1..=n copy tbl_pair
    map_key_next: Column<Advice>,  // next(key) for map_pair[0]

    q_map_tbl: Selector,   // enable table rows 0..=n (for lookup gating)
    q_map_first: Selector, // enforce map_pair[0] == (0,0)
    q_map_link: Selector,  // enforce map_pair[i+1] == tbl_pair[i]
    q_map_shift: Selector, // enforce map_key_next[i] == map_key[i+1]
    q_map_last: Selector,  // enforce map_key_next[last] == PAD_KEY
}

/// ---------- Join config (same pattern as GraphJoin5) ----------
#[derive(Clone, Debug)]
struct JoinConfig<F: Field + Ord> {
    // per-row membership flag: dst exists in next table?
    in_next: Column<Advice>,
    // gap witnesses if in_next == 0
    low: Column<Advice>,
    high: Column<Advice>,
    // attached value from next table (if in_next==1 else 0)
    val: Column<Advice>,

    // selectors
    q_lookup: Selector,
    q_lookup_complex: Selector,

    // LT checks for low < dst and dst < high when not in_next
    lt_low: LtConfig<F, NUM_BYTES>,
    lt_high: LtConfig<F, NUM_BYTES>,
}

/// ---------- Main circuit config ----------
#[derive(Clone, Debug)]
pub struct Path3OrdConfig<F: Field + Ord> {
    instance: Column<Instance>,

    // base edges r1,r2,r3: [src, dst] (all equal to Edge table)
    r: [[Column<Advice>; 2]; 3],

    // join steps: r1->T2, r2->T3
    join: [JoinConfig<F>; 2],

    // aggregators: agg[0]=r3->T3, agg[1]=r2->T2
    agg: [AggConfig<F>; 2],

    // ordering checks
    q_ord1: Selector,              // enable lt_ab on r1 rows
    q_ord2: Selector,              // enable lt_bc on r2 rows
    lt_ab: LtConfig<F, NUM_BYTES>, // r1.src < r1.dst
    lt_bc: LtConfig<F, NUM_BYTES>, // r2.src < r2.dst

    // glue gates (soundness) + filtered values
    q_r3_one: Selector,    // enforce agg0.val_in = 1
    q_r2_filter: Selector, // enforce agg1.val_in = join1.val * lt_bc
    q_r1_filter: Selector, // enforce fval_r1 = join0.val * lt_ab

    fval_r1: Column<Advice>, // filtered contribution per r1 row

    // final sum
    q_sum_first: Selector,
    q_sum_accu: Selector,
    sum: Column<Advice>,

    // output constrained to equal sum at chosen row
    out: Column<Advice>,
    q_out: Selector,
}

#[derive(Clone)]
pub struct Path3OrdCircuit<F: Field + Ord> {
    pub edges: Vec<Edge>,
    pub _marker: PhantomData<F>,
}

impl<F: Field + Ord> Default for Path3OrdCircuit<F> {
    fn default() -> Self {
        Self {
            edges: vec![],
            _marker: PhantomData,
        }
    }
}

pub struct Path3OrdChip<F: Field + Ord> {
    cfg: Path3OrdConfig<F>,
}

impl<F: Field + Ord> Path3OrdChip<F> {
    pub fn construct(cfg: Path3OrdConfig<F>) -> Self {
        Self { cfg }
    }

    // ---------------- helpers (host-side) ----------------

    fn sort_by_src(mut rows: Vec<[u64; 3]>) -> Vec<[u64; 3]> {
        // stable tie-breaking doesn't matter for correctness; (src, dst, val) just needs to be a permutation
        rows.sort_by_key(|r| r[0]);
        rows
    }

    fn run_sum_by_src(sorted: &[[u64; 3]]) -> Vec<u64> {
        let mut out = vec![0u64; sorted.len()];
        let mut acc: u128 = 0;
        let mut prev: Option<u64> = None;
        for (i, r) in sorted.iter().enumerate() {
            let src = r[0];
            let v = r[2] as u128;
            if prev == Some(src) {
                acc += v;
            } else {
                acc = v;
            }
            out[i] = acc as u64;
            prev = Some(src);
        }
        out
    }

    fn emit_pairs(sorted: &[[u64; 3]], run: &[u64]) -> Vec<[u64; 2]> {
        let n = sorted.len();
        let mut out = vec![[PAD_KEY, PAD_VAL]; n];
        for i in 0..n {
            let cur = sorted[i][0];
            let next = if i + 1 < n { sorted[i + 1][0] } else { PAD_KEY };
            let is_last = next != cur;
            if is_last {
                out[i] = [cur, run[i]];
            }
        }
        out
    }

    fn build_tbl_from_emit(emit: &[[u64; 2]], n: usize) -> Vec<[u64; 2]> {
        let mut pairs: Vec<[u64; 2]> = emit.iter().copied().filter(|p| p[0] != PAD_KEY).collect();
        pairs.sort_by_key(|p| p[0]);
        while pairs.len() < n {
            pairs.push([PAD_KEY, PAD_VAL]);
        }
        pairs.truncate(n);
        pairs
    }

    fn key_next_from_tbl(tbl: &[[u64; 2]]) -> Vec<u64> {
        let n = tbl.len();
        let mut out = vec![PAD_KEY; n];
        for i in 0..n {
            out[i] = if i + 1 < n { tbl[i + 1][0] } else { PAD_KEY };
        }
        out
    }

    fn map_from_tbl(tbl: &[[u64; 2]]) -> HashMap<u64, u64> {
        let mut m = HashMap::new();
        for [k, v] in tbl.iter().copied() {
            if k == PAD_KEY {
                continue;
            }
            m.insert(k, v);
        }
        m
    }

    fn gap_witness(keys_sorted_with_pad: &[u64], x: u64) -> (u64, u64, u64) {
        // returns (in, low, high)
        match keys_sorted_with_pad.binary_search(&x) {
            Ok(_) => (1, 0, PAD_KEY),
            Err(idx) => {
                let high = keys_sorted_with_pad[idx];
                let low = if idx == 0 {
                    0
                } else {
                    keys_sorted_with_pad[idx - 1]
                };
                (0, low, high)
            }
        }
    }

    // ---------------- configure gadgets ----------------

    fn configure_agg(
        meta: &mut ConstraintSystem<F>,
        src_col: Column<Advice>,
        dst_col: Column<Advice>,
    ) -> AggConfig<F> {
        // input val column
        let val_in = meta.advice_column();
        meta.enable_equality(val_in);

        // sorted triple
        let sorted = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        for c in sorted {
            meta.enable_equality(c);
        }

        // permutation: (src, dst, val_in) <-> sorted
        let q_perm_in = meta.complex_selector();
        let q_perm_out = meta.complex_selector();
        let perm_sort = PermAnyChip::configure(
            meta,
            q_perm_in,
            q_perm_out,
            vec![src_col, dst_col, val_in],
            sorted.to_vec(),
        );

        // sortedness on sorted[0] <= sorted[0]_next (uses sentinel row at n)
        let q_sort = meta.selector();
        let aux_src_eq = meta.advice_column();
        let iz_src_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort),
            |m| {
                m.query_advice(sorted[0], Rotation::next())
                    - m.query_advice(sorted[0], Rotation::cur())
            },
            aux_src_eq,
        );
        let lt_src_cur_next = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort),
            |m| m.query_advice(sorted[0], Rotation::cur()),
            |m| m.query_advice(sorted[0], Rotation::next()),
        );
        meta.create_gate("sorted src is nondecreasing", |m| {
            let q = m.query_selector(q_sort);
            let le = lt_src_cur_next.is_lt(m, None) + iz_src_eq.expr();
            vec![q * (le - Expression::Constant(F::ONE))]
        });

        // group-by on sorted src, sum sorted[2]
        let q_first = meta.selector();
        let q_accu = meta.selector();
        let q_emit = meta.selector();

        let run_sum = meta.advice_column();
        meta.enable_equality(run_sum);

        let aux_same_prev = meta.advice_column();
        let iz_same_prev = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_accu),
            |m| {
                m.query_advice(sorted[0], Rotation::cur())
                    - m.query_advice(sorted[0], Rotation::prev())
            },
            aux_same_prev,
        );

        let aux_same_next = meta.advice_column();
        let iz_same_next = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_emit),
            |m| {
                m.query_advice(sorted[0], Rotation::next())
                    - m.query_advice(sorted[0], Rotation::cur())
            },
            aux_same_next,
        );

        meta.create_gate("run_sum_first", |m| {
            let q = m.query_selector(q_first);
            let rs = m.query_advice(run_sum, Rotation::cur());
            let v = m.query_advice(sorted[2], Rotation::cur());
            vec![q * (rs - v)]
        });

        meta.create_gate("run_sum_accu", |m| {
            let q = m.query_selector(q_accu);
            let same = iz_same_prev.expr(); // 1 if same group
            let rs_cur = m.query_advice(run_sum, Rotation::cur());
            let rs_prev = m.query_advice(run_sum, Rotation::prev());
            let v = m.query_advice(sorted[2], Rotation::cur());
            vec![q * (rs_cur - (same * rs_prev + v))]
        });

        // emit padded (key,val)
        let emit_pair = [meta.advice_column(), meta.advice_column()];
        for c in emit_pair {
            meta.enable_equality(c);
        }

        meta.create_gate("emit group pair at last row of group", |m| {
            let q = m.query_selector(q_emit);
            let one = Expression::Constant(F::ONE);
            let same_next = iz_same_next.expr();
            let is_last = one.clone() - same_next;
            let not_last = one - is_last.clone();

            let cur_key = m.query_advice(sorted[0], Rotation::cur());
            let cur_sum = m.query_advice(run_sum, Rotation::cur());

            let out_k = m.query_advice(emit_pair[0], Rotation::cur());
            let out_v = m.query_advice(emit_pair[1], Rotation::cur());

            let pad_k = Expression::Constant(F::from(PAD_KEY));
            let pad_v = Expression::Constant(F::from(PAD_VAL));

            vec![
                q.clone() * (out_k - (is_last.clone() * cur_key + not_last.clone() * pad_k)),
                q * (out_v - (is_last * cur_sum + not_last * pad_v)),
            ]
        });

        // Build a padded table (key,val) that must be a permutation of emit_pair
        let tbl_pair = [meta.advice_column(), meta.advice_column()];
        let tbl_key_next = meta.advice_column();
        for c in tbl_pair {
            meta.enable_equality(c);
        }
        meta.enable_equality(tbl_key_next);

        let q_tbl_in = meta.complex_selector();
        let q_tbl_out = meta.complex_selector();
        let perm_tbl = PermAnyChip::configure(
            meta,
            q_tbl_in,
            q_tbl_out,
            emit_pair.to_vec(),
            tbl_pair.to_vec(),
        );

        // prove tbl_pair[0] sorted (only 0..n-2 are enabled in assignment)
        let q_tbl_sort = meta.selector();
        let aux_tbl_eq = meta.advice_column();
        let iz_tbl_key_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_tbl_sort),
            |m| {
                m.query_advice(tbl_pair[0], Rotation::next())
                    - m.query_advice(tbl_pair[0], Rotation::cur())
            },
            aux_tbl_eq,
        );
        let lt_tbl_key_cur_next = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_tbl_sort),
            |m| m.query_advice(tbl_pair[0], Rotation::cur()),
            |m| m.query_advice(tbl_pair[0], Rotation::next()),
        );
        meta.create_gate("tbl keys nondecreasing", |m| {
            let q = m.query_selector(q_tbl_sort);
            let le = lt_tbl_key_cur_next.is_lt(m, None) + iz_tbl_key_eq.expr();
            vec![q * (le - Expression::Constant(F::ONE))]
        });

        // enforce tbl_key_next[i] == tbl_key[i+1], last == PAD
        let q_tbl_shift = meta.selector();
        meta.create_gate("tbl_key_next = next(tbl_key)", |m| {
            let q = m.query_selector(q_tbl_shift);
            let kn = m.query_advice(tbl_key_next, Rotation::cur());
            let nextk = m.query_advice(tbl_pair[0], Rotation::next());
            vec![q * (kn - nextk)]
        });

        let q_tbl_last = meta.selector();
        meta.create_gate("tbl_key_next last = PAD", |m| {
            let q = m.query_selector(q_tbl_last);
            let kn = m.query_advice(tbl_key_next, Rotation::cur());
            vec![q * (kn - Expression::Constant(F::from(PAD_KEY)))]
        });

        // ---- map table (dummy row 0) ----
        let map_pair = [meta.advice_column(), meta.advice_column()];
        let map_key_next = meta.advice_column();
        for c in map_pair {
            meta.enable_equality(c);
        }
        meta.enable_equality(map_key_next);

        let q_map_tbl = meta.complex_selector();
        let q_map_first = meta.selector();
        let q_map_link = meta.selector();
        let q_map_shift = meta.selector();
        let q_map_last = meta.selector();

        meta.create_gate("map_first_is_zero", |m| {
            let q = m.query_selector(q_map_first);
            let k0 = m.query_advice(map_pair[0], Rotation::cur());
            let v0 = m.query_advice(map_pair[1], Rotation::cur());
            vec![q.clone() * k0, q * v0]
        });

        // link: map[i+1] == tbl[i]
        meta.create_gate("map_link_tbl_shift", |m| {
            let q = m.query_selector(q_map_link);
            let mk_next = m.query_advice(map_pair[0], Rotation::next());
            let mv_next = m.query_advice(map_pair[1], Rotation::next());
            let tk_cur = m.query_advice(tbl_pair[0], Rotation::cur());
            let tv_cur = m.query_advice(tbl_pair[1], Rotation::cur());
            vec![q.clone() * (mk_next - tk_cur), q * (mv_next - tv_cur)]
        });

        // map_key_next = next(map_key)
        meta.create_gate("map_key_next = next(map_key)", |m| {
            let q = m.query_selector(q_map_shift);
            let kn = m.query_advice(map_key_next, Rotation::cur());
            let nextk = m.query_advice(map_pair[0], Rotation::next());
            vec![q * (kn - nextk)]
        });

        // last: map_key_next[last] == PAD_KEY
        meta.create_gate("map_key_next last = PAD", |m| {
            let q = m.query_selector(q_map_last);
            let kn = m.query_advice(map_key_next, Rotation::cur());
            vec![q * (kn - Expression::Constant(F::from(PAD_KEY)))]
        });

        AggConfig {
            val_in,
            sorted,
            perm_sort,
            q_sort,
            lt_src_cur_next,
            iz_src_eq,

            q_first,
            q_accu,
            q_emit,
            run_sum,
            iz_same_prev,
            iz_same_next,

            emit_pair,
            tbl_pair,
            tbl_key_next,
            perm_tbl,

            q_tbl_sort,
            lt_tbl_key_cur_next,
            iz_tbl_key_eq,
            q_tbl_shift,
            q_tbl_last,

            map_pair,
            map_key_next,
            q_map_tbl,
            q_map_first,
            q_map_link,
            q_map_shift,
            q_map_last,
        }
    }

    fn configure_join(meta: &mut ConstraintSystem<F>, dst_col: Column<Advice>) -> JoinConfig<F> {
        let in_next = meta.advice_column();
        let low = meta.advice_column();
        let high = meta.advice_column();
        let val = meta.advice_column();
        for c in [in_next, low, high, val] {
            meta.enable_equality(c);
        }

        let q_lookup = meta.selector();
        let q_lookup_complex = meta.complex_selector();

        // lt_low: low < dst, lt_high: dst < high (enabled only when not in_next)
        let lt_low = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| {
                let q = m.query_selector(q_lookup);
                let inx = m.query_advice(in_next, Rotation::cur());
                q * (Expression::Constant(F::ONE) - inx)
            },
            |m| m.query_advice(low, Rotation::cur()),
            |m| m.query_advice(dst_col, Rotation::cur()),
        );
        let lt_high = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| {
                let q = m.query_selector(q_lookup);
                let inx = m.query_advice(in_next, Rotation::cur());
                q * (Expression::Constant(F::ONE) - inx)
            },
            |m| m.query_advice(dst_col, Rotation::cur()),
            |m| m.query_advice(high, Rotation::cur()),
        );

        meta.create_gate("join membership/gap basic logic", |m| {
            let q = m.query_selector(q_lookup);
            let inx = m.query_advice(in_next, Rotation::cur());
            let one = Expression::Constant(F::ONE);

            let not_in = one.clone() - inx.clone();
            let v = m.query_advice(val, Rotation::cur());

            let low_ok = lt_low.is_lt(m, None);
            let high_ok = lt_high.is_lt(m, None);

            vec![
                // in_next boolean
                q.clone() * inx.clone() * (one.clone() - inx.clone()),
                // if not in -> must satisfy low<dst<high
                q.clone() * not_in.clone() * (one.clone() - low_ok),
                q.clone() * not_in.clone() * (one.clone() - high_ok),
                // if not in -> val must be 0
                q * not_in * v,
            ]
        });

        JoinConfig {
            in_next,
            low,
            high,
            val,
            q_lookup,
            q_lookup_complex,
            lt_low,
            lt_high,
        }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> Path3OrdConfig<F> {
        let instance = meta.instance_column();
        meta.enable_equality(instance);

        let r: [[Column<Advice>; 2]; 3] =
            std::array::from_fn(|_| [meta.advice_column(), meta.advice_column()]);
        for i in 0..3 {
            meta.enable_equality(r[i][0]);
            meta.enable_equality(r[i][1]);
        }

        // join[0]: r1.dst joins to T2; join[1]: r2.dst joins to T3
        let join = [
            Self::configure_join(meta, r[0][1]),
            Self::configure_join(meta, r[1][1]),
        ];

        // agg[0]=r3->T3 ; agg[1]=r2->T2
        let agg = [
            Self::configure_agg(meta, r[2][0], r[2][1]),
            Self::configure_agg(meta, r[1][0], r[1][1]),
        ];

        // Ordering checks
        let q_ord1 = meta.selector();
        let q_ord2 = meta.selector();

        let lt_ab = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_ord1),
            |m| m.query_advice(r[0][0], Rotation::cur()),
            |m| m.query_advice(r[0][1], Rotation::cur()),
        );
        let lt_bc = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_ord2),
            |m| m.query_advice(r[1][0], Rotation::cur()),
            |m| m.query_advice(r[1][1], Rotation::cur()),
        );

        // filtered contribution per r1 row
        let fval_r1 = meta.advice_column();
        meta.enable_equality(fval_r1);

        // soundness glue
        let q_r3_one = meta.selector();
        let q_r2_filter = meta.selector();
        let q_r1_filter = meta.selector();

        // r3: val_in == 1
        meta.create_gate("r3 val_in is 1", |m| {
            let q = m.query_selector(q_r3_one);
            let v = m.query_advice(agg[0].val_in, Rotation::cur());
            vec![q * (v - Expression::Constant(F::ONE))]
        });

        // r2: agg[1].val_in == join[1].val * lt_bc
        meta.create_gate("r2 val_in filtered by b<c", |m| {
            let q = m.query_selector(q_r2_filter);
            let raw = m.query_advice(join[1].val, Rotation::cur());
            let bc = lt_bc.is_lt(m, None);
            let vin = m.query_advice(agg[1].val_in, Rotation::cur());
            vec![q * (vin - raw * bc)]
        });

        // r1: fval_r1 == join[0].val * lt_ab
        meta.create_gate("r1 filtered contribution", |m| {
            let q = m.query_selector(q_r1_filter);
            let raw = m.query_advice(join[0].val, Rotation::cur());
            let ab = lt_ab.is_lt(m, None);
            let fv = m.query_advice(fval_r1, Rotation::cur());
            vec![q * (fv - raw * ab)]
        });

        // sum fval_r1
        let q_sum_first = meta.selector();
        let q_sum_accu = meta.selector();
        let sum = meta.advice_column();
        meta.enable_equality(sum);

        meta.create_gate("sum_first", |m| {
            let q = m.query_selector(q_sum_first);
            let s = m.query_advice(sum, Rotation::cur());
            let v = m.query_advice(fval_r1, Rotation::cur());
            vec![q * (s - v)]
        });

        meta.create_gate("sum_accu", |m| {
            let q = m.query_selector(q_sum_accu);
            let s_cur = m.query_advice(sum, Rotation::cur());
            let s_prev = m.query_advice(sum, Rotation::prev());
            let v = m.query_advice(fval_r1, Rotation::cur());
            vec![q * (s_cur - (s_prev + v))]
        });

        // output constrained equal to sum (enable q_out at the chosen row)
        let out = meta.advice_column();
        meta.enable_equality(out);
        let q_out = meta.selector();
        meta.create_gate("out equals sum", |m| {
            let q = m.query_selector(q_out);
            let o = m.query_advice(out, Rotation::cur());
            let s = m.query_advice(sum, Rotation::cur());
            vec![q * (o - s)]
        });

        Path3OrdConfig {
            instance,
            r,
            join,
            agg,
            q_ord1,
            q_ord2,
            lt_ab,
            lt_bc,
            q_r3_one,
            q_r2_filter,
            q_r1_filter,
            fval_r1,
            q_sum_first,
            q_sum_accu,
            sum,
            out,
            q_out,
        }
    }

    /// expose public output cell
    pub fn expose_public(
        &self,
        layouter: &mut impl Layouter<F>,
        cell: AssignedCell<F, F>,
        row: usize,
    ) -> Result<(), Error> {
        layouter.constrain_instance(cell.cell(), self.cfg.instance, row)
    }

    // ---------------- assignment helpers ----------------

    fn assign_agg_stage(
        region: &mut Region<'_, F>,
        a: &AggConfig<F>,
        n: usize,
        sorted_rows: &Vec<[u64; 3]>,
        run: &Vec<u64>,
        emit: &Vec<[u64; 2]>,
        tbl: &Vec<[u64; 2]>,
        key_next: &Vec<u64>,
        val_in_vec: &Vec<u64>,
    ) -> Result<(), Error> {
        // val_in
        for i in 0..n {
            region.assign_advice(
                || "val_in",
                a.val_in,
                i,
                || Value::known(F::from(val_in_vec[i])),
            )?;
        }

        // sorted triple + sentinel row at n
        for i in 0..n {
            region.assign_advice(
                || "sorted_src",
                a.sorted[0],
                i,
                || Value::known(F::from(sorted_rows[i][0])),
            )?;
            region.assign_advice(
                || "sorted_dst",
                a.sorted[1],
                i,
                || Value::known(F::from(sorted_rows[i][1])),
            )?;
            region.assign_advice(
                || "sorted_val",
                a.sorted[2],
                i,
                || Value::known(F::from(sorted_rows[i][2])),
            )?;
        }
        region.assign_advice(
            || "sorted_src_s",
            a.sorted[0],
            n,
            || Value::known(F::from(PAD_KEY)),
        )?;
        region.assign_advice(
            || "sorted_dst_s",
            a.sorted[1],
            n,
            || Value::known(F::from(PAD_KEY)),
        )?;
        region.assign_advice(|| "sorted_val_s", a.sorted[2], n, || Value::known(F::ZERO))?;

        // run_sum
        for i in 0..n {
            region.assign_advice(|| "run_sum", a.run_sum, i, || Value::known(F::from(run[i])))?;
        }

        // emit pairs
        for i in 0..n {
            region.assign_advice(
                || "emit_k",
                a.emit_pair[0],
                i,
                || Value::known(F::from(emit[i][0])),
            )?;
            region.assign_advice(
                || "emit_v",
                a.emit_pair[1],
                i,
                || Value::known(F::from(emit[i][1])),
            )?;
        }

        // tbl pairs + key_next
        for i in 0..n {
            region.assign_advice(
                || "tbl_k",
                a.tbl_pair[0],
                i,
                || Value::known(F::from(tbl[i][0])),
            )?;
            region.assign_advice(
                || "tbl_v",
                a.tbl_pair[1],
                i,
                || Value::known(F::from(tbl[i][1])),
            )?;
            region.assign_advice(
                || "tbl_kn",
                a.tbl_key_next,
                i,
                || Value::known(F::from(key_next[i])),
            )?;
        }

        // ---- map table: rows 0..=n ----
        // row0 = (0,0)
        region.assign_advice(|| "map_k0", a.map_pair[0], 0, || Value::known(F::ZERO))?;
        region.assign_advice(|| "map_v0", a.map_pair[1], 0, || Value::known(F::ZERO))?;

        // rows 1..=n copy tbl[0..n-1]
        for i in 0..n {
            region.assign_advice(
                || "map_k",
                a.map_pair[0],
                i + 1,
                || Value::known(F::from(tbl[i][0])),
            )?;
            region.assign_advice(
                || "map_v",
                a.map_pair[1],
                i + 1,
                || Value::known(F::from(tbl[i][1])),
            )?;
        }

        // map_key_next[r] = map_key[r+1]
        // => for r=0..n-1: map_key_next[r] = tbl[r].key ; for r=n: PAD
        for r in 0..n {
            region.assign_advice(
                || "map_kn",
                a.map_key_next,
                r,
                || Value::known(F::from(tbl[r][0])),
            )?;
        }
        region.assign_advice(
            || "map_kn_last",
            a.map_key_next,
            n,
            || Value::known(F::from(PAD_KEY)),
        )?;

        // enable map selectors (table rows 0..=n)
        for i in 0..=n {
            a.q_map_tbl.enable(region, i)?;
        }
        a.q_map_first.enable(region, 0)?;
        for i in 0..n {
            a.q_map_link.enable(region, i)?;
            a.q_map_shift.enable(region, i)?;
        }
        a.q_map_last.enable(region, n)?;

        // enable perms
        for i in 0..n {
            a.perm_sort.q_perm1.enable(region, i)?;
            a.perm_sort.q_perm2.enable(region, i)?;
            a.perm_tbl.q_perm1.enable(region, i)?;
            a.perm_tbl.q_perm2.enable(region, i)?;
        }

        // sortedness on sorted src uses sentinel row -> enable 0..n-1
        for i in 0..n {
            a.q_sort.enable(region, i)?;
        }

        // tbl sortedness is only meaningful for 0..n-2
        for i in 0..n.saturating_sub(1) {
            if i + 1 < n {
                a.q_tbl_sort.enable(region, i)?;
                a.q_tbl_shift.enable(region, i)?;
            }
        }
        if n > 0 {
            a.q_tbl_last.enable(region, n - 1)?;
        }

        // group gates
        if n > 0 {
            a.q_first.enable(region, 0)?;
        }
        for i in 1..n {
            a.q_accu.enable(region, i)?;
        }
        for i in 0..n {
            a.q_emit.enable(region, i)?;
        }

        // assign helper chips
        let iz_same_prev_chip = IsZeroChip::construct(a.iz_same_prev.clone());
        let iz_same_next_chip = IsZeroChip::construct(a.iz_same_next.clone());
        let iz_src_eq_chip = IsZeroChip::construct(a.iz_src_eq.clone());
        let iz_tbl_eq_chip = IsZeroChip::construct(a.iz_tbl_key_eq.clone());

        let lt_src_chip = LtChip::<F, NUM_BYTES>::construct(a.lt_src_cur_next.clone());
        let lt_tbl_chip = LtChip::<F, NUM_BYTES>::construct(a.lt_tbl_key_cur_next.clone());

        // For sorted src: compare row i with row i+1; last compares to PAD sentinel
        for i in 0..n {
            let cur = sorted_rows[i][0];
            let next = if i + 1 < n {
                sorted_rows[i + 1][0]
            } else {
                PAD_KEY
            };

            lt_src_chip.assign(
                region,
                i,
                Value::known(F::from(cur)),
                Value::known(F::from(next)),
            )?;
            let diff_src = F::from(next) - F::from(cur);
            iz_src_eq_chip.assign(region, i, Value::known(diff_src))?;
        }

        // For tbl sortedness: only 0..n-2
        for i in 0..n.saturating_sub(1) {
            if i + 1 < n {
                lt_tbl_chip.assign(
                    region,
                    i,
                    Value::known(F::from(tbl[i][0])),
                    Value::known(F::from(tbl[i + 1][0])),
                )?;
                let diff_tbl = F::from(tbl[i + 1][0]) - F::from(tbl[i][0]);
                iz_tbl_eq_chip.assign(region, i, Value::known(diff_tbl))?;
            }
        }

        // same_prev (1..n-1)
        for i in 1..n {
            let diff = F::from(sorted_rows[i][0]) - F::from(sorted_rows[i - 1][0]);
            iz_same_prev_chip.assign(region, i, Value::known(diff))?;
        }
        // same_next (0..n-1), uses sentinel row at n
        for i in 0..n {
            let next_src = if i + 1 < n {
                sorted_rows[i + 1][0]
            } else {
                PAD_KEY
            };
            let diff = F::from(next_src) - F::from(sorted_rows[i][0]);
            iz_same_next_chip.assign(region, i, Value::known(diff))?;
        }

        Ok(())
    }

    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        edges_in: &[Edge],
    ) -> Result<AssignedCell<F, F>, Error> {
        let cfg = &self.cfg;
        let n = edges_in.len();

        // load LT tables
        // agg LTs
        for a in cfg.agg.iter() {
            LtChip::<F, NUM_BYTES>::construct(a.lt_src_cur_next.clone()).load(layouter)?;
            LtChip::<F, NUM_BYTES>::construct(a.lt_tbl_key_cur_next.clone()).load(layouter)?;
        }
        // join LTs
        for j in cfg.join.iter() {
            LtChip::<F, NUM_BYTES>::construct(j.lt_low.clone()).load(layouter)?;
            LtChip::<F, NUM_BYTES>::construct(j.lt_high.clone()).load(layouter)?;
        }
        // ordering LTs
        LtChip::<F, NUM_BYTES>::construct(cfg.lt_ab.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.lt_bc.clone()).load(layouter)?;

        if n == 0 {
            // trivial output 0
            let cell = layouter.assign_region(
                || "out0",
                |mut region| {
                    let c = region.assign_advice(|| "out", cfg.out, 0, || Value::known(F::ZERO))?;
                    Ok(c)
                },
            )?;
            return Ok(cell);
        }

        // Shift IDs inside the circuit to reserve key=0 for dummy map row.
        // (Works whether Edge fields are u32/u64/etc.)
        let edges: Vec<(u64, u64)> = edges_in
            .iter()
            .map(|e| {
                let s = e.src as u64;
                let d = e.dst as u64;
                (s + SHIFT_ID, d + SHIFT_ID)
            })
            .collect();

        // ---------------- host DP witnesses ----------------

        // Stage T3 from r3 (same edges): val=1 per edge, group by src
        let r3_rows: Vec<[u64; 3]> = edges.iter().map(|(s, d)| [*s, *d, 1u64]).collect();
        let r3_sorted = Self::sort_by_src(r3_rows.clone());
        let r3_run = Self::run_sum_by_src(&r3_sorted);
        let r3_emit = Self::emit_pairs(&r3_sorted, &r3_run);
        let t3_tbl = Self::build_tbl_from_emit(&r3_emit, n);
        let t3_next = Self::key_next_from_tbl(&t3_tbl);
        let t3_map = Self::map_from_tbl(&t3_tbl);

        let mut t3_keys: Vec<u64> = t3_tbl
            .iter()
            .map(|p| p[0])
            .filter(|&k| k != PAD_KEY)
            .collect();
        t3_keys.push(0);
        t3_keys.push(PAD_KEY);
        t3_keys.sort();
        t3_keys.dedup();

        // Join r2.dst into T3, then filter by (r2.src<r2.dst) to form val_in for T2
        let mut r2_in = vec![0u64; n];
        let mut r2_low = vec![0u64; n];
        let mut r2_high = vec![PAD_KEY; n];
        let mut r2_join_val = vec![0u64; n];
        let mut r2_val_in = vec![0u64; n];

        for (i, (s, d)) in edges.iter().enumerate() {
            let dst_c = *d;
            let (b, lo, hi) = Self::gap_witness(&t3_keys, dst_c);
            r2_in[i] = b;
            r2_low[i] = lo;
            r2_high[i] = hi;

            let raw = if b == 1 {
                *t3_map.get(&dst_c).unwrap_or(&0)
            } else {
                0
            };
            r2_join_val[i] = raw;

            let bc = if s < d { 1u64 } else { 0u64 };
            r2_val_in[i] = raw.saturating_mul(bc);
        }

        // Build T2 by aggregating rows (src, dst, val_in) grouped by src
        let r2_rows: Vec<[u64; 3]> = edges
            .iter()
            .enumerate()
            .map(|(i, (s, d))| [*s, *d, r2_val_in[i]])
            .collect();
        let r2_sorted = Self::sort_by_src(r2_rows.clone());
        let r2_run = Self::run_sum_by_src(&r2_sorted);
        let r2_emit = Self::emit_pairs(&r2_sorted, &r2_run);
        let t2_tbl = Self::build_tbl_from_emit(&r2_emit, n);
        let t2_next = Self::key_next_from_tbl(&t2_tbl);
        let t2_map = Self::map_from_tbl(&t2_tbl);

        let mut t2_keys: Vec<u64> = t2_tbl
            .iter()
            .map(|p| p[0])
            .filter(|&k| k != PAD_KEY)
            .collect();
        t2_keys.push(0);
        t2_keys.push(PAD_KEY);
        t2_keys.sort();
        t2_keys.dedup();

        // Join r1.dst into T2
        let mut r1_in = vec![0u64; n];
        let mut r1_low = vec![0u64; n];
        let mut r1_high = vec![PAD_KEY; n];
        let mut r1_join_val = vec![0u64; n];
        let mut r1_fval = vec![0u64; n];

        for (i, (s, d)) in edges.iter().enumerate() {
            let b = *d;
            let (inn, lo, hi) = Self::gap_witness(&t2_keys, b);
            r1_in[i] = inn;
            r1_low[i] = lo;
            r1_high[i] = hi;

            let raw = if inn == 1 {
                *t2_map.get(&b).unwrap_or(&0)
            } else {
                0
            };
            r1_join_val[i] = raw;

            let ab = if s < d { 1u64 } else { 0u64 };
            r1_fval[i] = raw.saturating_mul(ab);
        }

        // ---------------- circuit assignment ----------------

        let out_cell = layouter.assign_region(
            || "path3_ordered",
            |mut region| {
                // base tables r1,r2,r3 all equal to edges (self-join)
                for (i, (s, d)) in edges.iter().enumerate() {
                    // r1
                    region.assign_advice(
                        || "r1_src",
                        cfg.r[0][0],
                        i,
                        || Value::known(F::from(*s)),
                    )?;
                    region.assign_advice(
                        || "r1_dst",
                        cfg.r[0][1],
                        i,
                        || Value::known(F::from(*d)),
                    )?;
                    // r2
                    region.assign_advice(
                        || "r2_src",
                        cfg.r[1][0],
                        i,
                        || Value::known(F::from(*s)),
                    )?;
                    region.assign_advice(
                        || "r2_dst",
                        cfg.r[1][1],
                        i,
                        || Value::known(F::from(*d)),
                    )?;
                    // r3
                    region.assign_advice(
                        || "r3_src",
                        cfg.r[2][0],
                        i,
                        || Value::known(F::from(*s)),
                    )?;
                    region.assign_advice(
                        || "r3_dst",
                        cfg.r[2][1],
                        i,
                        || Value::known(F::from(*d)),
                    )?;
                }

                // join[1] for r2 -> T3
                {
                    let jc = &cfg.join[1];
                    let lt_low_chip = LtChip::<F, NUM_BYTES>::construct(jc.lt_low.clone());
                    let lt_high_chip = LtChip::<F, NUM_BYTES>::construct(jc.lt_high.clone());

                    for i in 0..n {
                        jc.q_lookup.enable(&mut region, i)?;
                        jc.q_lookup_complex.enable(&mut region, i)?;

                        region.assign_advice(
                            || "r2_in",
                            jc.in_next,
                            i,
                            || Value::known(F::from(r2_in[i])),
                        )?;
                        region.assign_advice(
                            || "r2_low",
                            jc.low,
                            i,
                            || Value::known(F::from(r2_low[i])),
                        )?;
                        region.assign_advice(
                            || "r2_high",
                            jc.high,
                            i,
                            || Value::known(F::from(r2_high[i])),
                        )?;
                        region.assign_advice(
                            || "r2_val",
                            jc.val,
                            i,
                            || Value::known(F::from(r2_join_val[i])),
                        )?;

                        let dst = edges[i].1;
                        lt_low_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(r2_low[i])),
                            Value::known(F::from(dst)),
                        )?;
                        lt_high_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(dst)),
                            Value::known(F::from(r2_high[i])),
                        )?;
                    }
                }

                // join[0] for r1 -> T2
                {
                    let jc = &cfg.join[0];
                    let lt_low_chip = LtChip::<F, NUM_BYTES>::construct(jc.lt_low.clone());
                    let lt_high_chip = LtChip::<F, NUM_BYTES>::construct(jc.lt_high.clone());

                    for i in 0..n {
                        jc.q_lookup.enable(&mut region, i)?;
                        jc.q_lookup_complex.enable(&mut region, i)?;

                        region.assign_advice(
                            || "r1_in",
                            jc.in_next,
                            i,
                            || Value::known(F::from(r1_in[i])),
                        )?;
                        region.assign_advice(
                            || "r1_low",
                            jc.low,
                            i,
                            || Value::known(F::from(r1_low[i])),
                        )?;
                        region.assign_advice(
                            || "r1_high",
                            jc.high,
                            i,
                            || Value::known(F::from(r1_high[i])),
                        )?;
                        region.assign_advice(
                            || "r1_val",
                            jc.val,
                            i,
                            || Value::known(F::from(r1_join_val[i])),
                        )?;

                        let dst = edges[i].1;
                        lt_low_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(r1_low[i])),
                            Value::known(F::from(dst)),
                        )?;
                        lt_high_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(dst)),
                            Value::known(F::from(r1_high[i])),
                        )?;
                    }
                }

                // ordering selectors + LT witnesses
                {
                    let lt_ab_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_ab.clone());
                    let lt_bc_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_bc.clone());

                    for i in 0..n {
                        cfg.q_ord1.enable(&mut region, i)?;
                        cfg.q_ord2.enable(&mut region, i)?;

                        let (s, d) = edges[i];

                        // r1.src < r1.dst
                        lt_ab_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(s)),
                            Value::known(F::from(d)),
                        )?;
                        // r2.src < r2.dst
                        lt_bc_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(s)),
                            Value::known(F::from(d)),
                        )?;
                    }
                }

                // glue selectors + filtered values
                for i in 0..n {
                    cfg.q_r3_one.enable(&mut region, i)?;
                    cfg.q_r2_filter.enable(&mut region, i)?;
                    cfg.q_r1_filter.enable(&mut region, i)?;

                    region.assign_advice(
                        || "fval_r1",
                        cfg.fval_r1,
                        i,
                        || Value::known(F::from(r1_fval[i])),
                    )?;
                }

                // agg stage 0: r3 -> T3 (val_in all ones)
                let r3_val_in = vec![1u64; n];
                Self::assign_agg_stage(
                    &mut region,
                    &cfg.agg[0],
                    n,
                    &r3_sorted,
                    &r3_run,
                    &r3_emit,
                    &t3_tbl,
                    &t3_next,
                    &r3_val_in,
                )?;

                // agg stage 1: r2 -> T2 (val_in = join_val * (b<c))
                Self::assign_agg_stage(
                    &mut region,
                    &cfg.agg[1],
                    n,
                    &r2_sorted,
                    &r2_run,
                    &r2_emit,
                    &t2_tbl,
                    &t2_next,
                    &r2_val_in,
                )?;

                // final sum over fval_r1
                let mut running: u128 = 0;
                for i in 0..n {
                    running += r1_fval[i] as u128;
                    region.assign_advice(
                        || "sum",
                        cfg.sum,
                        i,
                        || Value::known(F::from(running as u64)),
                    )?;
                }
                cfg.q_sum_first.enable(&mut region, 0)?;
                for i in 1..n {
                    cfg.q_sum_accu.enable(&mut region, i)?;
                }

                // output at last row, constrained equal to sum via q_out
                let out_row = n - 1;
                cfg.q_out.enable(&mut region, out_row)?;
                let out_cell = region.assign_advice(
                    || "out",
                    cfg.out,
                    out_row,
                    || Value::known(F::from((running as u64))),
                )?;
                Ok(out_cell)
            },
        )?;

        Ok(out_cell)
    }
}

// ---------------- IMPORTANT: join lookups are defined in Circuit::configure ----------------

/// Full constraint-system setup for `Path3OrdCircuit`.
/// Extracted verbatim so the circuit and any wrapper that embeds it
/// (see `crate::inline_bind`) configure IDENTICAL constraints -- the
/// chip's own `configure` alone is NOT sufficient here.
pub fn configure_path3ord_full<F: Field + Ord>(meta: &mut ConstraintSystem<F>) -> Path3OrdConfig<F> {
    let cfg = Path3OrdChip::<F>::configure(meta);

    // convenience
    let t3 = cfg.agg[0].clone(); // table from r3
    let t2 = cfg.agg[1].clone(); // table from r2

    // r1 joins to T2
    {
        let j = cfg.join[0].clone();
        let rel_dst = cfg.r[0][1];
        let t = t2.clone();

        // gap lookup: (1-in)*low in key AND (1-in)*high in key_next
        meta.lookup_any("gap r1->t2", move |m| {
            let q_in = m.query_selector(j.q_lookup_complex);
            let inx = m.query_advice(j.in_next, Rotation::cur());
            let gate = q_in * (Expression::Constant(F::ONE) - inx);

            let low = m.query_advice(j.low, Rotation::cur());
            let high = m.query_advice(j.high, Rotation::cur());

            let q_tbl = m.query_selector(t.q_map_tbl);
            let key = m.query_advice(t.map_pair[0], Rotation::cur());
            let keyn = m.query_advice(t.map_key_next, Rotation::cur());

            vec![
                (gate.clone() * low, q_tbl.clone() * key),
                (gate * high, q_tbl * keyn),
            ]
        });

        // map lookup: (in*dst, val) exists in (map_key, map_val)
        // when in=0, join-gate forces val=0 so tuple is (0,0) which exists at map row0
        meta.lookup_any("map r1->t2", move |m| {
            let q_in = m.query_selector(j.q_lookup_complex);
            let inx = m.query_advice(j.in_next, Rotation::cur());

            let dst = m.query_advice(rel_dst, Rotation::cur());
            let v = m.query_advice(j.val, Rotation::cur());

            let q_tbl = m.query_selector(t.q_map_tbl);
            let tk = m.query_advice(t.map_pair[0], Rotation::cur());
            let tv = m.query_advice(t.map_pair[1], Rotation::cur());

            vec![
                (q_in.clone() * inx.clone() * dst, q_tbl.clone() * tk),
                (q_in * v, q_tbl * tv),
            ]
        });
    }

    // r2 joins to T3
    {
        let j = cfg.join[1].clone();
        let rel_dst = cfg.r[1][1];
        let t = t3.clone();

        meta.lookup_any("gap r2->t3", move |m| {
            let q_in = m.query_selector(j.q_lookup_complex);
            let inx = m.query_advice(j.in_next, Rotation::cur());
            let gate = q_in * (Expression::Constant(F::ONE) - inx);

            let low = m.query_advice(j.low, Rotation::cur());
            let high = m.query_advice(j.high, Rotation::cur());

            let q_tbl = m.query_selector(t.q_map_tbl);
            let key = m.query_advice(t.map_pair[0], Rotation::cur());
            let keyn = m.query_advice(t.map_key_next, Rotation::cur());

            vec![
                (gate.clone() * low, q_tbl.clone() * key),
                (gate * high, q_tbl * keyn),
            ]
        });

        meta.lookup_any("map r2->t3", move |m| {
            let q_in = m.query_selector(j.q_lookup_complex);
            let inx = m.query_advice(j.in_next, Rotation::cur());

            let dst = m.query_advice(rel_dst, Rotation::cur());
            let v = m.query_advice(j.val, Rotation::cur());

            let q_tbl = m.query_selector(t.q_map_tbl);
            let tk = m.query_advice(t.map_pair[0], Rotation::cur());
            let tv = m.query_advice(t.map_pair[1], Rotation::cur());

            vec![
                (q_in.clone() * inx.clone() * dst, q_tbl.clone() * tk),
                (q_in * v, q_tbl * tv),
            ]
        });
    }

    cfg
}

impl<F: Field + Ord> Circuit<F> for Path3OrdCircuit<F> {
    type Config = Path3OrdConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        configure_path3ord_full(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = Path3OrdChip::construct(config);
        let out_cell = chip.assign(&mut layouter, &self.edges)?;
        chip.expose_public(&mut layouter, out_cell, 0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::graph_data_processing::read_edges;
    use crate::data::graph_data_processing::read_edges_csv;
    use halo2_proofs::dev::MockProver;

    use halo2_proofs::{
        plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit},
        poly::{
            commitment::Params,
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
    use halo2curves::pasta::{vesta, EqAffine, Fp};
    use rand::rngs::OsRng;
    use std::time::Instant;
    use std::{fs::File, io::Write, path::Path};

    fn generate_and_verify_proof<C: Circuit<Fp>>(
        circuit: C,
        public_input: &[Fp],
        proof_path: &str,
    ) {
        let params_path = &crate::paths::param_file(17);
        let mut fd = std::fs::File::open(&params_path).unwrap();
        let params = ParamsIPA::<vesta::Affine>::read(&mut fd).unwrap();

        let t0 = Instant::now();
        let vk = keygen_vk(&params, &circuit).expect("keygen_vk should not fail");
        println!("Time to generate vk {:?}", t0.elapsed());

        let t1 = Instant::now();
        let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk should not fail");
        println!("Time to generate pk {:?}", t1.elapsed());

        let mut rng = OsRng;
        let mut transcript = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);
        create_proof::<IPACommitmentScheme<_>, ProverIPA<_>, _, _, _, _>(
            &params,
            &pk,
            &[circuit],
            &[&[public_input]],
            &mut rng,
            &mut transcript,
        )
        .expect("proof generation should not fail");
        let proof = transcript.finalize();

        File::create(Path::new(proof_path))
            .expect("Failed to create proof file")
            .write_all(&proof)
            .expect("Failed to write proof");
        println!("Proof written to: {}", proof_path);

        let strategy = SingleStrategy::new(&params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        assert!(
            verify_proof(
                &params,
                pk.get_vk(),
                strategy,
                &[&[public_input]],
                &mut transcript
            )
            .is_ok(),
            "Proof verification failed"
        );
    }

    fn dp_expected(edges: &[Edge]) -> u64 {
        // multiplicity-aware DP:
        // T3[c] = outdegree(c)
        // T2[b] = Σ_{(b->c)} [b<c] * T3[c]
        // ans   = Σ_{(a->b)} [a<b] * T2[b]
        let mut outdeg: HashMap<u64, u64> = HashMap::new();
        for e in edges {
            let s = e.src as u64;
            *outdeg.entry(s).or_insert(0) += 1;
        }

        let mut t2: HashMap<u64, u128> = HashMap::new();
        for e in edges {
            let b = e.src as u64;
            let c = e.dst as u64;
            if b < c {
                let v = *outdeg.get(&c).unwrap_or(&0) as u128;
                *t2.entry(b).or_insert(0) += v;
            }
        }

        let mut ans: u128 = 0;
        for e in edges {
            let a = e.src as u64;
            let b = e.dst as u64;
            if a < b {
                ans += *t2.get(&b).unwrap_or(&0);
            }
        }
        ans as u64
    }

    #[test]
    fn test() {
        let base_path = &crate::paths::graph_dir();

        let mut edges = read_edges(&format!("{}/wiki/wiki_Vote.txt", base_path)).unwrap();
        // let mut edges =
        //     read_edges(&format!("{}/facebook/facebook_combined.txt", base_path)).unwrap();
        // let mut edges =
        //     read_edges_csv(&format!("{}/last/lastfm_asia_edges.csv", base_path)).unwrap();

        // edges.truncate(100);

        let cnt = dp_expected(&edges);

        let circuit = Path3OrdCircuit::<Fp> {
            edges,
            _marker: PhantomData,
        };

        let public_input = vec![Fp::from(cnt)];
        let k = 17;

        // let test = true;
        let test = false;

        if test {
            let prover = MockProver::run(k, &circuit, vec![public_input]).unwrap();
            prover.assert_satisfied();
        } else {
            let proof_path = &crate::paths::proof_file("wiki_proof_q1");
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
    }
}
