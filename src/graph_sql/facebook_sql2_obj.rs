//! 4-way path COUNT(*) with ordering: a<b<c<d
//!
//! Query:
//!   SELECT COUNT(*) AS cnt
//!   FROM Edge r1
//!   JOIN Edge r2 ON r1.dst = r2.src
//!   JOIN Edge r3 ON r2.dst = r3.src
//!   JOIN Edge r4 ON r3.dst = r4.src
//!   WHERE r1.src < r2.src
//!     AND r2.src < r3.src
//!     AND r3.src < r4.src;
//!
//! Variables:
//!   r1: (a -> b)
//!   r2: (b -> c)
//!   r3: (c -> d)
//!   r4: (d -> e)
//! Ordering means: a<b<c<d   (note: e unconstrained)
//!
//! DP plan:
//!   T4[d] = outdeg(d)                                      // from r4
//!   T3[c] = Σ_{(c->d) in r3} T4[d] * [c<d]                 // join r3.dst into T4
//!   T2[b] = Σ_{(b->c) in r2} T3[c] * [b<c]                 // join r2.dst into T3
//!   ans   = Σ_{(a->b) in r1} T2[b] * [a<b]                 // join r1.dst into T2
//!
//! IMPORTANT: selectors used inside meta.lookup_any(...) MUST be complex_selector().
//! - join.q_lookup_complex is complex_selector()
//! - agg.q_map_tbl (map-table gate) is complex_selector()

use halo2_proofs::halo2curves::ff::PrimeField;
use halo2_proofs::plonk::Expression;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};

use crate::data::graph_data_processing::Edge;

use std::collections::HashMap;
use std::marker::PhantomData;

const NUM_BYTES: usize = 8;
const MAX_SENTINEL: u64 = u64::MAX;
const PAD_KEY: u64 = MAX_SENTINEL; // pad key goes last in ASC
const PAD_VAL: u64 = 0;

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

#[derive(Clone, Debug)]
struct AggConfig<F: Field + Ord> {
    // Input triple: (src, dst, val_in)
    val_in: Column<Advice>,

    // Sorted triple columns (src, dst, val)
    sorted: [Column<Advice>; 3],

    // perm: (src, dst, val_in) <-> sorted
    perm_sort: PermAnyConfig,

    // prove sorted[0] is nondecreasing
    q_sort: Selector,
    lt_src_cur_next: LtConfig<F, NUM_BYTES>,
    iz_src_eq: IsZeroConfig<F>,

    // group helpers (by sorted src)
    q_first: Selector,
    q_accu: Selector,
    q_emit: Selector,
    run_sum: Column<Advice>,
    iz_same_prev: IsZeroConfig<F>,
    iz_same_next: IsZeroConfig<F>,

    // emitted padded group pairs (key,val)
    emit_pair: [Column<Advice>; 2],

    // padded table (key,val) permuted from emit_pair
    tbl_pair: [Column<Advice>; 2],
    perm_tbl: PermAnyConfig,

    // prove tbl keys nondecreasing
    q_tbl_sort: Selector,
    lt_tbl_key_cur_next: LtConfig<F, NUM_BYTES>,
    iz_tbl_key_eq: IsZeroConfig<F>,

    // === map table used by joins (has dummy row 0) ===
    map_pair: [Column<Advice>; 2], // (key,val) with row0=(0,0)
    map_key_next: Column<Advice>,  // next(key)

    // IMPORTANT: used in lookup_any => must be complex_selector()
    q_map_tbl: Selector,

    q_map_first: Selector,
    q_map_link: Selector,
    q_map_shift: Selector,
    q_map_last: Selector,
}

#[derive(Clone, Debug)]
struct JoinConfig<F: Field + Ord> {
    in_next: Column<Advice>,
    low: Column<Advice>,
    high: Column<Advice>,
    val: Column<Advice>,

    q_lookup: Selector,
    q_lookup_complex: Selector,

    lt_low: LtConfig<F, NUM_BYTES>,
    lt_high: LtConfig<F, NUM_BYTES>,
}

#[derive(Clone, Debug)]
pub struct GraphPath4OrderConfig<F: Field + Ord> {
    instance: Column<Instance>,

    // r1..r4 copies: [src, dst]
    r: [[Column<Advice>; 2]; 4],

    // joins:
    // join[2]: r3.dst -> T4
    // join[1]: r2.dst -> T3
    // join[0]: r1.dst -> T2
    join: [JoinConfig<F>; 3],

    // aggs:
    // agg[0]: r4 -> T4(outdeg by src)
    // agg[1]: r3 -> T3(sum by c) with val_in = join[2].val*[c<d]
    // agg[2]: r2 -> T2(sum by b) with val_in = join[1].val*[b<c]
    agg: [AggConfig<F>; 3],

    // filters:
    q_r3_filt: Selector,
    lt_cd: LtConfig<F, NUM_BYTES>,

    q_r2_filt: Selector,
    lt_bc: LtConfig<F, NUM_BYTES>,

    q_r1_contrib: Selector,
    lt_ab: LtConfig<F, NUM_BYTES>,

    contrib: Column<Advice>,

    // sum
    q_sum_first: Selector,
    q_sum_accu: Selector,
    sum: Column<Advice>,

    out: Column<Advice>,
}

#[derive(Clone)]
pub struct GraphPath4OrderCircuit<F: Field + Ord> {
    pub edges: Vec<Edge>, // reuse same Edge table as r1..r4 (self-joins)
    pub _marker: PhantomData<F>,
}

impl<F: Field + Ord> Default for GraphPath4OrderCircuit<F> {
    fn default() -> Self {
        Self {
            edges: vec![],
            _marker: PhantomData,
        }
    }
}

pub struct GraphPath4OrderChip<F: Field + Ord> {
    cfg: GraphPath4OrderConfig<F>,
}

impl<F: Field + Ord> GraphPath4OrderChip<F> {
    pub fn construct(cfg: GraphPath4OrderConfig<F>) -> Self {
        Self { cfg }
    }

    fn configure_agg(
        meta: &mut ConstraintSystem<F>,
        src_col: Column<Advice>,
        dst_col: Column<Advice>,
    ) -> AggConfig<F> {
        let val_in = meta.advice_column();
        meta.enable_equality(val_in);

        let sorted = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        for c in sorted {
            meta.enable_equality(c);
        }

        // PermAny selectors are typically selectors; not required by your rule.
        let q_perm_in = meta.complex_selector();
        let q_perm_out = meta.complex_selector();
        let perm_sort = PermAnyChip::configure(
            meta,
            q_perm_in,
            q_perm_out,
            vec![src_col, dst_col, val_in],
            sorted.to_vec(),
        );

        // sortedness of sorted[0]
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
        meta.create_gate("sorted src nondecreasing", |m| {
            let q = m.query_selector(q_sort);
            let le = lt_src_cur_next.is_lt(m, None) + iz_src_eq.expr();
            vec![q * (le - Expression::Constant(F::ONE))]
        });

        // group-by helpers
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
            let same = iz_same_prev.expr();
            let rs_cur = m.query_advice(run_sum, Rotation::cur());
            let rs_prev = m.query_advice(run_sum, Rotation::prev());
            let v = m.query_advice(sorted[2], Rotation::cur());
            vec![q * (rs_cur - (same * rs_prev + v))]
        });

        let emit_pair = [meta.advice_column(), meta.advice_column()];
        for c in emit_pair {
            meta.enable_equality(c);
        }
        meta.create_gate("emit group pair or pad", |m| {
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

        // padded table (key,val) as permutation of emit_pair
        let tbl_pair = [meta.advice_column(), meta.advice_column()];
        for c in tbl_pair {
            meta.enable_equality(c);
        }
        let q_tbl_in = meta.complex_selector();
        let q_tbl_out = meta.complex_selector();
        let perm_tbl = PermAnyChip::configure(
            meta,
            q_tbl_in,
            q_tbl_out,
            emit_pair.to_vec(),
            tbl_pair.to_vec(),
        );

        // prove tbl keys sorted
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

        // map table with dummy row0
        let map_pair = [meta.advice_column(), meta.advice_column()];
        let map_key_next = meta.advice_column();
        for c in map_pair {
            meta.enable_equality(c);
        }
        meta.enable_equality(map_key_next);

        // IMPORTANT: used inside lookup_any => complex_selector()
        let q_map_tbl = meta.complex_selector();
        let q_map_first = meta.selector();
        let q_map_link = meta.selector();
        let q_map_shift = meta.selector();
        let q_map_last = meta.selector();

        meta.create_gate("map first row is (0,0)", |m| {
            let q = m.query_selector(q_map_first);
            let k0 = m.query_advice(map_pair[0], Rotation::cur());
            let v0 = m.query_advice(map_pair[1], Rotation::cur());
            vec![q.clone() * k0, q * v0]
        });
        // map[i+1] = tbl[i]
        meta.create_gate("map links tbl (shifted)", |m| {
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
        meta.create_gate("map last next = PAD", |m| {
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
            perm_tbl,
            q_tbl_sort,
            lt_tbl_key_cur_next,
            iz_tbl_key_eq,
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
        let q_lookup_complex = meta.complex_selector(); // IMPORTANT for lookup_any

        // when not in_next: enforce low < dst and dst < high
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

        meta.create_gate("join membership/gap logic", |m| {
            let q = m.query_selector(q_lookup);
            let inx = m.query_advice(in_next, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let not_in = one.clone() - inx.clone();
            let v = m.query_advice(val, Rotation::cur());

            let low_ok = lt_low.is_lt(m, None);
            let high_ok = lt_high.is_lt(m, None);

            vec![
                q.clone() * inx.clone() * (one.clone() - inx.clone()), // boolean
                q.clone() * not_in.clone() * (one.clone() - low_ok),
                q.clone() * not_in.clone() * (one.clone() - high_ok),
                q * not_in * v, // if not_in => v=0
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

    pub fn configure(meta: &mut ConstraintSystem<F>) -> GraphPath4OrderConfig<F> {
        let instance = meta.instance_column();
        meta.enable_equality(instance);

        let out = meta.advice_column();
        meta.enable_equality(out);

        // r1..r4 distinct columns (self-join)
        let r: [[Column<Advice>; 2]; 4] =
            std::array::from_fn(|_| [meta.advice_column(), meta.advice_column()]);
        for i in 0..4 {
            meta.enable_equality(r[i][0]);
            meta.enable_equality(r[i][1]);
        }

        // joins (configured against each dst column)
        let join = [
            Self::configure_join(meta, r[0][1]), // r1.dst -> T2
            Self::configure_join(meta, r[1][1]), // r2.dst -> T3
            Self::configure_join(meta, r[2][1]), // r3.dst -> T4
        ];

        // aggs:
        // r4 -> T4
        // r3 -> T3
        // r2 -> T2
        let agg = [
            Self::configure_agg(meta, r[3][0], r[3][1]),
            Self::configure_agg(meta, r[2][0], r[2][1]),
            Self::configure_agg(meta, r[1][0], r[1][1]),
        ];

        // r3 filter: agg[1].val_in = join[2].val * [c<d]
        let q_r3_filt = meta.selector();
        let lt_cd = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| {
                let q = m.query_selector(q_r3_filt);
                let inx = m.query_advice(join[2].in_next, Rotation::cur());
                q * inx
            },
            |m| m.query_advice(r[2][0], Rotation::cur()), // c
            |m| m.query_advice(r[2][1], Rotation::cur()), // d
        );
        meta.create_gate("r3 val_in = join_val * [c<d]", |m| {
            let q = m.query_selector(q_r3_filt);
            let v = m.query_advice(join[2].val, Rotation::cur());
            let outv = m.query_advice(agg[1].val_in, Rotation::cur());
            let cd = lt_cd.is_lt(m, None);
            vec![q * (outv - v * cd)]
        });

        // r2 filter: agg[2].val_in = join[1].val * [b<c]
        let q_r2_filt = meta.selector();
        let lt_bc = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| {
                let q = m.query_selector(q_r2_filt);
                let inx = m.query_advice(join[1].in_next, Rotation::cur());
                q * inx
            },
            |m| m.query_advice(r[1][0], Rotation::cur()), // b
            |m| m.query_advice(r[1][1], Rotation::cur()), // c
        );
        meta.create_gate("r2 val_in = join_val * [b<c]", |m| {
            let q = m.query_selector(q_r2_filt);
            let v = m.query_advice(join[1].val, Rotation::cur());
            let outv = m.query_advice(agg[2].val_in, Rotation::cur());
            let bc = lt_bc.is_lt(m, None);
            vec![q * (outv - v * bc)]
        });

        // r1 contrib: contrib = join[0].val * [a<b]
        let contrib = meta.advice_column();
        meta.enable_equality(contrib);

        let q_r1_contrib = meta.selector();
        let lt_ab = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| {
                let q = m.query_selector(q_r1_contrib);
                let inx = m.query_advice(join[0].in_next, Rotation::cur());
                q * inx
            },
            |m| m.query_advice(r[0][0], Rotation::cur()), // a
            |m| m.query_advice(r[0][1], Rotation::cur()), // b
        );
        meta.create_gate("contrib = join_val * [a<b]", |m| {
            let q = m.query_selector(q_r1_contrib);
            let v = m.query_advice(join[0].val, Rotation::cur());
            let outc = m.query_advice(contrib, Rotation::cur());
            let ab = lt_ab.is_lt(m, None);
            vec![q * (outc - v * ab)]
        });

        // sum contrib
        let q_sum_first = meta.selector();
        let q_sum_accu = meta.selector();
        let sum = meta.advice_column();
        meta.enable_equality(sum);

        meta.create_gate("sum_first", |m| {
            let q = m.query_selector(q_sum_first);
            let s = m.query_advice(sum, Rotation::cur());
            let v = m.query_advice(contrib, Rotation::cur());
            vec![q * (s - v)]
        });
        meta.create_gate("sum_accu", |m| {
            let q = m.query_selector(q_sum_accu);
            let s_cur = m.query_advice(sum, Rotation::cur());
            let s_prev = m.query_advice(sum, Rotation::prev());
            let v = m.query_advice(contrib, Rotation::cur());
            vec![q * (s_cur - (s_prev + v))]
        });

        GraphPath4OrderConfig {
            instance,
            r,
            join,
            agg,
            q_r3_filt,
            lt_cd,
            q_r2_filt,
            lt_bc,
            q_r1_contrib,
            lt_ab,
            contrib,
            q_sum_first,
            q_sum_accu,
            sum,
            out,
        }
    }

    pub fn expose_public(
        &self,
        layouter: &mut impl Layouter<F>,
        cell: AssignedCell<F, F>,
        row: usize,
    ) -> Result<(), Error> {
        layouter.constrain_instance(cell.cell(), self.cfg.instance, row)
    }

    // ---------------- host helpers ----------------
    fn sort_by_src(mut rows: Vec<[u64; 3]>) -> Vec<[u64; 3]> {
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

    // ---------------- assign ----------------
    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        edges: &[Edge],
    ) -> Result<AssignedCell<F, F>, Error> {
        let cfg = &self.cfg;
        let n = edges.len();

        // load LT tables
        for a in cfg.agg.iter() {
            LtChip::<F, NUM_BYTES>::construct(a.lt_src_cur_next.clone()).load(layouter)?;
            LtChip::<F, NUM_BYTES>::construct(a.lt_tbl_key_cur_next.clone()).load(layouter)?;
        }
        for j in cfg.join.iter() {
            LtChip::<F, NUM_BYTES>::construct(j.lt_low.clone()).load(layouter)?;
            LtChip::<F, NUM_BYTES>::construct(j.lt_high.clone()).load(layouter)?;
        }
        LtChip::<F, NUM_BYTES>::construct(cfg.lt_ab.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.lt_bc.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.lt_cd.clone()).load(layouter)?;

        if n == 0 {
            let cell = layouter.assign_region(
                || "out0",
                |mut region| {
                    let c =
                        region.assign_advice(|| "out", cfg.out, 0, || Value::known(F::from(0)))?;
                    Ok(c)
                },
            )?;
            return Ok(cell);
        }

        // -------- host witnesses (DP) --------
        // Stage4: T4[d] = outdeg(d) from r4
        let r4_rows: Vec<[u64; 3]> = edges.iter().map(|e| [e.src, e.dst, 1u64]).collect();
        let r4_sorted = Self::sort_by_src(r4_rows.clone());
        let r4_run = Self::run_sum_by_src(&r4_sorted);
        let r4_emit = Self::emit_pairs(&r4_sorted, &r4_run);
        let t4_tbl = Self::build_tbl_from_emit(&r4_emit, n);
        let t4_map = Self::map_from_tbl(&t4_tbl);

        let mut t4_keys: Vec<u64> = t4_tbl
            .iter()
            .map(|p| p[0])
            .filter(|&k| k != PAD_KEY)
            .collect();
        t4_keys.push(0);
        t4_keys.push(PAD_KEY);
        t4_keys.sort();
        t4_keys.dedup();

        // Stage3: join r3.dst=d into T4[d], filter c<d, agg by c => T3
        let mut r3_in = vec![0u64; n];
        let mut r3_low = vec![0u64; n];
        let mut r3_high = vec![PAD_KEY; n];
        let mut r3_join_val = vec![0u64; n];
        let mut r3_val_filt = vec![0u64; n];
        let mut r3_rows: Vec<[u64; 3]> = Vec::with_capacity(n);

        for (i, e) in edges.iter().enumerate() {
            let c = e.src;
            let d = e.dst;

            let (inx, lo, hi) = Self::gap_witness(&t4_keys, d);
            r3_in[i] = inx;
            r3_low[i] = lo;
            r3_high[i] = hi;

            let v = if inx == 1 {
                *t4_map.get(&d).unwrap_or(&0)
            } else {
                0
            };
            r3_join_val[i] = v;

            let cd = if c < d { 1u64 } else { 0u64 };
            r3_val_filt[i] = v * cd;

            r3_rows.push([c, d, r3_val_filt[i]]);
        }

        let r3_sorted = Self::sort_by_src(r3_rows.clone());
        let r3_run = Self::run_sum_by_src(&r3_sorted);
        let r3_emit = Self::emit_pairs(&r3_sorted, &r3_run);
        let t3_tbl = Self::build_tbl_from_emit(&r3_emit, n);
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

        // Stage2: join r2.dst=c into T3[c], filter b<c, agg by b => T2
        let mut r2_in = vec![0u64; n];
        let mut r2_low = vec![0u64; n];
        let mut r2_high = vec![PAD_KEY; n];
        let mut r2_join_val = vec![0u64; n];
        let mut r2_val_filt = vec![0u64; n];
        let mut r2_rows: Vec<[u64; 3]> = Vec::with_capacity(n);

        for (i, e) in edges.iter().enumerate() {
            let b = e.src;
            let c = e.dst;

            let (inx, lo, hi) = Self::gap_witness(&t3_keys, c);
            r2_in[i] = inx;
            r2_low[i] = lo;
            r2_high[i] = hi;

            let v = if inx == 1 {
                *t3_map.get(&c).unwrap_or(&0)
            } else {
                0
            };
            r2_join_val[i] = v;

            let bc = if b < c { 1u64 } else { 0u64 };
            r2_val_filt[i] = v * bc;

            r2_rows.push([b, c, r2_val_filt[i]]);
        }

        let r2_sorted = Self::sort_by_src(r2_rows.clone());
        let r2_run = Self::run_sum_by_src(&r2_sorted);
        let r2_emit = Self::emit_pairs(&r2_sorted, &r2_run);
        let t2_tbl = Self::build_tbl_from_emit(&r2_emit, n);
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

        // Stage1: join r1.dst=b into T2[b], filter a<b, sum
        let mut r1_in = vec![0u64; n];
        let mut r1_low = vec![0u64; n];
        let mut r1_high = vec![PAD_KEY; n];
        let mut r1_join_val = vec![0u64; n];
        let mut contrib = vec![0u64; n];

        let mut answer_u128: u128 = 0;
        for (i, e) in edges.iter().enumerate() {
            let a = e.src;
            let b = e.dst;

            let (inx, lo, hi) = Self::gap_witness(&t2_keys, b);
            r1_in[i] = inx;
            r1_low[i] = lo;
            r1_high[i] = hi;

            let v = if inx == 1 {
                *t2_map.get(&b).unwrap_or(&0)
            } else {
                0
            };
            r1_join_val[i] = v;

            let ab = if a < b { 1u64 } else { 0u64 };
            contrib[i] = v * ab;
            answer_u128 += contrib[i] as u128;
        }
        let answer = answer_u128 as u64;

        // -------- assignment region --------
        let out_cell = layouter.assign_region(
            || "path4_order_witness",
            |mut region| {
                // assign base edges into r1..r4 copies
                for k in 0..4 {
                    for (i, e) in edges.iter().enumerate() {
                        region.assign_advice(
                            || "src",
                            cfg.r[k][0],
                            i,
                            || Value::known(F::from(e.src)),
                        )?;
                        region.assign_advice(
                            || "dst",
                            cfg.r[k][1],
                            i,
                            || Value::known(F::from(e.dst)),
                        )?;
                    }
                }

                // assign join helper
                let mut assign_join = |jc: &JoinConfig<F>,
                                       in_vec: &Vec<u64>,
                                       low_vec: &Vec<u64>,
                                       high_vec: &Vec<u64>,
                                       val_vec: &Vec<u64>,
                                       dst_vec: &Vec<u64>|
                 -> Result<(), Error> {
                    let lt_low_chip = LtChip::<F, NUM_BYTES>::construct(jc.lt_low.clone());
                    let lt_high_chip = LtChip::<F, NUM_BYTES>::construct(jc.lt_high.clone());
                    for i in 0..n {
                        jc.q_lookup.enable(&mut region, i)?;
                        jc.q_lookup_complex.enable(&mut region, i)?; // complex (used by lookups)

                        region.assign_advice(
                            || "in_next",
                            jc.in_next,
                            i,
                            || Value::known(F::from(in_vec[i])),
                        )?;
                        region.assign_advice(
                            || "low",
                            jc.low,
                            i,
                            || Value::known(F::from(low_vec[i])),
                        )?;
                        region.assign_advice(
                            || "high",
                            jc.high,
                            i,
                            || Value::known(F::from(high_vec[i])),
                        )?;
                        region.assign_advice(
                            || "val",
                            jc.val,
                            i,
                            || Value::known(F::from(val_vec[i])),
                        )?;

                        lt_low_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(low_vec[i])),
                            Value::known(F::from(dst_vec[i])),
                        )?;
                        lt_high_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(dst_vec[i])),
                            Value::known(F::from(high_vec[i])),
                        )?;
                    }
                    Ok(())
                };

                // join[2]: r3.dst -> T4   (dst_vec = edges[i].dst)
                assign_join(
                    &cfg.join[2],
                    &r3_in,
                    &r3_low,
                    &r3_high,
                    &r3_join_val,
                    &edges.iter().map(|e| e.dst).collect(),
                )?;

                // join[1]: r2.dst -> T3
                assign_join(
                    &cfg.join[1],
                    &r2_in,
                    &r2_low,
                    &r2_high,
                    &r2_join_val,
                    &edges.iter().map(|e| e.dst).collect(),
                )?;

                // join[0]: r1.dst -> T2
                assign_join(
                    &cfg.join[0],
                    &r1_in,
                    &r1_low,
                    &r1_high,
                    &r1_join_val,
                    &edges.iter().map(|e| e.dst).collect(),
                )?;

                // assign agg helper
                let mut assign_agg_stage = |a: &AggConfig<F>,
                                            base_len: usize,
                                            sorted_rows: &Vec<[u64; 3]>,
                                            run: &Vec<u64>,
                                            emit: &Vec<[u64; 2]>,
                                            tbl: &Vec<[u64; 2]>,
                                            val_in_vec: &Vec<u64>|
                 -> Result<(), Error> {
                    // val_in
                    for i in 0..base_len {
                        region.assign_advice(
                            || "val_in",
                            a.val_in,
                            i,
                            || Value::known(F::from(val_in_vec[i])),
                        )?;
                    }

                    // sorted triple + sentinel row
                    for i in 0..base_len {
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
                        base_len,
                        || Value::known(F::from(PAD_KEY)),
                    )?;
                    region.assign_advice(
                        || "sorted_dst_s",
                        a.sorted[1],
                        base_len,
                        || Value::known(F::from(PAD_KEY)),
                    )?;
                    region.assign_advice(
                        || "sorted_val_s",
                        a.sorted[2],
                        base_len,
                        || Value::known(F::from(0u64)),
                    )?;

                    // run_sum
                    for i in 0..base_len {
                        region.assign_advice(
                            || "run_sum",
                            a.run_sum,
                            i,
                            || Value::known(F::from(run[i])),
                        )?;
                    }

                    // emit pairs
                    for i in 0..base_len {
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

                    // tbl pairs
                    for i in 0..base_len {
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
                    }

                    // map_pair: row0=(0,0), rows 1..=n copy tbl[0..n-1]
                    region.assign_advice(
                        || "map_k0",
                        a.map_pair[0],
                        0,
                        || Value::known(F::from(0u64)),
                    )?;
                    region.assign_advice(
                        || "map_v0",
                        a.map_pair[1],
                        0,
                        || Value::known(F::from(0u64)),
                    )?;
                    for i in 0..base_len {
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

                    // map_key_next[0..n-1]=tbl[i].key, last row=PAD
                    for i in 0..base_len {
                        region.assign_advice(
                            || "map_kn",
                            a.map_key_next,
                            i,
                            || Value::known(F::from(tbl[i][0])),
                        )?;
                    }
                    region.assign_advice(
                        || "map_kn_last",
                        a.map_key_next,
                        base_len,
                        || Value::known(F::from(PAD_KEY)),
                    )?;

                    // enable map selectors
                    for i in 0..=base_len {
                        a.q_map_tbl.enable(&mut region, i)?; // complex (lookup table gate)
                    }
                    a.q_map_first.enable(&mut region, 0)?;
                    for i in 0..base_len {
                        a.q_map_link.enable(&mut region, i)?;
                        a.q_map_shift.enable(&mut region, i)?;
                    }
                    a.q_map_last.enable(&mut region, base_len)?;

                    // enable sortedness checks
                    for i in 0..base_len.saturating_sub(1) {
                        a.q_sort.enable(&mut region, i)?;
                        a.q_tbl_sort.enable(&mut region, i)?;
                    }

                    // group gates
                    if base_len > 0 {
                        a.q_first.enable(&mut region, 0)?;
                        a.q_emit.enable(&mut region, 0)?;
                    }
                    for i in 1..base_len {
                        a.q_accu.enable(&mut region, i)?;
                        a.q_emit.enable(&mut region, i)?;
                    }

                    // IsZero and Lt witness assignment
                    let iz_same_prev_chip = IsZeroChip::construct(a.iz_same_prev.clone());
                    let iz_same_next_chip = IsZeroChip::construct(a.iz_same_next.clone());
                    let iz_src_eq_chip = IsZeroChip::construct(a.iz_src_eq.clone());
                    let iz_tbl_eq_chip = IsZeroChip::construct(a.iz_tbl_key_eq.clone());
                    let lt_src_chip = LtChip::<F, NUM_BYTES>::construct(a.lt_src_cur_next.clone());
                    let lt_tbl_chip =
                        LtChip::<F, NUM_BYTES>::construct(a.lt_tbl_key_cur_next.clone());

                    for i in 0..base_len.saturating_sub(1) {
                        lt_src_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(sorted_rows[i][0])),
                            Value::known(F::from(sorted_rows[i + 1][0])),
                        )?;
                        lt_tbl_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(tbl[i][0])),
                            Value::known(F::from(tbl[i + 1][0])),
                        )?;
                    }
                    for i in 1..base_len {
                        let diff = F::from(sorted_rows[i][0]) - F::from(sorted_rows[i - 1][0]);
                        iz_same_prev_chip.assign(&mut region, i, Value::known(diff))?;
                    }
                    for i in 0..base_len {
                        let next_src = if i + 1 < base_len {
                            sorted_rows[i + 1][0]
                        } else {
                            PAD_KEY
                        };
                        let diff = F::from(next_src) - F::from(sorted_rows[i][0]);
                        iz_same_next_chip.assign(&mut region, i, Value::known(diff))?;
                    }
                    for i in 0..base_len.saturating_sub(1) {
                        let diff = F::from(sorted_rows[i + 1][0]) - F::from(sorted_rows[i][0]);
                        iz_src_eq_chip.assign(&mut region, i, Value::known(diff))?;
                    }
                    for i in 0..base_len.saturating_sub(1) {
                        let diff = F::from(tbl[i + 1][0]) - F::from(tbl[i][0]);
                        iz_tbl_eq_chip.assign(&mut region, i, Value::known(diff))?;
                    }

                    // NOTE: enabling perm selectors depends on your PermAnyConfig API.
                    // If your PermAnyChip uses fixed columns or internal selectors, keep your existing enable logic here.

                    Ok(())
                };

                // agg[0] r4 -> T4 with val_in=1
                let r4_vals = vec![1u64; n];
                assign_agg_stage(
                    &cfg.agg[0],
                    n,
                    &r4_sorted,
                    &r4_run,
                    &r4_emit,
                    &t4_tbl,
                    &r4_vals,
                )?;

                // agg[1] r3 -> T3 with val_in=r3_val_filt
                assign_agg_stage(
                    &cfg.agg[1],
                    n,
                    &r3_sorted,
                    &r3_run,
                    &r3_emit,
                    &t3_tbl,
                    &r3_val_filt,
                )?;

                // agg[2] r2 -> T2 with val_in=r2_val_filt
                assign_agg_stage(
                    &cfg.agg[2],
                    n,
                    &r2_sorted,
                    &r2_run,
                    &r2_emit,
                    &t2_tbl,
                    &r2_val_filt,
                )?;

                // assign r3 filter lt witnesses
                {
                    let lt_cd_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_cd.clone());
                    for i in 0..n {
                        cfg.q_r3_filt.enable(&mut region, i)?;
                        lt_cd_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(edges[i].src)),
                            Value::known(F::from(edges[i].dst)),
                        )?;
                    }
                }

                // assign r2 filter lt witnesses
                {
                    let lt_bc_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_bc.clone());
                    for i in 0..n {
                        cfg.q_r2_filt.enable(&mut region, i)?;
                        lt_bc_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(edges[i].src)),
                            Value::known(F::from(edges[i].dst)),
                        )?;
                    }
                }

                // contrib + lt_ab witnesses
                {
                    let lt_ab_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_ab.clone());
                    for i in 0..n {
                        cfg.q_r1_contrib.enable(&mut region, i)?;
                        region.assign_advice(
                            || "contrib",
                            cfg.contrib,
                            i,
                            || Value::known(F::from(contrib[i])),
                        )?;
                        lt_ab_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(edges[i].src)),
                            Value::known(F::from(edges[i].dst)),
                        )?;
                    }
                }

                // sum
                let mut running: u128 = 0;
                for i in 0..n {
                    running += contrib[i] as u128;
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

                // output at last row
                let out_cell = region.assign_advice(
                    || "out",
                    cfg.out,
                    n - 1,
                    || Value::known(F::from(answer)),
                )?;
                Ok(out_cell)
            },
        )?;

        Ok(out_cell)
    }
}

// ---- LOOKUPS (must use complex selectors for lookup-table gating) ----
impl<F: Field + Ord> Circuit<F> for GraphPath4OrderCircuit<F> {
    type Config = GraphPath4OrderConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let mut cfg = GraphPath4OrderChip::<F>::configure(meta);

        // Tables:
        // T4 = cfg.agg[0]
        // T3 = cfg.agg[1]
        // T2 = cfg.agg[2]
        let t4 = cfg.agg[0].clone();
        let t3 = cfg.agg[1].clone();
        let t2 = cfg.agg[2].clone();

        // helper: add join lookups (gap + map)
        let mut add_join =
            |step: usize, rel_dst: Column<Advice>, j: JoinConfig<F>, t: AggConfig<F>| {
                // Gap pair: (low, high) are consecutive keys around dst, when in_next=0
                meta.lookup_any(format!("gap pair step {}", step), move |m| {
                    let q_in = m.query_selector(j.q_lookup_complex); // complex
                    let inx = m.query_advice(j.in_next, Rotation::cur());
                    let gate = q_in * (Expression::Constant(F::ONE) - inx);

                    let low = m.query_advice(j.low, Rotation::cur());
                    let high = m.query_advice(j.high, Rotation::cur());

                    let q_tbl = m.query_selector(t.q_map_tbl); // complex
                    let key = m.query_advice(t.map_pair[0], Rotation::cur());
                    let keyn = m.query_advice(t.map_key_next, Rotation::cur());

                    vec![
                        (gate.clone() * low, q_tbl.clone() * key),
                        (gate * high, q_tbl * keyn),
                    ]
                });

                // Map: (in*dst, val) exists in (map_key, map_val)
                // If in=0, we force val=0 and in*dst=0, so (0,0) hits dummy row0.
                meta.lookup_any(format!("map step {}", step), move |m| {
                    let q_in = m.query_selector(j.q_lookup_complex); // complex
                    let inx = m.query_advice(j.in_next, Rotation::cur());

                    let dst = m.query_advice(rel_dst, Rotation::cur());
                    let v = m.query_advice(j.val, Rotation::cur());

                    let q_tbl = m.query_selector(t.q_map_tbl); // complex
                    let tk = m.query_advice(t.map_pair[0], Rotation::cur());
                    let tv = m.query_advice(t.map_pair[1], Rotation::cur());

                    vec![
                        (q_in.clone() * inx.clone() * dst, q_tbl.clone() * tk),
                        (q_in * v, q_tbl * tv),
                    ]
                });
            };

        // r3.dst -> T4
        add_join(2, cfg.r[2][1], cfg.join[2].clone(), t4);
        // r2.dst -> T3
        add_join(1, cfg.r[1][1], cfg.join[1].clone(), t3);
        // r1.dst -> T2
        add_join(0, cfg.r[0][1], cfg.join[0].clone(), t2);

        cfg
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = GraphPath4OrderChip::construct(config);
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
        let params_path = "/home2/binbin/PoneglyphDB/src/proof/param17";
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

    fn dp_count_path4_order(edges: &[Edge]) -> u64 {
        use std::collections::HashMap;

        // T4[d] = outdeg(d)
        let mut t4 = HashMap::<u64, u64>::new();
        for e in edges {
            *t4.entry(e.src).or_insert(0) += 1;
        }

        // T3[c] = Σ_{(c->d)} T4[d] * [c<d]
        let mut t3 = HashMap::<u64, u64>::new();
        for e in edges {
            let c = e.src;
            let d = e.dst;
            if c < d {
                let v = *t4.get(&d).unwrap_or(&0);
                *t3.entry(c).or_insert(0) += v;
            }
        }

        // T2[b] = Σ_{(b->c)} T3[c] * [b<c]
        let mut t2 = HashMap::<u64, u64>::new();
        for e in edges {
            let b = e.src;
            let c = e.dst;
            if b < c {
                let v = *t3.get(&c).unwrap_or(&0);
                *t2.entry(b).or_insert(0) += v;
            }
        }

        // ans = Σ_{(a->b)} T2[b] * [a<b]
        let mut ans: u128 = 0;
        for e in edges {
            let a = e.src;
            let b = e.dst;
            if a < b {
                ans += *t2.get(&b).unwrap_or(&0) as u128;
            }
        }

        ans as u64
    }

    #[test]
    fn test() {
        // Use REAL dataset R1.tsv as requested
        let base_path = "/home2/binbin/PoneglyphDB/src/graph_data";

        // let mut edges =
        //     read_edges_csv(&format!("{}/last/lastfm_asia_edges.csv", base_path)).unwrap();

        // let mut edges =
        //     read_edges(&format!("{}/facebook/facebook_combined.txt", base_path)).unwrap();

        let mut edges = read_edges(&format!("{}/wiki/wiki_Vote.txt", base_path)).unwrap();

        // edges.truncate(100);

        // expected COUNT(*) for:
        // (a->b),(b->c),(c->d),(d->e) with a<b<c<d
        let cnt = dp_count_path4_order(&edges);

        let circuit = GraphPath4OrderCircuit::<Fp> {
            edges,
            _marker: PhantomData,
        };

        let public_input = vec![Fp::from(cnt)];
        let k = 16;

        // let test = true;
        let test = false;

        if test {
            let prover = MockProver::run(k, &circuit, vec![public_input]).unwrap();
            prover.assert_satisfied();
        } else {
            let proof_path = "/home2/binbin/PoneglyphDB/src/proof/wiki_proof_q2";
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
    }
}
