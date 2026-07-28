use halo2_proofs::halo2curves::ff::PrimeField;
use halo2_proofs::plonk::Expression;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::circuits::conserve_idx::{
    assign_conserve, assign_row_index, configure_conserve, configure_row_index, ConserveConfig,
    RowIndexConfig,
};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};
use crate::circuits::card_preserve::{
    assign_cp_agg, assign_cp_join, assign_cp_root, build_cp_stage, configure_cp_agg,
    configure_cp_join, configure_cp_root, wire_cp_edge, CpAggConfig, CpJoinConfig, CpRootConfig,
};

use crate::data::graph_data_processing::Edge;

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

const NUM_BYTES: usize = 8;
const MAX_SENTINEL: u64 = u64::MAX;
const PAD_KEY: u64 = MAX_SENTINEL; // pad key goes last in ASC
const PAD_VAL: u64 = 0;

/// Every key handed to the Cardinality Preservation Check is shifted by this,
/// so key 0 stays free for the gadget's dummy table row. Node ids in the graph
/// datasets start at 0 (facebook_combined), and shifting preserves the
/// equality joins the check traverses.
const SHIFT_ID: u64 = 1;

/// Test hook, off in every benchmark path: when set, the prover moves one
/// joinable r1 tuple to the residual side and re-reduces the relations around
/// it, so the partition still passes Conservation and only condition (4) can
/// catch it. This is exactly the cheat a residual-side-only argument misses,
/// so the negative test in this module is what shows the Cardinality
/// Preservation Check is not vacuous.
pub static HIDE_ONE_CLEAN_TUPLE: AtomicBool = AtomicBool::new(false);

/// Test hook, off in every benchmark path: when set, the prover skips the
/// semijoin reduction entirely and declares every real tuple clean, leaving
/// the residual section empty. Conservation still holds and both channels of
/// condition (4) then agree trivially, so this is exactly the escape that
/// Pairwise Consistency has to close, and the third direction of the test in
/// this module is what shows condition (3) closes it.
pub static MARK_ALL_CLEAN: AtomicBool = AtomicBool::new(false);

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
    // pins the sorted view's sentinel cell at row n to PAD
    q_sentinel: Selector,
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

    // the leaf stage's input multiplicity is one per r4 tuple
    q_r4_val_one: Selector,

    contrib: Column<Advice>,

    // sum
    q_sum_first: Selector,
    q_sum_accu: Selector,
    sum: Column<Advice>,

    out: Column<Advice>,
    // ties the published cell to the accumulator
    q_out: Selector,

    // ---------------- (1) Conservation Check ----------------
    // R^_i == R^_i^c U+ R^_i^r over the INDEXED relation. r1..r4 are four
    // occurrences of ONE Edge table, so the shared rows are conserved four
    // times, once per node, each against that node's own indicator.
    row_idx: RowIndexConfig,
    cons: [ConserveConfig; 4],
    cflag: [Column<Advice>; 4],
    // booleanity of both copies of the indicator, and the masked clean keys
    // below, on ALL n rows of every relation rather than on a prover-chosen
    // prefix
    q_flag: Selector,

    // ---------------- Pairwise Consistency, condition (3) ----------------
    // The six lookups of the three tree edges read one masked key column per
    // side: `flag * (key + SHIFT_ID)` over the partition group's rows, which is
    // 0 on every row the clean part does not use. The clean row set the lookups
    // range over is therefore carried by the (boolean) indicator column, not by
    // a selector range derived from the private |R^c|, and the selectors below
    // are enabled on all n rows of every relation.
    q_pw_in: [Selector; 4],
    // pw_dst[e] masks the dst key of the parent of edge e (relations r1, r2, r3)
    // pw_src[e] masks the src key of the child of edge e (relations r2, r3, r4)
    pw_dst: [Column<Advice>; 3],
    pw_src: [Column<Advice>; 3],

    // ---------------- Cardinality Preservation Check, condition (4) --------
    // shifted join keys, sk[k][j] = r[k][j] + SHIFT_ID
    sk: [[Column<Advice>; 2]; 4],
    q_shift: Selector,

    // edge index 0 = (r3 parent, r4 child), 1 = (r2, r3), 2 = (r1, r2)
    cp_agg: [CpAggConfig<F, NUM_BYTES>; 3],
    cp_join: [CpJoinConfig<F, NUM_BYTES>; 3],
    cp_root: CpRootConfig,

    // the leaf's input-channel multiplicity, pinned to 1
    cp_one: Column<Advice>,
    // (mu_all, mu_cln) of the two internal nodes r3 and r2
    cp_mu: [[Column<Advice>; 2]; 2],
    q_cp_mu: Selector,
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

        // The group-boundary detector on the last real row reads the sentinel
        // cell at row n through `iz_same_next`, and nondecreasing alone does not
        // pin that cell. A prover that sets it equal to the last real key makes
        // `iz_same_next` report "same group" there, so the emit gate below
        // *demands* PAD instead of that group's sum and the highest-key group
        // disappears from the table the next join reads: the parent carrying
        // that key then certifies its absence with a gap witness that really
        // does hold in the forged table, takes val = 0, and the answer is
        // undercounted. This is the same gate as
        // `card_preserve.rs: "cp: sorted view sentinel is PAD"`, and pinning the
        // VALUE rather than requiring a strict last increase is deliberate: a
        // stage whose rows include padding keyed at PAD legitimately ends at
        // PAD, and those rows must not be emitted as a group.
        let q_sentinel = meta.selector();
        meta.create_gate("agg: sorted view sentinel is PAD", |m| {
            let q = m.query_selector(q_sentinel);
            vec![
                q * (m.query_advice(sorted[0], Rotation::cur())
                    - Expression::Constant(F::from(PAD_KEY))),
            ]
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
            q_sentinel,
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
        //
        // The three ordering chips below are range-gated on the bare filter
        // selector, not on `q * in_next` as in g_sql2_obj.rs. Their `lt` cell
        // is the relation's `keep` bit, which the Conservation Check and the
        // input channel of the Cardinality Preservation Check both read on
        // every base row, so it has to be constrained on every base row.
        // Gating on `in_next` left it free wherever the DP had already forced
        // the fetched value to 0. The witness was assigned on every row
        // already, so nothing on the DP side changes.
        let q_r3_filt = meta.selector();
        let lt_cd = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_r3_filt),
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

        // The leaf stage r4 has no filter of its own, so nothing above ties its
        // val_in column to anything: it was free advice, and a uniform val_in = v
        // makes T4 = v * outdeg, which the sorted view carries as an exact
        // permutation and which then scales T3, T2, contrib and the answer by v.
        // T4[d] must be the out-degree, so the input multiplicity is 1 per tuple.
        let q_r4_val_one = meta.selector();
        meta.create_gate("r4 val_in is 1", |m| {
            let q = m.query_selector(q_r4_val_one);
            vec![
                q * (m.query_advice(agg[0].val_in, Rotation::cur()) - Expression::Constant(F::ONE)),
            ]
        });

        // r2 filter: agg[2].val_in = join[1].val * [b<c]
        let q_r2_filt = meta.selector();
        let lt_bc = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_r2_filt),
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
            |m| m.query_selector(q_r1_contrib),
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

        // The published answer. `out` is the only cell of this circuit copied to
        // the instance column, and without this gate no polynomial identity
        // relates it to the accumulator: the prover could publish any COUNT and
        // every other condition of the file would still hold, because the sum
        // chain and `out` were connected only by the host-side witness. Degree 2.
        let q_out = meta.selector();
        meta.create_gate("out equals sum", |m| {
            let q = m.query_selector(q_out);
            let o = m.query_advice(out, Rotation::cur());
            let s = m.query_advice(sum, Rotation::cur());
            vec![q * (o - s)]
        });

        // ================= condition (1): Conservation Check =================
        // g_sql2_obj.rs has no partition at all, so one is introduced here:
        // a clean indicator per base row, and one shuffle per relation between
        // the filtered base rows and a materialized [R^c | R^r | PAD] layout.
        // The indicator rides along as the third column of both sides, which
        // is what binds it to the partition.
        fn cols3<FF: PrimeField>(meta: &mut ConstraintSystem<FF>) -> [Column<Advice>; 3] {
            [
                meta.advice_column(),
                meta.advice_column(),
                meta.advice_column(),
            ]
        }
        fn cols2<FF: PrimeField>(meta: &mut ConstraintSystem<FF>) -> [Column<Advice>; 2] {
            [meta.advice_column(), meta.advice_column()]
        }

        let cflag: [Column<Advice>; 4] = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        // ---------- (1) Conservation Check ----------
        // One permutation per node, between the indexed relation R^_i and the
        // concatenation of its two parts. The indices are distinct, so R^_i is
        // a set even though Edge is a bag, and the single permutation rules out
        // an occurrence being fabricated, lost, duplicated or counted on both
        // sides: no Non-Membership Check.
        let row_idx = configure_row_index::<F>(meta);
        let cons: [ConserveConfig; 4] =
            std::array::from_fn(|k| configure_conserve::<F>(meta, &row_idx, &r[k], cflag[k]));

        // ---------- Selector Check ----------
        // One bit per committed row per tree node. r1..r4 are four occurrences
        // of ONE Edge table, so each node carries its own bit column over the
        // shared rows. Selection is by position, so `g_sql2_obj.rs`'s four
        // filtered views, four partition groups and four Conservation shuffles
        // all go away: there is no second copy of the data to compare against
        // the first.
        //
        // Booleanity was a consequence of those shuffles before; now it is a
        // gate, and it is load-bearing. The clean channel of the propagation
        // multiplies by the bit at every node, so a non-boolean value like
        // 1 + d * s^{-1} would pay back exactly the deficit that deselecting a
        // participating tuple creates and the root equality would accept it.
        //
        // The predicate half confines the selection to the rows the WHERE
        // clause keeps: r1, r2 and r3 carry a<b, b<c and c<d respectively, and
        // r4 has none. `g_sql2_obj.rs` got this structurally, by padding the
        // indicator away wherever the predicate failed; here it is one gate,
        // and it is what makes the input channel of (3) count the join of the
        // predicate-filtered relations.
        let q_flag = meta.selector();
        {
            let cf = cflag;
            let pred: [Option<LtConfig<F, NUM_BYTES>>; 4] =
                [Some(lt_ab), Some(lt_bc), Some(lt_cd), None];
            meta.create_gate("selector is a bit and implies its predicate", move |m| {
                let q = m.query_selector(q_flag);
                let one = Expression::Constant(F::ONE);
                let mut cs = Vec::with_capacity(7);
                for k in 0..4 {
                    let c = m.query_advice(cf[k], Rotation::cur());
                    cs.push(q.clone() * c.clone() * (one.clone() - c.clone()));
                    if let Some(lt) = pred[k] {
                        cs.push(q.clone() * c * (one.clone() - lt.is_lt(m, None)));
                    }
                }
                cs
            });
        }

        // ============= condition (3): Pairwise Consistency =============
        // Two mutual Membership Checks per join-tree edge, each looking one
        // clean key column up directly in the adjacent relation's clean key
        // column. Both sides read the key column of the *partition* group over
        // its clean rows, never the base relation's column: a lookup against the
        // base rows would only certify membership in R_i, which is the weaker
        // statement g_sql2_obj.rs already made and is not condition (3). The
        // tuple columns of the partition group are tied to the base relation by
        // the Conservation Check above, so those rows are exactly pi_K(R^c).
        //
        // An earlier version of this file routed each direction through an
        // intermediate advice column holding the deduplicated key set of the
        // relation it looked into. That column was plain prover advice and
        // nothing bound it to the relation it claimed to enumerate, so setting
        // the two tables of an edge to each other's key sets satisfied both
        // lookups for an arbitrary partition and condition (3) was vacuous.
        // Looking the two key columns up in each other leaves no free advice:
        // the two containments now hold between the actual clean key columns
        // and together they are the set equality condition (3) asks for, at one
        // advice column and one complex selector less per side.
        //
        // Keys enter both sides shifted by SHIFT_ID: a lookup_any expression is
        // evaluated on every row of the circuit and both sides are 0 wherever
        // their selector is off or their row is not clean, so 0 is unavoidably in
        // the table, and node 0 is a real node in these datasets.
        //
        // Which rows the two containments range over is the delicate part. An
        // earlier version gated both sides with a selector enabled on the
        // partition group's clean prefix [0, n_cln), a range the prover derives
        // from the private partition. Then a row past n_cln + n_res carried no
        // constraint at all: a dangling tuple marked clean on the base side was
        // absorbed by the Conservation shuffle at such a tail row, a PAD triple
        // filled the residual slot it vacated, the multiset equality still held,
        // and no Pairwise Consistency lookup ever saw the smuggled row, so
        // conditions (9) and (10) stopped speaking about the same R^c. Shrinking
        // the prefix to nothing made the whole condition vacuous.
        //
        // The clean row set is therefore carried by the indicator column itself:
        // each side reads a masked key `flag * (key + SHIFT_ID)`, which is the
        // shifted key on a clean row and 0 on every other row, and the selector
        // is enabled on all n rows of the relation. 0 is in every table (the rows
        // past n have the selector off), so the masked-out rows are free. The
        // mask lives in its own advice column rather than in the lookup
        // expression on purpose: a lookup's required degree is
        // 2 + input_degree + table_degree, so multiplying inside the expression
        // would have pushed both sides to degree 3 and the whole constraint
        // system from 7 to 8.
        //
        // fresh complex selectors: a simple selector may not appear in a lookup
        // expression, so the q_cln_flag ones above cannot be reused here
        let q_pw_in: [Selector; 4] = std::array::from_fn(|_| meta.complex_selector());
        let pw_dst: [Column<Advice>; 3] = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let pw_src: [Column<Advice>; 3] = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        {
            let pd = pw_dst;
            let ps = pw_src;
            let cf = cflag;
            let rr = r;
            meta.create_gate("pw: masked clean keys", move |m| {
                let q = m.query_selector(q_flag);
                let sh = Expression::Constant(F::from(SHIFT_ID));
                let mut cs = Vec::with_capacity(6);
                for e in 0..3 {
                    // parent of edge e is relation e, child is relation e + 1
                    let fp = m.query_advice(cf[e], Rotation::cur());
                    let kp = m.query_advice(rr[e][1], Rotation::cur());
                    cs.push(
                        q.clone()
                            * (m.query_advice(pd[e], Rotation::cur()) - fp * (kp + sh.clone())),
                    );
                    let fc = m.query_advice(cf[e + 1], Rotation::cur());
                    let kc = m.query_advice(rr[e + 1][0], Rotation::cur());
                    cs.push(
                        q.clone()
                            * (m.query_advice(ps[e], Rotation::cur()) - fc * (kc + sh.clone())),
                    );
                }
                cs
            });
        }

        let mut pw_edge = |name: &'static str,
                           q_in: Selector,
                           in_col: Column<Advice>,
                           q_t: Selector,
                           tbl_col: Column<Advice>| {
            meta.lookup_any(name, move |m| {
                let lhs = m.query_selector(q_in) * m.query_advice(in_col, Rotation::cur());
                let rhs = m.query_selector(q_t) * m.query_advice(tbl_col, Rotation::cur());
                vec![(lhs, rhs)]
            });
        };

        let pw_names: [[&'static str; 2]; 3] = [
            ["pw: r1^c dst in r2^c src", "pw: r2^c src in r1^c dst"],
            ["pw: r2^c dst in r3^c src", "pw: r3^c src in r2^c dst"],
            ["pw: r3^c dst in r4^c src", "pw: r4^c src in r3^c dst"],
        ];
        for e in 0..3 {
            // parent of edge e is r_{e+1}, child is r_{e+2}
            let par_key = pw_dst[e]; // masked dst of the parent
            let chi_key = pw_src[e]; // masked src of the child
            pw_edge(pw_names[e][0], q_pw_in[e], par_key, q_pw_in[e + 1], chi_key);
            pw_edge(pw_names[e][1], q_pw_in[e + 1], chi_key, q_pw_in[e], par_key);
        }

        // =========== condition (4): Cardinality Preservation Check ===========
        // Shifted keys first: key 0 is reserved for the gadget's dummy row.
        let sk: [[Column<Advice>; 2]; 4] = [cols2(meta), cols2(meta), cols2(meta), cols2(meta)];
        let q_shift = meta.selector();
        meta.create_gate("shifted keys for the cardinality check", move |m| {
            let q = m.query_selector(q_shift);
            let sh = Expression::Constant(F::from(SHIFT_ID));
            let mut cs = Vec::with_capacity(8);
            for k in 0..4 {
                for j in 0..2 {
                    cs.push(
                        q.clone()
                            * (m.query_advice(sk[k][j], Rotation::cur())
                                - m.query_advice(r[k][j], Rotation::cur())
                                - sh.clone()),
                    );
                }
            }
            cs
        });

        // One fixed column serves every Lt chip of the check, so the whole
        // check costs a single u8 range table.
        let cp_u8 = meta.fixed_column();

        let cp_one = meta.advice_column();
        let cp_mu: [[Column<Advice>; 2]; 2] = [cols2(meta), cols2(meta)];

        // Child side of each edge, over the child relation's own rows. The leaf
        // r4 carries (1, c_4); the internal nodes r3 and r2 carry the fresh mu
        // columns the recurrence gate below ties to their own child edge.
        let cp_agg = [
            configure_cp_agg::<F, NUM_BYTES>(
                meta,
                cp_u8,
                sk[3][0], // r4.src
                cp_one,
                cflag[3],
                PAD_KEY,
            ),
            configure_cp_agg::<F, NUM_BYTES>(
                meta,
                cp_u8,
                sk[2][0], // r3.src
                cp_mu[0][0],
                cp_mu[0][1],
                PAD_KEY,
            ),
            configure_cp_agg::<F, NUM_BYTES>(
                meta,
                cp_u8,
                sk[1][0], // r2.src
                cp_mu[1][0],
                cp_mu[1][1],
                PAD_KEY,
            ),
        ];

        // Parent side of each edge, over the parent relation's own rows.
        let parent_key = [sk[2][1], sk[1][1], sk[0][1]]; // r3.dst, r2.dst, r1.dst
        let cp_join = [
            configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, parent_key[0]),
            configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, parent_key[1]),
            configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, parent_key[2]),
        ];
        for e in 0..3 {
            wire_cp_edge(meta, &cp_join[e], &cp_agg[e], parent_key[e]);
        }

        let cp_root = configure_cp_root::<F>(meta);

        // The recurrences of both channels, one selector for all of them since
        // every relation occupies the same rows. Degree 3 each.
        let q_cp_mu = meta.selector();
        {
            let s_all = [cp_join[0].s_all, cp_join[1].s_all, cp_join[2].s_all];
            let s_cln = [cp_join[0].s_cln, cp_join[1].s_cln, cp_join[2].s_cln];
            // the bound keep * c of r3, r2, r1
            let cf = [cflag[2], cflag[1], cflag[0]];
            let mu_all = [cp_mu[0][0], cp_mu[1][0], cp_root.mu_all];
            let mu_cln = [cp_mu[0][1], cp_mu[1][1], cp_root.mu_cln];
            let pred = [lt_cd, lt_bc, lt_ab];
            meta.create_gate("cp: multiplicity recurrences along the path", move |m| {
                let q = m.query_selector(q_cp_mu);
                let mut cs = Vec::with_capacity(7);

                // leaf r4: one input-channel extension per tuple
                cs.push(
                    q.clone()
                        * (m.query_advice(cp_one, Rotation::cur()) - Expression::Constant(F::ONE)),
                );

                // r3, then r2, then the root r1
                for i in 0..3 {
                    let p = pred[i].is_lt(m, None);
                    cs.push(
                        q.clone()
                            * (m.query_advice(mu_all[i], Rotation::cur())
                                - p * m.query_advice(s_all[i], Rotation::cur())),
                    );
                    cs.push(
                        q.clone()
                            * (m.query_advice(mu_cln[i], Rotation::cur())
                                - m.query_advice(cf[i], Rotation::cur())
                                    * m.query_advice(s_cln[i], Rotation::cur())),
                    );
                }
                cs
            });
        }

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
            q_r4_val_one,
            contrib,
            q_sum_first,
            q_sum_accu,
            sum,
            out,
            q_out,

            row_idx,
            cons,
            cflag,
            q_flag,

            q_pw_in,
            pw_dst,
            pw_src,

            sk,
            q_shift,
            cp_agg,
            cp_join,
            cp_root,
            cp_one,
            cp_mu,
            q_cp_mu,
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

    /// The honest prover's clean instance: the fully reduced instance of the
    /// 4-path, i.e. the tuples of each relation that extend to a full join
    /// result. One up sweep along the chain, then one down sweep from the root,
    /// which is exact for a tree. `out[k]` is the indicator of relation
    /// `r_{k+1}`, in base row order.
    ///
    /// `drop_r1` removes one root tuple before the down sweep and is used only
    /// by the negative test: the result is still the fully reduced instance of
    /// a smaller instance, so Conservation, and pairwise consistency between
    /// the clean projections, both still hold, and only the clean join is
    /// smaller than the input join.
    fn reduce_clean(edges: &[Edge], drop_r1: Option<usize>) -> [Vec<u64>; 4] {
        let n = edges.len();
        // all four relations are the same Edge table, so the three ordering
        // predicates are the same per-row bit src < dst
        let lt: Vec<bool> = edges.iter().map(|e| e.src < e.dst).collect();

        // up sweep: which tuples extend downwards
        let keys4: HashSet<u64> = edges.iter().map(|e| e.src).collect();
        let up3: Vec<bool> = (0..n)
            .map(|i| lt[i] && keys4.contains(&edges[i].dst))
            .collect();
        let keys3: HashSet<u64> = (0..n).filter(|&i| up3[i]).map(|i| edges[i].src).collect();
        let up2: Vec<bool> = (0..n)
            .map(|i| lt[i] && keys3.contains(&edges[i].dst))
            .collect();
        let keys2: HashSet<u64> = (0..n).filter(|&i| up2[i]).map(|i| edges[i].src).collect();
        let mut cl1: Vec<bool> = (0..n)
            .map(|i| lt[i] && keys2.contains(&edges[i].dst))
            .collect();
        if let Some(i) = drop_r1 {
            cl1[i] = false;
        }

        // down sweep: which of those also extend upwards
        let d1: HashSet<u64> = (0..n).filter(|&i| cl1[i]).map(|i| edges[i].dst).collect();
        let cl2: Vec<bool> = (0..n)
            .map(|i| up2[i] && d1.contains(&edges[i].src))
            .collect();
        let d2: HashSet<u64> = (0..n).filter(|&i| cl2[i]).map(|i| edges[i].dst).collect();
        let cl3: Vec<bool> = (0..n)
            .map(|i| up3[i] && d2.contains(&edges[i].src))
            .collect();
        let d3: HashSet<u64> = (0..n).filter(|&i| cl3[i]).map(|i| edges[i].dst).collect();
        let cl4: Vec<bool> = (0..n).map(|i| d3.contains(&edges[i].src)).collect();

        let b2u = |v: Vec<bool>| v.into_iter().map(|b| b as u64).collect::<Vec<u64>>();
        [b2u(cl1), b2u(cl2), b2u(cl3), b2u(cl4)]
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

        // Every Lt chip of the Cardinality Preservation Check shares one u8
        // fixed column, so a single load covers all twelve of them.
        LtChip::<F, NUM_BYTES>::construct(cfg.cp_agg[0].lt_key_cur_next).load(layouter)?;

        // The empty-input layout enables no selector at all, so under a
        // verifying key generated for n == 0 nothing is constrained and the
        // published cell is free. That is not a hole this file can close: with no
        // rows there is no accumulator to tie `out` to, and every condition of
        // the gate is equally empty. Any deployment has to fix the verifying key
        // for the relation size it verifies.
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
        // Host-side truncation: the answer and the running sum are both carried
        // as u64, so a count past 2^64 would wrap. The in-circuit accumulator
        // adds in F and does not wrap, so the honest prover simply cannot satisfy
        // "sum_accu" past that point: this is a completeness limit at
        // astronomically large counts, not a soundness hole, and contorting the
        // circuit into u128 limbs would cost far more than it is worth.
        let answer = answer_u128 as u64;

        // -------- the partition, condition (1) --------
        // keep bits per relation: the ordering predicate of r1/r2/r3, and the
        // constant 1 of r4, which has none
        let lt_bit: Vec<u64> = edges.iter().map(|e| (e.src < e.dst) as u64).collect();
        let keep: [Vec<u64>; 4] = [
            lt_bit.clone(),
            lt_bit.clone(),
            lt_bit.clone(),
            vec![1u64; n],
        ];

        // The honest prover's R^c is the fully reduced instance: exactly the
        // tuples that extend to a full join result.
        let tamper = HIDE_ONE_CLEAN_TUPLE.load(Ordering::Relaxed);
        let mark_all = MARK_ALL_CLEAN.load(Ordering::Relaxed);
        let drop_r1 = if tamper {
            let honest = Self::reduce_clean(edges, None);
            (0..n).find(|&i| honest[0][i] == 1)
        } else {
            None
        };
        let cln: [Vec<u64>; 4] = if mark_all {
            // the escape condition (3) closes: no reduction at all, every real
            // tuple clean and the residual section empty
            std::array::from_fn(|k| keep[k].clone())
        } else {
            Self::reduce_clean(edges, drop_r1)
        };

        // the two sides of each Conservation Check: the filtered base rows, and
        // the partition laid out as [R^c rows | R^r rows | PAD rows]
        let mut filt_rows: Vec<Vec<[u64; 3]>> = Vec::with_capacity(4);
        let mut part_rows: Vec<Vec<[u64; 3]>> = Vec::with_capacity(4);
        let mut n_cln = [0usize; 4];
        let mut n_res = [0usize; 4];
        for k in 0..4 {
            let mut filt: Vec<[u64; 3]> = Vec::with_capacity(n);
            let mut cl: Vec<[u64; 3]> = Vec::new();
            let mut res: Vec<[u64; 3]> = Vec::new();
            for i in 0..n {
                if keep[k][i] == 1 {
                    filt.push([edges[i].src, edges[i].dst, cln[k][i]]);
                    if cln[k][i] == 1 {
                        cl.push([edges[i].src, edges[i].dst, 1]);
                    } else {
                        res.push([edges[i].src, edges[i].dst, 0]);
                    }
                } else {
                    filt.push([PAD_KEY, PAD_KEY, 0]);
                }
            }
            n_cln[k] = cl.len();
            n_res[k] = res.len();
            let mut part = cl;
            part.extend(res);
            while part.len() < n {
                part.push([PAD_KEY, PAD_KEY, 0]);
            }
            filt_rows.push(filt);
            part_rows.push(part);
        }

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

                    // enable sortedness checks. q_sort covers the comparison
                    // into the pinned sentinel row too, so the sorted key column
                    // is nondecreasing all the way to PAD.
                    for i in 0..base_len {
                        a.q_sort.enable(&mut region, i)?;
                    }
                    for i in 0..base_len.saturating_sub(1) {
                        a.q_tbl_sort.enable(&mut region, i)?;
                    }
                    // pin the sentinel cell the emit gate reads on the last real
                    // row, so the highest-key group is always recognised as a
                    // group end and cannot be suppressed
                    a.q_sentinel.enable(&mut region, base_len)?;

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

                    // the last comparison reads the PAD sentinel at row base_len
                    for i in 0..base_len {
                        let next_src = if i + 1 < base_len {
                            sorted_rows[i + 1][0]
                        } else {
                            PAD_KEY
                        };
                        lt_src_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(sorted_rows[i][0])),
                            Value::known(F::from(next_src)),
                        )?;
                    }
                    for i in 0..base_len.saturating_sub(1) {
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
                    for i in 0..base_len {
                        let next_src = if i + 1 < base_len {
                            sorted_rows[i + 1][0]
                        } else {
                            PAD_KEY
                        };
                        let diff = F::from(next_src) - F::from(sorted_rows[i][0]);
                        iz_src_eq_chip.assign(&mut region, i, Value::known(diff))?;
                    }
                    for i in 0..base_len.saturating_sub(1) {
                        let diff = F::from(tbl[i + 1][0]) - F::from(tbl[i][0]);
                        iz_tbl_eq_chip.assign(&mut region, i, Value::known(diff))?;
                    }

                    // The two shuffles of the stage. Leaving these selectors off
                    // made both arguments 0-vs-0 on every row, i.e. dead: the
                    // sorted view was then unrelated to (src, dst, val_in) and
                    // the T-table unrelated to the emitted group pairs, so the
                    // whole DP chain was severed from the base relation and any
                    // constant could be added to every table value without the
                    // map or gap lookups noticing. Both sides of each shuffle are
                    // enabled on exactly the same base rows [0, base_len), so the
                    // rows past the relation contribute an all-zero tuple to both
                    // sides and cancel.
                    for i in 0..base_len {
                        a.perm_sort.q_perm1.enable(&mut region, i)?;
                        a.perm_sort.q_perm2.enable(&mut region, i)?;
                        a.perm_tbl.q_perm1.enable(&mut region, i)?;
                        a.perm_tbl.q_perm2.enable(&mut region, i)?;
                    }

                    Ok(())
                };

                // agg[0] r4 -> T4 with val_in=1, pinned below by "r4 val_in is 1"
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

                // the leaf stage's input multiplicity is one per r4 tuple
                for i in 0..n {
                    cfg.q_r4_val_one.enable(&mut region, i)?;
                }

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

                // ================= condition (1): the partition =================
                // clean indicator and shifted keys on the base rows
                for k in 0..4 {
                    for i in 0..n {
                        region.assign_advice(
                            || "cflag",
                            cfg.cflag[k],
                            i,
                            || Value::known(F::from(cln[k][i])),
                        )?;
                        region.assign_advice(
                            || "sk_src",
                            cfg.sk[k][0],
                            i,
                            || Value::known(F::from(edges[i].src + SHIFT_ID)),
                        )?;
                        region.assign_advice(
                            || "sk_dst",
                            cfg.sk[k][1],
                            i,
                            || Value::known(F::from(edges[i].dst + SHIFT_ID)),
                        )?;
                    }
                }
                for i in 0..n {
                    cfg.q_shift.enable(&mut region, i)?;
                }

                // ---- (1) Conservation Check, one permutation per node ----
                let edge_rows: Vec<Vec<u64>> = edges
                    .iter()
                    .map(|e| vec![e.src, e.dst])
                    .collect();
                assign_row_index(&mut region, &cfg.row_idx, n)?;
                for k in 0..4 {
                    assign_conserve(&mut region, &cfg.cons[k], &edge_rows, &cln[k])?;
                }

                // ============= condition (3): Pairwise Consistency =============
                // The six lookups read one masked key column per side,
                // flag * (key + SHIFT_ID) over the partition group's rows, so the
                // clean row set they range over is the indicator column itself
                // and the selectors are enabled on all n rows. A row whose
                // indicator is 0 masks to 0, which every table contains.
                for k in 0..4 {
                    for i in 0..n {
                        cfg.q_pw_in[k].enable(&mut region, i)?;
                    }
                }
                for i in 0..n {
                    cfg.q_flag.enable(&mut region, i)?;
                    for e in 0..3 {
                        // parent of edge e is relation e, child is relation e + 1
                        let fp = F::from(cln[e][i]);
                        let kp = F::from(edges[i].dst) + F::from(SHIFT_ID);
                        region.assign_advice(
                            || "pw masked dst",
                            cfg.pw_dst[e],
                            i,
                            || Value::known(fp * kp),
                        )?;
                        let fc = F::from(cln[e + 1][i]);
                        let kc = F::from(edges[i].src) + F::from(SHIFT_ID);
                        region.assign_advice(
                            || "pw masked src",
                            cfg.pw_src[e],
                            i,
                            || Value::known(fc * kc),
                        )?;
                    }
                }

                // ============= CARDINALITY PRESERVATION CHECK =============
                // condition (4): one traversal of the chain r4 -> r3 -> r2 -> r1
                // carrying two multiplicities per tuple, then one equality
                // between the two root sums.
                for i in 0..n {
                    region.assign_advice(|| "cp_one", cfg.cp_one, i, || Value::known(F::ONE))?;
                    cfg.q_cp_mu.enable(&mut region, i)?;
                }

                let shifted_src: Vec<u64> = edges.iter().map(|e| e.src + SHIFT_ID).collect();
                let shifted_dst: Vec<u64> = edges.iter().map(|e| e.dst + SHIFT_ID).collect();

                // the leaf carries (key, v_all, v_cln) = (r4.src, 1, c_4)
                let mut rows: Vec<[u64; 3]> =
                    (0..n).map(|i| [shifted_src[i], 1, cln[3][i]]).collect();
                let mut cp_sums = (0u64, 0u64);

                for e in 0..3 {
                    let stage = build_cp_stage(&rows, PAD_KEY);
                    assign_cp_agg(&mut region, &cfg.cp_agg[e], &rows, &stage)?;
                    let fetched = assign_cp_join(
                        &mut region,
                        &cfg.cp_join[e],
                        &shifted_dst,
                        &stage,
                        PAD_KEY,
                    )?;

                    // parent of edge e: r3, then r2, then the root r1
                    let par = 2 - e;
                    let mu: Vec<(u64, u64)> = (0..n)
                        .map(|i| (keep[par][i] * fetched[i].0, cln[par][i] * fetched[i].1))
                        .collect();

                    if e < 2 {
                        for i in 0..n {
                            region.assign_advice(
                                || "cp mu_all",
                                cfg.cp_mu[e][0],
                                i,
                                || Value::known(F::from(mu[i].0)),
                            )?;
                            region.assign_advice(
                                || "cp mu_cln",
                                cfg.cp_mu[e][1],
                                i,
                                || Value::known(F::from(mu[i].1)),
                            )?;
                        }
                        // the parent becomes the child of the next edge, keyed
                        // by its own src
                        rows = (0..n).map(|i| [shifted_src[i], mu[i].0, mu[i].1]).collect();
                    } else {
                        cp_sums = assign_cp_root(&mut region, &cfg.cp_root, &mu)?;
                    }
                }

                if !tamper && !mark_all {
                    debug_assert_eq!(
                        cp_sums.0, cp_sums.1,
                        "cardinality preservation: |R^c join| != |R join|"
                    );
                    debug_assert_eq!(
                        cp_sums.0, answer,
                        "the input channel of condition (4) must count the query answer"
                    );
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

                // output at the last row, tied to the accumulator by "out equals
                // sum" so the published cell is the count the circuit computed
                let out_cell = region.assign_advice(
                    || "out",
                    cfg.out,
                    n - 1,
                    || Value::known(F::from(answer)),
                )?;
                cfg.q_out.enable(&mut region, n - 1)?;
                Ok(out_cell)
            },
        )?;

        Ok(out_cell)
    }
}

// ---- LOOKUPS (must use complex selectors for lookup-table gating) ----
/// Full constraint-system setup for `GraphPath4OrderCircuit`.
/// Extracted verbatim so the circuit and any wrapper that embeds it
/// (see `crate::inline_bind`) configure IDENTICAL constraints -- the
/// chip's own `configure` alone is NOT sufficient here.
pub fn configure_path4order_full<F: Field + Ord>(
    meta: &mut ConstraintSystem<F>,
) -> GraphPath4OrderConfig<F> {
    let mut cfg = GraphPath4OrderChip::<F>::configure(meta);

    // Tables:
    // T4 = cfg.agg[0]
    // T3 = cfg.agg[1]
    // T2 = cfg.agg[2]
    let t4 = cfg.agg[0].clone();
    let t3 = cfg.agg[1].clone();
    let t2 = cfg.agg[2].clone();

    // helper: add join lookups (gap + map)
    let mut add_join = |step: usize, rel_dst: Column<Advice>, j: JoinConfig<F>, t: AggConfig<F>| {
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

impl<F: Field + Ord> Circuit<F> for GraphPath4OrderCircuit<F> {
    type Config = GraphPath4OrderConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        configure_path4order_full(meta)
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

    use halo2_proofs::dev::{MockProver, VerifyFailure};

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
    use std::sync::atomic::Ordering;
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
    #[ignore = "inherited heavy end-to-end proof; the fast check is test_cardinality_preservation"]
    fn test() {
        // Use REAL dataset R1.tsv as requested
        let base_path = &crate::paths::graph_dir();

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
        // MockProver branch only: the full wiki_Vote slice is 103_689 rows, so
        // it needs 2^17, the same degree as the param file the proof path loads.
        let k = 17;

        // VPJOIN_MOCK=1 runs the MockProver check instead of a real proof.
        let test = std::env::var("VPJOIN_MOCK")
            .map(|v| v == "1")
            .unwrap_or(false);

        if test {
            let prover = MockProver::run(k, &circuit, vec![public_input]).unwrap();
            prover.assert_satisfied();
        } else {
            let proof_path = &crate::paths::proof_file("wiki_proof_q2_new");
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
    }

    /// The maximum gate degree of the whole constraint system. Every fix in this
    /// file has to stay at or below the degree the circuit already had, since a
    /// rise doubles every FFT of the prover.
    #[test]
    fn test_max_gate_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        let mut cs = ConstraintSystem::<Fp>::default();
        let _ = <GraphPath4OrderCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
        let degree = cs.degree();
        println!("cs.degree() = {}", degree);
        println!(
            "advice = {}, fixed = {}, selectors = {}, lookups = {}, shuffles = {}, gates = {}",
            cs.num_advice_columns(),
            cs.num_fixed_columns(),
            cs.num_selectors(),
            cs.lookups().len(),
            cs.shuffles().len(),
            cs.gates().len()
        );
        assert!(
            degree <= 7,
            "the maximum gate degree rose to {}, which costs more than any fix \
             in this file is worth",
            degree
        );
    }

    /// Fast correctness check of the One-Pass OBJ conditions this file carries:
    /// a truncated slice of the real dataset under MockProver, which verifies
    /// every gate, shuffle and lookup of the circuit without paying for a real
    /// proof, then the negative direction of condition (4), then the all-clean
    /// escape that condition (3) closes.
    #[test]
    fn test_cardinality_preservation() {
        // small enough for MockProver, large enough that the reduction really
        // drops tuples on every relation of the chain
        const N_EDGES: usize = 4000;
        let k = 14;

        let base_path = &crate::paths::graph_dir();
        let mut edges = read_edges(&format!("{}/wiki/wiki_Vote.txt", base_path)).unwrap();
        assert!(
            !edges.is_empty(),
            "dataset not found under {}/wiki/wiki_Vote.txt",
            base_path
        );
        edges.truncate(N_EDGES);

        // Non-vacuity of the third direction below: the slice must contain a
        // tuple that passes its ordering predicate yet is not in the reduced
        // instance, i.e. a dangling tuple the all-clean partition keeps. On an
        // acyclic query a pairwise consistent instance is exactly a fully
        // reduced one, so as soon as one relation's clean part is strictly
        // smaller than its filtered part, the all-clean partition violates
        // condition (3) on some edge.
        let honest = GraphPath4OrderChip::<Fp>::reduce_clean(&edges, None);
        let n_keep = edges.iter().filter(|e| e.src < e.dst).count();
        let n_cln: Vec<usize> = (0..4)
            .map(|r| honest[r].iter().filter(|&&c| c == 1).count())
            .collect();
        println!(
            "slice: {} edges, {} pass src<dst, |R^c| = {:?}",
            edges.len(),
            n_keep,
            n_cln
        );
        assert!(
            n_cln[0] < n_keep,
            "the slice has no dangling r1 tuple, so the all-clean partition \
             would legitimately satisfy condition (3)"
        );

        let cnt = dp_count_path4_order(&edges);
        let circuit = GraphPath4OrderCircuit::<Fp> {
            edges,
            _marker: PhantomData,
        };
        let public_input = vec![Fp::from(cnt)];

        let prover = MockProver::run(k, &circuit, vec![public_input.clone()]).unwrap();
        prover.assert_satisfied();

        // Negative direction: the same witness with one joinable r1 tuple
        // hidden in the residual side and the relations re-reduced around it,
        // so the partition still satisfies Conservation and the clean
        // projections still agree on every tree edge. Only condition (4) can
        // see this, so the circuit must now reject.
        super::HIDE_ONE_CLEAN_TUPLE.store(true, Ordering::Relaxed);
        let tampered = MockProver::run(k, &circuit, vec![public_input.clone()]).unwrap();
        let verdict = tampered.verify();
        super::HIDE_ONE_CLEAN_TUPLE.store(false, Ordering::Relaxed);

        let failures = verdict.expect_err("condition (4) accepted a hidden joinable tuple");
        {
            let mut kinds: Vec<String> = failures
                .iter()
                .map(|f| match f {
                    VerifyFailure::ConstraintNotSatisfied { constraint, .. } => {
                        format!("gate {}", constraint)
                    }
                    VerifyFailure::Lookup { name, .. } => format!("lookup {}", name),
                    other => format!("{}", other),
                })
                .collect();
            kinds.sort();
            kinds.dedup();
            println!("hidden joinable tuple rejected by: {:#?}", kinds);
        }
        assert!(
            failures
                .iter()
                .any(|f| format!("{:?}", f).contains("cardinality preservation")),
            "the circuit rejected, but not through the Cardinality Preservation Check: {:?}",
            failures
        );

        // Third direction: the all-clean partition. No reduction at all, every
        // real tuple declared clean and the residual section empty.
        // Conservation still holds and both channels of condition (4) compute
        // the same number on every row, so condition (4) alone accepts this.
        // Pairwise Consistency is what must reject it, through a "pw: " lookup.
        super::MARK_ALL_CLEAN.store(true, Ordering::Relaxed);
        let all_clean = MockProver::run(k, &circuit, vec![public_input]).unwrap();
        let verdict = all_clean.verify();
        super::MARK_ALL_CLEAN.store(false, Ordering::Relaxed);

        let failures = verdict.expect_err("condition (3) accepted the all-clean partition");
        {
            let mut kinds: Vec<String> = failures
                .iter()
                .map(|f| match f {
                    VerifyFailure::ConstraintNotSatisfied { constraint, .. } => {
                        format!("gate {}", constraint)
                    }
                    VerifyFailure::Lookup { name, .. } => format!("lookup {}", name),
                    other => format!("{}", other),
                })
                .collect();
            kinds.sort();
            kinds.dedup();
            println!("all-clean partition rejected by: {:#?}", kinds);
        }
        assert!(
            failures.iter().any(|f| matches!(
                f,
                VerifyFailure::Lookup { name, .. } if name.starts_with("pw: ")
            )),
            "the circuit rejected the all-clean partition, but not through a \
             Pairwise Consistency lookup: {:?}",
            failures
        );
    }
}
