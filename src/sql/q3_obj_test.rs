//! TPC-H Q3 under the updated One-Pass OBJ.
//!
//! Same query, same clean/residual witness and same aggregation as
//! `q3_obj.rs`. What changes is condition (4) of the gate, the one that rules
//! out a joinable tuple hidden in the residual side.
//!
//! `q3_obj.rs` argues it on the residual side: every residual order tuple
//! carries a membership flag per neighbour plus a gap witness, and the
//! Emptiness Logic gate asks that at least one neighbour miss. A condition
//! phrased over the residual relations alone cannot see a tuple wrongly moved
//! into the residual whose join partners stay clean, so this file replaces it
//! with the Cardinality Preservation Check of `crate::circuits::card_preserve`:
//! one traversal of the join tree carrying two multiplicities per tuple, one
//! counting join extensions over the inputs and one over the clean instance,
//! and a single equality constraint between the two root sums.
//!
//! Join tree: `orders` is the root, with `customer` as its child on
//! `c_custkey = o_custkey` and `lineitem` as its child on
//! `o_orderkey = l_orderkey`. Both children are leaves, so
//!
//!   v_all  = pred                     v_cln  = c * pred            (children)
//!   mu_all = pred_o * s_all_c * s_all_l
//!   mu_cln = c_o * pred_o * s_cln_c * s_cln_l                      (root)
//!
//! The clean indicator `c` is bound by the Conservation Check itself: each
//! relation's permutation carries one extra column holding `keep * c` on the
//! input side and a constant `1` on the clean rows against `0` on the residual
//! rows of the partition side, so the multiset equality forces the indicator on
//! a base row to mark exactly the occurrences that went to `R^c`.
//!
//! Everything else, including the ORDER BY proof over the emitted groups, is
//! unchanged from `q3_obj.rs`.

use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};
use crate::circuits::card_preserve::{
    assign_cp_agg, assign_cp_join, assign_cp_root, build_cp_stage, configure_cp_agg,
    configure_cp_join, configure_cp_root, wire_cp_edge, CpAggConfig, CpJoinConfig, CpRootConfig,
};

use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const NUM_BYTES: usize = 5;
const MAX_SENTINEL: u64 = (1u64 << (8 * NUM_BYTES)) - 1; // 2^40-1

const SCALE: u64 = 1000;

const PAD_OK: u64 = MAX_SENTINEL; // pad orderkey (max)
const PAD_DATE: u64 = MAX_SENTINEL; // pad orderdate (max -> last when ASC)
const PAD_SHIP: u64 = MAX_SENTINEL; // pad shippriority (max)
const PAD_REV: u64 = 0; // pad revenue (min -> last when DESC)

/// Test hook, off in every benchmark path: when set, the prover moves one
/// joinable tuple to the residual side and re-reduces the neighbours around
/// it, so the partition still passes Conservation, Non-Membership and Pairwise
/// Consistency and only condition (4) can catch it. This is exactly the cheat
/// a residual-side-only argument misses, so the negative test in this module
/// is what shows the Cardinality Preservation Check is not vacuous.
pub static HIDE_ONE_CLEAN_TUPLE: AtomicBool = AtomicBool::new(false);

/// Test hook, off in every benchmark path: when set, the prover skips the
/// semijoin reduction entirely and declares every tuple that passes its
/// predicate clean. Conservation still holds and both channels of condition (4)
/// then agree row by row, so this is the escape that only Pairwise Consistency
/// can close, and the negative direction for it in this module is what shows
/// condition (3) is doing work.
pub static MARK_ALL_CLEAN: AtomicBool = AtomicBool::new(false);

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

#[derive(Clone, Debug)]
pub struct TestCircuitConfig<F: Field + Ord> {
    q_enable: Vec<Selector>,

    customer: Vec<Column<Advice>>, // 2
    orders: Vec<Column<Advice>>,   // 4
    lineitem: Vec<Column<Advice>>, // 4

    check: Vec<Column<Advice>>,     // 0..2 used
    condition: Vec<Column<Advice>>, // 3

    // clean indicator per base row: [customer, orders, lineitem]
    cflag: Vec<Column<Advice>>,

    o_join: Vec<Column<Advice>>,    // 4
    o_disjoin: Vec<Column<Advice>>, // 4
    c_join: Vec<Column<Advice>>,    // 2
    c_disjoin: Vec<Column<Advice>>, // 2
    l_join: Vec<Column<Advice>>,    // 4
    l_disjoin: Vec<Column<Advice>>, // 4

    lt_compare_condition: Vec<LtConfig<F, NUM_BYTES>>,
    equal_condition: Vec<IsZeroConfig<F>>,

    instance: Column<Instance>,
    instance_test: Column<Advice>,

    // ---------------- Cardinality Preservation Check ----------------
    // condition (4): |R^c join| == |R join|, over the tree rooted at orders
    cp_agg_c: CpAggConfig<F, NUM_BYTES>,  // child customer, keyed by c_custkey
    cp_agg_l: CpAggConfig<F, NUM_BYTES>,  // child lineitem, keyed by l_orderkey
    cp_join_c: CpJoinConfig<F, NUM_BYTES>, // orders -> customer
    cp_join_l: CpJoinConfig<F, NUM_BYTES>, // orders -> lineitem
    cp_root: CpRootConfig,
    q_cp_mu: Selector, // the two root product gates

    // clean/residual flag on the partition side of each Conservation Check
    q_cln_flag: Vec<Selector>, // rows of R^c: flag == 1
    q_res_flag: Vec<Selector>, // rows of R^r: flag == 0

    // ---------------- permutation pads (orders) ----------------
    o_filt_pad: Vec<Column<Advice>>,
    o_part_pad: Vec<Column<Advice>>,
    perm_orders: PermAnyConfig,

    // ---------------- permutation pads (customer) ----------------
    c_filt_pad: Vec<Column<Advice>>,
    c_part_pad: Vec<Column<Advice>>,
    perm_customer: PermAnyConfig,

    // ---------------- permutation pads (lineitem) ----------------
    l_filt_pad: Vec<Column<Advice>>,
    l_part_pad: Vec<Column<Advice>>,
    perm_lineitem: PermAnyConfig,

    // -------- join<->join membership lookups (4 directions) --------
    // input selectors
    q_in_o_cust_in_c: Selector, // o_join.custkey -> c_join table
    q_in_c_cust_in_o: Selector, // c_join.custkey -> o_join table
    q_in_l_okey_in_o: Selector, // l_join.orderkey -> o_join table
    q_in_o_okey_in_l: Selector, // o_join.orderkey -> l_join table

    // table selectors (gate the table side!)

    // key-table advice columns

    // ---------- l_join -> l_sorted permutation ----------
    l_sorted: Vec<Column<Advice>>, // 4 cols (same as lineitem)
    perm_lsort: PermAnyConfig,

    q_line: Selector,  // enable line_rev + emit + same_next
    q_first: Selector, // run_sum[0] = line_rev[0]
    q_accu: Selector,  // enable run_sum recurrence (needs prev)

    // line-level helpers over l_sorted
    line_rev: Column<Advice>,
    run_sum: Column<Advice>,
    iz_same_prev: IsZeroConfig<F>, // cur_okey - prev_okey == 0 (only rows >=1)
    iz_same_next: IsZeroConfig<F>, // next_okey - cur_okey == 0 (rows 0..n-1)

    // ---------- emitted padded result (length = l_join.len()) ----------
    res_pad: Vec<Column<Advice>>, // [okey, odate, shippri, revenue]

    // attach (okey,odate,shippri) via lookup into o_join
    q_res_lookup: Selector,
    q_tbl_o_join: Selector, // gates table-side (o_join rows)

    // ---------- ORDER BY proof (res_pad -> res_sorted) ----------
    res_sorted: Vec<Column<Advice>>, // same 4 cols
    perm_res: PermAnyConfig,
    q_sort_res: Selector, // rows 0..n-2

    lt_rev_next_cur: LtConfig<F, NUM_BYTES>, // rev_next < rev_cur
    lt_date_cur_next: LtConfig<F, NUM_BYTES>, // date_cur < date_next
    iz_rev_eq: IsZeroConfig<F>,              // rev_cur - rev_next == 0
    iz_date_eq: IsZeroConfig<F>,             // date_cur - date_next == 0
}

#[derive(Debug, Clone)]
pub struct TestChip<F: Field + Ord> {
    config: TestCircuitConfig<F>,
}

impl<F: Field + Ord> TestChip<F> {
    pub fn construct(config: TestCircuitConfig<F>) -> Self {
        Self { config }
    }

    // ---------- small assignment helpers ----------
    fn assign_table_u64(
        region: &mut Region<'_, F>,
        tag: &'static str,
        cols: &[Column<Advice>],
        rows: &[Vec<u64>],
    ) -> Result<Vec<Vec<AssignedCell<F, F>>>, Error> {
        let mut out: Vec<Vec<AssignedCell<F, F>>> = Vec::with_capacity(rows.len());
        for (i, r) in rows.iter().enumerate() {
            let mut row_cells = Vec::with_capacity(cols.len());
            for (j, &v) in r.iter().enumerate() {
                let cell = region.assign_advice(|| tag, cols[j], i, || Value::known(F::from(v)))?;
                row_cells.push(cell);
            }
            out.push(row_cells);
        }
        Ok(out)
    }

    fn assign_table_f(
        region: &mut Region<'_, F>,
        tag: &'static str,
        cols: &[Column<Advice>],
        rows: &[Vec<F>],
    ) -> Result<(), Error> {
        for (i, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                region.assign_advice(|| tag, cols[j], i, || Value::known(v))?;
            }
        }
        Ok(())
    }

    fn assign_part_pad_and_link(
        region: &mut Region<'_, F>,
        tag: &'static str,
        part_cols: &[Column<Advice>],
        part_rows: &[Vec<F>],                   // total rows
        join_cells: &[Vec<AssignedCell<F, F>>], // join_len x width
        dis_cells: &[Vec<AssignedCell<F, F>>],  // dis_len x width
    ) -> Result<(), Error> {
        let join_len = join_cells.len();
        let dis_len = dis_cells.len();

        for (i, r) in part_rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                let part_cell =
                    region.assign_advice(|| tag, part_cols[j], i, || Value::known(v))?;

                // **THIS is the missing “query o_join/o_disjoin” part**
                // tie o_part_pad[i] == o_join[i] or o_disjoin[i-join_len] via equality constraints
                //
                // The last column of part_rows is the clean indicator, which has
                // no counterpart in the join/disjoin tables: it is pinned by
                // q_cln_flag / q_res_flag instead, so it is skipped here.
                if i < join_len {
                    if j < join_cells[i].len() {
                        region.constrain_equal(part_cell.cell(), join_cells[i][j].cell())?;
                    }
                } else if i < join_len + dis_len {
                    if j < dis_cells[i - join_len].len() {
                        region
                            .constrain_equal(part_cell.cell(), dis_cells[i - join_len][j].cell())?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> TestCircuitConfig<F> {
        let instance = meta.instance_column();
        meta.enable_equality(instance);
        let instance_test = meta.advice_column();
        meta.enable_equality(instance_test);

        let mut q_enable = vec![];
        for _ in 0..4 {
            q_enable.push(meta.selector());
        }

        let mut q_sort = vec![];
        for _ in 0..9 {
            q_sort.push(meta.selector());
        }

        let mut q_join = vec![];
        for i in 0..8 {
            if i < 2 {
                q_join.push(meta.selector());
            } else {
                q_join.push(meta.complex_selector());
            }
        }

        let q_accu = meta.selector();

        let customer = vec![meta.advice_column(), meta.advice_column()];
        let orders = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let lineitem = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();

        let mut condition = vec![];
        for _ in 0..3 {
            condition.push(meta.advice_column());
        }
        meta.enable_equality(condition[2]);

        let mut check = vec![];
        for _ in 0..4 {
            check.push(meta.advice_column());
        }

        // clean indicator per base row of customer / orders / lineitem
        let cflag = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for &c in cflag.iter() {
            meta.enable_equality(c);
        }

        let c_join = vec![meta.advice_column(), meta.advice_column()];
        let c_disjoin = vec![meta.advice_column(), meta.advice_column()];

        let o_join = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let o_disjoin = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();

        let l_join = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let l_disjoin = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();

        // enable equality on join/disjoin columns (needed for constrain_equal with *_part_pad)
        for &col in o_join
            .iter()
            .chain(o_disjoin.iter())
            .chain(c_join.iter())
            .chain(c_disjoin.iter())
            .chain(l_join.iter())
            .chain(l_disjoin.iter())
        {
            meta.enable_equality(col);
        }

        // ---------------- predicate chips ----------------
        // IsZero for c_mktsegment == :1  => check[0] in {0,1}
        let is_zero_aux = meta.advice_column();
        let mut equal_condition = vec![];
        let iz = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_enable[0]),
            |m| {
                m.query_advice(customer[0], Rotation::cur())
                    - m.query_advice(condition[0], Rotation::cur())
            },
            is_zero_aux,
        );
        equal_condition.push(iz.clone());

        meta.create_gate("c_mktsegment == :1 => check0", |m| {
            let s = m.query_selector(q_enable[0]);
            let out = m.query_advice(check[0], Rotation::cur());
            vec![
                s.clone() * (iz.expr() * (out.clone() - Expression::Constant(F::ONE))),
                s * (Expression::Constant(F::ONE) - iz.expr()) * out,
            ]
        });

        // Lt for o_orderdate < :2 => check[1] (also booleanize check[1])
        let mut lt_compare_condition = vec![];
        let lt_o = LtChip::configure(
            meta,
            |m| m.query_selector(q_enable[1]),
            |m| m.query_advice(orders[0], Rotation::cur()),
            |m| m.query_advice(condition[1], Rotation::cur()),
        );
        meta.create_gate("o_orderdate < :2 => check1", |m| {
            let s = m.query_selector(q_enable[1]);
            let out = m.query_advice(check[1], Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                s.clone() * (lt_o.is_lt(m, None) - out.clone()),
                s * out.clone() * (one - out), // boolean
            ]
        });
        lt_compare_condition.push(lt_o);

        // Lt for :2 < l_shipdate => check[2] (also booleanize check[2])
        let lt_l = LtChip::configure(
            meta,
            |m| m.query_selector(q_enable[2]),
            |m| m.query_advice(condition[2], Rotation::cur()),
            |m| m.query_advice(lineitem[3], Rotation::cur()),
        );
        meta.create_gate(":2 < l_shipdate => check2", |m| {
            let s = m.query_selector(q_enable[2]);
            let out = m.query_advice(check[2], Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                s.clone() * (lt_l.is_lt(m, None) - out.clone()),
                s * out.clone() * (one - out), // boolean
            ]
        });
        lt_compare_condition.push(lt_l);

        // ---------------- permutation configs (orders/customer/lineitem) ----------------
        fn mk_perm<FF: PrimeField>(
            meta: &mut ConstraintSystem<FF>,
            width: usize,
        ) -> (Vec<Column<Advice>>, Vec<Column<Advice>>, PermAnyConfig) {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            let mut a = vec![];
            let mut b = vec![];
            for _ in 0..width {
                a.push(meta.advice_column());
                b.push(meta.advice_column());
            }
            let perm = PermAnyChip::configure(meta, q1, q2, a.clone(), b.clone());
            (a, b, perm)
        }

        // One column wider than in q3_obj.rs: the last column of each pair
        // carries the clean indicator, so the Conservation Check binds it.
        let (o_filt_pad, o_part_pad, perm_orders) = mk_perm::<F>(meta, 5);
        let (c_filt_pad, c_part_pad, perm_customer) = mk_perm::<F>(meta, 3);
        let (l_filt_pad, l_part_pad, perm_lineitem) = mk_perm::<F>(meta, 5);

        // -------- link gates: filt_pad must equal (keep? base : PAD) --------
        // NOTE: this is what makes the permutation check talk about *real filtered rows*
        // The indicator column pads with 0, so a row dropped by the predicate
        // carries indicator 0 and contributes to neither channel of the
        // Cardinality Preservation Check.
        let pad_o: [u64; 5] = [MAX_SENTINEL, 0, MAX_SENTINEL, MAX_SENTINEL, 0];
        let pad_c: [u64; 3] = [MAX_SENTINEL, MAX_SENTINEL, 0];
        let pad_l: [u64; 5] = [MAX_SENTINEL, MAX_SENTINEL, MAX_SENTINEL, MAX_SENTINEL, 0];

        let mut link_filt_pad = |name: &'static str,
                                 q: Selector,
                                 keep_col: Column<Advice>,
                                 base: Vec<Column<Advice>>,
                                 filt: Vec<Column<Advice>>,
                                 pads: Vec<u64>| {
            meta.create_gate(name, move |m| {
                let q = m.query_selector(q);
                let keep = m.query_advice(keep_col, Rotation::cur());
                let one = Expression::Constant(F::ONE);
                let drop = one.clone() - keep.clone();

                let mut cs = vec![q.clone() * keep.clone() * (one.clone() - keep.clone())]; // keep boolean

                for j in 0..base.len() {
                    let b = m.query_advice(base[j], Rotation::cur());
                    let f = m.query_advice(filt[j], Rotation::cur());
                    let p = Expression::Constant(F::from(pads[j]));
                    cs.push(q.clone() * (f - (keep.clone() * b + drop.clone() * p)));
                }
                cs
            });
        };

        let mut o_base = orders.clone();
        o_base.push(cflag[1]);
        let mut c_base = customer.clone();
        c_base.push(cflag[0]);
        let mut l_base = lineitem.clone();
        l_base.push(cflag[2]);

        link_filt_pad(
            "link o_filt_pad = (check1? orders : PAD)",
            perm_orders.q_perm1,
            check[1],
            o_base,
            o_filt_pad.clone(),
            pad_o.to_vec(),
        );
        link_filt_pad(
            "link c_filt_pad = (check0? customer : PAD)",
            perm_customer.q_perm1,
            check[0],
            c_base,
            c_filt_pad.clone(),
            pad_c.to_vec(),
        );
        link_filt_pad(
            "link l_filt_pad = (check2? lineitem : PAD)",
            perm_lineitem.q_perm1,
            check[2],
            l_base,
            l_filt_pad.clone(),
            pad_l.to_vec(),
        );

        // -------- partition side of the indicator: 1 on R^c rows, 0 on R^r --------
        // The partition column group is laid out as [clean rows | residual rows
        // | pad rows], so one selector per section pins the indicator. Without
        // these the prover could mark a residual row clean and inflate the
        // clean channel of the check below.
        let q_cln_flag = (0..3).map(|_| meta.selector()).collect::<Vec<_>>();
        let q_res_flag = (0..3).map(|_| meta.selector()).collect::<Vec<_>>();

        for (idx, part) in [
            c_part_pad.clone(),
            o_part_pad.clone(),
            l_part_pad.clone(),
        ]
        .iter()
        .enumerate()
        {
            let flag_col = *part.last().unwrap();
            let q_c = q_cln_flag[idx];
            let q_r = q_res_flag[idx];
            meta.create_gate("clean indicator on the partition side", move |m| {
                let qc = m.query_selector(q_c);
                let qr = m.query_selector(q_r);
                let f = m.query_advice(flag_col, Rotation::cur());
                vec![
                    qc * (f.clone() - Expression::Constant(F::ONE)),
                    qr * f,
                ]
            });
        }

        // -------- membership lookup selectors/cols --------
        let q_in_o_cust_in_c = meta.complex_selector();
        let q_in_c_cust_in_o = meta.complex_selector();
        let q_in_l_okey_in_o = meta.complex_selector();
        let q_in_o_okey_in_l = meta.complex_selector();



        // -------- condition (3), Pairwise Consistency --------
        // Two mutual Membership Checks per tree edge, each looking one clean
        // relation's key column up directly in the adjacent clean relation's key
        // column. q3_obj.rs routed these through intermediate `tbl_*` advice
        // columns holding the deduplicated key sets, but nothing bound those
        // columns to the relations they claimed to enumerate: a prover could put
        // o_join's custkeys into the table o_join looks into and c_join's into
        // the table c_join looks into, and all four lookups would pass for an
        // arbitrary partition. Looking the columns up in each other removes the
        // free advice, and with it the escape, at one fewer column per edge
        // direction.
        //
        // A lookup input is 0 on every row where its selector is off, and the
        // table side is 0 on those rows too, so 0 is always in the table and the
        // gated-off rows cost nothing. Real custkeys and orderkeys are at least
        // 1 in TPC-H, so the containment is over the real keys.
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

        // edge (orders, customer) on custkey
        pw_edge(
            "pw: o_join.custkey in c_join.custkey",
            q_in_o_cust_in_c,
            o_join[2],
            q_in_c_cust_in_o,
            c_join[1],
        );
        pw_edge(
            "pw: c_join.custkey in o_join.custkey",
            q_in_c_cust_in_o,
            c_join[1],
            q_in_o_cust_in_c,
            o_join[2],
        );

        // edge (orders, lineitem) on orderkey
        pw_edge(
            "pw: l_join.orderkey in o_join.orderkey",
            q_in_l_okey_in_o,
            l_join[0],
            q_in_o_okey_in_l,
            o_join[3],
        );
        pw_edge(
            "pw: o_join.orderkey in l_join.orderkey",
            q_in_o_okey_in_l,
            o_join[3],
            q_in_l_okey_in_o,
            l_join[0],
        );

        // ---------------- Cardinality Preservation Check (condition (4)) ----------------
        // One fixed column serves every Lt chip of the check, so the whole
        // check costs a single u8 range table.
        let cp_u8 = meta.fixed_column();

        // Children of the root. Both are leaves, so their two multiplicity
        // columns are columns the circuit already has: the predicate bit is the
        // input channel and the bound indicator (keep * c) is the clean one.
        let cp_agg_c = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            customer[1], // c_custkey
            check[0],
            c_filt_pad[2],
            MAX_SENTINEL,
        );
        let cp_agg_l = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            lineitem[0], // l_orderkey
            check[2],
            l_filt_pad[4],
            MAX_SENTINEL,
        );

        // Parent side, on the rows of orders.
        let cp_join_c = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, orders[2]);
        let cp_join_l = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, orders[3]);
        wire_cp_edge(meta, &cp_join_c, &cp_agg_c, orders[2]);
        wire_cp_edge(meta, &cp_join_l, &cp_agg_l, orders[3]);

        // Root multiplicities and the single equality that compares the two
        // join cardinalities.
        let cp_root = configure_cp_root::<F>(meta);
        let q_cp_mu = meta.selector();
        {
            let s_all_c = cp_join_c.s_all;
            let s_cln_c = cp_join_c.s_cln;
            let s_all_l = cp_join_l.s_all;
            let s_cln_l = cp_join_l.s_cln;
            let pred_o = check[1];
            let cln_o = o_filt_pad[4];
            let mu_all = cp_root.mu_all;
            let mu_cln = cp_root.mu_cln;
            meta.create_gate("cp: root multiplicities over orders", move |m| {
                let q = m.query_selector(q_cp_mu);
                let all = m.query_advice(mu_all, Rotation::cur())
                    - m.query_advice(pred_o, Rotation::cur())
                        * m.query_advice(s_all_c, Rotation::cur())
                        * m.query_advice(s_all_l, Rotation::cur());
                let cln = m.query_advice(mu_cln, Rotation::cur())
                    - m.query_advice(cln_o, Rotation::cur())
                        * m.query_advice(s_cln_c, Rotation::cur())
                        * m.query_advice(s_cln_l, Rotation::cur());
                vec![q.clone() * all, q * cln]
            });
        }

        // Aggregate
        let q_line = meta.selector();
        let q_first = meta.selector();
        let q_accu = meta.selector();

        let q_res_lookup = meta.complex_selector();
        let q_tbl_o_join = meta.complex_selector();

        let q_sort_res = meta.selector();

        // l_sorted (4 cols)
        let l_sorted = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();

        // line helpers
        let line_rev = meta.advice_column();
        let run_sum = meta.advice_column();

        // emitted padded result + sorted result (each 4 cols)
        let res_pad = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let res_sorted = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();

        // perm: l_join <-> l_sorted
        let q_perm_l_in = meta.complex_selector();
        let q_perm_l_out = meta.complex_selector();
        let perm_lsort = PermAnyChip::configure(
            meta,
            q_perm_l_in,
            q_perm_l_out,
            l_join.clone(),
            l_sorted.clone(),
        );

        // perm: res_pad <-> res_sorted
        let q_perm_r_in = meta.complex_selector();
        let q_perm_r_out = meta.complex_selector();
        let perm_res = PermAnyChip::configure(
            meta,
            q_perm_r_in,
            q_perm_r_out,
            res_pad.clone(),
            res_sorted.clone(),
        );

        // Configure IsZero helpers (same_prev / same_next / eqs)
        let aux_same_prev = meta.advice_column();
        let aux_same_next = meta.advice_column();
        let aux_rev_eq = meta.advice_column();
        let aux_date_eq = meta.advice_column();

        let iz_same_prev = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_accu),
            |m| {
                m.query_advice(l_sorted[0], Rotation::cur())
                    - m.query_advice(l_sorted[0], Rotation::prev())
            },
            aux_same_prev,
        );

        let iz_same_next = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_line),
            |m| {
                m.query_advice(l_sorted[0], Rotation::next())
                    - m.query_advice(l_sorted[0], Rotation::cur())
            },
            aux_same_next,
        );

        // res_sorted: [0]=okey,[1]=odate,[2]=ship,[3]=rev
        let iz_rev_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| {
                m.query_advice(res_sorted[3], Rotation::cur())
                    - m.query_advice(res_sorted[3], Rotation::next())
            },
            aux_rev_eq,
        );

        let iz_date_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| {
                m.query_advice(res_sorted[1], Rotation::cur())
                    - m.query_advice(res_sorted[1], Rotation::next())
            },
            aux_date_eq,
        );

        // Configure revenue/date LT chips for ORDER BY gate
        let lt_rev_next_cur = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| m.query_advice(res_sorted[3], Rotation::next()), // rev_next
            |m| m.query_advice(res_sorted[3], Rotation::cur()),  // rev_cur
        );

        let lt_date_cur_next = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| m.query_advice(res_sorted[1], Rotation::cur()), // date_cur
            |m| m.query_advice(res_sorted[1], Rotation::next()), // date_next
        );

        // Gates: line_rev, run_sum, emit padded group row
        // line_rev = ext * (SCALE - disc)
        meta.create_gate("line_rev", |m| {
            let q = m.query_selector(q_line);
            let ext = m.query_advice(l_sorted[1], Rotation::cur());
            let disc = m.query_advice(l_sorted[2], Rotation::cur());
            let lr = m.query_advice(line_rev, Rotation::cur());
            let scale = Expression::Constant(F::from(SCALE));
            vec![q * (lr - ext * (scale - disc))]
        });

        // run_sum[0] = line_rev[0]
        meta.create_gate("run_sum_first", |m| {
            let q = m.query_selector(q_first);
            let rs = m.query_advice(run_sum, Rotation::cur());
            let lr = m.query_advice(line_rev, Rotation::cur());
            vec![q * (rs - lr)]
        });

        // run_sum[i] = same_prev * run_sum[i-1] + line_rev[i]
        meta.create_gate("run_sum_accu", |m| {
            let q = m.query_selector(q_accu);
            let same = iz_same_prev.expr(); // 1 if same group
            let rs_cur = m.query_advice(run_sum, Rotation::cur());
            let rs_prev = m.query_advice(run_sum, Rotation::prev());
            let lr = m.query_advice(line_rev, Rotation::cur());
            vec![q * (rs_cur - (same * rs_prev + lr))]
        });

        // emit group row only at last row of each orderkey group
        meta.create_gate("emit_res_pad", |m| {
            let q = m.query_selector(q_line);
            let one = Expression::Constant(F::ONE);
            let same_next = iz_same_next.expr();
            let is_last = one.clone() - same_next; // 1 if next != cur
            let not_last = one.clone() - is_last.clone();

            let cur_okey = m.query_advice(l_sorted[0], Rotation::cur());
            let rs = m.query_advice(run_sum, Rotation::cur());

            let out_okey = m.query_advice(res_pad[0], Rotation::cur());
            let out_date = m.query_advice(res_pad[1], Rotation::cur());
            let out_ship = m.query_advice(res_pad[2], Rotation::cur());
            let out_rev = m.query_advice(res_pad[3], Rotation::cur());

            let pad_ok = Expression::Constant(F::from(PAD_OK));
            let pad_date = Expression::Constant(F::from(PAD_DATE));
            let pad_ship = Expression::Constant(F::from(PAD_SHIP));
            let pad_rev = Expression::Constant(F::from(PAD_REV));

            vec![
                // okey / rev forced in both cases
                q.clone() * (out_okey - (is_last.clone() * cur_okey + not_last.clone() * pad_ok)),
                q.clone() * (out_rev - (is_last.clone() * rs + not_last.clone() * pad_rev)),
                // date/ship must be PAD when not last; last rows are constrained by lookup below
                q.clone() * not_last.clone() * (out_date - pad_date),
                q * not_last * (out_ship - pad_ship),
            ]
        });
        // Lookup: (orderkey, orderdate, shippriority) must exist in o_join
        meta.lookup_any("attach o_join attrs to res_pad", |m| {
            let q_in = m.query_selector(q_res_lookup);
            let q_tbl = m.query_selector(q_tbl_o_join);

            let one = Expression::Constant(F::ONE);
            let is_last = one - iz_same_next.expr(); // reuse same_next on l_sorted rows

            let gate = q_in * is_last;

            let ok = m.query_advice(res_pad[0], Rotation::cur());
            let od = m.query_advice(res_pad[1], Rotation::cur());
            let sp = m.query_advice(res_pad[2], Rotation::cur());

            vec![
                (
                    gate.clone() * ok,
                    q_tbl.clone() * m.query_advice(o_join[3], Rotation::cur()),
                ),
                (
                    gate.clone() * od,
                    q_tbl.clone() * m.query_advice(o_join[0], Rotation::cur()),
                ),
                (
                    gate * sp,
                    q_tbl * m.query_advice(o_join[1], Rotation::cur()),
                ),
            ]
        });

        // ORDER BY gate on res_sorted
        meta.create_gate("ORDER BY revenue DESC, o_orderdate ASC", |m| {
            let q = m.query_selector(q_sort_res);

            let rev_gt = lt_rev_next_cur.is_lt(m, None); // next < cur
            let rev_eq = iz_rev_eq.expr();

            let date_lt = lt_date_cur_next.is_lt(m, None); // cur < next
            let date_eq = iz_date_eq.expr();
            let date_le = date_lt + date_eq;

            vec![q * (rev_gt + rev_eq * date_le - Expression::Constant(F::ONE))]
        });

        TestCircuitConfig {
            q_enable,
            q_accu,

            customer,
            orders,
            lineitem,

            check,
            condition,
            cflag,

            o_join,
            o_disjoin,
            c_join,
            c_disjoin,
            l_join,
            l_disjoin,

            lt_compare_condition,
            equal_condition,

            instance,
            instance_test,

            cp_agg_c,
            cp_agg_l,
            cp_join_c,
            cp_join_l,
            cp_root,
            q_cp_mu,
            q_cln_flag,
            q_res_flag,

            o_filt_pad,
            o_part_pad,
            perm_orders,

            c_filt_pad,
            c_part_pad,
            perm_customer,

            l_filt_pad,
            l_part_pad,
            perm_lineitem,

            q_in_o_cust_in_c,
            q_in_c_cust_in_o,
            q_in_l_okey_in_o,
            q_in_o_okey_in_l,



            q_line,
            q_first,
            q_res_lookup,
            q_sort_res,
            l_sorted,
            line_rev,
            run_sum,
            res_pad,
            res_sorted,
            perm_lsort,
            perm_res,
            iz_same_next,
            iz_same_prev,
            q_tbl_o_join,
            lt_rev_next_cur,
            lt_date_cur_next,
            iz_rev_eq,
            iz_date_eq,
        }
    }

    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        customer: Vec<Vec<u64>>,
        orders: Vec<Vec<u64>>,
        lineitem: Vec<Vec<u64>>,
        condition: [u64; 2],
    ) -> Result<AssignedCell<F, F>, Error> {
        // chips
        let equal_chip = IsZeroChip::construct(self.config.equal_condition[0].clone());

        let lt_o_chip = LtChip::construct(self.config.lt_compare_condition[0].clone());
        lt_o_chip.load(layouter)?;

        let lt_l_chip = LtChip::construct(self.config.lt_compare_condition[1].clone());
        lt_l_chip.load(layouter)?;

        // Every Lt chip of the Cardinality Preservation Check shares one u8
        // fixed column, so a single load covers the whole check.
        LtChip::<F, NUM_BYTES>::construct(self.config.cp_agg_c.lt_key_cur_next).load(layouter)?;

        let iz_same_prev_chip = IsZeroChip::construct(self.config.iz_same_prev.clone());
        let iz_same_next_chip = IsZeroChip::construct(self.config.iz_same_next.clone());
        let iz_rev_eq_chip = IsZeroChip::construct(self.config.iz_rev_eq.clone());
        let iz_date_eq_chip = IsZeroChip::construct(self.config.iz_date_eq.clone());

        let lt_rev_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_rev_next_cur.clone());
        lt_rev_chip.load(layouter)?;
        let lt_date_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_date_cur_next.clone());
        lt_date_chip.load(layouter)?;

        let _start = Instant::now();

        // predicate flags
        let mut c_check = vec![];
        for i in 0..customer.len() {
            c_check.push(if customer[i][0] == condition[0] {
                1u64
            } else {
                0u64
            });
        }
        let mut o_check = vec![];
        for i in 0..orders.len() {
            o_check.push(orders[i][0] < condition[1]);
        }
        let mut l_check = vec![];
        for i in 0..lineitem.len() {
            l_check.push(lineitem[i][3] > condition[1]);
        }

        // filtered tables
        let c_combined: Vec<Vec<u64>> = customer
            .iter()
            .cloned()
            .filter(|r| r[0] == condition[0])
            .collect();
        let o_combined: Vec<Vec<u64>> = orders
            .iter()
            .cloned()
            .filter(|r| r[0] < condition[1])
            .collect();
        let l_combined: Vec<Vec<u64>> = lineitem
            .iter()
            .cloned()
            .filter(|r| r[3] > condition[1])
            .collect();

        // sets from filtered tables
        let c_keys: HashSet<u64> = c_combined.iter().map(|c| c[1]).collect();
        let l_orderkeys: HashSet<u64> = l_combined.iter().map(|l| l[0]).collect();

        // contributing vs noncontributing orders (on filtered orders)
        let mut contributing_orders = vec![];
        let mut noncontributing_orders = vec![];
        for o in o_combined.iter() {
            let ok = c_keys.contains(&o[2]) && l_orderkeys.contains(&o[3]);
            if ok {
                contributing_orders.push(o.clone());
            } else {
                noncontributing_orders.push(o.clone());
            }
        }

        let contributing_o_custkeys: HashSet<u64> =
            contributing_orders.iter().map(|o| o[2]).collect();
        let contributing_o_orderkeys: HashSet<u64> =
            contributing_orders.iter().map(|o| o[3]).collect();

        // contributing vs noncontributing customers (on filtered customers)
        let mut contributing_customers = vec![];
        let mut noncontributing_customers = vec![];
        for c in c_combined.iter() {
            if contributing_o_custkeys.contains(&c[1]) {
                contributing_customers.push(c.clone());
            } else {
                noncontributing_customers.push(c.clone());
            }
        }

        // contributing vs noncontributing lineitems (on filtered lineitems)
        let mut contributing_lineitems = vec![];
        let mut noncontributing_lineitems = vec![];
        for l in l_combined.iter() {
            if contributing_o_orderkeys.contains(&l[0]) {
                contributing_lineitems.push(l.clone());
            } else {
                noncontributing_lineitems.push(l.clone());
            }
        }

        // test hook only: hide one joinable order and re-reduce around it
        let tamper = HIDE_ONE_CLEAN_TUPLE.load(Ordering::Relaxed);
        if tamper && !contributing_orders.is_empty() {
            let hidden = contributing_orders.remove(0);
            noncontributing_orders.push(hidden);

            let ck: HashSet<u64> = contributing_orders.iter().map(|o| o[2]).collect();
            let ok: HashSet<u64> = contributing_orders.iter().map(|o| o[3]).collect();

            let (c_keep2, c_drop2): (Vec<_>, Vec<_>) = contributing_customers
                .drain(..)
                .partition(|c| ck.contains(&c[1]));
            contributing_customers = c_keep2;
            noncontributing_customers.extend(c_drop2);

            let (l_keep2, l_drop2): (Vec<_>, Vec<_>) = contributing_lineitems
                .drain(..)
                .partition(|l| ok.contains(&l[0]));
            contributing_lineitems = l_keep2;
            noncontributing_lineitems.extend(l_drop2);
        }

        // test hook only: declare everything clean, i.e. no reduction at all
        let all_clean = MARK_ALL_CLEAN.load(Ordering::Relaxed);
        if all_clean {
            contributing_orders = o_combined.clone();
            contributing_customers = c_combined.clone();
            contributing_lineitems = l_combined.clone();
            noncontributing_orders.clear();
            noncontributing_customers.clear();
            noncontributing_lineitems.clear();
        }

        let contributing_o_custkeys: HashSet<u64> =
            contributing_orders.iter().map(|o| o[2]).collect();
        let contributing_o_orderkeys: HashSet<u64> =
            contributing_orders.iter().map(|o| o[3]).collect();

        let join_value = vec![
            contributing_orders.clone(),    // orders
            contributing_customers.clone(), // customer
            contributing_lineitems.clone(), // lineitem
        ];

        let disjoin_value = vec![
            noncontributing_orders.clone(),
            noncontributing_customers.clone(),
            noncontributing_lineitems.clone(),
        ];

        // ---------------- permutation padding helpers ----------------
        fn pad_filter_u64(rows: &[Vec<u64>], keep: &[bool], pad: &[u64]) -> Vec<Vec<u64>> {
            rows.iter()
                .zip(keep.iter())
                .map(|(r, &k)| if k { r.clone() } else { pad.to_vec() })
                .collect()
        }
        fn pad_partition_u64(
            join: &[Vec<u64>],
            dis: &[Vec<u64>],
            total: usize,
            pad: &[u64],
        ) -> Vec<Vec<u64>> {
            let mut out: Vec<Vec<u64>> = Vec::with_capacity(total);
            out.extend_from_slice(join);
            out.extend_from_slice(dis);
            while out.len() < total {
                out.push(pad.to_vec());
            }
            out
        }
        fn to_field_rows<FF: Field + Ord>(u: &[Vec<u64>]) -> Vec<Vec<FF>> {
            u.iter()
                .map(|r| r.iter().map(|&x| FF::from(x)).collect())
                .collect()
        }

        // PAD rows: one column wider than in q3_obj.rs, the clean indicator,
        // which pads with 0 so a row dropped by the predicate is never clean
        let pad_o: [u64; 5] = [MAX_SENTINEL, 0, MAX_SENTINEL, MAX_SENTINEL, 0];
        let pad_c: [u64; 3] = [MAX_SENTINEL, MAX_SENTINEL, 0];
        let pad_l: [u64; 5] = [MAX_SENTINEL, MAX_SENTINEL, MAX_SENTINEL, MAX_SENTINEL, 0];

        let c_keep: Vec<bool> = c_check.iter().map(|&x| x == 1).collect();
        let o_keep: Vec<bool> = o_check.clone();
        let l_keep: Vec<bool> = l_check.clone();

        // ---------------- clean indicator per base row ----------------
        // A base row is clean iff it passes its predicate and its tuple went to
        // the clean side above. The tests are the same ones that built
        // join_value, so the indicator marks exactly the occurrences of R^c.
        let hidden_orders: HashSet<(u64, u64)> = noncontributing_orders
            .iter()
            .map(|o| (o[2], o[3]))
            .collect();
        let cln_o: Vec<u64> = orders
            .iter()
            .zip(o_keep.iter())
            .map(|(o, &k)| {
                (k && (all_clean
                    || (c_keys.contains(&o[2])
                        && l_orderkeys.contains(&o[3])
                        && !hidden_orders.contains(&(o[2], o[3]))))) as u64
            })
            .collect();
        let cln_c: Vec<u64> = customer
            .iter()
            .zip(c_keep.iter())
            .map(|(c, &k)| (k && (all_clean || contributing_o_custkeys.contains(&c[1]))) as u64)
            .collect();
        let cln_l: Vec<u64> = lineitem
            .iter()
            .zip(l_keep.iter())
            .map(|(l, &k)| (k && (all_clean || contributing_o_orderkeys.contains(&l[0]))) as u64)
            .collect();

        // the indicator rides along as the last column of each base relation,
        // so the existing filt_pad link gates bind it
        let ext = |rows: &Vec<Vec<u64>>, flag: &Vec<u64>| -> Vec<Vec<u64>> {
            rows.iter()
                .zip(flag.iter())
                .map(|(r, &f)| {
                    let mut v = r.clone();
                    v.push(f);
                    v
                })
                .collect()
        };
        let orders_ext = ext(&orders, &cln_o);
        let customer_ext = ext(&customer, &cln_c);
        let lineitem_ext = ext(&lineitem, &cln_l);

        // the partition side carries a constant 1 on the clean rows and 0 on
        // the residual rows, pinned by q_cln_flag / q_res_flag
        let with_flag = |rows: &Vec<Vec<u64>>, f: u64| -> Vec<Vec<u64>> {
            rows.iter()
                .map(|r| {
                    let mut v = r.clone();
                    v.push(f);
                    v
                })
                .collect()
        };
        let join_ext: Vec<Vec<Vec<u64>>> = join_value.iter().map(|t| with_flag(t, 1)).collect();
        let dis_ext: Vec<Vec<Vec<u64>>> = disjoin_value.iter().map(|t| with_flag(t, 0)).collect();

        // filt_pad values (these are ALSO constrained by the link gates in configure)
        let o_filt_pad_f: Vec<Vec<F>> =
            to_field_rows::<F>(&pad_filter_u64(&orders_ext, &o_keep, &pad_o));
        let c_filt_pad_f: Vec<Vec<F>> =
            to_field_rows::<F>(&pad_filter_u64(&customer_ext, &c_keep, &pad_c));
        let l_filt_pad_f: Vec<Vec<F>> =
            to_field_rows::<F>(&pad_filter_u64(&lineitem_ext, &l_keep, &pad_l));

        // part_pad values (we will additionally constrain_equal them to o_join/o_disjoin etc)
        let o_part_pad_f: Vec<Vec<F>> = to_field_rows::<F>(&pad_partition_u64(
            &join_ext[0],
            &dis_ext[0],
            orders.len(),
            &pad_o,
        ));
        let c_part_pad_f: Vec<Vec<F>> = to_field_rows::<F>(&pad_partition_u64(
            &join_ext[1],
            &dis_ext[1],
            customer.len(),
            &pad_c,
        ));
        let l_part_pad_f: Vec<Vec<F>> = to_field_rows::<F>(&pad_partition_u64(
            &join_ext[2],
            &dis_ext[2],
            lineitem.len(),
            &pad_l,
        ));

        // compute witnesses from o_join and l_join (NO join materialization)
        let o_rows = &join_value[0]; // [odate, shippri, custkey, okey]
        let l_rows = &join_value[2]; // [okey, ext, disc, shipdate]
        let m = o_rows.len();
        let n = l_rows.len();

        // map orderkey -> (orderdate, shippriority)
        let mut o_map: HashMap<u64, (u64, u64)> = HashMap::new();
        for r in o_rows.iter() {
            o_map.insert(r[3], (r[0], r[1]));
        }

        // l_sorted = sort l_rows by l_orderkey
        let mut l_sorted_u64 = l_rows.clone();
        l_sorted_u64.sort_by_key(|r| r[0]);

        // compute line_rev/run_sum and emit res_pad rows
        let mut line_rev_u64: Vec<u64> = vec![0; n];
        let mut run_sum_u64: Vec<u64> = vec![0; n];
        let mut res_pad_u64: Vec<[u64; 4]> = vec![[PAD_OK, PAD_DATE, PAD_SHIP, PAD_REV]; n];

        let mut acc: u128 = 0;
        let mut prev_ok: Option<u64> = None;

        for i in 0..n {
            let ok = l_sorted_u64[i][0];
            let ext = l_sorted_u64[i][1] as u128;
            let disc = l_sorted_u64[i][2] as u128;
            let lr = ext * ((SCALE as u128) - disc); // (scaled) revenue contribution
            line_rev_u64[i] = lr as u64;

            if prev_ok == Some(ok) {
                acc += lr;
            } else {
                acc = lr;
            }
            run_sum_u64[i] = acc as u64;

            let next_ok = if i + 1 < n { l_sorted_u64[i + 1][0] } else { 0 };
            let is_last = next_ok != ok;

            if is_last {
                let (od, sp) = o_map.get(&ok).copied().unwrap_or((0, 0));
                res_pad_u64[i] = [ok, od, sp, run_sum_u64[i]];
            }
            prev_ok = Some(ok);
        }

        // build res_sorted witness: sort group rows by (rev desc, odate asc), pad to length n
        let mut groups: Vec<[u64; 4]> = res_pad_u64
            .iter()
            .copied()
            .filter(|r| r[0] != PAD_OK)
            .collect();

        groups.sort_by(|a, b| b[3].cmp(&a[3]).then(a[1].cmp(&b[1])));

        let mut res_sorted_u64: Vec<[u64; 4]> = Vec::with_capacity(n);
        res_sorted_u64.extend(groups.into_iter());
        while res_sorted_u64.len() < n {
            res_sorted_u64.push([PAD_OK, PAD_DATE, PAD_SHIP, PAD_REV]);
        }

        layouter.assign_region(
            || "witness",
            |mut region| {
                // ---------------- base tables ----------------
                for i in 0..customer.len() {
                    self.config.q_enable[0].enable(&mut region, i)?;
                    for j in 0..2 {
                        region.assign_advice(
                            || "customer",
                            self.config.customer[j],
                            i,
                            || Value::known(F::from(customer[i][j])),
                        )?;
                    }
                    region.assign_advice(
                        || "check0",
                        self.config.check[0],
                        i,
                        || Value::known(F::from(c_check[i])),
                    )?;
                    region.assign_advice(
                        || "cond0",
                        self.config.condition[0],
                        i,
                        || Value::known(F::from(condition[0])),
                    )?;
                }

                for i in 0..orders.len() {
                    self.config.q_enable[1].enable(&mut region, i)?;
                    for j in 0..4 {
                        region.assign_advice(
                            || "orders",
                            self.config.orders[j],
                            i,
                            || Value::known(F::from(orders[i][j])),
                        )?;
                    }
                    region.assign_advice(
                        || "check1",
                        self.config.check[1],
                        i,
                        || Value::known(F::from(o_check[i] as u64)),
                    )?;
                    region.assign_advice(
                        || "cond1",
                        self.config.condition[1],
                        i,
                        || Value::known(F::from(condition[1])),
                    )?;
                }

                for i in 0..lineitem.len() {
                    self.config.q_enable[2].enable(&mut region, i)?;
                    region.assign_advice(
                        || "cond2",
                        self.config.condition[2],
                        i,
                        || Value::known(F::from(condition[1])),
                    )?;
                    for j in 0..4 {
                        region.assign_advice(
                            || "lineitem",
                            self.config.lineitem[j],
                            i,
                            || Value::known(F::from(lineitem[i][j])),
                        )?;
                    }
                    region.assign_advice(
                        || "check2",
                        self.config.check[2],
                        i,
                        || Value::known(F::from(l_check[i] as u64)),
                    )?;
                }

                // ---------------- join/disjoin witnesses (capture cells) ----------------
                let o_join_cells = Self::assign_table_u64(
                    &mut region,
                    "o_join",
                    &self.config.o_join,
                    &join_value[0],
                )?;
                let o_dis_cells = Self::assign_table_u64(
                    &mut region,
                    "o_disjoin",
                    &self.config.o_disjoin,
                    &disjoin_value[0],
                )?;

                let c_join_cells = Self::assign_table_u64(
                    &mut region,
                    "c_join",
                    &self.config.c_join,
                    &join_value[1],
                )?;
                let c_dis_cells = Self::assign_table_u64(
                    &mut region,
                    "c_disjoin",
                    &self.config.c_disjoin,
                    &disjoin_value[1],
                )?;

                let l_join_cells = Self::assign_table_u64(
                    &mut region,
                    "l_join",
                    &self.config.l_join,
                    &join_value[2],
                )?;
                let l_dis_cells = Self::assign_table_u64(
                    &mut region,
                    "l_disjoin",
                    &self.config.l_disjoin,
                    &disjoin_value[2],
                )?;

                // ---------------- predicate subchips ----------------
                for i in 0..customer.len() {
                    equal_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(customer[i][0]) - F::from(condition[0])),
                    )?;
                }
                for i in 0..orders.len() {
                    lt_o_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(orders[i][0])),
                        Value::known(F::from(condition[1])),
                    )?;
                }
                for i in 0..lineitem.len() {
                    lt_l_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(condition[1])),
                        Value::known(F::from(lineitem[i][3])),
                    )?;
                }

                // ===================== PERMUTATION PROOFS (the fix) =====================
                // 1) Enable shuffle selectors
                // 2) Assign filt_pad columns (already linked to base tables by link-gates)
                // 3) Assign part_pad columns AND constrain_equal them to (join || disjoin)
                // => shuffle now proves: filtered_rows == join ∪ disjoin  (as multisets)

                // ---- orders perm ----
                for i in 0..orders.len() {
                    self.config.perm_orders.q_perm1.enable(&mut region, i)?;
                    self.config.perm_orders.q_perm2.enable(&mut region, i)?;
                }
                Self::assign_table_f(
                    &mut region,
                    "o_filt_pad",
                    &self.config.o_filt_pad,
                    &o_filt_pad_f,
                )?;
                Self::assign_part_pad_and_link(
                    &mut region,
                    "o_part_pad",
                    &self.config.o_part_pad,
                    &o_part_pad_f,
                    &o_join_cells,
                    &o_dis_cells,
                )?;

                // ---- customer perm ----
                for i in 0..customer.len() {
                    self.config.perm_customer.q_perm1.enable(&mut region, i)?;
                    self.config.perm_customer.q_perm2.enable(&mut region, i)?;
                }
                Self::assign_table_f(
                    &mut region,
                    "c_filt_pad",
                    &self.config.c_filt_pad,
                    &c_filt_pad_f,
                )?;
                Self::assign_part_pad_and_link(
                    &mut region,
                    "c_part_pad",
                    &self.config.c_part_pad,
                    &c_part_pad_f,
                    &c_join_cells,
                    &c_dis_cells,
                )?;

                // ---- lineitem perm ----
                for i in 0..lineitem.len() {
                    self.config.perm_lineitem.q_perm1.enable(&mut region, i)?;
                    self.config.perm_lineitem.q_perm2.enable(&mut region, i)?;
                }
                Self::assign_table_f(
                    &mut region,
                    "l_filt_pad",
                    &self.config.l_filt_pad,
                    &l_filt_pad_f,
                )?;
                Self::assign_part_pad_and_link(
                    &mut region,
                    "l_part_pad",
                    &self.config.l_part_pad,
                    &l_part_pad_f,
                    &l_join_cells,
                    &l_dis_cells,
                )?;

                // ---- clean indicator on both sides of every Conservation Check ----
                // input side: the raw indicator on the base rows, which the
                // link gates turn into keep * indicator on the pad columns
                for (col, flags) in [
                    (self.config.cflag[0], &cln_c),
                    (self.config.cflag[1], &cln_o),
                    (self.config.cflag[2], &cln_l),
                ] {
                    for (i, &f) in flags.iter().enumerate() {
                        region.assign_advice(|| "cflag", col, i, || Value::known(F::from(f)))?;
                    }
                }
                // partition side: 1 on the clean rows, 0 on the residual rows
                for (idx, (n_cln, n_res)) in [
                    (join_value[1].len(), disjoin_value[1].len()),
                    (join_value[0].len(), disjoin_value[0].len()),
                    (join_value[2].len(), disjoin_value[2].len()),
                ]
                .iter()
                .enumerate()
                {
                    for i in 0..*n_cln {
                        self.config.q_cln_flag[idx].enable(&mut region, i)?;
                    }
                    for i in *n_cln..(*n_cln + *n_res) {
                        self.config.q_res_flag[idx].enable(&mut region, i)?;
                    }
                }

                // ===================== CARDINALITY PRESERVATION CHECK =====================
                // condition (4) of the One-Pass OBJ: the two multiplicity
                // channels are propagated to the root and their sums compared.
                //
                // Both children are leaves, so a child row's input-channel
                // multiplicity is its predicate bit and its clean-channel
                // multiplicity is that bit times the clean indicator.
                let cp_rows_c: Vec<[u64; 3]> = (0..customer.len())
                    .map(|i| [customer[i][1], c_check[i], cln_c[i]])
                    .collect();
                let cp_rows_l: Vec<[u64; 3]> = (0..lineitem.len())
                    .map(|i| [lineitem[i][0], l_check[i] as u64, cln_l[i]])
                    .collect();

                let cp_stage_c = build_cp_stage(&cp_rows_c, MAX_SENTINEL);
                let cp_stage_l = build_cp_stage(&cp_rows_l, MAX_SENTINEL);

                assign_cp_agg(&mut region, &self.config.cp_agg_c, &cp_rows_c, &cp_stage_c)?;
                assign_cp_agg(&mut region, &self.config.cp_agg_l, &cp_rows_l, &cp_stage_l)?;

                // parent side, on the rows of orders
                let o_custkeys: Vec<u64> = orders.iter().map(|o| o[2]).collect();
                let o_orderkeys: Vec<u64> = orders.iter().map(|o| o[3]).collect();
                let fetched_c = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_c,
                    &o_custkeys,
                    &cp_stage_c,
                    MAX_SENTINEL,
                )?;
                let fetched_l = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_l,
                    &o_orderkeys,
                    &cp_stage_l,
                    MAX_SENTINEL,
                )?;

                // root multiplicities and the equality between the two sums
                let cp_mu: Vec<(u64, u64)> = (0..orders.len())
                    .map(|i| {
                        let pred = o_check[i] as u64;
                        let cln = cln_o[i];
                        (
                            pred * fetched_c[i].0 * fetched_l[i].0,
                            cln * fetched_c[i].1 * fetched_l[i].1,
                        )
                    })
                    .collect();
                for i in 0..orders.len() {
                    self.config.q_cp_mu.enable(&mut region, i)?;
                }
                let (cp_all, cp_cln) =
                    assign_cp_root(&mut region, &self.config.cp_root, &cp_mu)?;
                if !tamper && !all_clean {
                    debug_assert_eq!(
                        cp_all, cp_cln,
                        "cardinality preservation: |R^c join| != |R join|"
                    );
                }

                // inputs: only enable on real rows of each join table
                for i in 0..join_value[0].len() {
                    self.config.q_in_o_cust_in_c.enable(&mut region, i)?; // o_join.cust -> c_join
                    self.config.q_in_o_okey_in_l.enable(&mut region, i)?; // o_join.okey -> l_join
                }

                for i in 0..join_value[1].len() {
                    self.config.q_in_c_cust_in_o.enable(&mut region, i)?; // c_join.cust -> o_join
                }

                for i in 0..join_value[2].len() {
                    self.config.q_in_l_okey_in_o.enable(&mut region, i)?; // l_join.okey -> o_join
                }

                // assign l_sorted (rows 0..n) + sentinel row n (needed for same_next on last row)
                for i in 0..n {
                    for j in 0..4 {
                        region.assign_advice(
                            || "l_sorted",
                            self.config.l_sorted[j],
                            i,
                            || Value::known(F::from(l_sorted_u64[i][j])),
                        )?;
                    }
                }
                for j in 0..4 {
                    region.assign_advice(
                        || "l_sorted_sentinel",
                        self.config.l_sorted[j],
                        n,
                        || Value::known(F::from(0u64)),
                    )?;
                }

                // assign line_rev, run_sum and res_pad/res_sorted
                for i in 0..n {
                    region.assign_advice(
                        || "line_rev",
                        self.config.line_rev,
                        i,
                        || Value::known(F::from(line_rev_u64[i])),
                    )?;
                    region.assign_advice(
                        || "run_sum",
                        self.config.run_sum,
                        i,
                        || Value::known(F::from(run_sum_u64[i])),
                    )?;

                    let rp = res_pad_u64[i];
                    for j in 0..4 {
                        region.assign_advice(
                            || "res_pad",
                            self.config.res_pad[j],
                            i,
                            || Value::known(F::from(rp[j])),
                        )?;
                    }

                    let rs = res_sorted_u64[i];
                    for j in 0..4 {
                        region.assign_advice(
                            || "res_sorted",
                            self.config.res_sorted[j],
                            i,
                            || Value::known(F::from(rs[j])),
                        )?;
                    }
                }

                // enable permutation selectors: l_join <-> l_sorted
                for i in 0..n {
                    self.config.perm_lsort.q_perm1.enable(&mut region, i)?;
                    self.config.perm_lsort.q_perm2.enable(&mut region, i)?;
                }

                // enable line/accu selectors
                if n > 0 {
                    self.config.q_line.enable(&mut region, 0)?;
                    self.config.q_first.enable(&mut region, 0)?;
                    self.config.q_res_lookup.enable(&mut region, 0)?;
                }
                for i in 0..n {
                    self.config.q_line.enable(&mut region, i)?;
                    self.config.q_res_lookup.enable(&mut region, i)?;
                }
                for i in 1..n {
                    self.config.q_accu.enable(&mut region, i)?;
                }

                // enable o_join table gate for lookup
                for i in 0..m {
                    self.config.q_tbl_o_join.enable(&mut region, i)?;
                }

                // enable permutation selectors: res_pad <-> res_sorted
                for i in 0..n {
                    self.config.perm_res.q_perm1.enable(&mut region, i)?;
                    self.config.perm_res.q_perm2.enable(&mut region, i)?;
                }

                // enable sort gate on rows 0..n-2
                for i in 0..n.saturating_sub(1) {
                    self.config.q_sort_res.enable(&mut region, i)?;
                }

                // same_prev only for i>=1
                for i in 1..n {
                    let diff = F::from(l_sorted_u64[i][0]) - F::from(l_sorted_u64[i - 1][0]);
                    iz_same_prev_chip.assign(&mut region, i, Value::known(diff))?;
                }
                // same_next for i=0..n-1 (needs sentinel row n assigned)
                for i in 0..n {
                    let next_ok = if i + 1 < n {
                        l_sorted_u64[i + 1][0]
                    } else {
                        0u64
                    };
                    let diff = F::from(next_ok) - F::from(l_sorted_u64[i][0]);
                    iz_same_next_chip.assign(&mut region, i, Value::known(diff))?;
                }

                // ORDER BY helpers on res_sorted: rows 0..n-2
                for i in 0..n.saturating_sub(1) {
                    let rev_cur = res_sorted_u64[i][3];
                    let rev_next = res_sorted_u64[i + 1][3];
                    let date_cur = res_sorted_u64[i][1];
                    let date_next = res_sorted_u64[i + 1][1];

                    iz_rev_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(rev_cur) - F::from(rev_next)),
                    )?;
                    iz_date_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(date_cur) - F::from(date_next)),
                    )?;

                    lt_rev_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(rev_next)),
                        Value::known(F::from(rev_cur)),
                    )?;
                    lt_date_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(date_cur)),
                        Value::known(F::from(date_next)),
                    )?;
                }

                // public output
                let out = region.assign_advice(
                    || "instance_test",
                    self.config.instance_test,
                    0,
                    || Value::known(F::from(1)),
                )?;
                Ok(out)
            },
        )
    }

    pub fn expose_public(
        &self,
        layouter: &mut impl Layouter<F>,
        cell: AssignedCell<F, F>,
        row: usize,
    ) -> Result<(), Error> {
        layouter.constrain_instance(cell.cell(), self.config.instance, row)
    }
}

// ---------------- Circuit wrapper ----------------
// Visibility only (revision): `pub` so the additive bench harness in
// `crate::bench_queries` can build this circuit outside the test module.
// No field, gate, or synthesis logic is changed.
pub struct MyCircuit<F> {
    pub customer: Vec<Vec<u64>>,
    pub orders: Vec<Vec<u64>>,
    pub lineitem: Vec<Vec<u64>>,
    pub condition: [u64; 2],
    pub _marker: PhantomData<F>,
}

impl<F: Copy + Default> Default for MyCircuit<F> {
    fn default() -> Self {
        Self {
            customer: Vec::new(),
            orders: Vec::new(),
            lineitem: Vec::new(),
            condition: [Default::default(); 2],
            _marker: PhantomData,
        }
    }
}

impl<F: Field + Ord> Circuit<F> for MyCircuit<F> {
    type Config = TestCircuitConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        TestChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = TestChip::construct(config);

        let out_cell = chip.assign(
            &mut layouter,
            self.customer.clone(),
            self.orders.clone(),
            self.lineitem.clone(),
            self.condition,
        )?;

        chip.expose_public(&mut layouter, out_cell, 0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::MyCircuit;
    use crate::data::data_processing;
    use chrono::{DateTime, NaiveDate, Utc};
    use halo2_proofs::dev::{MockProver, VerifyFailure};
    use std::marker::PhantomData;
    use std::sync::atomic::Ordering;

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
        let params_path = &crate::paths::param_file(16);
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

    #[test]
    #[ignore = "inherited heavy end-to-end test; the fast check is test_cardinality_preservation"]
    fn test_2() {
        use crate::data::data_processing;
        use std::collections::HashMap;

        // ---------------- paths ----------------
        let customer_file_path = &crate::paths::data_file("customer.tbl");
        let orders_file_path = &crate::paths::data_file("orders.tbl");
        let lineitem_file_path = &crate::paths::data_file("lineitem.tbl");
        let supplier_file_path = &crate::paths::data_file("supplier.tbl");

        // ---------------- load records ----------------
        let customers = data_processing::customer_read_records_from_file(customer_file_path)
            .expect("failed to read customer.tbl");
        let orders = data_processing::orders_read_records_from_file(orders_file_path)
            .expect("failed to read orders.tbl");
        let lineitems = data_processing::lineitem_read_records_from_file(lineitem_file_path)
            .expect("failed to read lineitem.tbl");
        let suppliers = data_processing::supplier_read_records_from_file(supplier_file_path)
            .expect("failed to read supplier.tbl");

        // ---------------- helpers ----------------
        fn max_freq(counts: &HashMap<u64, u64>) -> (u64, u64) {
            counts
                .iter()
                .max_by_key(|(_, &c)| c)
                .map(|(&k, &c)| (k, c))
                .unwrap_or((0u64, 0u64))
        }

        // ---------------- suppkey frequencies ----------------
        let mut supp_in_lineitem: HashMap<u64, u64> = HashMap::new();
        for r in lineitems.iter() {
            *supp_in_lineitem.entry(r.l_suppkey).or_default() += 1;
        }
        let (max_supp_li, max_cnt_li) = max_freq(&supp_in_lineitem);

        let mut supp_in_supplier: HashMap<u64, u64> = HashMap::new();
        for r in suppliers.iter() {
            *supp_in_supplier.entry(r.s_suppkey).or_default() += 1;
        }
        let (max_supp_s, max_cnt_s) = max_freq(&supp_in_supplier);

        // ---------------- custkey frequencies ----------------
        let mut cust_in_customer: HashMap<u64, u64> = HashMap::new();
        for r in customers.iter() {
            *cust_in_customer.entry(r.c_custkey).or_default() += 1;
        }
        let (max_cust_c, max_cnt_c) = max_freq(&cust_in_customer);

        let mut cust_in_orders: HashMap<u64, u64> = HashMap::new();
        for r in orders.iter() {
            *cust_in_orders.entry(r.o_custkey).or_default() += 1;
        }
        let (max_cust_o, max_cnt_o) = max_freq(&cust_in_orders);

        // ---------------- print results ----------------
        println!(
            "[suppkey] lineitem: max frequency = {} (suppkey={}) over {} rows",
            max_cnt_li,
            max_supp_li,
            lineitems.len()
        );
        println!(
            "[suppkey] supplier : max frequency = {} (suppkey={}) over {} rows",
            max_cnt_s,
            max_supp_s,
            suppliers.len()
        );

        println!(
            "[custkey] customer: max frequency = {} (custkey={}) over {} rows",
            max_cnt_c,
            max_cust_c,
            customers.len()
        );
        println!(
            "[custkey] orders  : max frequency = {} (custkey={}) over {} rows",
            max_cnt_o,
            max_cust_o,
            orders.len()
        );

        #[test]
        fn test_2() {
            use crate::data::data_processing;
            use std::collections::HashMap;

            // ---------------- paths ----------------
            let customer_file_path = &crate::paths::data_file("customer.tbl");
            let orders_file_path = &crate::paths::data_file("orders.tbl");
            let lineitem_file_path = &crate::paths::data_file("lineitem.tbl");
            let supplier_file_path = &crate::paths::data_file("supplier.tbl");

            // ---------------- load records ----------------
            let customers = data_processing::customer_read_records_from_file(customer_file_path)
                .expect("failed to read customer.tbl");
            let orders = data_processing::orders_read_records_from_file(orders_file_path)
                .expect("failed to read orders.tbl");
            let lineitems = data_processing::lineitem_read_records_from_file(lineitem_file_path)
                .expect("failed to read lineitem.tbl");
            let suppliers = data_processing::supplier_read_records_from_file(supplier_file_path)
                .expect("failed to read supplier.tbl");

            // ---------------- helpers ----------------
            fn max_freq(counts: &HashMap<u64, u64>) -> (u64, u64) {
                counts
                    .iter()
                    .max_by_key(|(_, &c)| c)
                    .map(|(&k, &c)| (k, c))
                    .unwrap_or((0u64, 0u64))
            }

            // ---------------- suppkey frequencies ----------------
            let mut supp_in_lineitem: HashMap<u64, u64> = HashMap::new();
            for r in lineitems.iter() {
                *supp_in_lineitem.entry(r.l_suppkey).or_default() += 1;
            }
            let (max_supp_li, max_cnt_li) = max_freq(&supp_in_lineitem);

            let mut supp_in_supplier: HashMap<u64, u64> = HashMap::new();
            for r in suppliers.iter() {
                *supp_in_supplier.entry(r.s_suppkey).or_default() += 1;
            }
            let (max_supp_s, max_cnt_s) = max_freq(&supp_in_supplier);

            // ---------------- custkey frequencies ----------------
            let mut cust_in_customer: HashMap<u64, u64> = HashMap::new();
            for r in customers.iter() {
                *cust_in_customer.entry(r.c_custkey).or_default() += 1;
            }
            let (max_cust_c, max_cnt_c) = max_freq(&cust_in_customer);

            let mut cust_in_orders: HashMap<u64, u64> = HashMap::new();
            for r in orders.iter() {
                *cust_in_orders.entry(r.o_custkey).or_default() += 1;
            }
            let (max_cust_o, max_cnt_o) = max_freq(&cust_in_orders);

            // ---------------- print results ----------------
            println!(
                "[suppkey] lineitem: max frequency = {} (suppkey={}) over {} rows",
                max_cnt_li,
                max_supp_li,
                lineitems.len()
            );
            println!(
                "[suppkey] supplier : max frequency = {} (suppkey={}) over {} rows",
                max_cnt_s,
                max_supp_s,
                suppliers.len()
            );

            println!(
                "[custkey] customer: max frequency = {} (custkey={}) over {} rows",
                max_cnt_c,
                max_cust_c,
                customers.len()
            );
            println!(
                "[custkey] orders  : max frequency = {} (custkey={}) over {} rows",
                max_cnt_o,
                max_cust_o,
                orders.len()
            );
            // [suppkey] lineitem: max frequency = 668 (suppkey=38) over 60175 rows
            // [suppkey] supplier : max frequency = 1 (suppkey=48) over 100 rows
            // [custkey] customer: max frequency = 1 (custkey=1202) over 1500 rows
            // [custkey] orders  : max frequency = 32 (custkey=643) over 15000 rows
        }
    }

    #[test]
    #[ignore = "inherited heavy end-to-end test; the fast check is test_cardinality_preservation"]
    fn test_1() {
        let k = 16;

        fn string_to_u64(s: &str) -> u64 {
            let mut result = 0;
            for (i, c) in s.chars().enumerate() {
                result += (i as u64 + 1) * (c as u64);
            }
            result
        }
        fn scale_by_1000(x: f64) -> u64 {
            (1000.0 * x) as u64
        }
        fn date_to_timestamp(date_str: &str) -> u64 {
            match NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
                Ok(date) => {
                    let datetime: DateTime<Utc> =
                        DateTime::<Utc>::from_utc(date.and_hms(0, 0, 0), Utc);
                    datetime.timestamp() as u64
                }
                Err(_) => 0,
            }
        }

        let customer_file_path = &crate::paths::data_file("customer.tbl");
        let orders_file_path = &crate::paths::data_file("orders.tbl");
        let lineitem_file_path = &crate::paths::data_file("lineitem.tbl");

        let mut customer: Vec<Vec<u64>> = Vec::new();
        let mut orders: Vec<Vec<u64>> = Vec::new();
        let mut lineitem: Vec<Vec<u64>> = Vec::new();

        if let Ok(records) = data_processing::customer_read_records_from_file(customer_file_path) {
            customer = records
                .iter()
                .map(|record| vec![string_to_u64(&record.c_mktsegment), record.c_custkey])
                .collect();
        }
        if let Ok(records) = data_processing::orders_read_records_from_file(orders_file_path) {
            orders = records
                .iter()
                .map(|record| {
                    vec![
                        date_to_timestamp(&record.o_orderdate),
                        record.o_shippriority,
                        record.o_custkey,
                        record.o_orderkey,
                    ]
                })
                .collect();
        }
        if let Ok(records) = data_processing::lineitem_read_records_from_file(lineitem_file_path) {
            lineitem = records
                .iter()
                .map(|record| {
                    vec![
                        record.l_orderkey,
                        scale_by_1000(record.l_extendedprice),
                        scale_by_1000(record.l_discount),
                        date_to_timestamp(&record.l_shipdate),
                    ]
                })
                .collect();
        }

        let condition = [string_to_u64("HOUSEHOLD"), date_to_timestamp("1995-03-25")];

        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            condition,
            _marker: PhantomData,
        };

        let public_input = vec![Fp::from(1)];

        // let test = true;
        let test = false;

        if test {
            let prover = MockProver::run(k, &circuit, vec![public_input]).unwrap();
            prover.assert_satisfied();
        } else {
            let proof_path = &crate::paths::proof_file("proof_obj_q3_test");
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
    }

    /// The truncated dataset slice both fast tests share: small enough for
    /// MockProver and for one real proof, large enough that the semijoin
    /// reduction actually drops tuples on all three relations.
    fn small_slice() -> (Vec<Vec<u64>>, Vec<Vec<u64>>, Vec<Vec<u64>>, [u64; 2]) {

        fn string_to_u64(s: &str) -> u64 {
            let mut result = 0;
            for (i, c) in s.chars().enumerate() {
                result += (i as u64 + 1) * (c as u64);
            }
            result
        }
        fn scale_by_1000(x: f64) -> u64 {
            (1000.0 * x) as u64
        }
        fn date_to_timestamp(date_str: &str) -> u64 {
            match NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
                Ok(date) => {
                    let datetime: DateTime<Utc> =
                        DateTime::<Utc>::from_utc(date.and_hms(0, 0, 0), Utc);
                    datetime.timestamp() as u64
                }
                Err(_) => 0,
            }
        }

        // a slice small enough for MockProver but large enough that the
        // reduction actually drops tuples on every relation
        const N_CUST: usize = 300;
        const N_ORD: usize = 2000;
        const N_LINE: usize = 8000;

        let mut customer: Vec<Vec<u64>> = Vec::new();
        let mut orders: Vec<Vec<u64>> = Vec::new();
        let mut lineitem: Vec<Vec<u64>> = Vec::new();

        if let Ok(records) = data_processing::customer_read_records_from_file(
            &crate::paths::data_file("customer.tbl"),
        ) {
            customer = records
                .iter()
                .take(N_CUST)
                .map(|record| vec![string_to_u64(&record.c_mktsegment), record.c_custkey])
                .collect();
        }
        if let Ok(records) =
            data_processing::orders_read_records_from_file(&crate::paths::data_file("orders.tbl"))
        {
            orders = records
                .iter()
                .take(N_ORD)
                .map(|record| {
                    vec![
                        date_to_timestamp(&record.o_orderdate),
                        record.o_shippriority,
                        record.o_custkey,
                        record.o_orderkey,
                    ]
                })
                .collect();
        }
        if let Ok(records) = data_processing::lineitem_read_records_from_file(
            &crate::paths::data_file("lineitem.tbl"),
        ) {
            lineitem = records
                .iter()
                .take(N_LINE)
                .map(|record| {
                    vec![
                        record.l_orderkey,
                        scale_by_1000(record.l_extendedprice),
                        scale_by_1000(record.l_discount),
                        date_to_timestamp(&record.l_shipdate),
                    ]
                })
                .collect();
        }

        assert!(
            !customer.is_empty() && !orders.is_empty() && !lineitem.is_empty(),
            "dataset files not found under {}",
            crate::paths::data_file("customer.tbl")
        );

        let condition = [string_to_u64("HOUSEHOLD"), date_to_timestamp("1995-03-25")];

        (customer, orders, lineitem, condition)
    }

    /// The real prover, not MockProver, on a truncated slice. MockProver checks
    /// every constraint but tolerates cells that are never read; this closes
    /// that gap by generating and verifying an actual proof, so the converted
    /// circuit is known to be provable and not only satisfiable.
    #[test]
    #[ignore = "real IPA proof, ~3 min in a debug build; run explicitly to check provability"]
    fn test_real_proof_small() {
        let (customer, orders, lineitem, condition) = small_slice();
        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            condition,
            _marker: PhantomData,
        };
        let t = Instant::now();
        // the shipped param16 is the smallest set on disk, and the slice fits it
        generate_and_verify_proof(
            circuit,
            &[Fp::from(1)],
            &crate::paths::proof_file("proof_obj_q3_test_small"),
        );
        println!("real proof of the truncated slice took {:?}", t.elapsed());
    }

    /// Fast correctness check of the Cardinality Preservation Check: a
    /// truncated slice of the dataset under MockProver, which verifies every
    /// gate, shuffle and lookup of the circuit without paying for a real proof.
    #[test]
    fn test_cardinality_preservation() {
        let k = 15;
        let (customer, orders, lineitem, condition) = small_slice();

        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            condition,
            _marker: PhantomData,
        };

        let prover = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        prover.assert_satisfied();

        // Negative direction: the same witness with one joinable tuple hidden
        // in the residual side, and the neighbours re-reduced around it so that
        // Conservation, Non-Membership and Pairwise Consistency all still hold.
        // Only condition (4) can see this, so the circuit must now reject.
        super::HIDE_ONE_CLEAN_TUPLE.store(true, Ordering::Relaxed);
        let tampered = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        let verdict = tampered.verify();
        super::HIDE_ONE_CLEAN_TUPLE.store(false, Ordering::Relaxed);

        let failures = verdict.expect_err("condition (4) accepted a hidden joinable tuple");
        assert!(
            failures
                .iter()
                .any(|f| format!("{:?}", f).contains("cardinality preservation")),
            "the circuit rejected, but not through the Cardinality Preservation Check: {:?}",
            failures
        );

        // Third direction: no reduction at all, every tuple that passes its
        // predicate declared clean. Conservation holds and condition (4) is
        // satisfied for free, since both channels then agree row by row, so this
        // is the escape that condition (3) exists to close.
        super::MARK_ALL_CLEAN.store(true, Ordering::Relaxed);
        let unreduced = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        let verdict = unreduced.verify();
        super::MARK_ALL_CLEAN.store(false, Ordering::Relaxed);

        let failures = verdict.expect_err("condition (3) accepted an unreduced clean instance");
        assert!(
            failures.iter().any(|f| matches!(
                f,
                VerifyFailure::Lookup { name, .. } if name.starts_with("pw: ")
            )),
            "the circuit rejected, but not through a Pairwise Consistency lookup: {:?}",
            failures
        );
    }
}
