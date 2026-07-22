use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};

use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::time::Instant;

const NUM_BYTES: usize = 5;
const MAX_SENTINEL: u64 = (1u64 << (8 * NUM_BYTES)) - 1; // 2^40-1

const SCALE: u64 = 1000;

const PAD_OK: u64 = MAX_SENTINEL; // pad orderkey (max)
const PAD_DATE: u64 = MAX_SENTINEL; // pad orderdate (max -> last when ASC)
const PAD_SHIP: u64 = MAX_SENTINEL; // pad shippriority (max)
const PAD_REV: u64 = 0; // pad revenue (min -> last when DESC)

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

#[derive(Clone, Debug)]
pub struct TestCircuitConfig<F: Field + Ord> {
    q_enable: Vec<Selector>,

    q_flagged_lookup: Selector,
    q_lookup_complex: Selector,

    customer: Vec<Column<Advice>>, // 2
    orders: Vec<Column<Advice>>,   // 4
    lineitem: Vec<Column<Advice>>, // 4

    flag_o_in_lc: Vec<Column<Advice>>,   // 2
    range_low_high: Vec<Column<Advice>>, // 4
    check: Vec<Column<Advice>>,          // 0..2 used
    condition: Vec<Column<Advice>>,      // 3

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

    // gap LT configs (gated by (1-in_*))
    lt_c_gap_low: LtConfig<F, NUM_BYTES>,
    lt_c_gap_high: LtConfig<F, NUM_BYTES>,
    lt_l_gap_low: LtConfig<F, NUM_BYTES>,
    lt_l_gap_high: LtConfig<F, NUM_BYTES>,

    // key tables for membership+gap
    c_key: Column<Advice>,
    c_key_next: Column<Advice>,
    l_key: Column<Advice>,
    l_key_next: Column<Advice>,

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
    q_tbl_c_cust: Selector,
    q_tbl_o_cust: Selector,
    q_tbl_o_okey: Selector,
    q_tbl_l_okey: Selector,

    // key-table advice columns
    tbl_c_custkey: Column<Advice>, // values: [0] + uniq(c_join.c_custkey)
    tbl_o_custkey: Column<Advice>, // values: [0] + uniq(o_join.o_custkey)
    tbl_o_orderkey: Column<Advice>, // values: [0] + uniq(o_join.o_orderkey)
    tbl_l_orderkey: Column<Advice>, // values: [0] + uniq(l_join.l_orderkey)

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
                if i < join_len {
                    region.constrain_equal(part_cell.cell(), join_cells[i][j].cell())?;
                } else if i < join_len + dis_len {
                    region.constrain_equal(part_cell.cell(), dis_cells[i - join_len][j].cell())?;
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

        let q_flagged_lookup = meta.selector();
        let q_lookup_complex = meta.complex_selector();

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

        let flag_o_in_lc = vec![meta.advice_column(), meta.advice_column()];
        let range_low_high = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();

        let mut check = vec![];
        for _ in 0..4 {
            check.push(meta.advice_column());
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

        // ---------------- gap + emptiness logic (over o_disjoin keys) ----------------
        let lt_c_gap_low = LtChip::configure(
            meta,
            |m| {
                let q = m.query_selector(q_flagged_lookup);
                let in_c = m.query_advice(flag_o_in_lc[0], Rotation::cur());
                q * (Expression::Constant(F::ONE) - in_c)
            },
            |m| m.query_advice(range_low_high[0], Rotation::cur()),
            |m| m.query_advice(o_disjoin[2], Rotation::cur()),
        );
        let lt_c_gap_high = LtChip::configure(
            meta,
            |m| {
                let q = m.query_selector(q_flagged_lookup);
                let in_c = m.query_advice(flag_o_in_lc[0], Rotation::cur());
                q * (Expression::Constant(F::ONE) - in_c)
            },
            |m| m.query_advice(o_disjoin[2], Rotation::cur()),
            |m| m.query_advice(range_low_high[1], Rotation::cur()),
        );

        let lt_l_gap_low = LtChip::configure(
            meta,
            |m| {
                let q = m.query_selector(q_flagged_lookup);
                let in_l = m.query_advice(flag_o_in_lc[1], Rotation::cur());
                q * (Expression::Constant(F::ONE) - in_l)
            },
            |m| m.query_advice(range_low_high[2], Rotation::cur()),
            |m| m.query_advice(o_disjoin[3], Rotation::cur()),
        );
        let lt_l_gap_high = LtChip::configure(
            meta,
            |m| {
                let q = m.query_selector(q_flagged_lookup);
                let in_l = m.query_advice(flag_o_in_lc[1], Rotation::cur());
                q * (Expression::Constant(F::ONE) - in_l)
            },
            |m| m.query_advice(o_disjoin[3], Rotation::cur()),
            |m| m.query_advice(range_low_high[3], Rotation::cur()),
        );

        meta.create_gate("Emptiness Logic", |m| {
            let q = m.query_selector(q_flagged_lookup);
            let in_c = m.query_advice(flag_o_in_lc[0], Rotation::cur());
            let in_l = m.query_advice(flag_o_in_lc[1], Rotation::cur());
            let one = Expression::Constant(F::ONE);

            let c_low_ok = lt_c_gap_low.is_lt(m, None);
            let c_high_ok = lt_c_gap_high.is_lt(m, None);
            let l_low_ok = lt_l_gap_low.is_lt(m, None);
            let l_high_ok = lt_l_gap_high.is_lt(m, None);

            vec![
                // flags boolean
                q.clone() * in_c.clone() * (one.clone() - in_c.clone()),
                q.clone() * in_l.clone() * (one.clone() - in_l.clone()),
                // cannot be in both
                q.clone() * in_c.clone() * in_l.clone(),
                // if not in C => low < val < high
                q.clone() * (one.clone() - in_c.clone()) * (one.clone() - c_low_ok),
                q.clone() * (one.clone() - in_c.clone()) * (one.clone() - c_high_ok),
                // if not in L => low < val < high
                q.clone() * (one.clone() - in_l.clone()) * (one.clone() - l_low_ok),
                q * (one - in_l) * (Expression::Constant(F::ONE) - l_high_ok),
            ]
        });

        // ---------------- membership/gap key tables ----------------
        let c_key = meta.advice_column();
        let c_key_next = meta.advice_column();
        let l_key = meta.advice_column();
        let l_key_next = meta.advice_column();

        meta.lookup_any("Customer member (disjoin orders)", |m| {
            let q = m.query_selector(q_lookup_complex);
            let in_c = m.query_advice(flag_o_in_lc[0], Rotation::cur());
            let val = m.query_advice(o_disjoin[2], Rotation::cur());
            let key = m.query_advice(c_key, Rotation::cur());
            vec![(q * in_c * val, key)]
        });

        meta.lookup_any("Customer gap pair (disjoin orders)", |m| {
            let q = m.query_selector(q_lookup_complex);
            let in_c = m.query_advice(flag_o_in_lc[0], Rotation::cur());
            let gate = q * (Expression::Constant(F::ONE) - in_c);
            let low = m.query_advice(range_low_high[0], Rotation::cur());
            let high = m.query_advice(range_low_high[1], Rotation::cur());
            let key = m.query_advice(c_key, Rotation::cur());
            let keyn = m.query_advice(c_key_next, Rotation::cur());
            vec![(gate.clone() * low, key), (gate * high, keyn)]
        });

        meta.lookup_any("Lineitem member (disjoin orders)", |m| {
            let q = m.query_selector(q_lookup_complex);
            let in_l = m.query_advice(flag_o_in_lc[1], Rotation::cur());
            let val = m.query_advice(o_disjoin[3], Rotation::cur());
            let key = m.query_advice(l_key, Rotation::cur());
            vec![(q * in_l * val, key)]
        });

        meta.lookup_any("Lineitem gap pair (disjoin orders)", |m| {
            let q = m.query_selector(q_lookup_complex);
            let in_l = m.query_advice(flag_o_in_lc[1], Rotation::cur());
            let gate = q * (Expression::Constant(F::ONE) - in_l);
            let low = m.query_advice(range_low_high[2], Rotation::cur());
            let high = m.query_advice(range_low_high[3], Rotation::cur());
            let key = m.query_advice(l_key, Rotation::cur());
            let keyn = m.query_advice(l_key_next, Rotation::cur());
            vec![(gate.clone() * low, key), (gate * high, keyn)]
        });

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

        let (o_filt_pad, o_part_pad, perm_orders) = mk_perm::<F>(meta, 4);
        let (c_filt_pad, c_part_pad, perm_customer) = mk_perm::<F>(meta, 2);
        let (l_filt_pad, l_part_pad, perm_lineitem) = mk_perm::<F>(meta, 4);

        // -------- link gates: filt_pad must equal (keep? base : PAD) --------
        // NOTE: this is what makes the permutation check talk about *real filtered rows*
        let pad_o: [u64; 4] = [MAX_SENTINEL, 0, MAX_SENTINEL, MAX_SENTINEL];
        let pad_c: [u64; 2] = [MAX_SENTINEL, MAX_SENTINEL];
        let pad_l: [u64; 4] = [MAX_SENTINEL, MAX_SENTINEL, MAX_SENTINEL, MAX_SENTINEL];

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

        link_filt_pad(
            "link o_filt_pad = (check1? orders : PAD)",
            perm_orders.q_perm1,
            check[1],
            orders.clone(),
            o_filt_pad.clone(),
            pad_o.to_vec(),
        );
        link_filt_pad(
            "link c_filt_pad = (check0? customer : PAD)",
            perm_customer.q_perm1,
            check[0],
            customer.clone(),
            c_filt_pad.clone(),
            pad_c.to_vec(),
        );
        link_filt_pad(
            "link l_filt_pad = (check2? lineitem : PAD)",
            perm_lineitem.q_perm1,
            check[2],
            lineitem.clone(),
            l_filt_pad.clone(),
            pad_l.to_vec(),
        );

        // -------- membership lookup selectors/cols --------
        let q_in_o_cust_in_c = meta.complex_selector();
        let q_in_c_cust_in_o = meta.complex_selector();
        let q_in_l_okey_in_o = meta.complex_selector();
        let q_in_o_okey_in_l = meta.complex_selector();

        let q_tbl_c_cust = meta.complex_selector();
        let q_tbl_o_cust = meta.complex_selector();
        let q_tbl_o_okey = meta.complex_selector();
        let q_tbl_l_okey = meta.complex_selector();

        let tbl_c_custkey = meta.advice_column();
        let tbl_o_custkey = meta.advice_column();
        let tbl_o_orderkey = meta.advice_column();
        let tbl_l_orderkey = meta.advice_column();

        // o_join.o_custkey ∈ c_join.c_custkey
        meta.lookup_any("o_join.custkey in c_join", |m| {
            let q_in = m.query_selector(q_in_o_cust_in_c);
            let q_t = m.query_selector(q_tbl_c_cust);

            let lhs = q_in * m.query_advice(o_join[2], Rotation::cur()); // o_custkey
            let rhs = q_t * m.query_advice(tbl_c_custkey, Rotation::cur());
            vec![(lhs, rhs)]
        });

        // c_join.c_custkey ∈ o_join.o_custkey
        meta.lookup_any("c_join.custkey in o_join", |m| {
            let q_in = m.query_selector(q_in_c_cust_in_o);
            let q_t = m.query_selector(q_tbl_o_cust);

            let lhs = q_in * m.query_advice(c_join[1], Rotation::cur()); // c_custkey
            let rhs = q_t * m.query_advice(tbl_o_custkey, Rotation::cur());
            vec![(lhs, rhs)]
        });

        // l_join.l_orderkey ∈ o_join.o_orderkey
        meta.lookup_any("l_join.orderkey in o_join", |m| {
            let q_in = m.query_selector(q_in_l_okey_in_o);
            let q_t = m.query_selector(q_tbl_o_okey);

            let lhs = q_in * m.query_advice(l_join[0], Rotation::cur()); // l_orderkey
            let rhs = q_t * m.query_advice(tbl_o_orderkey, Rotation::cur());
            vec![(lhs, rhs)]
        });

        // o_join.o_orderkey ∈ l_join.l_orderkey
        meta.lookup_any("o_join.orderkey in l_join", |m| {
            let q_in = m.query_selector(q_in_o_okey_in_l);
            let q_t = m.query_selector(q_tbl_l_okey);

            let lhs = q_in * m.query_advice(o_join[3], Rotation::cur()); // o_orderkey
            let rhs = q_t * m.query_advice(tbl_l_orderkey, Rotation::cur());
            vec![(lhs, rhs)]
        });

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

            q_flagged_lookup,
            q_lookup_complex,

            customer,
            orders,
            lineitem,

            flag_o_in_lc,
            range_low_high,
            check,
            condition,

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

            lt_c_gap_low,
            lt_c_gap_high,
            lt_l_gap_low,
            lt_l_gap_high,

            c_key,
            c_key_next,
            l_key,
            l_key_next,

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

            q_tbl_c_cust,
            q_tbl_o_cust,
            q_tbl_o_okey,
            q_tbl_l_okey,

            tbl_c_custkey,
            tbl_o_custkey,
            tbl_o_orderkey,
            tbl_l_orderkey,

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

        let lt_c_low_chip = LtChip::construct(self.config.lt_c_gap_low.clone());
        lt_c_low_chip.load(layouter)?;
        let lt_c_high_chip = LtChip::construct(self.config.lt_c_gap_high.clone());
        lt_c_high_chip.load(layouter)?;

        let lt_l_low_chip = LtChip::construct(self.config.lt_l_gap_low.clone());
        lt_l_low_chip.load(layouter)?;
        let lt_l_high_chip = LtChip::construct(self.config.lt_l_gap_high.clone());
        lt_l_high_chip.load(layouter)?;

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

        fn uniq_keys(mut v: Vec<u64>) -> Vec<u64> {
            v.push(0); // dummy so gated-off inputs (=>0) always in-table
            v.sort();
            v.dedup();
            v
        }

        let c_join_keys_tbl: Vec<u64> = uniq_keys(join_value[1].iter().map(|r| r[1]).collect()); // c_custkey
        let o_join_cust_tbl: Vec<u64> = uniq_keys(join_value[0].iter().map(|r| r[2]).collect()); // o_custkey
        let o_join_okey_tbl: Vec<u64> = uniq_keys(join_value[0].iter().map(|r| r[3]).collect()); // o_orderkey
        let l_join_okey_tbl: Vec<u64> = uniq_keys(join_value[2].iter().map(|r| r[0]).collect()); // l_orderkey

        // key vectors for membership+gap proof (must include 0 and MAX)
        let mut valid_c_keys: Vec<u64> = c_combined.iter().map(|r| r[1]).collect();
        valid_c_keys.push(0);
        valid_c_keys.push(MAX_SENTINEL);
        valid_c_keys.sort();
        valid_c_keys.dedup();

        let mut valid_l_keys: Vec<u64> = l_combined.iter().map(|r| r[0]).collect();
        valid_l_keys.push(0);
        valid_l_keys.push(MAX_SENTINEL);
        valid_l_keys.sort();
        valid_l_keys.dedup();

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

        // PAD rows
        let pad_o: [u64; 4] = [MAX_SENTINEL, 0, MAX_SENTINEL, MAX_SENTINEL];
        let pad_c: [u64; 2] = [MAX_SENTINEL, MAX_SENTINEL];
        let pad_l: [u64; 4] = [MAX_SENTINEL, MAX_SENTINEL, MAX_SENTINEL, MAX_SENTINEL];

        let c_keep: Vec<bool> = c_check.iter().map(|&x| x == 1).collect();
        let o_keep: Vec<bool> = o_check.clone();
        let l_keep: Vec<bool> = l_check.clone();

        // filt_pad values (these are ALSO constrained by the link gates in configure)
        let o_filt_pad_f: Vec<Vec<F>> =
            to_field_rows::<F>(&pad_filter_u64(&orders, &o_keep, &pad_o));
        let c_filt_pad_f: Vec<Vec<F>> =
            to_field_rows::<F>(&pad_filter_u64(&customer, &c_keep, &pad_c));
        let l_filt_pad_f: Vec<Vec<F>> =
            to_field_rows::<F>(&pad_filter_u64(&lineitem, &l_keep, &pad_l));

        // part_pad values (we will additionally constrain_equal them to o_join/o_disjoin etc)
        let o_part_pad_f: Vec<Vec<F>> = to_field_rows::<F>(&pad_partition_u64(
            &join_value[0],
            &disjoin_value[0],
            orders.len(),
            &pad_o,
        ));
        let c_part_pad_f: Vec<Vec<F>> = to_field_rows::<F>(&pad_partition_u64(
            &join_value[1],
            &disjoin_value[1],
            customer.len(),
            &pad_c,
        ));
        let l_part_pad_f: Vec<Vec<F>> = to_field_rows::<F>(&pad_partition_u64(
            &join_value[2],
            &disjoin_value[2],
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

                // ---------------- emptiness logic over disjoin orders ----------------
                let dis_o = &disjoin_value[0];
                for i in 0..dis_o.len() {
                    self.config.q_flagged_lookup.enable(&mut region, i)?;
                    self.config.q_lookup_complex.enable(&mut region, i)?;

                    let o_custkey = dis_o[i][2];
                    let o_orderkey = dis_o[i][3];

                    let c_idx = valid_c_keys.binary_search(&o_custkey);
                    let (in_c_u64, low_c_u64, high_c_u64) = match c_idx {
                        Ok(_) => (1u64, 0u64, MAX_SENTINEL),
                        Err(idx) => (0u64, valid_c_keys[idx - 1], valid_c_keys[idx]),
                    };

                    let l_idx = valid_l_keys.binary_search(&o_orderkey);
                    let (in_l_u64, low_l_u64, high_l_u64) = match l_idx {
                        Ok(_) => (1u64, 0u64, MAX_SENTINEL),
                        Err(idx) => (0u64, valid_l_keys[idx - 1], valid_l_keys[idx]),
                    };

                    let in_c = F::from(in_c_u64);
                    let in_l = F::from(in_l_u64);
                    let low_c = F::from(low_c_u64);
                    let high_c = F::from(high_c_u64);
                    let low_l = F::from(low_l_u64);
                    let high_l = F::from(high_l_u64);

                    region.assign_advice(
                        || "flag_c",
                        self.config.flag_o_in_lc[0],
                        i,
                        || Value::known(in_c),
                    )?;
                    region.assign_advice(
                        || "flag_l",
                        self.config.flag_o_in_lc[1],
                        i,
                        || Value::known(in_l),
                    )?;
                    region.assign_advice(
                        || "low_c",
                        self.config.range_low_high[0],
                        i,
                        || Value::known(low_c),
                    )?;
                    region.assign_advice(
                        || "high_c",
                        self.config.range_low_high[1],
                        i,
                        || Value::known(high_c),
                    )?;
                    region.assign_advice(
                        || "low_l",
                        self.config.range_low_high[2],
                        i,
                        || Value::known(low_l),
                    )?;
                    region.assign_advice(
                        || "high_l",
                        self.config.range_low_high[3],
                        i,
                        || Value::known(high_l),
                    )?;

                    lt_c_low_chip.assign(
                        &mut region,
                        i,
                        Value::known(low_c),
                        Value::known(F::from(o_custkey)),
                    )?;
                    lt_c_high_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(o_custkey)),
                        Value::known(high_c),
                    )?;
                    lt_l_low_chip.assign(
                        &mut region,
                        i,
                        Value::known(low_l),
                        Value::known(F::from(o_orderkey)),
                    )?;
                    lt_l_high_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(o_orderkey)),
                        Value::known(high_l),
                    )?;
                }

                // ---------------- key tables (row0 dummy) ----------------
                region.assign_advice(
                    || "c_key0",
                    self.config.c_key,
                    0,
                    || Value::known(F::from(0)),
                )?;
                region.assign_advice(
                    || "c_keyn0",
                    self.config.c_key_next,
                    0,
                    || Value::known(F::from(0)),
                )?;
                region.assign_advice(
                    || "l_key0",
                    self.config.l_key,
                    0,
                    || Value::known(F::from(0)),
                )?;
                region.assign_advice(
                    || "l_keyn0",
                    self.config.l_key_next,
                    0,
                    || Value::known(F::from(0)),
                )?;

                for i in 0..valid_c_keys.len() {
                    let k = valid_c_keys[i];
                    let kn = if i + 1 < valid_c_keys.len() {
                        valid_c_keys[i + 1]
                    } else {
                        MAX_SENTINEL
                    };
                    region.assign_advice(
                        || "c_key",
                        self.config.c_key,
                        i + 1,
                        || Value::known(F::from(k)),
                    )?;
                    region.assign_advice(
                        || "c_key_next",
                        self.config.c_key_next,
                        i + 1,
                        || Value::known(F::from(kn)),
                    )?;
                }
                for i in 0..valid_l_keys.len() {
                    let k = valid_l_keys[i];
                    let kn = if i + 1 < valid_l_keys.len() {
                        valid_l_keys[i + 1]
                    } else {
                        MAX_SENTINEL
                    };
                    region.assign_advice(
                        || "l_key",
                        self.config.l_key,
                        i + 1,
                        || Value::known(F::from(k)),
                    )?;
                    region.assign_advice(
                        || "l_key_next",
                        self.config.l_key_next,
                        i + 1,
                        || Value::known(F::from(kn)),
                    )?;
                }

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

                let h = *[
                    customer.len(),
                    orders.len(),
                    lineitem.len(),
                    join_value[0].len(),
                    join_value[1].len(),
                    join_value[2].len(),
                ]
                .iter()
                .max()
                .unwrap();

                // helper: fill a key-table column up to h rows, gating only real table rows
                fn assign_key_table<F: Field + Ord>(
                    region: &mut Region<'_, F>,
                    col: Column<Advice>,
                    q_tbl: Selector,
                    keys: &[u64],
                    h: usize,
                    name: &'static str,
                ) -> Result<(), Error> {
                    for i in 0..h {
                        let v = if i < keys.len() { keys[i] } else { 0u64 };
                        region.assign_advice(|| name, col, i, || Value::known(F::from(v)))?;
                        if i < keys.len() {
                            q_tbl.enable(region, i)?;
                        }
                    }
                    Ok(())
                }

                assign_key_table::<F>(
                    &mut region,
                    self.config.tbl_c_custkey,
                    self.config.q_tbl_c_cust,
                    &c_join_keys_tbl,
                    h,
                    "tbl_c_custkey",
                )?;
                assign_key_table::<F>(
                    &mut region,
                    self.config.tbl_o_custkey,
                    self.config.q_tbl_o_cust,
                    &o_join_cust_tbl,
                    h,
                    "tbl_o_custkey",
                )?;
                assign_key_table::<F>(
                    &mut region,
                    self.config.tbl_o_orderkey,
                    self.config.q_tbl_o_okey,
                    &o_join_okey_tbl,
                    h,
                    "tbl_o_orderkey",
                )?;
                assign_key_table::<F>(
                    &mut region,
                    self.config.tbl_l_orderkey,
                    self.config.q_tbl_l_okey,
                    &l_join_okey_tbl,
                    h,
                    "tbl_l_orderkey",
                )?;

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
    use halo2_proofs::dev::MockProver;
    use std::marker::PhantomData;

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
            let proof_path = &crate::paths::proof_file("proof_obj_q3");
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
    }
}
