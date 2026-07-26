use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};
use std::collections::HashSet;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};
use crate::circuits::card_preserve::{
    assign_cp_agg, assign_cp_join, assign_cp_root, build_cp_stage, configure_cp_agg,
    configure_cp_join, configure_cp_root, wire_cp_edge, CpAggConfig, CpJoinConfig, CpRootConfig,
};

const NUM_BYTES: usize = 5;
const MAX_SENTINEL: u64 = (1u64 << (8 * NUM_BYTES)) - 1;

// Padding tuned for ORDER BY (totalprice DESC, orderdate ASC)
const PAD_NAME: u64 = MAX_SENTINEL;
const PAD_CUST: u64 = MAX_SENTINEL;
const PAD_OKEY: u64 = MAX_SENTINEL;
const PAD_DATE: u64 = MAX_SENTINEL; // biggest => last when ASC
const PAD_TOTAL: u64 = 0; // smallest => last when DESC
const PAD_QSUM: u64 = 0;

/// Test hook, off in every benchmark path: when set, the prover moves one
/// joinable lineitem tuple to the residual side and re-reduces the partition
/// around it, so the partition still passes Conservation and stays a valid
/// reduced instance, and only condition (4) can catch it. This is exactly the
/// cheat a residual-side-only argument misses, so the negative test in this
/// module is what shows the Cardinality Preservation Check is not vacuous.
pub static HIDE_ONE_CLEAN_TUPLE: AtomicBool = AtomicBool::new(false);

/// Test hook, off in every benchmark path: when set, the prover skips the
/// semijoin reduction entirely and declares every real tuple clean.
/// Conservation still holds and both channels of condition (4) then agree on
/// every row, so the two root sums match for free. This is exactly the escape
/// that Pairwise Consistency has to close, and the third direction of the test
/// in this module is what shows it does.
pub static MARK_ALL_CLEAN: AtomicBool = AtomicBool::new(false);

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

#[derive(Clone, Debug)]
pub struct Q18Config<F: Field + Ord> {
    // base tables
    customer: Vec<Column<Advice>>, // [name, custkey]
    orders: Vec<Column<Advice>>,   // [okey, custkey, date, total]
    lineitem: Vec<Column<Advice>>, // [okey, qty]

    // condition (:1). The parameter is prover advice, so it is pinned twice:
    // `q_thresh_same` makes it one value for the whole proof instead of one per
    // row, and `q_thresh_dec` plus the u8 lookup on `thresh_byte` make that
    // value a 5-byte integer, which is what the Lt chip below needs to decide
    // the comparison. Binding it to the *public* :1 needs an extra instance
    // cell, which is a change to the callers, not to this file.
    cond_thresh: Column<Advice>,
    q_thresh_same: Selector,
    thresh_byte: Column<Advice>,
    q_thresh_byte: Selector,
    q_thresh_dec: Selector,

    // permutation: lineitem <-> l_sorted
    l_sorted: Vec<Column<Advice>>,
    perm_lsort: PermAnyConfig,

    // l_sorted really is sorted by l_orderkey
    q_lsort: Selector,
    q_lsort_last: Selector,
    lt_lkey_cur_next: LtConfig<F, NUM_BYTES>, // okey_cur < okey_next
    iz_lkey_eq: IsZeroConfig<F>,

    // grouping helpers
    q_line: Selector,
    q_first: Selector,
    q_accu: Selector,

    run_sum: Column<Advice>,
    iz_same_prev: IsZeroConfig<F>,
    iz_same_next: IsZeroConfig<F>,

    // having: threshold < group_sum (only on last row)
    lt_thresh_sum: LtConfig<F, NUM_BYTES>,
    heavy: Column<Advice>, // boolean on last rows (else can be 0)

    // result (padded, length = n_lineitem rows)
    res_pad: Vec<Column<Advice>>, // [c_name, c_cust, okey, date, total, sum_qty]
    res_sorted: Vec<Column<Advice>>, // same
    perm_res: PermAnyConfig,

    // lookups to attach attributes (only when emit=1)
    q_lookup_ord: Selector,
    q_lookup_cust: Selector,
    q_tbl_orders: Selector,
    q_tbl_customer: Selector,

    // ORDER BY proof
    q_sort: Selector,
    lt_total_next_cur: LtConfig<F, NUM_BYTES>, // total_next < total_cur
    lt_date_cur_next: LtConfig<F, NUM_BYTES>,  // date_cur < date_next
    iz_total_eq: IsZeroConfig<F>,
    iz_date_eq: IsZeroConfig<F>,

    // ---------------- clean/residual partition ----------------
    // condition (1): R_i == R_i^c U R_i^r, one Conservation Check per
    // relation, widened by the clean indicator so that it binds it
    cflag: Vec<Column<Advice>>, // per base row: [customer, orders, lineitem]
    part_c: Vec<Column<Advice>>, // [name, custkey, flag]
    part_o: Vec<Column<Advice>>, // [okey, custkey, date, total, flag]
    part_l: Vec<Column<Advice>>, // [okey, qty, flag]
    perm_cons: Vec<PermAnyConfig>, // [customer, orders, lineitem]
    q_cln_flag: Vec<Selector>,  // rows of R^c: flag == 1
    q_res_flag: Vec<Selector>,  // rows of R^r: flag == 0

    // ---------------- Pairwise Consistency (condition (3)) ----------------
    // pi_K(R_i^c) == pi_K(R_j^c) on every join tree edge, as two mutual
    // membership lookups per edge between the clean sections of the two
    // partition groups. One selector per relation, gating the clean rows: it
    // serves as the input selector of the lookups out of that relation and as
    // the table selector of the lookups into it, so the four lookups need no
    // column of their own.
    q_pw_cln: Vec<Selector>, // clean rows of [customer, orders, lineitem]

    // ---------------- Cardinality Preservation Check ----------------
    // condition (4): |R^c join| == |R join|, over the tree rooted at lineitem
    cp_agg_c: CpAggConfig<F, NUM_BYTES>, // child customer, keyed by c_custkey
    cp_agg_o: CpAggConfig<F, NUM_BYTES>, // child orders, keyed by o_orderkey
    cp_join_c: CpJoinConfig<F, NUM_BYTES>, // orders -> customer
    cp_join_o: CpJoinConfig<F, NUM_BYTES>, // lineitem -> orders
    cp_root: CpRootConfig,
    cp_ones: Column<Advice>, // input channel of the customer leaf, pinned to 1
    cp_v_all_o: Column<Advice>, // the two multiplicities of an orders tuple
    cp_v_cln_o: Column<Advice>,
    q_cp_one: Selector,  // pins cp_ones
    q_cp_mu_o: Selector, // the two product gates on orders
    q_cp_mu_l: Selector, // the two product gates at the root

    instance: Column<Instance>,
    instance_test: Column<Advice>,
}

#[derive(Clone, Debug)]
pub struct Q18Chip<F: Field + Ord> {
    pub config: Q18Config<F>,
}

impl<F: Field + Ord> Q18Chip<F> {
    pub fn construct(config: Q18Config<F>) -> Self {
        Self { config }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> Q18Config<F> {
        let instance = meta.instance_column();
        meta.enable_equality(instance);
        let instance_test = meta.advice_column();
        meta.enable_equality(instance_test);

        // base tables
        let customer = vec![meta.advice_column(), meta.advice_column()];
        let orders = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let lineitem = vec![meta.advice_column(), meta.advice_column()];

        // condition
        let cond_thresh = meta.advice_column();

        // selectors
        let q_line = meta.selector();
        let q_first = meta.selector();
        let q_accu = meta.selector();

        let q_lookup_ord = meta.complex_selector();
        let q_lookup_cust = meta.complex_selector();
        let q_tbl_orders = meta.complex_selector();
        let q_tbl_customer = meta.complex_selector();

        let q_sort = meta.selector();

        // l_sorted
        let l_sorted = vec![meta.advice_column(), meta.advice_column()];

        // PermAny: lineitem <-> l_sorted
        let q_perm_l_in = meta.complex_selector();
        let q_perm_l_out = meta.complex_selector();
        let perm_lsort = PermAnyChip::configure(
            meta,
            q_perm_l_in,
            q_perm_l_out,
            lineitem.clone(),
            l_sorted.clone(),
        );

        // One fixed column serves every Lt chip of this circuit that takes one,
        // the sortedness ladder just below and the whole Cardinality
        // Preservation Check further down, so they all share a single u8 range
        // table and one `load`.
        let cp_u8 = meta.fixed_column();

        // ---------- l_sorted really is sorted by l_orderkey ----------
        // `perm_lsort` is a shuffle: it ties l_sorted to lineitem as a multiset
        // and says nothing about the order of the rows. The boundary detectors
        // below compare a row's key with its neighbour's only, so on an
        // unsorted view a prover could lay one orderkey out as two
        // non-adjacent runs. Each run would then look like a group of its own
        // and carry its own partial sum(l_quantity): the aggregate of that
        // order comes out too small, or its result row is duplicated, or (when
        // no run passes HAVING) the order drops out of the result. Nothing
        // downstream sees it, since the two attribute lookups read the emitted
        // orderkey and the order's own attributes but never the sum, the ORDER
        // BY ladder reads only totalprice and orderdate, and the Cardinality
        // Preservation Check counts the join over the base relations and never
        // looks at this view at all. This gate is what makes "same key as my
        // neighbour" mean "same group". Row n holds a PAD_OKEY sentinel; the
        // ladder covers that pair too and the gate right below makes it strict,
        // so the last group's boundary is real.
        let q_lsort = meta.selector();
        let aux_lkey_eq = meta.advice_column();
        let iz_lkey_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_lsort),
            |m| {
                m.query_advice(l_sorted[0], Rotation::next())
                    - m.query_advice(l_sorted[0], Rotation::cur())
            },
            aux_lkey_eq,
        );
        let lt_lkey_cur_next = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            cp_u8,
            |m| m.query_selector(q_lsort),
            |m| m.query_advice(l_sorted[0], Rotation::cur()),
            |m| m.query_advice(l_sorted[0], Rotation::next()),
        );
        meta.create_gate("l_sorted key is nondecreasing", |m| {
            let q = m.query_selector(q_lsort);
            let le = lt_lkey_cur_next.is_lt(m, None) + iz_lkey_eq.expr();
            vec![q * (le - Expression::Constant(F::ONE))]
        });

        // The ladder above is nondecreasing, so on its own it would let the
        // sentinel repeat the last real key: iz_same_next would then read 0 on
        // row n-1, the last group would never see its boundary and would never
        // emit, which drops the highest orderkey from the result. The last pair
        // is therefore strict. The Lt chip's own gate already holds on that row
        // (q_lsort covers it), so this only reads its outcome.
        let q_lsort_last = meta.selector();
        meta.create_gate("l_sorted sentinel is above the last key", |m| {
            let q = m.query_selector(q_lsort_last);
            vec![q * (lt_lkey_cur_next.is_lt(m, None) - Expression::Constant(F::ONE))]
        });

        // grouping columns
        let run_sum = meta.advice_column();

        // same_prev / same_next
        let aux_same_prev = meta.advice_column();
        let aux_same_next = meta.advice_column();

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

        // run_sum gates
        // run_sum[0] = qty[0]
        meta.create_gate("run_sum_first", |m| {
            let q = m.query_selector(q_first);
            let rs = m.query_advice(run_sum, Rotation::cur());
            let qty = m.query_advice(l_sorted[1], Rotation::cur());
            vec![q * (rs - qty)]
        });
        // run_sum[i] = same_prev * run_sum[i-1] + qty[i]
        meta.create_gate("run_sum_accu", |m| {
            let q = m.query_selector(q_accu);
            let same = iz_same_prev.expr();
            let rs_cur = m.query_advice(run_sum, Rotation::cur());
            let rs_prev = m.query_advice(run_sum, Rotation::prev());
            let qty = m.query_advice(l_sorted[1], Rotation::cur());
            vec![q * (rs_cur - (same * rs_prev + qty))]
        });

        // ---------- the HAVING parameter :1 is one value for the whole proof ----------
        // `cond_thresh` is read on every row the group-by covers, as the lhs of
        // the Lt chip just below, and as bare advice it was a fresh prover
        // choice on each of those rows. That is a soundness hole in both
        // directions, independently of what the parameter is supposed to be: on
        // the last row of a group whose true sum is below the threshold the
        // prover writes a smaller value there, the Lt chip reports lt = 1, the
        // gate below then *forces* heavy = 1 and the group is emitted; on a
        // group whose sum is above it the prover writes a larger value and the
        // group is dropped. Both keep every other gate, lookup and shuffle
        // satisfied and the fixed columns, and hence the verifying key, are
        // untouched.
        //
        // Two gates close the per-row freedom.
        //
        // (a) One value per proof. A degree-2 identity between neighbouring
        // rows, enabled on every row the parameter is read on except the last,
        // makes the column constant there. The prover keeps one choice for the
        // whole proof instead of one per group, which is what turns "heavy" into
        // a single predicate rather than a per-group verdict.
        let q_thresh_same = meta.selector();
        meta.create_gate(
            "having: cond_thresh is one value for the whole proof",
            |m| {
                let q = m.query_selector(q_thresh_same);
                let cur = m.query_advice(cond_thresh, Rotation::cur());
                let next = m.query_advice(cond_thresh, Rotation::next());
                vec![q * (cur - next)]
            },
        );

        // (b) That single value is a NUM_BYTES-byte integer. Constancy alone is
        // not enough: the Lt gate is only
        // `lhs - rhs - from_bytes(diff) + lt * 2^40 == 0` with `diff` five u8
        // limbs, so it decides `lhs < rhs` only for `lhs` and `rhs` inside
        // [0, 2^40). A threshold of `p - 1` satisfies the gate with lt = 1 and
        // `diff = 2^40 - 1 - run_sum` for *every* group, so the HAVING filter
        // would pass everything while still being a single constant. The
        // decomposition below is checked once, on the row the constancy gate
        // propagates from, and its five limbs are looked up in the u8 column
        // that already serves every Lt chip here, so it costs one advice column
        // and one lookup argument.
        let thresh_byte = meta.advice_column();
        let q_thresh_byte = meta.complex_selector();
        let q_thresh_dec = meta.selector();
        meta.lookup_any("having: cond_thresh limb is a u8", |m| {
            let q = m.query_selector(q_thresh_byte);
            let limb = m.query_advice(thresh_byte, Rotation::cur());
            let u8_range = m.query_fixed(cp_u8, Rotation::cur());
            vec![(q * limb, u8_range)]
        });
        meta.create_gate("having: cond_thresh is a 5-byte integer", |m| {
            let q = m.query_selector(q_thresh_dec);
            let t = m.query_advice(cond_thresh, Rotation::cur());
            let mut acc = Expression::Constant(F::ZERO);
            for k in (0..NUM_BYTES).rev() {
                acc = acc * Expression::Constant(F::from(256u64))
                    + m.query_advice(thresh_byte, Rotation(k as i32));
            }
            vec![q * (t - acc)]
        });

        // HAVING: heavy = (thresh < run_sum) only on last rows
        let heavy = meta.advice_column();
        meta.enable_equality(heavy);

        let lt_thresh_sum = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| {
                let q = m.query_selector(q_line);
                let one = Expression::Constant(F::ONE);
                let is_last = one - iz_same_next.expr(); // only last row enables
                q * is_last
            },
            |m| m.query_advice(cond_thresh, Rotation::cur()),
            |m| m.query_advice(run_sum, Rotation::cur()),
        );

        // constrain heavy == lt_thresh_sum on last rows, and heavy boolean
        meta.create_gate("heavy flag on last rows", |m| {
            let q = m.query_selector(q_line);
            let one = Expression::Constant(F::ONE);
            let is_last = one.clone() - iz_same_next.expr();

            let h = m.query_advice(heavy, Rotation::cur());
            let lt = lt_thresh_sum.is_lt(m, None);

            vec![
                q.clone() * is_last.clone() * (h.clone() - lt),
                q * is_last * h.clone() * (one - h),
            ]
        });

        // result columns
        let res_pad = (0..6).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let res_sorted = (0..6).map(|_| meta.advice_column()).collect::<Vec<_>>();

        // permutation res_pad <-> res_sorted
        let q_perm_r_in = meta.complex_selector();
        let q_perm_r_out = meta.complex_selector();
        let perm_res = PermAnyChip::configure(
            meta,
            q_perm_r_in,
            q_perm_r_out,
            res_pad.clone(),
            res_sorted.clone(),
        );

        // emit gate: only if emit = is_last * heavy
        meta.create_gate("emit res_pad rows (pad otherwise)", |m| {
            let q = m.query_selector(q_line);
            let one = Expression::Constant(F::ONE);
            let is_last = one.clone() - iz_same_next.expr();
            let h = m.query_advice(heavy, Rotation::cur());
            let emit = is_last * h;
            let not_emit = one.clone() - emit.clone();

            let cur_okey = m.query_advice(l_sorted[0], Rotation::cur());
            let sum_qty = m.query_advice(run_sum, Rotation::cur());

            // res_pad indices:
            // 0=c_name,1=c_cust,2=o_okey,3=o_date,4=o_total,5=sum_qty
            let out_name = m.query_advice(res_pad[0], Rotation::cur());
            let out_cust = m.query_advice(res_pad[1], Rotation::cur());
            let out_okey = m.query_advice(res_pad[2], Rotation::cur());
            let out_date = m.query_advice(res_pad[3], Rotation::cur());
            let out_total = m.query_advice(res_pad[4], Rotation::cur());
            let out_sum = m.query_advice(res_pad[5], Rotation::cur());

            let pad_name = Expression::Constant(F::from(PAD_NAME));
            let pad_cust = Expression::Constant(F::from(PAD_CUST));
            let pad_okey = Expression::Constant(F::from(PAD_OKEY));
            let pad_date = Expression::Constant(F::from(PAD_DATE));
            let pad_total = Expression::Constant(F::from(PAD_TOTAL));
            let pad_qsum = Expression::Constant(F::from(PAD_QSUM));

            vec![
                // we always set okey & sum based on emit; other cols forced to PAD when not emit
                q.clone() * (out_okey - (emit.clone() * cur_okey + not_emit.clone() * pad_okey)),
                q.clone() * (out_sum - (emit.clone() * sum_qty + not_emit.clone() * pad_qsum)),
                q.clone() * not_emit.clone() * (out_name - pad_name),
                q.clone() * not_emit.clone() * (out_cust - pad_cust),
                q.clone() * not_emit.clone() * (out_date - pad_date),
                q * not_emit * (out_total - pad_total),
            ]
        });

        // ---------- Tuple lookup into orders (only when emit=1) ----------
        // (o_okey, o_cust, o_date, o_total) ∈ orders
        meta.lookup_any("attach orders tuple", |m| {
            let q_in = m.query_selector(q_lookup_ord);
            let q_tbl = m.query_selector(q_tbl_orders);

            let one = Expression::Constant(F::ONE);
            let is_last = one.clone() - iz_same_next.expr();
            let h = m.query_advice(heavy, Rotation::cur());
            let emit = is_last * h;
            let gate = q_in * emit;

            vec![
                (
                    gate.clone() * m.query_advice(res_pad[2], Rotation::cur()),
                    q_tbl.clone() * m.query_advice(orders[0], Rotation::cur()),
                ), // okey
                (
                    gate.clone() * m.query_advice(res_pad[1], Rotation::cur()),
                    q_tbl.clone() * m.query_advice(orders[1], Rotation::cur()),
                ), // cust
                (
                    gate.clone() * m.query_advice(res_pad[3], Rotation::cur()),
                    q_tbl.clone() * m.query_advice(orders[2], Rotation::cur()),
                ), // date
                (
                    gate * m.query_advice(res_pad[4], Rotation::cur()),
                    q_tbl * m.query_advice(orders[3], Rotation::cur()),
                ), // total
            ]
        });

        // ---------- Tuple lookup into customer (only when emit=1) ----------
        // (c_custkey, c_name) ∈ customer
        meta.lookup_any("attach customer tuple", |m| {
            let q_in = m.query_selector(q_lookup_cust);
            let q_tbl = m.query_selector(q_tbl_customer);

            let one = Expression::Constant(F::ONE);
            let is_last = one.clone() - iz_same_next.expr();
            let h = m.query_advice(heavy, Rotation::cur());
            let emit = is_last * h;
            let gate = q_in * emit;

            vec![
                (
                    gate.clone() * m.query_advice(res_pad[1], Rotation::cur()),
                    q_tbl.clone() * m.query_advice(customer[1], Rotation::cur()),
                ), // custkey
                (
                    gate * m.query_advice(res_pad[0], Rotation::cur()),
                    q_tbl * m.query_advice(customer[0], Rotation::cur()),
                ), // name
            ]
        });

        // ---------- ORDER BY (o_totalprice DESC, o_orderdate ASC) on res_sorted ----------
        let aux_total_eq = meta.advice_column();
        let aux_date_eq = meta.advice_column();

        let iz_total_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort),
            |m| {
                m.query_advice(res_sorted[4], Rotation::cur())
                    - m.query_advice(res_sorted[4], Rotation::next())
            },
            aux_total_eq,
        );
        let iz_date_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort),
            |m| {
                m.query_advice(res_sorted[3], Rotation::cur())
                    - m.query_advice(res_sorted[3], Rotation::next())
            },
            aux_date_eq,
        );

        let lt_total_next_cur = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort),
            |m| m.query_advice(res_sorted[4], Rotation::next()), // total_next
            |m| m.query_advice(res_sorted[4], Rotation::cur()),  // total_cur
        );
        let lt_date_cur_next = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort),
            |m| m.query_advice(res_sorted[3], Rotation::cur()), // date_cur
            |m| m.query_advice(res_sorted[3], Rotation::next()), // date_next
        );

        meta.create_gate("ORDER BY total DESC, date ASC", |m| {
            let q = m.query_selector(q_sort);

            let total_gt = lt_total_next_cur.is_lt(m, None); // next < cur
            let total_eq = iz_total_eq.expr();

            let date_lt = lt_date_cur_next.is_lt(m, None); // cur < next
            let date_eq = iz_date_eq.expr();
            let date_le = date_lt + date_eq;

            vec![q * (total_gt + total_eq * date_le - Expression::Constant(F::ONE))]
        });

        // ---------- clean/residual partition of every relation ----------
        // q18_obj.rs partitions nothing, so condition (4) has no second side to
        // compare against until the partition exists. One indicator column per
        // relation on the join tree plus one Conservation Check per relation is
        // the whole addition: the multiset equality between the base rows and
        // the [clean rows | residual rows] block is what binds the indicator to
        // the partition, and without it a prover could mark a residual row
        // clean, inflate the clean channel and make (4) vacuous. Q18 has no
        // in-relation predicate, so every base row is kept and the two blocks
        // fill the relation's n rows exactly, with no pad block.
        let cflag = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for &c in cflag.iter() {
            meta.enable_equality(c);
        }
        let part_c = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let part_o = (0..5).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let part_l = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();

        let mut c_base = customer.clone();
        c_base.push(cflag[0]);
        let mut o_base = orders.clone();
        o_base.push(cflag[1]);
        let mut l_base = lineitem.clone();
        l_base.push(cflag[2]);

        let mut perm_cons = Vec::new();
        for (base, part) in [
            (c_base, part_c.clone()),
            (o_base, part_o.clone()),
            (l_base, part_l.clone()),
        ] {
            let q_in = meta.complex_selector();
            let q_out = meta.complex_selector();
            perm_cons.push(PermAnyChip::configure(meta, q_in, q_out, base, part));
        }

        // partition side of the indicator: 1 on the R^c rows, 0 on the R^r rows
        let q_cln_flag = (0..3).map(|_| meta.selector()).collect::<Vec<_>>();
        let q_res_flag = (0..3).map(|_| meta.selector()).collect::<Vec<_>>();
        for (idx, part) in [part_c.clone(), part_o.clone(), part_l.clone()]
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
                vec![qc * (f.clone() - Expression::Constant(F::ONE)), qr * f]
            });
        }

        // ---------- Pairwise Consistency (condition (3)) ----------
        // Two mutual membership lookups per join tree edge, on the shared key,
        // each looking one clean relation's key column up directly in the
        // adjacent clean relation's key column.
        //
        // A partition group is laid out [clean rows | residual rows] and its
        // tuple columns are tied to the base relation by the Conservation Check
        // above, so the group's key column restricted to its clean prefix is
        // exactly pi_K(R^c). These lookups read those columns, never the base
        // relation: a lookup over the base rows would certify membership in R_j,
        // the weaker statement q18_obj.rs already made, and not condition (3).
        //
        // An earlier version routed each direction through an intermediate
        // advice column holding the deduplicated key set of the relation it
        // claimed to enumerate. Nothing bound that column to the relation, so
        // the prover could put pi_K(R_i^c) in the table R_i^c looks into and
        // pi_K(R_j^c) in the table R_j^c looks into and satisfy both directions
        // for an arbitrary partition, which made condition (3) vacuous. With the
        // two columns looked up in each other there is no free advice left, and
        // the two containments together are the set equality (3) asks for. It is
        // also cheaper: four advice columns and four complex selectors less.
        //
        // Without this, condition (4) has a trivial escape: the all-clean
        // partition satisfies Conservation and makes both channels of the
        // Cardinality Preservation Check agree on every row, so the two root
        // sums are equal for free. A dangling tuple left in R_i^c has no partner
        // in the clean part of its neighbour, so the key sets on that edge
        // differ and one of the four lookups below fails.
        //
        // On a row where a selector is off both sides of a lookup evaluate to 0,
        // so 0 is always in the table and the rows outside the clean prefixes
        // cost nothing. TPC-H custkeys and orderkeys are at least 1, so the
        // containment is over the real keys and needs no key shift.
        //
        // The selectors have to be fresh complex selectors, one per relation:
        // q_cln_flag is simple and a simple selector may appear on neither side
        // of a lookup.
        let q_pw_cln = (0..3).map(|_| meta.complex_selector()).collect::<Vec<_>>();

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
            "pw: o^c custkey in c^c",
            q_pw_cln[1],
            part_o[1],
            q_pw_cln[0],
            part_c[1],
        );
        pw_edge(
            "pw: c^c custkey in o^c",
            q_pw_cln[0],
            part_c[1],
            q_pw_cln[1],
            part_o[1],
        );

        // edge (lineitem, orders) on orderkey
        pw_edge(
            "pw: l^c orderkey in o^c",
            q_pw_cln[2],
            part_l[0],
            q_pw_cln[1],
            part_o[0],
        );
        pw_edge(
            "pw: o^c orderkey in l^c",
            q_pw_cln[1],
            part_o[0],
            q_pw_cln[2],
            part_l[0],
        );

        // ---------- Cardinality Preservation Check (condition (4)) ----------
        // Every Lt chip of the check reads the `cp_u8` column allocated above,
        // so the whole check costs a single u8 range table.

        // customer is a leaf and Q18 has no in-relation predicate, so its input
        // channel is the constant 1 and its clean channel is the bound
        // indicator. The constant has to be pinned: a prover free to choose it
        // could zero both channels at once and satisfy (4) with an empty R^c.
        let cp_ones = meta.advice_column();
        let q_cp_one = meta.selector();
        meta.create_gate("cp: input multiplicity of a customer tuple is 1", |m| {
            let q = m.query_selector(q_cp_one);
            vec![q * (m.query_advice(cp_ones, Rotation::cur()) - Expression::Constant(F::ONE))]
        });

        let cp_agg_c = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            customer[1], // c_custkey
            cp_ones,
            cflag[0],
            MAX_SENTINEL,
        );

        // parent side of orders -> customer, on the rows of orders
        let cp_join_c = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, orders[1]);
        wire_cp_edge(meta, &cp_join_c, &cp_agg_c, orders[1]);

        // orders is internal: its two multiplicities are the two sums fetched
        // on its own child edge, the clean one masked by its indicator
        let cp_v_all_o = meta.advice_column();
        let cp_v_cln_o = meta.advice_column();
        let q_cp_mu_o = meta.selector();
        {
            let s_all_c = cp_join_c.s_all;
            let s_cln_c = cp_join_c.s_cln;
            let cln_o = cflag[1];
            meta.create_gate("cp: multiplicities of an orders tuple", move |m| {
                let q = m.query_selector(q_cp_mu_o);
                let all = m.query_advice(cp_v_all_o, Rotation::cur())
                    - m.query_advice(s_all_c, Rotation::cur());
                let cln = m.query_advice(cp_v_cln_o, Rotation::cur())
                    - m.query_advice(cln_o, Rotation::cur())
                        * m.query_advice(s_cln_c, Rotation::cur());
                vec![q.clone() * all, q * cln]
            });
        }

        let cp_agg_o = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            orders[0], // o_orderkey
            cp_v_all_o,
            cp_v_cln_o,
            MAX_SENTINEL,
        );

        // parent side of lineitem -> orders, on the rows of lineitem
        let cp_join_o = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, lineitem[0]);
        wire_cp_edge(meta, &cp_join_o, &cp_agg_o, lineitem[0]);

        // Root multiplicities and the single equality that compares the two
        // join cardinalities.
        let cp_root = configure_cp_root::<F>(meta);
        let q_cp_mu_l = meta.selector();
        {
            let s_all_o = cp_join_o.s_all;
            let s_cln_o = cp_join_o.s_cln;
            let cln_l = cflag[2];
            let mu_all = cp_root.mu_all;
            let mu_cln = cp_root.mu_cln;
            meta.create_gate("cp: root multiplicities over lineitem", move |m| {
                let q = m.query_selector(q_cp_mu_l);
                let all = m.query_advice(mu_all, Rotation::cur())
                    - m.query_advice(s_all_o, Rotation::cur());
                let cln = m.query_advice(mu_cln, Rotation::cur())
                    - m.query_advice(cln_l, Rotation::cur())
                        * m.query_advice(s_cln_o, Rotation::cur());
                vec![q.clone() * all, q * cln]
            });
        }

        Q18Config {
            customer,
            orders,
            lineitem,
            cond_thresh,
            q_thresh_same,
            thresh_byte,
            q_thresh_byte,
            q_thresh_dec,
            l_sorted,
            perm_lsort,
            q_lsort,
            q_lsort_last,
            lt_lkey_cur_next,
            iz_lkey_eq,
            q_line,
            q_first,
            q_accu,
            run_sum,
            iz_same_prev,
            iz_same_next,
            lt_thresh_sum,
            heavy,
            res_pad,
            res_sorted,
            perm_res,
            q_lookup_ord,
            q_lookup_cust,
            q_tbl_orders,
            q_tbl_customer,
            q_sort,
            lt_total_next_cur,
            lt_date_cur_next,
            iz_total_eq,
            iz_date_eq,

            cflag,
            part_c,
            part_o,
            part_l,
            perm_cons,
            q_cln_flag,
            q_res_flag,

            q_pw_cln,

            cp_agg_c,
            cp_agg_o,
            cp_join_c,
            cp_join_o,
            cp_root,
            cp_ones,
            cp_v_all_o,
            cp_v_cln_o,
            q_cp_one,
            q_cp_mu_o,
            q_cp_mu_l,

            instance,
            instance_test,
        }
    }

    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        customer_u64: Vec<Vec<u64>>, // [name, custkey]
        orders_u64: Vec<Vec<u64>>,   // [okey, custkey, date, total]
        lineitem_u64: Vec<Vec<u64>>, // [okey, qty]
        threshold: u64,
    ) -> Result<AssignedCell<F, F>, Error> {
        // The HAVING comparison is a NUM_BYTES-byte Lt chip and the threshold is
        // now range-checked to that width in circuit, so a wider parameter has no
        // witness at all. Fail here rather than with an unsatisfied constraint.
        assert!(
            threshold <= MAX_SENTINEL,
            "the HAVING parameter :1 = {} does not fit the {}-byte comparison this circuit uses",
            threshold,
            NUM_BYTES
        );

        // chips
        let iz_same_prev_chip = IsZeroChip::construct(self.config.iz_same_prev.clone());
        let iz_same_next_chip = IsZeroChip::construct(self.config.iz_same_next.clone());

        let lt_thresh_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_thresh_sum.clone());
        lt_thresh_chip.load(layouter)?;

        let iz_total_eq_chip = IsZeroChip::construct(self.config.iz_total_eq.clone());
        let iz_date_eq_chip = IsZeroChip::construct(self.config.iz_date_eq.clone());

        let lt_total_chip =
            LtChip::<F, NUM_BYTES>::construct(self.config.lt_total_next_cur.clone());
        lt_total_chip.load(layouter)?;
        let lt_date_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_date_cur_next.clone());
        lt_date_chip.load(layouter)?;

        // The l_sorted sortedness ladder and every Lt chip of the Cardinality
        // Preservation Check share one u8 fixed column, so a single load covers
        // all of them.
        let iz_lkey_eq_chip = IsZeroChip::construct(self.config.iz_lkey_eq.clone());
        let lt_lkey_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_lkey_cur_next.clone());
        lt_lkey_chip.load(layouter)?;

        // prepare sorted lineitems
        let mut l_sorted = lineitem_u64.clone();
        l_sorted.sort_by_key(|r| r[0]); // by orderkey
        let n = l_sorted.len();

        // compute run_sum and heavy only on last row of group
        let mut run_sum_u64 = vec![0u64; n];
        let mut heavy_u64 = vec![0u64; n];

        let mut acc: u128 = 0;
        let mut prev_ok: Option<u64> = None;

        for i in 0..n {
            let ok = l_sorted[i][0];
            let qty = l_sorted[i][1] as u128;

            if prev_ok == Some(ok) {
                acc += qty;
            } else {
                acc = qty;
            }
            run_sum_u64[i] = acc as u64;

            // row n is the PAD_OKEY sentinel the sortedness ladder pins above
            // every real key, so the last real row always closes its group
            let next_ok = if i + 1 < n {
                l_sorted[i + 1][0]
            } else {
                PAD_OKEY
            };
            let is_last = next_ok != ok;

            if is_last && run_sum_u64[i] > threshold {
                heavy_u64[i] = 1;
            }
            prev_ok = Some(ok);
        }

        // maps for attachment
        use std::collections::HashMap;
        let mut orders_map: HashMap<u64, (u64, u64, u64)> = HashMap::new(); // okey -> (cust,date,total)
        for r in &orders_u64 {
            orders_map.insert(r[0], (r[1], r[2], r[3]));
        }
        let mut cust_map: HashMap<u64, u64> = HashMap::new(); // custkey -> name
        for r in &customer_u64 {
            cust_map.insert(r[1], r[0]);
        }

        // build res_pad (length = n)
        let mut res_pad = vec![[PAD_NAME, PAD_CUST, PAD_OKEY, PAD_DATE, PAD_TOTAL, PAD_QSUM]; n];
        for i in 0..n {
            // only last rows can emit
            let next_ok = if i + 1 < n {
                l_sorted[i + 1][0]
            } else {
                PAD_OKEY
            };
            let is_last = next_ok != l_sorted[i][0];

            if is_last && heavy_u64[i] == 1 {
                let okey = l_sorted[i][0];
                let sumq = run_sum_u64[i];
                let (cust, date, total) = orders_map.get(&okey).copied().unwrap_or((0, 0, 0));
                let name = cust_map.get(&cust).copied().unwrap_or(0);
                res_pad[i] = [name, cust, okey, date, total, sumq];
            }
        }

        // build res_sorted (sort emitted rows by total desc, date asc)
        let mut emitted: Vec<[u64; 6]> = res_pad
            .iter()
            .copied()
            .filter(|r| r[2] != PAD_OKEY)
            .collect();
        emitted.sort_by(|a, b| b[4].cmp(&a[4]).then(a[3].cmp(&b[3])));

        let mut res_sorted = Vec::with_capacity(n);
        res_sorted.extend(emitted.into_iter());
        while res_sorted.len() < n {
            res_sorted.push([PAD_NAME, PAD_CUST, PAD_OKEY, PAD_DATE, PAD_TOTAL, PAD_QSUM]);
        }

        // ---------------- clean/residual partition of every relation ----------------
        // The honest prover's R^c is the fully reduced instance: exactly the
        // tuples that extend to a full join result. On the path
        // lineitem -> orders -> customer that is a semijoin reduction in each
        // direction, iterated to a fixpoint. `alive` is the set of lineitem
        // tuples the prover is willing to call clean, which is everything
        // except under the test hook below.
        let reduce = |alive: &[bool]| -> (Vec<u64>, Vec<u64>, Vec<u64>) {
            let mut cln_c = vec![1u64; customer_u64.len()];
            let mut cln_o = vec![0u64; orders_u64.len()];
            let mut cln_l: Vec<u64> = alive.iter().map(|&a| a as u64).collect();
            loop {
                let ck: HashSet<u64> = customer_u64
                    .iter()
                    .zip(cln_c.iter())
                    .filter(|(_, &f)| f == 1)
                    .map(|(c, _)| c[1])
                    .collect();
                let lk: HashSet<u64> = lineitem_u64
                    .iter()
                    .zip(cln_l.iter())
                    .filter(|(_, &f)| f == 1)
                    .map(|(l, _)| l[0])
                    .collect();
                let o_new: Vec<u64> = orders_u64
                    .iter()
                    .map(|o| (ck.contains(&o[1]) && lk.contains(&o[0])) as u64)
                    .collect();
                let ok: HashSet<u64> = orders_u64
                    .iter()
                    .zip(o_new.iter())
                    .filter(|(_, &f)| f == 1)
                    .map(|(o, _)| o[0])
                    .collect();
                let oc: HashSet<u64> = orders_u64
                    .iter()
                    .zip(o_new.iter())
                    .filter(|(_, &f)| f == 1)
                    .map(|(o, _)| o[1])
                    .collect();
                let l_new: Vec<u64> = lineitem_u64
                    .iter()
                    .zip(alive.iter())
                    .map(|(l, &a)| (a && ok.contains(&l[0])) as u64)
                    .collect();
                let c_new: Vec<u64> = customer_u64
                    .iter()
                    .map(|c| oc.contains(&c[1]) as u64)
                    .collect();
                if o_new == cln_o && l_new == cln_l && c_new == cln_c {
                    break;
                }
                cln_o = o_new;
                cln_l = l_new;
                cln_c = c_new;
            }
            (cln_c, cln_o, cln_l)
        };

        let mut alive_l = vec![true; lineitem_u64.len()];

        // test hook only: skip the reduction and call every tuple clean. The
        // partition still conserves every relation and both channels of
        // condition (4) then agree on every row, so only Pairwise Consistency
        // can see that the clean side is not the reduced instance.
        let all_clean = MARK_ALL_CLEAN.load(Ordering::Relaxed);
        let (mut cln_c, mut cln_o, mut cln_l) = if all_clean {
            (
                vec![1u64; customer_u64.len()],
                vec![1u64; orders_u64.len()],
                vec![1u64; lineitem_u64.len()],
            )
        } else {
            reduce(&alive_l)
        };

        // test hook only: hide one joinable lineitem tuple and re-reduce around
        // it, so the partition is still a valid reduced instance of a smaller
        // input and only condition (4) can see the difference
        let tamper = HIDE_ONE_CLEAN_TUPLE.load(Ordering::Relaxed) && !all_clean;
        if tamper {
            if let Some(pos) = cln_l.iter().position(|&f| f == 1) {
                alive_l[pos] = false;
                let reduced = reduce(&alive_l);
                cln_c = reduced.0;
                cln_o = reduced.1;
                cln_l = reduced.2;
            }
        }

        // the partition side of a Conservation Check: the clean rows first,
        // each carrying a constant 1, then the residual rows carrying 0. The
        // return value is the block and the length of its clean prefix.
        let split = |rows: &Vec<Vec<u64>>, flag: &Vec<u64>| -> (Vec<Vec<u64>>, usize) {
            let mut out: Vec<Vec<u64>> = Vec::with_capacity(rows.len());
            for (r, &f) in rows.iter().zip(flag.iter()) {
                if f == 1 {
                    let mut v = r.clone();
                    v.push(1);
                    out.push(v);
                }
            }
            let n_cln = out.len();
            for (r, &f) in rows.iter().zip(flag.iter()) {
                if f == 0 {
                    let mut v = r.clone();
                    v.push(0);
                    out.push(v);
                }
            }
            (out, n_cln)
        };
        let part_c_rows = split(&customer_u64, &cln_c);
        let part_o_rows = split(&orders_u64, &cln_o);
        let part_l_rows = split(&lineitem_u64, &cln_l);

        // Pairwise Consistency needs no witness of its own: its four lookups run
        // between the key columns of the partition groups assigned below, gated
        // by q_pw_cln over each group's clean prefix.

        // assign region
        layouter.assign_region(
            || "Q18 witness",
            |mut region| {
                // base tables
                for i in 0..customer_u64.len() {
                    for j in 0..2 {
                        region.assign_advice(
                            || "customer",
                            self.config.customer[j],
                            i,
                            || Value::known(F::from(customer_u64[i][j])),
                        )?;
                    }
                    self.config.q_tbl_customer.enable(&mut region, i)?;
                }
                for i in 0..orders_u64.len() {
                    for j in 0..4 {
                        region.assign_advice(
                            || "orders",
                            self.config.orders[j],
                            i,
                            || Value::known(F::from(orders_u64[i][j])),
                        )?;
                    }
                    self.config.q_tbl_orders.enable(&mut region, i)?;
                }
                for i in 0..lineitem_u64.len() {
                    for j in 0..2 {
                        region.assign_advice(
                            || "lineitem",
                            self.config.lineitem[j],
                            i,
                            || Value::known(F::from(lineitem_u64[i][j])),
                        )?;
                    }
                }

                // condition column (threshold) over n rows (like your Q3 condition)
                for i in 0..n {
                    region.assign_advice(
                        || "threshold",
                        self.config.cond_thresh,
                        i,
                        || Value::known(F::from(threshold)),
                    )?;
                }
                // The parameter is the same on every one of those rows, which is
                // what `q_thresh_same` now requires, and it is a 5-byte integer,
                // which is what `q_thresh_dec` and the u8 lookup require. The
                // decomposition is witnessed once, on row 0, from which the
                // constancy ladder carries the value to every other row.
                for i in 0..n.saturating_sub(1) {
                    self.config.q_thresh_same.enable(&mut region, i)?;
                }
                if n >= NUM_BYTES {
                    for k in 0..NUM_BYTES {
                        region.assign_advice(
                            || "cond_thresh limb",
                            self.config.thresh_byte,
                            k,
                            || Value::known(F::from((threshold >> (8 * k)) & 0xff)),
                        )?;
                        self.config.q_thresh_byte.enable(&mut region, k)?;
                    }
                    self.config.q_thresh_dec.enable(&mut region, 0)?;
                }

                // l_sorted + sentinel row for same_next
                for i in 0..n {
                    for j in 0..2 {
                        region.assign_advice(
                            || "l_sorted",
                            self.config.l_sorted[j],
                            i,
                            || Value::known(F::from(l_sorted[i][j])),
                        )?;
                    }
                }
                // sentinel at row n for same_next on the last row. It carries
                // PAD_OKEY, not 0: the sortedness ladder below covers the pair
                // (n-1, n), so the sentinel is forced above every real key and
                // a prover cannot repeat the last key here to swallow the last
                // group's boundary.
                region.assign_advice(
                    || "l_sorted_sentinel okey",
                    self.config.l_sorted[0],
                    n,
                    || Value::known(F::from(PAD_OKEY)),
                )?;
                region.assign_advice(
                    || "l_sorted_sentinel qty",
                    self.config.l_sorted[1],
                    n,
                    || Value::known(F::from(0u64)),
                )?;

                // ---- l_sorted key is nondecreasing, sentinel row included ----
                for i in 0..n {
                    self.config.q_lsort.enable(&mut region, i)?;
                    if i + 1 == n {
                        self.config.q_lsort_last.enable(&mut region, i)?;
                    }
                    let cur = l_sorted[i][0];
                    let next = if i + 1 < n {
                        l_sorted[i + 1][0]
                    } else {
                        PAD_OKEY
                    };
                    iz_lkey_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(next) - F::from(cur)),
                    )?;
                    lt_lkey_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(cur)),
                        Value::known(F::from(next)),
                    )?;
                }

                // enable perm lineitem<->l_sorted
                for i in 0..n {
                    self.config.perm_lsort.q_perm1.enable(&mut region, i)?;
                    self.config.perm_lsort.q_perm2.enable(&mut region, i)?;
                }

                // assign run_sum and heavy
                for i in 0..n {
                    // selectors
                    self.config.q_line.enable(&mut region, i)?;
                    if i == 0 {
                        self.config.q_first.enable(&mut region, i)?;
                    }
                    if i >= 1 {
                        self.config.q_accu.enable(&mut region, i)?;
                    }

                    region.assign_advice(
                        || "run_sum",
                        self.config.run_sum,
                        i,
                        || Value::known(F::from(run_sum_u64[i])),
                    )?;
                    region.assign_advice(
                        || "heavy",
                        self.config.heavy,
                        i,
                        || Value::known(F::from(heavy_u64[i])),
                    )?;
                }

                // isZero assignments
                for i in 1..n {
                    let diff = F::from(l_sorted[i][0]) - F::from(l_sorted[i - 1][0]);
                    iz_same_prev_chip.assign(&mut region, i, Value::known(diff))?;
                }
                for i in 0..n {
                    let next_ok = if i + 1 < n {
                        l_sorted[i + 1][0]
                    } else {
                        PAD_OKEY
                    };
                    let diff = F::from(next_ok) - F::from(l_sorted[i][0]);
                    iz_same_next_chip.assign(&mut region, i, Value::known(diff))?;
                }

                // Lt threshold < run_sum (only constrained on last rows due to q_enable in configure)
                for i in 0..n {
                    lt_thresh_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(threshold)),
                        Value::known(F::from(run_sum_u64[i])),
                    )?;
                }

                // assign res_pad and res_sorted
                for i in 0..n {
                    for j in 0..6 {
                        region.assign_advice(
                            || "res_pad",
                            self.config.res_pad[j],
                            i,
                            || Value::known(F::from(res_pad[i][j])),
                        )?;
                        region.assign_advice(
                            || "res_sorted",
                            self.config.res_sorted[j],
                            i,
                            || Value::known(F::from(res_sorted[i][j])),
                        )?;
                    }
                }

                // enable lookup selectors (safe to enable for all rows; gate uses emit inside lookup)
                for i in 0..n {
                    self.config.q_lookup_ord.enable(&mut region, i)?;
                    self.config.q_lookup_cust.enable(&mut region, i)?;
                }

                // perm res_pad<->res_sorted
                for i in 0..n {
                    self.config.perm_res.q_perm1.enable(&mut region, i)?;
                    self.config.perm_res.q_perm2.enable(&mut region, i)?;
                }

                // ORDER BY helper chips (rows 0..n-2)
                for i in 0..n.saturating_sub(1) {
                    self.config.q_sort.enable(&mut region, i)?;

                    let total_cur = res_sorted[i][4];
                    let total_next = res_sorted[i + 1][4];
                    let date_cur = res_sorted[i][3];
                    let date_next = res_sorted[i + 1][3];

                    iz_total_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(total_cur) - F::from(total_next)),
                    )?;
                    iz_date_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(date_cur) - F::from(date_next)),
                    )?;

                    lt_total_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(total_next)),
                        Value::known(F::from(total_cur)),
                    )?;
                    lt_date_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(date_cur)),
                        Value::known(F::from(date_next)),
                    )?;
                }

                // ---- Conservation Check: R_i == R_i^c U R_i^r, one per relation ----
                // input side: the clean indicator on the base rows. Q18 has no
                // predicate, so nothing rescales it and it is the bit the clean
                // channel below uses directly. Partition side: the
                // [clean | residual] block, with the indicator pinned to 1 on
                // the first section and 0 on the second.
                for (idx, (flag_col, flags, part_cols, part_rows, n_cln)) in [
                    (
                        self.config.cflag[0],
                        &cln_c,
                        &self.config.part_c,
                        &part_c_rows.0,
                        part_c_rows.1,
                    ),
                    (
                        self.config.cflag[1],
                        &cln_o,
                        &self.config.part_o,
                        &part_o_rows.0,
                        part_o_rows.1,
                    ),
                    (
                        self.config.cflag[2],
                        &cln_l,
                        &self.config.part_l,
                        &part_l_rows.0,
                        part_l_rows.1,
                    ),
                ]
                .into_iter()
                .enumerate()
                {
                    for (i, &f) in flags.iter().enumerate() {
                        region.assign_advice(
                            || "clean indicator",
                            flag_col,
                            i,
                            || Value::known(F::from(f)),
                        )?;
                    }
                    for (i, row) in part_rows.iter().enumerate() {
                        for (j, &v) in row.iter().enumerate() {
                            region.assign_advice(
                                || "partition",
                                part_cols[j],
                                i,
                                || Value::known(F::from(v)),
                            )?;
                        }
                        self.config.perm_cons[idx].q_perm1.enable(&mut region, i)?;
                        self.config.perm_cons[idx].q_perm2.enable(&mut region, i)?;
                    }
                    for i in 0..n_cln {
                        self.config.q_cln_flag[idx].enable(&mut region, i)?;
                    }
                    for i in n_cln..part_rows.len() {
                        self.config.q_res_flag[idx].enable(&mut region, i)?;
                    }
                }

                // ---- Pairwise Consistency: pi_K(R_i^c) == pi_K(R_j^c) ----
                // One selector per relation over the clean prefix of its
                // partition group. It is the input selector of the two lookups
                // out of that relation and the table selector of the lookups
                // into it, so this loop is the whole assignment the condition
                // needs.
                for (idx, n_cln) in [part_c_rows.1, part_o_rows.1, part_l_rows.1]
                    .into_iter()
                    .enumerate()
                {
                    for i in 0..n_cln {
                        self.config.q_pw_cln[idx].enable(&mut region, i)?;
                    }
                }

                // ===================== CARDINALITY PRESERVATION CHECK =====================
                // condition (4) of the One-Pass OBJ: the two multiplicity
                // channels are propagated up the tree
                // customer -> orders -> lineitem and their root sums compared.
                //
                // customer is a leaf without a predicate, so its input channel
                // is the constant 1 and its clean channel is the indicator.
                for i in 0..customer_u64.len() {
                    region.assign_advice(
                        || "cp v_all customer",
                        self.config.cp_ones,
                        i,
                        || Value::known(F::ONE),
                    )?;
                    self.config.q_cp_one.enable(&mut region, i)?;
                }
                let cp_rows_c: Vec<[u64; 3]> = (0..customer_u64.len())
                    .map(|i| [customer_u64[i][1], 1, cln_c[i]])
                    .collect();
                let cp_stage_c = build_cp_stage(&cp_rows_c, MAX_SENTINEL);
                assign_cp_agg(&mut region, &self.config.cp_agg_c, &cp_rows_c, &cp_stage_c)?;

                // orders -> customer edge, on the rows of orders
                let o_custkeys: Vec<u64> = orders_u64.iter().map(|o| o[1]).collect();
                let fetched_c = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_c,
                    &o_custkeys,
                    &cp_stage_c,
                    MAX_SENTINEL,
                )?;

                // orders is internal: it carries the two sums it just fetched,
                // the clean one masked by its own indicator
                let cp_rows_o: Vec<[u64; 3]> = (0..orders_u64.len())
                    .map(|i| [orders_u64[i][0], fetched_c[i].0, cln_o[i] * fetched_c[i].1])
                    .collect();
                for i in 0..orders_u64.len() {
                    region.assign_advice(
                        || "cp v_all orders",
                        self.config.cp_v_all_o,
                        i,
                        || Value::known(F::from(cp_rows_o[i][1])),
                    )?;
                    region.assign_advice(
                        || "cp v_cln orders",
                        self.config.cp_v_cln_o,
                        i,
                        || Value::known(F::from(cp_rows_o[i][2])),
                    )?;
                    self.config.q_cp_mu_o.enable(&mut region, i)?;
                }
                let cp_stage_o = build_cp_stage(&cp_rows_o, MAX_SENTINEL);
                assign_cp_agg(&mut region, &self.config.cp_agg_o, &cp_rows_o, &cp_stage_o)?;

                // lineitem -> orders edge, on the rows of lineitem
                let l_orderkeys: Vec<u64> = lineitem_u64.iter().map(|l| l[0]).collect();
                let fetched_o = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_o,
                    &l_orderkeys,
                    &cp_stage_o,
                    MAX_SENTINEL,
                )?;

                // root multiplicities and the equality between the two sums
                let cp_mu: Vec<(u64, u64)> = (0..lineitem_u64.len())
                    .map(|i| (fetched_o[i].0, cln_l[i] * fetched_o[i].1))
                    .collect();
                for i in 0..lineitem_u64.len() {
                    self.config.q_cp_mu_l.enable(&mut region, i)?;
                }
                let (cp_all, cp_cln) = assign_cp_root(&mut region, &self.config.cp_root, &cp_mu)?;
                if !tamper && !all_clean {
                    debug_assert_eq!(
                        cp_all, cp_cln,
                        "cardinality preservation: |R^c join| != |R join|"
                    );
                }

                // public output (same style as your Q3)
                let out = region.assign_advice(
                    || "instance_test",
                    self.config.instance_test,
                    0,
                    || Value::known(F::from(1u64)),
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

#[derive(Clone, Debug)]
pub struct MyCircuit<F: Field + Ord> {
    pub customer: Vec<Vec<u64>>,
    pub orders: Vec<Vec<u64>>,
    pub lineitem: Vec<Vec<u64>>,
    pub threshold: u64, // <-- :1 in TPCH Q18
    pub _marker: PhantomData<F>,
}

impl<F: Field + Ord> Default for MyCircuit<F> {
    fn default() -> Self {
        Self {
            customer: vec![],
            orders: vec![],
            lineitem: vec![],
            threshold: 0,
            _marker: PhantomData,
        }
    }
}

impl<F: Field + Ord> Circuit<F> for MyCircuit<F> {
    type Config = Q18Config<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        Q18Chip::<F>::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = Q18Chip::<F>::construct(config);

        let out_cell = chip.assign(
            &mut layouter,
            self.customer.clone(),
            self.orders.clone(),
            self.lineitem.clone(),
            self.threshold, // <-- pass :1
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
    use halo2curves::pasta::{vesta, EqAffine, Fp};

    use halo2_proofs::{
        plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit},
        poly::{
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

    use halo2_proofs::poly::commitment::Params;
    use rand::rngs::OsRng;
    use std::collections::HashSet;
    use std::marker::PhantomData;
    use std::sync::atomic::Ordering;
    use std::time::Instant;
    use std::{fs::File, io::Write, path::Path};

    fn generate_and_verify_proof<C: Circuit<Fp>>(
        _k: u32,
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
    #[ignore = "inherited heavy end-to-end proof; the fast check is test_cardinality_preservation"]
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
            // customer = [c_name_u64, c_custkey]
            customer = records
                .iter()
                .map(|record| vec![string_to_u64(&record.c_name), record.c_custkey])
                .collect();
        }

        if let Ok(records) = data_processing::orders_read_records_from_file(orders_file_path) {
            // IMPORTANT: match Q18Chip expectation:
            // orders = [o_orderkey, o_custkey, o_orderdate_ts, o_totalprice_scaled]
            orders = records
                .iter()
                .map(|record| {
                    vec![
                        record.o_orderkey,
                        record.o_custkey,
                        date_to_timestamp(&record.o_orderdate),
                        scale_by_1000(record.o_totalprice),
                    ]
                })
                .collect();
        }

        if let Ok(records) = data_processing::lineitem_read_records_from_file(lineitem_file_path) {
            // lineitem = [l_orderkey, l_quantity]
            lineitem = records
                .iter()
                .map(|record| vec![record.l_orderkey, record.l_quantity])
                .collect();
        }

        // TPCH Q18 typical parameter is 300
        let threshold: u64 = 300;

        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            threshold,
            _marker: PhantomData,
        };

        let public_input = vec![Fp::from(1)];

        // With VPJOIN_MOCK=1 this checks every gate, lookup and shuffle at full
        // scale under MockProver, which does no cryptography at all, instead of
        // generating a real proof. That is the cheap way to confirm the circuit
        // still fits its degree on the whole dataset. Unset, it measures a real
        // keygen / prove / verify, which is the number the paper reports.
        let test = std::env::var("VPJOIN_MOCK")
            .map(|v| v == "1")
            .unwrap_or(false);

        if test {
            let prover = MockProver::run(k, &circuit, vec![public_input]).unwrap();
            prover.assert_satisfied();
        } else {
            let proof_path = &crate::paths::proof_file("proof_obj_q18");
            generate_and_verify_proof(k, circuit, &public_input, proof_path);
        }
    }

    /// The maximum gate degree drives the size of every FFT the prover runs, so
    /// a constraint added for soundness must not raise it. This probe builds the
    /// constraint system only, with no witness and no proof, and also prints the
    /// column and argument counts so the cost of a change is visible.
    #[test]
    fn test_max_gate_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        let mut cs = ConstraintSystem::<Fp>::default();
        let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
        println!("cs.degree() = {}", cs.degree());
        println!(
            "advice = {}, fixed = {}, selectors = {}, gates = {}, lookups = {}, shuffles = {}",
            cs.num_advice_columns(),
            cs.num_fixed_columns(),
            cs.num_selectors(),
            cs.gates().len(),
            cs.lookups().len(),
            cs.shuffles().len(),
        );
        // 9 is what the Cardinality Preservation Check costs here, measured
        // before the two HAVING gates were added and unchanged by them: both are
        // degree 2, and the u8 lookup on the threshold limbs has a degree-2
        // input like the Pairwise Consistency lookups. Anything added later that
        // raises this doubles every FFT the prover runs.
        assert!(
            cs.degree() <= 9,
            "the maximum gate degree rose to {}",
            cs.degree()
        );
    }

    /// Fast correctness check of the Cardinality Preservation Check: a
    /// truncated slice of the dataset under MockProver, which verifies every
    /// gate, shuffle and lookup of the circuit without paying for a real proof.
    #[test]
    fn test_cardinality_preservation() {
        let k = 15;

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

        // A slice small enough for MockProver but large enough that the
        // reduction drops tuples on every relation: 1607 of the 2010 orders
        // dangle on the customer side, 36 lineitem rows dangle on the orders
        // side, and 125 of the 300 customers end up residual, so both edges
        // exercise the gap witness and the sigma = 0 default. The one group
        // that passes HAVING (orderkey 6882, custkey 178) has its order and its
        // customer inside the slice, so the two attribute lookups still hold.
        const N_CUST: usize = 300;
        const N_ORD: usize = 2000;
        const N_LINE: usize = 8000;
        // Ten more orders, taken from further down the file so that their
        // orderkeys lie past the last one the lineitem slice covers. A prefix of
        // orders alone leaves every order's orderkey with a lineitem in the
        // slice, and then the fourth Pairwise Consistency lookup, `o^c orderkey
        // in l^c`, has nothing to reject in the all-clean direction below and
        // that direction rests on the other three. These ten give it a witness.
        // They carry no lineitem, so the honest reduction leaves them residual
        // and the first two directions see them only as ten more residual rows.
        const ORD_TAIL_FROM: usize = 4000;
        const ORD_TAIL: usize = 10;

        let mut customer: Vec<Vec<u64>> = Vec::new();
        let mut orders: Vec<Vec<u64>> = Vec::new();
        let mut lineitem: Vec<Vec<u64>> = Vec::new();

        if let Ok(records) = data_processing::customer_read_records_from_file(
            &crate::paths::data_file("customer.tbl"),
        ) {
            customer = records
                .iter()
                .take(N_CUST)
                .map(|record| vec![string_to_u64(&record.c_name), record.c_custkey])
                .collect();
        }
        if let Ok(records) =
            data_processing::orders_read_records_from_file(&crate::paths::data_file("orders.tbl"))
        {
            orders = records
                .iter()
                .take(N_ORD)
                .chain(records.iter().skip(ORD_TAIL_FROM).take(ORD_TAIL))
                .map(|record| {
                    vec![
                        record.o_orderkey,
                        record.o_custkey,
                        date_to_timestamp(&record.o_orderdate),
                        scale_by_1000(record.o_totalprice),
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
                .map(|record| vec![record.l_orderkey, record.l_quantity])
                .collect();
        }

        assert!(
            !customer.is_empty() && !orders.is_empty() && !lineitem.is_empty(),
            "dataset files not found under {}",
            crate::paths::data_file("customer.tbl")
        );

        // The all-clean partition of the third direction below is only rejected
        // by Pairwise Consistency if the slice really does contain a dangling
        // tuple, and each of the four lookups sees only its own direction of its
        // own edge. Count a witness for each one here, so that no direction of
        // the condition passes vacuously and the third direction below can
        // insist on all four.
        let c_keys: HashSet<u64> = customer.iter().map(|c| c[1]).collect();
        let l_keys: HashSet<u64> = lineitem.iter().map(|l| l[0]).collect();
        let o_keys: HashSet<u64> = orders.iter().map(|o| o[0]).collect();
        let o_custkeys: HashSet<u64> = orders.iter().map(|o| o[1]).collect();
        let pw_witnesses = [
            (
                "pw: o^c custkey in c^c",
                orders.iter().filter(|o| !c_keys.contains(&o[1])).count(),
            ),
            (
                "pw: c^c custkey in o^c",
                customer
                    .iter()
                    .filter(|c| !o_custkeys.contains(&c[1]))
                    .count(),
            ),
            (
                "pw: l^c orderkey in o^c",
                lineitem.iter().filter(|l| !o_keys.contains(&l[0])).count(),
            ),
            (
                "pw: o^c orderkey in l^c",
                orders.iter().filter(|o| !l_keys.contains(&o[0])).count(),
            ),
        ];
        for (name, witnesses) in pw_witnesses.iter() {
            assert!(
                *witnesses > 0,
                "no tuple of the slice violates `{}`, so that lookup would \
                 legitimately accept the all-clean partition and the third \
                 direction below would rest on the other three",
                name
            );
        }

        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            threshold: 300,
            _marker: PhantomData,
        };

        let prover = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        prover.assert_satisfied();

        // Negative direction: the same witness with one joinable lineitem tuple
        // hidden in the residual side, and the partition re-reduced around it so
        // that it is still a valid reduced instance and the Conservation Checks
        // all still hold. Only condition (4) can see this, so the circuit must
        // now reject.
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

        // Third direction, the escape that condition (3) closes: the all-clean
        // partition. Conservation still holds and both channels of condition (4)
        // then compute the same number on every row, so the two root sums agree
        // for free. Only Pairwise Consistency can see that the clean side is not
        // the reduced instance, and the dangling tuples counted above are what
        // it sees. Now that the two clean key columns are looked up in each other
        // rather than in a table the prover fills, this direction is a test of
        // the condition itself and not of an honest table filling, so it insists
        // that every one of the four lookups reject.
        super::MARK_ALL_CLEAN.store(true, Ordering::Relaxed);
        let all_clean = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        let verdict = all_clean.verify();
        super::MARK_ALL_CLEAN.store(false, Ordering::Relaxed);

        let failures = verdict.expect_err("condition (3) accepted the all-clean partition");
        assert!(
            failures.iter().any(|f| matches!(
                f,
                VerifyFailure::Lookup { name, .. } if name.starts_with("pw: ")
            )),
            "the circuit rejected, but not through Pairwise Consistency: {:?}",
            failures
        );
        for (name, witnesses) in pw_witnesses.iter() {
            assert!(
                failures.iter().any(|f| format!("{:?}", f).contains(name)),
                "`{}` accepted the all-clean partition although {} tuples of the \
                 slice violate it",
                name,
                witnesses
            );
        }
    }
}
