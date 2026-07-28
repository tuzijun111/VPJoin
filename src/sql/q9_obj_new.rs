//! TPC-H Q9 under the revised One-Pass OBJ.
//!
//! `q9_obj.rs` realizes the earlier four-condition gate: the prover materializes
//! a `[clean | residual | pad]` group per relation and a Conservation Check per
//! relation ties it back to the committed rows. This file realizes the revised
//! gate, in which the certified object is one selector bit per committed row.
//! The split then partitions the input occurrences by construction, so
//! Conservation and Non-Membership have nothing left to check and the six
//! partition groups, their six shuffles and the whole pad-section bookkeeping
//! disappear. Three conditions remain:
//!
//!   (1) Selector Check          c(1-c) = 0  and  c(1-b) = 0
//!   (2) Pairwise Consistency    pi_K(R_i^c) == pi_K(R_j^c) on all five edges
//!   (3) Cardinality Preservation  the two root sums agree
//!
//! Only `part` carries an in-relation predicate, so `b = p_keep` there and
//! `b == 1` on the other five relations.
//!
//! Two consequences worth recording.
//!
//! The KNOWN DESIGN GAP `q9_obj.rs` documents on its Pairwise Consistency
//! selectors is closed here. There, the ten lookups were gated by selectors
//! whose enabled range was the clean prefix of a partition group, so a
//! malicious circuit GENERATOR could publish a vk with all six ranges empty and
//! make the condition vacuous. Here the gating factor is the committed selector
//! COLUMN, which the Cardinality Preservation Check reads on the same rows, and
//! the fixed selectors only mark the relation's real rows, a function of
//! |R_i| alone.
//!
//! The aggregation moves from the clean prefix of `l_join_pad` onto the
//! committed lineitem rows. A deselected row keeps its place, its profit triple
//! is masked to the canonical PAD triple, and it joins the trailing PAD run of
//! the sorted view, which emits the result pad triple and is dropped by the
//! result shuffle exactly as the old pad section was.

use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};
use crate::circuits::conserve_idx::{
    assign_conserve, assign_row_index, configure_conserve, configure_row_index, ConserveConfig,
    RowIndexConfig,
};
use crate::circuits::card_preserve::{
    assign_cp_agg, assign_cp_join, assign_cp_root, build_cp_stage, configure_cp_agg,
    configure_cp_join, configure_cp_root, wire_cp_edge, CpAggConfig, CpJoinConfig, CpRootConfig,
};

use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

// ----------------- tuning -----------------
const NUM_BYTES: usize = 5;
const MAX_SENTINEL: u64 = (1u64 << (8 * NUM_BYTES)) - 1; // 2^40-1
const SCALE: u64 = 1000;

// pack (partkey, suppkey) into one u64 for partsupp key
// partkey up to ~200k, suppkey up to ~10k -> SHIFT=1<<20 is safe
const PS_SHIFT: u64 = 1u64 << 20;

// n_nationkey starts at 0 in TPC-H and key 0 is reserved twice over: it is the
// dummy row of the key-indexed tables of the Cardinality Preservation Check, and
// it is the value both sides of a Pairwise Consistency lookup take on every row
// where their selector is off, so a key of 0 would be in the table for free.
// Both sides of the supplier -> nation edge are therefore shifted by this
// constant, in the same direction, which preserves the equijoin.
const NAT_SHIFT: u64 = 1;

// index order of the relations, the one cflag / q_cln_flag / q_res_flag and the
// input side of the Pairwise Consistency lookups all use
const R_PART: usize = 0;
const R_SUPP: usize = 1;
const R_NAT: usize = 2;
const R_ORD: usize = 3;
const R_PS: usize = 4;
const R_LINE: usize = 5;

/// Test hook, off in every benchmark path: when set, the prover moves one
/// joinable lineitem tuple to the residual side and re-reduces the neighbours
/// around it, so the partition still passes Conservation, Non-Membership and
/// Pairwise Consistency and only condition (4) can catch it. This is exactly
/// the cheat a residual-side-only argument misses, so the negative test in this
/// module is what shows the Cardinality Preservation Check is not vacuous.
pub static HIDE_ONE_CLEAN_TUPLE: AtomicBool = AtomicBool::new(false);

/// Test hook, off in every benchmark path: when set, the prover skips the
/// semijoin reduction entirely and declares every real tuple clean, leaving the
/// residual section of every partition empty. Conservation still holds and both
/// channels of condition (4) then agree on every row, so this is exactly the
/// escape Pairwise Consistency has to close: a dangling tuple left in `R_i^c`
/// has no partner in the clean part of the adjacent relation, so the key sets on
/// that edge differ and a `pw: ` lookup must reject.
pub static MARK_ALL_CLEAN: AtomicBool = AtomicBool::new(false);

// ----------------- paddings -----------------
// Part: [p_partkey, p_name_hash]
const PAD_PKEY: u64 = MAX_SENTINEL;
const PAD_PNAME: u64 = MAX_SENTINEL;

// Supplier: [s_suppkey, s_nationkey]
const PAD_SKEY: u64 = MAX_SENTINEL;
const PAD_SNAT: u64 = MAX_SENTINEL;

// Nation: [n_nationkey, n_name_hash]
const PAD_NKEY: u64 = MAX_SENTINEL;
const PAD_NNAME: u64 = MAX_SENTINEL;

// Orders: [o_orderkey, o_year]
const PAD_OKEY: u64 = MAX_SENTINEL;
const PAD_OYEAR: u64 = 0; // for DESC, 0 is smallest

// Partsupp: [ps_key, ps_supplycost]
const PAD_PSKEY: u64 = MAX_SENTINEL;
const PAD_PSCOST: u64 = 0;

// Lineitem join working view: [l_orderkey,l_partkey,l_suppkey,l_qty,l_ext,l_disc]
const PAD_LOKEY: u64 = MAX_SENTINEL;
const PAD_LPKEY: u64 = MAX_SENTINEL;
const PAD_LSKEY: u64 = MAX_SENTINEL;
const PAD_LQTY: u64 = 0;
const PAD_LEXT: u64 = 0;
const PAD_LDISC: u64 = 0;

// Profit row: [nation_hash, year, amount]
const PAD_PROF_N: u64 = MAX_SENTINEL;
const PAD_PROF_Y: u64 = 0;
const PAD_PROF_A: u64 = 0;

// Result row: [nation_hash, year, sum_profit]
const PAD_RES_N: u64 = MAX_SENTINEL;
const PAD_RES_Y: u64 = 0;
const PAD_RES_S: u64 = 0;

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

#[derive(Clone, Debug)]
pub struct TestCircuitConfig<F: Field + Ord> {
    // selectors
    q_part_pred: Selector, // p_name_hash == :1

    // base tables
    part: Vec<Column<Advice>>,     // 2
    supplier: Vec<Column<Advice>>, // 2
    nation: Vec<Column<Advice>>,   // 2
    orders: Vec<Column<Advice>>,   // 2
    partsupp: Vec<Column<Advice>>, // 2
    lineitem: Vec<Column<Advice>>, // 6

    // condition (:1)
    cond: Column<Advice>,
    q_cond_eq: Selector, // :1 is the same on every part row

    // ---------------- (1) Conservation Check ----------------
    // R^_i == R^_i^c U+ R^_i^r over the INDEXED relation, one permutation each
    row_idx: RowIndexConfig,
    cons: Vec<ConserveConfig>, // 6, in the order R_PART .. R_LINE
    // the indicator c per committed row, same order
    cflag: Vec<Column<Advice>>, // 6
    // one complex selector per relation over its committed rows. It gates both
    // sides of every lookup this circuit runs and the booleanity gate; a simple
    // selector may appear on neither side of a lookup, which is why it is
    // complex. Its enabled range is |R_i|, so it is a function of the public
    // sizes and of nothing the prover chooses.
    q_row: Vec<Selector>, // 6

    // part predicate output, the bit b of the Selector Check
    p_keep: Column<Advice>, // boolean

    // ---------- attach columns on the committed lineitem rows ----------
    l_supplycost: Column<Advice>, // from partsupp
    l_year: Column<Advice>,       // from orders
    l_nationkey: Column<Advice>,  // from supplier
    l_nationname: Column<Advice>, // from nation

    // amount = ext*(SCALE-disc) - supplycost*qty*SCALE
    q_amount: Selector,
    amount: Column<Advice>,

    // profit table (unsorted + sorted) length = l_join_pad.len()
    profit: Vec<Column<Advice>>,        // 3 [nation_hash, year, amount]
    profit_sorted: Vec<Column<Advice>>, // 3
    perm_profit: PermAnyConfig,

    // enforce profit_sorted ORDER BY nation ASC, year DESC
    q_sort_profit: Selector,
    lt_nation_cur_next: LtConfig<F, NUM_BYTES>,
    lt_year_next_cur: LtConfig<F, NUM_BYTES>,
    iz_nation_eq: IsZeroConfig<F>,
    iz_year_eq: IsZeroConfig<F>,

    // grouping on (nation,year)
    q_first: Selector,
    q_accu: Selector,
    q_line: Selector,
    q_gkey_sentinel: Selector, // gkey at row n is 0
    q_gkey_last: Selector,     // the last profit row ends its group

    gkey: Column<Advice>, // packed group key
    run_sum: Column<Advice>,
    iz_same_prev: IsZeroConfig<F>,
    iz_same_next: IsZeroConfig<F>,

    // emitted padded results (length = profit_sorted.len())
    res_pad: Vec<Column<Advice>>, // 3 [nation,year,sum_profit] only on last rows else PAD
    res_sorted: Vec<Column<Advice>>, // 3
    perm_res: PermAnyConfig,

    // result ORDER BY nation ASC, year DESC
    q_sort_res: Selector,
    lt_res_nation_cur_next: LtConfig<F, NUM_BYTES>,
    lt_res_year_next_cur: LtConfig<F, NUM_BYTES>,
    iz_res_nation_eq: IsZeroConfig<F>,
    iz_res_year_eq: IsZeroConfig<F>,

    // ---------- tuple lookups (over the committed relations) ----------
    q_lkp_orders: Selector,
    q_lkp_supplier: Selector,
    q_lkp_nation: Selector,
    q_lkp_partsupp: Selector,

    instance: Column<Instance>,
    instance_test: Column<Advice>,

    iz_part: IsZeroConfig<F>,

    // ---------------- Cardinality Preservation Check ----------------
    // condition (4): |R^c join| == |R join|, over the tree rooted at lineitem
    cp_one: Column<Advice>, // constant 1, the input channel of an unfiltered leaf
    // Three derived key columns on the committed rows. They serve the
    // Cardinality Preservation Check, the Pairwise Consistency lookups and the
    // partsupp attach lookup alike, which is what lets `q9_obj.rs`'s separate
    // `l_pw_pskey` / `s_pw_nkey` / `n_pw_key` and `l_ps_key` all disappear.
    n_key_cp: Column<Advice>, // n_nationkey + NAT_SHIFT, on nation rows
    s_nkey_cp: Column<Advice>, // s_nationkey + NAT_SHIFT, on supplier rows
    l_ps_key_cp: Column<Advice>, // packed partsupp key, on base lineitem rows
    v_all_s: Column<Advice>, // supplier is internal: its two multiplicities
    v_cln_s: Column<Advice>,
    t_all: Column<Advice>, // the four root sigmas, folded pairwise per channel
    u_all: Column<Advice>,
    t_cln: Column<Advice>,
    u_cln: Column<Advice>,

    cp_agg_p: CpAggConfig<F, NUM_BYTES>, // child part, keyed by p_partkey
    cp_agg_o: CpAggConfig<F, NUM_BYTES>, // child orders, keyed by o_orderkey
    cp_agg_s: CpAggConfig<F, NUM_BYTES>, // child supplier, keyed by s_suppkey
    cp_agg_ps: CpAggConfig<F, NUM_BYTES>, // child partsupp, keyed by ps_key
    cp_agg_n: CpAggConfig<F, NUM_BYTES>, // child nation, keyed by n_nationkey

    cp_join_p: CpJoinConfig<F, NUM_BYTES>,  // lineitem -> part
    cp_join_o: CpJoinConfig<F, NUM_BYTES>,  // lineitem -> orders
    cp_join_s: CpJoinConfig<F, NUM_BYTES>,  // lineitem -> supplier
    cp_join_ps: CpJoinConfig<F, NUM_BYTES>, // lineitem -> partsupp
    cp_join_n: CpJoinConfig<F, NUM_BYTES>,  // supplier -> nation

    cp_root: CpRootConfig,
    q_cp_one: Selector, // cp_one == 1
    q_cp_nat: Selector, // nation rows: shifted key
    q_cp_sup: Selector, // supplier rows: shifted key and the two multiplicities
    q_cp_mu: Selector,  // lineitem rows: packed key and the two root products

}

#[derive(Clone, Debug)]
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
        part_rows: &[Vec<F>],
        join_cells: &[Vec<AssignedCell<F, F>>],
        dis_cells: &[Vec<AssignedCell<F, F>>],
    ) -> Result<(), Error> {
        let join_len = join_cells.len();
        let dis_len = dis_cells.len();

        for (i, r) in part_rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                let part_cell =
                    region.assign_advice(|| tag, part_cols[j], i, || Value::known(v))?;

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

        // base tables
        let part = vec![meta.advice_column(), meta.advice_column()];
        let supplier = vec![meta.advice_column(), meta.advice_column()];
        let nation = vec![meta.advice_column(), meta.advice_column()];
        let orders = vec![meta.advice_column(), meta.advice_column()];
        let partsupp = vec![meta.advice_column(), meta.advice_column()];
        let lineitem = (0..6).map(|_| meta.advice_column()).collect::<Vec<_>>();

        for &col in part
            .iter()
            .chain(supplier.iter())
            .chain(nation.iter())
            .chain(orders.iter())
            .chain(partsupp.iter())
            .chain(lineitem.iter())
        {
            meta.enable_equality(col);
        }

        // condition (:1 hash)
        let cond = meta.advice_column();

        // clean indicator per base row, one column per relation, in the order
        // [part, supplier, nation, orders, partsupp, lineitem]
        let cflag = (0..6).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for &c in cflag.iter() {
            meta.enable_equality(c);
        }

        // ---------- part predicate ----------
        let q_part_pred = meta.selector();
        let p_keep = meta.advice_column();

        let is_zero_aux = meta.advice_column();
        let iz_part = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_part_pred),
            |m| m.query_advice(part[1], Rotation::cur()) - m.query_advice(cond, Rotation::cur()),
            is_zero_aux,
        );

        // `:1` is the query parameter, read by `iz_part` at Rotation::cur on
        // every part row. It was a free advice cell per row, so the prover could
        // answer a different predicate on every row of part: `p_keep`, which is
        // the input channel of `cp_agg_p` and therefore of condition (10), was a
        // per-row choice rather than a function of one parameter. This gate makes
        // it one value for the whole proof, by equating consecutive rows over
        // exactly the rows `q_part_pred` reads it on.
        //
        // It does not make the parameter PUBLIC. Nothing in this circuit is: the
        // only instance cell carries F::ONE. Binding `cond` to the instance
        // column needs a second instance value, which every caller of this
        // circuit would have to pass, so it cannot be done from this file.
        let q_cond_eq = meta.selector();
        meta.create_gate("cond is one value for the whole proof", |m| {
            let q = m.query_selector(q_cond_eq);
            let cur = m.query_advice(cond, Rotation::cur());
            let next = m.query_advice(cond, Rotation::next());
            vec![q * (cur - next)]
        });

        // keep boolean and equal to iz_p.expr()
        meta.create_gate("p_keep = (p_name_hash == :1)", |m| {
            let q = m.query_selector(q_part_pred);
            let keep = m.query_advice(p_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone() * (keep.clone() - iz_part.expr()),
                q * keep.clone() * (one - keep),
            ]
        });

        // ---------- (1) Conservation Check ----------
        // One permutation argument per relation, between the indexed relation
        // R^_i and the concatenation of its two parts. The indices are
        // distinct, so R^_i is a set even though the relation is a bag, and
        // that single permutation rules out an occurrence being fabricated,
        // lost, duplicated or counted on both sides: no Non-Membership Check.
        let row_idx = configure_row_index::<F>(meta);
        let cons: Vec<ConserveConfig> = vec![
            configure_conserve::<F>(meta, &row_idx, &part, cflag[R_PART]),
            configure_conserve::<F>(meta, &row_idx, &supplier, cflag[R_SUPP]),
            configure_conserve::<F>(meta, &row_idx, &nation, cflag[R_NAT]),
            configure_conserve::<F>(meta, &row_idx, &orders, cflag[R_ORD]),
            configure_conserve::<F>(meta, &row_idx, &partsupp, cflag[R_PS]),
            configure_conserve::<F>(meta, &row_idx, &lineitem, cflag[R_LINE]),
        ];

        // ---------- Selector Check, the predicate half ----------
        // The prover supplies one bit per committed row. Selection is by
        // position, so no row can be fabricated, dropped or placed on both
        // sides and there is nothing for a Conservation or Non-Membership Check
        // to compare against: `q9_obj.rs`'s six partition groups, six shuffles
        // and pad-section gates all go away with the materialization.
        //
        // Two gates per relation. The booleanity gate keeps the bit from
        // scaling a multiplicity channel, and the predicate gate confines the
        // selection to the rows the WHERE clause keeps, which is what makes the
        // input channel of (3) count the join of the PREDICATE-FILTERED inputs.
        // Only `part` carries a predicate here, so the second gate is written
        // once; on the other five relations b == 1 and it would be vacuous.
        let q_row = (0..6).map(|_| meta.complex_selector()).collect::<Vec<_>>();
        for (idx, &c) in cflag.iter().enumerate() {
            let q = q_row[idx];
            meta.create_gate("selector is a bit", move |m| {
                let q = m.query_selector(q);
                let c = m.query_advice(c, Rotation::cur());
                vec![q * c.clone() * (Expression::Constant(F::ONE) - c)]
            });
        }
        {
            let q = q_row[R_PART];
            let c = cflag[R_PART];
            meta.create_gate("selected part row satisfies its predicate", move |m| {
                let q = m.query_selector(q);
                let c = m.query_advice(c, Rotation::cur());
                let b = m.query_advice(p_keep, Rotation::cur());
                vec![q * c * (Expression::Constant(F::ONE) - b)]
            });
        }

        // ---------- attach cols ----------
        // The packed partsupp key lives in `l_ps_key_cp` below, on the same
        // committed rows, so this file has one packed-key column where
        // `q9_obj.rs` had three.
        let l_supplycost = meta.advice_column();
        let l_year = meta.advice_column();
        let l_nationkey = meta.advice_column();
        let l_nationname = meta.advice_column();

        let q_amount = meta.selector();
        let amount = meta.advice_column();

        // amount formula.
        //
        // KNOWN COMPLETENESS LIMIT, not an under-constraint. The gate evaluates
        // ext*(SCALE-disc) - cost*qty*SCALE in F, so a row whose amount is
        // negative gets the field element p - |a|. The witness generator below
        // stores the low 64 bits of the i128 instead, F::from(2^64 - |a|), which
        // is a different element, so this gate is unsatisfiable on such a row and
        // NO proof exists for it. That direction is safe: it rejects honest
        // provers, it does not admit dishonest ones. It is reachable at full TPC-H
        // scale, where ps_supplycost can exceed l_extendedprice*(1-l_discount)/qty.
        // Fixing it means a sign/magnitude representation for `amount`, `profit[2]`
        // and both running sums, which is a redesign of the aggregation rather
        // than a constraint, so it is left documented here.
        //
        // Relatedly, `run_sum` and `res_pad[2]` carry no range check, so the
        // group sums this circuit certifies are field elements mod p, not
        // bounded integers. That is inherent to the field-arithmetic aggregation
        // these OBJ circuits all use.
        meta.create_gate("amount = ext*(SCALE-disc) - cost*qty*SCALE", |m| {
            let q = m.query_selector(q_amount);
            let qty = m.query_advice(lineitem[3], Rotation::cur());
            let ext = m.query_advice(lineitem[4], Rotation::cur());
            let disc = m.query_advice(lineitem[5], Rotation::cur());
            let cost = m.query_advice(l_supplycost, Rotation::cur());
            let a = m.query_advice(amount, Rotation::cur());

            let scale = Expression::Constant(F::from(SCALE));
            // ext*(SCALE-disc) - cost*qty*SCALE
            vec![q * (a - (ext * (scale.clone() - disc) - cost * qty * scale))]
        });

        // ---------- tuple lookups ----------
        // Every one of these now runs between the COMMITTED relations, with
        // both sides gated by the selector: a deselected row and a row past the
        // relation both contribute the all-zero tuple, which is what a
        // gated-off input row reads, and the shift by one keeps that dummy away
        // from any real tuple. `q9_obj.rs` read them off the `*_join_pad`
        // groups, which needed a Conservation Check each to mean anything.
        //
        // The part membership lookup `q9_obj.rs` runs here is gone: with both
        // sides on the committed relations it is character for character the
        // `pw: lineitem^c partkey in part^c` lookup below, so it was a
        // duplicate rather than a second check.
        let q_lkp_orders = meta.complex_selector();
        let q_lkp_supplier = meta.complex_selector();
        let q_lkp_nation = meta.complex_selector();
        let q_lkp_partsupp = meta.complex_selector();

        // Orders: (l_orderkey, l_year) in the selected orders
        meta.lookup_any("attach year from orders", |m| {
            let one = Expression::Constant(F::ONE);
            let q_in = m.query_selector(q_lkp_orders) * m.query_advice(cflag[R_LINE], Rotation::cur());
            let q_t = m.query_selector(q_row[R_ORD]) * m.query_advice(cflag[R_ORD], Rotation::cur());
            vec![
                (
                    q_in.clone() * (m.query_advice(lineitem[0], Rotation::cur()) + one.clone()),
                    q_t.clone() * (m.query_advice(orders[0], Rotation::cur()) + one.clone()),
                ),
                (
                    q_in * (m.query_advice(l_year, Rotation::cur()) + one.clone()),
                    q_t * (m.query_advice(orders[1], Rotation::cur()) + one),
                ),
            ]
        });

        // Supplier: (l_suppkey, l_nationkey) in the selected suppliers
        meta.lookup_any("attach nationkey from supplier", |m| {
            let one = Expression::Constant(F::ONE);
            let q_in = m.query_selector(q_lkp_supplier) * m.query_advice(cflag[R_LINE], Rotation::cur());
            let q_t = m.query_selector(q_row[R_SUPP]) * m.query_advice(cflag[R_SUPP], Rotation::cur());
            vec![
                (
                    q_in.clone() * (m.query_advice(lineitem[2], Rotation::cur()) + one.clone()),
                    q_t.clone() * (m.query_advice(supplier[0], Rotation::cur()) + one.clone()),
                ),
                (
                    q_in * (m.query_advice(l_nationkey, Rotation::cur()) + one.clone()),
                    q_t * (m.query_advice(supplier[1], Rotation::cur()) + one),
                ),
            ]
        });

        // Nation: (l_nationkey, l_nationname) in the selected nations
        meta.lookup_any("attach nation name from nation", |m| {
            let one = Expression::Constant(F::ONE);
            let q_in = m.query_selector(q_lkp_nation) * m.query_advice(cflag[R_LINE], Rotation::cur());
            let q_t = m.query_selector(q_row[R_NAT]) * m.query_advice(cflag[R_NAT], Rotation::cur());
            vec![
                (
                    q_in.clone() * (m.query_advice(l_nationkey, Rotation::cur()) + one.clone()),
                    q_t.clone() * (m.query_advice(nation[0], Rotation::cur()) + one.clone()),
                ),
                (
                    q_in * (m.query_advice(l_nationname, Rotation::cur()) + one.clone()),
                    q_t * (m.query_advice(nation[1], Rotation::cur()) + one),
                ),
            ]
        });

        // ---------------- Cardinality Preservation Check (condition (4)) ----------------
        // Join tree: lineitem is the root with part, orders, partsupp and
        // supplier as children, and nation is the child of supplier. The
        // supplier -> nation edge is a real tree edge here: the lookup above
        // collapses it into a 2-hop check off the lineitem row, but the
        // propagation below fetches nation's per-key sums on supplier's own rows
        // and folds them into supplier's multiplicities.
        //
        // One fixed column serves every Lt chip of the check, so the whole
        // check costs a single u8 range table.
        let cp_u8 = meta.fixed_column();

        let q_cp_one = meta.selector();
        let q_cp_nat = meta.selector();
        let q_cp_sup = meta.selector();
        let q_cp_mu = meta.selector();

        // Only part carries a predicate, so the input channel of the other
        // leaves is the constant 1 and needs a column pinned to it.
        let cp_one = meta.advice_column();
        meta.create_gate("cp: unfiltered leaf multiplicity is 1", |m| {
            let q = m.query_selector(q_cp_one);
            vec![q * (m.query_advice(cp_one, Rotation::cur()) - Expression::Constant(F::ONE))]
        });

        // n_nationkey can be 0, which is the dummy key of the gadget's tables,
        // so both sides of the supplier -> nation edge are shifted.
        let n_key_cp = meta.advice_column();
        meta.create_gate("cp: shifted nation key", |m| {
            let q = m.query_selector(q_cp_nat);
            let k = m.query_advice(n_key_cp, Rotation::cur());
            let b = m.query_advice(nation[0], Rotation::cur());
            vec![q * (k - (b + Expression::Constant(F::from(NAT_SHIFT))))]
        });
        let s_nkey_cp = meta.advice_column();
        meta.create_gate("cp: shifted supplier nation key", |m| {
            let q = m.query_selector(q_cp_sup);
            let k = m.query_advice(s_nkey_cp, Rotation::cur());
            let b = m.query_advice(supplier[1], Rotation::cur());
            vec![q * (k - (b + Expression::Constant(F::from(NAT_SHIFT))))]
        });

        // the partsupp key on the base lineitem rows, packed exactly the way
        // "ps_key packing" packs it on the join rows
        let l_ps_key_cp = meta.advice_column();
        meta.create_gate("cp: ps_key packing on base lineitem rows", |m| {
            let q = m.query_selector(q_cp_mu);
            let lp = m.query_advice(lineitem[1], Rotation::cur());
            let ls = m.query_advice(lineitem[2], Rotation::cur());
            let key = m.query_advice(l_ps_key_cp, Rotation::cur());
            let shift = Expression::Constant(F::from(PS_SHIFT));
            vec![q * (key - (lp * shift + ls))]
        });

        // Partsupp: (ps_key, supplycost) in the selected partsupps. It reads
        // the same packed key column the propagation uses.
        meta.lookup_any("attach supplycost from partsupp", |m| {
            let one = Expression::Constant(F::ONE);
            let q_in = m.query_selector(q_lkp_partsupp) * m.query_advice(cflag[R_LINE], Rotation::cur());
            let q_t = m.query_selector(q_row[R_PS]) * m.query_advice(cflag[R_PS], Rotation::cur());
            vec![
                (
                    q_in.clone() * (m.query_advice(l_ps_key_cp, Rotation::cur()) + one.clone()),
                    q_t.clone() * (m.query_advice(partsupp[0], Rotation::cur()) + one.clone()),
                ),
                (
                    q_in * (m.query_advice(l_supplycost, Rotation::cur()) + one.clone()),
                    q_t * (m.query_advice(partsupp[1], Rotation::cur()) + one),
                ),
            ]
        });

        // -------- (2) Pairwise Consistency on every tree edge --------
        // pi_K(R_i^c) == pi_K(R_j^c), as two mutual Membership Checks per edge,
        // each looking one endpoint's key column up directly in the other
        // endpoint's. Both sides read
        //
        //     q_row * c(t) * (t[K] + 1),
        //
        // so a deselected row and a row past the relation both read 0, 0 is in
        // every table, and the containment is over the selected keys only.
        //
        // This is where the realization differs most from `q9_obj.rs`. There the
        // gating factor was a Selector whose enabled range was the clean prefix
        // of a partition group, which the file documents as a KNOWN DESIGN GAP:
        // the ranges were related to the partition only by honest assignment, so
        // a malicious circuit GENERATOR could publish a vk with all six ranges
        // empty and make the condition vacuous while (1) and (3) still accepted
        // the all-clean selection. Here the gating factor is the committed
        // selector COLUMN, the same one the Cardinality Preservation Check reads
        // on the same rows, and the fixed selectors mark nothing but the
        // relation's real rows, a function of |R_i| alone. The gap is closed.
        //
        // The three derived key columns are the ones the propagation already
        // needs, so no column is added for this condition at all.
        {
            let mut pw_edge = |name: &'static str,
                               q_in: Selector,
                               c_in: Column<Advice>,
                               k_in: Column<Advice>,
                               q_tb: Selector,
                               c_tb: Column<Advice>,
                               k_tb: Column<Advice>| {
                meta.lookup_any(name, move |m| {
                    let one = Expression::Constant(F::ONE);
                    let lhs = m.query_selector(q_in)
                        * m.query_advice(c_in, Rotation::cur())
                        * (m.query_advice(k_in, Rotation::cur()) + one.clone());
                    let rhs = m.query_selector(q_tb)
                        * m.query_advice(c_tb, Rotation::cur())
                        * (m.query_advice(k_tb, Rotation::cur()) + one);
                    vec![(lhs, rhs)]
                });
            };

            // edge lineitem - part, on l_partkey = p_partkey
            pw_edge("pw: lineitem^c partkey in part^c",
                q_row[R_LINE], cflag[R_LINE], lineitem[1],
                q_row[R_PART], cflag[R_PART], part[0]);
            pw_edge("pw: part^c partkey in lineitem^c",
                q_row[R_PART], cflag[R_PART], part[0],
                q_row[R_LINE], cflag[R_LINE], lineitem[1]);

            // edge lineitem - orders, on l_orderkey = o_orderkey
            pw_edge("pw: lineitem^c orderkey in orders^c",
                q_row[R_LINE], cflag[R_LINE], lineitem[0],
                q_row[R_ORD], cflag[R_ORD], orders[0]);
            pw_edge("pw: orders^c orderkey in lineitem^c",
                q_row[R_ORD], cflag[R_ORD], orders[0],
                q_row[R_LINE], cflag[R_LINE], lineitem[0]);

            // edge lineitem - partsupp, on the packed (partkey, suppkey)
            pw_edge("pw: lineitem^c ps_key in partsupp^c",
                q_row[R_LINE], cflag[R_LINE], l_ps_key_cp,
                q_row[R_PS], cflag[R_PS], partsupp[0]);
            pw_edge("pw: partsupp^c ps_key in lineitem^c",
                q_row[R_PS], cflag[R_PS], partsupp[0],
                q_row[R_LINE], cflag[R_LINE], l_ps_key_cp);

            // edge lineitem - supplier, on l_suppkey = s_suppkey
            pw_edge("pw: lineitem^c suppkey in supplier^c",
                q_row[R_LINE], cflag[R_LINE], lineitem[2],
                q_row[R_SUPP], cflag[R_SUPP], supplier[0]);
            pw_edge("pw: supplier^c suppkey in lineitem^c",
                q_row[R_SUPP], cflag[R_SUPP], supplier[0],
                q_row[R_LINE], cflag[R_LINE], lineitem[2]);

            // edge supplier - nation, on the shifted s_nationkey = n_nationkey.
            // The NAT_SHIFT of both columns cancels, so the containment is the
            // one on the raw nation keys.
            pw_edge("pw: supplier^c nationkey in nation^c",
                q_row[R_SUPP], cflag[R_SUPP], s_nkey_cp,
                q_row[R_NAT], cflag[R_NAT], n_key_cp);
            pw_edge("pw: nation^c nationkey in supplier^c",
                q_row[R_NAT], cflag[R_NAT], n_key_cp,
                q_row[R_SUPP], cflag[R_SUPP], s_nkey_cp);
        }

        // Leaves. Part's input channel is its predicate bit and its clean
        // channel is the bound indicator keep * c on the input side of the
        // Conservation Check; the other three leaves are unfiltered.
        let cp_agg_p = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            part[0], // p_partkey
            p_keep,
            cflag[R_PART],
            MAX_SENTINEL,
        );
        let cp_agg_o = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            orders[0], // o_orderkey
            cp_one,
            cflag[3],
            MAX_SENTINEL,
        );
        let cp_agg_ps = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            partsupp[0], // packed ps_key
            cp_one,
            cflag[4],
            MAX_SENTINEL,
        );
        let cp_agg_n = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            n_key_cp, // n_nationkey + NAT_SHIFT
            cp_one,
            cflag[2],
            MAX_SENTINEL,
        );

        // Supplier is internal: fetch nation's sums on supplier's rows and fold
        // them into the two multiplicities supplier carries as a child of
        // lineitem.
        let cp_join_n = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, s_nkey_cp);
        wire_cp_edge(meta, &cp_join_n, &cp_agg_n, s_nkey_cp);

        let v_all_s = meta.advice_column();
        let v_cln_s = meta.advice_column();
        {
            let s_all_n = cp_join_n.s_all;
            let s_cln_n = cp_join_n.s_cln;
            let cln_s = cflag[1];
            meta.create_gate(
                "cp: supplier multiplicities from the nation edge",
                move |m| {
                    let q = m.query_selector(q_cp_sup);
                    let all = m.query_advice(v_all_s, Rotation::cur())
                        - m.query_advice(s_all_n, Rotation::cur());
                    let cln = m.query_advice(v_cln_s, Rotation::cur())
                        - m.query_advice(cln_s, Rotation::cur())
                            * m.query_advice(s_cln_n, Rotation::cur());
                    vec![q.clone() * all, q * cln]
                },
            );
        }
        let cp_agg_s = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            supplier[0], // s_suppkey
            v_all_s,
            v_cln_s,
            MAX_SENTINEL,
        );

        // Parent side, on the base rows of lineitem.
        let cp_join_p = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, lineitem[1]);
        let cp_join_o = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, lineitem[0]);
        let cp_join_s = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, lineitem[2]);
        let cp_join_ps = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, l_ps_key_cp);
        wire_cp_edge(meta, &cp_join_p, &cp_agg_p, lineitem[1]);
        wire_cp_edge(meta, &cp_join_o, &cp_agg_o, lineitem[0]);
        wire_cp_edge(meta, &cp_join_s, &cp_agg_s, lineitem[2]);
        wire_cp_edge(meta, &cp_join_ps, &cp_agg_ps, l_ps_key_cp);

        // Root multiplicities and the single equality that compares the two
        // join cardinalities. The root has four children, so each channel folds
        // its four sums through two intermediate columns and no gate exceeds
        // degree 4.
        let cp_root = configure_cp_root::<F>(meta);
        let t_all = meta.advice_column();
        let u_all = meta.advice_column();
        let t_cln = meta.advice_column();
        let u_cln = meta.advice_column();
        {
            let (s_all_p, s_cln_p) = (cp_join_p.s_all, cp_join_p.s_cln);
            let (s_all_o, s_cln_o) = (cp_join_o.s_all, cp_join_o.s_cln);
            let (s_all_s, s_cln_s) = (cp_join_s.s_all, cp_join_s.s_cln);
            let (s_all_ps, s_cln_ps) = (cp_join_ps.s_all, cp_join_ps.s_cln);
            let cln_l = cflag[5];
            let mu_all = cp_root.mu_all;
            let mu_cln = cp_root.mu_cln;
            meta.create_gate("cp: root multiplicities over lineitem", move |m| {
                let q = m.query_selector(q_cp_mu);

                let ta = m.query_advice(t_all, Rotation::cur());
                let ua = m.query_advice(u_all, Rotation::cur());
                let tc = m.query_advice(t_cln, Rotation::cur());
                let uc = m.query_advice(u_cln, Rotation::cur());

                vec![
                    q.clone()
                        * (ta.clone()
                            - m.query_advice(s_all_p, Rotation::cur())
                                * m.query_advice(s_all_o, Rotation::cur())),
                    q.clone()
                        * (ua.clone()
                            - m.query_advice(s_all_s, Rotation::cur())
                                * m.query_advice(s_all_ps, Rotation::cur())),
                    q.clone() * (m.query_advice(mu_all, Rotation::cur()) - ta * ua),
                    q.clone()
                        * (tc.clone()
                            - m.query_advice(s_cln_p, Rotation::cur())
                                * m.query_advice(s_cln_o, Rotation::cur())),
                    q.clone()
                        * (uc.clone()
                            - m.query_advice(s_cln_s, Rotation::cur())
                                * m.query_advice(s_cln_ps, Rotation::cur())),
                    q * (m.query_advice(mu_cln, Rotation::cur())
                        - m.query_advice(cln_l, Rotation::cur()) * tc * uc),
                ]
            });
        }

        // ---------- profit = [nationname, year, amount] ----------
        let profit = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        // The profit triple of a SELECTED row is its attached (nation, year,
        // amount); of a deselected row it is the canonical PAD triple. That one
        // masked gate does the work `q9_obj.rs` split between "profit row tie"
        // over the join prefix and "profit pad row" over the complement, and it
        // no longer depends on the clean rows being laid out first.
        //
        // The five attach columns and `amount` stay free on a deselected row,
        // and that is safe for the same reason as before: nothing else reads
        // them, and the triple the shuffle carries into the group-by is pinned.
        meta.create_gate("profit row: selected? attached : PAD", |m| {
            let q = m.query_selector(q_amount);
            let c = m.query_advice(cflag[R_LINE], Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let drop = one - c.clone();

            let pn = m.query_advice(profit[0], Rotation::cur());
            let py = m.query_advice(profit[1], Rotation::cur());
            let pa = m.query_advice(profit[2], Rotation::cur());

            vec![
                q.clone()
                    * (pn
                        - (c.clone() * m.query_advice(l_nationname, Rotation::cur())
                            + drop.clone() * Expression::Constant(F::from(PAD_PROF_N)))),
                q.clone()
                    * (py
                        - (c.clone() * m.query_advice(l_year, Rotation::cur())
                            + drop.clone() * Expression::Constant(F::from(PAD_PROF_Y)))),
                q * (pa
                    - (c * m.query_advice(amount, Rotation::cur())
                        + drop * Expression::Constant(F::from(PAD_PROF_A)))),
            ]
        });

        // The gate above is the only thing that ever reads the profit triple,
        // and it runs under `q_amount`, which is enabled on the join rows
        // [0, join_len) only. The rows from `join_len` to `lineitem.len()` are
        // therefore free advice, yet `perm_profit` below carries every one of
        // them into `profit_sorted`, and from there into the group-by
        // accumulator: the shuffle fixes the multiset and nothing else, so a
        // forged triple is an arbitrary extra summand. Re-using an existing
        // (nation, year) inflates that group's sum, and a fresh pair invents a
        // group. Pin those rows to the canonical PAD triple, which is what the
        // honest prover already writes, rather than gating them out of the
        // shuffle: the shuffle is Conservation for the profit table and must
        // keep covering every row.
        //
        // The five attach columns and `amount` are free on the same rows, but
        // they are read only by `q_amount`-gated gates and `q_lkp_*`-gated
        // lookups, so with the profit triple pinned there is nothing left for
        // them to move.
        let q_prof_pad = meta.selector();
        meta.create_gate("profit pad row", |m| {
            let q = m.query_selector(q_prof_pad);
            let pn = m.query_advice(profit[0], Rotation::cur());
            let py = m.query_advice(profit[1], Rotation::cur());
            let pa = m.query_advice(profit[2], Rotation::cur());
            vec![
                q.clone() * (pn - Expression::Constant(F::from(PAD_PROF_N))),
                q.clone() * (py - Expression::Constant(F::from(PAD_PROF_Y))),
                q * (pa - Expression::Constant(F::from(PAD_PROF_A))),
            ]
        });

        // profit_sorted permutation
        let profit_sorted = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let q_perm_pf_in = meta.complex_selector();
        let q_perm_pf_out = meta.complex_selector();
        let perm_profit = PermAnyChip::configure(
            meta,
            q_perm_pf_in,
            q_perm_pf_out,
            profit.clone(),
            profit_sorted.clone(),
        );

        // ---------- ORDER BY on profit_sorted: nation ASC, year DESC ----------
        let q_sort_profit = meta.selector();

        let aux_n_eq = meta.advice_column();
        let aux_y_eq = meta.advice_column();

        let iz_nation_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort_profit),
            |m| {
                m.query_advice(profit_sorted[0], Rotation::cur())
                    - m.query_advice(profit_sorted[0], Rotation::next())
            },
            aux_n_eq,
        );
        let iz_year_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort_profit),
            |m| {
                m.query_advice(profit_sorted[1], Rotation::cur())
                    - m.query_advice(profit_sorted[1], Rotation::next())
            },
            aux_y_eq,
        );

        let lt_nation_cur_next = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort_profit),
            |m| m.query_advice(profit_sorted[0], Rotation::cur()),
            |m| m.query_advice(profit_sorted[0], Rotation::next()),
        );
        let lt_year_next_cur = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort_profit),
            |m| m.query_advice(profit_sorted[1], Rotation::next()),
            |m| m.query_advice(profit_sorted[1], Rotation::cur()),
        );

        meta.create_gate("profit_sorted ORDER BY nation ASC, year DESC", |m| {
            let q = m.query_selector(q_sort_profit);

            let n_lt = lt_nation_cur_next.is_lt(m, None);
            let n_eq = iz_nation_eq.expr();

            let y_gt = lt_year_next_cur.is_lt(m, None); // next < cur
            let y_eq = iz_year_eq.expr();
            let y_ge = y_gt + y_eq;

            vec![q * (n_lt + n_eq * y_ge - Expression::Constant(F::ONE))]
        });

        // ---------- GROUP BY (nation,year) sum(amount) ----------
        let q_first = meta.selector();
        let q_accu = meta.selector();
        let q_line = meta.selector();

        let gkey = meta.advice_column();
        let run_sum = meta.advice_column();

        // pack gkey = nation * PS_SHIFT + year (PS_SHIFT is enough)
        meta.create_gate("gkey packing", |m| {
            let q = m.query_selector(q_line);
            let n = m.query_advice(profit_sorted[0], Rotation::cur());
            let y = m.query_advice(profit_sorted[1], Rotation::cur());
            let g = m.query_advice(gkey, Rotation::cur());
            let shift = Expression::Constant(F::from(PS_SHIFT));
            vec![q * (g - (n * shift + y))]
        });

        let aux_sp = meta.advice_column();
        let aux_sn = meta.advice_column();

        let iz_same_prev = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_accu),
            |m| m.query_advice(gkey, Rotation::cur()) - m.query_advice(gkey, Rotation::prev()),
            aux_sp,
        );
        let iz_same_next = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_line),
            |m| m.query_advice(gkey, Rotation::next()) - m.query_advice(gkey, Rotation::cur()),
            aux_sn,
        );

        // `iz_same_next` reads gkey at Rotation::next, so the last real row,
        // n-1, reads the sentinel cell at row n. "gkey packing" runs under
        // `q_line`, which covers rows [0, n) only, and the profit ORDER BY gate
        // covers [0, n-1), so nothing constrained that cell. A prover setting it
        // equal to gkey[n-1] makes `iz_same_next` report "same group" on the last
        // real row; the emit gate then writes the PAD triple instead of that
        // group's sums and the group disappears from the result, while both
        // ORDER BY ladders and conditions (7), (9) and (10) stay satisfied.
        // This is the same attack `card_preserve::configure_cp_agg` closes with
        // its own `q_sentinel`, and the same fix: pin the boundary cell.
        //
        // The pinned value is 0, which is what the honest prover writes and is
        // no real group key: gkey = nation_hash * PS_SHIFT + year, a real nation
        // hash is nonzero and the padding rows carry PAD_PROF_N, so every row of
        // `profit_sorted` has gkey >= PS_SHIFT. Pinning the VALUE rather than
        // demanding a strict increase across the last comparison is what keeps
        // the trailing PAD-keyed rows able to form their own group: the emit gate
        // writes (PAD_PROF_N, PAD_PROF_Y, 0) there, which is exactly the result
        // pad triple, so that group is dropped by the res shuffle rather than
        // wrongly emitted.
        let q_gkey_sentinel = meta.selector();
        meta.create_gate("gkey sentinel", |m| {
            let q = m.query_selector(q_gkey_sentinel);
            vec![q * m.query_advice(gkey, Rotation::cur())]
        });

        // The pin above is only as good as "no real gkey is 0", which holds for
        // any real nation table but is a statement about prover-supplied advice.
        // This second gate makes the last row a group end unconditionally, which
        // is what it is: `iz_same_next.expr()` is 1 - value * value_inv, so
        // forcing it to 0 forces value * value_inv = 1, hence gkey[n] != gkey[n-1]
        // whatever the two cells hold. One selector, one degree-3 gate, no new
        // column. The only witness it rules out is a table whose very last
        // profit row has gkey exactly 0, which the honest prover never produces.
        let q_gkey_last = meta.selector();
        meta.create_gate("last profit row ends its group", |m| {
            let q = m.query_selector(q_gkey_last);
            vec![q * iz_same_next.expr()]
        });

        // run_sum[0] = amount[0]
        meta.create_gate("run_sum_first", |m| {
            let q = m.query_selector(q_first);
            let rs = m.query_advice(run_sum, Rotation::cur());
            let a = m.query_advice(profit_sorted[2], Rotation::cur());
            vec![q * (rs - a)]
        });

        // run_sum[i] = same_prev*run_sum[i-1] + amount[i]
        meta.create_gate("run_sum_accu", |m| {
            let q = m.query_selector(q_accu);
            let same = iz_same_prev.expr();
            let rs_cur = m.query_advice(run_sum, Rotation::cur());
            let rs_prev = m.query_advice(run_sum, Rotation::prev());
            let a = m.query_advice(profit_sorted[2], Rotation::cur());
            vec![q * (rs_cur - (same * rs_prev + a))]
        });

        // emit res_pad only at last row of group
        let res_pad = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        meta.create_gate("emit res_pad (group last)", |m| {
            let q = m.query_selector(q_line);
            let one = Expression::Constant(F::ONE);
            let is_last = one.clone() - iz_same_next.expr();
            let not_last = one.clone() - is_last.clone();

            let n = m.query_advice(profit_sorted[0], Rotation::cur());
            let y = m.query_advice(profit_sorted[1], Rotation::cur());
            let rs = m.query_advice(run_sum, Rotation::cur());

            let out_n = m.query_advice(res_pad[0], Rotation::cur());
            let out_y = m.query_advice(res_pad[1], Rotation::cur());
            let out_s = m.query_advice(res_pad[2], Rotation::cur());

            let pad_n = Expression::Constant(F::from(PAD_RES_N));
            let pad_y = Expression::Constant(F::from(PAD_RES_Y));
            let pad_s = Expression::Constant(F::from(PAD_RES_S));

            vec![
                q.clone() * (out_n - (is_last.clone() * n + not_last.clone() * pad_n)),
                q.clone() * (out_y - (is_last.clone() * y + not_last.clone() * pad_y)),
                q * (out_s - (is_last * rs + not_last * pad_s)),
            ]
        });

        // perm res_pad -> res_sorted
        let res_sorted = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let q_perm_r_in = meta.complex_selector();
        let q_perm_r_out = meta.complex_selector();
        let perm_res = PermAnyChip::configure(
            meta,
            q_perm_r_in,
            q_perm_r_out,
            res_pad.clone(),
            res_sorted.clone(),
        );

        // ORDER BY on res_sorted: nation ASC, year DESC
        let q_sort_res = meta.selector();

        let aux_rn = meta.advice_column();
        let aux_ry = meta.advice_column();

        let iz_res_nation_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| {
                m.query_advice(res_sorted[0], Rotation::cur())
                    - m.query_advice(res_sorted[0], Rotation::next())
            },
            aux_rn,
        );
        let iz_res_year_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| {
                m.query_advice(res_sorted[1], Rotation::cur())
                    - m.query_advice(res_sorted[1], Rotation::next())
            },
            aux_ry,
        );

        let lt_res_nation_cur_next = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| m.query_advice(res_sorted[0], Rotation::cur()),
            |m| m.query_advice(res_sorted[0], Rotation::next()),
        );
        let lt_res_year_next_cur = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| m.query_advice(res_sorted[1], Rotation::next()),
            |m| m.query_advice(res_sorted[1], Rotation::cur()),
        );

        meta.create_gate("res_sorted ORDER BY nation ASC, year DESC", |m| {
            let q = m.query_selector(q_sort_res);

            let n_lt = lt_res_nation_cur_next.is_lt(m, None);
            let n_eq = iz_res_nation_eq.expr();

            let y_gt = lt_res_year_next_cur.is_lt(m, None);
            let y_eq = iz_res_year_eq.expr();
            let y_ge = y_gt + y_eq;

            vec![q * (n_lt + n_eq * y_ge - Expression::Constant(F::ONE))]
        });

        TestCircuitConfig {
            q_part_pred,

            part,
            supplier,
            nation,
            orders,
            partsupp,
            lineitem,

            cond,
            q_cond_eq,
            row_idx,
            cons,
            cflag,
            q_row,
            p_keep,

            l_supplycost,
            l_year,
            l_nationkey,
            l_nationname,

            q_amount,
            amount,

            profit,
            profit_sorted,
            perm_profit,

            q_sort_profit,
            lt_nation_cur_next,
            lt_year_next_cur,
            iz_nation_eq,
            iz_year_eq,

            q_first,
            q_accu,
            q_line,
            q_gkey_sentinel,
            q_gkey_last,

            gkey,
            run_sum,
            iz_same_prev,
            iz_same_next,

            res_pad,
            res_sorted,
            perm_res,

            q_sort_res,
            lt_res_nation_cur_next,
            lt_res_year_next_cur,
            iz_res_nation_eq,
            iz_res_year_eq,

            q_lkp_orders,
            q_lkp_supplier,
            q_lkp_nation,
            q_lkp_partsupp,


            instance,
            instance_test,

            iz_part,

            cp_one,
            n_key_cp,
            s_nkey_cp,
            l_ps_key_cp,
            v_all_s,
            v_cln_s,
            t_all,
            u_all,
            t_cln,
            u_cln,

            cp_agg_p,
            cp_agg_o,
            cp_agg_s,
            cp_agg_ps,
            cp_agg_n,

            cp_join_p,
            cp_join_o,
            cp_join_s,
            cp_join_ps,
            cp_join_n,

            cp_root,
            q_cp_one,
            q_cp_nat,
            q_cp_sup,
            q_cp_mu,

        }
    }

    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        part: Vec<Vec<u64>>,     // [p_partkey, p_name_hash]
        supplier: Vec<Vec<u64>>, // [s_suppkey, s_nationkey]
        nation: Vec<Vec<u64>>,   // [n_nationkey, n_name_hash]
        orders: Vec<Vec<u64>>,   // [o_orderkey, o_year]
        partsupp: Vec<Vec<u64>>, // [ps_key, ps_supplycost]
        lineitem: Vec<Vec<u64>>, // [l_orderkey,l_partkey,l_suppkey,l_qty,l_ext,l_disc]
        cond_hash: u64,          // :1 (simplified)
    ) -> Result<AssignedCell<F, F>, Error> {
        // chips
        // let iz_part_chip = IsZeroChip::construct(self.config.iz_part.clone());
        // let iz_part_chip2 = iz_part_chip.clone();
        // drop(iz_part_chip);

        let lt_n_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_nation_cur_next.clone());
        lt_n_chip.load(layouter)?;
        let lt_y_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_year_next_cur.clone());
        lt_y_chip.load(layouter)?;
        let lt_rn_chip =
            LtChip::<F, NUM_BYTES>::construct(self.config.lt_res_nation_cur_next.clone());
        lt_rn_chip.load(layouter)?;
        let lt_ry_chip =
            LtChip::<F, NUM_BYTES>::construct(self.config.lt_res_year_next_cur.clone());
        lt_ry_chip.load(layouter)?;

        // Every Lt chip of the Cardinality Preservation Check shares one u8
        // fixed column, so a single load covers the whole check.
        LtChip::<F, NUM_BYTES>::construct(self.config.cp_agg_p.lt_key_cur_next).load(layouter)?;

        let iz_n_eq_chip = IsZeroChip::construct(self.config.iz_nation_eq.clone());
        let iz_y_eq_chip = IsZeroChip::construct(self.config.iz_year_eq.clone());
        let iz_sp_chip = IsZeroChip::construct(self.config.iz_same_prev.clone());
        let iz_sn_chip = IsZeroChip::construct(self.config.iz_same_next.clone());
        let iz_rn_eq_chip = IsZeroChip::construct(self.config.iz_res_nation_eq.clone());
        let iz_ry_eq_chip = IsZeroChip::construct(self.config.iz_res_year_eq.clone());

        // ---------------- witness preprocessing ----------------
        // 1) Part filter (simplified LIKE): keep if p_name_hash == cond_hash
        let mut p_keep: Vec<u64> = vec![0; part.len()];
        for i in 0..part.len() {
            p_keep[i] = if part[i][1] == cond_hash { 1 } else { 0 };
        }
        let p_join: Vec<Vec<u64>> = part
            .iter()
            .cloned()
            .zip(p_keep.iter().cloned())
            .filter(|(_, k)| *k == 1)
            .map(|(r, _)| r)
            .collect();

        // p_filt_pad and p_join_pad are built further down, once the clean
        // instance is known: both carry the clean indicator as a third column.

        // lookup maps
        let partkeys_keep: HashSet<u64> = p_join.iter().map(|r| r[0]).collect();

        let sup_map: HashMap<u64, u64> = supplier.iter().map(|r| (r[0], r[1])).collect();
        let nat_map: HashMap<u64, u64> = nation.iter().map(|r| (r[0], r[1])).collect();
        let ord_map: HashMap<u64, u64> = orders.iter().map(|r| (r[0], r[1])).collect();
        let ps_map: HashMap<u64, u64> = partsupp.iter().map(|r| (r[0], r[1])).collect();

        // 2) contributing lineitems
        // keep if:
        // - l_partkey in partkeys_keep
        // - l_suppkey in supplier
        // - l_orderkey in orders
        // - partsupp has (l_partkey,l_suppkey)
        // - supplier.nationkey in nation
        let mut l_join: Vec<Vec<u64>> = vec![];
        let mut l_dis: Vec<Vec<u64>> = vec![];
        let mut l_cln: Vec<u64> = vec![0; lineitem.len()];

        for (i, r) in lineitem.iter().enumerate() {
            let okey = r[0];
            let pkey = r[1];
            let skey = r[2];

            let ps_key = pkey * PS_SHIFT + skey;

            let ok = partkeys_keep.contains(&pkey)
                && sup_map.contains_key(&skey)
                && ord_map.contains_key(&okey)
                && ps_map.contains_key(&ps_key)
                && nat_map.contains_key(&sup_map[&skey]);

            if ok {
                l_cln[i] = 1;
                l_join.push(r.clone());
            } else {
                l_dis.push(r.clone());
            }
        }

        // test hook only: hide one joinable lineitem in the residual side. The
        // clean parts of the other five relations are derived from l_join below,
        // so the neighbours are re-reduced around the hidden tuple and the
        // partition still satisfies conditions (1)-(3).
        let tamper = HIDE_ONE_CLEAN_TUPLE.load(Ordering::Relaxed);
        if tamper && !l_join.is_empty() {
            let hidden = l_join.remove(0);
            l_dis.push(hidden);
            if let Some(slot) = l_cln.iter_mut().find(|f| **f == 1) {
                *slot = 0;
            }
        }

        // test hook only: skip the reduction and declare every real tuple clean.
        // The partition groups keep the row order they have below, only the
        // boundary between the clean and the residual section moves to the end
        // of the real rows and every real row's flag becomes 1. Conservation
        // still holds, and with c == pred on every row the clean channel of
        // condition (4) equals its input channel row by row, so the two root
        // sums agree for free. Only condition (3) can see this.
        let all_clean = MARK_ALL_CLEAN.load(Ordering::Relaxed);
        if all_clean {
            for f in l_cln.iter_mut() {
                *f = 1;
            }
        }

        // 3) contributing orders/suppliers/partsupp/nation subsets (for join_pad tables)
        let l_okeys: HashSet<u64> = l_join.iter().map(|r| r[0]).collect();
        let l_skeys: HashSet<u64> = l_join.iter().map(|r| r[2]).collect();
        let l_pskeys: HashSet<u64> = l_join.iter().map(|r| r[1] * PS_SHIFT + r[2]).collect();
        let s_nkeys: HashSet<u64> = l_skeys
            .iter()
            .filter_map(|sk| sup_map.get(sk).copied())
            .collect();

        let o_join: Vec<Vec<u64>> = orders
            .iter()
            .cloned()
            .filter(|r| l_okeys.contains(&r[0]))
            .collect();
        let s_join: Vec<Vec<u64>> = supplier
            .iter()
            .cloned()
            .filter(|r| l_skeys.contains(&r[0]))
            .collect();
        let ps_join: Vec<Vec<u64>> = partsupp
            .iter()
            .cloned()
            .filter(|r| l_pskeys.contains(&r[0]))
            .collect();
        let n_join: Vec<Vec<u64>> = nation
            .iter()
            .cloned()
            .filter(|r| s_nkeys.contains(&r[0]))
            .collect();

        // Build join_pad tables for others by partitioning base into join/disjoin
        // disjoin = base - join (by key membership)
        let o_join_keys: HashSet<u64> = o_join.iter().map(|r| r[0]).collect();
        let s_join_keys: HashSet<u64> = s_join.iter().map(|r| r[0]).collect();
        let n_join_keys: HashSet<u64> = n_join.iter().map(|r| r[0]).collect();
        let ps_join_keys: HashSet<u64> = ps_join.iter().map(|r| r[0]).collect();

        let o_dis: Vec<Vec<u64>> = orders
            .iter()
            .cloned()
            .filter(|r| !o_join_keys.contains(&r[0]))
            .collect();
        let s_dis: Vec<Vec<u64>> = supplier
            .iter()
            .cloned()
            .filter(|r| !s_join_keys.contains(&r[0]))
            .collect();
        let n_dis: Vec<Vec<u64>> = nation
            .iter()
            .cloned()
            .filter(|r| !n_join_keys.contains(&r[0]))
            .collect();
        let ps_dis: Vec<Vec<u64>> = partsupp
            .iter()
            .cloned()
            .filter(|r| !ps_join_keys.contains(&r[0]))
            .collect();

        // ---------------- clean/residual partition of part ----------------
        // The clean side of every relation is the fully reduced instance: the
        // tuples that extend to a full join result. For the five relations above
        // that is what o_join / s_join / ps_join / n_join already are. For part
        // it is the kept rows whose partkey is used by a clean lineitem row,
        // which is a subset of p_join, so p_join_pad becomes
        // [clean | residual | pad] instead of [kept | pad].
        let l_pkeys: HashSet<u64> = l_join.iter().map(|r| r[1]).collect();
        let (p_cln, p_res): (Vec<Vec<u64>>, Vec<Vec<u64>>) = p_join
            .iter()
            .cloned()
            .partition(|r| l_pkeys.contains(&r[0]));

        // ---------------- clean indicator per base row ----------------
        // A base row is clean iff it passes its predicate and its tuple went to
        // the clean side above, so the indicator marks exactly the occurrences
        // of R^c.
        // Under the all-clean hook every real tuple is declared clean. On part
        // that still means keep * 1, because the link gate pads the indicator
        // with 0 and so forces a row dropped by the predicate to stay dirty.
        let cln_p: Vec<u64> = part
            .iter()
            .zip(p_keep.iter())
            .map(|(r, &k)| (k == 1 && (all_clean || l_pkeys.contains(&r[0]))) as u64)
            .collect();
        let cln_s: Vec<u64> = supplier
            .iter()
            .map(|r| (all_clean || l_skeys.contains(&r[0])) as u64)
            .collect();
        let cln_n: Vec<u64> = nation
            .iter()
            .map(|r| (all_clean || s_nkeys.contains(&r[0])) as u64)
            .collect();
        let cln_o: Vec<u64> = orders
            .iter()
            .map(|r| (all_clean || l_okeys.contains(&r[0])) as u64)
            .collect();
        let cln_ps: Vec<u64> = partsupp
            .iter()
            .map(|r| (all_clean || l_pskeys.contains(&r[0])) as u64)
            .collect();

        // ---------------- the aggregation witness ----------------
        // On the committed lineitem rows: a selected row carries its attached
        // (nation, year, amount), a deselected row the canonical PAD triple.
        // The `amount` gate covers every row, so a deselected row still needs an
        // amount consistent with a supplycost of 0.
        let mut profit_u64: Vec<[u64; 3]> =
            vec![[PAD_PROF_N, PAD_PROF_Y, PAD_PROF_A]; lineitem.len()];
        let mut amount_u64: Vec<u64> = vec![0; lineitem.len()];

        for i in 0..lineitem.len() {
            let okey = lineitem[i][0];
            let pkey = lineitem[i][1];
            let skey = lineitem[i][2];
            let qty = lineitem[i][3];
            let ext = lineitem[i][4];
            let disc = lineitem[i][5];

            let cost = if l_cln[i] == 1 {
                ps_map[&(pkey * PS_SHIFT + skey)]
            } else {
                0
            };

            // amount_scaled = ext*(SCALE-disc) - cost*qty*SCALE
            //
            // KNOWN COMPLETENESS LIMIT, see the "amount = ..." gate: for a < 0
            // this stores F::from(2^64 - |a|) while the gate computes p - |a|,
            // so the gate is unsatisfiable and the honest prover cannot prove
            // such a row at all. It is not an under-constraint, and repairing it
            // needs a signed representation for `amount`, `profit[2]` and both
            // running sums. A deselected row has cost 0, so its amount is
            // ext*(SCALE-disc) >= 0 and never hits this.
            let a: i128 = (ext as i128) * ((SCALE as i128) - (disc as i128))
                - (cost as i128) * (qty as i128) * (SCALE as i128);
            amount_u64[i] = (a as i128 as u128) as u64;

            if l_cln[i] == 1 {
                let year = ord_map[&okey];
                let nname = nat_map[&sup_map[&skey]];
                profit_u64[i] = [nname, year, amount_u64[i]];
            }
        }

        // profit_sorted = sort by (nation asc, year desc), keep amount aligned
        let mut profit_sorted_u64: Vec<[u64; 3]> = profit_u64.clone();
        profit_sorted_u64.sort_by(|a, b| a[0].cmp(&b[0]).then(b[1].cmp(&a[1])));

        // compute running group sum + emit res_pad
        let n = profit_sorted_u64.len();
        let mut run_sum_u64: Vec<u64> = vec![0; n];
        let mut res_pad_u64: Vec<[u64; 3]> = vec![[PAD_RES_N, PAD_RES_Y, PAD_RES_S]; n];

        let mut acc: u128 = 0;
        let mut prev_key: Option<(u64, u64)> = None;

        for i in 0..n {
            let nat = profit_sorted_u64[i][0];
            let year = profit_sorted_u64[i][1];
            let amt = profit_sorted_u64[i][2] as u128;

            if prev_key == Some((nat, year)) {
                acc = acc.wrapping_add(amt);
            } else {
                acc = amt;
            }
            run_sum_u64[i] = acc as u64;

            let next = if i + 1 < n {
                (profit_sorted_u64[i + 1][0], profit_sorted_u64[i + 1][1])
            } else {
                (0, 0)
            };
            let is_last = next != (nat, year);

            if is_last && nat != PAD_PROF_N {
                res_pad_u64[i] = [nat, year, run_sum_u64[i]];
            }
            prev_key = Some((nat, year));
        }

        // res_sorted = bring non-pad group rows first, sorted (already sorted), then pads
        let mut groups: Vec<[u64; 3]> = res_pad_u64
            .iter()
            .copied()
            .filter(|r| r[0] != PAD_RES_N)
            .collect();
        groups.sort_by(|a, b| a[0].cmp(&b[0]).then(b[1].cmp(&a[1]))); // nation asc, year desc

        let mut res_sorted_u64: Vec<[u64; 3]> = vec![];
        res_sorted_u64.extend(groups.into_iter());
        while res_sorted_u64.len() < n {
            res_sorted_u64.push([PAD_RES_N, PAD_RES_Y, PAD_RES_S]);
        }

        // --------------- assign region ---------------
        layouter.assign_region(
            || "Q9 witness",
            |mut region| {
                let iz_part_chip = IsZeroChip::construct(self.config.iz_part.clone());
                // base tables
                for i in 0..part.len() {
                    self.config.q_part_pred.enable(&mut region, i)?;
                    iz_part_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(part[i][1]) - F::from(cond_hash)),
                    )?;
                    for j in 0..2 {
                        region.assign_advice(
                            || "part",
                            self.config.part[j],
                            i,
                            || Value::known(F::from(part[i][j])),
                        )?;
                    }
                    region.assign_advice(
                        || "cond",
                        self.config.cond,
                        i,
                        || Value::known(F::from(cond_hash)),
                    )?;
                    region.assign_advice(
                        || "p_keep",
                        self.config.p_keep,
                        i,
                        || Value::known(F::from(p_keep[i])),
                    )?;
                }
                // :1 is one value for the whole proof: compare each part row's
                // cond cell with the next one, over all but the last part row.
                for i in 0..part.len().saturating_sub(1) {
                    self.config.q_cond_eq.enable(&mut region, i)?;
                }

                for i in 0..supplier.len() {
                    for j in 0..2 {
                        region.assign_advice(
                            || "supplier",
                            self.config.supplier[j],
                            i,
                            || Value::known(F::from(supplier[i][j])),
                        )?;
                    }
                }
                for i in 0..nation.len() {
                    for j in 0..2 {
                        region.assign_advice(
                            || "nation",
                            self.config.nation[j],
                            i,
                            || Value::known(F::from(nation[i][j])),
                        )?;
                    }
                }
                for i in 0..orders.len() {
                    for j in 0..2 {
                        region.assign_advice(
                            || "orders",
                            self.config.orders[j],
                            i,
                            || Value::known(F::from(orders[i][j])),
                        )?;
                    }
                }
                for i in 0..partsupp.len() {
                    for j in 0..2 {
                        region.assign_advice(
                            || "partsupp",
                            self.config.partsupp[j],
                            i,
                            || Value::known(F::from(partsupp[i][j])),
                        )?;
                    }
                }
                for i in 0..lineitem.len() {
                    for j in 0..6 {
                        region.assign_advice(
                            || "lineitem",
                            self.config.lineitem[j],
                            i,
                            || Value::known(F::from(lineitem[i][j])),
                        )?;
                    }
                }

                // ---- clean indicator on the base side of every Conservation Check ----
                // On part the link gate turns this into keep * indicator on the
                // pad columns; the other five relations have no predicate, so
                // the shuffle binds the column itself.
                for (col, flags) in [
                    (self.config.cflag[0], &cln_p),
                    (self.config.cflag[1], &cln_s),
                    (self.config.cflag[2], &cln_n),
                    (self.config.cflag[3], &cln_o),
                    (self.config.cflag[4], &cln_ps),
                    (self.config.cflag[5], &l_cln),
                ] {
                    for (i, &f) in flags.iter().enumerate() {
                        region.assign_advice(|| "cflag", col, i, || Value::known(F::from(f)))?;
                    }
                }
                // Selector Check and Pairwise Consistency need no witness of
                // their own beyond the bits assigned above: the ten lookups run
                // between the committed key columns, gated by those bits, and
                // the three derived key columns are the ones the propagation
                // already writes. The fixed selectors below mark nothing but
                // each relation's real rows.
                for i in 0..part.len() {
                    self.config.q_row[R_PART].enable(&mut region, i)?;
                }
                for i in 0..supplier.len() {
                    self.config.q_row[R_SUPP].enable(&mut region, i)?;
                }
                for i in 0..nation.len() {
                    self.config.q_row[R_NAT].enable(&mut region, i)?;
                }
                for i in 0..orders.len() {
                    self.config.q_row[R_ORD].enable(&mut region, i)?;
                }
                for i in 0..partsupp.len() {
                    self.config.q_row[R_PS].enable(&mut region, i)?;
                }
                for i in 0..lineitem.len() {
                    self.config.q_row[R_LINE].enable(&mut region, i)?;
                    self.config.q_lkp_orders.enable(&mut region, i)?;
                    self.config.q_lkp_supplier.enable(&mut region, i)?;
                    self.config.q_lkp_nation.enable(&mut region, i)?;
                    self.config.q_lkp_partsupp.enable(&mut region, i)?;
                    self.config.q_amount.enable(&mut region, i)?;
                }

                // assign attached cols + amount + profit on the committed
                // lineitem rows. A deselected row carries zeros in the attach
                // columns and the amount its own gate then forces, and its
                // profit triple is the PAD triple the masked gate demands.
                for i in 0..lineitem.len() {
                    let okey = lineitem[i][0];
                    let pkey = lineitem[i][1];
                    let skey = lineitem[i][2];
                    let ps_key = pkey * PS_SHIFT + skey;

                    let selected = l_cln[i] == 1;
                    let (year, nkey, nname, cost) = if selected {
                        let year = ord_map[&okey];
                        let nkey = sup_map[&skey];
                        let nname = nat_map[&nkey];
                        (year, nkey, nname, ps_map[&ps_key])
                    } else {
                        // `q_amount` covers every row now, so the amount gate
                        // still has to hold here. With cost 0 it reduces to
                        // ext * (SCALE - disc), which is what `amount_of` writes.
                        (0, 0, 0, 0)
                    };

                    region.assign_advice(
                        || "l_year",
                        self.config.l_year,
                        i,
                        || Value::known(F::from(year)),
                    )?;
                    region.assign_advice(
                        || "l_nationkey",
                        self.config.l_nationkey,
                        i,
                        || Value::known(F::from(nkey)),
                    )?;
                    region.assign_advice(
                        || "l_nationname",
                        self.config.l_nationname,
                        i,
                        || Value::known(F::from(nname)),
                    )?;
                    region.assign_advice(
                        || "l_supplycost",
                        self.config.l_supplycost,
                        i,
                        || Value::known(F::from(cost)),
                    )?;
                    region.assign_advice(
                        || "amount",
                        self.config.amount,
                        i,
                        || Value::known(F::from(amount_u64[i])),
                    )?;
                    for j in 0..3 {
                        region.assign_advice(
                            || "profit",
                            self.config.profit[j],
                            i,
                            || Value::known(F::from(profit_u64[i][j])),
                        )?;
                    }
                }

                // profit_sorted assignment + perm selectors
                for i in 0..n {
                    self.config.perm_profit.q_perm1.enable(&mut region, i)?;
                    self.config.perm_profit.q_perm2.enable(&mut region, i)?;
                    region.assign_advice(
                        || "profit_sorted",
                        self.config.profit_sorted[0],
                        i,
                        || Value::known(F::from(profit_sorted_u64[i][0])),
                    )?;
                    region.assign_advice(
                        || "profit_sorted",
                        self.config.profit_sorted[1],
                        i,
                        || Value::known(F::from(profit_sorted_u64[i][1])),
                    )?;
                    region.assign_advice(
                        || "profit_sorted",
                        self.config.profit_sorted[2],
                        i,
                        || Value::known(F::from(profit_sorted_u64[i][2])),
                    )?;
                }

                // enable sort gate on profit_sorted rows 0..n-2
                for i in 0..n.saturating_sub(1) {
                    self.config.q_sort_profit.enable(&mut region, i)?;
                    // isZero eq helpers
                    iz_n_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(
                            F::from(profit_sorted_u64[i][0]) - F::from(profit_sorted_u64[i + 1][0]),
                        ),
                    )?;
                    iz_y_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(
                            F::from(profit_sorted_u64[i][1]) - F::from(profit_sorted_u64[i + 1][1]),
                        ),
                    )?;

                    lt_n_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(profit_sorted_u64[i][0])),
                        Value::known(F::from(profit_sorted_u64[i + 1][0])),
                    )?;
                    lt_y_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(profit_sorted_u64[i + 1][1])),
                        Value::known(F::from(profit_sorted_u64[i][1])),
                    )?;
                }

                // group helpers: gkey, run_sum, res_pad
                for i in 0..n {
                    self.config.q_line.enable(&mut region, i)?;
                    region.assign_advice(
                        || "gkey",
                        self.config.gkey,
                        i,
                        || {
                            let g = profit_sorted_u64[i][0] * PS_SHIFT + profit_sorted_u64[i][1];
                            Value::known(F::from(g))
                        },
                    )?;
                    region.assign_advice(
                        || "run_sum",
                        self.config.run_sum,
                        i,
                        || Value::known(F::from(run_sum_u64[i])),
                    )?;

                    region.assign_advice(
                        || "res_pad",
                        self.config.res_pad[0],
                        i,
                        || Value::known(F::from(res_pad_u64[i][0])),
                    )?;
                    region.assign_advice(
                        || "res_pad",
                        self.config.res_pad[1],
                        i,
                        || Value::known(F::from(res_pad_u64[i][1])),
                    )?;
                    region.assign_advice(
                        || "res_pad",
                        self.config.res_pad[2],
                        i,
                        || Value::known(F::from(res_pad_u64[i][2])),
                    )?;
                }

                // sentinel for Rotation::next on last row, pinned by
                // "gkey sentinel"; the last real row is additionally forced to
                // be a group end by "last profit row ends its group"
                region.assign_advice(
                    || "gkey_sentinel",
                    self.config.gkey,
                    n,
                    || Value::known(F::from(0u64)),
                )?;
                self.config.q_gkey_sentinel.enable(&mut region, n)?;
                if n > 0 {
                    self.config.q_gkey_last.enable(&mut region, n - 1)?;
                }

                if n > 0 {
                    self.config.q_first.enable(&mut region, 0)?;
                }
                for i in 1..n {
                    self.config.q_accu.enable(&mut region, i)?;
                }

                // same_prev/same_next
                for i in 1..n {
                    let cur = profit_sorted_u64[i][0] * PS_SHIFT + profit_sorted_u64[i][1];
                    let prev = profit_sorted_u64[i - 1][0] * PS_SHIFT + profit_sorted_u64[i - 1][1];
                    iz_sp_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(cur) - F::from(prev)),
                    )?;
                }
                // sentinel gkey at row n for same_next convenience: set to 0 by just assuming next reads 0? we didn’t allocate row n,
                // so instead only assign same_next for i=0..n-2 and handle last row by skipping.
                for i in 0..n.saturating_sub(1) {
                    let next = profit_sorted_u64[i + 1][0] * PS_SHIFT + profit_sorted_u64[i + 1][1];
                    let cur = profit_sorted_u64[i][0] * PS_SHIFT + profit_sorted_u64[i][1];
                    iz_sn_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(next) - F::from(cur)),
                    )?;
                }
                // for last row, set diff = 0- cur (forces is_last=1), ok
                if n > 0 {
                    let cur = profit_sorted_u64[n - 1][0] * PS_SHIFT + profit_sorted_u64[n - 1][1];
                    iz_sn_chip.assign(
                        &mut region,
                        n - 1,
                        Value::known(F::from(0u64) - F::from(cur)),
                    )?;
                }

                // res_sorted + perm selectors + sort gate
                for i in 0..n {
                    self.config.perm_res.q_perm1.enable(&mut region, i)?;
                    self.config.perm_res.q_perm2.enable(&mut region, i)?;
                    region.assign_advice(
                        || "res_sorted",
                        self.config.res_sorted[0],
                        i,
                        || Value::known(F::from(res_sorted_u64[i][0])),
                    )?;
                    region.assign_advice(
                        || "res_sorted",
                        self.config.res_sorted[1],
                        i,
                        || Value::known(F::from(res_sorted_u64[i][1])),
                    )?;
                    region.assign_advice(
                        || "res_sorted",
                        self.config.res_sorted[2],
                        i,
                        || Value::known(F::from(res_sorted_u64[i][2])),
                    )?;
                }
                for i in 0..n.saturating_sub(1) {
                    self.config.q_sort_res.enable(&mut region, i)?;
                    iz_rn_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(
                            F::from(res_sorted_u64[i][0]) - F::from(res_sorted_u64[i + 1][0]),
                        ),
                    )?;
                    iz_ry_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(
                            F::from(res_sorted_u64[i][1]) - F::from(res_sorted_u64[i + 1][1]),
                        ),
                    )?;

                    lt_rn_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(res_sorted_u64[i][0])),
                        Value::known(F::from(res_sorted_u64[i + 1][0])),
                    )?;
                    lt_ry_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(res_sorted_u64[i + 1][1])),
                        Value::known(F::from(res_sorted_u64[i][1])),
                    )?;
                }

                // ===================== (1) CONSERVATION CHECK =====================
                assign_row_index(
                    &mut region,
                    &self.config.row_idx,
                    [
                        part.len(),
                        supplier.len(),
                        nation.len(),
                        orders.len(),
                        partsupp.len(),
                        lineitem.len(),
                    ]
                    .into_iter()
                    .max()
                    .unwrap_or(0),
                )?;
                for (idx, (rows, flags)) in [
                    (&part, &cln_p),
                    (&supplier, &cln_s),
                    (&nation, &cln_n),
                    (&orders, &cln_o),
                    (&partsupp, &cln_ps),
                    (&lineitem, &l_cln),
                ]
                .into_iter()
                .enumerate()
                {
                    assign_conserve(&mut region, &self.config.cons[idx], rows, flags)?;
                }

                // ===================== CARDINALITY PRESERVATION CHECK =====================
                // condition (4) of the One-Pass OBJ: the two multiplicity
                // channels are propagated over the join tree rooted at lineitem
                // and their sums at the root are compared. Both channels run
                // over the same rows, the same sorted views and the same key
                // lookups, so the clean channel only costs its own column,
                // running sum and product.

                // the constant 1: the input channel of the three unfiltered
                // leaves orders, partsupp and nation
                let cp_one_rows = orders.len().max(partsupp.len()).max(nation.len());
                for i in 0..cp_one_rows {
                    self.config.q_cp_one.enable(&mut region, i)?;
                    region.assign_advice(
                        || "cp_one",
                        self.config.cp_one,
                        i,
                        || Value::known(F::ONE),
                    )?;
                }

                // nation, the leaf hanging under supplier, on its shifted key
                let cp_rows_n: Vec<[u64; 3]> = (0..nation.len())
                    .map(|i| [nation[i][0] + NAT_SHIFT, 1, cln_n[i]])
                    .collect();
                for i in 0..nation.len() {
                    self.config.q_cp_nat.enable(&mut region, i)?;
                    region.assign_advice(
                        || "n_key_cp",
                        self.config.n_key_cp,
                        i,
                        || Value::known(F::from(cp_rows_n[i][0])),
                    )?;
                }
                let cp_stage_n = build_cp_stage(&cp_rows_n, MAX_SENTINEL);
                assign_cp_agg(&mut region, &self.config.cp_agg_n, &cp_rows_n, &cp_stage_n)?;

                // supplier is internal: fetch nation's two sums on supplier's own
                // rows and fold them into the multiplicities supplier carries as
                // a child of lineitem
                let s_nkeys_cp: Vec<u64> = supplier.iter().map(|r| r[1] + NAT_SHIFT).collect();
                let fetched_n = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_n,
                    &s_nkeys_cp,
                    &cp_stage_n,
                    MAX_SENTINEL,
                )?;
                let mut cp_rows_s: Vec<[u64; 3]> = Vec::with_capacity(supplier.len());
                for i in 0..supplier.len() {
                    self.config.q_cp_sup.enable(&mut region, i)?;
                    region.assign_advice(
                        || "s_nkey_cp",
                        self.config.s_nkey_cp,
                        i,
                        || Value::known(F::from(s_nkeys_cp[i])),
                    )?;
                    let v_all = fetched_n[i].0;
                    let v_cln = cln_s[i] * fetched_n[i].1;
                    region.assign_advice(
                        || "v_all_s",
                        self.config.v_all_s,
                        i,
                        || Value::known(F::from(v_all)),
                    )?;
                    region.assign_advice(
                        || "v_cln_s",
                        self.config.v_cln_s,
                        i,
                        || Value::known(F::from(v_cln)),
                    )?;
                    cp_rows_s.push([supplier[i][0], v_all, v_cln]);
                }
                let cp_stage_s = build_cp_stage(&cp_rows_s, MAX_SENTINEL);
                assign_cp_agg(&mut region, &self.config.cp_agg_s, &cp_rows_s, &cp_stage_s)?;

                // the three remaining children of the root are leaves, so their
                // input channel is the predicate bit (1 where there is none) and
                // their clean channel is that bit times the bound indicator
                let cp_rows_p: Vec<[u64; 3]> = (0..part.len())
                    .map(|i| [part[i][0], p_keep[i], cln_p[i]])
                    .collect();
                let cp_rows_o: Vec<[u64; 3]> = (0..orders.len())
                    .map(|i| [orders[i][0], 1, cln_o[i]])
                    .collect();
                let cp_rows_ps: Vec<[u64; 3]> = (0..partsupp.len())
                    .map(|i| [partsupp[i][0], 1, cln_ps[i]])
                    .collect();

                let cp_stage_p = build_cp_stage(&cp_rows_p, MAX_SENTINEL);
                let cp_stage_o = build_cp_stage(&cp_rows_o, MAX_SENTINEL);
                let cp_stage_ps = build_cp_stage(&cp_rows_ps, MAX_SENTINEL);

                assign_cp_agg(&mut region, &self.config.cp_agg_p, &cp_rows_p, &cp_stage_p)?;
                assign_cp_agg(&mut region, &self.config.cp_agg_o, &cp_rows_o, &cp_stage_o)?;
                assign_cp_agg(
                    &mut region,
                    &self.config.cp_agg_ps,
                    &cp_rows_ps,
                    &cp_stage_ps,
                )?;

                // parent side of the four root edges, on the base lineitem rows
                let l_pkeys_all: Vec<u64> = lineitem.iter().map(|r| r[1]).collect();
                let l_okeys_all: Vec<u64> = lineitem.iter().map(|r| r[0]).collect();
                let l_skeys_all: Vec<u64> = lineitem.iter().map(|r| r[2]).collect();
                let l_pskeys_all: Vec<u64> =
                    lineitem.iter().map(|r| r[1] * PS_SHIFT + r[2]).collect();

                let fetched_p = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_p,
                    &l_pkeys_all,
                    &cp_stage_p,
                    MAX_SENTINEL,
                )?;
                let fetched_o = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_o,
                    &l_okeys_all,
                    &cp_stage_o,
                    MAX_SENTINEL,
                )?;
                let fetched_s = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_s,
                    &l_skeys_all,
                    &cp_stage_s,
                    MAX_SENTINEL,
                )?;
                let fetched_ps = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_ps,
                    &l_pskeys_all,
                    &cp_stage_ps,
                    MAX_SENTINEL,
                )?;

                // root multiplicities, folded pairwise, and the equality between
                // the two sums
                let mut cp_mu: Vec<(u64, u64)> = Vec::with_capacity(lineitem.len());
                for i in 0..lineitem.len() {
                    self.config.q_cp_mu.enable(&mut region, i)?;
                    region.assign_advice(
                        || "l_ps_key_cp",
                        self.config.l_ps_key_cp,
                        i,
                        || Value::known(F::from(l_pskeys_all[i])),
                    )?;

                    let ta = fetched_p[i].0 * fetched_o[i].0;
                    let ua = fetched_s[i].0 * fetched_ps[i].0;
                    let tc = fetched_p[i].1 * fetched_o[i].1;
                    let uc = fetched_s[i].1 * fetched_ps[i].1;

                    for (col, v) in [
                        (self.config.t_all, ta),
                        (self.config.u_all, ua),
                        (self.config.t_cln, tc),
                        (self.config.u_cln, uc),
                    ] {
                        region.assign_advice(|| "cp fold", col, i, || Value::known(F::from(v)))?;
                    }

                    cp_mu.push((ta * ua, l_cln[i] * tc * uc));
                }
                let (cp_all, cp_cln) = assign_cp_root(&mut region, &self.config.cp_root, &cp_mu)?;
                if !tamper && !all_clean {
                    debug_assert_eq!(
                        cp_all, cp_cln,
                        "cardinality preservation: |R^c join| != |R join|"
                    );
                }

                // public output
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

// ---------------- Circuit wrapper ----------------
pub struct MyCircuit<F: Field + Ord> {
    pub part: Vec<Vec<u64>>,
    pub supplier: Vec<Vec<u64>>,
    pub nation: Vec<Vec<u64>>,
    pub orders: Vec<Vec<u64>>,
    pub partsupp: Vec<Vec<u64>>,
    pub lineitem: Vec<Vec<u64>>,
    pub cond_hash: u64,
    pub _marker: PhantomData<F>,
}

impl<F: Field + Ord> Default for MyCircuit<F> {
    fn default() -> Self {
        Self {
            part: vec![],
            supplier: vec![],
            nation: vec![],
            orders: vec![],
            partsupp: vec![],
            lineitem: vec![],
            cond_hash: 0,
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
            self.part.clone(),
            self.supplier.clone(),
            self.nation.clone(),
            self.orders.clone(),
            self.partsupp.clone(),
            self.lineitem.clone(),
            self.cond_hash,
        )?;

        chip.expose_public(&mut layouter, out_cell, 0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::MyCircuit;
    use crate::data::data_processing;

    use chrono::{Datelike, NaiveDate};
    use halo2_proofs::dev::{MockProver, VerifyFailure};
    use halo2curves::pasta::{vesta, EqAffine, Fp};

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

    use rand::rngs::OsRng;
    use std::marker::PhantomData;
    use std::sync::atomic::Ordering;
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
    fn year_from_date(date_str: &str) -> u64 {
        // TPCH uses DATE; parse YYYY-MM-DD and take year.
        if let Ok(d) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
            d.year() as u64
        } else {
            0
        }
    }

    #[test]
    #[ignore = "inherited heavy end-to-end proof; the fast check is test_cardinality_preservation"]
    fn test_1() {
        let k = 16;

        // Adjust paths as in your repo
        let part_path = &crate::paths::data_file("part.tbl");
        let supplier_path = &crate::paths::data_file("supplier.tbl");
        let nation_path = &crate::paths::data_file("nation.tbl");
        let orders_path = &crate::paths::data_file("orders.tbl");
        let partsupp_path = &crate::paths::data_file("partsupp.tbl");
        let lineitem_path = &crate::paths::data_file("lineitem.tbl");

        // ---------- load ----------
        // NOTE: function names may differ in your crate; rename accordingly.
        let mut part: Vec<Vec<u64>> = vec![];
        let mut supplier: Vec<Vec<u64>> = vec![];
        let mut nation: Vec<Vec<u64>> = vec![];
        let mut orders: Vec<Vec<u64>> = vec![];
        let mut partsupp: Vec<Vec<u64>> = vec![];
        let mut lineitem: Vec<Vec<u64>> = vec![];

        if let Ok(records) = data_processing::part_read_records_from_file(part_path) {
            // use only p_partkey, p_name
            part = records
                .iter()
                .map(|r| vec![r.p_partkey, string_to_u64(&r.p_name)])
                .collect();
        }
        if let Ok(records) = data_processing::supplier_read_records_from_file(supplier_path) {
            supplier = records
                .iter()
                .map(|r| vec![r.s_suppkey, r.s_nationkey])
                .collect();
        }
        if let Ok(records) = data_processing::nation_read_records_from_file(nation_path) {
            nation = records
                .iter()
                .map(|r| vec![r.n_nationkey, string_to_u64(&r.n_name)])
                .collect();
        }
        if let Ok(records) = data_processing::orders_read_records_from_file(orders_path) {
            // use only o_orderkey, o_year
            orders = records
                .iter()
                .map(|r| vec![r.o_orderkey, year_from_date(&r.o_orderdate)])
                .collect();
        }
        if let Ok(records) = data_processing::partsupp_read_records_from_file(partsupp_path) {
            // use ps_key packed + ps_supplycost scaled
            partsupp = records
                .iter()
                .map(|r| {
                    let key = r.ps_partkey * super::PS_SHIFT + r.ps_suppkey;
                    vec![key, scale_by_1000(r.ps_supplycost)]
                })
                .collect();
        }
        if let Ok(records) = data_processing::lineitem_read_records_from_file(lineitem_path) {
            // [l_orderkey,l_partkey,l_suppkey,l_quantity,l_extendedprice,l_discount]
            lineitem = records
                .iter()
                .map(|r| {
                    vec![
                        r.l_orderkey,
                        r.l_partkey,
                        r.l_suppkey,
                        r.l_quantity,
                        scale_by_1000(r.l_extendedprice),
                        scale_by_1000(r.l_discount),
                    ]
                })
                .collect();
        }

        // Q9 param example: ':1' (simplified as equality hash)
        let cond_hash = string_to_u64("green");

        let circuit = MyCircuit::<Fp> {
            part,
            supplier,
            nation,
            orders,
            partsupp,
            lineitem,
            cond_hash,
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
            let proof_path = &crate::paths::proof_file("proof_obj_q9_new");
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
    }

    /// The maximum gate degree of the whole circuit. Every FFT of a real proof
    /// is sized by it, so a fix that raises it costs far more than it saves.
    #[test]
    fn test_max_gate_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        let mut cs = ConstraintSystem::<Fp>::default();
        let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
        let degree = cs.degree();
        println!("cs.degree() = {}", degree);
        println!(
            "COST advice={} fixed={} instance={} selectors={} gates={} polys={} lookups={} shuffles={}",
            cs.num_advice_columns(),
            cs.num_fixed_columns(),
            cs.num_instance_columns(),
            cs.num_selectors(),
            cs.gates().len(),
            cs.gates().iter().map(|g| g.polynomials().len()).sum::<usize>(),
            cs.lookups().len(),
            cs.shuffles().len(),
        );
        // 8, one above the 7 of `q9_obj.rs`: gating both sides of a lookup by
        // the selector COLUMN rather than by a selector RANGE costs one degree
        // on each side. It buys no FFT, which is the number that matters:
        // halo2 sizes the extended domain at the next power of two above
        // degree - 1, so 7, 8 and 9 all run on an 8x domain and only 10 doubles
        // it. Anything that pushes this past 9 does.
        assert!(degree <= 9, "the maximum gate degree rose to {}", degree);
    }

    /// Fast correctness check of the Cardinality Preservation Check: a
    /// truncated slice of the real dataset under MockProver, which verifies
    /// every gate, shuffle and lookup of the circuit without paying for a real
    /// proof.
    #[test]
    fn test_cardinality_preservation() {
        let k = 14;

        // supplier, nation and partsupp are kept whole: the two parts the
        // predicate keeps sit at p_partkey 848 and 1559 and their partsupp rows
        // are spread over the whole table. Part, orders and lineitem are
        // truncated, which is what keeps the slice small. Truncating part below
        // the largest l_partkey also leaves lineitem rows whose partkey occurs
        // in no part tuple, so the slice exercises the gap witness and the
        // sigma = 0 default of the check as well as the present-key lookup.
        const N_PART: usize = 1600;
        const N_ORD: usize = 4000;
        const N_LINE: usize = 8000;

        let mut part: Vec<Vec<u64>> = vec![];
        let mut supplier: Vec<Vec<u64>> = vec![];
        let mut nation: Vec<Vec<u64>> = vec![];
        let mut orders: Vec<Vec<u64>> = vec![];
        let mut partsupp: Vec<Vec<u64>> = vec![];
        let mut lineitem: Vec<Vec<u64>> = vec![];

        if let Ok(records) =
            data_processing::part_read_records_from_file(&crate::paths::data_file("part.tbl"))
        {
            part = records
                .iter()
                .take(N_PART)
                .map(|r| vec![r.p_partkey, string_to_u64(&r.p_name)])
                .collect();
        }
        if let Ok(records) = data_processing::supplier_read_records_from_file(
            &crate::paths::data_file("supplier.tbl"),
        ) {
            supplier = records
                .iter()
                .map(|r| vec![r.s_suppkey, r.s_nationkey])
                .collect();
        }
        if let Ok(records) =
            data_processing::nation_read_records_from_file(&crate::paths::data_file("nation.tbl"))
        {
            nation = records
                .iter()
                .map(|r| vec![r.n_nationkey, string_to_u64(&r.n_name)])
                .collect();
        }
        if let Ok(records) =
            data_processing::orders_read_records_from_file(&crate::paths::data_file("orders.tbl"))
        {
            orders = records
                .iter()
                .take(N_ORD)
                .map(|r| vec![r.o_orderkey, year_from_date(&r.o_orderdate)])
                .collect();
        }
        if let Ok(records) = data_processing::partsupp_read_records_from_file(
            &crate::paths::data_file("partsupp.tbl"),
        ) {
            partsupp = records
                .iter()
                .map(|r| {
                    let key = r.ps_partkey * super::PS_SHIFT + r.ps_suppkey;
                    vec![key, scale_by_1000(r.ps_supplycost)]
                })
                .collect();
        }
        if let Ok(records) = data_processing::lineitem_read_records_from_file(
            &crate::paths::data_file("lineitem.tbl"),
        ) {
            lineitem = records
                .iter()
                .take(N_LINE)
                .map(|r| {
                    vec![
                        r.l_orderkey,
                        r.l_partkey,
                        r.l_suppkey,
                        r.l_quantity,
                        scale_by_1000(r.l_extendedprice),
                        scale_by_1000(r.l_discount),
                    ]
                })
                .collect();
        }

        assert!(
            !part.is_empty()
                && !supplier.is_empty()
                && !nation.is_empty()
                && !orders.is_empty()
                && !partsupp.is_empty()
                && !lineitem.is_empty(),
            "dataset files not found under {}",
            crate::paths::data_file("part.tbl")
        );

        // The simplified LIKE predicate is an equality on string_to_u64(p_name),
        // and string_to_u64("green") matches no part name at all, which would
        // leave the whole clean instance empty and condition (4) vacuous. This
        // is the name of part 848; part 1559 hashes to the same value, so two
        // part rows pass the predicate and the reduced instance is nonempty on
        // every one of the six relations.
        let cond_hash = string_to_u64("orange olive puff midnight almond");

        // The all-clean partition of the third direction below must really be
        // wrong on this slice, or condition (3) would accept it and that
        // direction would pass vacuously. Every lineitem row whose l_partkey is
        // not one of the parts the predicate keeps is a dangling tuple on the
        // lineitem -> part edge the moment the whole relation is declared clean:
        // part^c can never hold a row the predicate dropped, because the link
        // gate pads the clean indicator with 0.
        let kept_pkeys: std::collections::HashSet<u64> = part
            .iter()
            .filter(|r| r[1] == cond_hash)
            .map(|r| r[0])
            .collect();
        let dangling = lineitem
            .iter()
            .filter(|r| !kept_pkeys.contains(&r[1]))
            .count();
        assert!(
            !kept_pkeys.is_empty() && dangling > 0,
            "the slice has {} kept parts and {} lineitem rows dangling on the part edge, \
             so the all-clean partition would be a legitimate reduced instance",
            kept_pkeys.len(),
            dangling
        );

        let circuit = MyCircuit::<Fp> {
            part,
            supplier,
            nation,
            orders,
            partsupp,
            lineitem,
            cond_hash,
            _marker: PhantomData,
        };

        let prover = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        prover.assert_satisfied();

        // Negative direction: the same witness with one joinable lineitem tuple
        // hidden in the residual side, and the neighbours re-reduced around it
        // so that Conservation, Non-Membership and Pairwise Consistency all
        // still hold. Only condition (4) can see this, so the circuit must now
        // reject.
        super::HIDE_ONE_CLEAN_TUPLE.store(true, Ordering::Relaxed);
        let tampered = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        let verdict = tampered.verify();
        super::HIDE_ONE_CLEAN_TUPLE.store(false, Ordering::Relaxed);

        // Checked when the sentinel and profit-pad pins were added: this
        // direction rejects through exactly one constraint, the root equality
        // "cp: cardinality preservation", and nothing else.
        let failures = verdict.expect_err("condition (4) accepted a hidden joinable tuple");
        assert!(
            failures
                .iter()
                .any(|f| format!("{:?}", f).contains("cardinality preservation")),
            "the circuit rejected, but not through the Cardinality Preservation Check: {:?}",
            failures
        );

        // Third direction: the escape condition (3) closes. The prover skips the
        // reduction and declares every real tuple clean, which leaves the
        // residual section empty. Conservation still holds and both channels of
        // condition (4) then compute the same number on every row, so nothing in
        // condition (1) or (4) objects. The partition is not the reduced
        // instance though, and the dangling tuples counted above have no partner
        // in the clean part of the adjacent relation, so a Pairwise Consistency
        // lookup must reject.
        super::MARK_ALL_CLEAN.store(true, Ordering::Relaxed);
        let unreduced = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        let verdict = unreduced.verify();
        super::MARK_ALL_CLEAN.store(false, Ordering::Relaxed);

        // Checked when the sentinel and profit-pad pins were added: this
        // direction rejects through Pairwise Consistency lookups only, and
        // through exactly three of them, "pw: lineitem^c partkey in part^c",
        // "pw: orders^c orderkey in lineitem^c" and
        // "pw: partsupp^c ps_key in lineitem^c". No gate objects.
        let failures = verdict.expect_err("condition (3) accepted the all-clean partition");
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
