//! TPC-H Query 5 proof (base tables only) with internal bag materialization.
//!
//! Proves:
//! 1) NR = nation ⋈ region, filter r_name == EUROPE  -> nr_out_pad(nk_shift, n_name_hash)
//! 2) CO = orders ⋈ customer, filter start<=odate<end -> co_out_pad(okey, nk_shift)
//! 3) LS = lineitem ⋈ supplier -> ls_mat(okey, nk_shift, ext, disc)
//! 4) Join LS with CO (packed (okey,nk)) and with NR (nk)
//! 5) GROUP BY nk -> SUM(ext*(1-disc))  (scaled by 1000)
//! 6) attach n_name_hash from NR
//! 7) ORDER BY revenue DESC
//!
//! IMPORTANT: we shift nationkey and regionkey by +1 in preprocessing to keep 0 as sentinel.
//!
//! UPDATE (padding extras):
//! - You can now pass padding extras for NR and CO materialized bags, and for the LS-join aggregation tail:
//!   * nr_pad_extra: extends NR permutation length by extra dummy PAD rows
//!   * co_pad_extra: extends CO permutation length by extra dummy PAD rows
//!   * ls_pad_extra: extends the LS-join aggregation/sort length by extra dummy PAD rows
//!
//! This is analogous to the triangle example where Bag sizes can be padded beyond the true witness size.
//!
//! UPDATE (condition (4) of the One-Pass OBJ):
//! Same query, same bags, same aggregation as `q5_obj.rs`. What changes is the
//! condition that rules out a joinable tuple hidden in the residual side.
//!
//! `q5_obj.rs` argues it on the residual side: every `ls_disjoin` row carries a
//! membership flag per neighbour bag plus a gap witness, and the "LS disjoin
//! emptiness" gate asks that at least one membership miss. A condition phrased
//! over the residual relation alone cannot see a tuple wrongly moved into the
//! residual whose join partners stay clean, so this file deletes that whole
//! construction and replaces it with the Cardinality Preservation Check of
//! `crate::circuits::card_preserve`: one traversal of the join tree carrying two
//! multiplicities per tuple, one counting join extensions over the inputs and
//! one over the clean instance, and a single equality constraint between the two
//! root sums.
//!
//! Q5 is cyclic, so the traversal runs over the TDJ cluster tree, not over the
//! base relations. Each materialized bag is one relation of the tree:
//!
//!   root  LS = lineitem |X| supplier      (one tuple per lineitem row)
//!     child CO = orders |X| customer      key `okey * SHIFT_NATION + nk_shift`
//!     child NR = nation |X| region        key `nk_shift`
//!
//! Both children are leaves and LS carries no in-relation predicate, so
//!
//!   v_all  = pred                  v_cln  = keep * c            (children)
//!   mu_all = s_all_co * s_all_nr
//!   mu_cln = c_ls * s_cln_co * s_cln_nr                          (root)
//!
//! with `pred` the bag's own predicate bit (`co_keep`, `nr_keep`; 1 for LS).
//! Every clean indicator is bound by the Conservation Check of its own bag:
//! each of the three permutations carries one extra column holding `keep * c` on
//! the input side against a constant 1 on the clean rows and 0 on the residual
//! rows of the partition side, so the multiset equality forces the indicator on
//! an input row to mark exactly the occurrences that went to `R^c`. `q5_obj.rs`
//! partitions only LS; the CO and NR bags gain the residual section they need to
//! be the second side of the count, laid out as
//! `[clean rows | residual rows | pad rows]` inside the existing `*_out_pad`
//! tables.
//!
//! UPDATE (condition (3) of the One-Pass OBJ):
//! Pairwise Consistency is now two mutual Membership Checks per edge of the
//! cluster tree, over the clean sections only:
//!
//!   edge (LS, CO) on `okey * SHIFT_NATION + nk_shift`:  LS^c <-> CO^c
//!   edge (LS, NR) on `nk_shift`:                        LS^c <-> NR^c
//!
//! Four lookups between the clean key columns themselves, with no intermediate
//! key table. `q5_obj.rs` has only the LS -> child direction, into advice columns
//! (`co_key` / `nr_key`) filled from the full filtered bags: a dangling LS tuple
//! is caught, because that direction is a real semijoin filter on the root, but a
//! CO^c or NR^c tuple matching no clean LS tuple contributes to neither channel
//! of condition (4) and is invisible, and nothing binds the two tables to the
//! relations they claim to enumerate. The mirror directions close the first hole
//! and dropping the tables closes the second.
//!
//! Everything else, including the group-by, the name attachment and the ORDER BY
//! proof, is unchanged from `q5_obj.rs`.

use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::lessthan_or_equal_generic::{
    LtEqGenericChip, LtEqGenericConfig, LtEqGenericInstruction,
};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};
use crate::circuits::card_preserve::{
    assign_cp_agg, assign_cp_join, assign_cp_root, build_cp_stage, configure_cp_agg,
    configure_cp_join, configure_cp_root, wire_cp_edge, CpAggConfig, CpJoinConfig, CpRootConfig,
};

use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

// pub(crate) so the multi-lane DP variant (`q5_obj_dp.rs`) shares one
// definition of the PAD/sentinel discipline instead of copying it.
pub(crate) const NUM_BYTES: usize = 7;
pub(crate) const MAX_SENTINEL: u64 = (1u64 << (8 * NUM_BYTES)) - 1;
pub(crate) const PAD_U64: u64 = MAX_SENTINEL;

pub(crate) const SCALE: u64 = 1000;
pub(crate) const PAD_REV: u64 = 0;

// pack (orderkey, nationkey_shift)
pub(crate) const SHIFT_NATION: u64 = 1u64 << 8; // nationkey_shift <= 25+1 fits

/// Test hook, off in every benchmark path: when set, the prover moves one
/// joinable LS tuple to the residual side and re-reduces the CO and NR bags
/// around it, so the partition still passes Conservation, Non-Membership and
/// Pairwise Consistency and only condition (4) can catch it. This is exactly
/// the cheat a residual-side-only argument misses, so the negative test in this
/// module is what shows the Cardinality Preservation Check is not vacuous.
pub static HIDE_ONE_CLEAN_TUPLE: AtomicBool = AtomicBool::new(false);

/// Test hook, off in every benchmark path: when set, the prover skips the
/// semijoin reduction entirely and declares every tuple that passes its
/// predicate clean, so all three residual sections are empty. Conservation still
/// holds and both channels of condition (4) then agree row by row, so this is the
/// escape that only Pairwise Consistency can close, and the negative direction
/// for it in this module is what shows condition (3) is doing work.
pub static MARK_ALL_CLEAN: AtomicBool = AtomicBool::new(false);

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

#[derive(Clone, Debug)]
pub struct Q5Config<F: Field + Ord> {
    // ---------------- base tables ----------------
    // customer: [c_custkey, c_nationkey_shift]
    customer: Vec<Column<Advice>>,
    // orders: [o_orderdate_ts, o_custkey, o_orderkey]
    orders: Vec<Column<Advice>>,
    // lineitem: [l_orderkey, l_suppkey, l_ext, l_disc] (scaled)
    lineitem: Vec<Column<Advice>>,
    // supplier: [s_suppkey, s_nationkey_shift]
    supplier: Vec<Column<Advice>>,
    // nation: [n_nationkey_shift, n_name_hash, n_regionkey_shift]
    nation: Vec<Column<Advice>>,
    // region: [r_regionkey_shift, r_name_hash]
    region_file: Vec<Column<Advice>>,

    // ---------------- conditions ----------------
    cond_europe: Column<Advice>,
    cond_start: Column<Advice>,
    cond_end: Column<Advice>,

    // ---------------- bag materialization: NR ----------------
    q_nr_join: Selector,              // enable nation->region tuple lookup
    q_nr_pred: Selector,              // enable isZero (r_name == EUROPE)
    nr_rname: Column<Advice>,         // looked-up region name hash for nation row
    nr_keep: Column<Advice>,          // boolean
    cflag_nr: Column<Advice>,         // clean indicator per nation row
    nr_pair: Vec<Column<Advice>>,     // [nk_shift, n_name_hash] per nation row
    nr_filt_pad: Vec<Column<Advice>>, // keep? [nr_pair, cflag] : [PAD, PAD, 0]
    nr_out_pad: Vec<Column<Advice>>,  // [clean rows | residual rows | pad rows]
    perm_nr: PermAnyConfig,
    iz_nr: IsZeroConfig<F>,

    // ---------------- bag materialization: CO ----------------
    q_oc_join: Selector, // orders->customer tuple lookup
    q_co_ge: Selector,   // start <= odate (LtEqGeneric)
    q_co_lt: Selector,   // odate < end (LtChip)
    q_co_and: Selector,  // keep = ge*lt
    co_ge_ok: Column<Advice>,
    co_lt_ok: Column<Advice>,
    co_keep: Column<Advice>,      // boolean
    cflag_co: Column<Advice>,     // clean indicator per order row
    co_nk: Column<Advice>,        // looked-up nationkey_shift for order row
    co_pair: Vec<Column<Advice>>, // [okey, nk_shift] per order row
    co_pkey: Column<Advice>,      // co_pair[0]*SHIFT_NATION + co_pair[1]
    co_filt_pad: Vec<Column<Advice>>,
    co_out_pad: Vec<Column<Advice>>, // [clean rows | residual rows | pad rows]
    perm_co: PermAnyConfig,
    lteq_start_le_odate: LtEqGenericConfig<F, NUM_BYTES>,
    lt_odate_lt_end: LtConfig<F, NUM_BYTES>,

    // ---------------- bag materialization: LS ----------------
    q_ls_join: Selector,         // lineitem->supplier tuple lookup
    ls_mat: Vec<Column<Advice>>, // [okey, nk_shift, ext, disc] per lineitem row
    cflag_ls: Column<Advice>,    // clean indicator per LS row
    ls_pkey: Column<Advice>,     // ls_mat[0]*SHIFT_NATION + ls_mat[1]

    // ---------------- LS partition: join/disjoin ----------------
    ls_join: Vec<Column<Advice>>,
    ls_disjoin: Vec<Column<Advice>>,
    ls_part_pad: Vec<Column<Advice>>, // 5 cols: the tuple plus the clean flag
    perm_ls: PermAnyConfig,

    // -------- condition (3), Pairwise Consistency --------
    // One complex selector per relation of the cluster tree, enabled over
    // exactly the clean section of that relation's partition group. Each one is
    // the input selector of the two lookups leaving its relation and the table
    // selector of the two entering it.
    q_pw_ls: Selector, // rows [0, |LS^c|) of ls_join
    q_pw_co: Selector, // rows [0, |CO^c|) of co_out_pad
    q_pw_nr: Selector, // rows [0, |NR^c|) of nr_out_pad

    // ---------------- Cardinality Preservation Check (condition (4)) ----------------
    // |R^c join| == |R join| over the cluster tree rooted at LS
    cp_agg_co: CpAggConfig<F, NUM_BYTES>,  // child CO, keyed by the packed key
    cp_agg_nr: CpAggConfig<F, NUM_BYTES>,  // child NR, keyed by nk_shift
    cp_join_co: CpJoinConfig<F, NUM_BYTES>, // LS -> CO
    cp_join_nr: CpJoinConfig<F, NUM_BYTES>, // LS -> NR
    cp_root: CpRootConfig,
    q_cp_mu: Selector, // the two root product gates

    // clean/residual flag on the partition side of each Conservation Check
    q_cln_flag: Vec<Selector>, // rows of R^c: flag == 1   [NR, CO, LS]
    q_res_flag: Vec<Selector>, // rows of R^r: flag == 0   [NR, CO, LS]

    // ---------------- aggregation over contributing LS ----------------
    ls_sorted: Vec<Column<Advice>>,
    perm_lsort: PermAnyConfig,

    q_line: Selector,
    q_first: Selector,
    q_accu: Selector,

    line_rev: Column<Advice>,
    run_sum: Column<Advice>,
    iz_same_prev: IsZeroConfig<F>,
    iz_same_next: IsZeroConfig<F>,

    // emitted padded result aligned with ls_sorted rows:
    // [nationkey_shift, n_name_hash, revenue]
    res_pad: Vec<Column<Advice>>,

    // attach (nk, name) via lookup into nr_out_pad
    q_res_lookup: Selector,

    // ORDER BY revenue DESC
    res_sorted: Vec<Column<Advice>>,
    perm_res: PermAnyConfig,
    q_sort_res: Selector,
    lteq_rev_next_le_cur: LtEqGenericConfig<F, NUM_BYTES>,

    // public
    instance: Column<Instance>,
    instance_test: Column<Advice>,
}

#[derive(Clone, Debug)]
pub struct Q5Chip<F: Field + Ord> {
    config: Q5Config<F>,
}

impl<F: Field + Ord> Q5Chip<F> {
    pub fn construct(config: Q5Config<F>) -> Self {
        Self { config }
    }

    // pub(crate): reused by the multi-lane DP variant (`q5_obj_dp.rs`).
    pub(crate) fn assign_table_u64(
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

    pub(crate) fn assign_table_f(
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

    pub(crate) fn assign_part_pad_and_link(
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

    pub fn configure(meta: &mut ConstraintSystem<F>) -> Q5Config<F> {
        // public
        let instance = meta.instance_column();
        meta.enable_equality(instance);
        let instance_test = meta.advice_column();
        meta.enable_equality(instance_test);

        // base tables
        let customer = vec![meta.advice_column(), meta.advice_column()];
        let orders = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let lineitem = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let supplier = vec![meta.advice_column(), meta.advice_column()];
        let nation = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let region_file = vec![meta.advice_column(), meta.advice_column()];

        // conditions
        let cond_europe = meta.advice_column();
        let cond_start = meta.advice_column();
        let cond_end = meta.advice_column();

        // ---------------- NR materialization ----------------
        let q_nr_join = meta.complex_selector();
        let q_nr_pred = meta.selector();

        let nr_rname = meta.advice_column();
        let nr_keep = meta.advice_column();
        let cflag_nr = meta.advice_column();

        let nr_pair = vec![meta.advice_column(), meta.advice_column()];
        // One column wider than in q5_obj.rs: the last column of each side
        // carries the clean indicator, so the Conservation Check binds it.
        let (nr_filt_pad, nr_out_pad, perm_nr) = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            let a = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
            let b = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
            let perm = PermAnyChip::configure(meta, q1, q2, a.clone(), b.clone());
            (a, b, perm)
        };

        // nation->region tuple lookup: (n_regionkey_shift, nr_rname) in region(r_regionkey_shift, r_name_hash)
        meta.lookup_any("nation_region_join", |m| {
            let q = m.query_selector(q_nr_join);
            vec![
                (
                    q.clone() * m.query_advice(nation[2], Rotation::cur()),
                    m.query_advice(region_file[0], Rotation::cur()),
                ),
                (
                    q * m.query_advice(nr_rname, Rotation::cur()),
                    m.query_advice(region_file[1], Rotation::cur()),
                ),
            ]
        });

        // isZero: nr_rname == EUROPE
        let iz_aux = meta.advice_column();
        let iz_nr = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_nr_pred),
            |m| {
                m.query_advice(nr_rname, Rotation::cur())
                    - m.query_advice(cond_europe, Rotation::cur())
            },
            iz_aux,
        );
        meta.create_gate("nr_keep = (r_name==EUROPE)", |m| {
            let q = m.query_selector(q_nr_pred);
            let out = m.query_advice(nr_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone() * iz_nr.expr() * (out.clone() - one.clone()),
                q * (one - iz_nr.expr()) * out,
            ]
        });

        // nr_pair is just nation columns (shifted key + name hash)
        meta.create_gate("nr_pair copies nation cols", |m| {
            let q = m.query_selector(q_nr_pred);
            let nk = m.query_advice(nation[0], Rotation::cur());
            let nm = m.query_advice(nation[1], Rotation::cur());
            let p0 = m.query_advice(nr_pair[0], Rotation::cur());
            let p1 = m.query_advice(nr_pair[1], Rotation::cur());
            vec![q.clone() * (p0 - nk), q * (p1 - nm)]
        });

        // link nr_filt_pad = keep? [nr_pair, cflag_nr] : [PAD, PAD, 0]
        // The indicator column pads with 0, so a nation dropped by the EUROPE
        // predicate is never clean and contributes to neither channel of the
        // Cardinality Preservation Check.
        let nr_base = [nr_pair[0], nr_pair[1], cflag_nr];
        let nr_base_pad = [PAD_U64, PAD_U64, 0u64];
        meta.create_gate("link nr_filt_pad", |m| {
            let q = m.query_selector(perm_nr.q_perm1);
            let keep = m.query_advice(nr_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let drop = one.clone() - keep.clone();
            let mut cs = vec![q.clone() * keep.clone() * (one.clone() - keep.clone())];
            for j in 0..3 {
                let b = m.query_advice(nr_base[j], Rotation::cur());
                let f = m.query_advice(nr_filt_pad[j], Rotation::cur());
                let p = Expression::Constant(F::from(nr_base_pad[j]));
                cs.push(q.clone() * (f - (keep.clone() * b + drop.clone() * p)));
            }
            cs
        });

        // ---------------- CO materialization ----------------
        let q_oc_join = meta.complex_selector();
        let q_co_ge = meta.selector();
        let q_co_lt = meta.selector();
        let q_co_and = meta.selector();

        let co_ge_ok = meta.advice_column();
        let co_lt_ok = meta.advice_column();
        let co_keep = meta.advice_column();
        let cflag_co = meta.advice_column();

        let co_nk = meta.advice_column();
        let co_pair = vec![meta.advice_column(), meta.advice_column()];
        let co_pkey = meta.advice_column();
        let (co_filt_pad, co_out_pad, perm_co) = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            let a = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
            let b = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
            let perm = PermAnyChip::configure(meta, q1, q2, a.clone(), b.clone());
            (a, b, perm)
        };

        // orders->customer tuple lookup: (o_custkey, co_nk) in customer(c_custkey, c_nationkey_shift)
        meta.lookup_any("orders_customer_join", |m| {
            let q = m.query_selector(q_oc_join);
            vec![
                (
                    q.clone() * m.query_advice(orders[1], Rotation::cur()),
                    m.query_advice(customer[0], Rotation::cur()),
                ),
                (
                    q * m.query_advice(co_nk, Rotation::cur()),
                    m.query_advice(customer[1], Rotation::cur()),
                ),
            ]
        });

        // co_pair = [o_orderkey, co_nk]
        meta.create_gate("co_pair copies orderkey and looked nk", |m| {
            let q = m.query_selector(q_co_and);
            let ok = m.query_advice(orders[2], Rotation::cur());
            let nk = m.query_advice(co_nk, Rotation::cur());
            let p0 = m.query_advice(co_pair[0], Rotation::cur());
            let p1 = m.query_advice(co_pair[1], Rotation::cur());
            vec![q.clone() * (p0 - ok), q * (p1 - nk)]
        });

        // The CO edge of the cluster tree is keyed by the composite
        // (orderkey, nationkey_shift), packed exactly the way the rest of the
        // file packs it. The Cardinality Preservation Check indexes the child
        // bag by a single column, so the packed key gets one.
        meta.create_gate("co_pkey = okey*SHIFT_NATION + nk", |m| {
            let q = m.query_selector(q_co_and);
            let ok = m.query_advice(co_pair[0], Rotation::cur());
            let nk = m.query_advice(co_pair[1], Rotation::cur());
            let pk = m.query_advice(co_pkey, Rotation::cur());
            vec![q * (pk - (ok * Expression::Constant(F::from(SHIFT_NATION)) + nk))]
        });

        // start <= odate (LtEqGeneric)
        let lteq_start_le_odate = LtEqGenericChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_co_ge),
            |m| vec![m.query_advice(cond_start, Rotation::cur())],
            |m| vec![m.query_advice(orders[0], Rotation::cur())],
        );
        meta.create_gate("co_ge_ok", |m| {
            let q = m.query_selector(q_co_ge);
            let out = m.query_advice(co_ge_ok, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone() * (lteq_start_le_odate.is_lt(m, None) - out.clone()),
                q * out.clone() * (one - out),
            ]
        });

        // odate < end (LtChip)
        let lt_odate_lt_end = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_co_lt),
            |m| m.query_advice(orders[0], Rotation::cur()),
            |m| m.query_advice(cond_end, Rotation::cur()),
        );
        meta.create_gate("co_lt_ok", |m| {
            let q = m.query_selector(q_co_lt);
            let out = m.query_advice(co_lt_ok, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone() * (lt_odate_lt_end.is_lt(m, None) - out.clone()),
                q * out.clone() * (one - out),
            ]
        });

        // co_keep = ge*lt
        meta.create_gate("co_keep = ge*lt", |m| {
            let q = m.query_selector(q_co_and);
            let ge = m.query_advice(co_ge_ok, Rotation::cur());
            let lt = m.query_advice(co_lt_ok, Rotation::cur());
            let keep = m.query_advice(co_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone() * (keep.clone() - ge.clone() * lt.clone()),
                q * keep.clone() * (one - keep),
            ]
        });

        // link co_filt_pad = keep? [co_pair, cflag_co] : [PAD, PAD, 0]
        let co_base = [co_pair[0], co_pair[1], cflag_co];
        let co_base_pad = [PAD_U64, PAD_U64, 0u64];
        meta.create_gate("link co_filt_pad", |m| {
            let q = m.query_selector(perm_co.q_perm1);
            let keep = m.query_advice(co_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let drop = one.clone() - keep.clone();
            let mut cs = vec![q.clone() * keep.clone() * (one.clone() - keep.clone())];
            for j in 0..3 {
                let b = m.query_advice(co_base[j], Rotation::cur());
                let f = m.query_advice(co_filt_pad[j], Rotation::cur());
                let p = Expression::Constant(F::from(co_base_pad[j]));
                cs.push(q.clone() * (f - (keep.clone() * b + drop.clone() * p)));
            }
            cs
        });

        // ---------------- LS materialization (lineitem ⋈ supplier) ----------------
        let q_ls_join = meta.complex_selector();
        let ls_mat = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        // tuple lookup (l_suppkey, ls_mat.nk) in supplier(s_suppkey, s_nationkey_shift)
        meta.lookup_any("lineitem_supplier_join", |m| {
            let q = m.query_selector(q_ls_join);
            vec![
                (
                    q.clone() * m.query_advice(lineitem[1], Rotation::cur()),
                    m.query_advice(supplier[0], Rotation::cur()),
                ),
                (
                    q * m.query_advice(ls_mat[1], Rotation::cur()),
                    m.query_advice(supplier[1], Rotation::cur()),
                ),
            ]
        });

        let cflag_ls = meta.advice_column();
        let ls_pkey = meta.advice_column();

        // the same packed key on the parent side of the LS -> CO edge
        meta.create_gate("ls_pkey = okey*SHIFT_NATION + nk", |m| {
            let q = m.query_selector(q_ls_join);
            let ok = m.query_advice(ls_mat[0], Rotation::cur());
            let nk = m.query_advice(ls_mat[1], Rotation::cur());
            let pk = m.query_advice(ls_pkey, Rotation::cur());
            vec![q * (pk - (ok * Expression::Constant(F::from(SHIFT_NATION)) + nk))]
        });

        // ls_mat copies orderkey/ext/disc from lineitem
        meta.create_gate("ls_mat copies lineitem cols", |m| {
            let q = m.query_selector(q_ls_join);
            let ok = m.query_advice(lineitem[0], Rotation::cur());
            let ext = m.query_advice(lineitem[2], Rotation::cur());
            let disc = m.query_advice(lineitem[3], Rotation::cur());

            let m_ok = m.query_advice(ls_mat[0], Rotation::cur());
            let m_ext = m.query_advice(ls_mat[2], Rotation::cur());
            let m_disc = m.query_advice(ls_mat[3], Rotation::cur());

            vec![
                q.clone() * (m_ok - ok),
                q.clone() * (m_ext - ext),
                q * (m_disc - disc),
            ]
        });

        // ---------------- LS partition permutation ----------------
        let ls_join = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let ls_disjoin = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        // One column wider than in q5_obj.rs: the clean indicator rides along,
        // so the LS Conservation Check binds it.
        let ls_part_pad = (0..5).map(|_| meta.advice_column()).collect::<Vec<_>>();

        for &c in ls_join
            .iter()
            .chain(ls_disjoin.iter())
            .chain(ls_part_pad.iter())
        {
            meta.enable_equality(c);
        }

        let perm_ls = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            let mut ls_in = ls_mat.clone();
            ls_in.push(cflag_ls);
            PermAnyChip::configure(meta, q1, q2, ls_in, ls_part_pad.clone())
        };

        // -------- partition side of the clean indicator: 1 on R^c rows, 0 on R^r --------
        // Each partition column group is laid out as [clean rows | residual rows
        // | pad rows], so one selector per section pins the indicator. Without
        // these the prover could mark a residual row clean and inflate the clean
        // channel of the check below.
        let q_cln_flag = (0..3).map(|_| meta.selector()).collect::<Vec<_>>();
        let q_res_flag = (0..3).map(|_| meta.selector()).collect::<Vec<_>>();

        for (idx, part) in [
            nr_out_pad.clone(),
            co_out_pad.clone(),
            ls_part_pad.clone(),
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
                vec![qc * (f.clone() - Expression::Constant(F::ONE)), qr * f]
            });
        }

        // ---------------- condition (3), Pairwise Consistency ----------------
        // Two mutual Membership Checks per edge of the cluster tree, each looking
        // one clean relation's key column up directly in the adjacent clean
        // relation's key column. Two things were wrong before:
        //
        //  * only the LS -> child direction existed, so a CO^c or NR^c tuple whose
        //    key matched no clean LS tuple contributed to neither channel of
        //    condition (4) and was invisible: the certified clean instance could
        //    carry dangling tuples in both child bags;
        //  * the tables the LS direction looked into were plain advice columns
        //    (`co_key` / `nr_key`) filled from the FULL filtered bags, and nothing
        //    in the circuit bound them to the relation they claimed to enumerate,
        //    so a prover could fill them with pi_K(LS^c) and pass for free.
        //
        // Looking the clean key columns up in each other, in both directions,
        // removes the free advice and closes both holes: the two containments give
        // the set equality condition (3) asks for, at one fewer advice column than
        // the one-directional version cost.
        //
        // A lookup input is 0 on every row where its selector is off, and the
        // table side is 0 on those rows too, so 0 is always in the table and the
        // gated-off (and unassigned) rows cost nothing. Real orderkeys are at
        // least 1 in TPC-H and every nationkey is stored shifted by +1, so both
        // keys are nonzero on the rows that matter and the containment is over the
        // real keys.
        let q_pw_ls = meta.complex_selector();
        let q_pw_co = meta.complex_selector();
        let q_pw_nr = meta.complex_selector();

        // Key columns of the PARTITION groups, never of the base relations:
        // ls_join is the clean section of the LS partition, and co_out_pad /
        // nr_out_pad are laid out as [clean rows | residual rows | pad rows]. The
        // (orderkey, nationkey_shift) key is packed inline with the same
        // SHIFT_NATION the rest of the file uses, which is why neither side needs
        // a materialized packed-key column of its own: co_pkey and ls_pkey live on
        // the input side of the two permutations, in base-relation row order, so
        // they are the wrong columns for this check.
        let ls_ok = ls_join[0];
        let ls_nk = ls_join[1];
        let co_ok = co_out_pad[0];
        let co_nk_p = co_out_pad[1];
        let nr_nk = nr_out_pad[0];

        // edge (LS, CO) on the packed (o_orderkey, nationkey_shift) key
        meta.lookup_any("pw: LS^c pkey in CO^c pkey", move |m| {
            let s = Expression::Constant(F::from(SHIFT_NATION));
            let lhs = m.query_selector(q_pw_ls)
                * (m.query_advice(ls_ok, Rotation::cur()) * s.clone()
                    + m.query_advice(ls_nk, Rotation::cur()));
            let rhs = m.query_selector(q_pw_co)
                * (m.query_advice(co_ok, Rotation::cur()) * s
                    + m.query_advice(co_nk_p, Rotation::cur()));
            vec![(lhs, rhs)]
        });
        meta.lookup_any("pw: CO^c pkey in LS^c pkey", move |m| {
            let s = Expression::Constant(F::from(SHIFT_NATION));
            let lhs = m.query_selector(q_pw_co)
                * (m.query_advice(co_ok, Rotation::cur()) * s.clone()
                    + m.query_advice(co_nk_p, Rotation::cur()));
            let rhs = m.query_selector(q_pw_ls)
                * (m.query_advice(ls_ok, Rotation::cur()) * s
                    + m.query_advice(ls_nk, Rotation::cur()));
            vec![(lhs, rhs)]
        });

        // edge (LS, NR) on nationkey_shift
        meta.lookup_any("pw: LS^c nationkey in NR^c nationkey", move |m| {
            let lhs = m.query_selector(q_pw_ls) * m.query_advice(ls_nk, Rotation::cur());
            let rhs = m.query_selector(q_pw_nr) * m.query_advice(nr_nk, Rotation::cur());
            vec![(lhs, rhs)]
        });
        meta.lookup_any("pw: NR^c nationkey in LS^c nationkey", move |m| {
            let lhs = m.query_selector(q_pw_nr) * m.query_advice(nr_nk, Rotation::cur());
            let rhs = m.query_selector(q_pw_ls) * m.query_advice(ls_nk, Rotation::cur());
            vec![(lhs, rhs)]
        });

        // ---------------- Cardinality Preservation Check (condition (4)) ----------------
        // One fixed column serves every Lt chip of the check, so the whole check
        // costs a single u8 range table.
        let cp_u8 = meta.fixed_column();

        // Children of the root. Both are leaves, so their two multiplicity
        // columns are columns the circuit already has: the predicate bit is the
        // input channel and the bound indicator (keep * c, i.e. the last column
        // of the filt_pad side of the Conservation Check) is the clean one.
        let cp_agg_co = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            co_pkey, // packed (o_orderkey, nk_shift)
            co_keep,
            co_filt_pad[2],
            PAD_U64,
        );
        let cp_agg_nr = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            nr_pair[0], // nk_shift
            nr_keep,
            nr_filt_pad[2],
            PAD_U64,
        );

        // Parent side, on the rows of LS (one per lineitem row).
        let cp_join_co = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, ls_pkey);
        let cp_join_nr = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, ls_mat[1]);
        wire_cp_edge(meta, &cp_join_co, &cp_agg_co, ls_pkey);
        wire_cp_edge(meta, &cp_join_nr, &cp_agg_nr, ls_mat[1]);

        // Root multiplicities and the single equality that compares the two join
        // cardinalities.
        let cp_root = configure_cp_root::<F>(meta);
        let q_cp_mu = meta.selector();
        {
            let s_all_co = cp_join_co.s_all;
            let s_cln_co = cp_join_co.s_cln;
            let s_all_nr = cp_join_nr.s_all;
            let s_cln_nr = cp_join_nr.s_cln;
            let mu_all = cp_root.mu_all;
            let mu_cln = cp_root.mu_cln;
            meta.create_gate("cp: root multiplicities over LS", move |m| {
                let q = m.query_selector(q_cp_mu);
                // LS carries no in-relation predicate, so pred_LS == 1 and every
                // lineitem row contributes exactly one LS tuple.
                let all = m.query_advice(mu_all, Rotation::cur())
                    - m.query_advice(s_all_co, Rotation::cur())
                        * m.query_advice(s_all_nr, Rotation::cur());
                let cln = m.query_advice(mu_cln, Rotation::cur())
                    - m.query_advice(cflag_ls, Rotation::cur())
                        * m.query_advice(s_cln_co, Rotation::cur())
                        * m.query_advice(s_cln_nr, Rotation::cur());
                vec![q.clone() * all, q * cln]
            });
        }

        // ---------------- aggregation ----------------
        let ls_sorted = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let perm_lsort = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            // NOTE: still permute ls_join <-> ls_sorted, but you may enable more rows in witness
            // if you pass ls_pad_extra; those additional rows must be padded consistently on both sides.
            PermAnyChip::configure(meta, q1, q2, ls_join.clone(), ls_sorted.clone())
        };

        let q_line = meta.selector();
        let q_first = meta.selector();
        let q_accu = meta.selector();

        let line_rev = meta.advice_column();
        let run_sum = meta.advice_column();

        let aux_same_prev = meta.advice_column();
        let aux_same_next = meta.advice_column();
        let iz_same_prev = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_accu),
            |m| {
                m.query_advice(ls_sorted[1], Rotation::cur())
                    - m.query_advice(ls_sorted[1], Rotation::prev())
            },
            aux_same_prev,
        );
        let iz_same_next = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_line),
            |m| {
                m.query_advice(ls_sorted[1], Rotation::next())
                    - m.query_advice(ls_sorted[1], Rotation::cur())
            },
            aux_same_next,
        );

        // line_rev = ext*(SCALE-disc)
        meta.create_gate("line_rev", |m| {
            let q = m.query_selector(q_line);
            let ext = m.query_advice(ls_sorted[2], Rotation::cur());
            let disc = m.query_advice(ls_sorted[3], Rotation::cur());
            let lr = m.query_advice(line_rev, Rotation::cur());
            let scale = Expression::Constant(F::from(SCALE));
            vec![q * (lr - ext * (scale - disc))]
        });
        meta.create_gate("run_sum_first", |m| {
            let q = m.query_selector(q_first);
            let rs = m.query_advice(run_sum, Rotation::cur());
            let lr = m.query_advice(line_rev, Rotation::cur());
            vec![q * (rs - lr)]
        });
        meta.create_gate("run_sum_accu", |m| {
            let q = m.query_selector(q_accu);
            let same = iz_same_prev.expr();
            let rs_cur = m.query_advice(run_sum, Rotation::cur());
            let rs_prev = m.query_advice(run_sum, Rotation::prev());
            let lr = m.query_advice(line_rev, Rotation::cur());
            vec![q * (rs_cur - (same * rs_prev + lr))]
        });

        let res_pad = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let q_res_lookup = meta.complex_selector();

        // emit group row only at last row of each nation group
        meta.create_gate("emit_res_pad", |m| {
            let q = m.query_selector(q_line);
            let one = Expression::Constant(F::ONE);
            let is_last = one.clone() - iz_same_next.expr();
            let not_last = one.clone() - is_last.clone();

            let nk = m.query_advice(ls_sorted[1], Rotation::cur());
            let rs = m.query_advice(run_sum, Rotation::cur());

            let out_nk = m.query_advice(res_pad[0], Rotation::cur());
            let out_nm = m.query_advice(res_pad[1], Rotation::cur());
            let out_rev = m.query_advice(res_pad[2], Rotation::cur());

            let pad_nk = Expression::Constant(F::from(PAD_U64));
            let pad_nm = Expression::Constant(F::from(PAD_U64));
            let pad_rev = Expression::Constant(F::from(PAD_REV));

            vec![
                q.clone() * (out_nk - (is_last.clone() * nk + not_last.clone() * pad_nk)),
                q.clone() * (out_rev - (is_last.clone() * rs + not_last.clone() * pad_rev)),
                q * not_last * (out_nm - pad_nm),
            ]
        });

        // attach (nk,name) via lookup into nr_out_pad (tuple lookup)
        meta.lookup_any("attach name from NR_out", |m| {
            let q_in = m.query_selector(q_res_lookup);
            let one = Expression::Constant(F::ONE);
            let is_last = one - iz_same_next.expr();
            let gate = q_in * is_last;

            vec![
                (
                    gate.clone() * m.query_advice(res_pad[0], Rotation::cur()),
                    m.query_advice(nr_out_pad[0], Rotation::cur()),
                ),
                (
                    gate * m.query_advice(res_pad[1], Rotation::cur()),
                    m.query_advice(nr_out_pad[1], Rotation::cur()),
                ),
            ]
        });

        // ---------------- ORDER BY revenue DESC ----------------
        let res_sorted = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let perm_res = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            PermAnyChip::configure(meta, q1, q2, res_pad.clone(), res_sorted.clone())
        };
        let q_sort_res = meta.selector();

        let lteq_rev_next_le_cur = LtEqGenericChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| vec![m.query_advice(res_sorted[2], Rotation::next())],
            |m| vec![m.query_advice(res_sorted[2], Rotation::cur())],
        );
        meta.create_gate("ORDER BY revenue DESC", |m| {
            let q = m.query_selector(q_sort_res);
            vec![q * (lteq_rev_next_le_cur.is_lt(m, None) - Expression::Constant(F::ONE))]
        });

        Q5Config {
            customer,
            orders,
            lineitem,
            supplier,
            nation,
            region_file,

            cond_europe,
            cond_start,
            cond_end,

            q_nr_join,
            q_nr_pred,
            nr_rname,
            nr_keep,
            cflag_nr,
            nr_pair,
            nr_filt_pad,
            nr_out_pad,
            perm_nr,
            iz_nr,

            q_oc_join,
            q_co_ge,
            q_co_lt,
            q_co_and,
            co_ge_ok,
            co_lt_ok,
            co_keep,
            cflag_co,
            co_nk,
            co_pair,
            co_pkey,
            co_filt_pad,
            co_out_pad,
            perm_co,
            lteq_start_le_odate,
            lt_odate_lt_end,

            q_ls_join,
            ls_mat,
            cflag_ls,
            ls_pkey,

            ls_join,
            ls_disjoin,
            ls_part_pad,
            perm_ls,

            q_pw_ls,
            q_pw_co,
            q_pw_nr,

            cp_agg_co,
            cp_agg_nr,
            cp_join_co,
            cp_join_nr,
            cp_root,
            q_cp_mu,
            q_cln_flag,
            q_res_flag,

            ls_sorted,
            perm_lsort,

            q_line,
            q_first,
            q_accu,

            line_rev,
            run_sum,
            iz_same_prev,
            iz_same_next,

            res_pad,
            q_res_lookup,

            res_sorted,
            perm_res,
            q_sort_res,
            lteq_rev_next_le_cur,

            instance,
            instance_test,
        }
    }

    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        // base inputs
        customer: Vec<Vec<u64>>, // [custkey, nationkey] (unshifted in file)
        orders: Vec<Vec<u64>>,   // [odate_ts, custkey, orderkey]
        lineitem: Vec<Vec<u64>>, // [orderkey, suppkey, ext_scaled, disc_scaled]
        supplier: Vec<Vec<u64>>, // [suppkey, nationkey] (unshifted)
        nation: Vec<Vec<u64>>,   // [nationkey, name_hash, regionkey] (unshifted)
        region_file: Vec<Vec<u64>>, // [regionkey, name_hash] (unshifted)
        europe_hash: u64,
        start_ts: u64,
        end_ts: u64,
        // NEW: padding knobs
        nr_pad_extra: usize,
        co_pad_extra: usize,
        ls_pad_extra: usize,
    ) -> Result<AssignedCell<F, F>, Error> {
        // chips
        let iz_nr_chip = IsZeroChip::construct(self.config.iz_nr.clone());

        let lteq_ge_chip =
            LtEqGenericChip::<F, NUM_BYTES>::construct(self.config.lteq_start_le_odate.clone());
        lteq_ge_chip.load(layouter)?;

        let lt_end_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_odate_lt_end.clone());
        lt_end_chip.load(layouter)?;

        // Every Lt chip of the Cardinality Preservation Check shares one u8 fixed
        // column, so a single load covers the whole check. This replaces the four
        // gap chips of the deleted residual-side argument.
        LtChip::<F, NUM_BYTES>::construct(self.config.cp_agg_co.lt_key_cur_next).load(layouter)?;

        let iz_same_prev_chip = IsZeroChip::construct(self.config.iz_same_prev.clone());
        let iz_same_next_chip = IsZeroChip::construct(self.config.iz_same_next.clone());

        let lteq_rev_chip =
            LtEqGenericChip::<F, NUM_BYTES>::construct(self.config.lteq_rev_next_le_cur.clone());
        lteq_rev_chip.load(layouter)?;

        // helpers
        fn to_field_rows<FF: Field + Ord>(u: &[Vec<u64>]) -> Vec<Vec<FF>> {
            u.iter()
                .map(|r| r.iter().map(|&x| FF::from(x)).collect())
                .collect()
        }
        fn pad_filter_u64(rows: &[Vec<u64>], keep: &[bool], pad: &[u64]) -> Vec<Vec<u64>> {
            rows.iter()
                .zip(keep.iter())
                .map(|(r, &k)| if k { r.clone() } else { pad.to_vec() })
                .collect()
        }
        fn pad_out_u64(filtered: &[Vec<u64>], total: usize, pad: &[u64]) -> Vec<Vec<u64>> {
            let mut out: Vec<Vec<u64>> = Vec::with_capacity(total);
            out.extend_from_slice(filtered);
            while out.len() < total {
                out.push(pad.to_vec());
            }
            out
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

        // Derived intermediates come from ONE shared definition (see
        // `q5_derive`), which `bench_queries::q5_pads` also uses to size the
        // DP capacity for Bag 2 {L,S} -- so the two can never drift.
        let Q5Derived {
            nr_rname_u64,
            nr_keep_b,
            nr_pair_u64,
            nr_filtered,
            nk_to_name,
            co_nk_u64,
            co_ge_b,
            co_lt_b,
            co_keep_b,
            co_pair_u64,
            co_filtered,
            ls_mat_u64,
            // the partition is rebuilt below from the per-row clean indicator,
            // which is what the Cardinality Preservation Check needs
            ls_join_u64: _,
            ls_dis_u64: _,
            nr_filt_pad_u64_ext,
            co_filt_pad_u64_ext,
            nr_total,
            co_total,
            // the *_out_pad tables gain the clean/residual layout below
            nr_out_pad_u64: _,
            co_out_pad_u64: _,
        } = q5_derive(
            &customer,
            &orders,
            &lineitem,
            &supplier,
            &nation,
            &region_file,
            europe_hash,
            start_ts,
            end_ts,
            nr_pad_extra,
            co_pad_extra,
        );

        // ---------------- clean/residual witness over the cluster tree ----------------
        // The honest clean instance is the fully reduced one: an LS tuple is clean
        // iff it joins both bags, a CO tuple iff its packed key occurs in a clean
        // LS tuple, an NR tuple iff its nationkey does. That is a fixed point of
        // the semijoin reduction, so Conservation, Non-Membership and Pairwise
        // Consistency all hold on it and condition (4) holds with equality.
        let co_set: HashSet<u64> = co_filtered
            .iter()
            .map(|r| r[0] * SHIFT_NATION + r[1])
            .collect();
        let nr_set: HashSet<u64> = nr_filtered.iter().map(|r| r[0]).collect();

        let mut ls_cln_b: Vec<u64> = ls_mat_u64
            .iter()
            .map(|r| {
                (co_set.contains(&(r[0] * SHIFT_NATION + r[1])) && nr_set.contains(&r[1])) as u64
            })
            .collect();

        // test hook only: hide one joinable LS tuple in the residual side. The
        // CO and NR indicators below are then recomputed from the reduced clean
        // LS set, so the neighbours are re-reduced around it.
        let tamper = HIDE_ONE_CLEAN_TUPLE.load(Ordering::Relaxed);
        if tamper {
            if let Some(i) = ls_cln_b.iter().position(|&b| b == 1) {
                ls_cln_b[i] = 0;
            }
        }

        // test hook only: no reduction at all. Every tuple that passes its own
        // predicate is declared clean, so all three residual sections come out
        // empty and both channels of condition (4) agree row by row. Only
        // condition (3) can reject this.
        let all_clean = MARK_ALL_CLEAN.load(Ordering::Relaxed);
        if all_clean {
            for b in ls_cln_b.iter_mut() {
                *b = 1;
            }
        }

        let mut ls_join_u64: Vec<Vec<u64>> = vec![];
        let mut ls_dis_u64: Vec<Vec<u64>> = vec![];
        for (i, r) in ls_mat_u64.iter().enumerate() {
            if ls_cln_b[i] == 1 {
                ls_join_u64.push(r.clone());
            } else {
                ls_dis_u64.push(r.clone());
            }
        }

        let cln_ls_keys: HashSet<u64> = ls_join_u64
            .iter()
            .map(|r| r[0] * SHIFT_NATION + r[1])
            .collect();
        let cln_ls_nks: HashSet<u64> = ls_join_u64.iter().map(|r| r[1]).collect();

        // clean indicator per input row of the two children (keep folded in)
        let cln_co: Vec<u64> = (0..orders.len())
            .map(|i| {
                (co_keep_b[i]
                    && (all_clean
                        || cln_ls_keys
                            .contains(&(co_pair_u64[i][0] * SHIFT_NATION + co_pair_u64[i][1]))))
                    as u64
            })
            .collect();
        let cln_nr: Vec<u64> = (0..nation.len())
            .map(|i| {
                (nr_keep_b[i] && (all_clean || cln_ls_nks.contains(&nr_pair_u64[i][0]))) as u64
            })
            .collect();

        // ---- clean indicator on both sides of the three Conservation Checks ----
        // input side: the link gates turn the base indicator into keep * c
        let nr_filt_pad_u64_ext: Vec<Vec<u64>> = nr_filt_pad_u64_ext
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let mut v = r.clone();
                v.push(if i < nation.len() { cln_nr[i] } else { 0 });
                v
            })
            .collect();
        let co_filt_pad_u64_ext: Vec<Vec<u64>> = co_filt_pad_u64_ext
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let mut v = r.clone();
                v.push(if i < orders.len() { cln_co[i] } else { 0 });
                v
            })
            .collect();

        // partition side: [clean rows | residual rows | pad rows], the flag a
        // constant 1 then 0 then 0
        fn split_clean(
            pair: &[Vec<u64>],
            keep: &[bool],
            cln: &[u64],
            total: usize,
        ) -> (Vec<Vec<u64>>, usize, usize) {
            let mut out: Vec<Vec<u64>> = vec![];
            let mut res: Vec<Vec<u64>> = vec![];
            for i in 0..pair.len() {
                if !keep[i] {
                    continue;
                }
                let mut v = pair[i].clone();
                if cln[i] == 1 {
                    v.push(1);
                    out.push(v);
                } else {
                    v.push(0);
                    res.push(v);
                }
            }
            let n_cln = out.len();
            let n_res = res.len();
            out.extend(res);
            while out.len() < total {
                out.push(vec![PAD_U64, PAD_U64, 0]);
            }
            (out, n_cln, n_res)
        }
        let (nr_out_pad_u64, nr_cln_len, nr_res_len) =
            split_clean(&nr_pair_u64, &nr_keep_b, &cln_nr, nr_total);
        let (co_out_pad_u64, co_cln_len, co_res_len) =
            split_clean(&co_pair_u64, &co_keep_b, &cln_co, co_total);

        // condition (3) needs no key vectors any more: the four Pairwise
        // Consistency lookups run between the clean key columns themselves.

        // the partition side of the LS Conservation Check carries the flag as a
        // fifth column: 1 on the clean rows, 0 on the residual and pad rows
        let with_flag = |rows: &[Vec<u64>], f: u64| -> Vec<Vec<u64>> {
            rows.iter()
                .map(|r| {
                    let mut v = r.clone();
                    v.push(f);
                    v
                })
                .collect()
        };
        let pad5 = vec![PAD_U64, PAD_U64, PAD_U64, PAD_U64, 0];
        let ls_part_pad_u64 = pad_partition_u64(
            &with_flag(&ls_join_u64, 1),
            &with_flag(&ls_dis_u64, 0),
            lineitem.len(),
            &pad5,
        );
        let ls_part_pad_f: Vec<Vec<F>> = to_field_rows::<F>(&ls_part_pad_u64);

        // ---------- aggregation over ls_join (with optional extra padding rows) ----------
        // Use pad rows that sort to the end by nk=PAD_U64 and have ext=disc=0.
        let join_len = ls_join_u64.len();
        let n = join_len.saturating_add(ls_pad_extra).max(1);

        let mut ls_join_ext = ls_join_u64.clone();
        while ls_join_ext.len() < n {
            // pad row: [okey=0, nk=PAD_U64, ext=0, disc=0]
            ls_join_ext.push(vec![0u64, PAD_U64, 0u64, 0u64]);
        }

        let mut ls_sorted_u64 = ls_join_ext.clone();
        ls_sorted_u64.sort_by_key(|r| r[1]); // by nationkey_shift (PAD_U64 goes last)

        let mut line_rev_u64 = vec![0u64; n];
        let mut run_sum_u64 = vec![0u64; n];
        let mut res_pad_u64: Vec<[u64; 3]> = vec![[PAD_U64, PAD_U64, PAD_REV]; n];

        let mut acc: u128 = 0;
        let mut prev_nk: Option<u64> = None;
        for i in 0..n {
            let nk = ls_sorted_u64[i][1];
            let ext = ls_sorted_u64[i][2] as u128;
            let disc = ls_sorted_u64[i][3] as u128;
            let lr = ext * ((SCALE as u128) - disc);
            line_rev_u64[i] = lr as u64;

            if prev_nk == Some(nk) {
                acc += lr;
            } else {
                acc = lr;
            }
            run_sum_u64[i] = acc as u64;

            let next_nk = if i + 1 < n {
                ls_sorted_u64[i + 1][1]
            } else {
                0
            };

            // IMPORTANT: don't emit for PAD_U64 groups; keep res_pad as PAD rows
            if next_nk != nk && nk != PAD_U64 {
                let nm = *nk_to_name.get(&nk).unwrap_or(&0);
                res_pad_u64[i] = [nk, nm, run_sum_u64[i]];
            }
            prev_nk = Some(nk);
        }

        // sort result by revenue desc
        let mut groups: Vec<[u64; 3]> = res_pad_u64
            .iter()
            .copied()
            .filter(|r| r[0] != PAD_U64)
            .collect();
        groups.sort_by(|a, b| b[2].cmp(&a[2]));
        let mut res_sorted_u64: Vec<[u64; 3]> = vec![];
        res_sorted_u64.extend(groups);
        while res_sorted_u64.len() < n {
            res_sorted_u64.push([PAD_U64, PAD_U64, PAD_REV]);
        }

        // ---------- assign region ----------
        layouter.assign_region(
            || "q5 witness",
            |mut region| {
                // base tables
                for i in 0..customer.len() {
                    for j in 0..2 {
                        // shift nationkey inside the assigned customer table (keep base single-table input)
                        let v = if j == 1 {
                            customer[i][j] + 1
                        } else {
                            customer[i][j]
                        };
                        region.assign_advice(
                            || "customer",
                            self.config.customer[j],
                            i,
                            || Value::known(F::from(v)),
                        )?;
                    }
                }
                for i in 0..orders.len() {
                    for j in 0..3 {
                        region.assign_advice(
                            || "orders",
                            self.config.orders[j],
                            i,
                            || Value::known(F::from(orders[i][j])),
                        )?;
                    }
                    region.assign_advice(
                        || "cond_start",
                        self.config.cond_start,
                        i,
                        || Value::known(F::from(start_ts)),
                    )?;
                    region.assign_advice(
                        || "cond_end",
                        self.config.cond_end,
                        i,
                        || Value::known(F::from(end_ts)),
                    )?;
                }
                for i in 0..lineitem.len() {
                    for j in 0..4 {
                        region.assign_advice(
                            || "lineitem",
                            self.config.lineitem[j],
                            i,
                            || Value::known(F::from(lineitem[i][j])),
                        )?;
                    }
                }
                for i in 0..supplier.len() {
                    for j in 0..2 {
                        let v = if j == 1 {
                            supplier[i][j] + 1
                        } else {
                            supplier[i][j]
                        };
                        region.assign_advice(
                            || "supplier",
                            self.config.supplier[j],
                            i,
                            || Value::known(F::from(v)),
                        )?;
                    }
                }
                for i in 0..nation.len() {
                    // nation: shift n_nationkey and n_regionkey
                    region.assign_advice(
                        || "nation_nk",
                        self.config.nation[0],
                        i,
                        || Value::known(F::from(nation[i][0] + 1)),
                    )?;
                    region.assign_advice(
                        || "nation_nm",
                        self.config.nation[1],
                        i,
                        || Value::known(F::from(nation[i][1])),
                    )?;
                    region.assign_advice(
                        || "nation_rk",
                        self.config.nation[2],
                        i,
                        || Value::known(F::from(nation[i][2] + 1)),
                    )?;
                }
                for i in 0..region_file.len() {
                    region.assign_advice(
                        || "region_rk",
                        self.config.region_file[0],
                        i,
                        || Value::known(F::from(region_file[i][0] + 1)),
                    )?;
                    region.assign_advice(
                        || "region_nm",
                        self.config.region_file[1],
                        i,
                        || Value::known(F::from(region_file[i][1])),
                    )?;
                }

                // ---------- NR materialization assignments (real nation rows) ----------
                for i in 0..nation.len() {
                    self.config.q_nr_join.enable(&mut region, i)?;
                    self.config.q_nr_pred.enable(&mut region, i)?;

                    region.assign_advice(
                        || "cond_europe",
                        self.config.cond_europe,
                        i,
                        || Value::known(F::from(europe_hash)),
                    )?;
                    region.assign_advice(
                        || "nr_rname",
                        self.config.nr_rname,
                        i,
                        || Value::known(F::from(nr_rname_u64[i])),
                    )?;
                    region.assign_advice(
                        || "nr_keep",
                        self.config.nr_keep,
                        i,
                        || Value::known(F::from(nr_keep_b[i] as u64)),
                    )?;

                    region.assign_advice(
                        || "nr_pair_nk",
                        self.config.nr_pair[0],
                        i,
                        || Value::known(F::from(nr_pair_u64[i][0])),
                    )?;
                    region.assign_advice(
                        || "nr_pair_nm",
                        self.config.nr_pair[1],
                        i,
                        || Value::known(F::from(nr_pair_u64[i][1])),
                    )?;
                    region.assign_advice(
                        || "cflag_nr",
                        self.config.cflag_nr,
                        i,
                        || Value::known(F::from(cln_nr[i])),
                    )?;

                    iz_nr_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(nr_rname_u64[i]) - F::from(europe_hash)),
                    )?;
                }

                // ---------- NR extra padding rows (no join/pred selectors), but must assign cols used by link gate ----------
                for i in nation.len()..nr_total {
                    // q_nr_* not enabled
                    region.assign_advice(
                        || "cond_europe_pad",
                        self.config.cond_europe,
                        i,
                        || Value::known(F::from(europe_hash)),
                    )?;
                    region.assign_advice(
                        || "nr_rname_pad",
                        self.config.nr_rname,
                        i,
                        || Value::known(F::ZERO),
                    )?;
                    region.assign_advice(
                        || "nr_keep_pad",
                        self.config.nr_keep,
                        i,
                        || Value::known(F::ZERO),
                    )?;
                    region.assign_advice(
                        || "nr_pair_nk_pad",
                        self.config.nr_pair[0],
                        i,
                        || Value::known(F::from(PAD_U64)),
                    )?;
                    region.assign_advice(
                        || "nr_pair_nm_pad",
                        self.config.nr_pair[1],
                        i,
                        || Value::known(F::from(PAD_U64)),
                    )?;
                    region.assign_advice(
                        || "cflag_nr_pad",
                        self.config.cflag_nr,
                        i,
                        || Value::known(F::ZERO),
                    )?;
                }

                // enable permutation for NR and assign filt/out (extended)
                for i in 0..nr_total {
                    self.config.perm_nr.q_perm1.enable(&mut region, i)?;
                    self.config.perm_nr.q_perm2.enable(&mut region, i)?;
                }
                // partition side: 1 on the clean NR rows, 0 on the residual ones
                for i in 0..nr_cln_len {
                    self.config.q_cln_flag[0].enable(&mut region, i)?;
                }
                for i in nr_cln_len..(nr_cln_len + nr_res_len) {
                    self.config.q_res_flag[0].enable(&mut region, i)?;
                }
                Self::assign_table_f(
                    &mut region,
                    "nr_filt_pad",
                    &self.config.nr_filt_pad,
                    &to_field_rows::<F>(&nr_filt_pad_u64_ext),
                )?;
                Self::assign_table_f(
                    &mut region,
                    "nr_out_pad",
                    &self.config.nr_out_pad,
                    &to_field_rows::<F>(&nr_out_pad_u64),
                )?;

                // ---------- CO materialization assignments (real order rows) ----------
                for i in 0..orders.len() {
                    self.config.q_oc_join.enable(&mut region, i)?;
                    self.config.q_co_ge.enable(&mut region, i)?;
                    self.config.q_co_lt.enable(&mut region, i)?;
                    self.config.q_co_and.enable(&mut region, i)?;

                    region.assign_advice(
                        || "co_nk",
                        self.config.co_nk,
                        i,
                        || Value::known(F::from(co_nk_u64[i])),
                    )?;
                    region.assign_advice(
                        || "co_ge_ok",
                        self.config.co_ge_ok,
                        i,
                        || Value::known(F::from(co_ge_b[i] as u64)),
                    )?;
                    region.assign_advice(
                        || "co_lt_ok",
                        self.config.co_lt_ok,
                        i,
                        || Value::known(F::from(co_lt_b[i] as u64)),
                    )?;
                    region.assign_advice(
                        || "co_keep",
                        self.config.co_keep,
                        i,
                        || Value::known(F::from(co_keep_b[i] as u64)),
                    )?;

                    region.assign_advice(
                        || "co_pair_ok",
                        self.config.co_pair[0],
                        i,
                        || Value::known(F::from(co_pair_u64[i][0])),
                    )?;
                    region.assign_advice(
                        || "co_pair_nk",
                        self.config.co_pair[1],
                        i,
                        || Value::known(F::from(co_pair_u64[i][1])),
                    )?;
                    region.assign_advice(
                        || "co_pkey",
                        self.config.co_pkey,
                        i,
                        || {
                            Value::known(F::from(
                                co_pair_u64[i][0] * SHIFT_NATION + co_pair_u64[i][1],
                            ))
                        },
                    )?;
                    region.assign_advice(
                        || "cflag_co",
                        self.config.cflag_co,
                        i,
                        || Value::known(F::from(cln_co[i])),
                    )?;

                    // predicate chips
                    lteq_ge_chip.assign(
                        &mut region,
                        i,
                        &[F::from(start_ts)],
                        &[F::from(orders[i][0])],
                    )?;
                    lt_end_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(orders[i][0])),
                        Value::known(F::from(end_ts)),
                    )?;
                }

                // ---------- CO extra padding rows (no join/pred selectors), but must assign cols used by link gate ----------
                for i in orders.len()..co_total {
                    // q_co_* not enabled
                    region.assign_advice(
                        || "co_nk_pad",
                        self.config.co_nk,
                        i,
                        || Value::known(F::ZERO),
                    )?;
                    region.assign_advice(
                        || "co_ge_ok_pad",
                        self.config.co_ge_ok,
                        i,
                        || Value::known(F::ZERO),
                    )?;
                    region.assign_advice(
                        || "co_lt_ok_pad",
                        self.config.co_lt_ok,
                        i,
                        || Value::known(F::ZERO),
                    )?;
                    region.assign_advice(
                        || "co_keep_pad",
                        self.config.co_keep,
                        i,
                        || Value::known(F::ZERO),
                    )?;
                    region.assign_advice(
                        || "co_pair_ok_pad",
                        self.config.co_pair[0],
                        i,
                        || Value::known(F::from(PAD_U64)),
                    )?;
                    region.assign_advice(
                        || "co_pair_nk_pad",
                        self.config.co_pair[1],
                        i,
                        || Value::known(F::from(PAD_U64)),
                    )?;
                    region.assign_advice(
                        || "cflag_co_pad",
                        self.config.cflag_co,
                        i,
                        || Value::known(F::ZERO),
                    )?;
                }

                for i in 0..co_total {
                    self.config.perm_co.q_perm1.enable(&mut region, i)?;
                    self.config.perm_co.q_perm2.enable(&mut region, i)?;
                }
                // partition side: 1 on the clean CO rows, 0 on the residual ones
                for i in 0..co_cln_len {
                    self.config.q_cln_flag[1].enable(&mut region, i)?;
                }
                for i in co_cln_len..(co_cln_len + co_res_len) {
                    self.config.q_res_flag[1].enable(&mut region, i)?;
                }
                Self::assign_table_f(
                    &mut region,
                    "co_filt_pad",
                    &self.config.co_filt_pad,
                    &to_field_rows::<F>(&co_filt_pad_u64_ext),
                )?;
                Self::assign_table_f(
                    &mut region,
                    "co_out_pad",
                    &self.config.co_out_pad,
                    &to_field_rows::<F>(&co_out_pad_u64),
                )?;

                // ---------- LS materialization assignments ----------
                for i in 0..lineitem.len() {
                    self.config.q_ls_join.enable(&mut region, i)?;
                    for j in 0..4 {
                        region.assign_advice(
                            || "ls_mat",
                            self.config.ls_mat[j],
                            i,
                            || Value::known(F::from(ls_mat_u64[i][j])),
                        )?;
                    }
                    region.assign_advice(
                        || "ls_pkey",
                        self.config.ls_pkey,
                        i,
                        || {
                            Value::known(F::from(
                                ls_mat_u64[i][0] * SHIFT_NATION + ls_mat_u64[i][1],
                            ))
                        },
                    )?;
                    region.assign_advice(
                        || "cflag_ls",
                        self.config.cflag_ls,
                        i,
                        || Value::known(F::from(ls_cln_b[i])),
                    )?;
                }

                // ---------- LS join/disjoin witnesses + partition permutation ----------
                let ls_join_cells = Self::assign_table_u64(
                    &mut region,
                    "ls_join",
                    &self.config.ls_join,
                    &ls_join_u64, // REAL join rows only (for part_pad linking)
                )?;
                let ls_dis_cells = Self::assign_table_u64(
                    &mut region,
                    "ls_disjoin",
                    &self.config.ls_disjoin,
                    &ls_dis_u64,
                )?;

                // NEW: if ls_pad_extra > 0, we also need to assign extra padding rows into ls_join
                // so that perm_lsort (enabled on n rows) has cells on the LHS.
                for i in ls_join_u64.len()..n {
                    // pad row: [okey=0, nk=PAD_U64, ext=0, disc=0]
                    region.assign_advice(
                        || "ls_join_pad_okey",
                        self.config.ls_join[0],
                        i,
                        || Value::known(F::ZERO),
                    )?;
                    region.assign_advice(
                        || "ls_join_pad_nk",
                        self.config.ls_join[1],
                        i,
                        || Value::known(F::from(PAD_U64)),
                    )?;
                    region.assign_advice(
                        || "ls_join_pad_ext",
                        self.config.ls_join[2],
                        i,
                        || Value::known(F::ZERO),
                    )?;
                    region.assign_advice(
                        || "ls_join_pad_disc",
                        self.config.ls_join[3],
                        i,
                        || Value::known(F::ZERO),
                    )?;
                }

                for i in 0..lineitem.len() {
                    self.config.perm_ls.q_perm1.enable(&mut region, i)?;
                    self.config.perm_ls.q_perm2.enable(&mut region, i)?;
                }
                Self::assign_part_pad_and_link(
                    &mut region,
                    "ls_part_pad",
                    &self.config.ls_part_pad,
                    &ls_part_pad_f,
                    &ls_join_cells,
                    &ls_dis_cells,
                )?;
                // partition side: 1 on the clean LS rows, 0 on the residual ones
                for i in 0..ls_join_u64.len() {
                    self.config.q_cln_flag[2].enable(&mut region, i)?;
                }
                for i in ls_join_u64.len()..(ls_join_u64.len() + ls_dis_u64.len()) {
                    self.config.q_res_flag[2].enable(&mut region, i)?;
                }

                // ---------- condition (3), Pairwise Consistency ----------
                // One selector per relation, enabled over exactly the clean
                // section of its partition group. Each one is both the input
                // selector of the lookups leaving that relation and the table
                // selector of the lookups entering it, so there is no free advice
                // anywhere in the check.
                for i in 0..ls_join_u64.len() {
                    self.config.q_pw_ls.enable(&mut region, i)?;
                }
                for i in 0..co_cln_len {
                    self.config.q_pw_co.enable(&mut region, i)?;
                }
                for i in 0..nr_cln_len {
                    self.config.q_pw_nr.enable(&mut region, i)?;
                }

                // ===================== CARDINALITY PRESERVATION CHECK =====================
                // condition (4) of the One-Pass OBJ over the cluster tree rooted
                // at LS: both multiplicity channels are propagated from the two
                // child bags to LS and their root sums compared.
                //
                // Both children are leaves, so a child row's input-channel
                // multiplicity is its predicate bit and its clean-channel
                // multiplicity is that bit times the clean indicator.
                let cp_rows_co: Vec<[u64; 3]> = (0..orders.len())
                    .map(|i| {
                        [
                            co_pair_u64[i][0] * SHIFT_NATION + co_pair_u64[i][1],
                            co_keep_b[i] as u64,
                            cln_co[i],
                        ]
                    })
                    .collect();
                let cp_rows_nr: Vec<[u64; 3]> = (0..nation.len())
                    .map(|i| [nr_pair_u64[i][0], nr_keep_b[i] as u64, cln_nr[i]])
                    .collect();

                let cp_stage_co = build_cp_stage(&cp_rows_co, PAD_U64);
                let cp_stage_nr = build_cp_stage(&cp_rows_nr, PAD_U64);

                assign_cp_agg(&mut region, &self.config.cp_agg_co, &cp_rows_co, &cp_stage_co)?;
                assign_cp_agg(&mut region, &self.config.cp_agg_nr, &cp_rows_nr, &cp_stage_nr)?;

                // parent side, on the rows of LS (one per lineitem row)
                let ls_pkeys: Vec<u64> = ls_mat_u64
                    .iter()
                    .map(|r| r[0] * SHIFT_NATION + r[1])
                    .collect();
                let ls_nks: Vec<u64> = ls_mat_u64.iter().map(|r| r[1]).collect();
                let fetched_co = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_co,
                    &ls_pkeys,
                    &cp_stage_co,
                    PAD_U64,
                )?;
                let fetched_nr = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_nr,
                    &ls_nks,
                    &cp_stage_nr,
                    PAD_U64,
                )?;

                // root multiplicities and the equality between the two sums
                let cp_mu: Vec<(u64, u64)> = (0..lineitem.len())
                    .map(|i| {
                        (
                            fetched_co[i].0 * fetched_nr[i].0,
                            ls_cln_b[i] * fetched_co[i].1 * fetched_nr[i].1,
                        )
                    })
                    .collect();
                for i in 0..lineitem.len() {
                    self.config.q_cp_mu.enable(&mut region, i)?;
                }
                let (cp_all, cp_cln) = assign_cp_root(&mut region, &self.config.cp_root, &cp_mu)?;
                if !tamper && !all_clean {
                    debug_assert_eq!(
                        cp_all, cp_cln,
                        "cardinality preservation: |R^c join| != |R join|"
                    );
                }

                // ---------- ls_join -> ls_sorted permutation ----------
                for i in 0..n {
                    self.config.perm_lsort.q_perm1.enable(&mut region, i)?;
                    self.config.perm_lsort.q_perm2.enable(&mut region, i)?;
                }

                for i in 0..n {
                    for j in 0..4 {
                        region.assign_advice(
                            || "ls_sorted",
                            self.config.ls_sorted[j],
                            i,
                            || Value::known(F::from(ls_sorted_u64[i][j])),
                        )?;
                    }
                }
                // sentinel row for same_next
                for j in 0..4 {
                    region.assign_advice(
                        || "ls_sorted_sentinel",
                        self.config.ls_sorted[j],
                        n,
                        || Value::known(F::from(0u64)),
                    )?;
                }

                // enable line/accu/selectors and assign helpers
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

                // assign line_rev/run_sum/res_pad/res_sorted
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
                    for j in 0..3 {
                        region.assign_advice(
                            || "res_pad",
                            self.config.res_pad[j],
                            i,
                            || Value::known(F::from(rp[j])),
                        )?;
                    }
                    let rs = res_sorted_u64[i];
                    for j in 0..3 {
                        region.assign_advice(
                            || "res_sorted",
                            self.config.res_sorted[j],
                            i,
                            || Value::known(F::from(rs[j])),
                        )?;
                    }
                }

                // same_prev / same_next
                for i in 1..n {
                    let diff = F::from(ls_sorted_u64[i][1]) - F::from(ls_sorted_u64[i - 1][1]);
                    iz_same_prev_chip.assign(&mut region, i, Value::known(diff))?;
                }
                for i in 0..n {
                    let next_nk = if i + 1 < n {
                        ls_sorted_u64[i + 1][1]
                    } else {
                        0u64
                    };
                    let diff = F::from(next_nk) - F::from(ls_sorted_u64[i][1]);
                    iz_same_next_chip.assign(&mut region, i, Value::known(diff))?;
                }

                // res_pad <-> res_sorted permutation and ORDER BY
                for i in 0..n {
                    self.config.perm_res.q_perm1.enable(&mut region, i)?;
                    self.config.perm_res.q_perm2.enable(&mut region, i)?;
                }
                for i in 0..n.saturating_sub(1) {
                    self.config.q_sort_res.enable(&mut region, i)?;
                    lteq_rev_chip.assign(
                        &mut region,
                        i,
                        &[F::from(res_sorted_u64[i + 1][2])],
                        &[F::from(res_sorted_u64[i][2])],
                    )?;
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

/// Everything Q5's witness generation derives from the base tables before
/// any circuit region is touched.
///
/// Extracted from `assign` VERBATIM so that the bag sizes needed for DP
/// capacity release (`bench_queries::q5_pads`) come from the SAME code the
/// circuit witnesses, and cannot drift from it.
pub struct Q5Derived {
    pub nr_rname_u64: Vec<u64>,
    pub nr_keep_b: Vec<bool>,
    pub nr_pair_u64: Vec<Vec<u64>>,
    pub nr_filtered: Vec<Vec<u64>>,
    pub nk_to_name: HashMap<u64, u64>,
    pub co_nk_u64: Vec<u64>,
    pub co_ge_b: Vec<bool>,
    pub co_lt_b: Vec<bool>,
    pub co_keep_b: Vec<bool>,
    pub co_pair_u64: Vec<Vec<u64>>,
    pub co_filtered: Vec<Vec<u64>>,
    pub ls_mat_u64: Vec<Vec<u64>>,
    pub ls_join_u64: Vec<Vec<u64>>,
    pub ls_dis_u64: Vec<Vec<u64>>,
    pub nr_filt_pad_u64_ext: Vec<Vec<u64>>,
    pub co_filt_pad_u64_ext: Vec<Vec<u64>>,
    pub nr_total: usize,
    pub co_total: usize,
    pub nr_out_pad_u64: Vec<Vec<u64>>,
    pub co_out_pad_u64: Vec<Vec<u64>>,
}

/// Derive Q5's intermediates from the base tables (pure; no circuit access).
///
/// Bag 1 {O,C} is `co_*` and is sized `orders.len() + co_pad_extra`; Bag 2
/// {L,S} is `ls_join_u64` and is sized `ls_join_u64.len() + ls_pad_extra` by
/// the caller. `nr_*` is the N join R dimension filter (one row per nation),
/// not a bag.
#[allow(clippy::too_many_arguments)]
pub fn q5_derive(
    customer: &[Vec<u64>],
    orders: &[Vec<u64>],
    lineitem: &[Vec<u64>],
    supplier: &[Vec<u64>],
    nation: &[Vec<u64>],
    region_file: &[Vec<u64>],
    europe_hash: u64,
    start_ts: u64,
    end_ts: u64,
    nr_pad_extra: usize,
    co_pad_extra: usize,
) -> Q5Derived {
    // NOTE: positional, not compacting -- a filtered-out row is REPLACED by a
    // PAD row at the same index, preserving order and length. Copied verbatim
    // from `assign`; an earlier hand-written "compact then pad" version broke
    // every downstream constraint.
    fn pad_filter_u64(rows: &[Vec<u64>], keep: &[bool], pad: &[u64]) -> Vec<Vec<u64>> {
        rows.iter()
            .zip(keep.iter())
            .map(|(r, &k)| if k { r.clone() } else { pad.to_vec() })
            .collect()
    }
    fn pad_out_u64(filtered: &[Vec<u64>], total: usize, pad: &[u64]) -> Vec<Vec<u64>> {
        let mut out: Vec<Vec<u64>> = Vec::with_capacity(total);
        out.extend_from_slice(filtered);
        while out.len() < total {
            out.push(pad.to_vec());
        }
        out
    }

    // ---------- build shifted maps ----------
    // shift nationkey+1, regionkey+1 to keep 0 as sentinel
    let mut cust_to_nk: HashMap<u64, u64> = HashMap::new();
    for r in customer.iter() {
        cust_to_nk.insert(r[0], r[1] + 1);
    }
    let mut supp_to_nk: HashMap<u64, u64> = HashMap::new();
    for r in supplier.iter() {
        supp_to_nk.insert(r[0], r[1] + 1);
    }
    let mut reg_to_name: HashMap<u64, u64> = HashMap::new();
    for r in region_file.iter() {
        reg_to_name.insert(r[0] + 1, r[1]); // regionkey_shift
    }

    // ---------- NR derivation (nation ⋈ region, filter EUROPE) ----------
    let mut nr_rname_u64 = vec![0u64; nation.len()];
    let mut nr_keep_b = vec![false; nation.len()];
    let mut nr_pair_u64: Vec<Vec<u64>> = vec![vec![0, 0]; nation.len()];

    for i in 0..nation.len() {
        let nk_shift = nation[i][0] + 1;
        let nm_hash = nation[i][1];
        let rk_shift = nation[i][2] + 1;
        let rname = *reg_to_name.get(&rk_shift).unwrap_or(&0);
        nr_rname_u64[i] = rname;
        nr_keep_b[i] = rname == europe_hash;
        nr_pair_u64[i] = vec![nk_shift, nm_hash];
    }
    let nr_filtered: Vec<Vec<u64>> = nr_pair_u64
        .iter()
        .cloned()
        .zip(nr_keep_b.iter())
        .filter(|(_, &k)| k)
        .map(|(r, _)| r)
        .collect();

    let pad2 = vec![PAD_U64; 2];
    let nr_filt_pad_u64 = pad_filter_u64(&nr_pair_u64, &nr_keep_b, &pad2);

    // NEW: allow NR to be padded beyond nation.len()
    let nr_total = nation.len().saturating_add(nr_pad_extra).max(1);
    let mut nr_filt_pad_u64_ext = nr_filt_pad_u64.clone();
    while nr_filt_pad_u64_ext.len() < nr_total {
        nr_filt_pad_u64_ext.push(pad2.clone());
    }
    let nr_out_pad_u64 = pad_out_u64(&nr_filtered, nr_total, &pad2);

    // map nk_shift -> name_hash for filtered NR
    let mut nk_to_name: HashMap<u64, u64> = HashMap::new();
    for r in nr_filtered.iter() {
        nk_to_name.insert(r[0], r[1]);
    }

    // ---------- CO derivation (orders ⋈ customer, filter by date range) ----------
    let mut co_nk_u64 = vec![0u64; orders.len()];
    let mut co_ge_b = vec![false; orders.len()];
    let mut co_lt_b = vec![false; orders.len()];
    let mut co_keep_b = vec![false; orders.len()];
    let mut co_pair_u64: Vec<Vec<u64>> = vec![vec![0, 0]; orders.len()];

    for i in 0..orders.len() {
        let odate = orders[i][0];
        let cust = orders[i][1];
        let okey = orders[i][2];

        let nk_shift = *cust_to_nk.get(&cust).unwrap_or(&0);
        co_nk_u64[i] = nk_shift;

        co_ge_b[i] = start_ts <= odate;
        co_lt_b[i] = odate < end_ts;
        co_keep_b[i] = co_ge_b[i] && co_lt_b[i];

        co_pair_u64[i] = vec![okey, nk_shift];
    }

    let co_filtered: Vec<Vec<u64>> = co_pair_u64
        .iter()
        .cloned()
        .zip(co_keep_b.iter())
        .filter(|(_, &k)| k)
        .map(|(r, _)| r)
        .collect();

    let co_filt_pad_u64 = pad_filter_u64(&co_pair_u64, &co_keep_b, &pad2);

    // NEW: allow CO to be padded beyond orders.len()
    let co_total = orders.len().saturating_add(co_pad_extra).max(1);
    let mut co_filt_pad_u64_ext = co_filt_pad_u64.clone();
    while co_filt_pad_u64_ext.len() < co_total {
        co_filt_pad_u64_ext.push(pad2.clone());
    }
    let co_out_pad_u64 = pad_out_u64(&co_filtered, co_total, &pad2);

    // build key sets for join
    let co_set: HashSet<u64> = co_filtered
        .iter()
        .map(|r| r[0] * SHIFT_NATION + r[1])
        .collect();

    let nr_set: HashSet<u64> = nr_filtered.iter().map(|r| r[0]).collect();

    // ---------- LS materialization (lineitem ⋈ supplier) ----------
    let mut ls_mat_u64: Vec<Vec<u64>> = vec![vec![0, 0, 0, 0]; lineitem.len()];
    for i in 0..lineitem.len() {
        let okey = lineitem[i][0];
        let supp = lineitem[i][1];
        let ext = lineitem[i][2];
        let disc = lineitem[i][3];
        let nk_shift = *supp_to_nk.get(&supp).unwrap_or(&0);
        ls_mat_u64[i] = vec![okey, nk_shift, ext, disc];
    }

    // ---------- partition LS into join/disjoin (relative to CO and NR) ----------
    let mut ls_join_u64 = vec![];
    let mut ls_dis_u64 = vec![];
    for r in ls_mat_u64.iter() {
        let packed = r[0] * SHIFT_NATION + r[1];
        let ok = co_set.contains(&packed) && nr_set.contains(&r[1]);
        if ok {
            ls_join_u64.push(r.clone());
        } else {
            ls_dis_u64.push(r.clone());
        }
    }

    Q5Derived {
        nr_rname_u64,
        nr_keep_b,
        nr_pair_u64,
        nr_filtered,
        nk_to_name,
        co_nk_u64,
        co_ge_b,
        co_lt_b,
        co_keep_b,
        co_pair_u64,
        co_filtered,
        ls_mat_u64,
        ls_join_u64,
        ls_dis_u64,
        nr_filt_pad_u64_ext,
        co_filt_pad_u64_ext,
        nr_total,
        co_total,
        nr_out_pad_u64,
        co_out_pad_u64,
    }
}

// ---------------- Circuit wrapper ----------------
pub struct MyCircuit<F> {
    // base tables (UNSHIFTED keys as loaded from .tbl/.csv)
    pub customer: Vec<Vec<u64>>, // [c_custkey, c_nationkey]
    pub orders: Vec<Vec<u64>>,   // [o_orderdate_ts, o_custkey, o_orderkey]
    pub lineitem: Vec<Vec<u64>>, // [l_orderkey, l_suppkey, l_extendedprice_scaled, l_discount_scaled]
    pub supplier: Vec<Vec<u64>>, // [s_suppkey, s_nationkey]
    pub nation: Vec<Vec<u64>>,   // [n_nationkey, n_name_hash, n_regionkey]
    pub region: Vec<Vec<u64>>,   // [r_regionkey, r_name_hash]

    // query parameters
    pub europe_hash: u64,
    pub start_ts: u64,
    pub end_ts: u64,

    // NEW: padding knobs (analogy to bag1_pad_extra / bag2_pad_extra)
    pub nr_pad_extra: usize,
    pub co_pad_extra: usize,
    pub ls_pad_extra: usize,

    pub _marker: PhantomData<F>,
}

impl<F: Copy + Default> Default for MyCircuit<F> {
    fn default() -> Self {
        Self {
            customer: vec![],
            orders: vec![],
            lineitem: vec![],
            supplier: vec![],
            nation: vec![],
            region: vec![],
            europe_hash: 0,
            start_ts: 0,
            end_ts: 0,
            nr_pad_extra: 0,
            co_pad_extra: 0,
            ls_pad_extra: 0,
            _marker: PhantomData,
        }
    }
}

impl<F: Field + Ord> Circuit<F> for MyCircuit<F> {
    type Config = Q5Config<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        Q5Chip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = Q5Chip::construct(config);

        let out_cell = chip.assign(
            &mut layouter,
            self.customer.clone(),
            self.orders.clone(),
            self.lineitem.clone(),
            self.supplier.clone(),
            self.nation.clone(),
            self.region.clone(),
            self.europe_hash,
            self.start_ts,
            self.end_ts,
            self.nr_pad_extra,
            self.co_pad_extra,
            self.ls_pad_extra,
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
    use rand::rngs::OsRng;

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
        let mut result = 0u64;
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
                let datetime: DateTime<Utc> = DateTime::<Utc>::from_utc(date.and_hms(0, 0, 0), Utc);
                datetime.timestamp() as u64
            }
            Err(_) => 0,
        }
    }

    #[test]
    #[ignore = "inherited heavy end-to-end proof; the fast check is test_cardinality_preservation"]
    fn test_1() {
        // ---------------- paths ----------------
        let customer_file_path = &crate::paths::data_file("customer.tbl");
        let orders_file_path = &crate::paths::data_file("orders.tbl");
        let lineitem_file_path = &crate::paths::data_file("lineitem.tbl");
        let supplier_file_path = &crate::paths::data_file("supplier.tbl");
        let nation_file_path = &crate::paths::data_file("nation.tbl");
        let region_file_path = &crate::paths::data_file("region.cvs"); // keep your repo spelling

        // customer: [c_custkey, c_nationkey]
        let mut customer: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::customer_read_records_from_file(customer_file_path) {
            customer = records
                .iter()
                .map(|r| vec![r.c_custkey, r.c_nationkey])
                .collect();
        }

        // orders: [o_orderdate_ts, o_custkey, o_orderkey]
        let mut orders: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::orders_read_records_from_file(orders_file_path) {
            orders = records
                .iter()
                .map(|r| vec![date_to_timestamp(&r.o_orderdate), r.o_custkey, r.o_orderkey])
                .collect();
        }

        // lineitem: [l_orderkey, l_suppkey, l_extendedprice_scaled, l_discount_scaled]
        let mut lineitem: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::lineitem_read_records_from_file(lineitem_file_path) {
            lineitem = records
                .iter()
                .map(|r| {
                    vec![
                        r.l_orderkey,
                        r.l_suppkey,
                        scale_by_1000(r.l_extendedprice),
                        scale_by_1000(r.l_discount),
                    ]
                })
                .collect();
        }

        // supplier: [s_suppkey, s_nationkey]
        let mut supplier: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::supplier_read_records_from_file(supplier_file_path) {
            supplier = records
                .iter()
                .map(|r| vec![r.s_suppkey, r.s_nationkey])
                .collect();
        }

        // nation: [n_nationkey, n_name_hash, n_regionkey]
        let mut nation: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::nation_read_records_from_file(nation_file_path) {
            nation = records
                .iter()
                .map(|r| vec![r.n_nationkey, string_to_u64(&r.n_name), r.n_regionkey])
                .collect();
        }

        // region: [r_regionkey, r_name_hash]
        let mut region: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::region_read_records_from_cvs(region_file_path) {
            region = records
                .iter()
                .map(|r| vec![r.r_regionkey, string_to_u64(&r.r_name)])
                .collect();
        }

        // ---------------- query params ----------------
        let europe_hash = string_to_u64("EUROPE");
        let start_ts = date_to_timestamp("1997-01-01");
        let end_ts = date_to_timestamp("1998-01-01"); // strict < end

        let privacy = match std::env::var("VPJOIN_PRIVACY")
            .as_deref()
            .unwrap_or("legacy")
        {
            "rjs" => crate::bench_queries::Privacy::Rjs,
            "dp" => crate::bench_queries::Privacy::Dp {
                epsilon: std::env::var("VPJOIN_EPS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.1),
                delta: std::env::var("VPJOIN_DELTA")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1e-5),
            },
            _ => crate::bench_queries::Privacy::Legacy,
        };
        let (nr_pad_extra, co_pad_extra, ls_pad_extra) = crate::bench_queries::q5_pads(privacy);
        println!(
            "[q5 test] privacy={} pads: nr={} co={} ls={}",
            privacy.label(),
            nr_pad_extra,
            co_pad_extra,
            ls_pad_extra
        );

        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            supplier,
            nation,
            region,
            europe_hash,
            start_ts,
            end_ts,
            nr_pad_extra,
            co_pad_extra,
            ls_pad_extra,
            _marker: PhantomData,
        };

        let public_input: Vec<Fp> = vec![Fp::from(1u64)];

        let k = crate::bench_queries::degree_for("q5", "tpch-60K", privacy);

        // let test = true;
        let test = false;

        if test {
            let prover = MockProver::run(k, &circuit, vec![public_input]).unwrap();
            prover.assert_satisfied();
        } else {
            let proof_path = &crate::paths::proof_file("proof_q5_obj_dp_test");
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
    }

    /// Fast correctness check of the Cardinality Preservation Check: a truncated
    /// slice of the dataset under MockProver, which verifies every gate, shuffle
    /// and lookup of the circuit without paying for a real proof.
    #[test]
    fn test_cardinality_preservation() {
        let k = 14;

        // customer, supplier, nation and region are taken whole because the three
        // tuple lookups of the bag materialization require every referenced key to
        // be present in its dimension table. orders and lineitem are truncated.
        //
        // On this slice every bag really does split, so no section of any
        // partition is vacuous: NR is 4 clean / 1 residual / 20 pad rows, CO is
        // 13 clean / 601 residual / 1386 pad rows, LS is 13 clean / 7987
        // residual, and both sides of condition (4) count 13 join results.
        const N_ORD: usize = 2000;
        const N_LINE: usize = 8000;

        let mut customer: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::customer_read_records_from_file(
            &crate::paths::data_file("customer.tbl"),
        ) {
            customer = records
                .iter()
                .map(|r| vec![r.c_custkey, r.c_nationkey])
                .collect();
        }

        let mut orders: Vec<Vec<u64>> = vec![];
        if let Ok(records) =
            data_processing::orders_read_records_from_file(&crate::paths::data_file("orders.tbl"))
        {
            orders = records
                .iter()
                .take(N_ORD)
                .map(|r| vec![date_to_timestamp(&r.o_orderdate), r.o_custkey, r.o_orderkey])
                .collect();
        }

        let mut lineitem: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::lineitem_read_records_from_file(
            &crate::paths::data_file("lineitem.tbl"),
        ) {
            lineitem = records
                .iter()
                .take(N_LINE)
                .map(|r| {
                    vec![
                        r.l_orderkey,
                        r.l_suppkey,
                        scale_by_1000(r.l_extendedprice),
                        scale_by_1000(r.l_discount),
                    ]
                })
                .collect();
        }

        let mut supplier: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::supplier_read_records_from_file(
            &crate::paths::data_file("supplier.tbl"),
        ) {
            supplier = records
                .iter()
                .map(|r| vec![r.s_suppkey, r.s_nationkey])
                .collect();
        }

        let mut nation: Vec<Vec<u64>> = vec![];
        if let Ok(records) =
            data_processing::nation_read_records_from_file(&crate::paths::data_file("nation.tbl"))
        {
            nation = records
                .iter()
                .map(|r| vec![r.n_nationkey, string_to_u64(&r.n_name), r.n_regionkey])
                .collect();
        }

        let mut region: Vec<Vec<u64>> = vec![];
        if let Ok(records) =
            data_processing::region_read_records_from_cvs(&crate::paths::data_file("region.cvs"))
        {
            region = records
                .iter()
                .map(|r| vec![r.r_regionkey, string_to_u64(&r.r_name)])
                .collect();
        }

        assert!(
            !customer.is_empty()
                && !orders.is_empty()
                && !lineitem.is_empty()
                && !supplier.is_empty()
                && !nation.is_empty()
                && !region.is_empty(),
            "dataset files not found under {}",
            crate::paths::data_file("customer.tbl")
        );

        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            supplier,
            nation,
            region,
            europe_hash: string_to_u64("EUROPE"),
            start_ts: date_to_timestamp("1996-01-01"),
            end_ts: date_to_timestamp("1998-01-01"),
            nr_pad_extra: 0,
            co_pad_extra: 0,
            ls_pad_extra: 0,
            _marker: PhantomData,
        };

        let prover = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        prover.assert_satisfied();

        // Negative direction: the same witness with one joinable LS tuple hidden
        // in the residual side, and the CO and NR bags re-reduced around it so
        // that Conservation, Non-Membership and Pairwise Consistency all still
        // hold. Only condition (4) can see this, so the circuit must now reject.
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

        // Third direction: no reduction at all, every tuple that passes its own
        // predicate declared clean and all three residual sections empty.
        // Conservation holds and condition (4) is satisfied for free, since both
        // channels then agree row by row, so this is the escape that condition (3)
        // exists to close. The two child bags now carry dangling tuples as well,
        // which is what the mirror direction of each edge catches.
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
} // end mod tests

// nation:   25
// part:     2000
// customer: 1500
// orders:   15000
// lineitem: 60175
// partsupp: 8000
// supplier: 100
// region:   5
