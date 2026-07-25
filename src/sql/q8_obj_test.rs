//! TPC-H Q8 under the updated One-Pass OBJ.
//!
//! Same query, same filters and same aggregation as `q8_obj.rs`. What this file
//! adds is the certification story of the gate, and in particular condition (4),
//! the one that rules out a joinable tuple hidden in the residual side.
//!
//! `q8_obj.rs` has no partition at all: it verifies the join by one-directional
//! lookups out of a prefix of the permuted lineitem table, and the prefix length
//! is a free choice of the prover. Nothing bounds the clean part from below, so
//! there is no old condition (4) here to replace; the partition and the check
//! are both introduced.
//!
//! The eight relations of the query are laid out as the acyclic join tree
//!
//!   lineitem (root)
//!     |- part                                  l_partkey  = p_partkey
//!     |- supplier                              l_suppkey  = s_suppkey
//!     |    `- nation as n2                     s_nationkey = n_nationkey
//!     `- orders                                l_orderkey = o_orderkey
//!          `- customer                         o_custkey  = c_custkey
//!               `- nation as n1                c_nationkey = n_nationkey
//!                    `- region                 n_regionkey = r_regionkey
//!
//! and each relation gets a Conservation Check (a `PermAnyChip` shuffle) between
//! its base rows, which carry `keep * c`, and a partition column group laid out
//! as `[clean rows | residual rows | pad rows]`. Two fresh selectors pin the
//! indicator to `1` on the clean section and to `0` on the residual one, so the
//! multiset equality forces the base-row indicator to mark exactly the
//! occurrences that went to `R^c`. Clean and residual live in one group, so
//! Non-Membership is structural.
//!
//! Condition (3), Pairwise Consistency, is two mutual Membership Checks per
//! edge of that tree over the CLEAN sections of the two partitions:
//!
//!   pi_K_ij(R_i^c) == pi_K_ij(R_j^c)     for every edge (R_i, R_j)
//!
//! Seven edges, so fourteen `pw: ` lookups, each looking one clean key column
//! up directly in the adjacent relation's clean key column. There is no
//! intermediate key table: an earlier version routed each direction through a
//! `pw_tbl` advice column holding the deduplicated key set, but nothing bound
//! that column to the relation it claimed to enumerate, so a prover could put
//! `pi_K(R_i^c)` into the table `R_i^c` looks into and `pi_K(R_j^c)` into the
//! other and satisfy both directions for an arbitrary partition. Looking the two
//! columns up in each other leaves no free advice to forge, and the two
//! containments together are the set equality. The key column always comes from
//! the partition group, never from the base relation: a lookup over the base
//! rows would only certify membership in `R_j`, which is the weaker statement
//! `q8_obj.rs` already made. Every key is shifted by `SHIFT_ID` on both sides of
//! every lookup, because a `lookup_any` expression is evaluated on every row of
//! the circuit and is `0` wherever its selector is off, so `0` is always in the
//! table and an unshifted key `0` would be accepted for free. Without condition
//! (3) condition (4) has a trivial
//! escape: the all-clean partition conserves every relation and makes both
//! channels of the Cardinality Preservation Check agree on every row, so a
//! dangling tuple left in `R_i^c` needs the key sets of an edge to disagree
//! before anything sees it.
//!
//! Condition (4) is then the Cardinality Preservation Check of
//! `crate::circuits::card_preserve`: one traversal of the tree above carrying
//! two multiplicities per tuple,
//!
//!   mu_all(t) = pred(t)        * prod_j sigma_all_j(t[K_j])
//!   mu_cln(t) = c(t) * pred(t) * prod_j sigma_cln_j(t[K_j])
//!
//! one counting join extensions over the inputs and one over the clean
//! instance, and a single equality between the two root sums. `pred` is the
//! per-tuple predicate bit the circuit already forces: `p_keep` on part,
//! `o_keep` on orders, the new `r_keep` on region (r_name == MIDDLE EAST), and
//! `1` on lineitem, supplier, customer and nation. The products are folded
//! through intermediate advice columns so that no new gate exceeds degree 4.
//!
//! Every edge of this tree joins on a primary key of the child, so both
//! multiplicities are 0/1 and the root sums count lineitem rows: the input
//! channel counts the joinable ones and the clean channel the ones the prover
//! declared clean. That is exactly the quantity the prefix trick leaves free.
//!
//! Nationkeys and regionkeys start at 0 in TPC-H and key 0 is reserved for the
//! gadget's dummy table row, so those two keys are shifted by `SHIFT_ID` on
//! both sides of their edges, which preserves the equality join.
//!
//! Everything else, including the volume arithmetic, the per-year sums and the
//! two ORDER BY proofs, is unchanged from `q8_obj.rs`.
//
// Halo2 “object-style” circuit for TPC-H Query 8 (simplified):
//
// - We treat `o_year` as a preprocessed integer input (you said we may simplify/ignore extract()).
// - Date filter is approximated by `o_year ∈ {1995, 1996}`.
// - volume is computed in scaled integers: vol = l_extendedprice * (SCALE - l_discount)
//   (both extprice and discount are expected pre-scaled by SCALE=1000).
// - mkt_share is proved as a field fraction: share * den == num
//   (no fixed-point conversion; ratio is still correct because both sums share the same scaling).
//
// Tables expected (as Vec<Vec<u64>>):
// region   : [r_regionkey, r_name_hash]
// nation   : [n_nationkey, n_regionkey, n_name_hash]
// customer : [c_custkey, c_nationkey]
// orders   : [o_orderkey, o_custkey, o_year]
// part     : [p_partkey, p_type_hash]
// supplier : [s_suppkey, s_nationkey]
// lineitem : [l_orderkey, l_partkey, l_suppkey, l_extendedprice_scaled, l_discount_scaled]
//
// Parameter inputs:
// - cond_nation_hash        (':1' in query)
// - const_region_name_hash  (hash("MIDDLE EAST"))
// - const_part_type_hash    (hash("PROMO BRUSHED COPPER"))

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

// ----------------- tuning -----------------
const NUM_BYTES: usize = 5;
const SCALE: u64 = 1000;

// ----------------- paddings -----------------
// year ASC => pad year should be "very large" so pads sort to the end
const MAX_SENTINEL: u64 = (1u64 << (8 * NUM_BYTES)) - 1; // 2^40-1
const PAD_YEAR: u64 = MAX_SENTINEL;

// Part: [p_partkey, p_type_hash]
const PAD_PKEY: u64 = MAX_SENTINEL;
const PAD_PTYPE: u64 = MAX_SENTINEL;

// Orders (filtered table used for lookup): [o_orderkey, o_year, o_custkey]
// o_custkey rides along because the orders -> customer edge of condition (3)
// needs the key on the partition side, not on the base rows.
const PAD_OKEY: u64 = MAX_SENTINEL;
const PAD_OYEAR: u64 = PAD_YEAR;
const PAD_OCUST: u64 = MAX_SENTINEL;

// Lineitem join pad: [l_orderkey,l_partkey,l_suppkey,l_ext,l_disc]
const PAD_LOKEY: u64 = MAX_SENTINEL;
const PAD_LPKEY: u64 = MAX_SENTINEL;
const PAD_LSKEY: u64 = MAX_SENTINEL;
const PAD_LEXT: u64 = 0;
const PAD_LDISC: u64 = 0;

// all_nations row: [year, volume, nation_hash]
const PAD_AN_YEAR: u64 = PAD_YEAR;
const PAD_AN_VOL: u64 = 0;
const PAD_AN_NAT: u64 = MAX_SENTINEL;

// result row: [year, num, den, share]
const PAD_RES_YEAR: u64 = PAD_YEAR;
const PAD_RES_NUM: u64 = 0;
const PAD_RES_DEN: u64 = 0;
// share is a field element; for pads set 0

// Region partition pad: [r_regionkey, r_name_hash] plus the clean indicator
const PAD_RKEY: u64 = MAX_SENTINEL;
const PAD_RNAME: u64 = MAX_SENTINEL;

// Key 0 is reserved for the dummy row of the key-indexed tables of the
// Cardinality Preservation Check, and n_nationkey / r_regionkey start at 0 in
// TPC-H. Both sides of those two edges are shifted by this constant, which
// preserves the equality join.
const SHIFT_ID: u64 = 1;
// ------------------------------------------------

/// Test hook, off in every benchmark path: when set, the prover moves one
/// joinable lineitem tuple to the residual side and re-reduces the neighbours
/// around it, so the partition still passes Conservation, Non-Membership and
/// Pairwise Consistency and only condition (4) can catch it. This is exactly
/// the cheat a prefix-only or residual-side-only argument misses, so the
/// negative test in this module is what shows the Cardinality Preservation
/// Check is not vacuous.
pub static HIDE_ONE_CLEAN_TUPLE: AtomicBool = AtomicBool::new(false);

/// Test hook, off in every benchmark path: when set, the prover skips the
/// semijoin reduction entirely and declares every real tuple clean, so the
/// residual section of every relation stays empty. Conservation still holds and
/// both channels of condition (4) then agree on every row, so `sum_cln ==
/// sum_all` passes for free. This is exactly the escape that Pairwise
/// Consistency has to close, and the third direction of the test in this module
/// is what shows it does.
pub static MARK_ALL_CLEAN: AtomicBool = AtomicBool::new(false);

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

#[derive(Clone, Debug)]
pub struct TestCircuitConfig<F: Field + Ord> {
    // public instance
    instance: Column<Instance>,
    instance_test: Column<Advice>,

    // base tables
    region: Vec<Column<Advice>>,   // 2
    nation: Vec<Column<Advice>>,   // 3
    customer: Vec<Column<Advice>>, // 2
    orders: Vec<Column<Advice>>,   // 3
    part: Vec<Column<Advice>>,     // 2
    supplier: Vec<Column<Advice>>, // 2
    lineitem: Vec<Column<Advice>>, // 5

    // parameters (advice constants repeated down the region)
    cond_nation: Column<Advice>, // ':1' hash
    const_rname: Column<Advice>, // hash("MIDDLE EAST")
    const_ptype: Column<Advice>, // hash("PROMO BRUSHED COPPER")
    target_rkey: Column<Advice>, // witness: regionkey for MIDDLE EAST (proved by lookup)

    // ---------- region membership proof for target_rkey ----------
    q_tbl_region: Selector,
    q_lkp_target_region: Selector,

    // ---------- clean indicator per base row ----------
    // one per relation of the join tree, in the order
    // [region, nation as n1, nation as n2, customer, orders, part, supplier,
    //  lineitem]
    cflag: Vec<Column<Advice>>, // 8

    // ---------- part type filter (PROMO BRUSHED COPPER) ----------
    q_part_pred: Selector,
    q_part_link: Selector,
    p_keep: Column<Advice>,
    iz_part_type: IsZeroConfig<F>,
    p_filt_pad: Vec<Column<Advice>>, // 3 (the last column is keep * c)
    p_join_pad: Vec<Column<Advice>>, // 3 ([clean | residual | pad] rows)
    perm_part: PermAnyConfig,
    q_emit: Selector,
    q_emit_last: Selector,

    // ---------- customer -> nation(region) attach, and customer keep ----------
    q_tbl_nation: Selector,
    q_tbl_customer: Selector,
    q_lkp_c_nat_region: Selector, // (c_nationkey, c_regionkey) ∈ nation
    c_regionkey: Column<Advice>,
    c_keep: Column<Advice>,
    iz_c_keep: IsZeroConfig<F>,

    // ---------- orders filter: year in {1995,1996} and customer in region ----------
    q_orders_attach_c: Selector, // (o_custkey, o_c_nationkey) ∈ customer
    o_c_nationkey: Column<Advice>, // attached
    q_orders_attach_r: Selector, // (o_c_nationkey, o_c_regionkey) ∈ nation
    o_c_regionkey: Column<Advice>, // attached
    o_c_keep: Column<Advice>,    // computed is_zero(o_c_regionkey - target_rkey)
    iz_o_c_keep: IsZeroConfig<F>,
    year_keep: Column<Advice>, // computed is_zero((year-1995)*(year-1996))
    iz_year_keep: IsZeroConfig<F>,
    o_keep: Column<Advice>,       // o_keep = year_keep * o_c_keep
    q_orders_keep_gate: Selector, // enforces o_keep
    // filtered orders table used for lineitem lookup
    q_orders_filt_gate: Selector,
    o_filt_pad: Vec<Column<Advice>>, // 4 (keep? [okey,oyear,ckey,c] : PAD)
    o_join_pad: Vec<Column<Advice>>, // 4 ([clean | residual | pad]) length=orders.len()
    perm_orders: PermAnyConfig,

    // ---------- lineitem reorder: clean rows then residual then PAD ----------
    l_join_pad: Vec<Column<Advice>>, // 6, length=lineitem.len()
    perm_lineitem: PermAnyConfig,

    // ---------- join lookups (only enabled on join prefix rows) ----------
    q_lkp_partkey: Selector,  // l_partkey ∈ p_join_pad
    q_tbl_partf: Selector,    // table enable for p_join_pad
    q_lkp_orders: Selector,   // (l_orderkey, l_year) ∈ o_join_pad
    q_tbl_ordersf: Selector,  // table enable for o_join_pad
    q_lkp_supplier: Selector, // (l_suppkey, l_s_nationkey) ∈ supplier
    q_tbl_supplier: Selector,
    q_lkp_nation2: Selector, // (l_s_nationkey, l_nation_hash) ∈ nation (n2)
    // attached on join rows
    l_year: Column<Advice>,
    l_s_nationkey: Column<Advice>,
    l_nation_hash: Column<Advice>,

    // ---------- volume and all_nations ----------
    q_volume: Selector,
    volume: Column<Advice>, // vol = ext*(SCALE - disc)

    all_nations: Vec<Column<Advice>>, // 3 [year, volume, nation_hash], len=lineitem.len()
    all_nations_sorted: Vec<Column<Advice>>, // 3
    perm_all_nations: PermAnyConfig,

    // ORDER BY year ASC on all_nations_sorted
    q_sort_year: Selector,
    lt_year_cur_next: LtConfig<F, NUM_BYTES>,
    iz_year_eq: IsZeroConfig<F>,

    // ---------- grouping per year: sums ----------
    q_is_cond: Selector,
    cond_flag: Column<Advice>, // (nation == ':1')
    iz_is_cond: IsZeroConfig<F>,

    q_first: Selector,
    q_accu: Selector,
    q_line: Selector,

    iz_same_prev: IsZeroConfig<F>,
    iz_same_next: IsZeroConfig<F>,

    run_num: Column<Advice>,
    run_den: Column<Advice>,

    // emit results (sparse)
    res_pad: Vec<Column<Advice>>, // 4 [year, num, den, share]
    // compact results (group rows first, pads last)
    res_sorted: Vec<Column<Advice>>, // 4
    perm_res: PermAnyConfig,

    // order by year on res_sorted
    q_sort_res: Selector,
    lt_res_year_cur_next: LtConfig<F, NUM_BYTES>,
    iz_res_year_eq: IsZeroConfig<F>,

    // share constraint on res_sorted: share * den = num
    q_share: Selector,

    // ---------------- Conservation Checks of the introduced partition ----------------
    // region also gets its in-relation predicate here, r_name == MIDDLE EAST,
    // because the input channel of condition (4) has to apply it.
    q_region_pred: Selector,
    r_keep: Column<Advice>,
    iz_r_keep: IsZeroConfig<F>,
    r_filt_pad: Vec<Column<Advice>>, // 3
    r_part_pad: Vec<Column<Advice>>, // 3
    perm_region: PermAnyConfig,

    // the relations that carry no predicate need no filt/pad link gate: their
    // base rows plus the raw indicator are the input side of the shuffle
    n1_part_pad: Vec<Column<Advice>>, // 4
    perm_nation1: PermAnyConfig,
    n2_part_pad: Vec<Column<Advice>>, // 4
    perm_nation2: PermAnyConfig,
    c_part_pad: Vec<Column<Advice>>, // 3
    perm_customer: PermAnyConfig,
    s_part_pad: Vec<Column<Advice>>, // 3
    perm_supplier: PermAnyConfig,

    // clean/residual flag on the partition side of every Conservation Check
    q_cln_flag: Vec<Selector>, // 8, rows of R^c: flag == 1
    q_res_flag: Vec<Selector>, // 8, rows of R^r: flag == 0

    // ---------------- Pairwise Consistency (condition (3)) ----------------
    // Two mutual Membership Checks per join-tree edge, both over the clean
    // sections of the two partition groups. One input selector per relation,
    // enabled over the rows [0, n_cln) of its partition group, in the same
    // relation order as q_cln_flag: [region, n1, n2, customer, orders, part,
    // supplier, lineitem].
    // The same eight selectors are also the table selectors: a direction of an
    // edge reads the input out of one relation's clean section and the table out
    // of the other's, so there is no separate key table and nothing to forge.
    q_pw_in: Vec<Selector>, // 8

    // ---------------- Cardinality Preservation Check (condition (4)) ----------------
    // shifted copies of the two keys that can be 0
    q_nation_cp: Selector,
    q_cust_cp: Selector,
    q_supp_cp: Selector,
    nk_shift: Column<Advice>,   // n_nationkey + SHIFT_ID, child key of n1 and n2
    n_rk_shift: Column<Advice>, // n_regionkey + SHIFT_ID, parent key of n1 -> region
    rk_shift: Column<Advice>,   // r_regionkey + SHIFT_ID, child key of region
    c_nk_shift: Column<Advice>, // c_nationkey + SHIFT_ID, parent key of customer -> n1
    s_nk_shift: Column<Advice>, // s_nationkey + SHIFT_ID, parent key of supplier -> n2

    // one child stage and one parent side per tree edge
    cp_agg_region: CpAggConfig<F, NUM_BYTES>,
    cp_join_region: CpJoinConfig<F, NUM_BYTES>, // nation as n1 -> region
    cp_agg_n1: CpAggConfig<F, NUM_BYTES>,
    cp_join_n1: CpJoinConfig<F, NUM_BYTES>, // customer -> nation as n1
    cp_agg_cust: CpAggConfig<F, NUM_BYTES>,
    cp_join_cust: CpJoinConfig<F, NUM_BYTES>, // orders -> customer
    cp_agg_ord: CpAggConfig<F, NUM_BYTES>,
    cp_join_ord: CpJoinConfig<F, NUM_BYTES>, // lineitem -> orders
    cp_agg_n2: CpAggConfig<F, NUM_BYTES>,
    cp_join_n2: CpJoinConfig<F, NUM_BYTES>, // supplier -> nation as n2
    cp_agg_supp: CpAggConfig<F, NUM_BYTES>,
    cp_join_supp: CpJoinConfig<F, NUM_BYTES>, // lineitem -> supplier
    cp_agg_part: CpAggConfig<F, NUM_BYTES>,
    cp_join_part: CpJoinConfig<F, NUM_BYTES>, // lineitem -> part
    cp_root: CpRootConfig,

    // the multiplicities the caller owns: one constant, the clean channel of
    // each internal node, and the two folded products at the root
    cp_one: Column<Advice>,         // pred of the nation leaf, constant 1
    cp_mu_n1: Column<Advice>,       // clean multiplicity of nation as n1
    cp_mu_cust: Column<Advice>,     // clean multiplicity of customer
    cp_mu_supp: Column<Advice>,     // clean multiplicity of supplier
    cp_mu_ord: Vec<Column<Advice>>, // 2: both multiplicities of orders
    cp_aux: Vec<Column<Advice>>,    // 2: folded partial products at the root
    q_cp_n1: Selector,
    q_cp_cust: Selector,
    q_cp_supp: Selector,
    q_cp_ord: Selector,
    q_cp_root: Selector,
}

#[derive(Clone, Debug)]
pub struct TestChip<F: Field + Ord> {
    config: TestCircuitConfig<F>,
}

impl<F: Field + Ord> TestChip<F> {
    pub fn construct(config: TestCircuitConfig<F>) -> Self {
        Self { config }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> TestCircuitConfig<F> {
        // instance
        let instance = meta.instance_column();
        meta.enable_equality(instance);
        let instance_test = meta.advice_column();
        meta.enable_equality(instance_test);

        // base tables
        let region = vec![meta.advice_column(), meta.advice_column()];
        let nation = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let customer = vec![meta.advice_column(), meta.advice_column()];
        let orders = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let part = vec![meta.advice_column(), meta.advice_column()];
        let supplier = vec![meta.advice_column(), meta.advice_column()];
        let lineitem = (0..5).map(|_| meta.advice_column()).collect::<Vec<_>>();

        // parameters
        let cond_nation = meta.advice_column();
        let const_rname = meta.advice_column();
        let const_ptype = meta.advice_column();
        let target_rkey = meta.advice_column();

        // enable equality on all advice columns that will be used in perms / constrain_equal
        for &c in region
            .iter()
            .chain(nation.iter())
            .chain(customer.iter())
            .chain(orders.iter())
            .chain(part.iter())
            .chain(supplier.iter())
            .chain(lineitem.iter())
        {
            meta.enable_equality(c);
        }
        meta.enable_equality(cond_nation);
        meta.enable_equality(const_rname);
        meta.enable_equality(const_ptype);
        meta.enable_equality(target_rkey);

        // clean indicator per base row, one column per relation of the join
        // tree: [region, nation as n1, nation as n2, customer, orders, part,
        // supplier, lineitem]
        let cflag = (0..8).map(|_| meta.advice_column()).collect::<Vec<_>>();

        // ---------- region membership lookup for target_rkey ----------
        let q_tbl_region = meta.complex_selector();
        let q_lkp_target_region = meta.complex_selector();

        // we will lookup (target_rkey, const_rname) in region table (r_regionkey, r_name_hash)
        meta.lookup_any("target_rkey is the regionkey of MIDDLE EAST", |m| {
            let q_in = m.query_selector(q_lkp_target_region);
            let q_t = m.query_selector(q_tbl_region);
            vec![
                (
                    q_in.clone() * m.query_advice(target_rkey, Rotation::cur()),
                    q_t.clone() * m.query_advice(region[0], Rotation::cur()),
                ),
                (
                    q_in * m.query_advice(const_rname, Rotation::cur()),
                    q_t * m.query_advice(region[1], Rotation::cur()),
                ),
            ]
        });

        // ---------- part filter p_type == const_ptype ----------
        let q_part_pred = meta.selector();
        let q_part_link = meta.selector();

        let p_keep = meta.advice_column();
        meta.enable_equality(p_keep);

        let aux_part = meta.advice_column();
        meta.enable_equality(aux_part);

        let iz_part_type = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_part_pred),
            |m| {
                m.query_advice(part[1], Rotation::cur())
                    - m.query_advice(const_ptype, Rotation::cur())
            },
            aux_part,
        );

        // p_keep = iz_part_type.expr() and boolean
        meta.create_gate("p_keep = (p_type == const_ptype)", |m| {
            let q = m.query_selector(q_part_pred);
            let keep = m.query_advice(p_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone() * (keep.clone() - iz_part_type.expr()),
                q * keep.clone() * (one - keep),
            ]
        });

        // filtered padded and join padded, one column wider than in q8_obj.rs:
        // the last column carries the clean indicator, so this Conservation
        // Check binds it
        let p_filt_pad = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let p_join_pad = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for &c in p_filt_pad.iter().chain(p_join_pad.iter()) {
            meta.enable_equality(c);
        }

        let q_perm_p1 = meta.complex_selector();
        let q_perm_p2 = meta.complex_selector();
        let perm_part = PermAnyChip::configure(
            meta,
            q_perm_p1,
            q_perm_p2,
            p_filt_pad.clone(),
            p_join_pad.clone(),
        );

        // link p_filt_pad = keep? part : PAD
        let q_part_link = meta.selector();
        meta.create_gate("p_filt_pad = keep? part : PAD", |m| {
            let q = m.query_selector(q_part_link);
            let keep = m.query_advice(p_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let dropv = one.clone() - keep.clone();

            let pad0 = Expression::Constant(F::from(PAD_PKEY));
            let pad1 = Expression::Constant(F::from(PAD_PTYPE));

            let b0 = m.query_advice(part[0], Rotation::cur());
            let b1 = m.query_advice(part[1], Rotation::cur());
            // the indicator pads with 0, so a row dropped by the predicate is
            // never clean and contributes to neither channel of condition (4)
            let b2 = m.query_advice(cflag[5], Rotation::cur());

            let f0 = m.query_advice(p_filt_pad[0], Rotation::cur());
            let f1 = m.query_advice(p_filt_pad[1], Rotation::cur());
            let f2 = m.query_advice(p_filt_pad[2], Rotation::cur());

            vec![
                q.clone() * (f0 - (keep.clone() * b0 + dropv.clone() * pad0)),
                q.clone() * (f1 - (keep.clone() * b1 + dropv * pad1)),
                q * (f2 - keep * b2),
            ]
        });

        // ---------- customer attach: (c_nationkey, c_regionkey) in nation ----------
        let q_tbl_nation = meta.complex_selector();
        let q_tbl_customer = meta.complex_selector();

        let q_lkp_c_nat_region = meta.complex_selector();
        let c_regionkey = meta.advice_column();
        meta.enable_equality(c_regionkey);

        meta.lookup_any("attach c_regionkey from nation via c_nationkey", |m| {
            let q_in = m.query_selector(q_lkp_c_nat_region);
            let q_t = m.query_selector(q_tbl_nation);
            vec![
                (
                    q_in.clone() * m.query_advice(customer[1], Rotation::cur()),
                    q_t.clone() * m.query_advice(nation[0], Rotation::cur()),
                ),
                (
                    q_in * m.query_advice(c_regionkey, Rotation::cur()),
                    q_t * m.query_advice(nation[1], Rotation::cur()),
                ),
            ]
        });

        // c_keep = is_zero(c_regionkey - target_rkey)
        let c_keep = meta.advice_column();
        meta.enable_equality(c_keep);
        let aux_ck = meta.advice_column();
        meta.enable_equality(aux_ck);

        let iz_c_keep = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_tbl_customer), // reuse customer-table selector as enable
            |m| {
                m.query_advice(c_regionkey, Rotation::cur())
                    - m.query_advice(target_rkey, Rotation::cur())
            },
            aux_ck,
        );
        meta.create_gate("c_keep = (c_regionkey == target_rkey)", |m| {
            let q = m.query_selector(q_tbl_customer);
            let keep = m.query_advice(c_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone() * (keep.clone() - iz_c_keep.expr()),
                q * keep.clone() * (one - keep),
            ]
        });

        // ---------- orders attach customer + nation region, then keep ----------
        let q_orders_attach_c = meta.complex_selector();
        let o_c_nationkey = meta.advice_column();
        meta.enable_equality(o_c_nationkey);

        // (o_custkey, o_c_nationkey) in customer
        meta.lookup_any("attach o_c_nationkey from customer via o_custkey", |m| {
            let q_in = m.query_selector(q_orders_attach_c);
            let q_t = m.query_selector(q_tbl_customer);
            vec![
                (
                    q_in.clone() * m.query_advice(orders[1], Rotation::cur()),
                    q_t.clone() * m.query_advice(customer[0], Rotation::cur()),
                ),
                (
                    q_in * m.query_advice(o_c_nationkey, Rotation::cur()),
                    q_t * m.query_advice(customer[1], Rotation::cur()),
                ),
            ]
        });

        let q_orders_attach_r = meta.complex_selector();
        let o_c_regionkey = meta.advice_column();
        meta.enable_equality(o_c_regionkey);

        // (o_c_nationkey, o_c_regionkey) in nation
        meta.lookup_any("attach o_c_regionkey from nation via o_c_nationkey", |m| {
            let q_in = m.query_selector(q_orders_attach_r);
            let q_t = m.query_selector(q_tbl_nation);
            vec![
                (
                    q_in.clone() * m.query_advice(o_c_nationkey, Rotation::cur()),
                    q_t.clone() * m.query_advice(nation[0], Rotation::cur()),
                ),
                (
                    q_in * m.query_advice(o_c_regionkey, Rotation::cur()),
                    q_t * m.query_advice(nation[1], Rotation::cur()),
                ),
            ]
        });

        // o_c_keep = is_zero(o_c_regionkey - target_rkey)
        let o_c_keep = meta.advice_column();
        meta.enable_equality(o_c_keep);
        let aux_ock = meta.advice_column();
        meta.enable_equality(aux_ock);

        let iz_o_c_keep = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_orders_attach_r),
            |m| {
                m.query_advice(o_c_regionkey, Rotation::cur())
                    - m.query_advice(target_rkey, Rotation::cur())
            },
            aux_ock,
        );
        meta.create_gate("o_c_keep = (o_c_regionkey == target_rkey)", |m| {
            let q = m.query_selector(q_orders_attach_r);
            let keep = m.query_advice(o_c_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone() * (keep.clone() - iz_o_c_keep.expr()),
                q * keep.clone() * (one - keep),
            ]
        });

        // year_keep = is_zero((year-1995)*(year-1996))
        let year_keep = meta.advice_column();
        meta.enable_equality(year_keep);
        let aux_yk = meta.advice_column();
        meta.enable_equality(aux_yk);

        let iz_year_keep = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_orders_attach_r),
            |m| {
                let y = m.query_advice(orders[2], Rotation::cur());
                let y1 = y.clone() - Expression::Constant(F::from(1995u64));
                let y2 = y - Expression::Constant(F::from(1996u64));
                y1 * y2
            },
            aux_yk,
        );
        meta.create_gate("year_keep = (year in {1995,1996})", |m| {
            let q = m.query_selector(q_orders_attach_r);
            let keep = m.query_advice(year_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone() * (keep.clone() - iz_year_keep.expr()),
                q * keep.clone() * (one - keep),
            ]
        });

        // o_keep = year_keep * o_c_keep
        let o_keep = meta.advice_column();
        meta.enable_equality(o_keep);
        let q_orders_keep_gate = meta.selector();

        meta.create_gate("o_keep = year_keep * o_c_keep", |m| {
            let q = m.query_selector(q_orders_keep_gate);
            let ok = m.query_advice(o_keep, Rotation::cur());
            let yk = m.query_advice(year_keep, Rotation::cur());
            let ck = m.query_advice(o_c_keep, Rotation::cur());
            vec![q * (ok - yk * ck)]
        });

        // filtered orders tables (okey, oyear, ckey, keep*c) for lookup. o_custkey
        // is carried on the partition side because the orders -> customer edge of
        // condition (3) has to read that key out of the clean section.
        let q_orders_filt_gate = meta.selector();
        let o_filt_pad = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let o_join_pad = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for &c in o_filt_pad.iter().chain(o_join_pad.iter()) {
            meta.enable_equality(c);
        }

        // o_filt_pad = keep? [o_orderkey, o_year, o_custkey, c] : PAD
        meta.create_gate("o_filt_pad = keep? order : PAD", |m| {
            let q = m.query_selector(q_orders_filt_gate);
            let keep = m.query_advice(o_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let dropv = one.clone() - keep.clone();

            let pad0 = Expression::Constant(F::from(PAD_OKEY));
            let pad1 = Expression::Constant(F::from(PAD_OYEAR));
            let pad2 = Expression::Constant(F::from(PAD_OCUST));

            let b0 = m.query_advice(orders[0], Rotation::cur());
            let b1 = m.query_advice(orders[2], Rotation::cur());
            let b2 = m.query_advice(orders[1], Rotation::cur());
            let b3 = m.query_advice(cflag[4], Rotation::cur());

            let f0 = m.query_advice(o_filt_pad[0], Rotation::cur());
            let f1 = m.query_advice(o_filt_pad[1], Rotation::cur());
            let f2 = m.query_advice(o_filt_pad[2], Rotation::cur());
            let f3 = m.query_advice(o_filt_pad[3], Rotation::cur());

            vec![
                q.clone() * (f0 - (keep.clone() * b0 + dropv.clone() * pad0)),
                q.clone() * (f1 - (keep.clone() * b1 + dropv.clone() * pad1)),
                q.clone() * (f2 - (keep.clone() * b2 + dropv * pad2)),
                q * (f3 - keep * b3),
            ]
        });

        // perm o_filt_pad -> o_join_pad
        let q_perm_o1 = meta.complex_selector();
        let q_perm_o2 = meta.complex_selector();
        let perm_orders = PermAnyChip::configure(
            meta,
            q_perm_o1,
            q_perm_o2,
            o_filt_pad.clone(),
            o_join_pad.clone(),
        );

        // ---------- lineitem permutation to l_join_pad (clean rows first) ----------
        // One column wider than in q8_obj.rs: lineitem carries no predicate, so
        // the raw indicator is itself the input side of this Conservation Check
        // and the extra column of l_join_pad is its partition side.
        let l_join_pad = (0..6).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for &c in l_join_pad.iter() {
            meta.enable_equality(c);
        }
        let mut l_base = lineitem.clone();
        l_base.push(cflag[7]);
        let q_perm_l1 = meta.complex_selector();
        let q_perm_l2 = meta.complex_selector();
        let perm_lineitem =
            PermAnyChip::configure(meta, q_perm_l1, q_perm_l2, l_base, l_join_pad.clone());

        // ---------- join lookups for lineitem join prefix ----------
        let q_lkp_partkey = meta.complex_selector();
        let q_tbl_partf = meta.complex_selector();

        // membership: l_partkey ∈ p_join_pad[0]
        meta.lookup_any("l.partkey in filtered part table", |m| {
            let q_in = m.query_selector(q_lkp_partkey);
            let q_t = m.query_selector(q_tbl_partf);
            vec![(
                q_in * m.query_advice(l_join_pad[1], Rotation::cur()),
                q_t * m.query_advice(p_join_pad[0], Rotation::cur()),
            )]
        });

        let q_lkp_orders = meta.complex_selector();
        let q_tbl_ordersf = meta.complex_selector();
        let l_year = meta.advice_column();
        meta.enable_equality(l_year);

        // (l_orderkey, l_year) ∈ o_join_pad
        meta.lookup_any("attach year from filtered orders", |m| {
            let q_in = m.query_selector(q_lkp_orders);
            let q_t = m.query_selector(q_tbl_ordersf);
            vec![
                (
                    q_in.clone() * m.query_advice(l_join_pad[0], Rotation::cur()),
                    q_t.clone() * m.query_advice(o_join_pad[0], Rotation::cur()),
                ),
                (
                    q_in * m.query_advice(l_year, Rotation::cur()),
                    q_t * m.query_advice(o_join_pad[1], Rotation::cur()),
                ),
            ]
        });

        let q_lkp_supplier = meta.complex_selector();
        let q_tbl_supplier = meta.complex_selector();
        let l_s_nationkey = meta.advice_column();
        meta.enable_equality(l_s_nationkey);

        // (l_suppkey, l_s_nationkey) ∈ supplier
        meta.lookup_any("attach supplier nationkey", |m| {
            let q_in = m.query_selector(q_lkp_supplier);
            let q_t = m.query_selector(q_tbl_supplier);
            vec![
                (
                    q_in.clone() * m.query_advice(l_join_pad[2], Rotation::cur()),
                    q_t.clone() * m.query_advice(supplier[0], Rotation::cur()),
                ),
                (
                    q_in * m.query_advice(l_s_nationkey, Rotation::cur()),
                    q_t * m.query_advice(supplier[1], Rotation::cur()),
                ),
            ]
        });

        let q_lkp_nation2 = meta.complex_selector();
        let l_nation_hash = meta.advice_column();
        meta.enable_equality(l_nation_hash);

        // (l_s_nationkey, l_nation_hash) ∈ nation (n2: supplier nation -> nation name hash)
        meta.lookup_any("attach nation namehash for supplier nation", |m| {
            let q_in = m.query_selector(q_lkp_nation2);
            let q_t = m.query_selector(q_tbl_nation);
            vec![
                (
                    q_in.clone() * m.query_advice(l_s_nationkey, Rotation::cur()),
                    q_t.clone() * m.query_advice(nation[0], Rotation::cur()),
                ),
                (
                    q_in * m.query_advice(l_nation_hash, Rotation::cur()),
                    q_t * m.query_advice(nation[2], Rotation::cur()),
                ),
            ]
        });

        // ---------- volume gate ----------
        let q_volume = meta.selector();
        let volume = meta.advice_column();
        meta.enable_equality(volume);

        meta.create_gate("volume = ext*(SCALE - disc)", |m| {
            let q = m.query_selector(q_volume);
            let ext = m.query_advice(l_join_pad[3], Rotation::cur());
            let disc = m.query_advice(l_join_pad[4], Rotation::cur());
            let v = m.query_advice(volume, Rotation::cur());
            let scale = Expression::Constant(F::from(SCALE));
            vec![q * (v - ext * (scale - disc))]
        });

        // all_nations row: [year, volume, nation_hash]
        let all_nations = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let all_nations_sorted = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        for &c in all_nations.iter().chain(all_nations_sorted.iter()) {
            meta.enable_equality(c);
        }

        // tie all_nations on join rows
        meta.create_gate("all_nations = (year, volume, nation)", |m| {
            let q = m.query_selector(q_volume); // same enable as join rows
            vec![
                q.clone()
                    * (m.query_advice(all_nations[0], Rotation::cur())
                        - m.query_advice(l_year, Rotation::cur())),
                q.clone()
                    * (m.query_advice(all_nations[1], Rotation::cur())
                        - m.query_advice(volume, Rotation::cur())),
                q * (m.query_advice(all_nations[2], Rotation::cur())
                    - m.query_advice(l_nation_hash, Rotation::cur())),
            ]
        });

        // perm all_nations -> all_nations_sorted
        let q_perm_an1 = meta.complex_selector();
        let q_perm_an2 = meta.complex_selector();
        let perm_all_nations = PermAnyChip::configure(
            meta,
            q_perm_an1,
            q_perm_an2,
            all_nations.clone(),
            all_nations_sorted.clone(),
        );

        // sort by year ASC: cur <= next
        let q_sort_year = meta.selector();

        let aux_year_eq = meta.advice_column();
        meta.enable_equality(aux_year_eq);

        let iz_year_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort_year),
            |m| {
                m.query_advice(all_nations_sorted[0], Rotation::cur())
                    - m.query_advice(all_nations_sorted[0], Rotation::next())
            },
            aux_year_eq,
        );

        let lt_year_cur_next = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort_year),
            |m| m.query_advice(all_nations_sorted[0], Rotation::cur()),
            |m| m.query_advice(all_nations_sorted[0], Rotation::next()),
        );

        meta.create_gate("ORDER BY year ASC on all_nations_sorted", |m| {
            let q = m.query_selector(q_sort_year);
            let lt = lt_year_cur_next.is_lt(m, None);
            let eq = iz_year_eq.expr();
            vec![q * (lt + eq - Expression::Constant(F::ONE))]
        });

        // ---------- cond flag (nation == ':1') ----------
        let q_is_cond = meta.selector();
        let cond_flag = meta.advice_column();
        meta.enable_equality(cond_flag);

        let aux_is_cond = meta.advice_column();
        meta.enable_equality(aux_is_cond);

        let iz_is_cond = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_is_cond),
            |m| {
                m.query_advice(all_nations_sorted[2], Rotation::cur())
                    - m.query_advice(cond_nation, Rotation::cur())
            },
            aux_is_cond,
        );

        meta.create_gate("cond_flag = (nation == :1)", |m| {
            let q = m.query_selector(q_is_cond);
            let f = m.query_advice(cond_flag, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone() * (f.clone() - iz_is_cond.expr()),
                q * f.clone() * (one - f),
            ]
        });

        // ---------- group by year: running sums ----------
        let q_first = meta.selector();
        let q_accu = meta.selector();
        let q_line = meta.selector();

        let aux_sp = meta.advice_column();
        let aux_sn = meta.advice_column();
        meta.enable_equality(aux_sp);
        meta.enable_equality(aux_sn);

        let iz_same_prev = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_accu),
            |m| {
                m.query_advice(all_nations_sorted[0], Rotation::cur())
                    - m.query_advice(all_nations_sorted[0], Rotation::prev())
            },
            aux_sp,
        );
        let iz_same_next = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_line),
            |m| {
                m.query_advice(all_nations_sorted[0], Rotation::next())
                    - m.query_advice(all_nations_sorted[0], Rotation::cur())
            },
            aux_sn,
        );

        let run_num = meta.advice_column();
        let run_den = meta.advice_column();
        meta.enable_equality(run_num);
        meta.enable_equality(run_den);

        // first row
        meta.create_gate("run sums first", |m| {
            let q = m.query_selector(q_first);
            let v = m.query_advice(all_nations_sorted[1], Rotation::cur());
            let f = m.query_advice(cond_flag, Rotation::cur());
            let rn = m.query_advice(run_num, Rotation::cur());
            let rd = m.query_advice(run_den, Rotation::cur());
            vec![q.clone() * (rd - v.clone()), q * (rn - f * v)]
        });

        // accumulation
        meta.create_gate("run sums accu", |m| {
            let q = m.query_selector(q_accu);
            let same = iz_same_prev.expr();
            let v = m.query_advice(all_nations_sorted[1], Rotation::cur());
            let f = m.query_advice(cond_flag, Rotation::cur());

            let rn_cur = m.query_advice(run_num, Rotation::cur());
            let rn_prev = m.query_advice(run_num, Rotation::prev());
            let rd_cur = m.query_advice(run_den, Rotation::cur());
            let rd_prev = m.query_advice(run_den, Rotation::prev());

            vec![
                q.clone() * (rd_cur - (same.clone() * rd_prev + v.clone())),
                q * (rn_cur - (same * rn_prev + f * v)),
            ]
        });

        // emit sparse results: on last row of year group, output (year, num, den, share), else PAD
        let res_pad = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for &c in res_pad.iter() {
            meta.enable_equality(c);
        }

        let q_emit = meta.selector();
        let q_emit_last = meta.selector();
        let q_share = meta.selector(); // later applied on res_sorted rows

        // gate: res_pad = last? values : PAD
        meta.create_gate("emit res_pad on group last (normal rows)", |m| {
            let q = m.query_selector(q_emit);
            let one = Expression::Constant(F::ONE);

            let is_last = one.clone() - iz_same_next.expr();
            let not_last = one.clone() - is_last.clone();

            let y = m.query_advice(all_nations_sorted[0], Rotation::cur());
            let rn = m.query_advice(run_num, Rotation::cur());
            let rd = m.query_advice(run_den, Rotation::cur());

            let out_y = m.query_advice(res_pad[0], Rotation::cur());
            let out_n = m.query_advice(res_pad[1], Rotation::cur());
            let out_d = m.query_advice(res_pad[2], Rotation::cur());
            let out_s = m.query_advice(res_pad[3], Rotation::cur());

            let pad_y = Expression::Constant(F::from(PAD_RES_YEAR));
            let pad_n = Expression::Constant(F::from(PAD_RES_NUM));
            let pad_d = Expression::Constant(F::from(PAD_RES_DEN));
            let pad_s = Expression::Constant(F::ZERO);

            vec![
                q.clone() * (out_y - (is_last.clone() * y + not_last.clone() * pad_y)),
                q.clone() * (out_n - (is_last.clone() * rn + not_last.clone() * pad_n)),
                q.clone() * (out_d - (is_last.clone() * rd + not_last.clone() * pad_d)),
                q * (out_s.clone() - (is_last * out_s + not_last * pad_s)),
            ]
        });

        meta.create_gate("emit res_pad on last row", |m| {
            let q = m.query_selector(q_emit_last);

            let y = m.query_advice(all_nations_sorted[0], Rotation::cur());
            let rn = m.query_advice(run_num, Rotation::cur());
            let rd = m.query_advice(run_den, Rotation::cur());

            let out_y = m.query_advice(res_pad[0], Rotation::cur());
            let out_n = m.query_advice(res_pad[1], Rotation::cur());
            let out_d = m.query_advice(res_pad[2], Rotation::cur());
            // out_s unconstrained here (fine), share is constrained later on res_sorted.

            vec![
                q.clone() * (out_y - y),
                q.clone() * (out_n - rn),
                q * (out_d - rd),
            ]
        });

        // res_sorted (compact) by permutation
        let res_sorted = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for &c in res_sorted.iter() {
            meta.enable_equality(c);
        }

        let q_perm_r1 = meta.complex_selector();
        let q_perm_r2 = meta.complex_selector();
        let perm_res = PermAnyChip::configure(
            meta,
            q_perm_r1,
            q_perm_r2,
            res_pad.clone(),
            res_sorted.clone(),
        );

        // ORDER BY year ASC on res_sorted: cur <= next
        let q_sort_res = meta.selector();
        let aux_res_eq = meta.advice_column();
        meta.enable_equality(aux_res_eq);

        let iz_res_year_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| {
                m.query_advice(res_sorted[0], Rotation::cur())
                    - m.query_advice(res_sorted[0], Rotation::next())
            },
            aux_res_eq,
        );

        let lt_res_year_cur_next = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| m.query_advice(res_sorted[0], Rotation::cur()),
            |m| m.query_advice(res_sorted[0], Rotation::next()),
        );

        meta.create_gate("ORDER BY year ASC on res_sorted", |m| {
            let q = m.query_selector(q_sort_res);
            let lt = lt_res_year_cur_next.is_lt(m, None);
            let eq = iz_res_year_eq.expr();
            vec![q * (lt + eq - Expression::Constant(F::ONE))]
        });

        // share constraint on res_sorted rows: share * den = num
        meta.create_gate("share*den = num (field)", |m| {
            let q = m.query_selector(q_share);
            let y = m.query_advice(res_sorted[0], Rotation::cur());
            let num = m.query_advice(res_sorted[1], Rotation::cur());
            let den = m.query_advice(res_sorted[2], Rotation::cur());
            let share = m.query_advice(res_sorted[3], Rotation::cur());

            // allow pads: if y==PAD_YEAR then num=den=share=0 by witness, constraint holds.
            let _ = y;
            vec![q * (share * den - num)]
        });

        // ============ introduced partition: Conservation Checks ============
        // q8_obj.rs verifies the join out of a prover-chosen prefix of
        // l_join_pad and partitions nothing, so there is no old condition (4)
        // to remove here. The partition itself is what has to be introduced:
        // per relation the base rows carrying keep * c are shuffled onto a
        // column group laid out as [clean rows | residual rows | pad rows], and
        // two selectors pin the indicator over the first two sections.

        // region carries the in-relation predicate r_name == MIDDLE EAST, which
        // the input channel of condition (4) has to apply, so materialize it.
        let q_region_pred = meta.selector();
        let r_keep = meta.advice_column();
        meta.enable_equality(r_keep);
        let aux_rk = meta.advice_column();
        meta.enable_equality(aux_rk);

        let iz_r_keep = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_region_pred),
            |m| {
                m.query_advice(region[1], Rotation::cur())
                    - m.query_advice(const_rname, Rotation::cur())
            },
            aux_rk,
        );
        meta.create_gate("r_keep = (r_name == const_rname)", |m| {
            let q = m.query_selector(q_region_pred);
            let keep = m.query_advice(r_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone() * (keep.clone() - iz_r_keep.expr()),
                q * keep.clone() * (one - keep),
            ]
        });

        let r_filt_pad = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let r_part_pad = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for &c in r_filt_pad.iter().chain(r_part_pad.iter()) {
            meta.enable_equality(c);
        }
        meta.create_gate("r_filt_pad = keep? region : PAD", |m| {
            let q = m.query_selector(q_region_pred);
            let keep = m.query_advice(r_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let dropv = one - keep.clone();

            let pad0 = Expression::Constant(F::from(PAD_RKEY));
            let pad1 = Expression::Constant(F::from(PAD_RNAME));

            let b0 = m.query_advice(region[0], Rotation::cur());
            let b1 = m.query_advice(region[1], Rotation::cur());
            let b2 = m.query_advice(cflag[0], Rotation::cur());

            let f0 = m.query_advice(r_filt_pad[0], Rotation::cur());
            let f1 = m.query_advice(r_filt_pad[1], Rotation::cur());
            let f2 = m.query_advice(r_filt_pad[2], Rotation::cur());

            vec![
                q.clone() * (f0 - (keep.clone() * b0 + dropv.clone() * pad0)),
                q.clone() * (f1 - (keep.clone() * b1 + dropv * pad1)),
                q * (f2 - keep * b2),
            ]
        });
        let q_perm_rg1 = meta.complex_selector();
        let q_perm_rg2 = meta.complex_selector();
        let perm_region = PermAnyChip::configure(
            meta,
            q_perm_rg1,
            q_perm_rg2,
            r_filt_pad.clone(),
            r_part_pad.clone(),
        );

        // nation, customer, supplier and lineitem carry no predicate, so their
        // base rows plus the raw indicator are the input side of the shuffle and
        // no link gate is needed: keep is 1 everywhere.
        let n1_part_pad = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let n2_part_pad = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let c_part_pad = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let s_part_pad = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();

        let mut n1_base = nation.clone();
        n1_base.push(cflag[1]);
        let mut n2_base = nation.clone();
        n2_base.push(cflag[2]);
        let mut c_base = customer.clone();
        c_base.push(cflag[3]);
        let mut s_base = supplier.clone();
        s_base.push(cflag[6]);

        let q_perm_n11 = meta.complex_selector();
        let q_perm_n12 = meta.complex_selector();
        let perm_nation1 =
            PermAnyChip::configure(meta, q_perm_n11, q_perm_n12, n1_base, n1_part_pad.clone());
        let q_perm_n21 = meta.complex_selector();
        let q_perm_n22 = meta.complex_selector();
        let perm_nation2 =
            PermAnyChip::configure(meta, q_perm_n21, q_perm_n22, n2_base, n2_part_pad.clone());
        let q_perm_c1 = meta.complex_selector();
        let q_perm_c2 = meta.complex_selector();
        let perm_customer =
            PermAnyChip::configure(meta, q_perm_c1, q_perm_c2, c_base, c_part_pad.clone());
        let q_perm_s1 = meta.complex_selector();
        let q_perm_s2 = meta.complex_selector();
        let perm_supplier =
            PermAnyChip::configure(meta, q_perm_s1, q_perm_s2, s_base, s_part_pad.clone());

        // -------- partition side of the indicator: 1 on R^c rows, 0 on R^r --------
        // Without these the prover could mark a residual row clean and inflate
        // the clean channel of the check below. The pad rows need no gate: the
        // shuffle already pins them, since the link gates pad the indicator
        // with 0.
        let q_cln_flag = (0..8).map(|_| meta.selector()).collect::<Vec<_>>();
        let q_res_flag = (0..8).map(|_| meta.selector()).collect::<Vec<_>>();

        for (idx, part_group) in [
            r_part_pad.clone(),
            n1_part_pad.clone(),
            n2_part_pad.clone(),
            c_part_pad.clone(),
            o_join_pad.clone(),
            p_join_pad.clone(),
            s_part_pad.clone(),
            l_join_pad.clone(),
        ]
        .iter()
        .enumerate()
        {
            let flag_col = *part_group.last().unwrap();
            let q_c = q_cln_flag[idx];
            let q_r = q_res_flag[idx];
            meta.create_gate("clean indicator on the partition side", move |m| {
                let qc = m.query_selector(q_c);
                let qr = m.query_selector(q_r);
                let f = m.query_advice(flag_col, Rotation::cur());
                vec![qc * (f.clone() - Expression::Constant(F::ONE)), qr * f]
            });
        }

        // ============ Pairwise Consistency (condition (3)) ============
        // pi_K_ij(R_i^c) == pi_K_ij(R_j^c) on every edge of the join tree, as two
        // mutual Membership Checks. Both sides read the key out of the relation's
        // partition group restricted to its clean section, which the Conservation
        // Check above ties to the base relation, so this really is a statement
        // about R^c and not about R.
        //
        // The selectors have to be complex: a simple selector may not appear in a
        // lookup expression, so the q_cln_flag selectors of the block above cannot
        // be reused. One per relation is enough, since every lookup out of a
        // relation runs over the same clean row range, and the same selector then
        // serves as the table selector of the opposite direction.
        //
        // There is no intermediate key table any more. Routing each direction
        // through a pw_tbl advice column holding the deduplicated key set left
        // that column unbound to the relation it claimed to enumerate: a prover
        // could store pi_K(R_i^c) in the table R_i^c looks into and pi_K(R_j^c)
        // in the other, and both directions would pass for an arbitrary
        // partition. Looking the two clean key columns up in each other leaves no
        // free advice, and the two containments are the set equality.
        let q_pw_in = (0..8).map(|_| meta.complex_selector()).collect::<Vec<_>>();

        // Every key is shifted by SHIFT_ID on both sides of every lookup, which
        // preserves the containment. A lookup_any expression is evaluated on every
        // row of the circuit and both sides are 0 wherever their selector is off,
        // so 0 is always a member of the table; without the shift a genuine key 0
        // (n_nationkey and r_regionkey start at 0 in TPC-H) would be accepted
        // whether or not it is in R_j^c.
        let mut pw_edge = |name: &'static str,
                           q_in: Selector,
                           key_col: Column<Advice>,
                           q_t: Selector,
                           tbl_col: Column<Advice>| {
            meta.lookup_any(name, move |m| {
                let q_in = m.query_selector(q_in);
                let q_t = m.query_selector(q_t);
                let shift = Expression::Constant(F::from(SHIFT_ID));
                vec![(
                    q_in * (m.query_advice(key_col, Rotation::cur()) + shift.clone()),
                    q_t * (m.query_advice(tbl_col, Rotation::cur()) + shift),
                )]
            });
        };

        // edge lineitem -- part on l_partkey = p_partkey
        pw_edge(
            "pw: lineitem^c partkey in part^c",
            q_pw_in[7],
            l_join_pad[1],
            q_pw_in[5],
            p_join_pad[0],
        );
        pw_edge(
            "pw: part^c partkey in lineitem^c",
            q_pw_in[5],
            p_join_pad[0],
            q_pw_in[7],
            l_join_pad[1],
        );

        // edge lineitem -- orders on l_orderkey = o_orderkey
        pw_edge(
            "pw: lineitem^c orderkey in orders^c",
            q_pw_in[7],
            l_join_pad[0],
            q_pw_in[4],
            o_join_pad[0],
        );
        pw_edge(
            "pw: orders^c orderkey in lineitem^c",
            q_pw_in[4],
            o_join_pad[0],
            q_pw_in[7],
            l_join_pad[0],
        );

        // edge lineitem -- supplier on l_suppkey = s_suppkey
        pw_edge(
            "pw: lineitem^c suppkey in supplier^c",
            q_pw_in[7],
            l_join_pad[2],
            q_pw_in[6],
            s_part_pad[0],
        );
        pw_edge(
            "pw: supplier^c suppkey in lineitem^c",
            q_pw_in[6],
            s_part_pad[0],
            q_pw_in[7],
            l_join_pad[2],
        );

        // edge supplier -- nation as n2 on s_nationkey = n_nationkey
        pw_edge(
            "pw: supplier^c nationkey in nation2^c",
            q_pw_in[6],
            s_part_pad[1],
            q_pw_in[2],
            n2_part_pad[0],
        );
        pw_edge(
            "pw: nation2^c nationkey in supplier^c",
            q_pw_in[2],
            n2_part_pad[0],
            q_pw_in[6],
            s_part_pad[1],
        );

        // edge orders -- customer on o_custkey = c_custkey
        pw_edge(
            "pw: orders^c custkey in customer^c",
            q_pw_in[4],
            o_join_pad[2],
            q_pw_in[3],
            c_part_pad[0],
        );
        pw_edge(
            "pw: customer^c custkey in orders^c",
            q_pw_in[3],
            c_part_pad[0],
            q_pw_in[4],
            o_join_pad[2],
        );

        // edge customer -- nation as n1 on c_nationkey = n_nationkey
        pw_edge(
            "pw: customer^c nationkey in nation1^c",
            q_pw_in[3],
            c_part_pad[1],
            q_pw_in[1],
            n1_part_pad[0],
        );
        pw_edge(
            "pw: nation1^c nationkey in customer^c",
            q_pw_in[1],
            n1_part_pad[0],
            q_pw_in[3],
            c_part_pad[1],
        );

        // edge nation as n1 -- region on n_regionkey = r_regionkey
        pw_edge(
            "pw: nation1^c regionkey in region^c",
            q_pw_in[1],
            n1_part_pad[1],
            q_pw_in[0],
            r_part_pad[0],
        );
        pw_edge(
            "pw: region^c regionkey in nation1^c",
            q_pw_in[0],
            r_part_pad[0],
            q_pw_in[1],
            n1_part_pad[1],
        );

        // ============ Cardinality Preservation Check (condition (4)) ============
        // Two of the seven edges are keyed on a 0-based id, so both of their
        // sides get a shifted copy of the key: key 0 is reserved for the dummy
        // row of the gadget's key-indexed tables.
        let q_nation_cp = meta.selector();
        let q_cust_cp = meta.selector();
        let q_supp_cp = meta.selector();

        let nk_shift = meta.advice_column();
        let n_rk_shift = meta.advice_column();
        let rk_shift = meta.advice_column();
        let c_nk_shift = meta.advice_column();
        let s_nk_shift = meta.advice_column();
        let cp_one = meta.advice_column();
        for &c in [nk_shift, n_rk_shift, rk_shift, c_nk_shift, s_nk_shift, cp_one].iter() {
            meta.enable_equality(c);
        }

        meta.create_gate("cp: shifted nation keys and the nation leaf predicate", |m| {
            let q = m.query_selector(q_nation_cp);
            let s = Expression::Constant(F::from(SHIFT_ID));
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone()
                    * (m.query_advice(nk_shift, Rotation::cur())
                        - m.query_advice(nation[0], Rotation::cur())
                        - s.clone()),
                q.clone()
                    * (m.query_advice(n_rk_shift, Rotation::cur())
                        - m.query_advice(nation[1], Rotation::cur())
                        - s),
                q * (m.query_advice(cp_one, Rotation::cur()) - one),
            ]
        });
        meta.create_gate("cp: shifted region key", |m| {
            let q = m.query_selector(q_region_pred);
            let s = Expression::Constant(F::from(SHIFT_ID));
            vec![
                q * (m.query_advice(rk_shift, Rotation::cur())
                    - m.query_advice(region[0], Rotation::cur())
                    - s),
            ]
        });
        meta.create_gate("cp: shifted customer nationkey", |m| {
            let q = m.query_selector(q_cust_cp);
            let s = Expression::Constant(F::from(SHIFT_ID));
            vec![
                q * (m.query_advice(c_nk_shift, Rotation::cur())
                    - m.query_advice(customer[1], Rotation::cur())
                    - s),
            ]
        });
        meta.create_gate("cp: shifted supplier nationkey", |m| {
            let q = m.query_selector(q_supp_cp);
            let s = Expression::Constant(F::from(SHIFT_ID));
            vec![
                q * (m.query_advice(s_nk_shift, Rotation::cur())
                    - m.query_advice(supplier[1], Rotation::cur())
                    - s),
            ]
        });

        // One fixed column serves every Lt chip of the check, so the whole
        // check costs a single u8 range table instead of one per comparator.
        let cp_u8 = meta.fixed_column();

        // ---- the branch region <- nation as n1 <- customer <- orders ----
        // region is a leaf: the input channel is its predicate bit and the clean
        // channel is the bound keep * c column of its Conservation Check.
        let cp_agg_region = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            rk_shift,
            r_keep,
            r_filt_pad[2],
            MAX_SENTINEL,
        );
        let cp_join_region = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, n_rk_shift);
        wire_cp_edge(meta, &cp_join_region, &cp_agg_region, n_rk_shift);

        // nation as n1 has no predicate, so its input-channel multiplicity is
        // exactly the sum fetched on its region edge and costs no gate at all;
        // only the clean channel needs one product.
        let cp_mu_n1 = meta.advice_column();
        let q_cp_n1 = meta.selector();
        {
            let s_cln = cp_join_region.s_cln;
            let c = cflag[1];
            meta.create_gate("cp: clean multiplicity of nation as n1", move |m| {
                let q = m.query_selector(q_cp_n1);
                vec![
                    q * (m.query_advice(cp_mu_n1, Rotation::cur())
                        - m.query_advice(c, Rotation::cur())
                            * m.query_advice(s_cln, Rotation::cur())),
                ]
            });
        }
        let cp_agg_n1 = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            nk_shift,
            cp_join_region.s_all,
            cp_mu_n1,
            MAX_SENTINEL,
        );
        let cp_join_n1 = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, c_nk_shift);
        wire_cp_edge(meta, &cp_join_n1, &cp_agg_n1, c_nk_shift);

        let cp_mu_cust = meta.advice_column();
        let q_cp_cust = meta.selector();
        {
            let s_cln = cp_join_n1.s_cln;
            let c = cflag[3];
            meta.create_gate("cp: clean multiplicity of customer", move |m| {
                let q = m.query_selector(q_cp_cust);
                vec![
                    q * (m.query_advice(cp_mu_cust, Rotation::cur())
                        - m.query_advice(c, Rotation::cur())
                            * m.query_advice(s_cln, Rotation::cur())),
                ]
            });
        }
        let cp_agg_cust = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            customer[0],
            cp_join_n1.s_all,
            cp_mu_cust,
            MAX_SENTINEL,
        );
        let cp_join_cust = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, orders[1]);
        wire_cp_edge(meta, &cp_join_cust, &cp_agg_cust, orders[1]);

        // orders is the one internal node that also carries a predicate, so
        // both of its channels need a product.
        let cp_mu_ord = (0..2).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let q_cp_ord = meta.selector();
        {
            let (mu_all, mu_cln) = (cp_mu_ord[0], cp_mu_ord[1]);
            let (s_all, s_cln) = (cp_join_cust.s_all, cp_join_cust.s_cln);
            let pred = o_keep;
            let cln = o_filt_pad[3];
            meta.create_gate("cp: multiplicities of orders", move |m| {
                let q = m.query_selector(q_cp_ord);
                vec![
                    q.clone()
                        * (m.query_advice(mu_all, Rotation::cur())
                            - m.query_advice(pred, Rotation::cur())
                                * m.query_advice(s_all, Rotation::cur())),
                    q * (m.query_advice(mu_cln, Rotation::cur())
                        - m.query_advice(cln, Rotation::cur())
                            * m.query_advice(s_cln, Rotation::cur())),
                ]
            });
        }
        let cp_agg_ord = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            orders[0],
            cp_mu_ord[0],
            cp_mu_ord[1],
            MAX_SENTINEL,
        );
        let cp_join_ord = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, lineitem[0]);
        wire_cp_edge(meta, &cp_join_ord, &cp_agg_ord, lineitem[0]);

        // ---- the branch nation as n2 <- supplier ----
        // nation as n2 is a leaf without a predicate, so its input channel is
        // the constant 1 column.
        let cp_agg_n2 = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            nk_shift,
            cp_one,
            cflag[2],
            MAX_SENTINEL,
        );
        let cp_join_n2 = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, s_nk_shift);
        wire_cp_edge(meta, &cp_join_n2, &cp_agg_n2, s_nk_shift);

        let cp_mu_supp = meta.advice_column();
        let q_cp_supp = meta.selector();
        {
            let s_cln = cp_join_n2.s_cln;
            let c = cflag[6];
            meta.create_gate("cp: clean multiplicity of supplier", move |m| {
                let q = m.query_selector(q_cp_supp);
                vec![
                    q * (m.query_advice(cp_mu_supp, Rotation::cur())
                        - m.query_advice(c, Rotation::cur())
                            * m.query_advice(s_cln, Rotation::cur())),
                ]
            });
        }
        let cp_agg_supp = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            supplier[0],
            cp_join_n2.s_all,
            cp_mu_supp,
            MAX_SENTINEL,
        );
        let cp_join_supp = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, lineitem[2]);
        wire_cp_edge(meta, &cp_join_supp, &cp_agg_supp, lineitem[2]);

        // ---- the part leaf ----
        let cp_agg_part = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            part[0],
            p_keep,
            p_filt_pad[2],
            MAX_SENTINEL,
        );
        let cp_join_part = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, lineitem[1]);
        wire_cp_edge(meta, &cp_join_part, &cp_agg_part, lineitem[1]);

        // ---- the root: lineitem, three children and no predicate ----
        let cp_root = configure_cp_root::<F>(meta);
        let cp_aux = (0..2).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let q_cp_root = meta.selector();
        {
            let (aux_all, aux_cln) = (cp_aux[0], cp_aux[1]);
            let cln_l = cflag[7];
            let (p_all, p_cln) = (cp_join_part.s_all, cp_join_part.s_cln);
            let (o_all, o_cln) = (cp_join_ord.s_all, cp_join_ord.s_cln);
            let (s_all, s_cln) = (cp_join_supp.s_all, cp_join_supp.s_cln);
            let mu_all = cp_root.mu_all;
            let mu_cln = cp_root.mu_cln;
            meta.create_gate("cp: root multiplicities over lineitem", move |m| {
                let q = m.query_selector(q_cp_root);
                // three children, so both products are folded through one
                // intermediate column each to stay at degree 4
                let f_aux_all = m.query_advice(aux_all, Rotation::cur())
                    - m.query_advice(p_all, Rotation::cur())
                        * m.query_advice(o_all, Rotation::cur());
                let f_all = m.query_advice(mu_all, Rotation::cur())
                    - m.query_advice(aux_all, Rotation::cur())
                        * m.query_advice(s_all, Rotation::cur());
                let f_aux_cln = m.query_advice(aux_cln, Rotation::cur())
                    - m.query_advice(cln_l, Rotation::cur())
                        * m.query_advice(p_cln, Rotation::cur());
                let f_cln = m.query_advice(mu_cln, Rotation::cur())
                    - m.query_advice(aux_cln, Rotation::cur())
                        * m.query_advice(o_cln, Rotation::cur())
                        * m.query_advice(s_cln, Rotation::cur());
                vec![
                    q.clone() * f_aux_all,
                    q.clone() * f_all,
                    q.clone() * f_aux_cln,
                    q * f_cln,
                ]
            });
        }

        TestCircuitConfig {
            instance,
            instance_test,

            region,
            nation,
            customer,
            orders,
            part,
            supplier,
            lineitem,

            cond_nation,
            const_rname,
            const_ptype,
            target_rkey,

            q_tbl_region,
            q_lkp_target_region,

            q_part_pred,
            p_keep,
            iz_part_type,
            p_filt_pad,
            p_join_pad,
            perm_part,

            q_tbl_nation,
            q_tbl_customer,
            q_lkp_c_nat_region,
            c_regionkey,
            c_keep,
            iz_c_keep,

            q_orders_attach_c,
            o_c_nationkey,
            q_orders_attach_r,
            o_c_regionkey,
            o_c_keep,
            iz_o_c_keep,
            year_keep,
            iz_year_keep,
            o_keep,
            q_orders_keep_gate,
            q_orders_filt_gate,
            o_filt_pad,
            o_join_pad,
            perm_orders,

            l_join_pad,
            perm_lineitem,

            q_lkp_partkey,
            q_tbl_partf,
            q_lkp_orders,
            q_tbl_ordersf,
            q_lkp_supplier,
            q_tbl_supplier,
            q_lkp_nation2,
            l_year,
            l_s_nationkey,
            l_nation_hash,

            q_volume,
            volume,

            all_nations,
            all_nations_sorted,
            perm_all_nations,

            q_sort_year,
            lt_year_cur_next,
            iz_year_eq,

            q_is_cond,
            cond_flag,
            iz_is_cond,

            q_first,
            q_accu,
            q_line,
            iz_same_prev,
            iz_same_next,
            run_num,
            run_den,

            res_pad,
            res_sorted,
            perm_res,

            q_sort_res,
            lt_res_year_cur_next,
            iz_res_year_eq,

            q_share,
            q_part_link,
            q_emit,
            q_emit_last,

            cflag,

            q_region_pred,
            r_keep,
            iz_r_keep,
            r_filt_pad,
            r_part_pad,
            perm_region,

            n1_part_pad,
            perm_nation1,
            n2_part_pad,
            perm_nation2,
            c_part_pad,
            perm_customer,
            s_part_pad,
            perm_supplier,

            q_cln_flag,
            q_res_flag,

            q_pw_in,

            q_nation_cp,
            q_cust_cp,
            q_supp_cp,
            nk_shift,
            n_rk_shift,
            rk_shift,
            c_nk_shift,
            s_nk_shift,

            cp_agg_region,
            cp_join_region,
            cp_agg_n1,
            cp_join_n1,
            cp_agg_cust,
            cp_join_cust,
            cp_agg_ord,
            cp_join_ord,
            cp_agg_n2,
            cp_join_n2,
            cp_agg_supp,
            cp_join_supp,
            cp_agg_part,
            cp_join_part,
            cp_root,

            cp_one,
            cp_mu_n1,
            cp_mu_cust,
            cp_mu_supp,
            cp_mu_ord,
            cp_aux,
            q_cp_n1,
            q_cp_cust,
            q_cp_supp,
            q_cp_ord,
            q_cp_root,
        }
    }

    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        region_t: Vec<Vec<u64>>,
        nation_t: Vec<Vec<u64>>,
        customer_t: Vec<Vec<u64>>,
        orders_t: Vec<Vec<u64>>,
        part_t: Vec<Vec<u64>>,
        supplier_t: Vec<Vec<u64>>,
        lineitem_t: Vec<Vec<u64>>,
        cond_nation_hash: u64,
        const_region_name_hash: u64,
        const_part_type_hash: u64,
    ) -> Result<AssignedCell<F, F>, Error> {
        // load lt tables
        let lt_year_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_year_cur_next.clone());
        lt_year_chip.load(layouter)?;
        let lt_res_year_chip =
            LtChip::<F, NUM_BYTES>::construct(self.config.lt_res_year_cur_next.clone());
        lt_res_year_chip.load(layouter)?;

        // Every Lt chip of the Cardinality Preservation Check shares one u8
        // fixed column, so a single load covers the whole check.
        LtChip::<F, NUM_BYTES>::construct(self.config.cp_agg_region.lt_key_cur_next).load(layouter)?;

        // ---------- preprocessing (pure witness computations) ----------

        // find target regionkey for MIDDLE EAST
        let mut target_rkey_u64 = PAD_YEAR;
        for r in region_t.iter() {
            if r.len() >= 2 && r[1] == const_region_name_hash {
                target_rkey_u64 = r[0];
                break;
            }
        }

        // nation maps
        let nat_region: HashMap<u64, u64> = nation_t.iter().map(|r| (r[0], r[1])).collect();
        let nat_name: HashMap<u64, u64> = nation_t.iter().map(|r| (r[0], r[2])).collect();

        // customer map
        let cust_nat: HashMap<u64, u64> = customer_t.iter().map(|r| (r[0], r[1])).collect();

        // supplier map
        let supp_nat: HashMap<u64, u64> = supplier_t.iter().map(|r| (r[0], r[1])).collect();

        // part filter
        let mut p_keep_u64 = vec![0u64; part_t.len()];
        for i in 0..part_t.len() {
            p_keep_u64[i] = if part_t[i][1] == const_part_type_hash {
                1
            } else {
                0
            };
        }
        let p_join: Vec<Vec<u64>> = part_t
            .iter()
            .cloned()
            .zip(p_keep_u64.iter().cloned())
            .filter(|(_, k)| *k == 1)
            .map(|(r, _)| r)
            .collect();
        let partkeys_keep: HashSet<u64> = p_join.iter().map(|r| r[0]).collect();

        // region predicate: r_name == MIDDLE EAST. q8_obj.rs proves this only
        // indirectly, through the witnessed target_rkey and its lookup, but the
        // input channel of condition (4) needs the bit per region row.
        let r_keep_u64: Vec<u64> = region_t
            .iter()
            .map(|r| (r[1] == const_region_name_hash) as u64)
            .collect();

        // customer keep: in target region
        let mut c_region_u64 = vec![PAD_YEAR; customer_t.len()];
        let mut c_keep_u64 = vec![0u64; customer_t.len()];
        for (i, c) in customer_t.iter().enumerate() {
            let nkey = c[1];
            let rkey = *nat_region.get(&nkey).unwrap_or(&PAD_YEAR);
            c_region_u64[i] = rkey;
            c_keep_u64[i] = if rkey == target_rkey_u64 { 1 } else { 0 };
        }

        // orders keep: year in {1995,1996} and customer keep
        let mut year_keep_u64 = vec![0u64; orders_t.len()];
        let mut o_c_keep_u64 = vec![0u64; orders_t.len()];
        let mut o_keep_u64 = vec![0u64; orders_t.len()];
        let mut o_c_nation_u64 = vec![PAD_YEAR; orders_t.len()];
        let mut o_c_region_u64 = vec![PAD_YEAR; orders_t.len()];

        let mut keep_orderkeys: HashSet<u64> = HashSet::new();

        for (i, o) in orders_t.iter().enumerate() {
            let okey = o[0];
            let ckey = o[1];
            let year = o[2];

            // year keep
            year_keep_u64[i] = if year == 1995 || year == 1996 { 1 } else { 0 };

            // customer keep
            let nkey = *cust_nat.get(&ckey).unwrap_or(&PAD_YEAR);
            o_c_nation_u64[i] = nkey;
            let rkey = *nat_region.get(&nkey).unwrap_or(&PAD_YEAR);
            o_c_region_u64[i] = rkey;

            let ck = if rkey == target_rkey_u64 { 1 } else { 0 };
            o_c_keep_u64[i] = ck;

            let ok = year_keep_u64[i] * ck;
            o_keep_u64[i] = ok;

            if ok == 1 {
                keep_orderkeys.insert(okey);
            }
        }

        // ---------- the clean instance ----------
        // A lineitem row is joinable when all three of its children have a
        // partner that survives its own subtree, which is exactly the condition
        // q8_obj.rs uses to build its join prefix.
        let l_ok_u64: Vec<u64> = lineitem_t
            .iter()
            .map(|r| {
                (keep_orderkeys.contains(&r[0])
                    && partkeys_keep.contains(&r[1])
                    && supp_nat.contains_key(&r[2])
                    && nat_name.contains_key(&supp_nat[&r[2]])) as u64
            })
            .collect();

        // The honest prover's `R^c` is the fully reduced instance, so at the
        // root it is exactly the set of joinable rows.
        let tamper = HIDE_ONE_CLEAN_TUPLE.load(Ordering::Relaxed);
        let all_clean = MARK_ALL_CLEAN.load(Ordering::Relaxed);
        let mut cln_l = l_ok_u64.clone();
        if tamper {
            if let Some(i) = cln_l.iter().position(|&x| x == 1) {
                cln_l[i] = 0;
            }
        }

        // Top-down half of the reduction: a tuple of a child is clean when it is
        // the partner of a clean tuple of its parent.
        let cl_pkeys: HashSet<u64> = (0..lineitem_t.len())
            .filter(|&i| cln_l[i] == 1)
            .map(|i| lineitem_t[i][1])
            .collect();
        let cl_okeys: HashSet<u64> = (0..lineitem_t.len())
            .filter(|&i| cln_l[i] == 1)
            .map(|i| lineitem_t[i][0])
            .collect();
        let cl_skeys: HashSet<u64> = (0..lineitem_t.len())
            .filter(|&i| cln_l[i] == 1)
            .map(|i| lineitem_t[i][2])
            .collect();

        let cln_p: Vec<u64> = (0..part_t.len())
            .map(|i| (p_keep_u64[i] == 1 && cl_pkeys.contains(&part_t[i][0])) as u64)
            .collect();
        let cln_o: Vec<u64> = (0..orders_t.len())
            .map(|i| (o_keep_u64[i] == 1 && cl_okeys.contains(&orders_t[i][0])) as u64)
            .collect();
        let cln_s: Vec<u64> = (0..supplier_t.len())
            .map(|i| cl_skeys.contains(&supplier_t[i][0]) as u64)
            .collect();

        let cl_snkeys: HashSet<u64> = (0..supplier_t.len())
            .filter(|&i| cln_s[i] == 1)
            .map(|i| supplier_t[i][1])
            .collect();
        let cln_n2: Vec<u64> = (0..nation_t.len())
            .map(|i| cl_snkeys.contains(&nation_t[i][0]) as u64)
            .collect();

        let cl_ckeys: HashSet<u64> = (0..orders_t.len())
            .filter(|&i| cln_o[i] == 1)
            .map(|i| orders_t[i][1])
            .collect();
        let cln_c: Vec<u64> = (0..customer_t.len())
            .map(|i| cl_ckeys.contains(&customer_t[i][0]) as u64)
            .collect();

        let cl_cnkeys: HashSet<u64> = (0..customer_t.len())
            .filter(|&i| cln_c[i] == 1)
            .map(|i| customer_t[i][1])
            .collect();
        let cln_n1: Vec<u64> = (0..nation_t.len())
            .map(|i| cl_cnkeys.contains(&nation_t[i][0]) as u64)
            .collect();

        let cl_nrkeys: HashSet<u64> = (0..nation_t.len())
            .filter(|&i| cln_n1[i] == 1)
            .map(|i| nation_t[i][1])
            .collect();
        let cln_r: Vec<u64> = (0..region_t.len())
            .map(|i| (r_keep_u64[i] == 1 && cl_nrkeys.contains(&region_t[i][0])) as u64)
            .collect();

        // Test hook: skip the reduction and declare every real tuple clean, so
        // that the residual section of every relation stays empty. The indicator
        // is then keep, which is what the link gates pad to on dropped rows, so
        // Conservation still holds and both channels of condition (4) agree on
        // every row. Only Pairwise Consistency can see that this partition is not
        // the reduced instance.
        let (cln_l, cln_p, cln_o, cln_s, cln_n2, cln_c, cln_n1, cln_r) = if all_clean {
            (
                vec![1u64; lineitem_t.len()],
                p_keep_u64.clone(),
                o_keep_u64.clone(),
                vec![1u64; supplier_t.len()],
                vec![1u64; nation_t.len()],
                vec![1u64; customer_t.len()],
                vec![1u64; nation_t.len()],
                r_keep_u64.clone(),
            )
        } else {
            (cln_l, cln_p, cln_o, cln_s, cln_n2, cln_c, cln_n1, cln_r)
        };

        // ---------- the two sides of every Conservation Check ----------
        // input side: keep? [row..., c] : PAD, which the link gates force.
        let filt_side = |rows: &[Vec<u64>],
                         keep: &[u64],
                         cln: &[u64],
                         pad: &[u64]|
         -> Vec<Vec<u64>> {
            (0..rows.len())
                .map(|i| {
                    if keep[i] == 1 {
                        let mut v = rows[i].clone();
                        v.push(cln[i]);
                        v
                    } else {
                        pad.to_vec()
                    }
                })
                .collect()
        };
        // partition side: [clean rows | residual rows | pad rows], with the
        // indicator held constant over each of the first two sections.
        let part_side = |rows: &[Vec<u64>],
                         keep: &[u64],
                         cln: &[u64],
                         pad: &[u64]|
         -> (Vec<Vec<u64>>, usize, usize) {
            let mut out: Vec<Vec<u64>> = vec![];
            for i in 0..rows.len() {
                if keep[i] == 1 && cln[i] == 1 {
                    let mut v = rows[i].clone();
                    v.push(1);
                    out.push(v);
                }
            }
            let n_cln = out.len();
            for i in 0..rows.len() {
                if keep[i] == 1 && cln[i] == 0 {
                    let mut v = rows[i].clone();
                    v.push(0);
                    out.push(v);
                }
            }
            let n_res = out.len() - n_cln;
            while out.len() < rows.len() {
                out.push(pad.to_vec());
            }
            (out, n_cln, n_res)
        };

        let ones = |n: usize| vec![1u64; n];

        // part and orders keep their filter-and-pad projection, one column wider
        let pad_p = vec![PAD_PKEY, PAD_PTYPE, 0];
        let p_filt_pad_u64 = filt_side(&part_t, &p_keep_u64, &cln_p, &pad_p);
        let (p_join_pad_u64, p_n_cln, p_n_res) =
            part_side(&part_t, &p_keep_u64, &cln_p, &pad_p);

        // [o_orderkey, o_year, o_custkey]: the custkey rides along so that the
        // orders -> customer edge of condition (3) can read it out of the clean
        // section of this partition group.
        let orders_proj: Vec<Vec<u64>> = orders_t.iter().map(|r| vec![r[0], r[2], r[1]]).collect();
        let pad_o = vec![PAD_OKEY, PAD_OYEAR, PAD_OCUST, 0];
        let o_filt_pad_u64 = filt_side(&orders_proj, &o_keep_u64, &cln_o, &pad_o);
        let (o_join_pad_u64, o_n_cln, o_n_res) =
            part_side(&orders_proj, &o_keep_u64, &cln_o, &pad_o);

        // the relations without a predicate: every row is clean or residual, so
        // the partition side has no pad section at all
        let pad_r = vec![PAD_RKEY, PAD_RNAME, 0];
        let r_filt_pad_u64 = filt_side(&region_t, &r_keep_u64, &cln_r, &pad_r);
        let (r_part_pad_u64, r_n_cln, r_n_res) =
            part_side(&region_t, &r_keep_u64, &cln_r, &pad_r);

        let pad_n = vec![MAX_SENTINEL, MAX_SENTINEL, MAX_SENTINEL, 0];
        let (n1_part_pad_u64, n1_n_cln, n1_n_res) =
            part_side(&nation_t, &ones(nation_t.len()), &cln_n1, &pad_n);
        let (n2_part_pad_u64, n2_n_cln, n2_n_res) =
            part_side(&nation_t, &ones(nation_t.len()), &cln_n2, &pad_n);

        let pad_c = vec![MAX_SENTINEL, MAX_SENTINEL, 0];
        let (c_part_pad_u64, c_n_cln, c_n_res) =
            part_side(&customer_t, &ones(customer_t.len()), &cln_c, &pad_c);
        let (s_part_pad_u64, s_n_cln, s_n_res) =
            part_side(&supplier_t, &ones(supplier_t.len()), &cln_s, &pad_c);

        // the root: l_join_pad is already laid out as [clean | residual | pad],
        // so the clean prefix the join lookups run over IS the clean section
        let pad_l = vec![PAD_LOKEY, PAD_LPKEY, PAD_LSKEY, PAD_LEXT, PAD_LDISC, 0];
        let (l_join_pad_u64, join_len, _l_n_res) =
            part_side(&lineitem_t, &ones(lineitem_t.len()), &cln_l, &pad_l);

        // Pairwise Consistency needs no host-side key table: each direction looks
        // one clean key column up directly in the adjacent clean key column, so
        // the only witness it consumes is the partition groups already built
        // above and the q_pw_in row ranges enabled over their clean sections.

        // build all_nations (aligned to l_join_pad rows): join rows real, others PAD
        let mut all_nations_u64: Vec<[u64; 3]> =
            vec![[PAD_AN_YEAR, PAD_AN_VOL, PAD_AN_NAT]; lineitem_t.len()];

        // maps for quick attach:
        let orders_year: HashMap<u64, u64> = orders_t.iter().map(|r| (r[0], r[2])).collect();

        for i in 0..join_len {
            let r = &l_join_pad_u64[i];
            let okey = r[0];
            let skey = r[2];
            let ext = r[3];
            let disc = r[4];

            // Under the MARK_ALL_CLEAN hook the clean section holds every lineitem
            // row, so these attaches can miss; on the honest witness they never
            // do, because a clean row is joinable by construction.
            let year = *orders_year.get(&okey).unwrap_or(&PAD_YEAR);
            let nkey = *supp_nat.get(&skey).unwrap_or(&PAD_YEAR);
            let nname = *nat_name.get(&nkey).unwrap_or(&PAD_AN_NAT);

            let vol_u64 = ((ext as u128) * ((SCALE - disc) as u128) % (u64::MAX as u128)) as u64;

            all_nations_u64[i] = [year, vol_u64, nname];
        }

        // sorted by year asc
        let mut all_nations_sorted_u64 = all_nations_u64.clone();
        all_nations_sorted_u64.sort_by(|a, b| a[0].cmp(&b[0])); // year asc, pads (PAD_YEAR) end

        // group sums and emit sparse results
        let n = all_nations_sorted_u64.len();
        let mut cond_flag_u64 = vec![0u64; n];
        let mut run_num_u64 = vec![0u64; n];
        let mut run_den_u64 = vec![0u64; n];
        let mut res_pad_u64: Vec<[u64; 4]> =
            vec![[PAD_RES_YEAR, PAD_RES_NUM, PAD_RES_DEN, 0u64]; n];

        let mut acc_num: u128 = 0;
        let mut acc_den: u128 = 0;
        let mut prev_year: Option<u64> = None;

        for i in 0..n {
            let y = all_nations_sorted_u64[i][0];
            let v = all_nations_sorted_u64[i][1] as u128;
            let nat = all_nations_sorted_u64[i][2];

            let is_cond = if nat == cond_nation_hash { 1u64 } else { 0u64 };
            cond_flag_u64[i] = is_cond;

            if prev_year == Some(y) {
                acc_den = acc_den.wrapping_add(v);
                if is_cond == 1 {
                    acc_num = acc_num.wrapping_add(v);
                }
            } else {
                acc_den = v;
                acc_num = if is_cond == 1 { v } else { 0 };
            }
            run_den_u64[i] = acc_den as u64;
            run_num_u64[i] = acc_num as u64;

            let next_year = if i + 1 < n {
                all_nations_sorted_u64[i + 1][0]
            } else {
                PAD_YEAR
            };
            let is_last = next_year != y;

            if is_last && y != PAD_YEAR {
                // witness share in field later; store placeholder 0 here
                res_pad_u64[i] = [y, run_num_u64[i], run_den_u64[i], 0u64];
            }

            prev_year = Some(y);
        }

        // compact results: bring real rows first (already year-sorted), then pads
        let mut real_rows: Vec<[u64; 4]> = res_pad_u64
            .iter()
            .copied()
            .filter(|r| r[0] != PAD_YEAR)
            .collect();
        real_rows.sort_by(|a, b| a[0].cmp(&b[0]));

        let mut res_sorted_u64: Vec<[u64; 4]> = vec![];
        res_sorted_u64.extend(real_rows.into_iter());
        while res_sorted_u64.len() < n {
            res_sorted_u64.push([PAD_RES_YEAR, PAD_RES_NUM, PAD_RES_DEN, 0u64]);
        }

        // ---------- assignment ----------
        let out_cell = layouter.assign_region(
            || "Q8 witness",
            |mut region| {
                // chips
                let iz_part_chip = IsZeroChip::construct(self.config.iz_part_type.clone());
                let iz_ck_chip = IsZeroChip::construct(self.config.iz_c_keep.clone());
                let iz_ock_chip = IsZeroChip::construct(self.config.iz_o_c_keep.clone());
                let iz_yk_chip = IsZeroChip::construct(self.config.iz_year_keep.clone());
                let iz_year_eq_chip = IsZeroChip::construct(self.config.iz_year_eq.clone());
                let iz_is_cond_chip = IsZeroChip::construct(self.config.iz_is_cond.clone());
                let iz_sp_chip = IsZeroChip::construct(self.config.iz_same_prev.clone());
                let iz_sn_chip = IsZeroChip::construct(self.config.iz_same_next.clone());
                let iz_res_eq_chip = IsZeroChip::construct(self.config.iz_res_year_eq.clone());
                let iz_rk_chip = IsZeroChip::construct(self.config.iz_r_keep.clone());

                // ---- base tables assignment ----
                for i in 0..region_t.len() {
                    self.config.q_tbl_region.enable(&mut region, i)?;
                    region.assign_advice(
                        || "regionkey",
                        self.config.region[0],
                        i,
                        || Value::known(F::from(region_t[i][0])),
                    )?;
                    region.assign_advice(
                        || "rname",
                        self.config.region[1],
                        i,
                        || Value::known(F::from(region_t[i][1])),
                    )?;

                    // region predicate, clean indicator and the shifted key of
                    // the region leaf of the Cardinality Preservation Check
                    self.config.q_region_pred.enable(&mut region, i)?;
                    region.assign_advice(
                        || "r_keep",
                        self.config.r_keep,
                        i,
                        || Value::known(F::from(r_keep_u64[i])),
                    )?;
                    iz_rk_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(region_t[i][1]) - F::from(const_region_name_hash)),
                    )?;
                    region.assign_advice(
                        || "cflag region",
                        self.config.cflag[0],
                        i,
                        || Value::known(F::from(cln_r[i])),
                    )?;
                    region.assign_advice(
                        || "rk_shift",
                        self.config.rk_shift,
                        i,
                        || Value::known(F::from(region_t[i][0] + SHIFT_ID)),
                    )?;
                    self.config.perm_region.q_perm1.enable(&mut region, i)?;
                    self.config.perm_region.q_perm2.enable(&mut region, i)?;
                    for j in 0..3 {
                        region.assign_advice(
                            || "r_filt_pad",
                            self.config.r_filt_pad[j],
                            i,
                            || Value::known(F::from(r_filt_pad_u64[i][j])),
                        )?;
                        region.assign_advice(
                            || "r_part_pad",
                            self.config.r_part_pad[j],
                            i,
                            || Value::known(F::from(r_part_pad_u64[i][j])),
                        )?;
                    }
                }

                for i in 0..nation_t.len() {
                    self.config.q_tbl_nation.enable(&mut region, i)?;
                    region.assign_advice(
                        || "nkey",
                        self.config.nation[0],
                        i,
                        || Value::known(F::from(nation_t[i][0])),
                    )?;
                    region.assign_advice(
                        || "n_rkey",
                        self.config.nation[1],
                        i,
                        || Value::known(F::from(nation_t[i][1])),
                    )?;
                    region.assign_advice(
                        || "nname",
                        self.config.nation[2],
                        i,
                        || Value::known(F::from(nation_t[i][2])),
                    )?;

                    // nation is two relations of the join tree, n1 under
                    // customer and n2 under supplier, so it carries two
                    // indicators and two Conservation Checks over the same rows
                    self.config.q_nation_cp.enable(&mut region, i)?;
                    region.assign_advice(
                        || "nk_shift",
                        self.config.nk_shift,
                        i,
                        || Value::known(F::from(nation_t[i][0] + SHIFT_ID)),
                    )?;
                    region.assign_advice(
                        || "n_rk_shift",
                        self.config.n_rk_shift,
                        i,
                        || Value::known(F::from(nation_t[i][1] + SHIFT_ID)),
                    )?;
                    region.assign_advice(
                        || "cp_one",
                        self.config.cp_one,
                        i,
                        || Value::known(F::ONE),
                    )?;
                    region.assign_advice(
                        || "cflag nation n1",
                        self.config.cflag[1],
                        i,
                        || Value::known(F::from(cln_n1[i])),
                    )?;
                    region.assign_advice(
                        || "cflag nation n2",
                        self.config.cflag[2],
                        i,
                        || Value::known(F::from(cln_n2[i])),
                    )?;
                    self.config.perm_nation1.q_perm1.enable(&mut region, i)?;
                    self.config.perm_nation1.q_perm2.enable(&mut region, i)?;
                    self.config.perm_nation2.q_perm1.enable(&mut region, i)?;
                    self.config.perm_nation2.q_perm2.enable(&mut region, i)?;
                    for j in 0..4 {
                        region.assign_advice(
                            || "n1_part_pad",
                            self.config.n1_part_pad[j],
                            i,
                            || Value::known(F::from(n1_part_pad_u64[i][j])),
                        )?;
                        region.assign_advice(
                            || "n2_part_pad",
                            self.config.n2_part_pad[j],
                            i,
                            || Value::known(F::from(n2_part_pad_u64[i][j])),
                        )?;
                    }
                }

                for i in 0..customer_t.len() {
                    self.config.q_tbl_customer.enable(&mut region, i)?;
                    region.assign_advice(
                        || "ckey",
                        self.config.customer[0],
                        i,
                        || Value::known(F::from(customer_t[i][0])),
                    )?;
                    region.assign_advice(
                        || "c_nkey",
                        self.config.customer[1],
                        i,
                        || Value::known(F::from(customer_t[i][1])),
                    )?;

                    self.config.q_cust_cp.enable(&mut region, i)?;
                    region.assign_advice(
                        || "c_nk_shift",
                        self.config.c_nk_shift,
                        i,
                        || Value::known(F::from(customer_t[i][1] + SHIFT_ID)),
                    )?;
                    region.assign_advice(
                        || "cflag customer",
                        self.config.cflag[3],
                        i,
                        || Value::known(F::from(cln_c[i])),
                    )?;
                    self.config.perm_customer.q_perm1.enable(&mut region, i)?;
                    self.config.perm_customer.q_perm2.enable(&mut region, i)?;
                    for j in 0..3 {
                        region.assign_advice(
                            || "c_part_pad",
                            self.config.c_part_pad[j],
                            i,
                            || Value::known(F::from(c_part_pad_u64[i][j])),
                        )?;
                    }
                }

                for i in 0..orders_t.len() {
                    region.assign_advice(
                        || "okey",
                        self.config.orders[0],
                        i,
                        || Value::known(F::from(orders_t[i][0])),
                    )?;
                    region.assign_advice(
                        || "o_cust",
                        self.config.orders[1],
                        i,
                        || Value::known(F::from(orders_t[i][1])),
                    )?;
                    region.assign_advice(
                        || "oyear",
                        self.config.orders[2],
                        i,
                        || Value::known(F::from(orders_t[i][2])),
                    )?;
                }

                for i in 0..part_t.len() {
                    region.assign_advice(
                        || "pkey",
                        self.config.part[0],
                        i,
                        || Value::known(F::from(part_t[i][0])),
                    )?;
                    region.assign_advice(
                        || "ptype",
                        self.config.part[1],
                        i,
                        || Value::known(F::from(part_t[i][1])),
                    )?;
                }

                for i in 0..supplier_t.len() {
                    self.config.q_tbl_supplier.enable(&mut region, i)?;
                    region.assign_advice(
                        || "skey",
                        self.config.supplier[0],
                        i,
                        || Value::known(F::from(supplier_t[i][0])),
                    )?;
                    region.assign_advice(
                        || "s_nkey",
                        self.config.supplier[1],
                        i,
                        || Value::known(F::from(supplier_t[i][1])),
                    )?;

                    self.config.q_supp_cp.enable(&mut region, i)?;
                    region.assign_advice(
                        || "s_nk_shift",
                        self.config.s_nk_shift,
                        i,
                        || Value::known(F::from(supplier_t[i][1] + SHIFT_ID)),
                    )?;
                    region.assign_advice(
                        || "cflag supplier",
                        self.config.cflag[6],
                        i,
                        || Value::known(F::from(cln_s[i])),
                    )?;
                    self.config.perm_supplier.q_perm1.enable(&mut region, i)?;
                    self.config.perm_supplier.q_perm2.enable(&mut region, i)?;
                    for j in 0..3 {
                        region.assign_advice(
                            || "s_part_pad",
                            self.config.s_part_pad[j],
                            i,
                            || Value::known(F::from(s_part_pad_u64[i][j])),
                        )?;
                    }
                }

                for i in 0..lineitem_t.len() {
                    for j in 0..5 {
                        region.assign_advice(
                            || "lineitem",
                            self.config.lineitem[j],
                            i,
                            || Value::known(F::from(lineitem_t[i][j])),
                        )?;
                    }
                    // the root carries no predicate, so its raw indicator is the
                    // input side of its Conservation Check
                    region.assign_advice(
                        || "cflag lineitem",
                        self.config.cflag[7],
                        i,
                        || Value::known(F::from(cln_l[i])),
                    )?;
                }

                // ---- parameters repeated on row 0..max_len ----
                let max_len = region_t
                    .len()
                    .max(nation_t.len())
                    .max(customer_t.len())
                    .max(orders_t.len())
                    .max(part_t.len())
                    .max(supplier_t.len())
                    .max(lineitem_t.len())
                    .max(1);

                for i in 0..max_len {
                    region.assign_advice(
                        || "cond_nation",
                        self.config.cond_nation,
                        i,
                        || Value::known(F::from(cond_nation_hash)),
                    )?;
                    region.assign_advice(
                        || "const_rname",
                        self.config.const_rname,
                        i,
                        || Value::known(F::from(const_region_name_hash)),
                    )?;
                    region.assign_advice(
                        || "const_ptype",
                        self.config.const_ptype,
                        i,
                        || Value::known(F::from(const_part_type_hash)),
                    )?;
                    region.assign_advice(
                        || "target_rkey",
                        self.config.target_rkey,
                        i,
                        || Value::known(F::from(target_rkey_u64)),
                    )?;
                }

                // prove target_rkey exists (do it once at row 0)
                self.config.q_lkp_target_region.enable(&mut region, 0)?;

                // ---- part predicate + p_keep + p_filt_pad + p_join_pad + perm selectors ----
                for i in 0..part_t.len() {
                    self.config.q_part_pred.enable(&mut region, i)?;
                    self.config.q_part_link.enable(&mut region, i)?;
                    self.config.perm_part.q_perm1.enable(&mut region, i)?;
                    self.config.perm_part.q_perm2.enable(&mut region, i)?;
                    self.config.q_tbl_partf.enable(&mut region, i)?;
                    // link gate
                    // enable part-link selector on all part rows
                    // (we used q_part_link = selector created in configure but not stored;
                    //  simplest: just reuse q_part_pred for iz assignment and separately assign p_filt_pad by witness.)
                    iz_part_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(part_t[i][1]) - F::from(const_part_type_hash)),
                    )?;

                    region.assign_advice(
                        || "p_keep",
                        self.config.p_keep,
                        i,
                        || Value::known(F::from(p_keep_u64[i])),
                    )?;
                    region.assign_advice(
                        || "cflag part",
                        self.config.cflag[5],
                        i,
                        || Value::known(F::from(cln_p[i])),
                    )?;

                    // p_filt_pad + p_join_pad, the last column being keep * c
                    for j in 0..3 {
                        region.assign_advice(
                            || "p_filt",
                            self.config.p_filt_pad[j],
                            i,
                            || Value::known(F::from(p_filt_pad_u64[i][j])),
                        )?;
                        region.assign_advice(
                            || "p_join",
                            self.config.p_join_pad[j],
                            i,
                            || Value::known(F::from(p_join_pad_u64[i][j])),
                        )?;
                    }
                }

                // ---- customer attach c_regionkey, c_keep ----
                for i in 0..customer_t.len() {
                    self.config.q_lkp_c_nat_region.enable(&mut region, i)?;
                    region.assign_advice(
                        || "c_regionkey",
                        self.config.c_regionkey,
                        i,
                        || Value::known(F::from(c_region_u64[i])),
                    )?;
                    region.assign_advice(
                        || "c_keep",
                        self.config.c_keep,
                        i,
                        || Value::known(F::from(c_keep_u64[i])),
                    )?;
                    iz_ck_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(c_region_u64[i]) - F::from(target_rkey_u64)),
                    )?;
                }

                // ---- orders attach customer/nation, keep flags, filtered orders tables, perm ----
                for i in 0..orders_t.len() {
                    self.config.q_orders_attach_c.enable(&mut region, i)?;
                    self.config.q_orders_attach_r.enable(&mut region, i)?;
                    self.config.q_orders_keep_gate.enable(&mut region, i)?;
                    self.config.q_orders_filt_gate.enable(&mut region, i)?;

                    self.config.perm_orders.q_perm1.enable(&mut region, i)?;
                    self.config.perm_orders.q_perm2.enable(&mut region, i)?;
                    self.config.q_tbl_ordersf.enable(&mut region, i)?;

                    // attached fields
                    region.assign_advice(
                        || "o_c_nationkey",
                        self.config.o_c_nationkey,
                        i,
                        || Value::known(F::from(o_c_nation_u64[i])),
                    )?;
                    region.assign_advice(
                        || "o_c_regionkey",
                        self.config.o_c_regionkey,
                        i,
                        || Value::known(F::from(o_c_region_u64[i])),
                    )?;
                    region.assign_advice(
                        || "o_c_keep",
                        self.config.o_c_keep,
                        i,
                        || Value::known(F::from(o_c_keep_u64[i])),
                    )?;
                    iz_ock_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(o_c_region_u64[i]) - F::from(target_rkey_u64)),
                    )?;

                    region.assign_advice(
                        || "year_keep",
                        self.config.year_keep,
                        i,
                        || Value::known(F::from(year_keep_u64[i])),
                    )?;
                    // is_zero input: (y-1995)*(y-1996)
                    let y_f = F::from(orders_t[i][2]);
                    let t_f = (y_f - F::from(1995u64)) * (y_f - F::from(1996u64));
                    iz_yk_chip.assign(&mut region, i, Value::known(t_f))?;

                    region.assign_advice(
                        || "o_keep",
                        self.config.o_keep,
                        i,
                        || Value::known(F::from(o_keep_u64[i])),
                    )?;
                    region.assign_advice(
                        || "cflag orders",
                        self.config.cflag[4],
                        i,
                        || Value::known(F::from(cln_o[i])),
                    )?;

                    // filtered orders tables, the last column being keep * c
                    for j in 0..4 {
                        region.assign_advice(
                            || "o_filt",
                            self.config.o_filt_pad[j],
                            i,
                            || Value::known(F::from(o_filt_pad_u64[i][j])),
                        )?;
                        region.assign_advice(
                            || "o_join",
                            self.config.o_join_pad[j],
                            i,
                            || Value::known(F::from(o_join_pad_u64[i][j])),
                        )?;
                    }
                }

                // ---- lineitem permutation to l_join_pad ----
                for i in 0..lineitem_t.len() {
                    self.config.perm_lineitem.q_perm1.enable(&mut region, i)?;
                    self.config.perm_lineitem.q_perm2.enable(&mut region, i)?;
                    for j in 0..6 {
                        region.assign_advice(
                            || "l_join_pad",
                            self.config.l_join_pad[j],
                            i,
                            || Value::known(F::from(l_join_pad_u64[i][j])),
                        )?;
                    }
                }

                // ---- clean indicator on the partition side of every Conservation Check ----
                // 1 on the rows of R^c, 0 on the rows of R^r; the pad rows are
                // pinned by the shuffle itself.
                for (idx, (n_cln, n_res)) in [
                    (r_n_cln, r_n_res),
                    (n1_n_cln, n1_n_res),
                    (n2_n_cln, n2_n_res),
                    (c_n_cln, c_n_res),
                    (o_n_cln, o_n_res),
                    (p_n_cln, p_n_res),
                    (s_n_cln, s_n_res),
                    (join_len, lineitem_t.len() - join_len),
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
                    // Pairwise Consistency: the clean rows of this relation, and
                    // only those. The same selector is the input side of every
                    // lookup out of this relation and the table side of every
                    // lookup into it.
                    for i in 0..*n_cln {
                        self.config.q_pw_in[idx].enable(&mut region, i)?;
                    }
                }

                // ---- enable join-only selectors on prefix join_len ----
                for i in 0..join_len {
                    self.config.q_lkp_partkey.enable(&mut region, i)?;
                    self.config.q_lkp_orders.enable(&mut region, i)?;
                    self.config.q_lkp_supplier.enable(&mut region, i)?;
                    self.config.q_lkp_nation2.enable(&mut region, i)?;
                    self.config.q_volume.enable(&mut region, i)?;
                }

                // ---- assign attached fields + volume + all_nations (join rows real; others pad) ----
                for i in 0..lineitem_t.len() {
                    if i < join_len {
                        let r = &l_join_pad_u64[i];
                        let okey = r[0];
                        let skey = r[2];
                        let year = *orders_year.get(&okey).unwrap_or(&PAD_YEAR);
                        let nkey = *supp_nat.get(&skey).unwrap_or(&PAD_YEAR);
                        let nname = *nat_name.get(&nkey).unwrap_or(&PAD_AN_NAT);

                        region.assign_advice(
                            || "l_year",
                            self.config.l_year,
                            i,
                            || Value::known(F::from(year)),
                        )?;
                        region.assign_advice(
                            || "l_s_nationkey",
                            self.config.l_s_nationkey,
                            i,
                            || Value::known(F::from(nkey)),
                        )?;
                        region.assign_advice(
                            || "l_nation_hash",
                            self.config.l_nation_hash,
                            i,
                            || Value::known(F::from(nname)),
                        )?;

                        let v = all_nations_u64[i][1];
                        region.assign_advice(
                            || "volume",
                            self.config.volume,
                            i,
                            || Value::known(F::from(v)),
                        )?;

                        region.assign_advice(
                            || "an_year",
                            self.config.all_nations[0],
                            i,
                            || Value::known(F::from(all_nations_u64[i][0])),
                        )?;
                        region.assign_advice(
                            || "an_vol",
                            self.config.all_nations[1],
                            i,
                            || Value::known(F::from(all_nations_u64[i][1])),
                        )?;
                        region.assign_advice(
                            || "an_nat",
                            self.config.all_nations[2],
                            i,
                            || Value::known(F::from(all_nations_u64[i][2])),
                        )?;
                    } else {
                        region.assign_advice(
                            || "l_year",
                            self.config.l_year,
                            i,
                            || Value::known(F::from(PAD_YEAR)),
                        )?;
                        region.assign_advice(
                            || "l_s_nationkey",
                            self.config.l_s_nationkey,
                            i,
                            || Value::known(F::from(PAD_YEAR)),
                        )?;
                        region.assign_advice(
                            || "l_nation_hash",
                            self.config.l_nation_hash,
                            i,
                            || Value::known(F::from(PAD_AN_NAT)),
                        )?;
                        region.assign_advice(
                            || "volume",
                            self.config.volume,
                            i,
                            || Value::known(F::ZERO),
                        )?;

                        region.assign_advice(
                            || "an_year",
                            self.config.all_nations[0],
                            i,
                            || Value::known(F::from(PAD_AN_YEAR)),
                        )?;
                        region.assign_advice(
                            || "an_vol",
                            self.config.all_nations[1],
                            i,
                            || Value::known(F::from(PAD_AN_VOL)),
                        )?;
                        region.assign_advice(
                            || "an_nat",
                            self.config.all_nations[2],
                            i,
                            || Value::known(F::from(PAD_AN_NAT)),
                        )?;
                    }
                }

                // ---- all_nations_sorted + permutation selectors ----
                for i in 0..n {
                    self.config
                        .perm_all_nations
                        .q_perm1
                        .enable(&mut region, i)?;
                    self.config
                        .perm_all_nations
                        .q_perm2
                        .enable(&mut region, i)?;

                    region.assign_advice(
                        || "anS_year",
                        self.config.all_nations_sorted[0],
                        i,
                        || Value::known(F::from(all_nations_sorted_u64[i][0])),
                    )?;
                    region.assign_advice(
                        || "anS_vol",
                        self.config.all_nations_sorted[1],
                        i,
                        || Value::known(F::from(all_nations_sorted_u64[i][1])),
                    )?;
                    region.assign_advice(
                        || "anS_nat",
                        self.config.all_nations_sorted[2],
                        i,
                        || Value::known(F::from(all_nations_sorted_u64[i][2])),
                    )?;
                }

                // ---- sorting constraint on all_nations_sorted ----
                for i in 0..n.saturating_sub(1) {
                    self.config.q_sort_year.enable(&mut region, i)?;
                    iz_year_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(
                            F::from(all_nations_sorted_u64[i][0])
                                - F::from(all_nations_sorted_u64[i + 1][0]),
                        ),
                    )?;
                    lt_year_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(all_nations_sorted_u64[i][0])),
                        Value::known(F::from(all_nations_sorted_u64[i + 1][0])),
                    )?;
                }

                // ---- cond_flag, group selectors, same_prev/next, running sums ----
                for i in 0..n {
                    // Enable the cond_flag constraints (you had this commented out)
                    self.config.q_is_cond.enable(&mut region, i)?;

                    region.assign_advice(
                        || "cond_flag",
                        self.config.cond_flag,
                        i,
                        || Value::known(F::from(cond_flag_u64[i])),
                    )?;

                    iz_is_cond_chip.assign(
                        &mut region,
                        i,
                        Value::known(
                            F::from(all_nations_sorted_u64[i][2]) - F::from(cond_nation_hash),
                        ),
                    )?;

                    // Only enable q_line where Rotation::next() is valid
                    if i + 1 < n {
                        self.config.q_line.enable(&mut region, i)?;
                        iz_sn_chip.assign(
                            &mut region,
                            i,
                            Value::known(
                                F::from(all_nations_sorted_u64[i + 1][0])
                                    - F::from(all_nations_sorted_u64[i][0]),
                            ),
                        )?;
                    }

                    region.assign_advice(
                        || "run_num",
                        self.config.run_num,
                        i,
                        || Value::known(F::from(run_num_u64[i])),
                    )?;
                    region.assign_advice(
                        || "run_den",
                        self.config.run_den,
                        i,
                        || Value::known(F::from(run_den_u64[i])),
                    )?;
                }

                if n > 0 {
                    for i in 0..n - 1 {
                        self.config.q_emit.enable(&mut region, i)?;
                    }
                    self.config.q_emit_last.enable(&mut region, n - 1)?;
                }

                if n > 0 {
                    self.config.q_first.enable(&mut region, 0)?;
                }
                for i in 1..n {
                    self.config.q_accu.enable(&mut region, i)?;
                    iz_sp_chip.assign(
                        &mut region,
                        i,
                        Value::known(
                            F::from(all_nations_sorted_u64[i][0])
                                - F::from(all_nations_sorted_u64[i - 1][0]),
                        ),
                    )?;
                }
                for i in 0..n.saturating_sub(1) {
                    iz_sn_chip.assign(
                        &mut region,
                        i,
                        Value::known(
                            F::from(all_nations_sorted_u64[i + 1][0])
                                - F::from(all_nations_sorted_u64[i][0]),
                        ),
                    )?;
                }
                // if n > 0 {
                //     iz_sn_chip.assign(
                //         &mut region,
                //         n - 1,
                //         Value::known(F::from(PAD_YEAR) - F::from(all_nations_sorted_u64[n - 1][0])),
                //     )?;
                // }

                // ---- perm res_pad -> res_sorted ----
                for i in 0..n {
                    self.config.perm_res.q_perm1.enable(&mut region, i)?;
                    self.config.perm_res.q_perm2.enable(&mut region, i)?;
                    // self.config.q_emit.enable(&mut region, i)?;

                    // assign res_sorted, and compute share witness on real rows:
                    let y = res_sorted_u64[i][0];
                    let num = res_sorted_u64[i][1];
                    let den = res_sorted_u64[i][2];

                    // share in field: share = num / den if den != 0, else 0
                    let share_f = if y != PAD_YEAR && den != 0 {
                        let numf = F::from(num);
                        let denf = F::from(den);
                        numf * denf.invert().unwrap()
                    } else {
                        F::ZERO
                    };

                    region.assign_advice(
                        || "resS_year",
                        self.config.res_sorted[0],
                        i,
                        || Value::known(F::from(y)),
                    )?;
                    region.assign_advice(
                        || "resS_num",
                        self.config.res_sorted[1],
                        i,
                        || Value::known(F::from(num)),
                    )?;
                    region.assign_advice(
                        || "resS_den",
                        self.config.res_sorted[2],
                        i,
                        || Value::known(F::from(den)),
                    )?;
                    region.assign_advice(
                        || "resS_share",
                        self.config.res_sorted[3],
                        i,
                        || Value::known(share_f),
                    )?;
                }

                // ---- order + share constraints on res_sorted ----
                for i in 0..n.saturating_sub(1) {
                    self.config.q_sort_res.enable(&mut region, i)?;
                    iz_res_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(
                            F::from(res_sorted_u64[i][0]) - F::from(res_sorted_u64[i + 1][0]),
                        ),
                    )?;
                    lt_res_year_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(res_sorted_u64[i][0])),
                        Value::known(F::from(res_sorted_u64[i + 1][0])),
                    )?;
                }
                for i in 0..n {
                    self.config.q_share.enable(&mut region, i)?;
                }

                // ---- assign res_pad (THIS WAS MISSING) ----
                for i in 0..n {
                    let y = res_pad_u64[i][0];
                    let num = res_pad_u64[i][1];
                    let den = res_pad_u64[i][2];

                    // share = num/den for real rows, else 0
                    let share_f = if y != PAD_YEAR && den != 0 {
                        F::from(num) * F::from(den).invert().unwrap()
                    } else {
                        F::ZERO
                    };

                    region.assign_advice(
                        || "res_pad_year",
                        self.config.res_pad[0],
                        i,
                        || Value::known(F::from(y)),
                    )?;
                    region.assign_advice(
                        || "res_pad_num",
                        self.config.res_pad[1],
                        i,
                        || Value::known(F::from(num)),
                    )?;
                    region.assign_advice(
                        || "res_pad_den",
                        self.config.res_pad[2],
                        i,
                        || Value::known(F::from(den)),
                    )?;
                    region.assign_advice(
                        || "res_pad_share",
                        self.config.res_pad[3],
                        i,
                        || Value::known(share_f),
                    )?;
                }

                // ===================== CARDINALITY PRESERVATION CHECK =====================
                // condition (4) of the One-Pass OBJ. One traversal of the join
                // tree carrying two multiplicities per tuple: it starts at the
                // region leaf and at the nation leaf, meets at lineitem, and
                // ends in the single equality between the two root sums. Both
                // channels ride the same rows, the same sorted views and the
                // same key lookups; the clean one only adds its own column,
                // running sum and product.

                // ---- region leaf: [key, pred, keep * c] ----
                let cp_rows_r: Vec<[u64; 3]> = (0..region_t.len())
                    .map(|i| {
                        [
                            region_t[i][0] + SHIFT_ID,
                            r_keep_u64[i],
                            r_filt_pad_u64[i][2],
                        ]
                    })
                    .collect();
                let cp_stage_r = build_cp_stage(&cp_rows_r, MAX_SENTINEL);
                assign_cp_agg(&mut region, &self.config.cp_agg_region, &cp_rows_r, &cp_stage_r)?;

                // ---- nation as n1: no predicate, one child ----
                let n_rkeys: Vec<u64> = nation_t.iter().map(|r| r[1] + SHIFT_ID).collect();
                let fetched_r = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_region,
                    &n_rkeys,
                    &cp_stage_r,
                    MAX_SENTINEL,
                )?;
                let cp_rows_n1: Vec<[u64; 3]> = (0..nation_t.len())
                    .map(|i| {
                        [
                            nation_t[i][0] + SHIFT_ID,
                            fetched_r[i].0,
                            cln_n1[i] * fetched_r[i].1,
                        ]
                    })
                    .collect();
                for i in 0..nation_t.len() {
                    self.config.q_cp_n1.enable(&mut region, i)?;
                    region.assign_advice(
                        || "cp_mu_n1",
                        self.config.cp_mu_n1,
                        i,
                        || Value::known(F::from(cp_rows_n1[i][2])),
                    )?;
                }
                let cp_stage_n1 = build_cp_stage(&cp_rows_n1, MAX_SENTINEL);
                assign_cp_agg(&mut region, &self.config.cp_agg_n1, &cp_rows_n1, &cp_stage_n1)?;

                // ---- customer: no predicate, one child ----
                let c_nkeys: Vec<u64> = customer_t.iter().map(|r| r[1] + SHIFT_ID).collect();
                let fetched_n1 = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_n1,
                    &c_nkeys,
                    &cp_stage_n1,
                    MAX_SENTINEL,
                )?;
                let cp_rows_c: Vec<[u64; 3]> = (0..customer_t.len())
                    .map(|i| {
                        [
                            customer_t[i][0],
                            fetched_n1[i].0,
                            cln_c[i] * fetched_n1[i].1,
                        ]
                    })
                    .collect();
                for i in 0..customer_t.len() {
                    self.config.q_cp_cust.enable(&mut region, i)?;
                    region.assign_advice(
                        || "cp_mu_cust",
                        self.config.cp_mu_cust,
                        i,
                        || Value::known(F::from(cp_rows_c[i][2])),
                    )?;
                }
                let cp_stage_c = build_cp_stage(&cp_rows_c, MAX_SENTINEL);
                assign_cp_agg(&mut region, &self.config.cp_agg_cust, &cp_rows_c, &cp_stage_c)?;

                // ---- orders: the one internal node with a predicate ----
                let o_ckeys: Vec<u64> = orders_t.iter().map(|r| r[1]).collect();
                let fetched_c = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_cust,
                    &o_ckeys,
                    &cp_stage_c,
                    MAX_SENTINEL,
                )?;
                let cp_rows_o: Vec<[u64; 3]> = (0..orders_t.len())
                    .map(|i| {
                        [
                            orders_t[i][0],
                            o_keep_u64[i] * fetched_c[i].0,
                            o_filt_pad_u64[i][3] * fetched_c[i].1,
                        ]
                    })
                    .collect();
                for i in 0..orders_t.len() {
                    self.config.q_cp_ord.enable(&mut region, i)?;
                    region.assign_advice(
                        || "cp_mu_ord all",
                        self.config.cp_mu_ord[0],
                        i,
                        || Value::known(F::from(cp_rows_o[i][1])),
                    )?;
                    region.assign_advice(
                        || "cp_mu_ord cln",
                        self.config.cp_mu_ord[1],
                        i,
                        || Value::known(F::from(cp_rows_o[i][2])),
                    )?;
                }
                let cp_stage_o = build_cp_stage(&cp_rows_o, MAX_SENTINEL);
                assign_cp_agg(&mut region, &self.config.cp_agg_ord, &cp_rows_o, &cp_stage_o)?;

                // ---- nation as n2: a leaf, so its input channel is the constant 1 ----
                let cp_rows_n2: Vec<[u64; 3]> = (0..nation_t.len())
                    .map(|i| [nation_t[i][0] + SHIFT_ID, 1, cln_n2[i]])
                    .collect();
                let cp_stage_n2 = build_cp_stage(&cp_rows_n2, MAX_SENTINEL);
                assign_cp_agg(&mut region, &self.config.cp_agg_n2, &cp_rows_n2, &cp_stage_n2)?;

                // ---- supplier: no predicate, one child ----
                let s_nkeys: Vec<u64> = supplier_t.iter().map(|r| r[1] + SHIFT_ID).collect();
                let fetched_n2 = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_n2,
                    &s_nkeys,
                    &cp_stage_n2,
                    MAX_SENTINEL,
                )?;
                let cp_rows_s: Vec<[u64; 3]> = (0..supplier_t.len())
                    .map(|i| {
                        [
                            supplier_t[i][0],
                            fetched_n2[i].0,
                            cln_s[i] * fetched_n2[i].1,
                        ]
                    })
                    .collect();
                for i in 0..supplier_t.len() {
                    self.config.q_cp_supp.enable(&mut region, i)?;
                    region.assign_advice(
                        || "cp_mu_supp",
                        self.config.cp_mu_supp,
                        i,
                        || Value::known(F::from(cp_rows_s[i][2])),
                    )?;
                }
                let cp_stage_s = build_cp_stage(&cp_rows_s, MAX_SENTINEL);
                assign_cp_agg(&mut region, &self.config.cp_agg_supp, &cp_rows_s, &cp_stage_s)?;

                // ---- part leaf: [key, pred, keep * c] ----
                let cp_rows_p: Vec<[u64; 3]> = (0..part_t.len())
                    .map(|i| [part_t[i][0], p_keep_u64[i], p_filt_pad_u64[i][2]])
                    .collect();
                let cp_stage_p = build_cp_stage(&cp_rows_p, MAX_SENTINEL);
                assign_cp_agg(&mut region, &self.config.cp_agg_part, &cp_rows_p, &cp_stage_p)?;

                // ---- the root: three parent sides over the lineitem base rows ----
                let l_pkeys: Vec<u64> = lineitem_t.iter().map(|r| r[1]).collect();
                let l_okeys: Vec<u64> = lineitem_t.iter().map(|r| r[0]).collect();
                let l_skeys: Vec<u64> = lineitem_t.iter().map(|r| r[2]).collect();
                let fetched_p = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_part,
                    &l_pkeys,
                    &cp_stage_p,
                    MAX_SENTINEL,
                )?;
                let fetched_o = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_ord,
                    &l_okeys,
                    &cp_stage_o,
                    MAX_SENTINEL,
                )?;
                let fetched_s = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_supp,
                    &l_skeys,
                    &cp_stage_s,
                    MAX_SENTINEL,
                )?;

                let mut cp_mu: Vec<(u64, u64)> = Vec::with_capacity(lineitem_t.len());
                for i in 0..lineitem_t.len() {
                    // folded through cp_aux so that the three-child product
                    // never exceeds degree 4 in one constraint
                    let aux_all = fetched_p[i].0 * fetched_o[i].0;
                    let aux_cln = cln_l[i] * fetched_p[i].1;
                    let mu_all = aux_all * fetched_s[i].0;
                    let mu_cln = aux_cln * fetched_o[i].1 * fetched_s[i].1;

                    self.config.q_cp_root.enable(&mut region, i)?;
                    region.assign_advice(
                        || "cp_aux all",
                        self.config.cp_aux[0],
                        i,
                        || Value::known(F::from(aux_all)),
                    )?;
                    region.assign_advice(
                        || "cp_aux cln",
                        self.config.cp_aux[1],
                        i,
                        || Value::known(F::from(aux_cln)),
                    )?;
                    cp_mu.push((mu_all, mu_cln));
                }
                let (cp_all, cp_cln) = assign_cp_root(&mut region, &self.config.cp_root, &cp_mu)?;
                // Under either test hook the two root sums are expected to
                // disagree (HIDE_ONE_CLEAN_TUPLE) or to agree on a partition that
                // is not the reduced instance (MARK_ALL_CLEAN), so neither
                // invariant below holds there.
                if !tamper && !all_clean {
                    debug_assert_eq!(
                        cp_all, cp_cln,
                        "cardinality preservation: |R^c join| != |R join|"
                    );
                    debug_assert_eq!(
                        cp_cln as usize, join_len,
                        "the clean channel must count the clean lineitem rows"
                    );
                }

                // ---- public output (keep the same convention as your previous files) ----
                let out = region.assign_advice(
                    || "instance_test",
                    self.config.instance_test,
                    0,
                    || Value::known(F::from(1u64)),
                )?;
                Ok(out)
            },
        )?;

        Ok(out_cell)
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
    pub region: Vec<Vec<u64>>,
    pub nation: Vec<Vec<u64>>,
    pub customer: Vec<Vec<u64>>,
    pub orders: Vec<Vec<u64>>,
    pub part: Vec<Vec<u64>>,
    pub supplier: Vec<Vec<u64>>,
    pub lineitem: Vec<Vec<u64>>,

    pub cond_nation_hash: u64,
    pub const_region_name_hash: u64,
    pub const_part_type_hash: u64,

    pub _marker: PhantomData<F>,
}

impl<F: Field + Ord> Default for MyCircuit<F> {
    fn default() -> Self {
        Self {
            region: vec![],
            nation: vec![],
            customer: vec![],
            orders: vec![],
            part: vec![],
            supplier: vec![],
            lineitem: vec![],
            cond_nation_hash: 0,
            const_region_name_hash: 0,
            const_part_type_hash: 0,
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
            self.region.clone(),
            self.nation.clone(),
            self.customer.clone(),
            self.orders.clone(),
            self.part.clone(),
            self.supplier.clone(),
            self.lineitem.clone(),
            self.cond_nation_hash,
            self.const_region_name_hash,
            self.const_part_type_hash,
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
            commitment::Params, // IMPORTANT: needed for ParamsIPA::read(...)
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
        let s = s.trim(); // <-- add this
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

        // Adjust paths to your repo
        let region_path = &crate::paths::data_file("region.cvs");
        let nation_path = &crate::paths::data_file("nation.tbl");
        let customer_path = &crate::paths::data_file("customer.tbl");
        let orders_path = &crate::paths::data_file("orders.tbl");
        let part_path = &crate::paths::data_file("part.tbl");
        let supplier_path = &crate::paths::data_file("supplier.tbl");
        let lineitem_path = &crate::paths::data_file("lineitem.tbl");

        let mut region: Vec<Vec<u64>> = vec![];
        let mut nation: Vec<Vec<u64>> = vec![];
        let mut customer: Vec<Vec<u64>> = vec![];
        let mut orders: Vec<Vec<u64>> = vec![];
        let mut part: Vec<Vec<u64>> = vec![];
        let mut supplier: Vec<Vec<u64>> = vec![];
        let mut lineitem: Vec<Vec<u64>> = vec![];

        // Load
        if let Ok(records) = data_processing::region_read_records_from_cvs(region_path) {
            region = records
                .iter()
                .map(|r| vec![r.r_regionkey, string_to_u64(&r.r_name)])
                .collect();
        }

        if let Ok(records) = data_processing::nation_read_records_from_file(nation_path) {
            nation = records
                .iter()
                .map(|r| vec![r.n_nationkey, r.n_regionkey, string_to_u64(&r.n_name)])
                .collect();
        }

        if let Ok(records) = data_processing::customer_read_records_from_file(customer_path) {
            customer = records
                .iter()
                .map(|r| vec![r.c_custkey, r.c_nationkey])
                .collect();
        }

        if let Ok(records) = data_processing::orders_read_records_from_file(orders_path) {
            orders = records
                .iter()
                .map(|r| vec![r.o_orderkey, r.o_custkey, year_from_date(&r.o_orderdate)])
                .collect();
        }

        if let Ok(records) = data_processing::part_read_records_from_file(part_path) {
            part = records
                .iter()
                .map(|r| vec![r.p_partkey, string_to_u64(&r.p_type)])
                .collect();
        }

        if let Ok(records) = data_processing::supplier_read_records_from_file(supplier_path) {
            supplier = records
                .iter()
                .map(|r| vec![r.s_suppkey, r.s_nationkey])
                .collect();
        }

        if let Ok(records) = data_processing::lineitem_read_records_from_file(lineitem_path) {
            lineitem = records
                .iter()
                .map(|r| {
                    vec![
                        r.l_orderkey,
                        r.l_partkey,
                        r.l_suppkey,
                        scale_by_1000(r.l_extendedprice),
                        scale_by_1000(r.l_discount),
                    ]
                })
                .collect();
        }

        // Query constants
        let const_region_name_hash = string_to_u64("MIDDLE EAST");
        let const_part_type_hash = string_to_u64("PROMO BRUSHED COPPER");

        // ':1' parameter (nation name)
        // choose any nation string that exists in your dataset
        let cond_nation_hash = string_to_u64("EGYPT");

        let circuit = MyCircuit::<Fp> {
            region,
            nation,
            customer,
            orders,
            part,
            supplier,
            lineitem,
            cond_nation_hash,
            const_region_name_hash,
            const_part_type_hash,
            _marker: PhantomData,
        };

        let public_input = vec![Fp::from(1)];

        // let test = true;
        let test = false;

        if test {
            let prover = MockProver::run(k, &circuit, vec![public_input]).unwrap();
            prover.assert_satisfied();
        } else {
            let proof_path = &crate::paths::proof_file("proof_obj_q8_test");
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
    }

    /// Fast correctness check of the Cardinality Preservation Check: a truncated
    /// slice of the real dataset under MockProver, which verifies every gate,
    /// shuffle and lookup of the circuit without paying for a real proof.
    #[test]
    fn test_cardinality_preservation() {
        let k = 14;

        // A slice small enough for MockProver but large enough that the
        // reduction drops tuples on every relation and still leaves a non-empty
        // join. The three small tables and customer are taken whole: every
        // orders row has to find its customer, and every customer its nation,
        // because those two attach lookups run on all rows.
        const N_ORD: usize = 3000;
        const N_LINE: usize = 8000;

        let mut region: Vec<Vec<u64>> = vec![];
        let mut nation: Vec<Vec<u64>> = vec![];
        let mut customer: Vec<Vec<u64>> = vec![];
        let mut orders: Vec<Vec<u64>> = vec![];
        let mut part: Vec<Vec<u64>> = vec![];
        let mut supplier: Vec<Vec<u64>> = vec![];
        let mut lineitem: Vec<Vec<u64>> = vec![];

        if let Ok(records) =
            data_processing::region_read_records_from_cvs(&crate::paths::data_file("region.cvs"))
        {
            region = records
                .iter()
                .map(|r| vec![r.r_regionkey, string_to_u64(&r.r_name)])
                .collect();
        }
        if let Ok(records) =
            data_processing::nation_read_records_from_file(&crate::paths::data_file("nation.tbl"))
        {
            nation = records
                .iter()
                .map(|r| vec![r.n_nationkey, r.n_regionkey, string_to_u64(&r.n_name)])
                .collect();
        }
        if let Ok(records) = data_processing::customer_read_records_from_file(
            &crate::paths::data_file("customer.tbl"),
        ) {
            customer = records
                .iter()
                .map(|r| vec![r.c_custkey, r.c_nationkey])
                .collect();
        }
        if let Ok(records) =
            data_processing::orders_read_records_from_file(&crate::paths::data_file("orders.tbl"))
        {
            orders = records
                .iter()
                .take(N_ORD)
                .map(|r| vec![r.o_orderkey, r.o_custkey, year_from_date(&r.o_orderdate)])
                .collect();
        }
        if let Ok(records) =
            data_processing::part_read_records_from_file(&crate::paths::data_file("part.tbl"))
        {
            part = records
                .iter()
                .map(|r| vec![r.p_partkey, string_to_u64(&r.p_type)])
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
                        scale_by_1000(r.l_extendedprice),
                        scale_by_1000(r.l_discount),
                    ]
                })
                .collect();
        }

        assert!(
            !region.is_empty()
                && !nation.is_empty()
                && !customer.is_empty()
                && !orders.is_empty()
                && !part.is_empty()
                && !supplier.is_empty()
                && !lineitem.is_empty(),
            "dataset files not found under {}",
            crate::paths::data_file("lineitem.tbl")
        );

        // Non-vacuity of the third direction: the slice has to contain a dangling
        // tuple, otherwise the all-clean partition really is the reduced instance
        // and Pairwise Consistency would accept it for a good reason. A lineitem
        // row whose l_partkey passes no part predicate is exactly such a tuple on
        // the lineitem -- part edge, and it is what the first `pw: ` lookup sees.
        let const_part_type_hash = string_to_u64("PROMO BRUSHED COPPER");
        let kept_partkeys: std::collections::HashSet<u64> = part
            .iter()
            .filter(|r| r[1] == const_part_type_hash)
            .map(|r| r[0])
            .collect();
        let dangling = lineitem
            .iter()
            .filter(|r| !kept_partkeys.contains(&r[1]))
            .count();
        assert!(
            dangling > 0,
            "the slice has no dangling lineitem tuple, so the third direction \
             would pass vacuously; widen it"
        );

        let circuit = MyCircuit::<Fp> {
            region,
            nation,
            customer,
            orders,
            part,
            supplier,
            lineitem,
            cond_nation_hash: string_to_u64("EGYPT"),
            const_region_name_hash: string_to_u64("MIDDLE EAST"),
            const_part_type_hash,
            _marker: PhantomData,
        };

        let t0 = Instant::now();
        let prover = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        prover.assert_satisfied();
        println!("positive direction: {:?}", t0.elapsed());
        drop(prover);

        // Negative direction: the same witness with one joinable lineitem tuple
        // hidden in the residual side, and the neighbours re-reduced around it so
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

        // Third direction: the escape condition (3) closes. Every real tuple is
        // declared clean and the residual side is empty, so Conservation holds
        // and both channels of condition (4) compute the same number on every
        // row, which makes sum_cln == sum_all pass for free. The partition is
        // nevertheless not the reduced instance, and the dangling tuples asserted
        // above must now be caught by a `pw: ` lookup.
        super::MARK_ALL_CLEAN.store(true, Ordering::Relaxed);
        let all_clean = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        let verdict = all_clean.verify();
        super::MARK_ALL_CLEAN.store(false, Ordering::Relaxed);

        let failures = verdict.expect_err("condition (3) accepted the all-clean partition");
        let pw: Vec<String> = failures
            .iter()
            .filter(|f| matches!(f, VerifyFailure::Lookup { name, .. } if name.starts_with("pw: ")))
            .map(|f| format!("{:?}", f))
            .collect();
        assert!(
            !pw.is_empty(),
            "the circuit rejected the all-clean partition, but not through a \
             Pairwise Consistency lookup"
        );
        println!(
            "all-clean partition: {} failures, {} of them Pairwise Consistency, first: {}",
            failures.len(),
            pw.len(),
            pw[0]
        );
    }
}
