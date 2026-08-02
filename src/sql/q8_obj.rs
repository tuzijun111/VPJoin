use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};
use crate::circuits::card_preserve::{
    assign_cp_agg, assign_cp_join, assign_cp_root, build_cp_stage, configure_cp_agg,
    configure_cp_join, configure_cp_root, wire_cp_edge, CpAggConfig, CpJoinConfig, CpRootConfig,
};
use crate::circuits::conserve_idx::{
    assign_conserve, assign_row_index, configure_conserve, configure_row_index, ConserveConfig,
    RowIndexConfig,
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

/// Number of committed input columns, i.e. the width of
/// `bench_queries::TpchInput::columns()` for q8:
/// region 2 + nation 3 + customer 2 + orders 3 + part 2 + supplier 2 +
/// lineitem 5. `q8_bound::NC` is checked against it.
pub const NUM_COMMITTED: usize = 19;
// ------------------------------------------------

/// Test hook, off in every benchmark path: when set, the prover moves one
/// joinable lineitem tuple to the residual side and re-reduces the neighbours
/// around it, so the partition still passes Conservation, Non-Membership and
/// Pairwise Consistency and only condition (4) can catch it. This is exactly
/// the cheat a prefix-only or residual-side-only argument misses, so the
/// negative test in this module is what shows the Cardinality Preservation
/// Check is not vacuous.
thread_local! {
    static HIDE_ONE_CLEAN_TUPLE_TL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// THREAD-LOCAL, not a process-wide flag: `cargo test` runs modules in parallel
/// threads while circuit synthesis is single-threaded, so a global would corrupt
/// every other circuit being assigned at that moment. As an `AtomicBool` this
/// made witness-binding tests pass in isolation and fail in the full suite.
pub fn set_hide_one_clean_tuple(on: bool) {
    HIDE_ONE_CLEAN_TUPLE_TL.with(|c| c.set(on));
}

fn hide_one_clean_tuple() -> bool {
    HIDE_ONE_CLEAN_TUPLE_TL.with(|c| c.get())
}

/// Test hook, off in every benchmark path: when set, the prover skips the
/// semijoin reduction entirely and declares every real tuple clean, so the
/// residual section of every relation stays empty. Conservation still holds and
/// both channels of condition (4) then agree on every row, so `sum_cln ==
/// sum_all` passes for free. This is exactly the escape that Pairwise
/// Consistency has to close, and the third direction of the test in this module
/// is what shows it does.
thread_local! {
    static MARK_ALL_CLEAN_TL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// THREAD-LOCAL, not a process-wide flag: `cargo test` runs modules in parallel
/// threads while circuit synthesis is single-threaded, so a global would corrupt
/// every other circuit being assigned at that moment. As an `AtomicBool` this
/// made witness-binding tests pass in isolation and fail in the full suite.
pub fn set_mark_all_clean(on: bool) {
    MARK_ALL_CLEAN_TL.with(|c| c.set(on));
}

fn mark_all_clean() -> bool {
    MARK_ALL_CLEAN_TL.with(|c| c.get())
}

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
    // the four above are constant down every row they are read on
    q_param_const: Selector,

    // ---------- region membership proof for target_rkey ----------
    q_tbl_region: Selector,
    q_lkp_target_region: Selector,

    // ---------- clean indicator per base row ----------
    // one per relation of the join tree, in the order
    // [region, nation as n1, nation as n2, customer, orders, part, supplier,
    //  lineitem]
    cflag: Vec<Column<Advice>>, // 8

    // ---------------- (1) Conservation Check ----------------
    // R^_i == R^_i^c U+ R^_i^r over the INDEXED relation, one permutation per
    // node of the join tree, in the same order as `cflag`
    row_idx: RowIndexConfig,
    cons: Vec<ConserveConfig>, // 8

    // one complex selector per node of the join tree over its committed rows,
    // in the same order as `cflag`. It gates both sides of every lookup and the
    // Selector Check gates; a simple selector may appear on neither side of a
    // lookup, which is why it is complex. Its extent is |R_i|, a public size.
    q_row: Vec<Selector>, // 8

    // ---------- part type filter (PROMO BRUSHED COPPER) ----------
    q_part_pred: Selector,
    p_keep: Column<Advice>,
    iz_part_type: IsZeroConfig<F>,
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

    // ---------- join lookups, over the committed relations ----------
    q_lkp_orders: Selector,   // (l_orderkey, l_year) in the selected orders
    q_lkp_supplier: Selector, // (l_suppkey, l_s_nationkey) in the selected suppliers
    q_tbl_supplier: Selector,
    q_lkp_nation2: Selector, // (l_s_nationkey, l_nation_hash) in the selected nations
    // attached on the committed lineitem rows
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

    // share constraint on res_sorted: share * den = num, plus share = 0 when den = 0
    q_share: Selector,
    iz_res_den: IsZeroConfig<F>,

    // ---------------- region's in-relation predicate ----------------
    // r_name == MIDDLE EAST, because the input channel of (3) has to apply it
    q_region_pred: Selector,
    r_keep: Column<Advice>,
    iz_r_keep: IsZeroConfig<F>,

    // ---------------- Cardinality Preservation Check (condition (4)) ----------------
    // shifted copies of the two keys that can be 0
    q_nation_cp: Selector,
    q_cust_cp: Selector,
    q_supp_cp: Selector,
    nk_shift: Column<Advice>, // n_nationkey + SHIFT_ID, child key of n1 and n2
    n_rk_shift: Column<Advice>, // n_regionkey + SHIFT_ID, parent key of n1 -> region
    rk_shift: Column<Advice>, // r_regionkey + SHIFT_ID, child key of region
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

        // Every one of these four is free advice read at Rotation::cur() on many
        // rows, and nothing tied any of them across rows:
        //
        //   cond_nation  inside iz_is_cond on every all_nations_sorted row, so
        //                the ':1' nation was a per-row choice and the numerator
        //                of every year could be anything up to its denominator
        //                (set cond_nation[i] = nation[i] and cond_flag is 1
        //                everywhere, which makes share == 1 for every year);
        //   const_ptype  inside iz_part_type on every part row, so the p_type
        //                filter was a per-row choice and the circuit did not
        //                certify WHICH type was kept;
        //   const_rname  inside iz_r_keep on every region row and in the
        //                target_rkey membership lookup at row 0;
        //   target_rkey  inside iz_o_c_keep on every orders row, so the region
        //                filter could be satisfied by a DIFFERENT region on
        //                every order even though row 0 proved the pair
        //                (target_rkey, const_rname) is a real region row.
        //
        // One degree-1 gate comparing Rotation::cur() to Rotation::next(),
        // enabled on every row but the last one these columns are assigned on,
        // turns each per-row choice into a single per-proof choice. With
        // target_rkey constant the row-0 lookup then really does pin the region
        // filter to one region of the base table, and region names being unique
        // makes that region a function of const_rname.
        //
        // What this cannot do from inside this file is tie the three constants to
        // public inputs: expose_public below constrains only instance_test row 0
        // to the constant 1, so no query parameter and no result cell is a public
        // input and the verifier has nothing to compare the answer against.
        // Fixing that needs a wider instance vector and a change at every call
        // site, which are shared files.
        let q_param_const = meta.selector();
        meta.create_gate("query parameters are constant down the rows", |m| {
            let q = m.query_selector(q_param_const);
            let step = |m: &mut VirtualCells<'_, F>, c: Column<Advice>| {
                m.query_advice(c, Rotation::cur()) - m.query_advice(c, Rotation::next())
            };
            let d0 = step(m, cond_nation);
            let d1 = step(m, const_rname);
            let d2 = step(m, const_ptype);
            let d3 = step(m, target_rkey);
            vec![q.clone() * d0, q.clone() * d1, q.clone() * d2, q * d3]
        });

        // clean indicator per base row, one column per relation of the join
        // tree: [region, nation as n1, nation as n2, customer, orders, part,
        // supplier, lineitem]
        let cflag = (0..8).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for &c in cflag.iter() {
            meta.enable_equality(c);
        }

        // ---------- (1) Conservation Check ----------
        // One permutation argument per node of the join tree, between the
        // indexed relation R^_i and the concatenation of its two parts. The
        // indices are distinct, so R^_i is a set even though the relation is a
        // bag, and the single permutation rules out an occurrence being
        // fabricated, lost, duplicated or counted on both sides: no
        // Non-Membership Check. `nation` appears twice in the tree, as n1 and
        // as n2, so its rows are conserved twice, once per occurrence, each
        // against its own indicator.
        let row_idx = configure_row_index::<F>(meta);
        let cons: Vec<ConserveConfig> = vec![
            configure_conserve::<F>(meta, &row_idx, &region, cflag[0]),
            configure_conserve::<F>(meta, &row_idx, &nation, cflag[1]),
            configure_conserve::<F>(meta, &row_idx, &nation, cflag[2]),
            configure_conserve::<F>(meta, &row_idx, &customer, cflag[3]),
            configure_conserve::<F>(meta, &row_idx, &orders, cflag[4]),
            configure_conserve::<F>(meta, &row_idx, &part, cflag[5]),
            configure_conserve::<F>(meta, &row_idx, &supplier, cflag[6]),
            configure_conserve::<F>(meta, &row_idx, &lineitem, cflag[7]),
        ];

        // ---------- Selector Check, booleanity half ----------
        // The prover supplies one bit per committed row per node of the join
        // tree. Selection is by position, so no row can be fabricated, dropped
        // or placed on both sides and there is nothing for a Conservation or
        // Non-Membership Check to compare against.
        //
        // Booleanity used to come free: the Conservation shuffle matched each
        // input row against a partition row whose flag column was pinned to 1
        // or 0. With no partition it needs its own gate, and it is load-bearing,
        // because the clean channel of the propagation multiplies by the bit at
        // every node. The predicate half, c(t)(1 - b(t)) = 0, is written below
        // next to each of the three predicates that exist (region, orders,
        // part); the other five nodes have b == 1.
        let q_row = (0..8).map(|_| meta.complex_selector()).collect::<Vec<_>>();
        for (idx, &c) in cflag.iter().enumerate() {
            let q = q_row[idx];
            meta.create_gate("selector is a bit", move |m| {
                let q = m.query_selector(q);
                let c = m.query_advice(c, Rotation::cur());
                vec![q * c.clone() * (Expression::Constant(F::ONE) - c)]
            });
        }

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

        // the predicate half of the Selector Check on part
        {
            let q = q_row[5];
            let c = cflag[5];
            meta.create_gate("selected part row satisfies its predicate", move |m| {
                let q = m.query_selector(q);
                let c = m.query_advice(c, Rotation::cur());
                let b = m.query_advice(p_keep, Rotation::cur());
                vec![q * c * (Expression::Constant(F::ONE) - b)]
            });
        }

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
        //
        // Dead column, kept only so the layout matches q8_obj.rs: c_keep is read
        // by nothing outside its own gate below. It feeds no filter, no shuffle
        // and neither channel of condition (10) -- the customer row of the check
        // takes its input multiplicity from fetched_n1, not from here -- so the
        // region filter is in practice applied on orders alone, through o_c_keep.
        // That is sound, because the orders -> customer -> nation1 -> region chain
        // of Pairwise Consistency is what ties a clean order to a clean region
        // row; it is just not a second place the filter is enforced.
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

        // the predicate half of the Selector Check on orders. `q8_obj.rs` got
        // this structurally, from the link gate that padded the indicator away
        // wherever `o_keep` was 0; here it is one constraint, and it is what
        // makes the input channel of (3) count the join of the
        // predicate-filtered relations.
        {
            let q = q_row[4];
            let c = cflag[4];
            meta.create_gate("selected orders row satisfies its predicate", move |m| {
                let q = m.query_selector(q);
                let c = m.query_advice(c, Rotation::cur());
                let b = m.query_advice(o_keep, Rotation::cur());
                vec![q * c * (Expression::Constant(F::ONE) - b)]
            });
        }

        // ---------- join lookups, over the committed relations ----------
        // Every one runs between committed columns with both sides gated by the
        // selector: a deselected row and a row past the relation both
        // contribute the all-zero tuple, which is what a gated-off input row
        // reads, and the shift by one keeps that dummy away from any real
        // tuple. `q8_obj.rs` read them off the `*_join_pad` groups, which
        // needed a Conservation Check each to mean anything.
        //
        // The part membership lookup `q8_obj.rs` runs here is gone: with both
        // sides on the committed relations it is character for character the
        // `pw: lineitem^c partkey in part^c` lookup below.
        let q_lkp_orders = meta.complex_selector();
        let l_year = meta.advice_column();
        meta.enable_equality(l_year);

        // (l_orderkey, l_year) in the selected orders
        meta.lookup_any("attach year from filtered orders", |m| {
            let one = Expression::Constant(F::ONE);
            let q_in = m.query_selector(q_lkp_orders) * m.query_advice(cflag[7], Rotation::cur());
            let q_t = m.query_selector(q_row[4]) * m.query_advice(cflag[4], Rotation::cur());
            vec![
                (
                    q_in.clone() * (m.query_advice(lineitem[0], Rotation::cur()) + one.clone()),
                    q_t.clone() * (m.query_advice(orders[0], Rotation::cur()) + one.clone()),
                ),
                (
                    q_in * (m.query_advice(l_year, Rotation::cur()) + one.clone()),
                    q_t * (m.query_advice(orders[2], Rotation::cur()) + one),
                ),
            ]
        });

        let q_lkp_supplier = meta.complex_selector();
        let q_tbl_supplier = meta.complex_selector();
        let l_s_nationkey = meta.advice_column();
        meta.enable_equality(l_s_nationkey);

        // (l_suppkey, l_s_nationkey) in the selected suppliers
        meta.lookup_any("attach supplier nationkey", |m| {
            let one = Expression::Constant(F::ONE);
            let q_in = m.query_selector(q_lkp_supplier) * m.query_advice(cflag[7], Rotation::cur());
            let q_t = m.query_selector(q_row[6]) * m.query_advice(cflag[6], Rotation::cur());
            vec![
                (
                    q_in.clone() * (m.query_advice(lineitem[2], Rotation::cur()) + one.clone()),
                    q_t.clone() * (m.query_advice(supplier[0], Rotation::cur()) + one.clone()),
                ),
                (
                    q_in * (m.query_advice(l_s_nationkey, Rotation::cur()) + one.clone()),
                    q_t * (m.query_advice(supplier[1], Rotation::cur()) + one),
                ),
            ]
        });

        let q_lkp_nation2 = meta.complex_selector();
        let l_nation_hash = meta.advice_column();
        meta.enable_equality(l_nation_hash);

        // (l_s_nationkey, l_nation_hash) in the selected nations, read in the
        // n2 role, which is the occurrence supplier hangs under
        meta.lookup_any("attach nation namehash for supplier nation", |m| {
            let one = Expression::Constant(F::ONE);
            let q_in = m.query_selector(q_lkp_nation2) * m.query_advice(cflag[7], Rotation::cur());
            let q_t = m.query_selector(q_row[2]) * m.query_advice(cflag[2], Rotation::cur());
            vec![
                (
                    q_in.clone() * (m.query_advice(l_s_nationkey, Rotation::cur()) + one.clone()),
                    q_t.clone() * (m.query_advice(nation[0], Rotation::cur()) + one.clone()),
                ),
                (
                    q_in * (m.query_advice(l_nation_hash, Rotation::cur()) + one.clone()),
                    q_t * (m.query_advice(nation[2], Rotation::cur()) + one),
                ),
            ]
        });

        // ---------- volume gate ----------
        let q_volume = meta.selector();
        let volume = meta.advice_column();
        meta.enable_equality(volume);

        meta.create_gate("volume = ext*(SCALE - disc)", |m| {
            let q = m.query_selector(q_volume);
            let ext = m.query_advice(lineitem[3], Rotation::cur());
            let disc = m.query_advice(lineitem[4], Rotation::cur());
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

        // The all_nations triple of a SELECTED row is its attached
        // (year, volume, nation); of a deselected row it is the canonical PAD
        // triple. That one masked gate does the work `q8_obj.rs` split between
        // "all_nations = (year, volume, nation)" over the join prefix and
        // "all_nations rows past the clean section are PAD" over the
        // complement, and it no longer depends on the clean rows being laid out
        // first. Without a pin on the deselected rows the shuffle would carry
        // free advice into the per-year running sums, where an injected triple
        // with a fresh year forms its own group and one with an existing year
        // and nation ':1' moves that year's numerator.
        //
        // The three attach columns and `volume` stay free on a deselected row,
        // which is safe for the same reason as before: nothing else reads them,
        // and the triple the shuffle carries is pinned.
        meta.create_gate("all_nations row: selected? attached : PAD", |m| {
            let q = m.query_selector(q_volume);
            let c = m.query_advice(cflag[7], Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let drop = one - c.clone();
            vec![
                q.clone()
                    * (m.query_advice(all_nations[0], Rotation::cur())
                        - (c.clone() * m.query_advice(l_year, Rotation::cur())
                            + drop.clone() * Expression::Constant(F::from(PAD_AN_YEAR)))),
                q.clone()
                    * (m.query_advice(all_nations[1], Rotation::cur())
                        - (c.clone() * m.query_advice(volume, Rotation::cur())
                            + drop.clone() * Expression::Constant(F::from(PAD_AN_VOL)))),
                q * (m.query_advice(all_nations[2], Rotation::cur())
                    - (c * m.query_advice(l_nation_hash, Rotation::cur())
                        + drop * Expression::Constant(F::from(PAD_AN_NAT)))),
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

        // The gate above degenerates on a zero-denominator row: with den = 0 it
        // reads -num = 0, which pins num and says nothing at all about share, so
        // that answer cell was uncertified advice. Two rows reach the case: the
        // PAD group at the end of the sorted view, whose volumes are all 0, and
        // any real year whose volumes cancel. Force share to 0 there. Same shape
        // and same degree as the IsZero gates the file already carries, and the
        // honest witness already writes 0 whenever den is 0.
        let aux_res_den = meta.advice_column();
        meta.enable_equality(aux_res_den);
        let iz_res_den = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_share),
            |m| m.query_advice(res_sorted[2], Rotation::cur()),
            aux_res_den,
        );
        {
            let share_col = res_sorted[3];
            let iz = iz_res_den.clone();
            meta.create_gate("share = 0 when den = 0", move |m| {
                let q = m.query_selector(q_share);
                vec![q * iz.expr() * m.query_advice(share_col, Rotation::cur())]
            });
        }

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

        // the predicate half of the Selector Check on region
        {
            let q = q_row[0];
            let c = cflag[0];
            meta.create_gate("selected region row satisfies its predicate", move |m| {
                let q = m.query_selector(q);
                let c = m.query_advice(c, Rotation::cur());
                let b = m.query_advice(r_keep, Rotation::cur());
                vec![q * c * (Expression::Constant(F::ONE) - b)]
            });
        }

        // ============ (2) Pairwise Consistency ============
        // pi_K_ij(R_i^c) == pi_K_ij(R_j^c) on every edge of the join tree, as
        // two mutual Membership Checks. Both sides read the COMMITTED key
        // column gated by that node's selector bit:
        //
        //     q_row * c(t) * (t[K] + SHIFT_ID),
        //
        // so a deselected row and a row past the relation both read 0, 0 is in
        // every table, and the containment is over the selected keys only.
        // n_nationkey and r_regionkey start at 0 in TPC-H, which is why the
        // shift is there: without it a genuine key 0 would be accepted whether
        // or not it is in R_j^c.
        //
        // `q8_obj.rs` read these off the clean prefix of a materialized
        // partition group, which meant pi_K(R^c) only because a Conservation
        // Check tied the group to the base rows, and which baked |R_i^c| into
        // the fixed columns. Here the gating factor is the committed bit, the
        // same one the propagation reads on the same rows, and the selectors
        // mark nothing but each relation's real rows.
        //
        // nation appears twice in the tree, as n1 under region and as n2 under
        // supplier, so the two occurrences read the SAME committed columns
        // through their own bits, cflag[1] and cflag[2].
        let mut pw_edge = |name: &'static str,
                           q_in: Selector,
                           c_in: Column<Advice>,
                           key_col: Column<Advice>,
                           q_t: Selector,
                           c_t: Column<Advice>,
                           tbl_col: Column<Advice>| {
            meta.lookup_any(name, move |m| {
                let shift = Expression::Constant(F::from(SHIFT_ID));
                let q_in = m.query_selector(q_in) * m.query_advice(c_in, Rotation::cur());
                let q_t = m.query_selector(q_t) * m.query_advice(c_t, Rotation::cur());
                vec![(
                    q_in * (m.query_advice(key_col, Rotation::cur()) + shift.clone()),
                    q_t * (m.query_advice(tbl_col, Rotation::cur()) + shift),
                )]
            });
        };

        // edge lineitem -- part on l_partkey = p_partkey
        pw_edge(
            "pw: lineitem^c partkey in part^c",
            q_row[7],
            cflag[7],
            lineitem[1],
            q_row[5],
            cflag[5],
            part[0],
        );
        pw_edge(
            "pw: part^c partkey in lineitem^c",
            q_row[5],
            cflag[5],
            part[0],
            q_row[7],
            cflag[7],
            lineitem[1],
        );

        // edge lineitem -- orders on l_orderkey = o_orderkey
        pw_edge(
            "pw: lineitem^c orderkey in orders^c",
            q_row[7],
            cflag[7],
            lineitem[0],
            q_row[4],
            cflag[4],
            orders[0],
        );
        pw_edge(
            "pw: orders^c orderkey in lineitem^c",
            q_row[4],
            cflag[4],
            orders[0],
            q_row[7],
            cflag[7],
            lineitem[0],
        );

        // edge lineitem -- supplier on l_suppkey = s_suppkey
        pw_edge(
            "pw: lineitem^c suppkey in supplier^c",
            q_row[7],
            cflag[7],
            lineitem[2],
            q_row[6],
            cflag[6],
            supplier[0],
        );
        pw_edge(
            "pw: supplier^c suppkey in lineitem^c",
            q_row[6],
            cflag[6],
            supplier[0],
            q_row[7],
            cflag[7],
            lineitem[2],
        );

        // edge supplier -- nation as n2 on s_nationkey = n_nationkey
        pw_edge(
            "pw: supplier^c nationkey in nation2^c",
            q_row[6],
            cflag[6],
            supplier[1],
            q_row[2],
            cflag[2],
            nation[0],
        );
        pw_edge(
            "pw: nation2^c nationkey in supplier^c",
            q_row[2],
            cflag[2],
            nation[0],
            q_row[6],
            cflag[6],
            supplier[1],
        );

        // edge orders -- customer on o_custkey = c_custkey
        pw_edge(
            "pw: orders^c custkey in customer^c",
            q_row[4],
            cflag[4],
            orders[1],
            q_row[3],
            cflag[3],
            customer[0],
        );
        pw_edge(
            "pw: customer^c custkey in orders^c",
            q_row[3],
            cflag[3],
            customer[0],
            q_row[4],
            cflag[4],
            orders[1],
        );

        // edge customer -- nation as n1 on c_nationkey = n_nationkey
        pw_edge(
            "pw: customer^c nationkey in nation1^c",
            q_row[3],
            cflag[3],
            customer[1],
            q_row[1],
            cflag[1],
            nation[0],
        );
        pw_edge(
            "pw: nation1^c nationkey in customer^c",
            q_row[1],
            cflag[1],
            nation[0],
            q_row[3],
            cflag[3],
            customer[1],
        );

        // edge nation as n1 -- region on n_regionkey = r_regionkey
        pw_edge(
            "pw: nation1^c regionkey in region^c",
            q_row[1],
            cflag[1],
            nation[1],
            q_row[0],
            cflag[0],
            region[0],
        );
        pw_edge(
            "pw: region^c regionkey in nation1^c",
            q_row[0],
            cflag[0],
            region[0],
            q_row[1],
            cflag[1],
            nation[1],
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
        for &c in [
            nk_shift, n_rk_shift, rk_shift, c_nk_shift, s_nk_shift, cp_one,
        ]
        .iter()
        {
            meta.enable_equality(c);
        }

        meta.create_gate(
            "cp: shifted nation keys and the nation leaf predicate",
            |m| {
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
            },
        );
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
        let cp_agg_region =
            configure_cp_agg::<F, NUM_BYTES>(meta, cp_u8, rk_shift, r_keep, cflag[0], MAX_SENTINEL);
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
            let cln = cflag[4];
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
        let cp_agg_n2 =
            configure_cp_agg::<F, NUM_BYTES>(meta, cp_u8, nk_shift, cp_one, cflag[2], MAX_SENTINEL);
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
        let cp_agg_part =
            configure_cp_agg::<F, NUM_BYTES>(meta, cp_u8, part[0], p_keep, cflag[5], MAX_SENTINEL);
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
            q_param_const,

            q_tbl_region,
            q_lkp_target_region,

            row_idx,
            cons,
            q_row,
            q_part_pred,
            p_keep,
            iz_part_type,

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

            q_lkp_orders,
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
            iz_res_den,
            q_emit,
            q_emit_last,

            cflag,

            q_region_pred,
            r_keep,
            iz_r_keep,

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

    /// Assign the query, and hand back the cells of the committed input columns
    /// alongside the public output.
    ///
    /// `in_cells[j][i]` is the cell this circuit witnesses for committed column
    /// `j` at row `i`, in the order `bench_queries::run_tpch` transposes the q8
    /// tables: region `[r_regionkey, hash(r_name)]`, nation `[n_nationkey,
    /// n_regionkey, hash(n_name)]`, customer `[c_custkey, c_nationkey]`, orders
    /// `[o_orderkey, o_custkey, year(o_orderdate)]`, part `[p_partkey,
    /// hash(p_type)]`, supplier `[s_suppkey, s_nationkey]`, lineitem
    /// `[l_orderkey, l_partkey, l_suppkey, 1000*l_extendedprice,
    /// 1000*l_discount]`. Every cell holds that value verbatim: the shifted and
    /// derived copies this circuit also needs (`rk_shift`, `nk_shift`,
    /// `n_rk_shift`, `c_nk_shift`, `s_nk_shift`, `volume`, ...) live in columns
    /// of their own and are never written back over a base column.
    ///
    /// Each column is zero-extended to `max_i |R_i|`, matching the extension
    /// `inline_bind::assign_bind_cells` and `inline_bind::bind_instance` apply,
    /// so `tie_columns` can pair them row for row.
    #[allow(clippy::too_many_arguments)]
    pub fn assign_with_input_cells(
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
    ) -> Result<(AssignedCell<F, F>, Vec<Vec<AssignedCell<F, F>>>), Error> {
        self.assign_inner(
            layouter,
            region_t,
            nation_t,
            customer_t,
            orders_t,
            part_t,
            supplier_t,
            lineitem_t,
            cond_nation_hash,
            const_region_name_hash,
            const_part_type_hash,
            true,
        )
    }

    /// `zero_extend` is the only difference between the two entry points: it
    /// writes the explicit zeros the tie needs on the rows past a short table.
    /// Off for the baseline, so the measured baseline keeps exactly the
    /// assignments it had before the binding existed; the extra writes belong to
    /// the binding layer and are charged to it.
    #[allow(clippy::too_many_arguments)]
    fn assign_inner(
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
        zero_extend: bool,
    ) -> Result<(AssignedCell<F, F>, Vec<Vec<AssignedCell<F, F>>>), Error> {
        // load lt tables
        let lt_year_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_year_cur_next.clone());
        lt_year_chip.load(layouter)?;
        let lt_res_year_chip =
            LtChip::<F, NUM_BYTES>::construct(self.config.lt_res_year_cur_next.clone());
        lt_res_year_chip.load(layouter)?;

        // Every Lt chip of the Cardinality Preservation Check shares one u8
        // fixed column, so a single load covers the whole check.
        LtChip::<F, NUM_BYTES>::construct(self.config.cp_agg_region.lt_key_cur_next)
            .load(layouter)?;

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
        let tamper = hide_one_clean_tuple();
        let all_clean = mark_all_clean();
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
        let filt_side =
            |rows: &[Vec<u64>], keep: &[u64], cln: &[u64], pad: &[u64]| -> Vec<Vec<u64>> {
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

        // The Selector Check and Pairwise Consistency consume no host-side
        // witness beyond the bits themselves: the fourteen lookups run between
        // the committed key columns, gated by those bits, and the propagation
        // reads the same columns on the same rows. `q8_obj.rs` built eight
        // partition groups, eight filtered views and their section boundaries
        // here; none of that exists any more.

        // maps for quick attach:
        let orders_year: HashMap<u64, u64> = orders_t.iter().map(|r| (r[0], r[2])).collect();

        // all_nations, on the committed lineitem rows: a selected row carries
        // its attached (year, volume, nation), a deselected row the canonical
        // PAD triple. The multiset is the same one `q8_obj.rs` produced from its
        // clean prefix plus pad section, so the sorted view and every per-year
        // sum below are unchanged; only the row a triple sits on has moved.
        let mut all_nations_u64: Vec<[u64; 3]> =
            vec![[PAD_AN_YEAR, PAD_AN_VOL, PAD_AN_NAT]; lineitem_t.len()];

        for i in 0..lineitem_t.len() {
            if cln_l[i] != 1 {
                continue;
            }
            let okey = lineitem_t[i][0];
            let skey = lineitem_t[i][2];
            let ext = lineitem_t[i][3];
            let disc = lineitem_t[i][4];

            // Under the MARK_ALL_CLEAN hook every lineitem row is selected, so
            // these attaches can miss; on the honest witness they never do,
            // because a selected row is joinable by construction.
            let year = *orders_year.get(&okey).unwrap_or(&PAD_YEAR);
            let nkey = *supp_nat.get(&skey).unwrap_or(&PAD_YEAR);
            let nname = *nat_name.get(&nkey).unwrap_or(&PAD_AN_NAT);

            // Two latent host-side limits, neither a soundness hole. SCALE-disc
            // underflows if a discount ever exceeds SCALE, and the product is
            // reduced here but computed in the field by the gate. Both only ever
            // break the HONEST prover: a witness the host mangles simply fails
            // the volume gate. Nothing range-checks volume, run_num or run_den
            // either, so the per-year sums are field sums; an honest TPC-H
            // instance never comes near the modulus, and range-checking three
            // columns would cost three Lt tables for no soundness gain.
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

        // The height every base table is zero-extended to, both here and in the
        // binding layer. Hoisted out of the region because the pad-out of the
        // committed columns runs before the parameter rows that first used it.
        let max_len = region_t
            .len()
            .max(nation_t.len())
            .max(customer_t.len())
            .max(orders_t.len())
            .max(part_t.len())
            .max(supplier_t.len())
            .max(lineitem_t.len())
            .max(1);

        // ---------- assignment ----------
        let (out_cell, in_cells) = layouter.assign_region(
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
                let iz_res_den_chip = IsZeroChip::construct(self.config.iz_res_den.clone());
                let iz_rk_chip = IsZeroChip::construct(self.config.iz_r_keep.clone());

                // The cells of the committed columns, in the order
                // `bench_queries::run_tpch` transposes the tables for q8:
                // region 2, nation 3, customer 2, orders 3, part 2, supplier 2,
                // lineitem 5. Every one of them holds the committed value
                // VERBATIM at the committed row index, which is what lets
                // q8_bound.rs copy-constrain the binding to THIS witness.
                let mut in_cells: Vec<Vec<AssignedCell<F, F>>> = vec![Vec::new(); NUM_COMMITTED];

                // ---- base tables assignment ----
                for i in 0..region_t.len() {
                    self.config.q_tbl_region.enable(&mut region, i)?;
                    in_cells[0].push(region.assign_advice(
                        || "regionkey",
                        self.config.region[0],
                        i,
                        || Value::known(F::from(region_t[i][0])),
                    )?);
                    in_cells[1].push(region.assign_advice(
                        || "rname",
                        self.config.region[1],
                        i,
                        || Value::known(F::from(region_t[i][1])),
                    )?);

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
                    self.config.q_row[0].enable(&mut region, i)?;
                }

                for i in 0..nation_t.len() {
                    self.config.q_tbl_nation.enable(&mut region, i)?;
                    in_cells[2].push(region.assign_advice(
                        || "nkey",
                        self.config.nation[0],
                        i,
                        || Value::known(F::from(nation_t[i][0])),
                    )?);
                    in_cells[3].push(region.assign_advice(
                        || "n_rkey",
                        self.config.nation[1],
                        i,
                        || Value::known(F::from(nation_t[i][1])),
                    )?);
                    in_cells[4].push(region.assign_advice(
                        || "nname",
                        self.config.nation[2],
                        i,
                        || Value::known(F::from(nation_t[i][2])),
                    )?);

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
                    self.config.q_row[1].enable(&mut region, i)?;
                    self.config.q_row[2].enable(&mut region, i)?;
                }

                for i in 0..customer_t.len() {
                    self.config.q_tbl_customer.enable(&mut region, i)?;
                    in_cells[5].push(region.assign_advice(
                        || "ckey",
                        self.config.customer[0],
                        i,
                        || Value::known(F::from(customer_t[i][0])),
                    )?);
                    in_cells[6].push(region.assign_advice(
                        || "c_nkey",
                        self.config.customer[1],
                        i,
                        || Value::known(F::from(customer_t[i][1])),
                    )?);

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
                    self.config.q_row[3].enable(&mut region, i)?;
                }

                for i in 0..orders_t.len() {
                    in_cells[7].push(region.assign_advice(
                        || "okey",
                        self.config.orders[0],
                        i,
                        || Value::known(F::from(orders_t[i][0])),
                    )?);
                    in_cells[8].push(region.assign_advice(
                        || "o_cust",
                        self.config.orders[1],
                        i,
                        || Value::known(F::from(orders_t[i][1])),
                    )?);
                    in_cells[9].push(region.assign_advice(
                        || "oyear",
                        self.config.orders[2],
                        i,
                        || Value::known(F::from(orders_t[i][2])),
                    )?);
                }

                for i in 0..part_t.len() {
                    in_cells[10].push(region.assign_advice(
                        || "pkey",
                        self.config.part[0],
                        i,
                        || Value::known(F::from(part_t[i][0])),
                    )?);
                    in_cells[11].push(region.assign_advice(
                        || "ptype",
                        self.config.part[1],
                        i,
                        || Value::known(F::from(part_t[i][1])),
                    )?);
                }

                for i in 0..supplier_t.len() {
                    self.config.q_tbl_supplier.enable(&mut region, i)?;
                    in_cells[12].push(region.assign_advice(
                        || "skey",
                        self.config.supplier[0],
                        i,
                        || Value::known(F::from(supplier_t[i][0])),
                    )?);
                    in_cells[13].push(region.assign_advice(
                        || "s_nkey",
                        self.config.supplier[1],
                        i,
                        || Value::known(F::from(supplier_t[i][1])),
                    )?);

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
                    self.config.q_row[6].enable(&mut region, i)?;
                }

                for i in 0..lineitem_t.len() {
                    for j in 0..5 {
                        in_cells[14 + j].push(region.assign_advice(
                            || "lineitem",
                            self.config.lineitem[j],
                            i,
                            || Value::known(F::from(lineitem_t[i][j])),
                        )?);
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

                // ---- the committed columns, zero-extended to a common height ----
                // `inline_bind::assign_bind_cells` lays the binding out over
                // max_i |R_i| rows and zero-extends every shorter column, and
                // `bind_instance` evaluates the same zero-extension, so the tie
                // needs a witness cell opposite each of those rows. These rows
                // carry no selector of this circuit -- every gate, lookup and
                // shuffle over a base table is gated on that table's own row
                // range -- and an unassigned advice cell is already zero to
                // both the prover and MockProver (`CellValue::Unassigned =>
                // Value::Real(ZERO)`), so writing the zero explicitly changes
                // no constraint. It only gives the copy constraint a cell.
                if zero_extend {
                    let mut next_slot = 0usize;
                    for (cols, len) in [
                        (&self.config.region, region_t.len()),
                        (&self.config.nation, nation_t.len()),
                        (&self.config.customer, customer_t.len()),
                        (&self.config.orders, orders_t.len()),
                        (&self.config.part, part_t.len()),
                        (&self.config.supplier, supplier_t.len()),
                        (&self.config.lineitem, lineitem_t.len()),
                    ] {
                        for &col in cols.iter() {
                            let slot = next_slot;
                            next_slot += 1;
                            // rows 0..len were assigned verbatim above
                            for i in len..max_len {
                                in_cells[slot].push(region.assign_advice(
                                    || "committed column zero-extension",
                                    col,
                                    i,
                                    || Value::known(F::ZERO),
                                )?);
                            }
                        }
                    }
                    debug_assert_eq!(next_slot, NUM_COMMITTED);
                }

                // ---- parameters repeated on row 0..max_len ----
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

                // Force the four parameter columns to be constant over exactly the
                // rows they are assigned on, which is a superset of every row any
                // gate reads them at. Enabled on 0..max_len-1 so that
                // Rotation::next() of the last enabled row is still inside the
                // region.
                for i in 0..max_len.saturating_sub(1) {
                    self.config.q_param_const.enable(&mut region, i)?;
                }

                // prove target_rkey exists (do it once at row 0)
                self.config.q_lkp_target_region.enable(&mut region, 0)?;

                // ---- part predicate + p_keep + p_filt_pad + p_join_pad + perm selectors ----
                for i in 0..part_t.len() {
                    self.config.q_part_pred.enable(&mut region, i)?;
                    self.config.q_row[5].enable(&mut region, i)?;
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
                    self.config.q_row[4].enable(&mut region, i)?;

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
                }

                // ---- the lineitem selector bit, on the committed rows ----
                for i in 0..lineitem_t.len() {
                    self.config.q_row[7].enable(&mut region, i)?;
                    region.assign_advice(
                        || "cflag lineitem",
                        self.config.cflag[7],
                        i,
                        || Value::known(F::from(cln_l[i])),
                    )?;
                }

                // ---- (1) Conservation Check ----
                assign_row_index(
                    &mut region,
                    &self.config.row_idx,
                    [
                        region_t.len(),
                        nation_t.len(),
                        customer_t.len(),
                        orders_t.len(),
                        part_t.len(),
                        supplier_t.len(),
                        lineitem_t.len(),
                    ]
                    .into_iter()
                    .max()
                    .unwrap_or(0),
                )?;
                for (idx, (rows, flags)) in [
                    (&region_t, &cln_r),
                    (&nation_t, &cln_n1),
                    (&nation_t, &cln_n2),
                    (&customer_t, &cln_c),
                    (&orders_t, &cln_o),
                    (&part_t, &cln_p),
                    (&supplier_t, &cln_s),
                    (&lineitem_t, &cln_l),
                ]
                .into_iter()
                .enumerate()
                {
                    assign_conserve(&mut region, &self.config.cons[idx], rows, flags)?;
                }

                // Selector Check and Pairwise Consistency need no witness of
                // their own beyond the bits assigned with the base rows above:
                // the fourteen lookups run between the committed key columns,
                // gated by those bits, and the fixed selectors mark nothing but
                // each relation's real rows.

                // The join lookups and the volume gate cover every committed
                // lineitem row now, not a clean prefix: a deselected row reads
                // the all-zero tuple on both sides of each lookup and its
                // all_nations triple is masked to PAD.
                for i in 0..lineitem_t.len() {
                    self.config.q_lkp_orders.enable(&mut region, i)?;
                    self.config.q_lkp_supplier.enable(&mut region, i)?;
                    self.config.q_lkp_nation2.enable(&mut region, i)?;
                    self.config.q_volume.enable(&mut region, i)?;
                }

                // ---- attached fields, volume and all_nations, in place ----
                // A selected row carries its attached (year, nationkey, name),
                // its volume and the real triple; a deselected row carries
                // zeros in the attach columns, the volume its own gate then
                // forces, and the canonical PAD triple, which is what the
                // masked gate demands. `volume` and the attach columns stay
                // free on a deselected row as far as the gate is concerned, so
                // writing 0 there just satisfies the volume gate.
                for i in 0..lineitem_t.len() {
                    let selected = cln_l[i] == 1;
                    let (year, nkey, nname) = if selected {
                        let okey = lineitem_t[i][0];
                        let skey = lineitem_t[i][2];
                        let year = *orders_year.get(&okey).unwrap_or(&PAD_YEAR);
                        let nkey = *supp_nat.get(&skey).unwrap_or(&PAD_YEAR);
                        (year, nkey, *nat_name.get(&nkey).unwrap_or(&PAD_AN_NAT))
                    } else {
                        (0, 0, 0)
                    };
                    // the volume gate runs on every row, so it always has to hold
                    let vol =
                        (lineitem_t[i][3] as u128) * ((SCALE as u128) - (lineitem_t[i][4] as u128));
                    let an = if selected {
                        [year, vol as u64, nname]
                    } else {
                        [PAD_AN_YEAR, PAD_AN_VOL, PAD_AN_NAT]
                    };

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
                    region.assign_advice(
                        || "volume",
                        self.config.volume,
                        i,
                        || Value::known(F::from(vol as u64)),
                    )?;
                    for j in 0..3 {
                        region.assign_advice(
                            || "all_nations",
                            self.config.all_nations[j],
                            i,
                            || Value::known(F::from(an[j])),
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
                    // inverse witness of the denominator, so that "share = 0 when
                    // den = 0" can see which rows are the degenerate ones
                    iz_res_den_chip.assign(&mut region, i, Value::known(F::from(den)))?;
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
                    .map(|i| [region_t[i][0] + SHIFT_ID, r_keep_u64[i], cln_r[i]])
                    .collect();
                let cp_stage_r = build_cp_stage(&cp_rows_r, MAX_SENTINEL);
                assign_cp_agg(
                    &mut region,
                    &self.config.cp_agg_region,
                    &cp_rows_r,
                    &cp_stage_r,
                )?;

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
                assign_cp_agg(
                    &mut region,
                    &self.config.cp_agg_n1,
                    &cp_rows_n1,
                    &cp_stage_n1,
                )?;

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
                assign_cp_agg(
                    &mut region,
                    &self.config.cp_agg_cust,
                    &cp_rows_c,
                    &cp_stage_c,
                )?;

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
                            cln_o[i] * fetched_c[i].1,
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
                assign_cp_agg(
                    &mut region,
                    &self.config.cp_agg_ord,
                    &cp_rows_o,
                    &cp_stage_o,
                )?;

                // ---- nation as n2: a leaf, so its input channel is the constant 1 ----
                let cp_rows_n2: Vec<[u64; 3]> = (0..nation_t.len())
                    .map(|i| [nation_t[i][0] + SHIFT_ID, 1, cln_n2[i]])
                    .collect();
                let cp_stage_n2 = build_cp_stage(&cp_rows_n2, MAX_SENTINEL);
                assign_cp_agg(
                    &mut region,
                    &self.config.cp_agg_n2,
                    &cp_rows_n2,
                    &cp_stage_n2,
                )?;

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
                assign_cp_agg(
                    &mut region,
                    &self.config.cp_agg_supp,
                    &cp_rows_s,
                    &cp_stage_s,
                )?;

                // ---- part leaf: [key, pred, keep * c] ----
                let cp_rows_p: Vec<[u64; 3]> = (0..part_t.len())
                    .map(|i| [part_t[i][0], p_keep_u64[i], cln_p[i]])
                    .collect();
                let cp_stage_p = build_cp_stage(&cp_rows_p, MAX_SENTINEL);
                assign_cp_agg(
                    &mut region,
                    &self.config.cp_agg_part,
                    &cp_rows_p,
                    &cp_stage_p,
                )?;

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
                        cp_cln as usize,
                        cln_l.iter().filter(|&&f| f == 1).count(),
                        "the clean channel must count the selected lineitem rows"
                    );
                }

                // ---- public output (keep the same convention as your previous files) ----
                let out = region.assign_advice(
                    || "instance_test",
                    self.config.instance_test,
                    0,
                    || Value::known(F::from(1u64)),
                )?;
                Ok((out, in_cells))
            },
        )?;

        Ok((out_cell, in_cells))
    }

    /// The signature every other caller uses, unchanged. Only `q8_bound.rs`
    /// needs the input cells, and only because it has to copy-constrain the
    /// binding to them.
    #[allow(clippy::too_many_arguments)]
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
        self.assign_inner(
            layouter,
            region_t,
            nation_t,
            customer_t,
            orders_t,
            part_t,
            supplier_t,
            lineitem_t,
            cond_nation_hash,
            const_region_name_hash,
            const_part_type_hash,
            false,
        )
        .map(|(out, _)| out)
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
            let proof_path = &crate::paths::proof_file("proof_obj_q8_new");
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
    }

    /// `cs.degree()` of this circuit is 7, and it comes from a lookup, not from a
    /// gate: the widest gate is degree 6 and the permutation argument is a
    /// constant 3. The soundness fixes added six gates and no lookup, no shuffle
    /// and no equality-bearing column that could move any of those three, so the
    /// only way they could raise the degree is by beating 6 with a gate of their
    /// own. They do not: five are degree 2 and the sixth is degree 4, the same as
    /// every other `IsZero` gate in the file. A rise would double every FFT of the
    /// prover and cost far more than the fixes are worth.
    #[test]
    fn test_max_gate_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        let mut cs = ConstraintSystem::<Fp>::default();
        let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
        let degree = cs.degree();

        // The names of the gates the soundness fixes introduced. cs.degree() is a
        // max over the gates, the lookups, the shuffles and the permutation, and
        // the fixes only ever added gates, so comparing the added gates against
        // the pre-existing ones is exactly the before/after test.
        const ADDED: [&str; 5] = [
            "query parameters are constant down the rows",
            "all_nations rows past the clean section are PAD",
            "share = 0 when den = 0",
            "region partition pad rows are the PAD tuple",
            "orders partition pad rows are the PAD tuple",
        ];

        let mut added_max = 0usize;
        let mut rest_max = 0usize;
        for g in cs.gates() {
            let d = g
                .polynomials()
                .iter()
                .map(|p| p.degree())
                .max()
                .unwrap_or(0);
            if ADDED.contains(&g.name()) || g.name() == "part partition pad rows are the PAD tuple"
            {
                added_max = added_max.max(d);
                println!("added gate {:?} has degree {}", g.name(), d);
            } else {
                rest_max = rest_max.max(d);
            }
        }
        // cs.degree() is 7 while the widest gate is 6: the 7 comes from a lookup,
        // and no lookup changed.
        println!(
            "cs.degree() = {}, max over pre-existing gates = {}, max over added gates = {}",
            degree, rest_max, added_max
        );
        println!(
            "COST advice={} fixed={} selectors={} gates={} lookups={} shuffles={}",
            cs.num_advice_columns(),
            cs.num_fixed_columns(),
            cs.num_selectors(),
            cs.gates().len(),
            cs.lookups().len(),
            cs.shuffles().len(),
        );
        assert!(
            added_max <= rest_max,
            "an added soundness gate has degree {} against a pre-existing maximum \
             of {}, so it raised cs.degree() and doubled the prover's FFTs",
            added_max,
            rest_max
        );
        // 8, one above the 7 of `q8_obj.rs`: gating both sides of a lookup by
        // the selector COLUMN rather than by a selector RANGE costs one degree
        // on each side. It buys no FFT, which is the number that matters: halo2
        // sizes the extended domain at the next power of two above degree - 1,
        // so 7, 8 and 9 all run on an 8x domain and only 10 doubles it.
        assert!(
            degree <= 9,
            "cs.degree() rose to {}, which would double every FFT the prover runs",
            degree
        );
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
        super::set_hide_one_clean_tuple(true);
        let tampered = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        let verdict = tampered.verify();
        super::set_hide_one_clean_tuple(false);

        let failures = verdict.expect_err("condition (4) accepted a hidden joinable tuple");
        // Name every constraint that fired, so that a newly added gate cannot
        // silently take over this direction and destroy the evidence that the
        // Cardinality Preservation Check is the thing catching the cheat.
        let names: Vec<String> = failures
            .iter()
            .map(|f| match f {
                VerifyFailure::ConstraintNotSatisfied { constraint, .. } => {
                    format!("gate {}", constraint)
                }
                VerifyFailure::Lookup { name, .. } => format!("lookup {}", name),
                VerifyFailure::Shuffle { name, .. } => format!("shuffle {}", name),
                other => format!("{:?}", other),
            })
            .collect();
        println!(
            "hidden joinable tuple: {} failures, constraints: {:?}",
            failures.len(),
            names
        );
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
        super::set_mark_all_clean(true);
        let all_clean = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        let verdict = all_clean.verify();
        super::set_mark_all_clean(false);

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
        let other: Vec<String> = failures
            .iter()
            .filter(
                |f| !matches!(f, VerifyFailure::Lookup { name, .. } if name.starts_with("pw: ")),
            )
            .map(|f| match f {
                VerifyFailure::ConstraintNotSatisfied { constraint, .. } => {
                    format!("gate {}", constraint)
                }
                VerifyFailure::Lookup { name, .. } => format!("lookup {}", name),
                VerifyFailure::Shuffle { name, .. } => format!("shuffle {}", name),
                o => format!("{:?}", o),
            })
            .collect();
        println!(
            "all-clean partition: {} failures, {} of them Pairwise Consistency, first: {}",
            failures.len(),
            pw.len(),
            pw[0]
        );
        println!("all-clean partition, non-pw failures: {:?}", other);
    }
}
