// q8_obj.rs
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

use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

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

// Orders (filtered table used for lookup): [o_orderkey, o_year]
const PAD_OKEY: u64 = MAX_SENTINEL;
const PAD_OYEAR: u64 = PAD_YEAR;

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
// ------------------------------------------------

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

    // ---------- part type filter (PROMO BRUSHED COPPER) ----------
    q_part_pred: Selector,
    q_part_link: Selector,
    p_keep: Column<Advice>,
    iz_part_type: IsZeroConfig<F>,
    p_filt_pad: Vec<Column<Advice>>, // 2
    p_join_pad: Vec<Column<Advice>>, // 2
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
    o_filt_pad: Vec<Column<Advice>>, // 2 (keep? [okey,oyear] : PAD)
    o_join_pad: Vec<Column<Advice>>, // 2 (kept orders + PAD) length=orders.len()
    perm_orders: PermAnyConfig,

    // ---------- lineitem reorder: join prefix then dis then PAD ----------
    l_join_pad: Vec<Column<Advice>>, // 5, length=lineitem.len()
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

        // filtered padded and join padded
        let p_filt_pad = vec![meta.advice_column(), meta.advice_column()];
        let p_join_pad = vec![meta.advice_column(), meta.advice_column()];
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

            let f0 = m.query_advice(p_filt_pad[0], Rotation::cur());
            let f1 = m.query_advice(p_filt_pad[1], Rotation::cur());

            vec![
                q.clone() * (f0 - (keep.clone() * b0 + dropv.clone() * pad0)),
                q * (f1 - (keep * b1 + dropv * pad1)),
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

        // filtered orders tables (okey, oyear) for lookup
        let q_orders_filt_gate = meta.selector();
        let o_filt_pad = vec![meta.advice_column(), meta.advice_column()];
        let o_join_pad = vec![meta.advice_column(), meta.advice_column()];
        for &c in o_filt_pad.iter().chain(o_join_pad.iter()) {
            meta.enable_equality(c);
        }

        // o_filt_pad = keep? [o_orderkey, o_year] : PAD
        meta.create_gate("o_filt_pad = keep? order : PAD", |m| {
            let q = m.query_selector(q_orders_filt_gate);
            let keep = m.query_advice(o_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let dropv = one.clone() - keep.clone();

            let pad0 = Expression::Constant(F::from(PAD_OKEY));
            let pad1 = Expression::Constant(F::from(PAD_OYEAR));

            let b0 = m.query_advice(orders[0], Rotation::cur());
            let b1 = m.query_advice(orders[2], Rotation::cur());

            let f0 = m.query_advice(o_filt_pad[0], Rotation::cur());
            let f1 = m.query_advice(o_filt_pad[1], Rotation::cur());

            vec![
                q.clone() * (f0 - (keep.clone() * b0 + dropv.clone() * pad0)),
                q * (f1 - (keep * b1 + dropv * pad1)),
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

        // ---------- lineitem permutation to l_join_pad (reorder join rows first) ----------
        let l_join_pad = (0..5).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for &c in l_join_pad.iter() {
            meta.enable_equality(c);
        }
        let q_perm_l1 = meta.complex_selector();
        let q_perm_l2 = meta.complex_selector();
        let perm_lineitem = PermAnyChip::configure(
            meta,
            q_perm_l1,
            q_perm_l2,
            lineitem.clone(),
            l_join_pad.clone(),
        );

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

        // p_filt_pad, p_join_pad
        let p_filt_pad_u64: Vec<Vec<u64>> = part_t
            .iter()
            .zip(p_keep_u64.iter())
            .map(|(r, &k)| {
                if k == 1 {
                    r.clone()
                } else {
                    vec![PAD_PKEY, PAD_PTYPE]
                }
            })
            .collect();

        let mut p_join_pad_u64: Vec<Vec<u64>> = vec![];
        p_join_pad_u64.extend(p_join.iter().cloned());
        while p_join_pad_u64.len() < part_t.len() {
            p_join_pad_u64.push(vec![PAD_PKEY, PAD_PTYPE]);
        }

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

        // build filtered orders table (okey, year)
        let o_filt_pad_u64: Vec<Vec<u64>> = orders_t
            .iter()
            .zip(o_keep_u64.iter())
            .map(|(r, &k)| {
                if k == 1 {
                    vec![r[0], r[2]]
                } else {
                    vec![PAD_OKEY, PAD_OYEAR]
                }
            })
            .collect();

        let mut o_join_rows: Vec<Vec<u64>> = orders_t
            .iter()
            .zip(o_keep_u64.iter())
            .filter(|(_, &k)| k == 1)
            .map(|(r, _)| vec![r[0], r[2]])
            .collect();
        while o_join_rows.len() < orders_t.len() {
            o_join_rows.push(vec![PAD_OKEY, PAD_OYEAR]);
        }
        let o_join_pad_u64 = o_join_rows;

        // reorder lineitems: join prefix where all joins pass
        let mut l_join: Vec<Vec<u64>> = vec![];
        let mut l_dis: Vec<Vec<u64>> = vec![];

        for r in lineitem_t.iter() {
            let okey = r[0];
            let pkey = r[1];
            let skey = r[2];

            let ok = keep_orderkeys.contains(&okey)
                && partkeys_keep.contains(&pkey)
                && supp_nat.contains_key(&skey)
                && nat_name.contains_key(&supp_nat[&skey]);

            if ok {
                l_join.push(r.clone());
            } else {
                l_dis.push(r.clone());
            }
        }
        let join_len = l_join.len();

        let mut l_join_pad_u64: Vec<Vec<u64>> = vec![];
        l_join_pad_u64.extend(l_join.into_iter());
        l_join_pad_u64.extend(l_dis.into_iter());
        while l_join_pad_u64.len() < lineitem_t.len() {
            l_join_pad_u64.push(vec![PAD_LOKEY, PAD_LPKEY, PAD_LSKEY, PAD_LEXT, PAD_LDISC]);
        }

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

            let year = orders_year[&okey];
            let nkey = supp_nat[&skey];
            let nname = nat_name[&nkey];

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

                    // p_filt_pad + p_join_pad
                    region.assign_advice(
                        || "p_filt0",
                        self.config.p_filt_pad[0],
                        i,
                        || Value::known(F::from(p_filt_pad_u64[i][0])),
                    )?;
                    region.assign_advice(
                        || "p_filt1",
                        self.config.p_filt_pad[1],
                        i,
                        || Value::known(F::from(p_filt_pad_u64[i][1])),
                    )?;
                    region.assign_advice(
                        || "p_join0",
                        self.config.p_join_pad[0],
                        i,
                        || Value::known(F::from(p_join_pad_u64[i][0])),
                    )?;
                    region.assign_advice(
                        || "p_join1",
                        self.config.p_join_pad[1],
                        i,
                        || Value::known(F::from(p_join_pad_u64[i][1])),
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

                    // filtered orders tables
                    region.assign_advice(
                        || "o_filt0",
                        self.config.o_filt_pad[0],
                        i,
                        || Value::known(F::from(o_filt_pad_u64[i][0])),
                    )?;
                    region.assign_advice(
                        || "o_filt1",
                        self.config.o_filt_pad[1],
                        i,
                        || Value::known(F::from(o_filt_pad_u64[i][1])),
                    )?;
                    region.assign_advice(
                        || "o_join0",
                        self.config.o_join_pad[0],
                        i,
                        || Value::known(F::from(o_join_pad_u64[i][0])),
                    )?;
                    region.assign_advice(
                        || "o_join1",
                        self.config.o_join_pad[1],
                        i,
                        || Value::known(F::from(o_join_pad_u64[i][1])),
                    )?;
                }

                // ---- lineitem permutation to l_join_pad ----
                for i in 0..lineitem_t.len() {
                    self.config.perm_lineitem.q_perm1.enable(&mut region, i)?;
                    self.config.perm_lineitem.q_perm2.enable(&mut region, i)?;
                    for j in 0..5 {
                        region.assign_advice(
                            || "l_join_pad",
                            self.config.l_join_pad[j],
                            i,
                            || Value::known(F::from(l_join_pad_u64[i][j])),
                        )?;
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
                        let year = orders_year[&okey];
                        let nkey = supp_nat[&skey];
                        let nname = nat_name[&nkey];

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
    use halo2_proofs::dev::MockProver;
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
    use std::time::Instant;
    use std::{fs::File, io::Write, path::Path};

    fn generate_and_verify_proof<C: Circuit<Fp>>(
        circuit: C,
        public_input: &[Fp],
        proof_path: &str,
    ) {
        let params_path = "/home2/binbin/PoneglyphDB/src/proof/param16";
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
    fn test_1() {
        let k = 16;

        // Adjust paths to your repo
        let region_path = "/home2/binbin/PoneglyphDB/src/data/region.cvs";
        let nation_path = "/home2/binbin/PoneglyphDB/src/data/nation.tbl";
        let customer_path = "/home2/binbin/PoneglyphDB/src/data/customer.tbl";
        let orders_path = "/home2/binbin/PoneglyphDB/src/data/orders.tbl";
        let part_path = "/home2/binbin/PoneglyphDB/src/data/part.tbl";
        let supplier_path = "/home2/binbin/PoneglyphDB/src/data/supplier.tbl";
        let lineitem_path = "/home2/binbin/PoneglyphDB/src/data/lineitem.tbl";

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
            let proof_path = "/home2/binbin/PoneglyphDB/src/proof/proof_obj_q8";
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
    }
}
