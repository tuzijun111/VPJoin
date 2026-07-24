use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::lessthan_or_equal_generic::{
    LtEqGenericChip, LtEqGenericConfig, LtEqGenericInstruction,
};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};

use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use std::marker::PhantomData;

// One shared definition of the PAD/sentinel discipline (from q5_obj).
use super::q5_obj::{
    q5_derive, Q5Chip, Q5Derived, MAX_SENTINEL, NUM_BYTES, PAD_REV, PAD_U64, SCALE, SHIFT_NATION,
};

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

pub const LANE_ROWS: usize = 63_000;

pub const SEG: usize = 32;

/// Public structural cap on the lane count.
pub const MAX_LANES: usize = 16;

/// c = ceil(n / lane_rows): the PUBLIC lane count hosting an n-row pipeline.
pub fn lanes_for(n: usize, lane_rows: usize) -> usize {
    assert!(lane_rows > 0, "lane_rows must be positive");
    ((n + lane_rows - 1) / lane_rows).max(1)
}

/// Lane count for a released capacity, `Err` when the release needs more lanes
/// than the structural cap allows.
///
/// A release that does not fit is an INFEASIBLE REQUEST, not a bug: a sweep
/// must be able to report the cell and move on, so this returns the same
/// information the panicking path used to abort with.
pub fn try_lanes_for_capacity(capacity: usize) -> Result<usize, String> {
    let c = lanes_for(capacity, LANE_ROWS);
    if c > MAX_LANES {
        return Err(format!(
            "released capacity {} needs {} lanes at base degree k={} ({} usable rows per \
             lane), above MAX_LANES={}",
            capacity, c, DP_LANE_K, LANE_ROWS, MAX_LANES
        ));
    }
    Ok(c)
}

// `Circuit::configure` has no access to the circuit instance (the
// `circuit-params` feature of halo2 is not enabled in this build), so the
// caller stamps the PUBLIC lane count here before keygen / MockProver runs;
// `synthesize` asserts the resulting config matches `self.num_lanes`.
// Thread-local so parallel tests with different lane counts cannot race.
thread_local! {
    static CONFIG_LANES: std::cell::Cell<usize> = const { std::cell::Cell::new(1) };
}

/// Stamp the lane count the NEXT `configure` call on this thread lays out.
pub fn set_config_lanes(c: usize) {
    assert!(
        (1..=MAX_LANES).contains(&c),
        "lane count {} outside 1..={}",
        c,
        MAX_LANES
    );
    CONFIG_LANES.with(|l| l.set(c));
}

/// MockProver fault injection used by the negative tests in this file.
/// Production callers leave this at `None`; it only perturbs the witness,
/// never the constraint system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tamper {
    #[default]
    None,
    /// Bump one emitted lane partial sum by 1 (lane emit gate and the drain
    /// shuffle must both reject).
    LanePartialSum,
    /// Overwrite one merge sentinel's nation key (merge sentinel gate and
    /// the drain shuffle must both reject).
    MergeSentinel,
}

/// One lane: a full structural replica of q5_obj's ls_join + aggregation
/// column group, minus the attach-name/ORDER BY tail (those run once over
/// the merge region).
#[derive(Clone, Debug)]
pub struct LaneConfig<F: Field + Ord> {
    ls_join: Vec<Column<Advice>>,   // [okey, nk_shift, ext, disc]
    ls_sorted: Vec<Column<Advice>>, // lane-local sort by nk_shift
    perm_lsort: PermAnyConfig,      // ls_join <-> ls_sorted (lane-local)
    line_rev: Column<Advice>,
    run_sum: Column<Advice>,
    iz_same_prev: IsZeroConfig<F>,
    iz_same_next: IsZeroConfig<F>,
    // emitted group rows aligned with ls_sorted: [nk_shift, partial_sum];
    // non-emitting rows hold the SENTINEL (PAD_U64, 0)
    res_pad: Vec<Column<Advice>>,
    // fixed marker of this lane's merge segment [l*SEG, (l+1)*SEG)
    seg: Column<Fixed>,
}

#[derive(Clone, Debug)]
pub struct Q5DpConfig<F: Field + Ord> {
    // ---------------- base tables ----------------
    customer: Vec<Column<Advice>>,
    orders: Vec<Column<Advice>>,
    lineitem: Vec<Column<Advice>>,
    supplier: Vec<Column<Advice>>,
    nation: Vec<Column<Advice>>,
    region_file: Vec<Column<Advice>>,

    // ---------------- conditions ----------------
    cond_europe: Column<Advice>,
    cond_start: Column<Advice>,
    cond_end: Column<Advice>,

    // ---------------- bag materialization: NR ----------------
    q_nr_join: Selector,
    q_nr_pred: Selector,
    nr_rname: Column<Advice>,
    nr_keep: Column<Advice>,
    nr_pair: Vec<Column<Advice>>,
    nr_filt_pad: Vec<Column<Advice>>,
    nr_out_pad: Vec<Column<Advice>>,
    perm_nr: PermAnyConfig,
    iz_nr: IsZeroConfig<F>,

    // ---------------- bag materialization: CO ----------------
    q_oc_join: Selector,
    q_co_ge: Selector,
    q_co_lt: Selector,
    q_co_and: Selector,
    co_ge_ok: Column<Advice>,
    co_lt_ok: Column<Advice>,
    co_keep: Column<Advice>,
    co_nk: Column<Advice>,
    co_pair: Vec<Column<Advice>>,
    co_filt_pad: Vec<Column<Advice>>,
    co_out_pad: Vec<Column<Advice>>,
    perm_co: PermAnyConfig,
    lteq_start_le_odate: LtEqGenericConfig<F, NUM_BYTES>,
    lt_odate_lt_end: LtConfig<F, NUM_BYTES>,

    // ---------------- bag materialization: LS ----------------
    q_ls_join: Selector,
    ls_mat: Vec<Column<Advice>>,

    // ---------------- LS partition: disjoin side ----------------
    ls_disjoin: Vec<Column<Advice>>,
    ls_part_pad: Vec<Column<Advice>>,
    perm_ls: PermAnyConfig,

    // ---------------- membership + gap proof on ls_disjoin ----------------
    q_flagged_lookup: Selector,
    q_lookup_complex: Selector,
    flags_in: Vec<Column<Advice>>,
    range_low_high: Vec<Column<Advice>>,

    lt_co_gap_low: LtConfig<F, NUM_BYTES>,
    lt_co_gap_high: LtConfig<F, NUM_BYTES>,
    lt_nr_gap_low: LtConfig<F, NUM_BYTES>,
    lt_nr_gap_high: LtConfig<F, NUM_BYTES>,

    co_key: Column<Advice>,
    co_key_next: Column<Advice>,
    nr_key: Column<Advice>,
    nr_key_next: Column<Advice>,

    // ---------------- lanes (shared selectors, per-lane columns) ----------
    // All lanes are active on the SAME rows 0..lane_rows, so one selector of
    // each kind serves every lane; only the columns replicate.
    lanes: Vec<LaneConfig<F>>,
    q_join_member: Selector, // membership lookups, all lane rows
    q_lane_line: Selector,
    q_lane_first: Selector,
    q_lane_accu: Selector,
    q_drain: Selector, // LHS gate of the per-lane drain shuffles (complex)

    // ---------------- cross-lane merge ----------------
    merge_nk: Column<Advice>,
    merge_sum: Column<Advice>,
    merge_real: Column<Advice>, // 1 on drained group rows, 0 on sentinels
    q_merge: Selector,
    msort: Vec<Column<Advice>>, // merge sorted by nk, sentinels last
    perm_merge: PermAnyConfig,

    // ---------------- final aggregation over the merge region -------------
    q_m_line: Selector,
    q_m_first: Selector,
    q_m_accu: Selector,
    m_run_sum: Column<Advice>,
    iz_m_same_prev: IsZeroConfig<F>,
    iz_m_same_next: IsZeroConfig<F>,
    // [nationkey_shift, n_name_hash, revenue], as q5_obj's res_pad
    m_res_pad: Vec<Column<Advice>>,
    q_m_res_lookup: Selector,

    // ORDER BY revenue DESC
    m_res_sorted: Vec<Column<Advice>>,
    perm_mres: PermAnyConfig,
    q_sort_mres: Selector,
    lteq_rev_next_le_cur: LtEqGenericConfig<F, NUM_BYTES>,

    // public
    instance: Column<Instance>,
    instance_test: Column<Advice>,
}

#[derive(Clone, Debug)]
pub struct Q5DpChip<F: Field + Ord> {
    config: Q5DpConfig<F>,
}

impl<F: Field + Ord> Q5DpChip<F> {
    pub fn construct(config: Q5DpConfig<F>) -> Self {
        Self { config }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> Q5DpConfig<F> {
        let num_lanes = CONFIG_LANES.with(|l| l.get());
        assert!(
            (1..=MAX_LANES).contains(&num_lanes),
            "configured lane count {} outside 1..={} (call set_config_lanes first)",
            num_lanes,
            MAX_LANES
        );

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

        // ---------------- NR materialization (as q5_obj) ----------------
        let q_nr_join = meta.complex_selector();
        let q_nr_pred = meta.selector();

        let nr_rname = meta.advice_column();
        let nr_keep = meta.advice_column();

        let nr_pair = vec![meta.advice_column(), meta.advice_column()];
        let (nr_filt_pad, nr_out_pad, perm_nr) = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            let a = vec![meta.advice_column(), meta.advice_column()];
            let b = vec![meta.advice_column(), meta.advice_column()];
            let perm = PermAnyChip::configure(meta, q1, q2, a.clone(), b.clone());
            (a, b, perm)
        };

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

        meta.create_gate("nr_pair copies nation cols", |m| {
            let q = m.query_selector(q_nr_pred);
            let nk = m.query_advice(nation[0], Rotation::cur());
            let nm = m.query_advice(nation[1], Rotation::cur());
            let p0 = m.query_advice(nr_pair[0], Rotation::cur());
            let p1 = m.query_advice(nr_pair[1], Rotation::cur());
            vec![q.clone() * (p0 - nk), q * (p1 - nm)]
        });

        meta.create_gate("link nr_filt_pad", |m| {
            let q = m.query_selector(perm_nr.q_perm1);
            let keep = m.query_advice(nr_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let drop = one.clone() - keep.clone();
            let mut cs = vec![q.clone() * keep.clone() * (one.clone() - keep.clone())];
            for j in 0..2 {
                let b = m.query_advice(nr_pair[j], Rotation::cur());
                let f = m.query_advice(nr_filt_pad[j], Rotation::cur());
                let p = Expression::Constant(F::from(PAD_U64));
                cs.push(q.clone() * (f - (keep.clone() * b + drop.clone() * p)));
            }
            cs
        });

        // ---------------- CO materialization (as q5_obj) ----------------
        let q_oc_join = meta.complex_selector();
        let q_co_ge = meta.selector();
        let q_co_lt = meta.selector();
        let q_co_and = meta.selector();

        let co_ge_ok = meta.advice_column();
        let co_lt_ok = meta.advice_column();
        let co_keep = meta.advice_column();

        let co_nk = meta.advice_column();
        let co_pair = vec![meta.advice_column(), meta.advice_column()];
        let (co_filt_pad, co_out_pad, perm_co) = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            let a = vec![meta.advice_column(), meta.advice_column()];
            let b = vec![meta.advice_column(), meta.advice_column()];
            let perm = PermAnyChip::configure(meta, q1, q2, a.clone(), b.clone());
            (a, b, perm)
        };

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

        meta.create_gate("co_pair copies orderkey and looked nk", |m| {
            let q = m.query_selector(q_co_and);
            let ok = m.query_advice(orders[2], Rotation::cur());
            let nk = m.query_advice(co_nk, Rotation::cur());
            let p0 = m.query_advice(co_pair[0], Rotation::cur());
            let p1 = m.query_advice(co_pair[1], Rotation::cur());
            vec![q.clone() * (p0 - ok), q * (p1 - nk)]
        });

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

        meta.create_gate("link co_filt_pad", |m| {
            let q = m.query_selector(perm_co.q_perm1);
            let keep = m.query_advice(co_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let drop = one.clone() - keep.clone();
            let mut cs = vec![q.clone() * keep.clone() * (one.clone() - keep.clone())];
            for j in 0..2 {
                let b = m.query_advice(co_pair[j], Rotation::cur());
                let f = m.query_advice(co_filt_pad[j], Rotation::cur());
                let p = Expression::Constant(F::from(PAD_U64));
                cs.push(q.clone() * (f - (keep.clone() * b + drop.clone() * p)));
            }
            cs
        });

        // ---------------- LS materialization (as q5_obj) ----------------
        let q_ls_join = meta.complex_selector();
        let ls_mat = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

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
        // ls_join is now laned; ls_disjoin and ls_part_pad stay one group.
        let ls_disjoin = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let ls_part_pad = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        for &c in ls_disjoin.iter().chain(ls_part_pad.iter()) {
            meta.enable_equality(c);
        }

        let perm_ls = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            PermAnyChip::configure(meta, q1, q2, ls_mat.clone(), ls_part_pad.clone())
        };

        // ---------------- membership + gap proof (as q5_obj) ----------------
        let q_flagged_lookup = meta.selector();
        let q_lookup_complex = meta.complex_selector();
        let q_join_member = meta.complex_selector();

        let flags_in = vec![meta.advice_column(), meta.advice_column()];
        let range_low_high = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        let lt_co_gap_low = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| {
                let q = m.query_selector(q_flagged_lookup);
                let in_co = m.query_advice(flags_in[0], Rotation::cur());
                q * (Expression::Constant(F::ONE) - in_co)
            },
            |m| m.query_advice(range_low_high[0], Rotation::cur()),
            |m| {
                m.query_advice(ls_disjoin[0], Rotation::cur())
                    * Expression::Constant(F::from(SHIFT_NATION))
                    + m.query_advice(ls_disjoin[1], Rotation::cur())
            },
        );
        let lt_co_gap_high = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| {
                let q = m.query_selector(q_flagged_lookup);
                let in_co = m.query_advice(flags_in[0], Rotation::cur());
                q * (Expression::Constant(F::ONE) - in_co)
            },
            |m| {
                m.query_advice(ls_disjoin[0], Rotation::cur())
                    * Expression::Constant(F::from(SHIFT_NATION))
                    + m.query_advice(ls_disjoin[1], Rotation::cur())
            },
            |m| m.query_advice(range_low_high[1], Rotation::cur()),
        );

        let lt_nr_gap_low = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| {
                let q = m.query_selector(q_flagged_lookup);
                let in_nr = m.query_advice(flags_in[1], Rotation::cur());
                q * (Expression::Constant(F::ONE) - in_nr)
            },
            |m| m.query_advice(range_low_high[2], Rotation::cur()),
            |m| m.query_advice(ls_disjoin[1], Rotation::cur()),
        );
        let lt_nr_gap_high = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| {
                let q = m.query_selector(q_flagged_lookup);
                let in_nr = m.query_advice(flags_in[1], Rotation::cur());
                q * (Expression::Constant(F::ONE) - in_nr)
            },
            |m| m.query_advice(ls_disjoin[1], Rotation::cur()),
            |m| m.query_advice(range_low_high[3], Rotation::cur()),
        );

        let co_key = meta.advice_column();
        let co_key_next = meta.advice_column();
        let nr_key = meta.advice_column();
        let nr_key_next = meta.advice_column();

        meta.lookup_any("CO member (ls_disjoin)", |m| {
            let q = m.query_selector(q_lookup_complex);
            let in_co = m.query_advice(flags_in[0], Rotation::cur());
            let key = (m.query_advice(ls_disjoin[0], Rotation::cur())
                * Expression::Constant(F::from(SHIFT_NATION)))
                + m.query_advice(ls_disjoin[1], Rotation::cur());
            vec![(q * in_co * key, m.query_advice(co_key, Rotation::cur()))]
        });
        meta.lookup_any("CO gap pair (ls_disjoin)", |m| {
            let q = m.query_selector(q_lookup_complex);
            let in_co = m.query_advice(flags_in[0], Rotation::cur());
            let gate = q * (Expression::Constant(F::ONE) - in_co);
            vec![
                (
                    gate.clone() * m.query_advice(range_low_high[0], Rotation::cur()),
                    m.query_advice(co_key, Rotation::cur()),
                ),
                (
                    gate * m.query_advice(range_low_high[1], Rotation::cur()),
                    m.query_advice(co_key_next, Rotation::cur()),
                ),
            ]
        });

        meta.lookup_any("NR member (ls_disjoin)", |m| {
            let q = m.query_selector(q_lookup_complex);
            let in_nr = m.query_advice(flags_in[1], Rotation::cur());
            let nk = m.query_advice(ls_disjoin[1], Rotation::cur());
            vec![(q * in_nr * nk, m.query_advice(nr_key, Rotation::cur()))]
        });
        meta.lookup_any("NR gap pair (ls_disjoin)", |m| {
            let q = m.query_selector(q_lookup_complex);
            let in_nr = m.query_advice(flags_in[1], Rotation::cur());
            let gate = q * (Expression::Constant(F::ONE) - in_nr);
            vec![
                (
                    gate.clone() * m.query_advice(range_low_high[2], Rotation::cur()),
                    m.query_advice(nr_key, Rotation::cur()),
                ),
                (
                    gate * m.query_advice(range_low_high[3], Rotation::cur()),
                    m.query_advice(nr_key_next, Rotation::cur()),
                ),
            ]
        });

        meta.create_gate("LS disjoin emptiness (not both memberships)", |m| {
            let q = m.query_selector(q_flagged_lookup);
            let in_co = m.query_advice(flags_in[0], Rotation::cur());
            let in_nr = m.query_advice(flags_in[1], Rotation::cur());
            let one = Expression::Constant(F::ONE);

            let co_low_ok = lt_co_gap_low.is_lt(m, None);
            let co_high_ok = lt_co_gap_high.is_lt(m, None);
            let nr_low_ok = lt_nr_gap_low.is_lt(m, None);
            let nr_high_ok = lt_nr_gap_high.is_lt(m, None);

            vec![
                q.clone() * in_co.clone() * (one.clone() - in_co.clone()),
                q.clone() * in_nr.clone() * (one.clone() - in_nr.clone()),
                q.clone() * in_co.clone() * in_nr.clone(),
                q.clone() * (one.clone() - in_co.clone()) * (one.clone() - co_low_ok),
                q.clone() * (one.clone() - in_co.clone()) * (one.clone() - co_high_ok),
                q.clone() * (one.clone() - in_nr.clone()) * (one.clone() - nr_low_ok),
                q * (one.clone() - in_nr) * (one - nr_high_ok),
            ]
        });

        // ---------------- cross-lane merge columns ----------------
        // Declared before the lanes so each lane's drain shuffle can target
        // them.  One (nk, sum) pair plus a real/sentinel flag; lane l owns
        // the fixed segment [l*SEG, (l+1)*SEG).
        let merge_nk = meta.advice_column();
        let merge_sum = meta.advice_column();
        let merge_real = meta.advice_column();
        let q_merge = meta.selector();
        let q_drain = meta.complex_selector();

        // ---------------- lanes ----------------
        // Shared row-pattern selectors: every lane is active on the same rows
        // 0..lane_rows, so one selector of each kind gates all lanes' gates.
        let q_lane_line = meta.selector();
        let q_lane_first = meta.selector();
        let q_lane_accu = meta.selector();
        let q_lane_perm1 = meta.complex_selector();
        let q_lane_perm2 = meta.complex_selector();

        let mut lanes: Vec<LaneConfig<F>> = Vec::with_capacity(num_lanes);
        for _l in 0..num_lanes {
            let ls_join = vec![
                meta.advice_column(),
                meta.advice_column(),
                meta.advice_column(),
                meta.advice_column(),
            ];
            let ls_sorted = vec![
                meta.advice_column(),
                meta.advice_column(),
                meta.advice_column(),
                meta.advice_column(),
            ];
            // equality: ls_part_pad links block-wise into this lane
            for &c in ls_join.iter() {
                meta.enable_equality(c);
            }

            // join rows must be members of both sets (as q5_obj, per lane;
            // pad rows carry the MAX_SENTINEL keys, which both key tables
            // contain, so the selector covers ALL lane rows)
            meta.lookup_any("CO member (ls_join lane)", |m| {
                let q = m.query_selector(q_join_member);
                let key = (m.query_advice(ls_join[0], Rotation::cur())
                    * Expression::Constant(F::from(SHIFT_NATION)))
                    + m.query_advice(ls_join[1], Rotation::cur());
                vec![(q * key, m.query_advice(co_key, Rotation::cur()))]
            });
            meta.lookup_any("NR member (ls_join lane)", |m| {
                let q = m.query_selector(q_join_member);
                let nk = m.query_advice(ls_join[1], Rotation::cur());
                vec![(q * nk, m.query_advice(nr_key, Rotation::cur()))]
            });

            // lane-local sort by nation key
            let perm_lsort = PermAnyChip::configure(
                meta,
                q_lane_perm1,
                q_lane_perm2,
                ls_join.clone(),
                ls_sorted.clone(),
            );

            let line_rev = meta.advice_column();
            let run_sum = meta.advice_column();

            let aux_same_prev = meta.advice_column();
            let aux_same_next = meta.advice_column();
            let iz_same_prev = IsZeroChip::configure(
                meta,
                |m| m.query_selector(q_lane_accu),
                |m| {
                    m.query_advice(ls_sorted[1], Rotation::cur())
                        - m.query_advice(ls_sorted[1], Rotation::prev())
                },
                aux_same_prev,
            );
            let iz_same_next = IsZeroChip::configure(
                meta,
                |m| m.query_selector(q_lane_line),
                |m| {
                    m.query_advice(ls_sorted[1], Rotation::next())
                        - m.query_advice(ls_sorted[1], Rotation::cur())
                },
                aux_same_next,
            );

            // line_rev = ext*(SCALE-disc), run_sum: same gates as q5_obj
            meta.create_gate("lane line_rev", |m| {
                let q = m.query_selector(q_lane_line);
                let ext = m.query_advice(ls_sorted[2], Rotation::cur());
                let disc = m.query_advice(ls_sorted[3], Rotation::cur());
                let lr = m.query_advice(line_rev, Rotation::cur());
                let scale = Expression::Constant(F::from(SCALE));
                vec![q * (lr - ext * (scale - disc))]
            });
            meta.create_gate("lane run_sum_first", |m| {
                let q = m.query_selector(q_lane_first);
                let rs = m.query_advice(run_sum, Rotation::cur());
                let lr = m.query_advice(line_rev, Rotation::cur());
                vec![q * (rs - lr)]
            });
            meta.create_gate("lane run_sum_accu", |m| {
                let q = m.query_selector(q_lane_accu);
                let same = iz_same_prev.expr();
                let rs_cur = m.query_advice(run_sum, Rotation::cur());
                let rs_prev = m.query_advice(run_sum, Rotation::prev());
                let lr = m.query_advice(line_rev, Rotation::cur());
                vec![q * (rs_cur - (same * rs_prev + lr))]
            });

            // emit (nk, partial_sum) at group boundaries; every other row is
            // the SENTINEL (PAD_U64, 0).  Names are attached ONCE later, in
            // the merge stage, not per lane.
            let res_pad = vec![meta.advice_column(), meta.advice_column()];
            meta.create_gate("lane emit res_pad", |m| {
                let q = m.query_selector(q_lane_line);
                let one = Expression::Constant(F::ONE);
                let is_last = one.clone() - iz_same_next.expr();
                let not_last = one - is_last.clone();

                let nk = m.query_advice(ls_sorted[1], Rotation::cur());
                let rs = m.query_advice(run_sum, Rotation::cur());

                let out_nk = m.query_advice(res_pad[0], Rotation::cur());
                let out_sum = m.query_advice(res_pad[1], Rotation::cur());

                let pad_nk = Expression::Constant(F::from(PAD_U64));
                let pad_rev = Expression::Constant(F::from(PAD_REV));

                vec![
                    q.clone() * (out_nk - (is_last.clone() * nk + not_last.clone() * pad_nk)),
                    q * (out_sum - (is_last * rs + not_last * pad_rev)),
                ]
            });

            // drain this lane's emitted rows into merge segment l.  Over the
            // full domain both sides carry g_l real rows plus sentinels:
            //   LHS: lane rows contribute res_pad (real or sentinel), all
            //        other rows the constant SENTINEL via (1 - q_drain);
            //   RHS: segment-l rows contribute the merge table, all other
            //        rows the constant SENTINEL via (1 - seg).
            let seg = meta.fixed_column();
            meta.shuffle("drain lane to merge segment", |m| {
                let q = m.query_selector(q_drain);
                let sg = m.query_fixed(seg, Rotation::cur());
                let one = Expression::Constant(F::ONE);
                let pad = Expression::Constant(F::from(PAD_U64));

                let lhs_nk = q.clone() * m.query_advice(res_pad[0], Rotation::cur())
                    + (one.clone() - q.clone()) * pad.clone();
                let lhs_sum = q * m.query_advice(res_pad[1], Rotation::cur());

                let rhs_nk = sg.clone() * m.query_advice(merge_nk, Rotation::cur())
                    + (one - sg.clone()) * pad;
                let rhs_sum = sg * m.query_advice(merge_sum, Rotation::cur());

                vec![(lhs_nk, rhs_nk), (lhs_sum, rhs_sum)]
            });

            lanes.push(LaneConfig {
                ls_join,
                ls_sorted,
                perm_lsort,
                line_rev,
                run_sum,
                iz_same_prev,
                iz_same_next,
                res_pad,
                seg,
            });
        }

        // every merge row is either the SENTINEL or flagged real
        meta.create_gate("merge row is sentinel or flagged real", |m| {
            let q = m.query_selector(q_merge);
            let real = m.query_advice(merge_real, Rotation::cur());
            let nk = m.query_advice(merge_nk, Rotation::cur());
            let sum = m.query_advice(merge_sum, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let pad = Expression::Constant(F::from(PAD_U64));
            vec![
                q.clone() * real.clone() * (one.clone() - real.clone()),
                q.clone() * (one.clone() - real.clone()) * (nk - pad),
                q * (one - real) * sum,
            ]
        });

        // ---------------- final aggregation over the merge region ----------
        // merge -> msort (sorted by nk, sentinels last), then the same
        // grouping pattern as the lanes, summing lane partials per nation.
        let msort = vec![meta.advice_column(), meta.advice_column()];
        let perm_merge = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            PermAnyChip::configure(meta, q1, q2, vec![merge_nk, merge_sum], msort.clone())
        };

        let q_m_line = meta.selector();
        let q_m_first = meta.selector();
        let q_m_accu = meta.selector();

        let m_run_sum = meta.advice_column();

        let aux_m_same_prev = meta.advice_column();
        let aux_m_same_next = meta.advice_column();
        let iz_m_same_prev = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_m_accu),
            |m| {
                m.query_advice(msort[0], Rotation::cur())
                    - m.query_advice(msort[0], Rotation::prev())
            },
            aux_m_same_prev,
        );
        let iz_m_same_next = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_m_line),
            |m| {
                m.query_advice(msort[0], Rotation::next())
                    - m.query_advice(msort[0], Rotation::cur())
            },
            aux_m_same_next,
        );

        // merge rows already carry partial sums, so no line_rev stage here
        meta.create_gate("merge run_sum_first", |m| {
            let q = m.query_selector(q_m_first);
            let rs = m.query_advice(m_run_sum, Rotation::cur());
            let s = m.query_advice(msort[1], Rotation::cur());
            vec![q * (rs - s)]
        });
        meta.create_gate("merge run_sum_accu", |m| {
            let q = m.query_selector(q_m_accu);
            let same = iz_m_same_prev.expr();
            let rs_cur = m.query_advice(m_run_sum, Rotation::cur());
            let rs_prev = m.query_advice(m_run_sum, Rotation::prev());
            let s = m.query_advice(msort[1], Rotation::cur());
            vec![q * (rs_cur - (same * rs_prev + s))]
        });

        let m_res_pad = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let q_m_res_lookup = meta.complex_selector();

        // emit group row only at last row of each nation group (as q5_obj)
        meta.create_gate("emit merge res_pad", |m| {
            let q = m.query_selector(q_m_line);
            let one = Expression::Constant(F::ONE);
            let is_last = one.clone() - iz_m_same_next.expr();
            let not_last = one.clone() - is_last.clone();

            let nk = m.query_advice(msort[0], Rotation::cur());
            let rs = m.query_advice(m_run_sum, Rotation::cur());

            let out_nk = m.query_advice(m_res_pad[0], Rotation::cur());
            let out_nm = m.query_advice(m_res_pad[1], Rotation::cur());
            let out_rev = m.query_advice(m_res_pad[2], Rotation::cur());

            let pad_nk = Expression::Constant(F::from(PAD_U64));
            let pad_nm = Expression::Constant(F::from(PAD_U64));
            let pad_rev = Expression::Constant(F::from(PAD_REV));

            vec![
                q.clone() * (out_nk - (is_last.clone() * nk + not_last.clone() * pad_nk)),
                q.clone() * (out_rev - (is_last.clone() * rs + not_last.clone() * pad_rev)),
                q * not_last * (out_nm - pad_nm),
            ]
        });

        // attach (nk,name) via lookup into nr_out_pad, ONCE for all lanes
        meta.lookup_any("attach name from NR_out (merge)", |m| {
            let q_in = m.query_selector(q_m_res_lookup);
            let one = Expression::Constant(F::ONE);
            let is_last = one - iz_m_same_next.expr();
            let gate = q_in * is_last;

            vec![
                (
                    gate.clone() * m.query_advice(m_res_pad[0], Rotation::cur()),
                    m.query_advice(nr_out_pad[0], Rotation::cur()),
                ),
                (
                    gate * m.query_advice(m_res_pad[1], Rotation::cur()),
                    m.query_advice(nr_out_pad[1], Rotation::cur()),
                ),
            ]
        });

        // ---------------- ORDER BY revenue DESC (once, small) --------------
        let m_res_sorted = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let perm_mres = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            PermAnyChip::configure(meta, q1, q2, m_res_pad.clone(), m_res_sorted.clone())
        };
        let q_sort_mres = meta.selector();

        let lteq_rev_next_le_cur = LtEqGenericChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort_mres),
            |m| vec![m.query_advice(m_res_sorted[2], Rotation::next())],
            |m| vec![m.query_advice(m_res_sorted[2], Rotation::cur())],
        );
        meta.create_gate("ORDER BY revenue DESC (merge)", |m| {
            let q = m.query_selector(q_sort_mres);
            vec![q * (lteq_rev_next_le_cur.is_lt(m, None) - Expression::Constant(F::ONE))]
        });

        Q5DpConfig {
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
            co_nk,
            co_pair,
            co_filt_pad,
            co_out_pad,
            perm_co,
            lteq_start_le_odate,
            lt_odate_lt_end,

            q_ls_join,
            ls_mat,

            ls_disjoin,
            ls_part_pad,
            perm_ls,

            q_flagged_lookup,
            q_lookup_complex,
            flags_in,
            range_low_high,

            lt_co_gap_low,
            lt_co_gap_high,
            lt_nr_gap_low,
            lt_nr_gap_high,

            co_key,
            co_key_next,
            nr_key,
            nr_key_next,

            lanes,
            q_join_member,
            q_lane_line,
            q_lane_first,
            q_lane_accu,
            q_drain,

            merge_nk,
            merge_sum,
            merge_real,
            q_merge,
            msort,
            perm_merge,

            q_m_line,
            q_m_first,
            q_m_accu,
            m_run_sum,
            iz_m_same_prev,
            iz_m_same_next,
            m_res_pad,
            q_m_res_lookup,

            m_res_sorted,
            perm_mres,
            q_sort_mres,
            lteq_rev_next_le_cur,

            instance,
            instance_test,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        customer: Vec<Vec<u64>>,
        orders: Vec<Vec<u64>>,
        lineitem: Vec<Vec<u64>>,
        supplier: Vec<Vec<u64>>,
        nation: Vec<Vec<u64>>,
        region_file: Vec<Vec<u64>>,
        europe_hash: u64,
        start_ts: u64,
        end_ts: u64,
        nr_pad_extra: usize,
        co_pad_extra: usize,
        ls_pad_extra: usize,
        lane_rows: usize,
        tamper: Tamper,
    ) -> Result<AssignedCell<F, F>, Error> {
        // chips (same 7 u8-range loads as q5_obj: 7*256 rows before the main
        // region)
        let iz_nr_chip = IsZeroChip::construct(self.config.iz_nr.clone());

        let lteq_ge_chip =
            LtEqGenericChip::<F, NUM_BYTES>::construct(self.config.lteq_start_le_odate.clone());
        lteq_ge_chip.load(layouter)?;

        let lt_end_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_odate_lt_end.clone());
        lt_end_chip.load(layouter)?;

        let lt_co_low_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_co_gap_low.clone());
        lt_co_low_chip.load(layouter)?;
        let lt_co_high_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_co_gap_high.clone());
        lt_co_high_chip.load(layouter)?;
        let lt_nr_low_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_nr_gap_low.clone());
        lt_nr_low_chip.load(layouter)?;
        let lt_nr_high_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_nr_gap_high.clone());
        lt_nr_high_chip.load(layouter)?;

        let iz_m_same_prev_chip = IsZeroChip::construct(self.config.iz_m_same_prev.clone());
        let iz_m_same_next_chip = IsZeroChip::construct(self.config.iz_m_same_next.clone());

        let lteq_rev_chip =
            LtEqGenericChip::<F, NUM_BYTES>::construct(self.config.lteq_rev_next_le_cur.clone());
        lteq_rev_chip.load(layouter)?;

        // helpers
        fn to_field_rows<FF: Field + Ord>(u: &[Vec<u64>]) -> Vec<Vec<FF>> {
            u.iter()
                .map(|r| r.iter().map(|&x| FF::from(x)).collect())
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

        // Derived intermediates from the SAME definition as q5_obj and
        // `bench_queries::q5_pads`, so witness and DP release cannot drift.
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
            ls_join_u64,
            ls_dis_u64,
            nr_filt_pad_u64_ext,
            co_filt_pad_u64_ext,
            nr_total,
            co_total,
            nr_out_pad_u64,
            co_out_pad_u64,
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

        // key vectors for membership+gap
        let mut valid_co_keys: Vec<u64> = co_filtered
            .iter()
            .map(|r| r[0] * SHIFT_NATION + r[1])
            .collect();
        valid_co_keys.push(0);
        valid_co_keys.push(MAX_SENTINEL);
        valid_co_keys.sort();
        valid_co_keys.dedup();

        let mut valid_nr_keys: Vec<u64> = nr_filtered.iter().map(|r| r[0]).collect();
        valid_nr_keys.push(0);
        valid_nr_keys.push(MAX_SENTINEL);
        valid_nr_keys.sort();
        valid_nr_keys.dedup();

        let pad4 = vec![PAD_U64; 4];
        let ls_part_pad_u64 = pad_partition_u64(&ls_join_u64, &ls_dis_u64, lineitem.len(), &pad4);
        let ls_part_pad_f: Vec<Vec<F>> = to_field_rows::<F>(&ls_part_pad_u64);

        // ---------- lane witnesses ----------
        // n is the released pipeline length; ALL c lanes are fully assigned
        // to lane_rows rows (the last lane carries the rounding-up pads), so
        // the assigned structure is a function of the PUBLIC capacity only.
        let c = self.config.lanes.len();
        assert!(lane_rows >= 1, "lane_rows must be positive");
        assert!(
            c <= MAX_LANES,
            "lane count {} exceeds MAX_LANES {}",
            c,
            MAX_LANES
        );

        let join_len = ls_join_u64.len();
        let n = join_len.saturating_add(ls_pad_extra).max(1);
        assert!(
            c * lane_rows >= n,
            "released capacity n={} exceeds {} lanes x {} rows",
            n,
            c,
            lane_rows
        );
        let cap = c * lane_rows;

        // pad row: [okey=0, nk=PAD_U64, ext=0, disc=0] (q5_obj's convention)
        let mut ls_join_full = ls_join_u64.clone();
        while ls_join_full.len() < cap {
            ls_join_full.push(vec![0u64, PAD_U64, 0u64, 0u64]);
        }

        struct LaneWit {
            sorted: Vec<Vec<u64>>,
            line_rev: Vec<u64>,
            run_sum: Vec<u64>,
            res_pad: Vec<[u64; 2]>,
            drained: Vec<[u64; 2]>,
        }

        let mut lane_wit: Vec<LaneWit> = Vec::with_capacity(c);
        for l in 0..c {
            let mut sorted: Vec<Vec<u64>> =
                ls_join_full[l * lane_rows..(l + 1) * lane_rows].to_vec();
            sorted.sort_by_key(|r| r[1]); // by nationkey_shift (PAD_U64 last)

            let mut line_rev = vec![0u64; lane_rows];
            let mut run_sum = vec![0u64; lane_rows];
            let mut res_pad: Vec<[u64; 2]> = vec![[PAD_U64, PAD_REV]; lane_rows];
            let mut drained: Vec<[u64; 2]> = vec![];

            let mut acc: u128 = 0;
            let mut prev_nk: Option<u64> = None;
            for i in 0..lane_rows {
                let nk = sorted[i][1];
                let ext = sorted[i][2] as u128;
                let disc = sorted[i][3] as u128;
                let lr = ext * ((SCALE as u128) - disc);
                line_rev[i] = lr as u64;

                if prev_nk == Some(nk) {
                    acc += lr;
                } else {
                    acc = lr;
                }
                run_sum[i] = acc as u64;

                let next_nk = if i + 1 < lane_rows {
                    sorted[i + 1][1]
                } else {
                    0
                };

                // don't emit for PAD_U64 groups; keep res_pad as SENTINEL
                if next_nk != nk && nk != PAD_U64 {
                    res_pad[i] = [nk, run_sum[i]];
                    drained.push([nk, run_sum[i]]);
                }
                prev_nk = Some(nk);
            }
            assert!(
                drained.len() <= SEG,
                "lane {} emits {} groups, more than SEG={} (|nation| bound broken?)",
                l,
                drained.len(),
                SEG
            );
            lane_wit.push(LaneWit {
                sorted,
                line_rev,
                run_sum,
                res_pad,
                drained,
            });
        }

        // fault injection (negative MockProver tests only): the drained /
        // merge values stay honest, so exactly the lane-side constraints
        // must catch the corruption
        if tamper == Tamper::LanePartialSum {
            let cell = lane_wit
                .iter_mut()
                .flat_map(|w| w.res_pad.iter_mut())
                .find(|r| r[0] != PAD_U64)
                .expect("LanePartialSum tamper needs at least one real group");
            cell[1] = cell[1].wrapping_add(1);
        }

        // ---------- merge witness ----------
        let m_total = c * SEG;
        let mut merge_u64: Vec<[u64; 3]> = Vec::with_capacity(m_total); // [nk, sum, real]
        for w in lane_wit.iter() {
            for r in w.drained.iter() {
                merge_u64.push([r[0], r[1], 1]);
            }
            for _ in w.drained.len()..SEG {
                merge_u64.push([PAD_U64, 0, 0]);
            }
        }
        if tamper == Tamper::MergeSentinel {
            let row = merge_u64
                .iter_mut()
                .find(|r| r[2] == 0)
                .expect("MergeSentinel tamper needs a sentinel row");
            row[0] = 12_345;
        }

        let mut msort_u64: Vec<[u64; 2]> = merge_u64.iter().map(|r| [r[0], r[1]]).collect();
        msort_u64.sort_by_key(|r| r[0]); // by nk (PAD_U64 sentinels last)

        // ---------- final grouping + ORDER BY witness ----------
        let mut m_run_sum_u64 = vec![0u64; m_total];
        let mut m_res_pad_u64: Vec<[u64; 3]> = vec![[PAD_U64, PAD_U64, PAD_REV]; m_total];

        let mut acc: u128 = 0;
        let mut prev_nk: Option<u64> = None;
        for i in 0..m_total {
            let nk = msort_u64[i][0];
            let s = msort_u64[i][1] as u128;
            if prev_nk == Some(nk) {
                acc += s;
            } else {
                acc = s;
            }
            m_run_sum_u64[i] = acc as u64;

            let next_nk = if i + 1 < m_total {
                msort_u64[i + 1][0]
            } else {
                0
            };
            if next_nk != nk && nk != PAD_U64 {
                let nm = *nk_to_name.get(&nk).unwrap_or(&0);
                m_res_pad_u64[i] = [nk, nm, m_run_sum_u64[i]];
            }
            prev_nk = Some(nk);
        }

        // sort result by revenue desc (as q5_obj)
        let mut groups: Vec<[u64; 3]> = m_res_pad_u64
            .iter()
            .copied()
            .filter(|r| r[0] != PAD_U64)
            .collect();
        groups.sort_by(|a, b| b[2].cmp(&a[2]));
        let mut m_res_sorted_u64: Vec<[u64; 3]> = groups;
        while m_res_sorted_u64.len() < m_total {
            m_res_sorted_u64.push([PAD_U64, PAD_U64, PAD_REV]);
        }

        // ---------- assign region ----------
        layouter.assign_region(
            || "q5 dp witness",
            |mut region| {
                // base tables
                for i in 0..customer.len() {
                    for j in 0..2 {
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

                // ---------- NR materialization (as q5_obj) ----------
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

                    iz_nr_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(nr_rname_u64[i]) - F::from(europe_hash)),
                    )?;
                }

                // NR extra padding rows (as q5_obj)
                for i in nation.len()..nr_total {
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
                }

                for i in 0..nr_total {
                    self.config.perm_nr.q_perm1.enable(&mut region, i)?;
                    self.config.perm_nr.q_perm2.enable(&mut region, i)?;
                }
                Q5Chip::assign_table_f(
                    &mut region,
                    "nr_filt_pad",
                    &self.config.nr_filt_pad,
                    &to_field_rows::<F>(&nr_filt_pad_u64_ext),
                )?;
                Q5Chip::assign_table_f(
                    &mut region,
                    "nr_out_pad",
                    &self.config.nr_out_pad,
                    &to_field_rows::<F>(&nr_out_pad_u64),
                )?;

                // ---------- CO materialization (as q5_obj) ----------
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

                // CO extra padding rows (as q5_obj)
                for i in orders.len()..co_total {
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
                }

                for i in 0..co_total {
                    self.config.perm_co.q_perm1.enable(&mut region, i)?;
                    self.config.perm_co.q_perm2.enable(&mut region, i)?;
                }
                Q5Chip::assign_table_f(
                    &mut region,
                    "co_filt_pad",
                    &self.config.co_filt_pad,
                    &to_field_rows::<F>(&co_filt_pad_u64_ext),
                )?;
                Q5Chip::assign_table_f(
                    &mut region,
                    "co_out_pad",
                    &self.config.co_out_pad,
                    &to_field_rows::<F>(&co_out_pad_u64),
                )?;

                // ---------- LS materialization (as q5_obj) ----------
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
                }

                // ---------- LS join lanes ----------
                // Lane l hosts global pipeline rows [l*lane_rows,
                // (l+1)*lane_rows); cells of REAL join rows are collected in
                // global order for the block-wise part_pad linkage below.
                let mut ls_join_cells: Vec<Vec<AssignedCell<F, F>>> = Vec::with_capacity(join_len);
                for (l, lane) in self.config.lanes.iter().enumerate() {
                    for r in 0..lane_rows {
                        let g = l * lane_rows + r;
                        let mut row_cells = Vec::with_capacity(4);
                        for j in 0..4 {
                            let cell = region.assign_advice(
                                || "ls_join_lane",
                                lane.ls_join[j],
                                r,
                                || Value::known(F::from(ls_join_full[g][j])),
                            )?;
                            row_cells.push(cell);
                        }
                        if g < join_len {
                            ls_join_cells.push(row_cells);
                        }
                    }
                }

                let ls_dis_cells = Q5Chip::assign_table_u64(
                    &mut region,
                    "ls_disjoin",
                    &self.config.ls_disjoin,
                    &ls_dis_u64,
                )?;

                for i in 0..lineitem.len() {
                    self.config.perm_ls.q_perm1.enable(&mut region, i)?;
                    self.config.perm_ls.q_perm2.enable(&mut region, i)?;
                }
                // ls_part_pad row i is copy-linked to lane i/lane_rows offset
                // i%lane_rows for i < J, else to ls_disjoin (the collected
                // cell list is in global order, so the q5_obj helper applies
                // unchanged)
                Q5Chip::assign_part_pad_and_link(
                    &mut region,
                    "ls_part_pad",
                    &self.config.ls_part_pad,
                    &ls_part_pad_f,
                    &ls_join_cells,
                    &ls_dis_cells,
                )?;

                // ---------- key tables (row0 dummy, as q5_obj) ----------
                region.assign_advice(
                    || "co_key0",
                    self.config.co_key,
                    0,
                    || Value::known(F::from(0)),
                )?;
                region.assign_advice(
                    || "co_keyn0",
                    self.config.co_key_next,
                    0,
                    || Value::known(F::from(0)),
                )?;
                region.assign_advice(
                    || "nr_key0",
                    self.config.nr_key,
                    0,
                    || Value::known(F::from(0)),
                )?;
                region.assign_advice(
                    || "nr_keyn0",
                    self.config.nr_key_next,
                    0,
                    || Value::known(F::from(0)),
                )?;

                for i in 0..valid_co_keys.len() {
                    let k = valid_co_keys[i];
                    let kn = if i + 1 < valid_co_keys.len() {
                        valid_co_keys[i + 1]
                    } else {
                        MAX_SENTINEL
                    };
                    region.assign_advice(
                        || "co_key",
                        self.config.co_key,
                        i + 1,
                        || Value::known(F::from(k)),
                    )?;
                    region.assign_advice(
                        || "co_key_next",
                        self.config.co_key_next,
                        i + 1,
                        || Value::known(F::from(kn)),
                    )?;
                }
                for i in 0..valid_nr_keys.len() {
                    let k = valid_nr_keys[i];
                    let kn = if i + 1 < valid_nr_keys.len() {
                        valid_nr_keys[i + 1]
                    } else {
                        MAX_SENTINEL
                    };
                    region.assign_advice(
                        || "nr_key",
                        self.config.nr_key,
                        i + 1,
                        || Value::known(F::from(k)),
                    )?;
                    region.assign_advice(
                        || "nr_key_next",
                        self.config.nr_key_next,
                        i + 1,
                        || Value::known(F::from(kn)),
                    )?;
                }

                // ---------- disjoin emptiness witnesses (as q5_obj) --------
                for i in 0..ls_dis_u64.len() {
                    self.config.q_flagged_lookup.enable(&mut region, i)?;
                    self.config.q_lookup_complex.enable(&mut region, i)?;

                    let ok = ls_dis_u64[i][0];
                    let nk = ls_dis_u64[i][1];
                    let packed = ok * SHIFT_NATION + nk;

                    let co_idx = valid_co_keys.binary_search(&packed);
                    let (in_co, low_co, high_co) = match co_idx {
                        Ok(_) => (1u64, 0u64, MAX_SENTINEL),
                        Err(idx) => (0u64, valid_co_keys[idx - 1], valid_co_keys[idx]),
                    };

                    let nr_idx = valid_nr_keys.binary_search(&nk);
                    let (in_nr, low_nr, high_nr) = match nr_idx {
                        Ok(_) => (1u64, 0u64, MAX_SENTINEL),
                        Err(idx) => (0u64, valid_nr_keys[idx - 1], valid_nr_keys[idx]),
                    };

                    region.assign_advice(
                        || "in_co",
                        self.config.flags_in[0],
                        i,
                        || Value::known(F::from(in_co)),
                    )?;
                    region.assign_advice(
                        || "in_nr",
                        self.config.flags_in[1],
                        i,
                        || Value::known(F::from(in_nr)),
                    )?;

                    region.assign_advice(
                        || "co_low",
                        self.config.range_low_high[0],
                        i,
                        || Value::known(F::from(low_co)),
                    )?;
                    region.assign_advice(
                        || "co_high",
                        self.config.range_low_high[1],
                        i,
                        || Value::known(F::from(high_co)),
                    )?;
                    region.assign_advice(
                        || "nr_low",
                        self.config.range_low_high[2],
                        i,
                        || Value::known(F::from(low_nr)),
                    )?;
                    region.assign_advice(
                        || "nr_high",
                        self.config.range_low_high[3],
                        i,
                        || Value::known(F::from(high_nr)),
                    )?;

                    lt_co_low_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(low_co)),
                        Value::known(F::from(packed)),
                    )?;
                    lt_co_high_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(packed)),
                        Value::known(F::from(high_co)),
                    )?;
                    lt_nr_low_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(low_nr)),
                        Value::known(F::from(nk)),
                    )?;
                    lt_nr_high_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(nk)),
                        Value::known(F::from(high_nr)),
                    )?;
                }

                // ---------- shared lane selectors ----------
                // ALL lane rows carry the membership lookups: pad rows hold
                // the MAX_SENTINEL keys, which both key tables contain, so
                // (unlike q5_obj's join-rows-only enabling) the selector
                // pattern is independent of the true join size.
                for i in 0..lane_rows {
                    self.config.q_join_member.enable(&mut region, i)?;
                    self.config.q_lane_line.enable(&mut region, i)?;
                    self.config.q_drain.enable(&mut region, i)?;
                    // every lane shares the same q_perm1/q_perm2 pair, so
                    // enabling through lane 0 covers them all
                    self.config.lanes[0]
                        .perm_lsort
                        .q_perm1
                        .enable(&mut region, i)?;
                    self.config.lanes[0]
                        .perm_lsort
                        .q_perm2
                        .enable(&mut region, i)?;
                }
                self.config.q_lane_first.enable(&mut region, 0)?;
                for i in 1..lane_rows {
                    self.config.q_lane_accu.enable(&mut region, i)?;
                }

                // ---------- per-lane aggregation ----------
                for (lane, w) in self.config.lanes.iter().zip(lane_wit.iter()) {
                    for i in 0..lane_rows {
                        for j in 0..4 {
                            region.assign_advice(
                                || "ls_sorted_lane",
                                lane.ls_sorted[j],
                                i,
                                || Value::known(F::from(w.sorted[i][j])),
                            )?;
                        }
                        region.assign_advice(
                            || "lane_line_rev",
                            lane.line_rev,
                            i,
                            || Value::known(F::from(w.line_rev[i])),
                        )?;
                        region.assign_advice(
                            || "lane_run_sum",
                            lane.run_sum,
                            i,
                            || Value::known(F::from(w.run_sum[i])),
                        )?;
                        region.assign_advice(
                            || "lane_res_nk",
                            lane.res_pad[0],
                            i,
                            || Value::known(F::from(w.res_pad[i][0])),
                        )?;
                        region.assign_advice(
                            || "lane_res_sum",
                            lane.res_pad[1],
                            i,
                            || Value::known(F::from(w.res_pad[i][1])),
                        )?;
                    }
                    // sentinel row for same_next (as q5_obj)
                    for j in 0..4 {
                        region.assign_advice(
                            || "ls_sorted_lane_sentinel",
                            lane.ls_sorted[j],
                            lane_rows,
                            || Value::known(F::ZERO),
                        )?;
                    }

                    // same_prev / same_next (lane-internal rotations only)
                    let iz_prev_chip = IsZeroChip::construct(lane.iz_same_prev.clone());
                    let iz_next_chip = IsZeroChip::construct(lane.iz_same_next.clone());
                    for i in 1..lane_rows {
                        let diff = F::from(w.sorted[i][1]) - F::from(w.sorted[i - 1][1]);
                        iz_prev_chip.assign(&mut region, i, Value::known(diff))?;
                    }
                    for i in 0..lane_rows {
                        let next_nk = if i + 1 < lane_rows {
                            w.sorted[i + 1][1]
                        } else {
                            0u64
                        };
                        let diff = F::from(next_nk) - F::from(w.sorted[i][1]);
                        iz_next_chip.assign(&mut region, i, Value::known(diff))?;
                    }
                }

                // ---------- merge region ----------
                for i in 0..m_total {
                    self.config.q_merge.enable(&mut region, i)?;
                    self.config.perm_merge.q_perm1.enable(&mut region, i)?;
                    self.config.perm_merge.q_perm2.enable(&mut region, i)?;
                    self.config.q_m_line.enable(&mut region, i)?;
                    self.config.q_m_res_lookup.enable(&mut region, i)?;
                }
                self.config.q_m_first.enable(&mut region, 0)?;
                for i in 1..m_total {
                    self.config.q_m_accu.enable(&mut region, i)?;
                }

                // fixed segment markers: lane l owns rows [l*SEG, (l+1)*SEG)
                for (l, lane) in self.config.lanes.iter().enumerate() {
                    for i in 0..m_total {
                        let v = if i / SEG == l { F::ONE } else { F::ZERO };
                        region.assign_fixed(|| "seg_marker", lane.seg, i, || Value::known(v))?;
                    }
                }

                for i in 0..m_total {
                    region.assign_advice(
                        || "merge_nk",
                        self.config.merge_nk,
                        i,
                        || Value::known(F::from(merge_u64[i][0])),
                    )?;
                    region.assign_advice(
                        || "merge_sum",
                        self.config.merge_sum,
                        i,
                        || Value::known(F::from(merge_u64[i][1])),
                    )?;
                    region.assign_advice(
                        || "merge_real",
                        self.config.merge_real,
                        i,
                        || Value::known(F::from(merge_u64[i][2])),
                    )?;

                    region.assign_advice(
                        || "msort_nk",
                        self.config.msort[0],
                        i,
                        || Value::known(F::from(msort_u64[i][0])),
                    )?;
                    region.assign_advice(
                        || "msort_sum",
                        self.config.msort[1],
                        i,
                        || Value::known(F::from(msort_u64[i][1])),
                    )?;

                    region.assign_advice(
                        || "m_run_sum",
                        self.config.m_run_sum,
                        i,
                        || Value::known(F::from(m_run_sum_u64[i])),
                    )?;

                    for j in 0..3 {
                        region.assign_advice(
                            || "m_res_pad",
                            self.config.m_res_pad[j],
                            i,
                            || Value::known(F::from(m_res_pad_u64[i][j])),
                        )?;
                        region.assign_advice(
                            || "m_res_sorted",
                            self.config.m_res_sorted[j],
                            i,
                            || Value::known(F::from(m_res_sorted_u64[i][j])),
                        )?;
                    }
                }
                // sentinel row for merge same_next
                for j in 0..2 {
                    region.assign_advice(
                        || "msort_sentinel",
                        self.config.msort[j],
                        m_total,
                        || Value::known(F::ZERO),
                    )?;
                }

                for i in 1..m_total {
                    let diff = F::from(msort_u64[i][0]) - F::from(msort_u64[i - 1][0]);
                    iz_m_same_prev_chip.assign(&mut region, i, Value::known(diff))?;
                }
                for i in 0..m_total {
                    let next_nk = if i + 1 < m_total {
                        msort_u64[i + 1][0]
                    } else {
                        0u64
                    };
                    let diff = F::from(next_nk) - F::from(msort_u64[i][0]);
                    iz_m_same_next_chip.assign(&mut region, i, Value::known(diff))?;
                }

                // m_res_pad <-> m_res_sorted permutation and ORDER BY
                for i in 0..m_total {
                    self.config.perm_mres.q_perm1.enable(&mut region, i)?;
                    self.config.perm_mres.q_perm2.enable(&mut region, i)?;
                }
                for i in 0..m_total.saturating_sub(1) {
                    self.config.q_sort_mres.enable(&mut region, i)?;
                    lteq_rev_chip.assign(
                        &mut region,
                        i,
                        &[F::from(m_res_sorted_u64[i + 1][2])],
                        &[F::from(m_res_sorted_u64[i][2])],
                    )?;
                }

                // public output (same as q5_obj)
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

    // padding knobs (as q5_obj; from `bench_queries::q5_pads`)
    pub nr_pad_extra: usize,
    pub co_pad_extra: usize,
    pub ls_pad_extra: usize,

    // lane geometry: num_lanes = lanes_for(n, lane_rows) with
    // n = |ls_join| + ls_pad_extra.  Production uses lane_rows = LANE_ROWS;
    // the MockProver tests shrink it to exercise multi-lane layouts at
    // small k.  Must match the value stamped via `set_config_lanes`.
    pub lane_rows: usize,
    pub num_lanes: usize,

    // MockProver fault injection (negative tests only)
    pub tamper: Tamper,

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
            lane_rows: LANE_ROWS,
            num_lanes: 1,
            tamper: Tamper::None,
            _marker: PhantomData,
        }
    }
}

impl<F: Field + Ord> Circuit<F> for MyCircuit<F> {
    type Config = Q5DpConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        // keep the lane geometry: it is circuit STRUCTURE, not witness
        Self {
            lane_rows: self.lane_rows,
            num_lanes: self.num_lanes,
            ..Self::default()
        }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        Q5DpChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        assert_eq!(
            config.lanes.len(),
            self.num_lanes,
            "configured lane count does not match the circuit: call \
             set_config_lanes(num_lanes) before keygen / MockProver"
        );
        let chip = Q5DpChip::construct(config);

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
            self.lane_rows,
            self.tamper,
        )?;

        chip.expose_public(&mut layouter, out_cell, 0)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DP lane run
//
// One code path for the `#[ignore]`d `tests::test_dp_lanes` below and for
// `src/bin/dp_lane_bench.rs`.  Nothing here is measurement policy beyond
// "keys once, then `reps` proofs": the geometry is exactly what the test used
// to compute inline.
// ---------------------------------------------------------------------------

/// Degree Q5's DP lane circuit is proved at. PINNED: the released capacity is
/// absorbed by lanes, not by a bigger domain.
pub const DP_LANE_K: u32 = 16;

/// k = 16 leaves this many rows to the range-chip loads before the main region
/// (7 u8 tables of 256 rows each).
const CHIP_LOAD_ROWS: usize = 7 * 256;

/// Build the Q5 DP lane circuit and the plan describing it, or explain why the
/// released capacity cannot be built.
///
/// `Err` covers every way this configuration can be rejected BEFORE proving:
/// the lane cap and the structural fit of the tallest column group at
/// [`DP_LANE_K`]. Both are properties of the request, so a sweep reports them
/// and continues.
///
/// Stamps the lane count for the next `configure` on this thread, so callers
/// must keygen / prove on the thread that called this.
fn try_dp_lane_setup(
    privacy: crate::bench_queries::Privacy,
) -> Result<
    (
        MyCircuit<halo2curves::pasta::Fp>,
        crate::dp_lane::DpLanePlan,
    ),
    String,
> {
    // Tables AND pads from bench_queries: one source of truth with the sweep
    // and the DP release.
    let input = crate::bench_queries::tpch_inputs("q5", privacy);
    let crate::bench_queries::TpchInput::Q5 {
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
    } = input
    else {
        unreachable!("tpch_inputs(\"q5\") returns Q5")
    };

    // lane count from the released capacity (public post-processing).
    // `q5_ls_true` is the same `q5_derive(.., 0, 0)` the release was calibrated
    // to, memoized so a sweep derives it once.
    let ls_true = crate::bench_queries::q5_ls_true();
    let n = ls_true + ls_pad_extra;
    let c = try_lanes_for_capacity(n)?;

    // k is FIXED at 16: every column group must fit under the 7 u8-range chip
    // loads plus blinding slack
    let tallest = lineitem
        .len() // ls_mat / ls_part_pad, the tallest FIXED group
        .max(orders.len() + co_pad_extra)
        .max(nation.len() + nr_pad_extra)
        .max(LANE_ROWS + 1) // laned pipeline + sentinel row
        .max(c * SEG + 1); // merge region + sentinel row
    if CHIP_LOAD_ROWS + tallest + 64 > 1usize << DP_LANE_K {
        return Err(format!(
            "tallest column group ({} rows) does not fit k={} ({} rows, of which {} go to \
             the u8 range-table loads and 64 to blinding)",
            tallest,
            DP_LANE_K,
            1usize << DP_LANE_K,
            CHIP_LOAD_ROWS
        ));
    }

    // `c` is in 1..=MAX_LANES by the check above, so this is an invariant.
    set_config_lanes(c);
    let circuit = MyCircuit::<halo2curves::pasta::Fp> {
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
        lane_rows: LANE_ROWS,
        num_lanes: c,
        tamper: Tamper::None,
        _marker: PhantomData,
    };
    let plan = crate::dp_lane::DpLanePlan {
        query: "q5".to_string(),
        dataset: "tpch-60K".to_string(),
        k: DP_LANE_K,
        lane_rows: LANE_ROWS,
        lanes: vec![c],
        capacity: vec![n],
        true_size: vec![ls_true],
        pads: vec![nr_pad_extra, co_pad_extra, ls_pad_extra],
    };
    Ok((circuit, plan))
}

/// [`try_dp_lane_setup`] for callers that treat an infeasible release as fatal.
fn dp_lane_setup(
    privacy: crate::bench_queries::Privacy,
) -> (
    MyCircuit<halo2curves::pasta::Fp>,
    crate::dp_lane::DpLanePlan,
) {
    try_dp_lane_setup(privacy).unwrap_or_else(|e| panic!("{}", e))
}

/// Geometry only, `Err` when this release cannot be built: no SRS is read, no
/// key is built, nothing is proved. Use this from a sweep, which must report an
/// infeasible cell and carry on.
pub fn try_plan_dp_lanes(
    privacy: crate::bench_queries::Privacy,
) -> Result<crate::dp_lane::DpLanePlan, String> {
    try_dp_lane_setup(privacy).map(|(_, plan)| plan)
}

/// Geometry only: no SRS is read, no key is built, nothing is proved.
/// Panics on an infeasible release; see [`try_plan_dp_lanes`].
pub fn plan_dp_lanes(privacy: crate::bench_queries::Privacy) -> crate::dp_lane::DpLanePlan {
    dp_lane_setup(privacy).1
}

/// Real IPA proving at k = 16 with the DP release hosted in lanes.
///
/// The verifying and proving keys are built ONCE, outside the timed region;
/// then `reps` proofs are generated and every one of them is verified.
/// `proof_path` is `Some` only for callers that want the last proof on disk
/// (the test keeps writing it, `dp_lane_bench` does not).
pub fn run_dp_lanes(
    privacy: crate::bench_queries::Privacy,
    reps: usize,
    proof_path: Option<&str>,
) -> crate::dp_lane::DpLaneRun {
    use halo2_proofs::poly::{
        commitment::Params,
        ipa::{
            commitment::{IPACommitmentScheme, ParamsIPA},
            multiopen::ProverIPA,
            strategy::SingleStrategy,
        },
        VerificationStrategy,
    };
    use halo2_proofs::transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    };
    use halo2curves::pasta::{vesta, EqAffine, Fp};
    use std::time::Instant;

    assert!(reps >= 1, "reps must be at least 1");
    let (circuit, plan) = dp_lane_setup(privacy);

    let params_path = crate::paths::param_file(DP_LANE_K);
    let mut fd = std::fs::File::open(&params_path).unwrap_or_else(|e| {
        panic!(
            "open {}: {} -- generate it with `cargo run --release --bin gen_params -- {}`",
            params_path, e, DP_LANE_K
        )
    });
    let params = ParamsIPA::<vesta::Affine>::read(&mut fd).expect("read params");

    let t = Instant::now();
    let vk = keygen_vk(&params, &circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &circuit).expect("keygen_pk should not fail");
    let keygen_s = t.elapsed().as_secs_f64();

    // Rebuilding the circuit per repetition (the witness tables are moved into
    // it) is untimed; only `create_proof` is.
    let mut rebuild = {
        let mut next = Some(circuit);
        move || match next.take() {
            Some(c) => c,
            None => dp_lane_setup(privacy).0,
        }
    };

    let public_input: Vec<Fp> = vec![Fp::from(1u64)];
    let mut prove_s = Vec::with_capacity(reps);
    let mut verify_total = 0.0;
    let mut proof_bytes = 0usize;
    let mut last_proof = Vec::new();

    for _ in 0..reps {
        let circuit = rebuild();
        let mut transcript = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);
        let t = Instant::now();
        create_proof::<IPACommitmentScheme<_>, ProverIPA<_>, _, _, _, _>(
            &params,
            &pk,
            &[circuit],
            &[&[&public_input[..]]],
            &mut rand::rngs::OsRng,
            &mut transcript,
        )
        .expect("proof generation should not fail");
        prove_s.push(t.elapsed().as_secs_f64());
        let proof = transcript.finalize();
        proof_bytes = proof.len();

        let strategy = SingleStrategy::new(&params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        let t = Instant::now();
        assert!(
            verify_proof(
                &params,
                pk.get_vk(),
                strategy,
                &[&[&public_input[..]]],
                &mut transcript
            )
            .is_ok(),
            "proof verification failed"
        );
        verify_total += t.elapsed().as_secs_f64();
        last_proof = proof;
    }

    if let Some(p) = proof_path {
        use std::io::Write;
        std::fs::File::create(std::path::Path::new(p))
            .expect("Failed to create proof file")
            .write_all(&last_proof)
            .expect("Failed to write proof");
    }

    crate::dp_lane::DpLaneRun {
        plan,
        keygen_s,
        verify_s: verify_total / prove_s.len() as f64,
        prove_s,
        proof_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::{lanes_for, set_config_lanes, MyCircuit, Tamper, LANE_ROWS, MAX_LANES};

    use halo2_proofs::dev::MockProver;
    use halo2curves::pasta::Fp;

    use std::marker::PhantomData;
    use std::time::Instant;

    /// Tiny synthetic Q5 instance for the MockProver tests.
    ///
    /// Nations 0..2 are EUROPE, nation 3 is not; orders o10/o20/o30 are in
    /// the date window, o40 is out, o50 belongs to the non-EUROPE customer.
    /// Exactly 6 lineitem rows join (o10/s1 x2 -> nation 0, o20/s2 x3 ->
    /// nation 1, o30/s3 -> nation 2) and 3 land in the disjoin (window fail,
    /// non-EUROPE, customer/supplier nation mismatch).
    #[allow(clippy::type_complexity)]
    fn synthetic_tables() -> (
        Vec<Vec<u64>>, // customer
        Vec<Vec<u64>>, // orders
        Vec<Vec<u64>>, // lineitem
        Vec<Vec<u64>>, // supplier
        Vec<Vec<u64>>, // nation
        Vec<Vec<u64>>, // region
        u64,           // europe_hash
        u64,           // start_ts
        u64,           // end_ts
    ) {
        let eur = 7u64;
        let region = vec![vec![0, eur], vec![1, 8]];
        let nation = vec![
            vec![0, 101, 0],
            vec![1, 102, 0],
            vec![2, 103, 0],
            vec![3, 104, 1],
        ];
        let supplier = vec![vec![1, 0], vec![2, 1], vec![3, 2], vec![4, 3]];
        let customer = vec![vec![1, 0], vec![2, 1], vec![3, 2], vec![4, 3]];
        let (start_ts, end_ts) = (100u64, 200u64);
        let orders = vec![
            vec![150, 1, 10], // in window, nation 0
            vec![150, 2, 20], // in window, nation 1
            vec![150, 3, 30], // in window, nation 2
            vec![50, 1, 40],  // out of window
            vec![150, 4, 50], // non-EUROPE customer
        ];
        let lineitem = vec![
            vec![10, 1, 10, 100], // join, nation 0, rev 9000
            vec![10, 1, 20, 0],   // join, nation 0, rev 20000
            vec![20, 2, 5, 0],    // join, nation 1, rev 5000
            vec![20, 2, 7, 0],    // join, nation 1, rev 7000
            vec![20, 2, 3, 0],    // join, nation 1, rev 3000 (lands in lane 1)
            vec![30, 3, 4, 0],    // join, nation 2, rev 4000 (lands in lane 1)
            vec![40, 1, 6, 0],    // disjoin: order out of window
            vec![50, 4, 6, 0],    // disjoin: non-EUROPE nation
            vec![10, 2, 6, 0],    // disjoin: customer/supplier nation mismatch
        ];
        (
            customer, orders, lineitem, supplier, nation, region, eur, start_ts, end_ts,
        )
    }

    fn three_lane_circuit(tamper: Tamper) -> (MyCircuit<Fp>, usize) {
        let (customer, orders, lineitem, supplier, nation, region, eur, start_ts, end_ts) =
            synthetic_tables();

        // sanity: the synthetic join really has 6 rows, so with 6 pad rows
        // and 4-row lanes the pipeline spans 3 lanes and the nation-1 group
        // spans the lane 0 / lane 1 boundary
        let d = crate::sql::q5_obj::q5_derive(
            &customer, &orders, &lineitem, &supplier, &nation, &region, eur, start_ts, end_ts, 0, 0,
        );
        assert_eq!(d.ls_join_u64.len(), 6);

        let lane_rows = 4;
        let ls_pad_extra = 6; // n = 6 + 6 = 12
        let c = lanes_for(6 + ls_pad_extra, lane_rows);
        assert_eq!(c, 3);

        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            supplier,
            nation,
            region,
            europe_hash: eur,
            start_ts,
            end_ts,
            nr_pad_extra: 0,
            co_pad_extra: 0,
            ls_pad_extra,
            lane_rows,
            num_lanes: c,
            tamper,
            _marker: PhantomData,
        };
        (circuit, c)
    }

    #[test]
    fn lane_math() {
        assert_eq!(lanes_for(1, LANE_ROWS), 1);
        assert_eq!(lanes_for(LANE_ROWS, LANE_ROWS), 1);
        assert_eq!(lanes_for(LANE_ROWS + 1, LANE_ROWS), 2);
        assert_eq!(lanes_for(13 * LANE_ROWS, LANE_ROWS), 13);
        assert_eq!(lanes_for(13 * LANE_ROWS - 1, LANE_ROWS), 13);
        // released capacities at the small-epsilon end (~770K) stay capped
        assert!(lanes_for(770_000, LANE_ROWS) <= MAX_LANES);
        // degenerate release still gets one lane
        assert_eq!(lanes_for(0, LANE_ROWS), 1);
    }

    /// c = 3 lanes at k = 11: groups spanning lanes, sentinel/merge logic,
    /// attach-name and ORDER BY over the merge region.
    #[test]
    fn mock_three_lanes() {
        let (circuit, c) = three_lane_circuit(Tamper::None);
        set_config_lanes(c);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(1u64)]]).unwrap();
        prover.assert_satisfied();
    }

    /// Corrupting one emitted lane partial sum must break the lane emit gate
    /// and the drain shuffle.
    #[test]
    fn mock_reject_tampered_lane_partial_sum() {
        let (circuit, c) = three_lane_circuit(Tamper::LanePartialSum);
        set_config_lanes(c);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(1u64)]]).unwrap();
        assert!(
            prover.verify().is_err(),
            "tampered lane partial sum must not verify"
        );
    }

    /// Corrupting one merge sentinel must break the sentinel gate and the
    /// drain shuffle.
    #[test]
    fn mock_reject_tampered_merge_sentinel() {
        let (circuit, c) = three_lane_circuit(Tamper::MergeSentinel);
        set_config_lanes(c);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(1u64)]]).unwrap();
        assert!(
            prover.verify().is_err(),
            "tampered merge sentinel must not verify"
        );
    }

    /// c = 1 degenerates to q5_obj's single-group layout: both circuits must
    /// accept the same input (and both expose the same public output [1]).
    #[test]
    fn mock_single_lane_matches_q5_obj() {
        let (customer, orders, lineitem, supplier, nation, region, eur, start_ts, end_ts) =
            synthetic_tables();
        let public: Vec<Fp> = vec![Fp::from(1u64)];

        let baseline = crate::sql::q5_obj::MyCircuit::<Fp> {
            customer: customer.clone(),
            orders: orders.clone(),
            lineitem: lineitem.clone(),
            supplier: supplier.clone(),
            nation: nation.clone(),
            region: region.clone(),
            europe_hash: eur,
            start_ts,
            end_ts,
            nr_pad_extra: 0,
            co_pad_extra: 0,
            ls_pad_extra: 3,
            _marker: PhantomData,
        };
        let prover = MockProver::run(11, &baseline, vec![public.clone()]).unwrap();
        prover.assert_satisfied();

        let lane_rows = 16; // n = 6 + 3 = 9 fits one lane
        let c = lanes_for(6 + 3, lane_rows);
        assert_eq!(c, 1);
        set_config_lanes(c);
        let laned = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            supplier,
            nation,
            region,
            europe_hash: eur,
            start_ts,
            end_ts,
            nr_pad_extra: 0,
            co_pad_extra: 0,
            ls_pad_extra: 3,
            lane_rows,
            num_lanes: c,
            tamper: Tamper::None,
            _marker: PhantomData,
        };
        let prover = MockProver::run(11, &laned, vec![public]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    #[ignore = "real k=16 IPA proving over full TPC-H; run explicitly"]
    fn test_dp_lanes() {
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

        let proof_path = crate::paths::proof_file("proof_q5_dp_lanes");
        let t_total = Instant::now();
        let run = super::run_dp_lanes(privacy, 1, Some(&proof_path));
        let p = &run.plan;
        println!(
            "[q5 dp-lanes] privacy={} pads: nr={} co={} ls={}",
            privacy.label(),
            p.pads[0],
            p.pads[1],
            p.pads[2]
        );
        println!(
            "[q5 dp-lanes] ls_true={} n={} lanes={} lane_rows={} effective_capacity={}",
            p.true_size[0],
            p.capacity[0],
            p.lanes[0],
            p.lane_rows,
            p.lanes[0] * p.lane_rows
        );
        println!("Proof written to: {}", proof_path);
        println!(
            "[q5 dp-lanes] lanes={} keygen {:.2}s prove {:.2}s verify {:.2}s \
             total prove+verify {:?}",
            p.lanes[0],
            run.keygen_s,
            run.prove_mean(),
            run.verify_s,
            t_total.elapsed()
        );
    }
} // end mod tests
