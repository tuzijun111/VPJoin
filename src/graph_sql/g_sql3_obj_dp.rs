use halo2_proofs::plonk::Expression;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::data::graph_data_processing::Edge;

// One shared definition of the PAD / packing conventions, the chips and the
// host-side witness derivation (from g_sql3_obj).
use super::g_sql3_obj::{
    gq3_derive, pack2, AggSumByKeyChip, AggSumByKeyConfig, Field, IndexedViewChip,
    IndexedViewConfig, MapLookupChip, MapLookupConfig, NUM_BYTES, PACK_SHIFT, PAD_U64,
};

use std::collections::BTreeMap;
use std::marker::PhantomData;

/// Rows the u8 range-table `load` regions may claim ahead of the witness
/// region. Five distinct u8 columns are loaded: the four shared ones (the sort
/// chip of each indexed view, and AggSumByKey's two sort chips) plus the single
/// column every lane's four Lt chips share. Each `load` writes 256 fixed rows.
/// Reserving all five is conservative: the floor planner places regions per
/// column, so in practice they overlap the witness region.
pub const PREAMBLE_ROWS: usize = 5 * 256;

/// Blinding rows halo2 keeps at the bottom of every advice column.
pub const BLINDING_SLACK: usize = 64;

/// Default base degree: the Revealing-Join-Size degree of gq3 on lastfm.
/// facebook and wiki prove at k = 22; pass their degree to [`lane_rows_for`].
pub const BASE_DEGREE: u32 = 18;

/// Usable rows per lane at the default base degree.
pub const LANE_ROWS: usize = (1usize << BASE_DEGREE) - PREAMBLE_ROWS - BLINDING_SLACK;

/// Public structural cap on the lane count. GQ3/GQ4 releases at eps = 0.01 run
/// into the dozens of lanes (gq3-lastfm needs 44 at k = 18, gq4-lastfm needs
/// about 144), so this is deliberately far above q5_obj_dp's cap of 16.
pub const MAX_LANES: usize = 64;

/// Usable rows per lane at circuit degree `k`.
pub fn lane_rows_for(k: u32) -> usize {
    let rows = 1usize << k;
    assert!(
        rows > PREAMBLE_ROWS + BLINDING_SLACK,
        "degree k={} is too small to host a lane",
        k
    );
    rows - PREAMBLE_ROWS - BLINDING_SLACK
}

/// c = ceil(n / lane_rows): the PUBLIC lane count hosting an n-row pipeline.
pub fn lanes_for(n: usize, lane_rows: usize) -> usize {
    assert!(lane_rows > 0, "lane_rows must be positive");
    ((n + lane_rows - 1) / lane_rows).max(1)
}

/// Lane count for a released Bag1 capacity at a caller-supplied base degree,
/// `Err` when the release needs more lanes than the structural cap allows.
///
/// A release that does not fit is an INFEASIBLE REQUEST, not a bug: a sweep
/// must be able to report the cell and move on, so this returns the same
/// information [`lanes_for_capacity`] aborts with.
pub fn try_lanes_for_capacity(capacity: usize, base_degree: u32) -> Result<usize, String> {
    let lane_rows = lane_rows_for(base_degree);
    let c = lanes_for(capacity, lane_rows);
    if c > MAX_LANES {
        return Err(format!(
            "released capacity {} needs {} lanes at base degree k={} ({} usable rows per \
             lane), above MAX_LANES={}",
            capacity, c, base_degree, lane_rows, MAX_LANES
        ));
    }
    Ok(c)
}

/// Lane count for a released Bag1 capacity at a caller-supplied base degree,
/// with the structural cap enforced. Panics on an infeasible capacity; see
/// [`try_lanes_for_capacity`].
pub fn lanes_for_capacity(capacity: usize, base_degree: u32) -> usize {
    try_lanes_for_capacity(capacity, base_degree).unwrap_or_else(|e| panic!("{}", e))
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
    /// Bump one lane row's contribution by 1 (the contrib gate and the lane's
    /// own prefix-sum gates must reject).
    LaneContrib,
    /// Bump one lane total by 1 (the copy constraint into the totals stage and
    /// the totals accumulator must both reject).
    LaneTotal,
    /// Break the pad-row convention on the last rounding-up row (the
    /// "bag1 real + key" gate and the r1 membership lookup must reject).
    PadRow,
}

/// One lane: a full structural replica of g_sql3_obj's Bag1 column group.
/// `out` is NOT replicated; it lives once in the cross-lane totals stage.
#[derive(Clone, Debug)]
pub struct LaneConfig<F: Field + Ord> {
    // Bag1 rows: A->B->C, plus the source-edge ids and per-group indices
    t12_a: Column<Advice>,
    t12_b: Column<Advice>,
    t12_c: Column<Advice>,
    t12_i_r1: Column<Advice>,
    t12_j_r2: Column<Advice>,
    t12_r1_eid: Column<Advice>,
    t12_r2_eid: Column<Advice>,
    t12_real: Column<Advice>,

    // membership-or-gap probe of the closing-edge message
    msg_lookup: MapLookupConfig<F>,

    // ordering predicates A<B and B<C
    lt_ab: LtConfig<F, NUM_BYTES>,
    lt_bc: LtConfig<F, NUM_BYTES>,

    // per-row contribution and the lane-local prefix sum
    contrib: Column<Advice>,
    run_sum: Column<Advice>,
}

#[derive(Clone, Debug)]
pub struct Gq3DpConfig<F: Field + Ord> {
    instance: Column<Instance>,

    // ---------------- fixed-height sections (shared, never laned) ----------
    // indexed views of the edge multiset; the table side of the r1/r2/r3
    // membership lookups. Height tracks |E|, which is public in every regime.
    in_by_dst: IndexedViewConfig<F>,  // key=dst, val=src
    out_by_src: IndexedViewConfig<F>, // key=src, val=dst

    // Bag2 rows: edge C->A (closer)
    t3_c: Column<Advice>,
    t3_a: Column<Advice>,
    t3_j_r3: Column<Advice>,
    t3_r3_eid: Column<Advice>,
    t3_real: Column<Advice>,
    q_t3_lookup: Selector,
    q_t3_msg_in: Selector,

    // message: msg_key=pack2(A,C), msg_val=count. Its map table is the table
    // side of every lane's msg_lookup, reached only through a lookup, so lanes
    // and this group stay fully decoupled.
    agg_msg: AggSumByKeyConfig<F>,

    // ---------------- lanes (shared selectors, per-lane columns) ----------
    // All lanes are active on the SAME rows 0..lane_rows, so one selector of
    // each kind serves every lane; only the columns replicate.
    lanes: Vec<LaneConfig<F>>,
    lane_u8: Column<Fixed>, // one u8 range table for all 4c lane Lt chips
    q_t12_lookup: Selector, // r1/r2 membership lookups (complex)
    q_t12_key: Selector,
    q_flag: Selector,
    q_complex: Selector, // msg member / msg gap lookups (complex)
    q_order: Selector,
    q_contrib: Selector,
    q_sum0: Selector, // lane-local prefix sum, row 0 of every lane
    q_sum: Selector,  // lane-local prefix sum, rows 1..lane_rows

    // ---------------- cross-lane totals (the only stitch) ------------------
    lane_tot: Column<Advice>, // row l = lane l's total, by copy constraint
    tot_run: Column<Advice>,  // prefix sum of lane_tot over rows 0..c
    q_tot0: Selector,
    q_tot: Selector,

    out: Column<Advice>,
    q_out: Selector,
}

#[derive(Clone, Debug)]
pub struct Gq3DpChip<F: Field + Ord> {
    cfg: Gq3DpConfig<F>,
}

impl<F: Field + Ord> Gq3DpChip<F> {
    pub fn construct(cfg: Gq3DpConfig<F>) -> Self {
        Self { cfg }
    }

    /// One full structural replica of the Bag1 column group. Every lane gets
    /// the same gates and the same lookups; only the columns differ.
    #[allow(clippy::too_many_arguments)]
    fn configure_lane(
        meta: &mut ConstraintSystem<F>,
        in_by_dst: &IndexedViewConfig<F>,
        out_by_src: &IndexedViewConfig<F>,
        agg_msg: &AggSumByKeyConfig<F>,
        lane_u8: Column<Fixed>,
        q_t12_lookup: Selector,
        q_t12_key: Selector,
        q_flag: Selector,
        q_complex: Selector,
        q_order: Selector,
        q_contrib: Selector,
        q_sum0: Selector,
        q_sum: Selector,
    ) -> LaneConfig<F> {
        let t12_a = meta.advice_column();
        let t12_b = meta.advice_column();
        let t12_c = meta.advice_column();
        let t12_i_r1 = meta.advice_column();
        let t12_j_r2 = meta.advice_column();
        let t12_r1_eid = meta.advice_column();
        let t12_r2_eid = meta.advice_column();
        let t12_real = meta.advice_column();
        // NOTE: no `enable_equality` on the lane columns. halo2 charges for
        // every column in the permutation argument whether or not a copy
        // constraint ever touches it, so a per-lane column that is only read by
        // gates and lookups must stay out of it: otherwise the permutation cost
        // grows with the lane count for nothing. `run_sum` below is the single
        // lane column that is genuinely copied (into the totals stage).

        // r1 via InByDst: key=B, idx=i_r1 -> val=A, eid=r1_eid.
        // The input side is per lane, the table side is the shared view: a
        // lookup argument is a multiset relation over the whole domain, so
        // adding lane rows against the same table needs no stitching.
        meta.lookup_any("bag1 r1 from in_by_dst", |m| {
            let q = m.query_selector(q_t12_lookup);
            vec![
                (
                    q.clone() * m.query_advice(t12_b, Rotation::cur()),
                    m.query_advice(in_by_dst.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t12_i_r1, Rotation::cur()),
                    m.query_advice(in_by_dst.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t12_a, Rotation::cur()),
                    m.query_advice(in_by_dst.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t12_r1_eid, Rotation::cur()),
                    m.query_advice(in_by_dst.sorted_eid, Rotation::cur()),
                ),
            ]
        });
        // r2 via OutBySrc: key=B, idx=j_r2 -> val=C, eid=r2_eid
        meta.lookup_any("bag1 r2 from out_by_src", |m| {
            let q = m.query_selector(q_t12_lookup);
            vec![
                (
                    q.clone() * m.query_advice(t12_b, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t12_j_r2, Rotation::cur()),
                    m.query_advice(out_by_src.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t12_c, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t12_r2_eid, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_eid, Rotation::cur()),
                ),
            ]
        });

        // Message lookup per Bag1 row: membership when the key is in the map,
        // otherwise a gap proof with val forced to 0.
        let msg_lookup = MapLookupChip::<F>::configure_with(
            meta,
            agg_msg.map_key,
            agg_msg.map_val,
            agg_msg.map_key_next,
            agg_msg.q_map_tbl,
            q_flag,
            q_complex,
            lane_u8,
            lane_u8,
            false, // probe columns are never copied; keep them out of the permutation
        );

        // Bag1 key: msg_lookup.key = pack2(A,C); Bag1 real boolean
        meta.create_gate("bag1 real + key", |m| {
            let q = m.query_selector(q_t12_key);

            let a = m.query_advice(t12_a, Rotation::cur());
            let c = m.query_advice(t12_c, Rotation::cur());
            let key_expr = a * Expression::Constant(F::from(PACK_SHIFT)) + c;

            let real = m.query_advice(t12_real, Rotation::cur());
            let one = Expression::Constant(F::ONE);

            vec![
                q.clone() * real.clone() * (one.clone() - real.clone()),
                q * (m.query_advice(msg_lookup.key, Rotation::cur()) - key_expr),
            ]
        });

        // ordering checks for Bag1: A<B and B<C, enabled only if real=1
        let lt_ab = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            lane_u8,
            |m| m.query_selector(q_order) * m.query_advice(t12_real, Rotation::cur()),
            |m| m.query_advice(t12_a, Rotation::cur()),
            |m| m.query_advice(t12_b, Rotation::cur()),
        );
        let lt_bc = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            lane_u8,
            |m| m.query_selector(q_order) * m.query_advice(t12_real, Rotation::cur()),
            |m| m.query_advice(t12_b, Rotation::cur()),
            |m| m.query_advice(t12_c, Rotation::cur()),
        );

        // contrib = t12_real * msg_val * lt_ab * lt_bc (read by gates only)
        let contrib = meta.advice_column();
        meta.create_gate("contrib gate", |m| {
            let q = m.query_selector(q_contrib);

            let real = m.query_advice(t12_real, Rotation::cur());
            let msgv = m.query_advice(msg_lookup.val, Rotation::cur());
            let ab = lt_ab.is_lt(m, None);
            let bc = lt_bc.is_lt(m, None);

            let outc = m.query_advice(contrib, Rotation::cur());
            vec![q * (outc - real * msgv * ab * bc)]
        });

        // lane-local prefix sum of contrib: q_sum0 fires at THIS lane's row 0,
        // so no accumulator ever crosses a lane boundary.
        let run_sum = meta.advice_column();
        meta.enable_equality(run_sum);
        meta.create_gate("sum0", |m| {
            let q = m.query_selector(q_sum0);
            vec![
                q * (m.query_advice(run_sum, Rotation::cur())
                    - m.query_advice(contrib, Rotation::cur())),
            ]
        });
        meta.create_gate("sum", |m| {
            let q = m.query_selector(q_sum);
            vec![
                q * (m.query_advice(run_sum, Rotation::cur())
                    - (m.query_advice(run_sum, Rotation::prev())
                        + m.query_advice(contrib, Rotation::cur()))),
            ]
        });

        LaneConfig {
            t12_a,
            t12_b,
            t12_c,
            t12_i_r1,
            t12_j_r2,
            t12_r1_eid,
            t12_r2_eid,
            t12_real,
            msg_lookup,
            lt_ab,
            lt_bc,
            contrib,
            run_sum,
        }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> Gq3DpConfig<F> {
        let num_lanes = CONFIG_LANES.with(|l| l.get());
        assert!(
            (1..=MAX_LANES).contains(&num_lanes),
            "configured lane count {} outside 1..={} (call set_config_lanes first)",
            num_lanes,
            MAX_LANES
        );

        let instance = meta.instance_column();
        meta.enable_equality(instance);

        // ---------------- fixed-height sections ----------------
        let in_by_dst = IndexedViewChip::<F>::configure(meta);
        let out_by_src = IndexedViewChip::<F>::configure(meta);

        // Bag2 columns (r3: C->A)
        let t3_c = meta.advice_column();
        let t3_a = meta.advice_column();
        let t3_j_r3 = meta.advice_column();
        let t3_r3_eid = meta.advice_column();
        let t3_real = meta.advice_column();
        for c in [t3_c, t3_a, t3_j_r3, t3_r3_eid, t3_real] {
            meta.enable_equality(c);
        }
        let q_t3_lookup = meta.complex_selector();
        let q_t3_msg_in = meta.selector();

        // Bag2 lookup (r3) via OutBySrc: key=C, idx=j_r3 -> val=A, eid=r3_eid
        meta.lookup_any("bag2 r3 from out_by_src", |m| {
            let q = m.query_selector(q_t3_lookup);
            vec![
                (
                    q.clone() * m.query_advice(t3_c, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t3_j_r3, Rotation::cur()),
                    m.query_advice(out_by_src.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t3_a, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t3_r3_eid, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_eid, Rotation::cur()),
                ),
            ]
        });

        // Message aggregator (single copy: its map table is a lookup table)
        let agg_msg = AggSumByKeyChip::<F>::configure(meta);

        // Tie agg_msg inputs to Bag2 rows:
        // keep = t3_real
        // in_key = keep ? pack2(A,C) : PAD
        // in_val = keep
        meta.create_gate("msg input from bag2 edges", |m| {
            let q = m.query_selector(q_t3_msg_in);

            let a = m.query_advice(t3_a, Rotation::cur());
            let c = m.query_advice(t3_c, Rotation::cur());
            let key_expr = a * Expression::Constant(F::from(PACK_SHIFT)) + c;

            let keep = m.query_advice(t3_real, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let pad = Expression::Constant(F::from(PAD_U64));
            let selected_key = keep.clone() * key_expr + (one - keep.clone()) * pad;

            vec![
                q.clone() * (m.query_advice(agg_msg.in_key, Rotation::cur()) - selected_key),
                q * (m.query_advice(agg_msg.in_val, Rotation::cur()) - keep),
            ]
        });

        // ---------------- lanes ----------------
        // Shared selectors: every lane is active on the same rows, so one
        // selector of each kind drives all c replicas.
        let q_t12_lookup = meta.complex_selector();
        let q_t12_key = meta.selector();
        let q_flag = meta.selector();
        let q_complex = meta.complex_selector();
        let q_order = meta.selector();
        let q_contrib = meta.selector();
        let q_sum0 = meta.selector();
        let q_sum = meta.selector();

        // One u8 range table for every lane Lt chip: `load` costs one 256-row
        // region per distinct u8 column, and that must not grow with c.
        let lane_u8 = meta.fixed_column();

        let lanes: Vec<LaneConfig<F>> = (0..num_lanes)
            .map(|_| {
                Self::configure_lane(
                    meta,
                    &in_by_dst,
                    &out_by_src,
                    &agg_msg,
                    lane_u8,
                    q_t12_lookup,
                    q_t12_key,
                    q_flag,
                    q_complex,
                    q_order,
                    q_contrib,
                    q_sum0,
                    q_sum,
                )
            })
            .collect();

        // ---------------- cross-lane totals ----------------
        // Row l holds lane l's total, copied in from that lane's last run_sum
        // cell; tot_run adds them with a degree-2 accumulator. A flat
        // degree-(1+c) sum gate would blow past cs.degree() = 7 for c > 6.
        let lane_tot = meta.advice_column();
        let tot_run = meta.advice_column();
        meta.enable_equality(lane_tot);
        meta.enable_equality(tot_run);
        let q_tot0 = meta.selector();
        let q_tot = meta.selector();

        meta.create_gate("lane totals first", |m| {
            let q = m.query_selector(q_tot0);
            vec![
                q * (m.query_advice(tot_run, Rotation::cur())
                    - m.query_advice(lane_tot, Rotation::cur())),
            ]
        });
        meta.create_gate("lane totals accu", |m| {
            let q = m.query_selector(q_tot);
            vec![
                q * (m.query_advice(tot_run, Rotation::cur())
                    - (m.query_advice(tot_run, Rotation::prev())
                        + m.query_advice(lane_tot, Rotation::cur()))),
            ]
        });

        // out equals tot_run at the last totals row
        let out = meta.advice_column();
        meta.enable_equality(out);
        let q_out = meta.selector();
        meta.create_gate("out = tot_run", |m| {
            let q = m.query_selector(q_out);
            vec![
                q * (m.query_advice(out, Rotation::cur())
                    - m.query_advice(tot_run, Rotation::cur())),
            ]
        });

        Gq3DpConfig {
            instance,
            in_by_dst,
            out_by_src,
            t3_c,
            t3_a,
            t3_j_r3,
            t3_r3_eid,
            t3_real,
            q_t3_lookup,
            q_t3_msg_in,
            agg_msg,
            lanes,
            lane_u8,
            q_t12_lookup,
            q_t12_key,
            q_flag,
            q_complex,
            q_order,
            q_contrib,
            q_sum0,
            q_sum,
            lane_tot,
            tot_run,
            q_tot0,
            q_tot,
            out,
            q_out,
        }
    }

    /// Assign one lane's block of the padded Bag1 pipeline.
    ///
    /// `base` is the global index of this lane's row 0, so lane l covers the
    /// padded rows `[l*lane_rows, (l+1)*lane_rows)`. Rows past `t12.len()` are
    /// pad rows and are witnessed with exactly the pad convention g_sql3_obj
    /// uses: all-zero tuple, key 0, in_set 1, val 0, contrib 0. Map row 0 is
    /// (0,0) and view row 0 is (0,0,0), so a pad row satisfies every lookup.
    ///
    /// Returns the lane's last `run_sum` cell (the value the totals stage
    /// copies) together with that total.
    #[allow(clippy::too_many_arguments)]
    fn assign_lane(
        lane: &LaneConfig<F>,
        region: &mut Region<'_, F>,
        lane_rows: usize,
        base: usize,
        t12: &[(u64, u64, u64, u64, u64, u64, u64)],
        msg_map: &BTreeMap<u64, u64>,
        keys: &[u64],
        tamper: Tamper,
        pad_row_global: usize,
    ) -> Result<(AssignedCell<F, F>, u64), Error> {
        let lt_ab_chip = LtChip::<F, NUM_BYTES>::construct(lane.lt_ab);
        let lt_bc_chip = LtChip::<F, NUM_BYTES>::construct(lane.lt_bc);
        let lt_low_chip = LtChip::<F, NUM_BYTES>::construct(lane.msg_lookup.lt_low);
        let lt_high_chip = LtChip::<F, NUM_BYTES>::construct(lane.msg_lookup.lt_high);

        let real12 = t12.len();
        let mut running: u64 = 0;
        let mut last_cell: Option<AssignedCell<F, F>> = None;

        for r in 0..lane_rows {
            let i = base + r;

            let (a, b, c, i_r1, j_r2, r1_eid, r2_eid, real) = if i < real12 {
                let (a, b, c, i_r1, j_r2, r1_eid, r2_eid) = t12[i];
                (a, b, c, i_r1, j_r2, r1_eid, r2_eid, 1u64)
            } else {
                (0, 0, 0, 0, 0, 0, 0, 0u64)
            };

            // Tamper::PadRow breaks the pad convention on the very last
            // rounding-up row: t12_a moves but msg key stays 0.
            let a_written = if tamper == Tamper::PadRow && i == pad_row_global {
                7u64
            } else {
                a
            };

            region.assign_advice(
                || "t12_a",
                lane.t12_a,
                r,
                || Value::known(F::from(a_written)),
            )?;
            region.assign_advice(|| "t12_b", lane.t12_b, r, || Value::known(F::from(b)))?;
            region.assign_advice(|| "t12_c", lane.t12_c, r, || Value::known(F::from(c)))?;
            region.assign_advice(
                || "t12_i_r1",
                lane.t12_i_r1,
                r,
                || Value::known(F::from(i_r1)),
            )?;
            region.assign_advice(
                || "t12_j_r2",
                lane.t12_j_r2,
                r,
                || Value::known(F::from(j_r2)),
            )?;
            region.assign_advice(
                || "t12_r1_eid",
                lane.t12_r1_eid,
                r,
                || Value::known(F::from(r1_eid)),
            )?;
            region.assign_advice(
                || "t12_r2_eid",
                lane.t12_r2_eid,
                r,
                || Value::known(F::from(r2_eid)),
            )?;
            region.assign_advice(
                || "t12_real",
                lane.t12_real,
                r,
                || Value::known(F::from(real)),
            )?;

            // LT witnesses (constraints disabled when real=0)
            lt_ab_chip.assign(
                region,
                r,
                Value::known(F::from(a)),
                Value::known(F::from(b)),
            )?;
            lt_bc_chip.assign(
                region,
                r,
                Value::known(F::from(b)),
                Value::known(F::from(c)),
            )?;

            // message lookup witness
            let key = if real == 1 { pack2(a, c) } else { 0 };
            let inside = if key == 0 || msg_map.contains_key(&key) {
                1u64
            } else {
                0u64
            };
            let val = if inside == 1 {
                *msg_map.get(&key).unwrap_or(&0)
            } else {
                0u64
            };

            let (low, high) = if inside == 0 {
                match keys.binary_search(&key) {
                    Ok(_) => (0u64, PAD_U64),
                    Err(pos) => {
                        let lo = keys[pos.saturating_sub(1)];
                        let hi = keys[pos];
                        (lo, hi)
                    }
                }
            } else {
                (0u64, PAD_U64)
            };

            region.assign_advice(
                || "msg_key",
                lane.msg_lookup.key,
                r,
                || Value::known(F::from(key)),
            )?;
            region.assign_advice(
                || "msg_in_set",
                lane.msg_lookup.in_set,
                r,
                || Value::known(F::from(inside)),
            )?;
            region.assign_advice(
                || "msg_low",
                lane.msg_lookup.low,
                r,
                || Value::known(F::from(low)),
            )?;
            region.assign_advice(
                || "msg_high",
                lane.msg_lookup.high,
                r,
                || Value::known(F::from(high)),
            )?;
            region.assign_advice(
                || "msg_val",
                lane.msg_lookup.val,
                r,
                || Value::known(F::from(val)),
            )?;

            lt_low_chip.assign(
                region,
                r,
                Value::known(F::from(low)),
                Value::known(F::from(key)),
            )?;
            lt_high_chip.assign(
                region,
                r,
                Value::known(F::from(key)),
                Value::known(F::from(high)),
            )?;

            let ab = if a < b { 1u64 } else { 0u64 };
            let bc = if b < c { 1u64 } else { 0u64 };
            let contrib_u64 = real * val * ab * bc;

            // Tamper::LaneContrib perturbs the cell only; run_sum keeps the
            // honest value, so the contrib gate and the sum gate both fail.
            let contrib_written = if tamper == Tamper::LaneContrib && base == 0 && r == 0 {
                contrib_u64 + 1
            } else {
                contrib_u64
            };
            region.assign_advice(
                || "contrib",
                lane.contrib,
                r,
                || Value::known(F::from(contrib_written)),
            )?;

            running = if r == 0 {
                contrib_u64
            } else {
                running.wrapping_add(contrib_u64)
            };
            last_cell = Some(region.assign_advice(
                || "run_sum",
                lane.run_sum,
                r,
                || Value::known(F::from(running)),
            )?);
        }

        Ok((
            last_cell.expect("lane_rows > 0 guarantees a last row"),
            running,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        edges: &[Edge],
        bag1_pad_extra: usize,
        bag2_pad_extra: usize,
        lane_rows: usize,
        tamper: Tamper,
    ) -> Result<AssignedCell<F, F>, Error> {
        let cfg = self.cfg.clone();
        let num_lanes = cfg.lanes.len();
        assert!(lane_rows > 0, "lane_rows must be positive");

        // Load all LT tables used. The shared chips keep a u8 column each; all
        // 4c lane chips share `lane_u8`, so this is 5 load regions for any c.
        LtChip::<F, NUM_BYTES>::construct(cfg.in_by_dst.lt_key).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.out_by_src.lt_key).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.agg_msg.lt_key).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.agg_msg.lt_out_key).load(layouter)?;
        debug_assert_eq!(cfg.lanes[0].lt_ab.u8, cfg.lane_u8);
        LtChip::<F, NUM_BYTES>::construct(cfg.lanes[0].lt_ab).load(layouter)?;

        let in_view_chip = IndexedViewChip::<F>::construct(cfg.in_by_dst.clone());
        let out_view_chip = IndexedViewChip::<F>::construct(cfg.out_by_src.clone());
        let agg_msg_chip = AggSumByKeyChip::<F>::construct(cfg.agg_msg.clone());

        layouter.assign_region(
            || "triangle_path_closer dp-lanes witness",
            |mut region| {
                // -------------------
                // Host-side derivation, shared verbatim with g_sql3_obj
                // -------------------
                let derived = gq3_derive(edges);
                let n_base = derived.in_rows.len();

                in_view_chip.assign(&mut region, n_base, &derived.in_rows)?;
                out_view_chip.assign(&mut region, n_base, &derived.out_rows)?;

                // -------------------
                // Bag2 + message aggregation: fixed-height under Privacy::Dp
                // (GQ3's Bag2 IS the Edge relation, whose size is public), and
                // small enough to stay a single group under Privacy::Legacy.
                // It occupies its own columns, so its height is independent of
                // `lane_rows`; only the domain 2^k bounds it, and the harness
                // checks that fit alongside the lane geometry.
                // -------------------
                let real3 = derived.t3.len();
                let n3 = std::cmp::max(real3 + bag2_pad_extra, 1);

                let mut agg_in: Vec<(u64, u64)> = vec![(PAD_U64, 0); n3];

                for i in 0..n3 {
                    cfg.q_t3_lookup.enable(&mut region, i)?;
                    cfg.q_t3_msg_in.enable(&mut region, i)?;

                    if i < real3 {
                        let (c, a, j_r3, r3_eid) = derived.t3[i];

                        region.assign_advice(
                            || "t3_c",
                            cfg.t3_c,
                            i,
                            || Value::known(F::from(c)),
                        )?;
                        region.assign_advice(
                            || "t3_a",
                            cfg.t3_a,
                            i,
                            || Value::known(F::from(a)),
                        )?;
                        region.assign_advice(
                            || "t3_j_r3",
                            cfg.t3_j_r3,
                            i,
                            || Value::known(F::from(j_r3)),
                        )?;
                        region.assign_advice(
                            || "t3_r3_eid",
                            cfg.t3_r3_eid,
                            i,
                            || Value::known(F::from(r3_eid)),
                        )?;
                        region.assign_advice(
                            || "t3_real",
                            cfg.t3_real,
                            i,
                            || Value::known(F::ONE),
                        )?;

                        agg_in[i] = (pack2(a, c), 1);
                    } else {
                        // padding
                        region.assign_advice(|| "t3_c0", cfg.t3_c, i, || Value::known(F::ZERO))?;
                        region.assign_advice(|| "t3_a0", cfg.t3_a, i, || Value::known(F::ZERO))?;
                        region.assign_advice(
                            || "t3_j0",
                            cfg.t3_j_r3,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t3_eid0",
                            cfg.t3_r3_eid,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t3_real0",
                            cfg.t3_real,
                            i,
                            || Value::known(F::ZERO),
                        )?;

                        agg_in[i] = (PAD_U64, 0);
                    }
                }

                let _emitted = agg_msg_chip.assign(&mut region, n3, &agg_in)?;

                // -------------------
                // Bag1: the pad-driven pipeline, spread over the lanes
                // -------------------
                let real12 = derived.t12.len();
                let n12 = std::cmp::max(real12 + bag1_pad_extra, 1);
                let capacity = num_lanes * lane_rows;
                assert!(
                    capacity >= n12,
                    "released Bag1 capacity {} exceeds {} lanes of {} rows",
                    n12,
                    num_lanes,
                    lane_rows
                );

                // Shared selectors: enabled once on rows 0..lane_rows, which
                // switches ON every lane at the same time. The lane count is a
                // function of the released capacity only, so nothing here can
                // depend on the true bag size.
                for r in 0..lane_rows {
                    cfg.q_t12_lookup.enable(&mut region, r)?;
                    cfg.q_t12_key.enable(&mut region, r)?;
                    cfg.q_flag.enable(&mut region, r)?;
                    cfg.q_complex.enable(&mut region, r)?;
                    cfg.q_order.enable(&mut region, r)?;
                    cfg.q_contrib.enable(&mut region, r)?;
                    if r == 0 {
                        cfg.q_sum0.enable(&mut region, r)?;
                    } else {
                        cfg.q_sum.enable(&mut region, r)?;
                    }
                }

                let mut lane_cells: Vec<AssignedCell<F, F>> = Vec::with_capacity(num_lanes);
                let mut lane_totals: Vec<u64> = Vec::with_capacity(num_lanes);
                for (l, lane) in cfg.lanes.iter().enumerate() {
                    let (cell, total) = Self::assign_lane(
                        lane,
                        &mut region,
                        lane_rows,
                        l * lane_rows,
                        &derived.t12,
                        &derived.msg_map,
                        &derived.keys,
                        tamper,
                        capacity - 1,
                    )?;
                    lane_cells.push(cell);
                    lane_totals.push(total);
                }

                // -------------------
                // The only cross-lane wiring: c copy constraints plus a
                // degree-2 accumulator over the c lane totals.
                // -------------------
                let mut acc: u64 = 0;
                for l in 0..num_lanes {
                    let tot = lane_totals[l];
                    let written = if tamper == Tamper::LaneTotal && l == 0 {
                        tot + 1
                    } else {
                        tot
                    };
                    let tot_cell = region.assign_advice(
                        || "lane_tot",
                        cfg.lane_tot,
                        l,
                        || Value::known(F::from(written)),
                    )?;
                    region.constrain_equal(lane_cells[l].cell(), tot_cell.cell())?;

                    if l == 0 {
                        cfg.q_tot0.enable(&mut region, l)?;
                        acc = tot;
                    } else {
                        cfg.q_tot.enable(&mut region, l)?;
                        acc = acc.wrapping_add(tot);
                    }
                    region.assign_advice(
                        || "tot_run",
                        cfg.tot_run,
                        l,
                        || Value::known(F::from(acc)),
                    )?;
                }

                let out_row = num_lanes - 1;
                cfg.q_out.enable(&mut region, out_row)?;
                let out_cell = region.assign_advice(
                    || "out",
                    cfg.out,
                    out_row,
                    || Value::known(F::from(acc)),
                )?;
                Ok(out_cell)
            },
        )
    }

    pub fn expose_public(
        &self,
        layouter: &mut impl Layouter<F>,
        cell: AssignedCell<F, F>,
        row: usize,
    ) -> Result<(), Error> {
        layouter.constrain_instance(cell.cell(), self.cfg.instance, row)
    }
}

/// Wrapper circuit. `lane_rows` and `num_lanes` are circuit STRUCTURE derived
/// from the publicly released capacity, not witness.
pub struct MyCircuit<F: Field + Ord> {
    pub edges: Vec<Edge>,
    pub bag1_pad_extra: usize,
    pub bag2_pad_extra: usize,
    pub lane_rows: usize,
    pub num_lanes: usize,
    pub tamper: Tamper,
    pub _marker: PhantomData<F>,
}

impl<F: Field + Ord> Default for MyCircuit<F> {
    fn default() -> Self {
        Self {
            edges: vec![],
            bag1_pad_extra: 0,
            bag2_pad_extra: 0,
            lane_rows: LANE_ROWS,
            num_lanes: 1,
            tamper: Tamper::None,
            _marker: PhantomData,
        }
    }
}

impl<F: Field + Ord> Circuit<F> for MyCircuit<F> {
    type Config = Gq3DpConfig<F>;
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
        Gq3DpChip::<F>::configure(meta)
    }

    fn synthesize(&self, cfg: Self::Config, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        assert_eq!(
            cfg.lanes.len(),
            self.num_lanes,
            "configured lane count does not match the circuit: call \
             set_config_lanes(num_lanes) before keygen / MockProver"
        );
        let chip = Gq3DpChip::<F>::construct(cfg);
        let out = chip.assign(
            &mut layouter,
            &self.edges,
            self.bag1_pad_extra,
            self.bag2_pad_extra,
            self.lane_rows,
            self.tamper,
        )?;
        chip.expose_public(&mut layouter, out, 0)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DP lane run
//
// One code path for the `#[ignore]`d `tests::test_dp_lanes` below and for
// `src/bin/dp_lane_bench.rs`.
// ---------------------------------------------------------------------------

/// Geometry the released Bag1 capacity implies, plus the edges it was derived
/// from. No SRS is read and no key is built here.
struct Gq3DpSetup {
    edges: Vec<Edge>,
    bag1_pad_extra: usize,
    bag2_pad_extra: usize,
    plan: crate::dp_lane::DpLanePlan,
}

/// Geometry the released Bag1 capacity implies, or an explanation of why this
/// configuration cannot be built.
///
/// `Err` covers every way a row can be rejected BEFORE proving: the lane cap on
/// the laned bag, and the structural fit of the tallest column group at the
/// pinned degree. Both are properties of the request, so a sweep reports them
/// and continues.
fn try_dp_lane_setup(
    dataset: &str,
    privacy: crate::bench_queries::Privacy,
) -> Result<Gq3DpSetup, String> {
    let edges = crate::bench_queries::load_graph(dataset);
    let (bag1_pad_extra, bag2_pad_extra) =
        crate::bench_queries::graph_pads("gq3", dataset, &edges, privacy);

    // The degree is PINNED at the Revealing-Join-Size value; the DP release is
    // absorbed by lanes, not by a bigger domain.
    let k = crate::bench_queries::degree_for("gq3", dataset, crate::bench_queries::Privacy::Rjs);
    let lane_rows = lane_rows_for(k);

    let stats = crate::bench_queries::bag_stats("gq3", &edges);
    let n12 = stats.bag1_size as usize + bag1_pad_extra;
    let c = try_lanes_for_capacity(n12, k)?;

    // every column group must fit under the range-table loads plus slack
    let n3 = edges.len() + bag2_pad_extra;
    let tallest = (edges.len() + 2).max(n3 + 1).max(lane_rows).max(c);
    if PREAMBLE_ROWS + tallest + BLINDING_SLACK > 1usize << k {
        return Err(format!(
            "tallest column group ({} rows) does not fit k={} ({} rows, of which {} go to \
             the u8 range-table loads and {} to blinding)",
            tallest,
            k,
            1usize << k,
            PREAMBLE_ROWS,
            BLINDING_SLACK
        ));
    }

    let plan = crate::dp_lane::DpLanePlan {
        query: "gq3".to_string(),
        dataset: dataset.to_string(),
        k,
        lane_rows,
        lanes: vec![c],
        capacity: vec![n12],
        true_size: vec![stats.bag1_size as usize],
        pads: vec![bag1_pad_extra, bag2_pad_extra],
    };
    Ok(Gq3DpSetup {
        edges,
        bag1_pad_extra,
        bag2_pad_extra,
        plan,
    })
}

/// [`try_dp_lane_setup`] for callers that treat an infeasible release as fatal.
fn dp_lane_setup(dataset: &str, privacy: crate::bench_queries::Privacy) -> Gq3DpSetup {
    try_dp_lane_setup(dataset, privacy).unwrap_or_else(|e| panic!("{}", e))
}

/// Geometry only, `Err` when this release cannot be built: no SRS is read, no
/// key is built, nothing is proved. Use this from a sweep, which must report an
/// infeasible cell and carry on.
pub fn try_plan_dp_lanes(
    dataset: &str,
    privacy: crate::bench_queries::Privacy,
) -> Result<crate::dp_lane::DpLanePlan, String> {
    try_dp_lane_setup(dataset, privacy).map(|s| s.plan)
}

/// Geometry only: no SRS is read, no key is built, nothing is proved.
/// Panics on an infeasible release; see [`try_plan_dp_lanes`].
pub fn plan_dp_lanes(
    dataset: &str,
    privacy: crate::bench_queries::Privacy,
) -> crate::dp_lane::DpLanePlan {
    dp_lane_setup(dataset, privacy).plan
}

/// Real IPA proving at the Revealing-Join-Size degree with the DP release
/// hosted in lanes.
///
/// The verifying and proving keys are built ONCE, outside the timed region;
/// then `reps` proofs are generated and every one of them is verified.
/// `proof_path` is `Some` only for callers that want the last proof on disk
/// (the test keeps writing it, `dp_lane_bench` does not).
pub fn run_dp_lanes(
    dataset: &str,
    privacy: crate::bench_queries::Privacy,
    reps: usize,
    proof_path: Option<&str>,
) -> crate::dp_lane::DpLaneRun {
    use halo2_proofs::poly::{
        ipa::{
            commitment::IPACommitmentScheme,
            multiopen::ProverIPA,
            strategy::SingleStrategy,
        },
        VerificationStrategy,
    };
    use halo2_proofs::transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    };
    use halo2curves::pasta::{EqAffine, Fp};
    use std::time::Instant;

    assert!(reps >= 1, "reps must be at least 1");
    let setup = dp_lane_setup(dataset, privacy);
    let k = setup.plan.k;
    let c = setup.plan.lanes[0];
    let cnt = crate::bench_queries::count_gq3(&setup.edges);

    set_config_lanes(c);
    let build = || MyCircuit::<Fp> {
        edges: setup.edges.clone(),
        bag1_pad_extra: setup.bag1_pad_extra,
        bag2_pad_extra: setup.bag2_pad_extra,
        lane_rows: setup.plan.lane_rows,
        num_lanes: c,
        tamper: Tamper::None,
        _marker: PhantomData,
    };

    // Shared loader: reads the persisted SRS, and generates and persists a
    // degree that is not shipped rather than aborting the sweep on it.
    let params = crate::bench_queries::params_for(k);

    let t = Instant::now();
    let vk = keygen_vk(&params, &build()).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &build()).expect("keygen_pk should not fail");
    let keygen_s = t.elapsed().as_secs_f64();

    let public_input: Vec<Fp> = vec![Fp::from(cnt)];
    let mut prove_s = Vec::with_capacity(reps);
    let mut verify_total = 0.0;
    let mut proof_bytes = 0usize;
    let mut last_proof = Vec::new();

    for _ in 0..reps {
        // Building the circuit (cloning the edge list into it) is untimed;
        // only `create_proof` is.
        let circuit = build();
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
        plan: setup.plan,
        keygen_s,
        verify_s: verify_total / prove_s.len() as f64,
        prove_s,
        proof_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        lane_rows_for, lanes_for, lanes_for_capacity, set_config_lanes, MyCircuit, Tamper,
        BASE_DEGREE, LANE_ROWS, MAX_LANES,
    };

    use crate::data::graph_data_processing::Edge;
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::Circuit;
    use halo2curves::pasta::Fp;

    use std::marker::PhantomData;
    use std::time::Instant;

    /// Seven directed edges over six nodes with exactly three ordered
    /// triangles: 0->1->2->0, 0->1->4->0 and 0->3->4->0.
    fn synthetic_edges() -> Vec<Edge> {
        [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0), (1, 4)]
            .into_iter()
            .map(|(src, dst)| Edge { src, dst })
            .collect()
    }

    /// 6 wedges survive the A<B pre-filter, so 5 pad rows give a released
    /// capacity of 11, which at 4 rows per lane really spans 3 lanes and
    /// leaves row 11 as a rounding-up pad.
    fn three_lane_circuit(tamper: Tamper) -> (MyCircuit<Fp>, usize) {
        let edges = synthetic_edges();
        let lane_rows = 4;
        let bag1_pad_extra = 5;
        let c = lanes_for(6 + bag1_pad_extra, lane_rows);
        assert_eq!(c, 3);

        let circuit = MyCircuit::<Fp> {
            edges,
            bag1_pad_extra,
            bag2_pad_extra: 0,
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
        // degenerate release still gets one lane
        assert_eq!(lanes_for(0, LANE_ROWS), 1);

        // lane geometry at the two gq3 base degrees
        assert_eq!(lane_rows_for(BASE_DEGREE), LANE_ROWS);
        assert!(lane_rows_for(22) > 16 * LANE_ROWS);

        // measured releases: lastfm at eps=0.1 needs 2 lanes at k=18, and at
        // eps=0.01 it needs dozens but stays under the structural cap
        assert_eq!(lanes_for_capacity(442_919, 18), 2);
        assert!(lanes_for_capacity(11_369_932, 18) <= MAX_LANES);
        // facebook and wiki at eps=0.1 collapse to a single lane at k=22
        assert_eq!(lanes_for_capacity(3_287_746, 22), 1);
        assert_eq!(lanes_for_capacity(2_725_688, 22), 1);
    }

    /// c = 3 lanes at k = 11: the padded Bag1 pipeline really spans three
    /// lanes, each with its own prefix sum, and the totals stage adds them.
    #[test]
    fn mock_three_lanes() {
        let (circuit, c) = three_lane_circuit(Tamper::None);
        set_config_lanes(c);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(3u64)]]).unwrap();
        prover.assert_satisfied();
    }

    /// PRIVACY REGRESSION GUARD. Every lane must be a FULL structural replica,
    /// so the constraint system has to grow by exactly the same amount for each
    /// added lane and by nothing else. A cheap "padding-only overflow lane"
    /// (the leak this design exists to avoid) would show up here immediately as
    /// a non-constant increment, and so would a lane that quietly drops a gate,
    /// a lookup or a permutation column.
    #[test]
    fn lanes_are_structural_replicas() {
        use halo2_proofs::plonk::ConstraintSystem;

        fn shape(c: usize) -> [usize; 7] {
            set_config_lanes(c);
            let mut cs = ConstraintSystem::<Fp>::default();
            let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
            [
                cs.num_advice_columns(),
                cs.num_fixed_columns(),
                cs.num_selectors(),
                cs.lookups().len(),
                cs.shuffles().len(),
                cs.permutation().get_columns().len(),
                cs.degree(),
            ]
        }

        let shapes: Vec<[usize; 7]> = (1..=6).map(shape).collect();
        let step: Vec<usize> = (0..7).map(|i| shapes[1][i] - shapes[0][i]).collect();

        // one lane costs a fixed number of advice columns, lookup arguments and
        // permutation columns, and nothing else
        assert_eq!(
            step[1], 0,
            "fixed columns must not grow with the lane count"
        );
        assert_eq!(step[2], 0, "selectors must not grow with the lane count");
        assert_eq!(step[4], 0, "shuffles must not grow with the lane count");
        assert_eq!(step[6], 0, "degree must not grow with the lane count");
        assert!(step[0] > 0 && step[3] > 0, "a lane must cost something");
        // only run_sum is copied out of a lane
        assert_eq!(step[5], 1, "a lane must add exactly one permutation column");

        for w in shapes.windows(2) {
            for i in 0..7 {
                assert_eq!(
                    w[1][i] - w[0][i],
                    step[i],
                    "lane cost is not constant in component {i}: {:?} -> {:?}",
                    w[0],
                    w[1]
                );
            }
        }
    }

    /// The COUNT(*) must be invariant to the lane geometry, and the block-wise
    /// split must place every real Bag1 row in exactly one lane at the right
    /// offset. The geometries below cover: real rows spilling into the LAST
    /// lane (no rounding-up pad at all), one row per lane, an odd rounding-up
    /// tail, and the degenerate single-lane case.
    #[test]
    fn mock_lane_geometries() {
        // (lane_rows, bag1_pad_extra, expected lanes)
        let cases: [(usize, usize, usize); 6] = [
            (2, 0, 3), // 6 real rows, no pad: lane 2 is entirely real
            (1, 0, 6), // one row per lane, q_sum never fires
            (4, 5, 3), // 11 released rows over 12: one rounding-up pad
            (3, 1, 3), // 7 released rows over 9: two rounding-up pads
            (5, 0, 2), // real rows straddle the lane-0/lane-1 boundary
            (16, 5, 1),
        ];
        for (lane_rows, bag1_pad_extra, want_c) in cases {
            let c = lanes_for(6 + bag1_pad_extra, lane_rows);
            assert_eq!(c, want_c, "lane_rows={lane_rows} pad={bag1_pad_extra}");
            set_config_lanes(c);
            let circuit = MyCircuit::<Fp> {
                edges: synthetic_edges(),
                bag1_pad_extra,
                bag2_pad_extra: 0,
                lane_rows,
                num_lanes: c,
                tamper: Tamper::None,
                _marker: PhantomData,
            };
            let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(3u64)]]).unwrap();
            assert_eq!(
                prover.verify(),
                Ok(()),
                "lane_rows={lane_rows} pad={bag1_pad_extra} c={c}"
            );

            // ... and the same geometry must reject a wrong public count.
            set_config_lanes(c);
            let circuit = MyCircuit::<Fp> {
                edges: synthetic_edges(),
                bag1_pad_extra,
                bag2_pad_extra: 0,
                lane_rows,
                num_lanes: c,
                tamper: Tamper::None,
                _marker: PhantomData,
            };
            let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(4u64)]]).unwrap();
            assert!(
                prover.verify().is_err(),
                "lane_rows={lane_rows} pad={bag1_pad_extra}: wrong count must not verify"
            );
        }
    }

    /// Corrupting one lane row's contribution must break the contrib gate and
    /// that lane's prefix sum.
    #[test]
    fn mock_reject_tampered_lane_contrib() {
        let (circuit, c) = three_lane_circuit(Tamper::LaneContrib);
        set_config_lanes(c);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(3u64)]]).unwrap();
        assert!(
            prover.verify().is_err(),
            "tampered lane contribution must not verify"
        );
    }

    /// Corrupting one lane total must break the copy constraint out of the
    /// lane and the totals accumulator.
    #[test]
    fn mock_reject_tampered_lane_total() {
        let (circuit, c) = three_lane_circuit(Tamper::LaneTotal);
        set_config_lanes(c);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(3u64)]]).unwrap();
        assert!(
            prover.verify().is_err(),
            "tampered lane total must not verify"
        );
    }

    /// Breaking the pad convention on the last rounding-up row must break the
    /// "bag1 real + key" gate and the r1 membership lookup.
    #[test]
    fn mock_reject_tampered_pad_row() {
        let (circuit, c) = three_lane_circuit(Tamper::PadRow);
        set_config_lanes(c);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(3u64)]]).unwrap();
        assert!(prover.verify().is_err(), "tampered pad row must not verify");
    }

    /// c = 1 degenerates to g_sql3_obj's single-group layout: both circuits
    /// must accept the same input and expose the same COUNT(*).
    #[test]
    fn mock_single_lane_matches_g_sql3_obj() {
        let edges = synthetic_edges();
        let public: Vec<Fp> = vec![Fp::from(3u64)];

        let baseline = crate::graph_sql::g_sql3_obj::MyCircuit::<Fp> {
            edges: edges.clone(),
            bag1_pad_extra: 5,
            bag2_pad_extra: 0,
            _marker: PhantomData,
        };
        let prover = MockProver::run(11, &baseline, vec![public.clone()]).unwrap();
        prover.assert_satisfied();

        let lane_rows = 16; // n12 = 6 + 5 = 11 fits one lane
        let c = lanes_for(6 + 5, lane_rows);
        assert_eq!(c, 1);
        set_config_lanes(c);
        let laned = MyCircuit::<Fp> {
            edges,
            bag1_pad_extra: 5,
            bag2_pad_extra: 0,
            lane_rows,
            num_lanes: c,
            tamper: Tamper::None,
            _marker: PhantomData,
        };
        let prover = MockProver::run(11, &laned, vec![public]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    #[ignore = "real IPA proving over a full graph dataset; run explicitly"]
    fn test_dp_lanes() {
        let dataset = std::env::var("VPJOIN_DATASET").unwrap_or_else(|_| "lastfm".into());
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

        let proof_path = crate::paths::proof_file("proof_gq3_dp_lanes");
        let t_total = Instant::now();
        let run = super::run_dp_lanes(&dataset, privacy, 1, Some(&proof_path));
        let p = &run.plan;
        println!(
            "[gq3 dp-lanes] dataset={} privacy={} pads: bag1={} bag2={}",
            dataset,
            privacy.label(),
            p.pads[0],
            p.pads[1]
        );
        println!(
            "[gq3 dp-lanes] bag1_true={} n12={} lanes={} lane_rows={} effective_capacity={}",
            p.true_size[0],
            p.capacity[0],
            p.lanes[0],
            p.lane_rows,
            p.lanes[0] * p.lane_rows
        );
        println!("Proof written to: {}", proof_path);
        println!(
            "[gq3 dp-lanes] lanes={} keygen {:.2}s prove {:.2}s verify {:.2}s \
             total prove+verify {:?}",
            p.lanes[0],
            run.keygen_s,
            run.prove_mean(),
            run.verify_s,
            t_total.elapsed()
        );
    }
}
