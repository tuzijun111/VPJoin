//! Multi-lane DP-padding variant of the GQ4 4-cycle circuit (`g_sql4_obj`).
//!
//! WHY THIS FILE EXISTS
//! --------------------
//! GQ4 materializes BOTH of its bags at capacities that, under differential
//! privacy, are NOISY RELEASES rather than the true bag sizes. The released
//! capacities are public, but they are not powers of two, and a single column
//! group must be padded up to the next power of two. On lastfm the true bags
//! hold 232,943 rows each and Revealing-Join-Size proves at k = 18; the DP
//! release at (eps,delta) = (0.1,1e-5) is 788,220 rows for Bag1 and 621,070
//! for Bag2, which alone would push the circuit to k = 20. At eps = 0.01 the
//! releases are ~50x larger and the degree would reach k = 26. Every step
//! doubles the prover's work even though the capacity grew by far less.
//!
//! This circuit keeps the degree PINNED at the Revealing-Join-Size value and
//! hosts each released capacity in
//!
//!     c = ceil(capacity / lane_rows)
//!
//! parallel column-group LANES, so the per-bag cost grows with the released
//! capacity instead of doubling at each power of two. Same trick, same
//! conventions, as `sql::q5_obj_dp` does for TPC-H Q5 and `g_sql3_obj_dp` does
//! for the GQ3 triangle.
//!
//! WHAT A LANE IS
//! --------------
//! GQ4 has TWO pad-driven pipelines, so it has two kinds of lane, and both are
//! FULL structural replicas:
//!
//!   * A Bag2 lane replicates the T34 column group (the eight t34 columns and
//!     the C<D comparator) AND a complete AggSumByKey group-by, which turns
//!     that lane's block of Bag2 rows into a lane-local message map table.
//!   * A Bag1 lane replicates the T12 column group (the eight t12 columns, the
//!     A<B and B<C comparators, the contribution column and a prefix sum) plus
//!     one complete MapLookup probe PER Bag2 lane.
//!
//! Lanes of a kind are identical by construction and all of them are assigned
//! to the full `lane_rows` rows, so the assigned structure is a function of the
//! PUBLIC released capacities only. It never depends on the true bag sizes. A
//! cheaper "overflow lane" that only carries padding would leak exactly the
//! quantity DP is paying to hide, so there is no such thing here.
//!
//! WHY LANE-LOCAL AGGREGATION IS SOUND
//! -----------------------------------
//! The message a Bag2 row sends is `msg_val(pack2(A,C)) = COUNT(paths C->D->A
//! with C<D)`, a pure COUNT. Counts ADD across a partition, so splitting Bag2
//! block-wise over c2 lanes and grouping each block independently is exactly
//! correct: lane l's map holds `count_l(key)`, and
//!
//!     sum_l count_l(key) = count(key).
//!
//! A Bag1 row therefore probes all c2 maps and sums the c2 retrieved values.
//! Keys missing from a given lane's map are proven missing by that lane's gap
//! bracket and contribute 0, exactly as in the single-group circuit. No
//! ordering, no sentinel and no carry crosses a Bag2 lane boundary: the Bag2
//! side needs ZERO stitching.
//!
//! COST NOTE (read before scaling this up)
//! ---------------------------------------
//! Probing c2 lane-local maps costs c1 * c2 MapLookup replicas, so the Bag1
//! side grows QUADRATICALLY in the lane counts, not linearly. That is inherent
//! to resolving a PRIVATE key against a table of private size: with c2 separate
//! tables a lookup argument can only address one of them at a time. It is fine
//! when one side collapses to a single lane (facebook and wiki at eps = 0.1
//! give c1 = c2 = 1) and it is roughly break-even against the k+2 jump at
//! lastfm eps = 0.1 (c1 = 4, c2 = 3). Getting a genuinely linear GQ4 needs a
//! different join strategy, namely replacing the message map by an in-circuit
//! sort-merge of the two bags, which is a different circuit rather than a
//! laned version of this one. See the report accompanying this file.
//!
//! WHAT STAYS SINGLE COPY
//! ----------------------
//! Everything whose height is fixed by the PUBLIC edge count: the two indexed
//! views of the edge multiset (`in_by_dst`, `out_by_src`), which are the table
//! side of every r1/r2/r3/r4 membership lookup. Their height tracks |E|, which
//! is public in every privacy regime. The base Edge advice columns of
//! `g_sql4_obj` (`e_eid`, `e_src`, `e_dst`) carry no gate and no lookup there,
//! so they are dropped rather than replicated.
//!
//! WHERE THE ONLY STITCH IS
//! ------------------------
//! The single global accumulator is Bag1's running sum of `contrib`. Following
//! `q5_obj_dp`, each Bag1 lane is self-contained: it runs its own prefix sum
//! with its own `q_sum0` at its row 0, so no accumulator ever crosses a lane
//! boundary. The c1 lane totals are then copied into a c1-row totals stage that
//! adds them with a degree-2 accumulator, and `out` at row c1-1 goes to the
//! instance. That is the ONLY cross-lane wiring in the whole circuit: c1 copy
//! constraints and one small accumulator.
//!
//! MEASURED STRUCTURE (ConstraintSystem probe)
//! -------------------------------------------
//!   c1=1 c2=1 : advice  141  fixed 3  sel  39  lookups   78  shuf  4  perm  31
//!   c1=2 c2=1 : advice  192  fixed 3  sel  39  lookups  114  shuf  4  perm  32
//!   c1=1 c2=2 : advice  215  fixed 3  sel  53  lookups  122  shuf  6  perm  43
//!   c1=4 c2=3 : advice  580  fixed 3  sel  67  lookups  382  shuf  8  perm  58
//!   c1=9 c2=10: advice 2871  fixed 3  sel 165  lookups 2058  shuf 22  perm 147
//! cs.degree() is 7 for every configuration, as in `g_sql4_obj`. At c1=c2=1 the
//! circuit matches the single-group original (142 advice there: this file drops
//! the three unconstrained base-Edge columns and adds the two totals columns),
//! and the fixed-column count falls from 9 to 3 because every lane Lt chip
//! shares one u8 range column. Each extra Bag1 lane adds 51 advice and 36
//! lookup arguments PLUS 23 advice / 18 lookups per Bag2 lane it must probe;
//! each extra Bag2 lane adds 51 advice, 26 lookups and 2 shuffles PLUS the same
//! per-probe cost in every Bag1 lane. That is the c1*c2 term the cost note
//! above is about.
//!
//! The permutation column count grows by ONE per Bag1 lane and by nothing at
//! all per Bag2 lane or per probe: lane columns are deliberately kept out of
//! the permutation argument, since halo2 charges for every column in it whether
//! or not a copy constraint touches it, and the only lane cell that is ever
//! copied is a Bag1 lane's last `run_sum`. `lanes_are_structural_replicas`
//! locks that shape in, and with it the privacy property that every lane costs
//! exactly the same.
//!
//! Row budget per lane, at circuit degree k:
//!   lane_rows = 2^k - PREAMBLE_ROWS - BLINDING_SLACK
//! where PREAMBLE_ROWS covers the u8 range-table `load` regions (each is 256
//! fixed rows in a region of its own, and the floor planner is free to place
//! them ahead of the witness region) and BLINDING_SLACK covers the blinding
//! rows halo2 reserves at the bottom of every advice column.

use halo2_proofs::plonk::Expression;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::data::graph_data_processing::Edge;

// One shared definition of the PAD / packing conventions, the chips and the
// host-side witness derivation (from g_sql4_obj).
use super::g_sql4_obj::{
    gq4_derive, pack2, AggSumByKeyChip, AggSumByKeyConfig, Field, IndexedViewChip,
    IndexedViewConfig, MapLookupChip, MapLookupConfig, NUM_BYTES, PACK_SHIFT, PAD_U64,
};

use std::collections::BTreeMap;
use std::marker::PhantomData;

/// Rows the u8 range-table `load` regions may claim ahead of the witness
/// region. Three distinct u8 columns are loaded: the sort chip of each indexed
/// view (2), plus the single column every lane Lt chip shares (1). Each `load`
/// writes 256 fixed rows. Reserving all three is conservative: the floor
/// planner places regions per column, so in practice they overlap the witness
/// region.
pub const PREAMBLE_ROWS: usize = 3 * 256;

/// Blinding rows halo2 keeps at the bottom of every advice column.
pub const BLINDING_SLACK: usize = 64;

/// Default base degree: the Revealing-Join-Size degree of gq4 on lastfm.
/// facebook and wiki prove at k = 22; pass their degree to [`lane_rows_for`].
pub const BASE_DEGREE: u32 = 18;

/// Usable rows per lane at the default base degree.
pub const LANE_ROWS: usize = (1usize << BASE_DEGREE) - PREAMBLE_ROWS - BLINDING_SLACK;

/// Public structural cap on either lane count. GQ4 releases at eps = 0.01 run
/// into the dozens of lanes (gq4-lastfm would need about 145 at k = 18), so
/// this is deliberately far above q5_obj_dp's cap of 16. Note that the Bag1
/// side pays c1 * c2 probes, so a config near this cap is sizeable even when
/// each individual count is legal.
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

/// Lane count for a released bag capacity at a caller-supplied base degree,
/// with the structural cap enforced.
pub fn lanes_for_capacity(capacity: usize, base_degree: u32) -> usize {
    let lane_rows = lane_rows_for(base_degree);
    let c = lanes_for(capacity, lane_rows);
    assert!(
        c <= MAX_LANES,
        "released capacity {} needs {} lanes at base degree k={} ({} usable rows per lane), \
         above MAX_LANES={}",
        capacity,
        c,
        base_degree,
        lane_rows,
        MAX_LANES
    );
    c
}

// `Circuit::configure` has no access to the circuit instance (the
// `circuit-params` feature of halo2 is not enabled in this build), so the
// caller stamps the PUBLIC lane counts here before keygen / MockProver runs;
// `synthesize` asserts the resulting config matches the circuit.
// Thread-local so parallel tests with different lane counts cannot race.
thread_local! {
    static CONFIG_LANES: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((1, 1)) };
}

/// Stamp the (Bag1, Bag2) lane counts the NEXT `configure` call on this thread
/// lays out.
pub fn set_config_lanes(bag1_lanes: usize, bag2_lanes: usize) {
    for c in [bag1_lanes, bag2_lanes] {
        assert!(
            (1..=MAX_LANES).contains(&c),
            "lane count {} outside 1..={}",
            c,
            MAX_LANES
        );
    }
    CONFIG_LANES.with(|l| l.set((bag1_lanes, bag2_lanes)));
}

/// MockProver fault injection used by the negative tests in this file.
/// Production callers leave this at `None`; it only perturbs the witness,
/// never the constraint system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tamper {
    #[default]
    None,
    /// Bump one Bag1 lane row's contribution by 1 (the contrib gate and that
    /// lane's own prefix-sum gates must reject).
    LaneContrib,
    /// Bump one Bag1 lane total by 1 (the copy constraint into the totals
    /// stage and the totals accumulator must both reject).
    LaneTotal,
    /// Break the Bag1 pad-row convention on the last rounding-up row (the
    /// "bag1 real + key" gate and the r1 membership lookup must reject).
    Bag1PadRow,
    /// Break the Bag2 pad-row convention on the last rounding-up row (the
    /// r3 membership lookup must reject).
    Bag2PadRow,
    /// Zero the retrieved message value of every probe EXCEPT probe 0, while
    /// leaving `contrib` at its honest (all-lane) value. This is the direct
    /// negative test for the cross-lane message sum: if the "contrib gate" ever
    /// stopped summing all `c2` probes, or if a probe's membership lookup were
    /// dropped, this witness would verify. It must not.
    DropProbeVal,
}

/// One Bag2 lane: the T34 column group plus its own AggSumByKey group-by,
/// which turns this lane's block of Bag2 rows into a lane-local message map.
#[derive(Clone, Debug)]
pub struct Bag2LaneConfig<F: Field + Ord> {
    // Bag2 rows: C->D->A, plus the source-edge ids and per-group indices
    t34_c: Column<Advice>,
    t34_d: Column<Advice>,
    t34_a: Column<Advice>,
    t34_i_r3: Column<Advice>,
    t34_j_r4: Column<Advice>,
    t34_r3_eid: Column<Advice>,
    t34_r4_eid: Column<Advice>,
    t34_real: Column<Advice>,

    // ordering predicate C<D (a Bag2 row only sends a message when C<D)
    lt_cd: LtConfig<F, NUM_BYTES>,

    // lane-local message map: msg_key = pack2(A,C), msg_val = count
    agg: AggSumByKeyConfig<F>,
}

/// One Bag1 lane: the T12 column group, one MapLookup probe per Bag2 lane, the
/// contribution column and a LANE-LOCAL prefix sum. `out` is NOT replicated; it
/// lives once in the cross-lane totals stage.
#[derive(Clone, Debug)]
pub struct Bag1LaneConfig<F: Field + Ord> {
    // Bag1 rows: A->B->C, plus the source-edge ids and per-group indices
    t12_a: Column<Advice>,
    t12_b: Column<Advice>,
    t12_c: Column<Advice>,
    t12_i_r1: Column<Advice>,
    t12_j_r2: Column<Advice>,
    t12_r1_eid: Column<Advice>,
    t12_r2_eid: Column<Advice>,
    t12_real: Column<Advice>,

    // membership-or-gap probe of the message map, ONE PER BAG2 LANE.
    // Counts add across a partition, so the row's message value is the sum of
    // the c2 retrieved values.
    probes: Vec<MapLookupConfig<F>>,

    // ordering predicates A<B and B<C
    lt_ab: LtConfig<F, NUM_BYTES>,
    lt_bc: LtConfig<F, NUM_BYTES>,

    // per-row contribution and the lane-local prefix sum
    contrib: Column<Advice>,
    run_sum: Column<Advice>,
}

#[derive(Clone, Debug)]
pub struct Gq4DpConfig<F: Field + Ord> {
    instance: Column<Instance>,

    // ---------------- fixed-height sections (shared, never laned) ----------
    // indexed views of the edge multiset; the table side of every r1/r2/r3/r4
    // membership lookup. Height tracks |E|, which is public in every regime.
    in_by_dst: IndexedViewConfig<F>,  // key=dst, val=src
    out_by_src: IndexedViewConfig<F>, // key=src, val=dst

    /// One u8 range table for EVERY lane Lt chip, of either kind. `load` costs
    /// one 256-row region per distinct u8 column, and that must not grow with
    /// the lane counts.
    lane_u8: Column<Fixed>,

    // ---------------- Bag2 lanes ------------------------------------------
    // All Bag2 lanes are active on the SAME rows 0..lane_rows, so one selector
    // of each kind serves every lane; only the columns replicate. (The
    // per-lane AggSumByKey replicas keep their own selectors, since their chip
    // enables them itself.)
    bag2_lanes: Vec<Bag2LaneConfig<F>>,
    q_t34_lookup: Selector, // r3/r4 membership lookups (complex)
    q_t34_flag: Selector,
    q_cd: Selector,
    q_t34_msg_in: Selector,

    // ---------------- Bag1 lanes ------------------------------------------
    bag1_lanes: Vec<Bag1LaneConfig<F>>,
    q_t12_lookup: Selector, // r1/r2 membership lookups (complex)
    q_t12_key: Selector,
    q_flag: Selector,
    q_complex: Selector, // msg member / msg gap lookups (complex)
    q_order12: Selector,
    q_contrib: Selector,
    q_sum0: Selector, // lane-local prefix sum, row 0 of every Bag1 lane
    q_sum: Selector,  // lane-local prefix sum, rows 1..lane_rows

    // ---------------- cross-lane totals (the only stitch) ------------------
    lane_tot: Column<Advice>, // row l = Bag1 lane l's total, by copy constraint
    tot_run: Column<Advice>,  // prefix sum of lane_tot over rows 0..c1
    q_tot0: Selector,
    q_tot: Selector,

    out: Column<Advice>,
    q_out: Selector,
}

#[derive(Clone, Debug)]
pub struct Gq4DpChip<F: Field + Ord> {
    cfg: Gq4DpConfig<F>,
}

impl<F: Field + Ord> Gq4DpChip<F> {
    /// One full structural replica of the Bag2 column group plus its own
    /// message aggregator. Every Bag2 lane gets the same gates and the same
    /// lookups; only the columns differ.
    fn configure_bag2_lane(
        meta: &mut ConstraintSystem<F>,
        in_by_dst: &IndexedViewConfig<F>,
        out_by_src: &IndexedViewConfig<F>,
        lane_u8: Column<Fixed>,
        q_t34_lookup: Selector,
        q_t34_flag: Selector,
        q_cd: Selector,
        q_t34_msg_in: Selector,
    ) -> Bag2LaneConfig<F> {
        let t34_c = meta.advice_column();
        let t34_d = meta.advice_column();
        let t34_a = meta.advice_column();
        let t34_i_r3 = meta.advice_column();
        let t34_j_r4 = meta.advice_column();
        let t34_r3_eid = meta.advice_column();
        let t34_r4_eid = meta.advice_column();
        let t34_real = meta.advice_column();
        // NOTE: no `enable_equality` on the lane columns. halo2 charges for
        // every column in the permutation argument whether or not a copy
        // constraint ever touches it, so a per-lane column that is only read by
        // gates and lookups must stay out of it: otherwise the permutation cost
        // grows with the lane count for nothing. Nothing is copied out of a
        // Bag2 lane at all -- its message reaches Bag1 through a lookup.

        // r3 via InByDst: key=D, idx=i_r3 -> val=C, eid=r3_eid  (r3 is C->D).
        // The input side is per lane, the table side is the shared view: a
        // lookup argument is a multiset relation over the whole domain, so
        // adding lane rows against the same table needs no stitching.
        meta.lookup_any("bag2 r3 from in_by_dst", |m| {
            let q = m.query_selector(q_t34_lookup);
            vec![
                (
                    q.clone() * m.query_advice(t34_d, Rotation::cur()),
                    m.query_advice(in_by_dst.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t34_i_r3, Rotation::cur()),
                    m.query_advice(in_by_dst.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t34_c, Rotation::cur()),
                    m.query_advice(in_by_dst.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t34_r3_eid, Rotation::cur()),
                    m.query_advice(in_by_dst.sorted_eid, Rotation::cur()),
                ),
            ]
        });
        // r4 via OutBySrc: key=D, idx=j_r4 -> val=A, eid=r4_eid  (r4 is D->A).
        // This is also the closing-edge check: `val` is t34_a, so the edge
        // D->A that closes the cycle is proven to exist by the same lookup
        // that produces A.
        meta.lookup_any("bag2 r4 from out_by_src", |m| {
            let q = m.query_selector(q_t34_lookup);
            vec![
                (
                    q.clone() * m.query_advice(t34_d, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t34_j_r4, Rotation::cur()),
                    m.query_advice(out_by_src.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t34_a, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t34_r4_eid, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_eid, Rotation::cur()),
                ),
            ]
        });

        meta.create_gate("bag2 real boolean", |m| {
            let q = m.query_selector(q_t34_flag);
            let real = m.query_advice(t34_real, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![q * real.clone() * (one - real)]
        });

        // ordering in Bag2: C < D, enabled only if real=1
        let lt_cd = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            lane_u8,
            |m| m.query_selector(q_cd) * m.query_advice(t34_real, Rotation::cur()),
            |m| m.query_advice(t34_c, Rotation::cur()),
            |m| m.query_advice(t34_d, Rotation::cur()),
        );

        // Lane-local message aggregator. Grouping this lane's block on its own
        // is exactly right because the message is a COUNT and counts add
        // across a partition; see the header.
        let agg = AggSumByKeyChip::<F>::configure_with_u8(meta, lane_u8, lane_u8);

        // Tie this lane's agg inputs to its Bag2 rows:
        //   keep   = t34_real * [C<D]
        //   in_key = keep ? pack2(A,C) : PAD
        //   in_val = keep
        meta.create_gate("msg input from bag2", |m| {
            let q = m.query_selector(q_t34_msg_in);

            let a = m.query_advice(t34_a, Rotation::cur());
            let c = m.query_advice(t34_c, Rotation::cur());
            let key_expr = a * Expression::Constant(F::from(PACK_SHIFT)) + c;

            let real = m.query_advice(t34_real, Rotation::cur());
            let cd = lt_cd.is_lt(m, None);
            let keep = real * cd;

            let one = Expression::Constant(F::ONE);
            let pad = Expression::Constant(F::from(PAD_U64));
            let selected_key = keep.clone() * key_expr + (one - keep.clone()) * pad;

            vec![
                q.clone() * (m.query_advice(agg.in_key, Rotation::cur()) - selected_key),
                q * (m.query_advice(agg.in_val, Rotation::cur()) - keep),
            ]
        });

        Bag2LaneConfig {
            t34_c,
            t34_d,
            t34_a,
            t34_i_r3,
            t34_j_r4,
            t34_r3_eid,
            t34_r4_eid,
            t34_real,
            lt_cd,
            agg,
        }
    }

    /// One full structural replica of the Bag1 column group, carrying one
    /// MapLookup probe per Bag2 lane. Every Bag1 lane gets the same gates and
    /// the same lookups; only the columns differ.
    #[allow(clippy::too_many_arguments)]
    fn configure_bag1_lane(
        meta: &mut ConstraintSystem<F>,
        in_by_dst: &IndexedViewConfig<F>,
        out_by_src: &IndexedViewConfig<F>,
        bag2_lanes: &[Bag2LaneConfig<F>],
        lane_u8: Column<Fixed>,
        q_t12_lookup: Selector,
        q_t12_key: Selector,
        q_flag: Selector,
        q_complex: Selector,
        q_order12: Selector,
        q_contrib: Selector,
        q_sum0: Selector,
        q_sum: Selector,
    ) -> Bag1LaneConfig<F> {
        let t12_a = meta.advice_column();
        let t12_b = meta.advice_column();
        let t12_c = meta.advice_column();
        let t12_i_r1 = meta.advice_column();
        let t12_j_r2 = meta.advice_column();
        let t12_r1_eid = meta.advice_column();
        let t12_r2_eid = meta.advice_column();
        let t12_real = meta.advice_column();
        // NOTE: no `enable_equality` on the lane columns; see the Bag2 lane.
        // `run_sum` below is the single lane column that is genuinely copied
        // (into the totals stage).

        // r1 via InByDst: key=B, idx=i_r1 -> val=A, eid=r1_eid
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

        // One membership-or-gap probe per Bag2 lane. Each probe is a complete
        // replica pointed at that lane's map table; they share the two probe
        // selectors and the u8 column because every probe is live on every row.
        let probes: Vec<MapLookupConfig<F>> = bag2_lanes
            .iter()
            .map(|b2| {
                MapLookupChip::<F>::configure_with(
                    meta,
                    b2.agg.map_key,
                    b2.agg.map_val,
                    b2.agg.map_key_next,
                    b2.agg.q_map_tbl,
                    q_flag,
                    q_complex,
                    lane_u8,
                    lane_u8,
                    // probe cells are never copied; keep the c1*c2 replicas out
                    // of the permutation argument
                    false,
                )
            })
            .collect();

        // Bag1 real is boolean, and EVERY probe asks about the same key,
        // pack2(A,C).
        meta.create_gate("bag1 real + key", |m| {
            let q = m.query_selector(q_t12_key);

            let a = m.query_advice(t12_a, Rotation::cur());
            let c = m.query_advice(t12_c, Rotation::cur());
            let key_expr = a * Expression::Constant(F::from(PACK_SHIFT)) + c;

            let real = m.query_advice(t12_real, Rotation::cur());
            let one = Expression::Constant(F::ONE);

            let mut cs = vec![q.clone() * real.clone() * (one - real)];
            for p in probes.iter() {
                cs.push(q.clone() * (m.query_advice(p.key, Rotation::cur()) - key_expr.clone()));
            }
            cs
        });

        // ordering checks for Bag1: A<B and B<C, enabled only if real=1
        let lt_ab = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            lane_u8,
            |m| m.query_selector(q_order12) * m.query_advice(t12_real, Rotation::cur()),
            |m| m.query_advice(t12_a, Rotation::cur()),
            |m| m.query_advice(t12_b, Rotation::cur()),
        );
        let lt_bc = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            lane_u8,
            |m| m.query_selector(q_order12) * m.query_advice(t12_real, Rotation::cur()),
            |m| m.query_advice(t12_b, Rotation::cur()),
            |m| m.query_advice(t12_c, Rotation::cur()),
        );

        // contrib = t12_real * (sum over Bag2 lanes of msg_val) * [A<B] * [B<C]
        let contrib = meta.advice_column();
        meta.create_gate("contrib gate", |m| {
            let q = m.query_selector(q_contrib);

            let real = m.query_advice(t12_real, Rotation::cur());
            let mut msgv = Expression::Constant(F::ZERO);
            for p in probes.iter() {
                msgv = msgv + m.query_advice(p.val, Rotation::cur());
            }
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

        Bag1LaneConfig {
            t12_a,
            t12_b,
            t12_c,
            t12_i_r1,
            t12_j_r2,
            t12_r1_eid,
            t12_r2_eid,
            t12_real,
            probes,
            lt_ab,
            lt_bc,
            contrib,
            run_sum,
        }
    }

    pub fn construct(cfg: Gq4DpConfig<F>) -> Self {
        Self { cfg }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> Gq4DpConfig<F> {
        let (num_bag1_lanes, num_bag2_lanes) = CONFIG_LANES.with(|l| l.get());
        for c in [num_bag1_lanes, num_bag2_lanes] {
            assert!(
                (1..=MAX_LANES).contains(&c),
                "configured lane count {} outside 1..={} (call set_config_lanes first)",
                c,
                MAX_LANES
            );
        }

        let instance = meta.instance_column();
        meta.enable_equality(instance);

        // ---------------- fixed-height sections ----------------
        let in_by_dst = IndexedViewChip::<F>::configure(meta);
        let out_by_src = IndexedViewChip::<F>::configure(meta);

        let lane_u8 = meta.fixed_column();

        // ---------------- Bag2 lanes ----------------
        let q_t34_lookup = meta.complex_selector();
        let q_t34_flag = meta.selector();
        let q_cd = meta.selector();
        let q_t34_msg_in = meta.selector();

        let bag2_lanes: Vec<Bag2LaneConfig<F>> = (0..num_bag2_lanes)
            .map(|_| {
                Self::configure_bag2_lane(
                    meta,
                    &in_by_dst,
                    &out_by_src,
                    lane_u8,
                    q_t34_lookup,
                    q_t34_flag,
                    q_cd,
                    q_t34_msg_in,
                )
            })
            .collect();

        // ---------------- Bag1 lanes ----------------
        // Shared selectors: every Bag1 lane is active on the same rows, so one
        // selector of each kind drives all c1 replicas and all c1*c2 probes.
        let q_t12_lookup = meta.complex_selector();
        let q_t12_key = meta.selector();
        let q_flag = meta.selector();
        let q_complex = meta.complex_selector();
        let q_order12 = meta.selector();
        let q_contrib = meta.selector();
        let q_sum0 = meta.selector();
        let q_sum = meta.selector();

        let bag1_lanes: Vec<Bag1LaneConfig<F>> = (0..num_bag1_lanes)
            .map(|_| {
                Self::configure_bag1_lane(
                    meta,
                    &in_by_dst,
                    &out_by_src,
                    &bag2_lanes,
                    lane_u8,
                    q_t12_lookup,
                    q_t12_key,
                    q_flag,
                    q_complex,
                    q_order12,
                    q_contrib,
                    q_sum0,
                    q_sum,
                )
            })
            .collect();

        // ---------------- cross-lane totals ----------------
        // Row l holds Bag1 lane l's total, copied in from that lane's last
        // run_sum cell; tot_run adds them with a degree-2 accumulator. A flat
        // degree-(1+c1) sum gate would blow past cs.degree() = 7 for c1 > 6.
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

        Gq4DpConfig {
            instance,
            in_by_dst,
            out_by_src,
            lane_u8,
            bag2_lanes,
            q_t34_lookup,
            q_t34_flag,
            q_cd,
            q_t34_msg_in,
            bag1_lanes,
            q_t12_lookup,
            q_t12_key,
            q_flag,
            q_complex,
            q_order12,
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

    /// Assign one Bag2 lane's block of the padded Bag2 pipeline and run its
    /// lane-local aggregation.
    ///
    /// `base` is the global index of this lane's row 0, so lane l covers the
    /// padded rows `[l*lane_rows, (l+1)*lane_rows)`. Rows past `t34.len()` are
    /// pad rows and are witnessed with exactly the pad convention g_sql4_obj
    /// uses: all-zero tuple, real 0, agg input (PAD, 0). View row 0 is (0,0,0),
    /// so a pad row satisfies both membership lookups.
    ///
    /// Returns this lane's message map and the sorted key list the Bag1 probes
    /// need for their gap witnesses.
    #[allow(clippy::too_many_arguments)]
    fn assign_bag2_lane(
        lane: &Bag2LaneConfig<F>,
        region: &mut Region<'_, F>,
        lane_rows: usize,
        base: usize,
        t34: &[(u64, u64, u64, u64, u64, u64, u64)],
        tamper: Tamper,
        pad_row_global: usize,
    ) -> Result<(BTreeMap<u64, u64>, Vec<u64>), Error> {
        let lt_cd_chip = LtChip::<F, NUM_BYTES>::construct(lane.lt_cd);
        let agg_chip = AggSumByKeyChip::<F>::construct(lane.agg.clone());

        let real34 = t34.len();
        let mut agg_in: Vec<(u64, u64)> = vec![(PAD_U64, 0); lane_rows];
        let mut msg_map: BTreeMap<u64, u64> = BTreeMap::new();

        for r in 0..lane_rows {
            let i = base + r;

            let (c, d, a, i_r3, j_r4, r3_eid, r4_eid, real) = if i < real34 {
                let (c, d, a, i_r3, j_r4, r3_eid, r4_eid) = t34[i];
                (c, d, a, i_r3, j_r4, r3_eid, r4_eid, 1u64)
            } else {
                (0, 0, 0, 0, 0, 0, 0, 0u64)
            };

            // Tamper::Bag2PadRow breaks the pad convention on the very last
            // rounding-up row: t34_c moves, so the r3 lookup tuple leaves the
            // indexed view.
            let c_written = if tamper == Tamper::Bag2PadRow && i == pad_row_global {
                7u64
            } else {
                c
            };

            region.assign_advice(
                || "t34_c",
                lane.t34_c,
                r,
                || Value::known(F::from(c_written)),
            )?;
            region.assign_advice(|| "t34_d", lane.t34_d, r, || Value::known(F::from(d)))?;
            region.assign_advice(|| "t34_a", lane.t34_a, r, || Value::known(F::from(a)))?;
            region.assign_advice(
                || "t34_i_r3",
                lane.t34_i_r3,
                r,
                || Value::known(F::from(i_r3)),
            )?;
            region.assign_advice(
                || "t34_j_r4",
                lane.t34_j_r4,
                r,
                || Value::known(F::from(j_r4)),
            )?;
            region.assign_advice(
                || "t34_r3_eid",
                lane.t34_r3_eid,
                r,
                || Value::known(F::from(r3_eid)),
            )?;
            region.assign_advice(
                || "t34_r4_eid",
                lane.t34_r4_eid,
                r,
                || Value::known(F::from(r4_eid)),
            )?;
            region.assign_advice(
                || "t34_real",
                lane.t34_real,
                r,
                || Value::known(F::from(real)),
            )?;

            // LT witness (constraint disabled when real=0)
            lt_cd_chip.assign(
                region,
                r,
                Value::known(F::from(c)),
                Value::known(F::from(d)),
            )?;

            let keep = if real == 1 && c < d { 1u64 } else { 0u64 };
            if keep == 1 {
                let key = pack2(a, c);
                agg_in[r] = (key, 1);
                *msg_map.entry(key).or_default() += 1;
            } else {
                agg_in[r] = (PAD_U64, 0);
            }
        }

        // Lane-local group-by: sortedness, run sums, emit and the map table all
        // live inside this lane's own columns, with the lane's own first/last
        // guards. Nothing crosses a Bag2 lane boundary.
        let _emitted = agg_chip.assign(region, lane_rows, &agg_in)?;

        // Key list for the Bag1 gap witnesses against THIS lane's map.
        // IMPORTANT: must include 0 (the synthetic map row 0) and PAD.
        let mut keys: Vec<u64> = msg_map.keys().copied().collect();
        keys.push(0);
        keys.push(PAD_U64);
        keys.sort();
        keys.dedup();

        Ok((msg_map, keys))
    }

    /// Assign one Bag1 lane's block of the padded Bag1 pipeline.
    ///
    /// Rows past `t12.len()` are pad rows carrying the all-zero tuple, so their
    /// key is 0, which every Bag2 lane's map holds at its synthetic row 0 with
    /// value 0. Returns the lane's last `run_sum` cell together with its total.
    #[allow(clippy::too_many_arguments)]
    fn assign_bag1_lane(
        lane: &Bag1LaneConfig<F>,
        region: &mut Region<'_, F>,
        lane_rows: usize,
        base: usize,
        t12: &[(u64, u64, u64, u64, u64, u64, u64)],
        maps: &[(BTreeMap<u64, u64>, Vec<u64>)],
        tamper: Tamper,
        pad_row_global: usize,
    ) -> Result<(AssignedCell<F, F>, u64), Error> {
        let lt_ab_chip = LtChip::<F, NUM_BYTES>::construct(lane.lt_ab);
        let lt_bc_chip = LtChip::<F, NUM_BYTES>::construct(lane.lt_bc);

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

            // Tamper::Bag1PadRow breaks the pad convention on the very last
            // rounding-up row: t12_a moves but every probe key stays 0.
            let a_written = if tamper == Tamper::Bag1PadRow && i == pad_row_global {
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

            // order witnesses (constraints disabled when real=0)
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

            // One probe per Bag2 lane; the row's message value is their sum.
            let key = if real == 1 { pack2(a, c) } else { 0 };
            let mut msg_total: u64 = 0;
            for (p_idx, (probe, (map, keys))) in
                lane.probes.iter().zip(maps.iter()).enumerate()
            {
                let inside = if key == 0 || map.contains_key(&key) {
                    1u64
                } else {
                    0u64
                };
                let val = if inside == 1 {
                    *map.get(&key).unwrap_or(&0)
                } else {
                    0u64
                };
                let (low, high) = if inside == 0 {
                    match keys.binary_search(&key) {
                        Ok(_) => (0u64, PAD_U64),
                        Err(pos) => {
                            // pos < keys.len() because PAD_U64 is in `keys`
                            // and key < PAD_U64 here.
                            (keys[pos.saturating_sub(1)], keys[pos])
                        }
                    }
                } else {
                    (0u64, PAD_U64)
                };
                msg_total += val;

                region.assign_advice(
                    || "msg_key",
                    probe.key,
                    r,
                    || Value::known(F::from(key)),
                )?;
                region.assign_advice(
                    || "msg_in_set",
                    probe.in_set,
                    r,
                    || Value::known(F::from(inside)),
                )?;
                region.assign_advice(|| "msg_low", probe.low, r, || Value::known(F::from(low)))?;
                region.assign_advice(
                    || "msg_high",
                    probe.high,
                    r,
                    || Value::known(F::from(high)),
                )?;
                // Tamper::DropProbeVal keeps `msg_total` (hence `contrib`)
                // honest but writes 0 into every probe past the first.
                let val_written = if tamper == Tamper::DropProbeVal && p_idx > 0 {
                    0
                } else {
                    val
                };
                region.assign_advice(
                    || "msg_val",
                    probe.val,
                    r,
                    || Value::known(F::from(val_written)),
                )?;

                LtChip::<F, NUM_BYTES>::construct(probe.lt_low).assign(
                    region,
                    r,
                    Value::known(F::from(low)),
                    Value::known(F::from(key)),
                )?;
                LtChip::<F, NUM_BYTES>::construct(probe.lt_high).assign(
                    region,
                    r,
                    Value::known(F::from(key)),
                    Value::known(F::from(high)),
                )?;
            }

            let ab = if a < b { 1u64 } else { 0u64 };
            let bc = if b < c { 1u64 } else { 0u64 };
            let contrib_u64 = real * msg_total * ab * bc;

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
        let c1 = cfg.bag1_lanes.len();
        let c2 = cfg.bag2_lanes.len();
        assert!(lane_rows > 0, "lane_rows must be positive");

        let in_view_chip = IndexedViewChip::<F>::construct(cfg.in_by_dst.clone());
        let out_view_chip = IndexedViewChip::<F>::construct(cfg.out_by_src.clone());

        // Load all LT tables used. The two shared views keep a u8 column each;
        // every lane chip of either kind shares `lane_u8`, so this is 3 load
        // regions for any lane counts.
        in_view_chip.load(layouter)?;
        out_view_chip.load(layouter)?;
        debug_assert_eq!(cfg.bag2_lanes[0].lt_cd.u8, cfg.lane_u8);
        LtChip::<F, NUM_BYTES>::construct(cfg.bag2_lanes[0].lt_cd).load(layouter)?;

        layouter.assign_region(
            || "cycle4_ordered dp-lanes witness",
            |mut region| {
                // -------------------
                // Host-side derivation, shared verbatim with g_sql4_obj
                // -------------------
                let derived = gq4_derive(edges);
                let n_base = derived.in_rows.len();

                in_view_chip.assign(&mut region, n_base, &derived.in_rows)?;
                out_view_chip.assign(&mut region, n_base, &derived.out_rows)?;

                // -------------------
                // Bag2: the first pad-driven pipeline, spread over c2 lanes.
                // -------------------
                let real34 = derived.t34.len();
                let n34 = std::cmp::max(real34 + bag2_pad_extra, 1);
                let cap2 = c2 * lane_rows;
                assert!(
                    cap2 >= n34,
                    "released Bag2 capacity {} exceeds {} lanes of {} rows",
                    n34,
                    c2,
                    lane_rows
                );

                // Shared selectors: enabled once on rows 0..lane_rows, which
                // switches ON every Bag2 lane at the same time. The lane count
                // is a function of the released capacity only, so nothing here
                // can depend on the true bag size.
                for r in 0..lane_rows {
                    cfg.q_t34_lookup.enable(&mut region, r)?;
                    cfg.q_t34_flag.enable(&mut region, r)?;
                    cfg.q_cd.enable(&mut region, r)?;
                    cfg.q_t34_msg_in.enable(&mut region, r)?;
                }

                let mut maps: Vec<(BTreeMap<u64, u64>, Vec<u64>)> = Vec::with_capacity(c2);
                for (l, lane) in cfg.bag2_lanes.iter().enumerate() {
                    maps.push(Self::assign_bag2_lane(
                        lane,
                        &mut region,
                        lane_rows,
                        l * lane_rows,
                        &derived.t34,
                        tamper,
                        cap2 - 1,
                    )?);
                }
                debug_assert_eq!(
                    maps.iter()
                        .map(|(m, _)| m.values().sum::<u64>())
                        .sum::<u64>(),
                    derived.msg_map.values().sum::<u64>(),
                    "lane-local maps must partition the message multiset"
                );

                // -------------------
                // Bag1: the second pad-driven pipeline, spread over c1 lanes.
                // -------------------
                let real12 = derived.t12.len();
                let n12 = std::cmp::max(real12 + bag1_pad_extra, 1);
                let cap1 = c1 * lane_rows;
                assert!(
                    cap1 >= n12,
                    "released Bag1 capacity {} exceeds {} lanes of {} rows",
                    n12,
                    c1,
                    lane_rows
                );

                for r in 0..lane_rows {
                    cfg.q_t12_lookup.enable(&mut region, r)?;
                    cfg.q_t12_key.enable(&mut region, r)?;
                    cfg.q_flag.enable(&mut region, r)?;
                    cfg.q_complex.enable(&mut region, r)?;
                    cfg.q_order12.enable(&mut region, r)?;
                    cfg.q_contrib.enable(&mut region, r)?;
                    if r == 0 {
                        cfg.q_sum0.enable(&mut region, r)?;
                    } else {
                        cfg.q_sum.enable(&mut region, r)?;
                    }
                }

                let mut lane_cells: Vec<AssignedCell<F, F>> = Vec::with_capacity(c1);
                let mut lane_totals: Vec<u64> = Vec::with_capacity(c1);
                for (l, lane) in cfg.bag1_lanes.iter().enumerate() {
                    let (cell, total) = Self::assign_bag1_lane(
                        lane,
                        &mut region,
                        lane_rows,
                        l * lane_rows,
                        &derived.t12,
                        &maps,
                        tamper,
                        cap1 - 1,
                    )?;
                    lane_cells.push(cell);
                    lane_totals.push(total);
                }

                // -------------------
                // The only cross-lane wiring: c1 copy constraints plus a
                // degree-2 accumulator over the c1 Bag1 lane totals.
                // -------------------
                let mut acc: u64 = 0;
                for l in 0..c1 {
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

                let out_row = c1 - 1;
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

/// Wrapper circuit. `lane_rows`, `num_bag1_lanes` and `num_bag2_lanes` are
/// circuit STRUCTURE derived from the publicly released capacities, not
/// witness.
pub struct MyCircuit<F: Field + Ord> {
    pub edges: Vec<Edge>,
    pub bag1_pad_extra: usize,
    pub bag2_pad_extra: usize,
    pub lane_rows: usize,
    pub num_bag1_lanes: usize,
    pub num_bag2_lanes: usize,
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
            num_bag1_lanes: 1,
            num_bag2_lanes: 1,
            tamper: Tamper::None,
            _marker: PhantomData,
        }
    }
}

impl<F: Field + Ord> Circuit<F> for MyCircuit<F> {
    type Config = Gq4DpConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        // keep the lane geometry: it is circuit STRUCTURE, not witness
        Self {
            lane_rows: self.lane_rows,
            num_bag1_lanes: self.num_bag1_lanes,
            num_bag2_lanes: self.num_bag2_lanes,
            ..Self::default()
        }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        Gq4DpChip::<F>::configure(meta)
    }

    fn synthesize(&self, cfg: Self::Config, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        assert_eq!(
            (cfg.bag1_lanes.len(), cfg.bag2_lanes.len()),
            (self.num_bag1_lanes, self.num_bag2_lanes),
            "configured lane counts do not match the circuit: call \
             set_config_lanes(num_bag1_lanes, num_bag2_lanes) before keygen / MockProver"
        );
        let chip = Gq4DpChip::<F>::construct(cfg);
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

/// Geometry the two released capacities imply, plus the edges they were
/// derived from. No SRS is read and no key is built here.
struct Gq4DpSetup {
    edges: Vec<Edge>,
    bag1_pad_extra: usize,
    bag2_pad_extra: usize,
    plan: crate::dp_lane::DpLanePlan,
}

fn dp_lane_setup(dataset: &str, privacy: crate::bench_queries::Privacy) -> Gq4DpSetup {
    let edges = crate::bench_queries::load_graph(dataset);
    let (bag1_pad_extra, bag2_pad_extra) =
        crate::bench_queries::graph_pads("gq4", dataset, &edges, privacy);

    // The degree is PINNED at the Revealing-Join-Size value; the DP releases
    // are absorbed by lanes, not by a bigger domain.
    let k = crate::bench_queries::degree_for("gq4", dataset, crate::bench_queries::Privacy::Rjs);
    let lane_rows = lane_rows_for(k);

    let stats = crate::bench_queries::bag_stats("gq4", &edges);
    let n12 = stats.bag1_size as usize + bag1_pad_extra;
    let n34 = stats.bag2_size as usize + bag2_pad_extra;
    let c1 = lanes_for_capacity(n12, k);
    let c2 = lanes_for_capacity(n34, k);

    // Every column group must fit under the range-table loads plus slack.
    //
    // A lane region is `lane_rows + 1` rows (the group-by writes a sentinel at
    // `lane_rows` for its `Rotation::next()` comparisons), but `lane_rows` is
    // already `2^k - PREAMBLE_ROWS - BLINDING_SLACK`, and the sentinel plus
    // halo2's real blinding factors sit well inside the 64-row BLINDING_SLACK
    // reserve. Counting the sentinel here on TOP of the full reserve
    // overcounts by exactly one row and rejects every legal configuration, so
    // this check uses `lane_rows`, matching g_sql3_obj_dp. The authoritative
    // fit check against the constraint system halo2 actually builds is
    // `tests::structure_fits_base_degree`.
    let tallest = (edges.len() + 2).max(lane_rows).max(c1);
    assert!(
        PREAMBLE_ROWS + tallest + BLINDING_SLACK <= 1usize << k,
        "tallest column group ({} rows) does not fit k={}",
        tallest,
        k
    );

    let plan = crate::dp_lane::DpLanePlan {
        query: "gq4".to_string(),
        dataset: dataset.to_string(),
        k,
        lane_rows,
        lanes: vec![c1, c2],
        capacity: vec![n12, n34],
        true_size: vec![stats.bag1_size as usize, stats.bag2_size as usize],
        pads: vec![bag1_pad_extra, bag2_pad_extra],
    };
    Gq4DpSetup {
        edges,
        bag1_pad_extra,
        bag2_pad_extra,
        plan,
    }
}

/// Geometry only: no SRS is read, no key is built, nothing is proved.
pub fn plan_dp_lanes(
    dataset: &str,
    privacy: crate::bench_queries::Privacy,
) -> crate::dp_lane::DpLanePlan {
    dp_lane_setup(dataset, privacy).plan
}

/// Real IPA proving at the Revealing-Join-Size degree with both DP releases
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
    let setup = dp_lane_setup(dataset, privacy);
    let k = setup.plan.k;
    let (c1, c2) = (setup.plan.lanes[0], setup.plan.lanes[1]);
    let cnt = crate::bench_queries::count_gq4(&setup.edges);

    set_config_lanes(c1, c2);
    let build = || MyCircuit::<Fp> {
        edges: setup.edges.clone(),
        bag1_pad_extra: setup.bag1_pad_extra,
        bag2_pad_extra: setup.bag2_pad_extra,
        lane_rows: setup.plan.lane_rows,
        num_bag1_lanes: c1,
        num_bag2_lanes: c2,
        tamper: Tamper::None,
        _marker: PhantomData,
    };

    let params_path = crate::paths::param_file(k);
    let mut fd = std::fs::File::open(&params_path).unwrap_or_else(|e| {
        panic!(
            "open {}: {} -- generate it with `cargo run --release --bin gen_params -- {}`",
            params_path, e, k
        )
    });
    let params = ParamsIPA::<vesta::Affine>::read(&mut fd).expect("read params");

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

    /// Twelve directed edges over eight nodes: 0->1->3->d->0 for each
    /// d in {4,5,6,7,8}, so there are exactly FIVE ordered 4-cycles, all
    /// sharing the separator pair (A,C) = (0,3).
    ///
    /// The shape is chosen so the message map cannot live in one lane. Both
    /// bags materialize 11 rows, and the single separator key that carries the
    /// answer has multiplicity 5 in Bag2. With `LANE_ROWS_SMALL` = 4 those five
    /// rows cannot all land in one Bag2 lane, so `mock_three_lanes` only
    /// reaches the right answer if the Bag1 probes really do SUM the values
    /// retrieved from every Bag2 lane's lane-local map. See
    /// `heavy_key_spans_multiple_bag2_lanes`, which pins that down.
    fn synthetic_edges() -> Vec<Edge> {
        let mut edges = vec![Edge { src: 0, dst: 1 }, Edge { src: 1, dst: 3 }];
        for d in 4..9u64 {
            edges.push(Edge { src: 3, dst: d });
            edges.push(Edge { src: d, dst: 0 });
        }
        edges
    }

    /// True size of either bag on [`synthetic_edges`].
    const SYNTH_BAG_ROWS: usize = 11;
    /// Rows per lane in the small tests.
    const LANE_ROWS_SMALL: usize = 4;
    /// COUNT(*) on [`synthetic_edges`].
    const SYNTH_CNT: u64 = 5;

    /// Both bags hold 11 rows, so 1 pad row gives a released capacity of 12,
    /// which at 4 rows per lane spans exactly 3 lanes and leaves the last row
    /// of each pipeline as a rounding-up pad.
    fn three_lane_circuit(tamper: Tamper) -> (MyCircuit<Fp>, usize, usize) {
        let edges = synthetic_edges();
        let lane_rows = LANE_ROWS_SMALL;
        let bag1_pad_extra = 1;
        let bag2_pad_extra = 1;
        let c1 = lanes_for(SYNTH_BAG_ROWS + bag1_pad_extra, lane_rows);
        let c2 = lanes_for(SYNTH_BAG_ROWS + bag2_pad_extra, lane_rows);
        assert_eq!((c1, c2), (3, 3));

        let circuit = MyCircuit::<Fp> {
            edges,
            bag1_pad_extra,
            bag2_pad_extra,
            lane_rows,
            num_bag1_lanes: c1,
            num_bag2_lanes: c2,
            tamper,
            _marker: PhantomData,
        };
        (circuit, c1, c2)
    }

    /// The separator key that carries the whole answer has multiplicity 5 in
    /// Bag2 while a lane only holds 4 rows, so its count is necessarily split
    /// over at least two Bag2 lanes. This is what makes `mock_three_lanes` a
    /// real test of the cross-lane message sum rather than of a single map.
    #[test]
    fn heavy_key_spans_multiple_bag2_lanes() {
        use std::collections::BTreeMap;

        let edges = synthetic_edges();
        let derived = crate::graph_sql::g_sql4_obj::gq4_derive(&edges);
        assert_eq!(derived.t12.len(), SYNTH_BAG_ROWS);
        assert_eq!(derived.t34.len(), SYNTH_BAG_ROWS);

        // the heaviest message key, and the lanes its Bag2 rows fall into
        let (&heavy, &mult) = derived
            .msg_map
            .iter()
            .max_by_key(|(_, v)| **v)
            .expect("non-empty message map");
        assert_eq!(mult, 5, "the five 4-cycles all share one separator key");

        let mut per_lane: BTreeMap<usize, u64> = BTreeMap::new();
        for (i, (c, _d, a, _, _, _, _)) in derived.t34.iter().copied().enumerate() {
            if crate::graph_sql::g_sql4_obj::pack2(a, c) == heavy {
                *per_lane.entry(i / LANE_ROWS_SMALL).or_default() += 1;
            }
        }
        assert!(
            per_lane.len() >= 2,
            "multiplicity {} cannot fit one lane of {} rows, got {:?}",
            mult,
            LANE_ROWS_SMALL,
            per_lane
        );
        assert_eq!(per_lane.values().sum::<u64>(), mult);
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

        // lane geometry at the two gq4 base degrees
        assert_eq!(lane_rows_for(BASE_DEGREE), LANE_ROWS);
        assert!(lane_rows_for(22) > 15 * LANE_ROWS);

        // measured releases at eps=0.1: lastfm needs 4 Bag1 lanes and 3 Bag2
        // lanes at k=18 (it would otherwise need k=20)
        assert_eq!(lanes_for_capacity(788_220, 18), 4);
        assert_eq!(lanes_for_capacity(621_070, 18), 3);
        // facebook and wiki collapse to a single lane of each kind at k=22
        assert_eq!(lanes_for_capacity(3_703_832, 22), 1);
        assert_eq!(lanes_for_capacity(3_797_163, 22), 1);
        assert_eq!(lanes_for_capacity(3_142_942, 22), 1);
        assert_eq!(lanes_for_capacity(3_017_452, 22), 1);
        // facebook and wiki at eps=0.01 stay under the structural cap at k=22
        assert!(lanes_for_capacity(36_036_597, 22) <= MAX_LANES);
        assert!(lanes_for_capacity(39_114_118, 22) <= MAX_LANES);
    }

    /// PRIVACY AUDIT. Every lane of a kind must be a FULL structural replica:
    /// no lane may be cheaper than another, or the constraint system would leak
    /// where the real rows stop, which is exactly what the DP release pays to
    /// hide. That is equivalent to the circuit shape being EXACTLY BILINEAR in
    /// (c1, c2) -- constant, plus a fixed cost per Bag1 lane, plus a fixed cost
    /// per Bag2 lane, plus a fixed cost per (Bag1 lane, Bag2 lane) probe pair.
    /// Fit that model on the corners and check it on a whole grid.
    ///
    /// The same fit also pins down the sharing that keeps the preamble bounded:
    /// fixed columns, selectors and shuffles must not grow with c1 at all
    /// (every Bag1 lane is live on the same rows, so they share every selector,
    /// and they hold no range table and no shuffle of their own).
    #[test]
    fn lanes_are_structural_replicas() {
        use halo2_proofs::plonk::ConstraintSystem;

        const N: usize = 7;
        fn shape(c1: usize, c2: usize) -> [i64; N] {
            set_config_lanes(c1, c2);
            let mut cs = ConstraintSystem::<Fp>::default();
            let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
            [
                cs.num_advice_columns() as i64,
                cs.num_fixed_columns() as i64,
                cs.num_selectors() as i64,
                cs.lookups().len() as i64,
                cs.shuffles().len() as i64,
                cs.permutation().get_columns().len() as i64,
                cs.degree() as i64,
            ]
        }

        let s11 = shape(1, 1);
        let s21 = shape(2, 1);
        let s12 = shape(1, 2);
        let s22 = shape(2, 2);

        // per-Bag1-lane, per-Bag2-lane and per-probe-pair costs
        let mut d1 = [0i64; N]; // one more Bag1 lane, at c2 = 1
        let mut d2 = [0i64; N]; // one more Bag2 lane, at c1 = 1
        let mut d12 = [0i64; N]; // the extra probe replica the pair costs
        for i in 0..N {
            d1[i] = s21[i] - s11[i];
            d2[i] = s12[i] - s11[i];
            d12[i] = s22[i] - s21[i] - s12[i] + s11[i];
        }

        for (i, what) in [
            (1usize, "fixed columns"),
            (2, "selectors"),
            (4, "shuffles"),
            (6, "degree"),
        ] {
            assert_eq!(d1[i], 0, "{what} must not grow with the Bag1 lane count");
        }
        assert_eq!(d12[6], 0, "degree must not grow with the probe count");
        for (i, what) in [
            (1usize, "fixed columns"),
            (4, "shuffles"),
            (6, "degree"),
        ] {
            assert_eq!(
                d12[i], 0,
                "{what} must not carry a c1*c2 term (probes share them)"
            );
        }
        assert_eq!(d1[6], 0, "degree must stay 7 as Bag1 lanes are added");
        assert_eq!(d2[6], 0, "degree must stay 7 as Bag2 lanes are added");
        assert_eq!(
            d1[1], 0,
            "a lane must not claim a range-table column of its own"
        );
        assert_eq!(d2[1], 0, "a Bag2 lane shares the one lane u8 column");
        assert!(d1[0] > 0 && d2[0] > 0 && d12[0] > 0, "a lane must cost");
        // only run_sum leaves a Bag1 lane; a Bag2 lane exports nothing
        assert_eq!(
            d1[5], 1,
            "a Bag1 lane must add exactly one permutation column"
        );

        // Now the audit proper: the model must hold EXACTLY off the corners.
        // Any lane that were cheaper than its siblings, or any structure keyed
        // to the true bag size, would break the fit.
        for c1 in 1..=4usize {
            for c2 in 1..=4usize {
                let got = shape(c1, c2);
                for i in 0..N {
                    let want = s11[i]
                        + (c1 as i64 - 1) * d1[i]
                        + (c2 as i64 - 1) * d2[i]
                        + (c1 as i64 - 1) * (c2 as i64 - 1) * d12[i];
                    assert_eq!(
                        got[i], want,
                        "c1={c1} c2={c2}: component {i} is {} but a full replica costs {}",
                        got[i], want
                    );
                }
            }
        }
    }

    /// The COUNT(*) must be invariant to the lane geometry, and the block-wise
    /// split must place every real row of BOTH bags in exactly one lane at the
    /// right offset. The geometries below cover: no rounding-up pad at all (the
    /// last lane of each pipeline is entirely real), real rows straddling a
    /// lane boundary, an odd rounding-up tail, a lane holding a single real
    /// row, and `lane_rows` = 1, where `q_sum` never fires and every lane's
    /// prefix sum is just its one row.
    #[test]
    fn mock_lane_geometries() {
        // (lane_rows, bag1_pad_extra, bag2_pad_extra, expected c1, expected c2)
        let cases: [(usize, usize, usize, usize, usize); 6] = [
            (11, 0, 0, 1, 1), // exactly one full lane each, no pad anywhere
            (6, 0, 0, 2, 2),  // real rows straddle the lane-0/lane-1 boundary
            (3, 1, 1, 4, 4),  // 12 released rows over 12: one rounding-up pad
            (5, 3, 0, 3, 3),  // 14 released rows over 15 vs 11 over 15
            (2, 0, 0, 6, 6),  // the last lane holds a single real row
            (1, 0, 0, 11, 11), // one row per lane: q_sum never fires
        ];
        for (lane_rows, p1, p2, want_c1, want_c2) in cases {
            let c1 = lanes_for(SYNTH_BAG_ROWS + p1, lane_rows);
            let c2 = lanes_for(SYNTH_BAG_ROWS + p2, lane_rows);
            assert_eq!((c1, c2), (want_c1, want_c2), "lane_rows={lane_rows}");

            set_config_lanes(c1, c2);
            let circuit = MyCircuit::<Fp> {
                edges: synthetic_edges(),
                bag1_pad_extra: p1,
                bag2_pad_extra: p2,
                lane_rows,
                num_bag1_lanes: c1,
                num_bag2_lanes: c2,
                tamper: Tamper::None,
                _marker: PhantomData,
            };
            let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]).unwrap();
            assert_eq!(
                prover.verify(),
                Ok(()),
                "lane_rows={lane_rows} c1={c1} c2={c2}"
            );

            // ... and the same geometry must reject a wrong public count
            let prover =
                MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT + 1)]]).unwrap();
            assert!(
                prover.verify().is_err(),
                "lane_rows={lane_rows} c1={c1} c2={c2} accepted a wrong COUNT(*)"
            );
        }
    }

    /// `LANE_ROWS` is arithmetic on a CONSERVATIVE guess at the preamble and
    /// the blinding rows. Get it wrong and nothing fails until a multi-hour
    /// real proof aborts, so check it against the constraint system halo2
    /// actually builds, at the two production cells:
    ///   * lastfm  k = 18, c1 = 4, c2 = 3 (the cell lanes exist for)
    ///   * fb/wiki k = 22, c1 = c2 = 1
    /// A lane occupies `lane_rows + 1` region rows: the group-by writes a
    /// sentinel row at `lane_rows` for its `Rotation::next()` comparisons.
    #[test]
    fn structure_fits_base_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        for (k, c1, c2) in [(18u32, 4usize, 3usize), (22, 1, 1), (18, 1, 1)] {
            set_config_lanes(c1, c2);
            let mut cs = ConstraintSystem::<Fp>::default();
            let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);

            // the contrib gate must stay inside the original circuit's degree
            assert!(
                cs.degree() <= 7,
                "k={} c1={} c2={}: degree {} exceeds g_sql4_obj's 7",
                k,
                c1,
                c2,
                cs.degree()
            );

            let lane_rows = lane_rows_for(k);
            let needed = lane_rows + 1 + cs.blinding_factors() + 1;
            assert!(
                needed <= 1usize << k,
                "k={} c1={} c2={}: a lane needs {} rows (lane_rows {} + sentinel + {} blinding \
                 + 1) but 2^k = {}",
                k,
                c1,
                c2,
                needed,
                lane_rows,
                cs.blinding_factors(),
                1usize << k
            );
        }
    }

    /// c1 = c2 = 3 lanes at k = 11: both padded pipelines really span three
    /// lanes, each Bag2 lane builds its own message map, each Bag1 lane probes
    /// all three of them, and the totals stage adds the three Bag1 totals.
    #[test]
    fn mock_three_lanes() {
        let (circuit, c1, c2) = three_lane_circuit(Tamper::None);
        set_config_lanes(c1, c2);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]).unwrap();
        prover.assert_satisfied();
    }

    /// Corrupting one Bag1 lane row's contribution must break the contrib gate
    /// and that lane's prefix sum.
    #[test]
    fn mock_reject_tampered_lane_contrib() {
        let (circuit, c1, c2) = three_lane_circuit(Tamper::LaneContrib);
        set_config_lanes(c1, c2);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]).unwrap();
        assert!(
            prover.verify().is_err(),
            "tampered lane contribution must not verify"
        );
    }

    /// Corrupting one Bag1 lane total must break the copy constraint out of the
    /// lane and the totals accumulator.
    #[test]
    fn mock_reject_tampered_lane_total() {
        let (circuit, c1, c2) = three_lane_circuit(Tamper::LaneTotal);
        set_config_lanes(c1, c2);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]).unwrap();
        assert!(
            prover.verify().is_err(),
            "tampered lane total must not verify"
        );
    }

    /// Breaking the Bag1 pad convention on the last rounding-up row must break
    /// the "bag1 real + key" gate and the r1 membership lookup.
    #[test]
    fn mock_reject_tampered_bag1_pad_row() {
        let (circuit, c1, c2) = three_lane_circuit(Tamper::Bag1PadRow);
        set_config_lanes(c1, c2);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]).unwrap();
        assert!(
            prover.verify().is_err(),
            "tampered Bag1 pad row must not verify"
        );
    }

    /// Breaking the Bag2 pad convention on the last rounding-up row must break
    /// the r3 membership lookup into the shared indexed view.
    #[test]
    fn mock_reject_tampered_bag2_pad_row() {
        let (circuit, c1, c2) = three_lane_circuit(Tamper::Bag2PadRow);
        set_config_lanes(c1, c2);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]).unwrap();
        assert!(
            prover.verify().is_err(),
            "tampered Bag2 pad row must not verify"
        );
    }

    /// Zeroing the value retrieved from every Bag2 lane past the first, while
    /// leaving `contrib` honest, must be rejected. This is the permanent
    /// negative test for the cross-lane message sum: it fails only because the
    /// "contrib gate" adds all `c2` probe values and each probe's membership
    /// lookup is really enforced.
    #[test]
    fn mock_reject_dropped_probe_val() {
        let (circuit, c1, c2) = three_lane_circuit(Tamper::DropProbeVal);
        set_config_lanes(c1, c2);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]).unwrap();
        assert!(
            prover.verify().is_err(),
            "dropping the message value of a Bag2 lane must not verify"
        );
    }

    /// c1 != c2. The two released capacities are independent, and the real
    /// lastfm cell is c1 = 4, c2 = 3, so the symmetric case the other tests use
    /// must not be the only one exercised: `c1 * c2` probe replicas, `c2` maps
    /// per Bag1 lane and `c1` rows in the totals stage all have to line up.
    #[test]
    fn mock_asymmetric_lanes() {
        let lane_rows = 6;
        let bag1_pad_extra = 1; // n12 = 12 -> 2 Bag1 lanes
        let bag2_pad_extra = 7; // n34 = 18 -> 3 Bag2 lanes
        let c1 = lanes_for(SYNTH_BAG_ROWS + bag1_pad_extra, lane_rows);
        let c2 = lanes_for(SYNTH_BAG_ROWS + bag2_pad_extra, lane_rows);
        assert_eq!((c1, c2), (2, 3));

        set_config_lanes(c1, c2);
        let circuit = MyCircuit::<Fp> {
            edges: synthetic_edges(),
            bag1_pad_extra,
            bag2_pad_extra,
            lane_rows,
            num_bag1_lanes: c1,
            num_bag2_lanes: c2,
            tamper: Tamper::None,
            _marker: PhantomData,
        };
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]).unwrap();
        prover.assert_satisfied();
    }

    /// A released capacity large enough that the LAST Bag2 lane holds no real
    /// row at all. Its lane-local aggregator then sees an all-PAD input, emits
    /// nothing, and its map table degenerates to the single (0,0) row followed
    /// by PAD. Every Bag1 row must still be able to prove absence in it, which
    /// is the one gap bracket (0, PAD) that lane offers.
    #[test]
    fn mock_all_pad_bag2_lane() {
        let lane_rows = LANE_ROWS_SMALL; // 4
        let bag1_pad_extra = 1; // n12 = 12 -> 3 Bag1 lanes
        let bag2_pad_extra = 5; // n34 = 16 -> 4 Bag2 lanes, the last all pad
        let c1 = lanes_for(SYNTH_BAG_ROWS + bag1_pad_extra, lane_rows);
        let c2 = lanes_for(SYNTH_BAG_ROWS + bag2_pad_extra, lane_rows);
        assert_eq!((c1, c2), (3, 4));
        // lane 3 covers padded rows 12..16, and the bag only has 11 real rows
        assert!(3 * lane_rows >= SYNTH_BAG_ROWS);

        set_config_lanes(c1, c2);
        let circuit = MyCircuit::<Fp> {
            edges: synthetic_edges(),
            bag1_pad_extra,
            bag2_pad_extra,
            lane_rows,
            num_bag1_lanes: c1,
            num_bag2_lanes: c2,
            tamper: Tamper::None,
            _marker: PhantomData,
        };
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]).unwrap();
        prover.assert_satisfied();
    }

    /// c1 = c2 = 1 degenerates to g_sql4_obj's single-group layout: both
    /// circuits must accept the same input and expose the same COUNT(*).
    #[test]
    fn mock_single_lane_matches_g_sql4_obj() {
        let edges = synthetic_edges();
        let public: Vec<Fp> = vec![Fp::from(SYNTH_CNT)];

        let baseline = crate::graph_sql::g_sql4_obj::MyCircuit::<Fp> {
            edges: edges.clone(),
            bag1_pad_extra: 1,
            bag2_pad_extra: 1,
            _marker: PhantomData,
        };
        let prover = MockProver::run(11, &baseline, vec![public.clone()]).unwrap();
        prover.assert_satisfied();

        let lane_rows = 16; // n12 = n34 = 11 + 1 = 12 fits one lane of each kind
        let c1 = lanes_for(SYNTH_BAG_ROWS + 1, lane_rows);
        let c2 = lanes_for(SYNTH_BAG_ROWS + 1, lane_rows);
        assert_eq!((c1, c2), (1, 1));
        set_config_lanes(c1, c2);
        let laned = MyCircuit::<Fp> {
            edges,
            bag1_pad_extra: 1,
            bag2_pad_extra: 1,
            lane_rows,
            num_bag1_lanes: c1,
            num_bag2_lanes: c2,
            tamper: Tamper::None,
            _marker: PhantomData,
        };
        let prover = MockProver::run(11, &laned, vec![public]).unwrap();
        prover.assert_satisfied();
    }

    /// Real IPA proving at the Revealing-Join-Size degree with both DP releases
    /// hosted in lanes. Multi-hour on the production datasets, so it is ignored
    /// by default. Run it explicitly, for example:
    ///
    ///   VPJOIN_DATASET=lastfm VPJOIN_PRIVACY=dp VPJOIN_EPS=0.1 \
    ///   VPJOIN_DELTA=1e-5 cargo test --release \
    ///     graph_sql::g_sql4_obj_dp::tests::test_dp_lanes -- --ignored --nocapture
    ///
    /// The same code path is what `cargo run --bin dp_lane_bench -- gq4` drives.
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

        let proof_path = crate::paths::proof_file("proof_gq4_dp_lanes");
        let t_total = Instant::now();
        let run = super::run_dp_lanes(&dataset, privacy, 1, Some(&proof_path));
        let p = &run.plan;
        println!(
            "[gq4 dp-lanes] dataset={} privacy={} pads: bag1={} bag2={}",
            dataset,
            privacy.label(),
            p.pads[0],
            p.pads[1]
        );
        println!(
            "[gq4 dp-lanes] bag1_true={} n12={} lanes={} | bag2_true={} n34={} lanes={} \
             | lane_rows={} effective_capacity={}/{} | probe replicas={}",
            p.true_size[0],
            p.capacity[0],
            p.lanes[0],
            p.true_size[1],
            p.capacity[1],
            p.lanes[1],
            p.lane_rows,
            p.lanes[0] * p.lane_rows,
            p.lanes[1] * p.lane_rows,
            p.lanes[0] * p.lanes[1]
        );
        println!("Proof written to: {}", proof_path);
        println!(
            "[gq4 dp-lanes] bag1_lanes={} bag2_lanes={} keygen {:.2}s prove {:.2}s \
             verify {:.2}s total prove+verify {:?}",
            p.lanes[0],
            p.lanes[1],
            run.keygen_s,
            run.prove_mean(),
            run.verify_s,
            t_total.elapsed()
        );
    }
}
