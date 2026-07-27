//! GQ3 (triangle) under the One-Pass OBJ, with the DP release hosted in lanes.
//!
//! This is the multi-lane DP-padding variant of `g_sql3_obj.rs`: same query,
//! same Path+Closer decomposition, same host-side derivation, same released
//! capacity and padding scheme, but the padded Bag1 pipeline is spread over up
//! to [`MAX_LANES`] structural replicas of the Bag1 column group so a release
//! larger than the domain still fits at the pinned degree.
//!
//! It now carries the same One-Pass OBJ conditions its single-group sibling
//! does, over a prover-supplied partition of each bag into a clean part `R^c`
//! and a residual part `R^r`:
//!
//!   (7)  `R_i == R_i^c U R_i^r`                    Conservation
//!   (9)  `pi_K(Bag1^c) == pi_K(Bag2^c)`            Pairwise Consistency
//!   (10) `|R^c join| == |R join|`                  Cardinality Preservation
//!
//! Condition (8), Non-Membership, is deliberately not enforced, exactly as in
//! the sibling.
//!
//! # How each condition holds over the UNION of the lanes
//!
//! A condition that is checked per lane in a way that lets a prover move a
//! violation into another lane is not checked at all, so each of the three is
//! spelled out here.
//!
//! * (7) is per lane and that is *stronger* than the union statement, not
//!   weaker. Lane `l` owns the padded Bag1 rows
//!   `[l*lane_rows, l*lane_rows + lane_live_rows(l))` and its own 8-column
//!   partition group, and one shuffle per lane forces lane `l`'s multiset of
//!   `(attributes, clean indicator)` to equal its own group's
//!   `[clean | residual | pad]` multiset. The lane row blocks are disjoint and
//!   cover the whole released capacity EXACTLY -- every lane but the last is a
//!   full `lane_rows` and the last stops at the capacity -- so summing the `c`
//!   lane statements gives `Bag1 == Bag1^c U Bag1^r`. Nothing can be moved
//!   between lanes, because a lane's shuffle only ever sees its own columns.
//!   Bag2 is not laned (its height is |E| + pads, public in every regime), so
//!   its Conservation Check is the sibling's, verbatim.
//!
//! * (9) has one direction per orientation, and only one of them is a union
//!   statement over the lanes.
//!   - `pi_K(Bag1^c) subset pi_K(Bag2^c)` is `c` lookups, one per lane, all
//!     into the SAME table (the clean key column of the Bag2 rows). A union of
//!     inputs is contained in a table iff every input is, so per-lane here IS
//!     the union statement.
//!   - `pi_K(Bag2^c) subset pi_K(Bag1^c)` is the dangerous direction: the
//!     table is the union of `c` per-lane clean key columns and a lookup cannot
//!     express "in lane 1 OR in lane 2". Doing it per lane would demand that
//!     EVERY lane contain EVERY clean Bag2 key, which no honest witness
//!     satisfies. It is realized instead by a prover-supplied host bit per
//!     (lane, Bag2 row): one gate forces the host bits of a CLEAN Bag2 row to
//!     sum to exactly 1 over the lanes, and lane `l`'s lookup checks the rows
//!     it hosts against its own clean key column. So the prover has to name a
//!     lane for every clean Bag2 tuple and that lane then has to really contain
//!     the key, which is exactly membership in the union. A prover that cannot
//!     name one fails the coverage gate.
//!
//! * (10) is accumulated ACROSS the lanes and compared exactly once. Bag2 is
//!   the only child of the cluster tree and it is a leaf, so its child stage is
//!   the sibling's `configure_cp_agg` over the (unlaned) Bag2 rows, keyed by the
//!   packed separator, carrying both channels. Two things then make the root
//!   side cheap and lane-safe:
//!     - the input channel needs no second traversal at all. On this tree
//!       `sigma_all(key)` is the number of real Bag2 edges with that key, which
//!       is exactly the `msg_val` the answer is already read off, so
//!       `mu_all = t12_real * [A<B] * [B<C] * msg_val` IS the existing `contrib`
//!       column and the input-side root sum IS `tot_run`, the cross-lane total
//!       the COUNT(*) is taken from.
//!     - the clean channel is one column `mu_cln` per lane row, pinned by a
//!       single lookup `(cln12 * key, mu_cln)` into the child table's
//!       `(key, sum_cln)` pair. With `cln12 = 0` the key factor is 0 and the
//!       only table row with key 0 is the pinned dummy `(0, 0, 0)`, so the
//!       lookup alone forces `mu_cln = cln12 * sigma_cln(key)` with no slack in
//!       either direction and no separate gap witness.
//!   Each lane prefix-sums `mu_cln` locally, the `c` lane totals are copied into
//!   the cross-lane totals stage and accumulated there, and ONE gate at the last
//!   totals row compares the two grand totals. There is no per-lane equality, so
//!   a hidden tuple's contribution cannot be parked in a lane that balances on
//!   its own.
//!
//! Everything else, including the message table, the ordering checks, the
//! COUNT(*) prefix sum and the whole lane geometry, is unchanged.

use halo2_proofs::plonk::Expression;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};
use crate::circuits::card_preserve::{
    assign_cp_agg, build_cp_stage, configure_cp_agg, CpAggConfig,
};
use crate::data::graph_data_processing::Edge;

// One shared definition of the PAD / packing conventions, the chips and the
// host-side witness derivation (from g_sql3_obj).
use super::g_sql3_obj::{
    gq3_derive, pack2, AggSumByKeyChip, AggSumByKeyConfig, Field, IndexedViewChip,
    IndexedViewConfig, MapLookupChip, MapLookupConfig, NUM_BYTES, PACK_SHIFT, PAD_U64,
};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::marker::PhantomData;

/// Rows the u8 range-table `load` regions may claim ahead of the witness
/// region. Six distinct u8 columns are loaded: the four shared ones (the sort
/// chip of each indexed view, and AggSumByKey's two sort chips), the single
/// column every lane's four Lt chips share, and the one column every Lt chip of
/// the Cardinality Preservation child stage shares. Each `load` writes 256 fixed
/// rows.
///
/// The reserve stays at five columns' worth on purpose. It is a conservative
/// margin, not a count: `SimpleFloorPlanner` starts a region at the first row
/// free in the columns that region touches, and each `load` touches only its own
/// fixed column, so all of them begin at row 0 and none of them displaces the
/// witness region. The constant is also part of the released capacity scheme
/// (`lane_rows_for` and every lane count derived from it), which must not move
/// when a soundness condition is added.
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

/// Live rows of lane `l`: the rows that are assigned and gated.
///
/// Every lane but the last is full. The LAST one stops at the released
/// capacity rather than at `c * lane_rows`, so the rows the lane count rounds
/// up to never exist: they are not assigned, no selector covers them and no
/// table side reads them.
///
/// TWO TIERS OF PADDING MEET HERE AND ONLY THE SECOND IS DROPPED.
///   * true bag size -> `capacity` is the DP pad. It is INSIDE the capacity, so
///     it stays assigned and stays gated exactly as before. Hiding the true
///     size is what the release pays for.
///   * `capacity` -> `c * lane_rows` is pure lane quantization, and that is what
///     goes away. Nothing but the arithmetic of the lane count put it there.
///
/// The boundary is a function of `capacity`, `c` and `lane_rows` only, all
/// three of which are public: the capacity is the released value the harness
/// prints and already the thing that fixes the lane count. No row range in this
/// file may be a function of the true bag size, since selectors are fixed
/// columns and a selector pattern lands in the verifying key.
pub fn lane_live_rows(l: usize, num_lanes: usize, lane_rows: usize, capacity: usize) -> usize {
    assert!(l < num_lanes, "lane {} outside 0..{}", l, num_lanes);
    assert!(
        capacity > (num_lanes - 1) * lane_rows && capacity <= num_lanes * lane_rows,
        "released capacity {} is not hosted by exactly {} lanes of {} rows",
        capacity,
        num_lanes,
        lane_rows
    );
    if l + 1 < num_lanes {
        lane_rows
    } else {
        capacity - (num_lanes - 1) * lane_rows
    }
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
///
/// It is a circuit field rather than a global flag on purpose: the tests of this
/// module run in parallel in one process, and a `static` hook would let one
/// test's fault injection leak into another's witness.
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
    /// Break the pad-row convention on the LAST row of the released capacity,
    /// which is the last live row of the last lane (the "bag1 real + key" gate
    /// and the r1 membership lookup must reject).
    PadRow,
    /// Move one joinable Bag2 tuple to the residual side and re-reduce both bags
    /// around it, so the partition still conserves both bags and the two clean
    /// sides still agree on the separator key. Only condition (10) can see this,
    /// and only because its root sums are accumulated across the lanes.
    HideCleanTuple,
    /// Skip the semijoin reduction and declare every real tuple clean. Both bags
    /// are still conserved and both channels of condition (10) then carry the
    /// same multiplicity on every row, so only condition (9) can reject it.
    MarkAllClean,
    /// Point one clean Bag2 tuple's host bit at a lane that does NOT hold a clean
    /// wedge with its key. The coverage gate still sees exactly one host, so only
    /// that lane's Pairwise Consistency lookup can reject it. This is what
    /// separates "some lane holds this key" from "the prover may name any lane".
    HostWrongLane,
    /// Move the LAST clean row of every lane's partition group into the residual
    /// block by flipping its flag, leaving the bag's own indicator honest. The
    /// flag stays boolean and non-increasing, so only that lane's Conservation
    /// shuffle can reject it: this is what shows condition (7) is not vacuous and
    /// that it really is enforced lane by lane.
    PartitionFlag,
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

    // ---------------- One-Pass OBJ, this lane's share ----------------
    /// bound clean indicator on this lane's Bag1 rows: 1 only on a row that is
    /// real, passes A<B<C and went to `Bag1^c`
    cln12: Column<Advice>,
    /// `real == [B != 0]`, so a pad row cannot be promoted to a real tuple and a
    /// real row cannot be demoted while keeping its attributes
    iz_t12_pad: IsZeroConfig<F>,
    /// condition (7): `[clean | residual | pad]` side of this lane's
    /// Conservation Check; columns are (A,B,C,i_r1,j_r2,r1_eid,r2_eid,flag)
    part12: [Column<Advice>; 8],
    /// `cln12 * pack2(A,C)`: the clean separator key of this lane, and the only
    /// column conditions (9) and (10) read the key through. Materializing the
    /// product keeps every lookup that reads it at input degree 2.
    ck12: Column<Advice>,
    /// condition (9), the union direction: host bit on the BAG2 rows, 1 on the
    /// clean Bag2 rows whose key this lane is claimed to carry
    host3: Column<Advice>,
    /// condition (10): `cln12 * sigma_cln(pack2(A,C))`, pinned by one lookup
    mu_cln: Column<Advice>,
    /// lane-local prefix sum of `mu_cln`; its last cell is copied into the
    /// cross-lane totals stage
    cln_run: Column<Advice>,
}

/// The row selectors one lane's machinery reads. Bundled because there are
/// thirteen of them and because there are now TWO sets: a selector is one fixed
/// column, so it is on or off for every lane at once, and the last lane is
/// live on fewer rows than the rest.
///
/// The `full` set serves lanes `0..c-1`, enabled on all `lane_rows` rows. The
/// `last` set serves lane `c-1`, enabled on its [`lane_live_rows`] rows only,
/// and folds the thirteen down to four columns, one per distinct ROW RANGE,
/// which is all a selector is: `0..live` for every gate, `0..live` for every
/// lookup and the shuffle (complex, since only a complex selector may be read
/// by one), `1..live` for the two prefix sums and `0..live-1` for the gate that
/// reads `Rotation::next()`. The full lanes keep their thirteen so their layout
/// is unchanged.
#[derive(Clone, Copy, Debug)]
struct LaneSelectors {
    q_t12_lookup: Selector,
    q_t12_key: Selector,
    q_flag: Selector,
    q_complex: Selector,
    q_order: Selector,
    q_contrib: Selector,
    q_sum0: Selector,
    q_sum: Selector,
    q_cln_bind: Selector,
    q_part12: Selector,
    q_part12_mono: Selector,
    q_perm12_in: Selector,
    q_perm12_out: Selector,
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

    // ---------------- Bag2's share of the One-Pass OBJ (never laned) -------
    /// bound clean indicator on the Bag2 rows
    cln3: Column<Advice>,
    /// `real == [C != 0]`
    iz_t3_pad: IsZeroConfig<F>,
    q_cln3_bind: Selector,
    /// `cln3 * pack2(A,C)`: the clean separator key of Bag2. It is the table of
    /// one direction of condition (9) and the input of the other, so both
    /// directions run between the two clean key columns with nothing in between.
    ck3: Column<Advice>,
    /// condition (7) for Bag2: `[clean | residual | pad]` side, columns are
    /// (C,A,j_r3,r3_eid,flag)
    part3: Vec<Column<Advice>>,
    perm_bag2: PermAnyConfig,
    q_part3: Selector,
    q_part3_mono: Selector,
    /// condition (9): exactly one lane hosts each clean Bag2 tuple
    q_host3: Selector,
    /// condition (10): the child stage of the single cluster-tree edge, over the
    /// Bag2 rows. Its map is a lookup table, so every lane reads it without any
    /// stitching, exactly like `agg_msg`.
    cp_agg: CpAggConfig<F, NUM_BYTES>,
    cp_u8: Column<Fixed>,

    // ---------------- lanes (shared selectors, per-lane columns) ----------
    // Lanes 0..c-1 are active on the SAME rows 0..lane_rows, so one selector of
    // each kind serves all of them; only the columns replicate. Lane c-1 stops
    // at the released capacity and reads `last` instead.
    lanes: Vec<LaneConfig<F>>,
    lane_u8: Column<Fixed>, // one u8 range table for all 4c lane Lt chips
    full: LaneSelectors,    // rows 0..lane_rows, every lane but the last
    last: LaneSelectors,    // rows 0..lane_live_rows(c-1, ..), the last lane

    // ---------------- cross-lane totals (the only stitch) ------------------
    lane_tot: Column<Advice>, // row l = lane l's total, by copy constraint
    tot_run: Column<Advice>,  // prefix sum of lane_tot over rows 0..c
    lane_cln: Column<Advice>, // row l = lane l's clean total, by copy constraint
    tot_cln: Column<Advice>,  // prefix sum of lane_cln over rows 0..c
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

    /// One multiset equality between two column groups on the same rows, with
    /// NO `enable_equality` on either group.
    ///
    /// `PermAnyChip::configure` builds the same shuffle, but it also enables
    /// equality on every column it is handed, and halo2 charges the permutation
    /// argument for every column that has equality enabled whether or not a copy
    /// constraint ever touches it. A lane's Conservation Check reads 16 per-lane
    /// columns that no copy constraint reaches, so routing it through
    /// `PermAnyChip` would grow the permutation argument by 16 columns per lane
    /// for nothing. Bag2's Conservation Check is not laned and does use
    /// `PermAnyChip`, exactly as the sibling does.
    fn lane_shuffle(
        meta: &mut ConstraintSystem<F>,
        name: &'static str,
        q_in: Selector,
        q_out: Selector,
        input: Vec<Column<Advice>>,
        table: Vec<Column<Advice>>,
    ) {
        assert_eq!(input.len(), table.len());
        meta.shuffle(name, move |m| {
            let qi = m.query_selector(q_in);
            let qo = m.query_selector(q_out);
            input
                .iter()
                .zip(table.iter())
                .map(|(i, t)| {
                    (
                        qi.clone() * m.query_advice(*i, Rotation::cur()),
                        qo.clone() * m.query_advice(*t, Rotation::cur()),
                    )
                })
                .collect()
        });
    }

    /// One full structural replica of the Bag1 column group. Every lane gets
    /// the same gates, the same lookups and the same shuffle; only the columns
    /// and the row selectors differ.
    #[allow(clippy::too_many_arguments)]
    fn configure_lane(
        meta: &mut ConstraintSystem<F>,
        in_by_dst: &IndexedViewConfig<F>,
        out_by_src: &IndexedViewConfig<F>,
        agg_msg: &AggSumByKeyConfig<F>,
        cp_agg: &CpAggConfig<F, NUM_BYTES>,
        ck3: Column<Advice>,
        q_t3_tbl: Selector,
        lane_u8: Column<Fixed>,
        sel: &LaneSelectors,
    ) -> LaneConfig<F> {
        let LaneSelectors {
            q_t12_lookup,
            q_t12_key,
            q_flag,
            q_complex,
            q_order,
            q_contrib,
            q_sum0,
            q_sum,
            q_cln_bind,
            q_part12,
            q_part12_mono,
            q_perm12_in,
            q_perm12_out,
        } = *sel;
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
        //
        // Every table side is multiplied by the view's own `q_tbl`, which the
        // shared `IndexedViewChip::assign` enables on exactly the rows
        // [0, n_base) its permutation, sort and idx recurrences cover. Left
        // ungated (as this file did), the view's one-past-the-end sentinel row
        // and EVERY row above it -- and in a laned circuit there are `lane_rows`
        // of them, far more than the view is tall -- are live table entries that
        // no constraint of the view touches. That is an unlimited supply of
        // forged view tuples: a prover writes any (key, idx, val, eid) it likes
        // on a free row and a lane row then "proves" a join it never had. A
        // lookup input is 0 wherever its own selector is off and a table row with
        // `q_tbl` off contributes 0 on every column, so the all-zero tuple stays
        // in the table and the gated-off input rows still cost nothing.
        meta.lookup_any("bag1 r1 from in_by_dst", |m| {
            let q = m.query_selector(q_t12_lookup);
            let t = m.query_selector(in_by_dst.q_tbl);
            vec![
                (
                    q.clone() * m.query_advice(t12_b, Rotation::cur()),
                    t.clone() * m.query_advice(in_by_dst.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t12_i_r1, Rotation::cur()),
                    t.clone() * m.query_advice(in_by_dst.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t12_a, Rotation::cur()),
                    t.clone() * m.query_advice(in_by_dst.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t12_r1_eid, Rotation::cur()),
                    t * m.query_advice(in_by_dst.sorted_eid, Rotation::cur()),
                ),
            ]
        });
        // r2 via OutBySrc: key=B, idx=j_r2 -> val=C, eid=r2_eid
        meta.lookup_any("bag1 r2 from out_by_src", |m| {
            let q = m.query_selector(q_t12_lookup);
            let t = m.query_selector(out_by_src.q_tbl);
            vec![
                (
                    q.clone() * m.query_advice(t12_b, Rotation::cur()),
                    t.clone() * m.query_advice(out_by_src.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t12_j_r2, Rotation::cur()),
                    t.clone() * m.query_advice(out_by_src.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t12_c, Rotation::cur()),
                    t.clone() * m.query_advice(out_by_src.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t12_r2_eid, Rotation::cur()),
                    t * m.query_advice(out_by_src.sorted_eid, Rotation::cur()),
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

        // ============ One-Pass OBJ: this lane's share ============
        let cln12 = meta.advice_column();
        let ck12 = meta.advice_column();
        let host3 = meta.advice_column();
        let mu_cln = meta.advice_column();

        // `real == [this row carries a tuple at all]`. The real bit is the
        // input-channel multiplicity of a Bag1 row, and it is NOT inside the
        // Conservation shuffle (the shuffled tuple is (attributes, indicator)),
        // so booleanity alone leaves it free advice on a pad row. Node ids are
        // SHIFT_ID-shifted, so B is nonzero on every real wedge and zero on the
        // all-zero pad tuple, and the r1/r2 lookups only accept a zero key
        // against the (0,0,0) dummy view row. So `real == [B != 0]` is exactly
        // "this row carries a tuple", at one advice column and one degree-3
        // constraint per lane.
        let aux_t12_pad = meta.advice_column();
        let iz_t12_pad = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_cln_bind),
            |m| m.query_advice(t12_b, Rotation::cur()),
            aux_t12_pad,
        );

        // The bound clean indicator: boolean, implies the row is real and passes
        // the bag's own predicate A<B<C, i.e. cln12 == t12_real * [A<B] * [B<C] *
        // c_1. The last polynomial pins the clean key column this lane's three
        // OBJ lookups read, so each of them stays at input degree 2.
        meta.create_gate("bag1 clean indicator (lane)", |m| {
            let q = m.query_selector(q_cln_bind);
            let cln = m.query_advice(cln12, Rotation::cur());
            let real = m.query_advice(t12_real, Rotation::cur());
            let ab = lt_ab.is_lt(m, None);
            let bc = lt_bc.is_lt(m, None);
            let one = Expression::Constant(F::ONE);
            let key = m.query_advice(msg_lookup.key, Rotation::cur());
            vec![
                q.clone() * cln.clone() * (one.clone() - cln.clone()),
                q.clone() * cln.clone() * (one.clone() - real.clone()),
                q.clone() * cln.clone() * (one.clone() - ab),
                q.clone() * cln.clone() * (one.clone() - bc),
                q.clone() * (real - (one - iz_t12_pad.expr())),
                q * (m.query_advice(ck12, Rotation::cur()) - cln * key),
            ]
        });

        // ---- condition (7), this lane's Conservation Check ----
        // Lane l's Bag1 rows, carrying the bound indicator, are a permutation of
        // lane l's own [clean | residual | pad] group. The flag is 1 over the
        // clean rows and 0 after them, so the multiset equality forces the
        // indicator on a bag row to mark exactly the tuples that went to R^c.
        // Per lane is STRONGER than the union statement: the lane row blocks are
        // disjoint and cover the whole released capacity, so summing the c lane
        // equalities gives Bag1 == Bag1^c U Bag1^r, and nothing can be shifted
        // between lanes because a lane's shuffle only sees its own columns.
        let part12: [Column<Advice>; 8] = [(); 8].map(|_| meta.advice_column());
        Self::lane_shuffle(
            meta,
            "bag1 conservation (lane)",
            q_perm12_in,
            q_perm12_out,
            vec![
                t12_a, t12_b, t12_c, t12_i_r1, t12_j_r2, t12_r1_eid, t12_r2_eid, cln12,
            ],
            part12.to_vec(),
        );

        // The partition side's flag is boolean and non-increasing, so the group
        // really is [R^c block | R^r and pad block] with no free tail. The block
        // extents are not selector ranges here, which is why nothing about the
        // clean/residual ratio is baked into the vk of this circuit.
        // The two live under separate selectors, so they must be separate gates:
        // a gate is checked for cell assignment wherever ANY of its selectors is
        // on, and the monotonicity polynomial reads `Rotation::next()`, which does
        // not exist at the last row of the lane.
        {
            let flag = part12[7];
            meta.create_gate("bag1 partition flag is boolean (lane)", move |m| {
                let q = m.query_selector(q_part12);
                let one = Expression::Constant(F::ONE);
                let f = m.query_advice(flag, Rotation::cur());
                vec![q * f.clone() * (one - f)]
            });
            meta.create_gate("bag1 partition flag is non-increasing (lane)", move |m| {
                let qm = m.query_selector(q_part12_mono);
                let one = Expression::Constant(F::ONE);
                let f = m.query_advice(flag, Rotation::cur());
                let f_next = m.query_advice(flag, Rotation::next());
                vec![qm * f_next * (one - f)]
            });
        }

        // ---- condition (9), Pairwise Consistency on the packed (A,C) key ----
        // Direction 1: pi_K(Bag1^c restricted to this lane) subset pi_K(Bag2^c).
        // Every lane looks into the SAME table, so the union of the lane inputs
        // is contained iff each lane's input is: per lane IS the union statement
        // in this direction. Both sides are the two clean key columns themselves,
        // with no intermediate deduplicated table (which nothing would bind to
        // the relation it claims to enumerate). 0 is in the table on every row
        // where the table selector is off, and a real packed key is at least
        // PACK_SHIFT + 1, so the containment is over the real clean keys only.
        meta.lookup_any("pw: bag1^c key in bag2^c (lane)", |m| {
            vec![(
                m.query_selector(q_t12_lookup) * m.query_advice(ck12, Rotation::cur()),
                m.query_selector(q_t3_tbl) * m.query_advice(ck3, Rotation::cur()),
            )]
        });

        // Direction 2, the union direction: pi_K(Bag2^c) subset pi_K(Bag1^c),
        // whose table is the union of the c per-lane clean key columns. A lookup
        // cannot say "in lane 1 OR in lane 2", and asking every lane to contain
        // every clean Bag2 key would reject every honest witness. So the prover
        // names the hosting lane with `host3`, and the gate "pw: every clean bag2
        // tuple is hosted by exactly one lane" in `configure` forces those bits
        // to sum to 1 on each clean Bag2 row. This lookup then checks the rows
        // this lane claims against this lane's own clean keys. A host bit on a
        // non-clean Bag2 row is harmless, since `ck3` is already 0 there.
        meta.lookup_any("pw: bag2^c key in bag1^c (lane host)", |m| {
            let gate = m.query_selector(q_t3_tbl) * m.query_advice(host3, Rotation::cur());
            vec![(
                gate * m.query_advice(ck3, Rotation::cur()),
                m.query_selector(q_t12_lookup) * m.query_advice(ck12, Rotation::cur()),
            )]
        });

        // ---- condition (10), the clean channel of the Cardinality
        // Preservation Check ----
        // `mu_cln` must be `cln12 * sigma_cln(pack2(A,C))`, and this single
        // lookup is the whole of it. With cln12 = 1 the pair (key, mu_cln) has to
        // be a row of the child table, whose keys are unique, so mu_cln is that
        // key's clean sum exactly. With cln12 = 0 the key factor is 0 and the
        // only table row with key 0 is the dummy (0,0,0) the child stage pins, so
        // mu_cln is forced to 0. There is no slack in either direction, which is
        // why no separate gap witness is needed on this channel: the absence
        // certification the input channel needs lives in the message probe above,
        // which is the same fetch with the same gap proof.
        meta.lookup_any("cp: clean sigma from the child table (lane)", |m| {
            let q = m.query_selector(q_t12_lookup);
            let t = m.query_selector(cp_agg.q_map_tbl);
            vec![
                (
                    q.clone() * m.query_advice(ck12, Rotation::cur()),
                    t.clone() * m.query_advice(cp_agg.map[0], Rotation::cur()),
                ),
                (
                    q * m.query_advice(mu_cln, Rotation::cur()),
                    t * m.query_advice(cp_agg.map[2], Rotation::cur()),
                ),
            ]
        });

        // lane-local prefix sum of the clean channel, on the same two selectors
        // as the answer's prefix sum. Its last cell is copied into the cross-lane
        // totals stage, where the two grand totals meet in ONE equality: a
        // per-lane equality is exactly the split a prover would exploit.
        let cln_run = meta.advice_column();
        meta.enable_equality(cln_run);
        meta.create_gate("cp: clean sum first row of the lane", |m| {
            let q = m.query_selector(q_sum0);
            vec![
                q * (m.query_advice(cln_run, Rotation::cur())
                    - m.query_advice(mu_cln, Rotation::cur())),
            ]
        });
        meta.create_gate("cp: clean sum accumulate in the lane", |m| {
            let q = m.query_selector(q_sum);
            vec![
                q * (m.query_advice(cln_run, Rotation::cur())
                    - (m.query_advice(cln_run, Rotation::prev())
                        + m.query_advice(mu_cln, Rotation::cur()))),
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
            cln12,
            iz_t12_pad,
            part12,
            ck12,
            host3,
            mu_cln,
            cln_run,
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

        // Bag2 lookup (r3) via OutBySrc: key=C, idx=j_r3 -> val=A, eid=r3_eid.
        // Table side gated by `q_tbl` for the reason given at the r1 lookup.
        meta.lookup_any("bag2 r3 from out_by_src", |m| {
            let q = m.query_selector(q_t3_lookup);
            let t = m.query_selector(out_by_src.q_tbl);
            vec![
                (
                    q.clone() * m.query_advice(t3_c, Rotation::cur()),
                    t.clone() * m.query_advice(out_by_src.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t3_j_r3, Rotation::cur()),
                    t.clone() * m.query_advice(out_by_src.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t3_a, Rotation::cur()),
                    t.clone() * m.query_advice(out_by_src.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t3_r3_eid, Rotation::cur()),
                    t * m.query_advice(out_by_src.sorted_eid, Rotation::cur()),
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

        // ============ Bag2's share of the One-Pass OBJ (never laned) ============
        // Bag2 IS the Edge relation, whose size is public in every regime, so it
        // stays a single fixed-height group and its three conditions are the
        // sibling's verbatim.
        let cln3 = meta.advice_column();
        let ck3 = meta.advice_column();
        meta.enable_equality(cln3);
        let q_cln3_bind = meta.selector();

        let aux_t3_pad = meta.advice_column();
        let iz_t3_pad = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_cln3_bind),
            |m| m.query_advice(t3_c, Rotation::cur()),
            aux_t3_pad,
        );

        // `t3_real` is a multiplicity of the input channel, so it must be boolean
        // AND pinned to the row's own attributes: with t3_real = 1 on a pad row
        // the gate above publishes in_key = pack2(0,0) = 0, the reserved dummy
        // key, and with t3_real = 0 on a real row that row's key leaves the
        // message map and the child table at once. Node ids are SHIFT_ID-shifted,
        // so `real == [C != 0]` closes both.
        meta.create_gate("bag2 clean indicator", |m| {
            let q = m.query_selector(q_cln3_bind);
            let cln = m.query_advice(cln3, Rotation::cur());
            let real = m.query_advice(t3_real, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone() * real.clone() * (one.clone() - real.clone()),
                q.clone() * cln.clone() * (one.clone() - cln.clone()),
                // clean implies real, i.e. cln3 == t3_real * c_3
                q.clone() * cln.clone() * (one.clone() - real.clone()),
                q.clone() * (real - (one - iz_t3_pad.expr())),
                // the clean key column both directions of condition (9) use.
                // in_key is already pinned to pack2(A,C) on a real row, and
                // cln3 implies real, so this is pi_K of the clean side exactly.
                q * (m.query_advice(ck3, Rotation::cur())
                    - cln * m.query_advice(agg_msg.in_key, Rotation::cur())),
            ]
        });

        // condition (7) for Bag2. Not laned, so `PermAnyChip` costs a constant
        // number of permutation columns here and is used as in the sibling.
        let part3 = (0..5).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let q_perm3_in = meta.complex_selector();
        let q_perm3_out = meta.complex_selector();
        let perm_bag2 = PermAnyChip::configure(
            meta,
            q_perm3_in,
            q_perm3_out,
            vec![t3_c, t3_a, t3_j_r3, t3_r3_eid, cln3],
            part3.clone(),
        );
        let q_part3 = meta.selector();
        let q_part3_mono = meta.selector();
        {
            let flag = part3[4];
            meta.create_gate("bag2 partition flag is boolean", move |m| {
                let q = m.query_selector(q_part3);
                let one = Expression::Constant(F::ONE);
                let f = m.query_advice(flag, Rotation::cur());
                vec![q * f.clone() * (one - f)]
            });
            meta.create_gate("bag2 partition flag is non-increasing", move |m| {
                let qm = m.query_selector(q_part3_mono);
                let one = Expression::Constant(F::ONE);
                let f = m.query_advice(flag, Rotation::cur());
                let f_next = m.query_advice(flag, Rotation::next());
                vec![qm * f_next * (one - f)]
            });
        }

        // condition (10), child side: Bag2 is the only child of the cluster tree
        // and it is a leaf, so its two multiplicity columns are columns the
        // circuit already has. `in_key` / `in_val` are pinned to the Bag2
        // attributes by the gate above and the clean channel is the bound
        // indicator. One fixed column serves every Lt chip of the stage.
        let cp_u8 = meta.fixed_column();
        let cp_agg = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            agg_msg.in_key, // t3_real ? pack2(A,C) : PAD
            agg_msg.in_val, // t3_real
            cln3,
            PAD_U64,
        );

        // ---------------- lanes ----------------
        // Shared selectors: lanes 0..c-1 are active on the same rows, so one
        // selector of each kind drives all of them. That is what keeps the
        // selector count constant in the lane count even though every lane
        // carries a full copy of the OBJ machinery.
        let q_t12_lookup = meta.complex_selector();
        let q_t12_key = meta.selector();
        let q_flag = meta.selector();
        let q_complex = meta.complex_selector();
        let q_order = meta.selector();
        let q_contrib = meta.selector();
        let q_sum0 = meta.selector();
        let q_sum = meta.selector();
        let q_cln_bind = meta.selector();
        let q_part12 = meta.selector();
        let q_part12_mono = meta.selector();
        let q_perm12_in = meta.complex_selector();
        let q_perm12_out = meta.complex_selector();
        let full = LaneSelectors {
            q_t12_lookup,
            q_t12_key,
            q_flag,
            q_complex,
            q_order,
            q_contrib,
            q_sum0,
            q_sum,
            q_cln_bind,
            q_part12,
            q_part12_mono,
            q_perm12_in,
            q_perm12_out,
        };

        // The last lane stops at the released capacity, so it cannot share the
        // ones above: a selector is a fixed column and turns its gate on for
        // every lane at once. Four more columns cover its four distinct row
        // ranges, whatever c is, and `q_sum0` stays shared because row 0 is live
        // in every lane.
        //
        // `q_last_complex` is complex because it gates lookups and a shuffle;
        // `q_last_row` is simple and drives gates only, exactly like the nine
        // simple selectors it stands in for.
        let q_last_row = meta.selector();
        let q_last_complex = meta.complex_selector();
        let q_last_sum = meta.selector();
        let q_last_mono = meta.selector();
        let last = LaneSelectors {
            q_t12_lookup: q_last_complex,
            q_t12_key: q_last_row,
            q_flag: q_last_row,
            q_complex: q_last_complex,
            q_order: q_last_row,
            q_contrib: q_last_row,
            q_sum0,
            q_sum: q_last_sum,
            q_cln_bind: q_last_row,
            q_part12: q_last_row,
            q_part12_mono: q_last_mono,
            q_perm12_in: q_last_complex,
            q_perm12_out: q_last_complex,
        };

        // One u8 range table for every lane Lt chip: `load` costs one 256-row
        // region per distinct u8 column, and that must not grow with c.
        let lane_u8 = meta.fixed_column();

        let lanes: Vec<LaneConfig<F>> = (0..num_lanes)
            .map(|l| {
                Self::configure_lane(
                    meta,
                    &in_by_dst,
                    &out_by_src,
                    &agg_msg,
                    &cp_agg,
                    ck3,
                    // `q_t3_lookup` is complex and enabled on exactly the Bag2
                    // rows, so it doubles as the table selector of one direction
                    // of condition (9) and the input selector of the other.
                    q_t3_lookup,
                    lane_u8,
                    if l + 1 == num_lanes { &last } else { &full },
                )
            })
            .collect();

        // condition (9), the union direction: every clean Bag2 tuple names
        // exactly one hosting lane. Together with each lane's
        // "pw: bag2^c key in bag1^c (lane host)" lookup this is membership of
        // pi_K(Bag2^c) in the UNION of the lanes' clean key columns; a prover
        // that cannot name a lane fails right here. The polynomial grows with the
        // lane count but its DEGREE does not, since the bits are summed.
        let q_host3 = meta.selector();
        {
            let hosts: Vec<Column<Advice>> = lanes.iter().map(|l| l.host3).collect();
            meta.create_gate(
                "pw: every clean bag2 tuple is hosted by exactly one lane",
                move |m| {
                    let q = m.query_selector(q_host3);
                    let one = Expression::Constant(F::ONE);
                    let mut polys = Vec::with_capacity(hosts.len() + 1);
                    let mut sum = Expression::Constant(F::ZERO);
                    for h in hosts.iter() {
                        let hq = m.query_advice(*h, Rotation::cur());
                        polys.push(q.clone() * hq.clone() * (one.clone() - hq.clone()));
                        sum = sum + hq;
                    }
                    polys.push(q * m.query_advice(cln3, Rotation::cur()) * (sum - one));
                    polys
                },
            );
        }

        // ---------------- cross-lane totals ----------------
        // Row l holds lane l's total, copied in from that lane's last run_sum
        // cell; tot_run adds them with a degree-2 accumulator. A flat
        // degree-(1+c) sum gate would blow past cs.degree() = 7 for c > 6.
        let lane_tot = meta.advice_column();
        let tot_run = meta.advice_column();
        let lane_cln = meta.advice_column();
        let tot_cln = meta.advice_column();
        meta.enable_equality(lane_tot);
        meta.enable_equality(tot_run);
        meta.enable_equality(lane_cln);
        let q_tot0 = meta.selector();
        let q_tot = meta.selector();

        meta.create_gate("lane totals first", |m| {
            let q = m.query_selector(q_tot0);
            vec![
                q.clone()
                    * (m.query_advice(tot_run, Rotation::cur())
                        - m.query_advice(lane_tot, Rotation::cur())),
                q * (m.query_advice(tot_cln, Rotation::cur())
                    - m.query_advice(lane_cln, Rotation::cur())),
            ]
        });
        meta.create_gate("lane totals accu", |m| {
            let q = m.query_selector(q_tot);
            vec![
                q.clone()
                    * (m.query_advice(tot_run, Rotation::cur())
                        - (m.query_advice(tot_run, Rotation::prev())
                            + m.query_advice(lane_tot, Rotation::cur()))),
                q * (m.query_advice(tot_cln, Rotation::cur())
                    - (m.query_advice(tot_cln, Rotation::prev())
                        + m.query_advice(lane_cln, Rotation::cur()))),
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

        // ---- condition (10): |Bag1^c |X| Bag2^c| == |Bag1 |X| Bag2| ----
        // THE one equality of the Cardinality Preservation Check, at the last
        // totals row, over the two GRAND totals.
        //
        // `tot_run` is the input-channel root sum: on this cluster tree
        // sigma_all(key) is the number of real Bag2 edges with that key, which is
        // exactly the msg_val the answer is read off, so
        // mu_all = t12_real * [A<B] * [B<C] * msg_val is the `contrib` column and
        // its cross-lane total is `tot_run`. `tot_cln` is the clean-channel root
        // sum of the same traversal.
        //
        // Both are accumulated ACROSS the lanes before they meet. A per-lane
        // equality would be strictly weaker and is the escape this layout has to
        // avoid: a prover would hide a joinable tuple in one lane and re-balance
        // that lane's two channels against each other, and every lane's own
        // equality would still hold.
        meta.create_gate("cp: cardinality preservation over all lanes", |m| {
            let q = m.query_selector(q_out);
            vec![
                q * (m.query_advice(tot_run, Rotation::cur())
                    - m.query_advice(tot_cln, Rotation::cur())),
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
            cln3,
            iz_t3_pad,
            q_cln3_bind,
            ck3,
            part3,
            perm_bag2,
            q_part3,
            q_part3_mono,
            q_host3,
            cp_agg,
            cp_u8,
            lanes,
            lane_u8,
            full,
            last,
            lane_tot,
            tot_run,
            lane_cln,
            tot_cln,
            q_tot0,
            q_tot,
            out,
            q_out,
        }
    }

    /// Assign one lane's block of the padded Bag1 pipeline.
    ///
    /// `base` is the global index of this lane's row 0 and `live_rows` its
    /// [`lane_live_rows`], so lane l covers the padded rows
    /// `[l*lane_rows, l*lane_rows + live_rows)`: a full lane_rows except in the
    /// last lane, which stops at the released capacity. Rows past `t12.len()`
    /// are DP pad rows and are witnessed with exactly the pad convention
    /// g_sql3_obj uses: all-zero tuple, key 0, in_set 1, val 0, contrib 0. Map
    /// row 0 is (0,0) and view row 0 is (0,0,0), so a pad row satisfies every
    /// lookup.
    ///
    /// Returns the lane's last `run_sum` cell (the value the totals stage
    /// copies) together with that total.
    #[allow(clippy::too_many_arguments)]
    fn assign_lane(
        lane: &LaneConfig<F>,
        region: &mut Region<'_, F>,
        live_rows: usize,
        base: usize,
        t12: &[(u64, u64, u64, u64, u64, u64, u64)],
        msg_map: &BTreeMap<u64, u64>,
        keys: &[u64],
        cln12: &[u64],
        sigma_cln: &HashMap<u64, (u64, u64)>,
        tamper: Tamper,
        pad_row_global: usize,
    ) -> Result<(AssignedCell<F, F>, u64, AssignedCell<F, F>, u64), Error> {
        let lt_ab_chip = LtChip::<F, NUM_BYTES>::construct(lane.lt_ab);
        let lt_bc_chip = LtChip::<F, NUM_BYTES>::construct(lane.lt_bc);
        let lt_low_chip = LtChip::<F, NUM_BYTES>::construct(lane.msg_lookup.lt_low);
        let lt_high_chip = LtChip::<F, NUM_BYTES>::construct(lane.msg_lookup.lt_high);
        let iz_pad_chip = IsZeroChip::construct(lane.iz_t12_pad.clone());

        let real12 = t12.len();
        let mut running: u64 = 0;
        let mut cln_running: u64 = 0;
        let mut last_cell: Option<AssignedCell<F, F>> = None;
        let mut last_cln_cell: Option<AssignedCell<F, F>> = None;

        // ---- condition (7): this lane's [clean | residual | pad] group ----
        // Rows past `real12` are pad rows, whose clean indicator is 0, so the
        // clean and residual blocks only ever hold this lane's real rows and the
        // group's tail is the all-zero pad tuple with flag 0. The flag comes out
        // non-increasing, which is what its gate asks for.
        //
        // The group is exactly `live_rows` tall, which is what the shuffle needs:
        // both of its sides are gated by this lane's own selectors over the same
        // rows, so the multiset equality is over the live rows and nothing else.
        let mut part_rows: Vec<[u64; 8]> = Vec::with_capacity(live_rows);
        for pass in 0..2 {
            for r in 0..live_rows {
                let i = base + r;
                if i >= real12 {
                    continue;
                }
                let clean = cln12[i];
                if (pass == 0) != (clean == 1) {
                    continue;
                }
                let (a, b, c, i_r1, j_r2, r1_eid, r2_eid) = t12[i];
                part_rows.push([a, b, c, i_r1, j_r2, r1_eid, r2_eid, clean]);
            }
        }
        while part_rows.len() < live_rows {
            part_rows.push([0u64; 8]);
        }
        // test hook only: demote the last clean row of this lane's group. The flag
        // column stays boolean and non-increasing, and no residual row of the bag
        // carries the same attributes (a wedge's clean bit is a function of its
        // key and its predicate), so only the shuffle can see it.
        if tamper == Tamper::PartitionFlag {
            if let Some(last_clean) = part_rows.iter().rposition(|row| row[7] == 1) {
                part_rows[last_clean][7] = 0;
            }
        }
        for (r, row) in part_rows.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                region.assign_advice(
                    || "part12",
                    lane.part12[j],
                    r,
                    || Value::known(F::from(*v)),
                )?;
            }
        }

        for r in 0..live_rows {
            let i = base + r;

            let (a, b, c, i_r1, j_r2, r1_eid, r2_eid, real) = if i < real12 {
                let (a, b, c, i_r1, j_r2, r1_eid, r2_eid) = t12[i];
                (a, b, c, i_r1, j_r2, r1_eid, r2_eid, 1u64)
            } else {
                (0, 0, 0, 0, 0, 0, 0, 0u64)
            };

            // Tamper::PadRow breaks the pad convention on the last row of the
            // released capacity: t12_a moves but msg key stays 0.
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

            // `real == [B != 0]` witness, on every row of the lane
            iz_pad_chip.assign(region, r, Value::known(F::from(b)))?;

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

            // ---- the clean side of this row: conditions (7), (9) and (10) ----
            // The clean indicator is a function of the WHOLE bag, not of this
            // lane: a row's clean bit is decided by the global semijoin reduction
            // in `assign`, so slicing the bag into lanes never changes it.
            let clean = if i < real12 { cln12[i] } else { 0 };
            region.assign_advice(
                || "cln12",
                lane.cln12,
                r,
                || Value::known(F::from(clean)),
            )?;
            region.assign_advice(
                || "ck12",
                lane.ck12,
                r,
                || Value::known(F::from(if clean == 1 { key } else { 0 })),
            )?;
            // mu_cln = cln12 * sigma_cln(key). Its lookup pins it, so the value
            // written here has to be the child table's clean sum for this key;
            // on a residual, pad or predicate-dropped row it is 0.
            let mu_cln_u64 = if clean == 1 {
                sigma_cln.get(&key).map(|v| v.1).unwrap_or(0)
            } else {
                0
            };
            region.assign_advice(
                || "mu_cln",
                lane.mu_cln,
                r,
                || Value::known(F::from(mu_cln_u64)),
            )?;
            cln_running = if r == 0 {
                mu_cln_u64
            } else {
                cln_running.wrapping_add(mu_cln_u64)
            };
            last_cln_cell = Some(region.assign_advice(
                || "cln_run",
                lane.cln_run,
                r,
                || Value::known(F::from(cln_running)),
            )?);
        }

        // The drain into the totals stage is the cell at row `live_rows - 1`,
        // which is now the last row of the RELEASED capacity in the last lane
        // rather than the last row of the lane's full height. Both prefix-sum
        // gates stop there with it.
        Ok((
            last_cell.expect("live_rows > 0 guarantees a last row"),
            running,
            last_cln_cell.expect("live_rows > 0 guarantees a last row"),
            cln_running,
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
        // 4c lane chips share `lane_u8` and every Lt chip of the Cardinality
        // Preservation child stage shares `cp_u8`, so this is 6 load regions for
        // any c. Each one touches only its own fixed column, so they all start at
        // row 0 and none of them displaces the witness region.
        LtChip::<F, NUM_BYTES>::construct(cfg.in_by_dst.lt_key).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.out_by_src.lt_key).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.agg_msg.lt_key).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.agg_msg.lt_out_key).load(layouter)?;
        debug_assert_eq!(cfg.lanes[0].lt_ab.u8, cfg.lane_u8);
        LtChip::<F, NUM_BYTES>::construct(cfg.lanes[0].lt_ab).load(layouter)?;
        debug_assert_eq!(cfg.cp_agg.lt_key_cur_next.u8, cfg.cp_u8);
        LtChip::<F, NUM_BYTES>::construct(cfg.cp_agg.lt_key_cur_next).load(layouter)?;

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
                // Row counts of the two bags. The partition witness below is
                // computed before either bag is assigned, so both counts and both
                // padding schemes are settled here, unchanged.
                // -------------------
                let real3 = derived.t3.len();
                let n3 = std::cmp::max(real3 + bag2_pad_extra, 1);
                let real12 = derived.t12.len();
                // The RELEASED capacity: the public value the lane count is
                // derived from, and now also where the last lane stops.
                let capacity = std::cmp::max(real12 + bag1_pad_extra, 1);
                assert!(
                    num_lanes * lane_rows >= capacity,
                    "released Bag1 capacity {} exceeds {} lanes of {} rows",
                    capacity,
                    num_lanes,
                    lane_rows
                );

                // ---------------- clean / residual partition of the two bags ----------------
                // Identical to the sibling's, and computed over the WHOLE bag: a
                // tuple's clean bit is a property of the relation, not of the lane
                // it happens to land in, so the lane split never enters here.
                //
                // The honest clean part of a relation is its fully reduced
                // instance. A Bag1 wedge contributes only when it is real and
                // A<B<C, and a wedge and a closing edge extend each other exactly
                // when they agree on the packed separator key, so the reduction of
                // this two-node cluster tree is the intersection of the two key
                // sets. On a two-node tree that single simultaneous pass is
                // already a fixed point of semijoin reduction, which is what
                // condition (9) needs.
                //
                // Only the REAL rows carry a clean bit; every pad row of every
                // lane has indicator 0, and `assign_lane` reads that off the
                // length of these vectors.
                let pred12: Vec<bool> = (0..real12)
                    .map(|i| {
                        derived.t12[i].0 < derived.t12[i].1 && derived.t12[i].1 < derived.t12[i].2
                    })
                    .collect();
                let key12: Vec<u64> = (0..real12)
                    .map(|i| pack2(derived.t12[i].0, derived.t12[i].2))
                    .collect();

                let keys3: HashSet<u64> = derived
                    .t3
                    .iter()
                    .map(|&(c, a, _, _)| pack2(a, c))
                    .collect();
                let keys12: HashSet<u64> =
                    (0..real12).filter(|&i| pred12[i]).map(|i| key12[i]).collect();

                let mut cln3: Vec<u64> = (0..real3)
                    .map(|i| keys12.contains(&pack2(derived.t3[i].1, derived.t3[i].0)) as u64)
                    .collect();
                let mut cln12: Vec<u64> = (0..real12)
                    .map(|i| (pred12[i] && keys3.contains(&key12[i])) as u64)
                    .collect();

                // Test hook only. Hide one joinable Bag2 tuple in the residual
                // side and iterate the reduction until nothing moves, so the
                // partition still conserves both bags and the two clean sides are
                // still pairwise consistent: only condition (10) can see it, and
                // only because its root sums cross the lanes.
                if tamper == Tamper::HideCleanTuple {
                    if let Some(hidden) = (0..real3).find(|&i| cln3[i] == 1) {
                        cln3[hidden] = 0;
                        loop {
                            let mut moved = false;
                            let live3: HashSet<u64> = (0..real3)
                                .filter(|&i| cln3[i] == 1)
                                .map(|i| pack2(derived.t3[i].1, derived.t3[i].0))
                                .collect();
                            for i in 0..real12 {
                                if cln12[i] == 1 && !live3.contains(&key12[i]) {
                                    cln12[i] = 0;
                                    moved = true;
                                }
                            }
                            let live12: HashSet<u64> = (0..real12)
                                .filter(|&i| cln12[i] == 1)
                                .map(|i| key12[i])
                                .collect();
                            for i in 0..real3 {
                                if cln3[i] == 1
                                    && !live12
                                        .contains(&pack2(derived.t3[i].1, derived.t3[i].0))
                                {
                                    cln3[i] = 0;
                                    moved = true;
                                }
                            }
                            if !moved {
                                break;
                            }
                        }
                    }
                }

                // Test hook only. Skip the reduction and declare every real tuple
                // clean; the indicator still has to satisfy its binding gate, so a
                // Bag1 wedge its own predicate drops stays out. Conservation still
                // holds and both channels of condition (10) then carry the same
                // multiplicity on every row, so only condition (9) can reject it.
                if tamper == Tamper::MarkAllClean {
                    for v in cln3.iter_mut() {
                        *v = 1;
                    }
                    for i in 0..real12 {
                        cln12[i] = pred12[i] as u64;
                    }
                }

                // ---------------- condition (9), the union direction ----------------
                // For every clean Bag2 tuple, the lane that hosts a clean Bag1
                // wedge with the same separator key. One lane per key is enough,
                // so this is a key -> lane map over the clean Bag1 rows; the
                // coverage gate rejects a clean Bag2 key that has none, which is
                // exactly what the all-clean partition produces.
                let mut host_of_key: HashMap<u64, usize> = HashMap::new();
                for j in 0..real12 {
                    if cln12[j] == 1 {
                        host_of_key.entry(key12[j]).or_insert(j / lane_rows);
                    }
                }

                // test hook only: the (lane, key) pairs the lanes really hold, so
                // Tamper::HostWrongLane can name a lane that does not hold the key
                let clean_lane_keys: HashSet<(usize, u64)> = if tamper == Tamper::HostWrongLane {
                    (0..real12)
                        .filter(|&j| cln12[j] == 1)
                        .map(|j| (j / lane_rows, key12[j]))
                        .collect()
                } else {
                    HashSet::new()
                };
                let mut wrong_host_placed = false;

                // -------------------
                // Bag2 + message aggregation: fixed-height under Privacy::Dp
                // (GQ3's Bag2 IS the Edge relation, whose size is public), and
                // small enough to stay a single group under Privacy::Legacy.
                // It occupies its own columns, so its height is independent of
                // `lane_rows`; only the domain 2^k bounds it, and the harness
                // checks that fit alongside the lane geometry.
                // -------------------
                let mut agg_in: Vec<(u64, u64)> = vec![(PAD_U64, 0); n3];
                let iz_t3_pad_chip = IsZeroChip::construct(cfg.iz_t3_pad.clone());

                for i in 0..n3 {
                    cfg.q_t3_lookup.enable(&mut region, i)?;
                    cfg.q_t3_msg_in.enable(&mut region, i)?;

                    // clean indicator, the input side of Bag2's Conservation
                    // Check, its partition-flag gates and the host-bit coverage
                    // gate, all over the same rows as the bag itself
                    cfg.q_cln3_bind.enable(&mut region, i)?;
                    cfg.q_host3.enable(&mut region, i)?;
                    cfg.q_part3.enable(&mut region, i)?;
                    if i + 1 < n3 {
                        cfg.q_part3_mono.enable(&mut region, i)?;
                    }
                    cfg.perm_bag2.q_perm1.enable(&mut region, i)?;
                    cfg.perm_bag2.q_perm2.enable(&mut region, i)?;

                    let clean = if i < real3 { cln3[i] } else { 0 };
                    region.assign_advice(
                        || "cln3",
                        cfg.cln3,
                        i,
                        || Value::known(F::from(clean)),
                    )?;
                    let ck3_u64 = if clean == 1 {
                        pack2(derived.t3[i].1, derived.t3[i].0)
                    } else {
                        0
                    };
                    region.assign_advice(
                        || "ck3",
                        cfg.ck3,
                        i,
                        || Value::known(F::from(ck3_u64)),
                    )?;
                    iz_t3_pad_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(if i < real3 { derived.t3[i].0 } else { 0 })),
                    )?;

                    // one host bit per lane; exactly one is 1 on a clean row
                    let mut host = if clean == 1 {
                        host_of_key.get(&ck3_u64).copied()
                    } else {
                        None
                    };
                    if tamper == Tamper::HostWrongLane && clean == 1 && !wrong_host_placed {
                        if let Some(wrong) =
                            (0..num_lanes).find(|&l| !clean_lane_keys.contains(&(l, ck3_u64)))
                        {
                            host = Some(wrong);
                            wrong_host_placed = true;
                        }
                    }
                    for (l, lane) in cfg.lanes.iter().enumerate() {
                        region.assign_advice(
                            || "host3",
                            lane.host3,
                            i,
                            || Value::known(F::from((host == Some(l)) as u64)),
                        )?;
                    }

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

                // ---- condition (7) for Bag2: [clean | residual | pad] ----
                // The pad rows are the all-zero tuple the bag already pads with,
                // and SHIFT_ID keeps 0 out of the real tuples, so the two
                // multisets agree exactly and the flag comes out non-increasing.
                let mut part3_rows: Vec<[u64; 5]> = Vec::with_capacity(n3);
                for pass in 0..2 {
                    for i in 0..real3 {
                        if (pass == 0) != (cln3[i] == 1) {
                            continue;
                        }
                        let (c, a, j_r3, r3_eid) = derived.t3[i];
                        part3_rows.push([c, a, j_r3, r3_eid, cln3[i]]);
                    }
                }
                while part3_rows.len() < n3 {
                    part3_rows.push([0u64; 5]);
                }
                for (i, row) in part3_rows.iter().enumerate() {
                    for (j, v) in row.iter().enumerate() {
                        region.assign_advice(
                            || "part3",
                            cfg.part3[j],
                            i,
                            || Value::known(F::from(*v)),
                        )?;
                    }
                }

                // ---- condition (10), child side ----
                // Bag2 is a leaf, so its input-channel multiplicity is its real
                // bit and its clean-channel multiplicity is the bound indicator;
                // both columns are already assigned above and the key column is
                // the same `agg_msg.in_key` the message table sorts. The stage
                // publishes one (key, sum_all, sum_cln) row per key, and every
                // lane reads `sum_cln` out of it through a lookup.
                let cp_rows: Vec<[u64; 3]> = (0..n3)
                    .map(|i| {
                        [
                            agg_in[i].0,
                            agg_in[i].1,
                            if i < real3 { cln3[i] } else { 0 },
                        ]
                    })
                    .collect();
                let cp_stage = build_cp_stage(&cp_rows, PAD_U64);
                assign_cp_agg(&mut region, &cfg.cp_agg, &cp_rows, &cp_stage)?;

                // -------------------
                // Bag1: the pad-driven pipeline, spread over the lanes
                // -------------------
                // Two selector sets, one row range each. `full` covers all
                // `lane_rows` rows and switches ON every lane but the last;
                // `last` covers the live rows of the last lane, which stop at
                // the RELEASED capacity. Both ranges are functions of the
                // capacity, `c` and `lane_rows`, all public, so nothing here can
                // depend on the true bag size -- and that is still true of the
                // OBJ selectors: none of their enable ranges is a function of
                // the clean/residual split, which is why no lane's shape leaks
                // |Bag1^c|.
                //
                // Row 0 is live in every lane, so `q_sum0` stays shared and is
                // enabled once, outside both loops.
                let live_last = lane_live_rows(num_lanes - 1, num_lanes, lane_rows, capacity);
                cfg.full.q_sum0.enable(&mut region, 0)?;
                if num_lanes > 1 {
                    for r in 0..lane_rows {
                        cfg.full.q_t12_lookup.enable(&mut region, r)?;
                        cfg.full.q_t12_key.enable(&mut region, r)?;
                        cfg.full.q_flag.enable(&mut region, r)?;
                        cfg.full.q_complex.enable(&mut region, r)?;
                        cfg.full.q_order.enable(&mut region, r)?;
                        cfg.full.q_contrib.enable(&mut region, r)?;
                        cfg.full.q_cln_bind.enable(&mut region, r)?;
                        cfg.full.q_part12.enable(&mut region, r)?;
                        if r + 1 < lane_rows {
                            cfg.full.q_part12_mono.enable(&mut region, r)?;
                        }
                        cfg.full.q_perm12_in.enable(&mut region, r)?;
                        cfg.full.q_perm12_out.enable(&mut region, r)?;
                        if r > 0 {
                            cfg.full.q_sum.enable(&mut region, r)?;
                        }
                    }
                }
                // The same list against the same names, so the two ranges stay
                // visibly parallel. Several of these names alias one column in
                // the `last` set, and enabling a selector twice on a row is
                // idempotent; `Assignment::enable_selector` is a no-op under the
                // prover in any case, since selectors are fixed and were
                // committed at keygen.
                for r in 0..live_last {
                    cfg.last.q_t12_lookup.enable(&mut region, r)?;
                    cfg.last.q_t12_key.enable(&mut region, r)?;
                    cfg.last.q_flag.enable(&mut region, r)?;
                    cfg.last.q_complex.enable(&mut region, r)?;
                    cfg.last.q_order.enable(&mut region, r)?;
                    cfg.last.q_contrib.enable(&mut region, r)?;
                    cfg.last.q_cln_bind.enable(&mut region, r)?;
                    cfg.last.q_part12.enable(&mut region, r)?;
                    if r + 1 < live_last {
                        cfg.last.q_part12_mono.enable(&mut region, r)?;
                    }
                    cfg.last.q_perm12_in.enable(&mut region, r)?;
                    cfg.last.q_perm12_out.enable(&mut region, r)?;
                    if r > 0 {
                        cfg.last.q_sum.enable(&mut region, r)?;
                    }
                }

                let mut lane_cells: Vec<AssignedCell<F, F>> = Vec::with_capacity(num_lanes);
                let mut lane_totals: Vec<u64> = Vec::with_capacity(num_lanes);
                let mut lane_cln_cells: Vec<AssignedCell<F, F>> = Vec::with_capacity(num_lanes);
                let mut lane_cln_totals: Vec<u64> = Vec::with_capacity(num_lanes);
                for (l, lane) in cfg.lanes.iter().enumerate() {
                    let (cell, total, cln_cell, cln_total) = Self::assign_lane(
                        lane,
                        &mut region,
                        lane_live_rows(l, num_lanes, lane_rows, capacity),
                        l * lane_rows,
                        &derived.t12,
                        &derived.msg_map,
                        &derived.keys,
                        &cln12,
                        &cp_stage.map,
                        tamper,
                        capacity - 1,
                    )?;
                    lane_cells.push(cell);
                    lane_totals.push(total);
                    lane_cln_cells.push(cln_cell);
                    lane_cln_totals.push(cln_total);
                }

                // -------------------
                // The only cross-lane wiring: 2c copy constraints plus two
                // degree-2 accumulators over the c lane totals, one per channel
                // of the Cardinality Preservation Check.
                // -------------------
                let mut acc: u64 = 0;
                let mut acc_cln: u64 = 0;
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

                    let cln_tot = lane_cln_totals[l];
                    let cln_cell = region.assign_advice(
                        || "lane_cln",
                        cfg.lane_cln,
                        l,
                        || Value::known(F::from(cln_tot)),
                    )?;
                    region.constrain_equal(lane_cln_cells[l].cell(), cln_cell.cell())?;

                    if l == 0 {
                        cfg.q_tot0.enable(&mut region, l)?;
                        acc = tot;
                        acc_cln = cln_tot;
                    } else {
                        cfg.q_tot.enable(&mut region, l)?;
                        acc = acc.wrapping_add(tot);
                        acc_cln = acc_cln.wrapping_add(cln_tot);
                    }
                    region.assign_advice(
                        || "tot_run",
                        cfg.tot_run,
                        l,
                        || Value::known(F::from(acc)),
                    )?;
                    region.assign_advice(
                        || "tot_cln",
                        cfg.tot_cln,
                        l,
                        || Value::known(F::from(acc_cln)),
                    )?;
                }

                // condition (10) over the union of the lanes. `acc` is the
                // input-channel root sum (which is also the certified COUNT(*))
                // and `acc_cln` is the clean-channel root sum; the gate
                // "cp: cardinality preservation over all lanes" compares them at
                // the last totals row.
                if tamper == Tamper::None {
                    debug_assert_eq!(
                        acc, acc_cln,
                        "cardinality preservation: |R^c join| != |R join|"
                    );
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
        lane_live_rows, lane_rows_for, lanes_for, lanes_for_capacity, set_config_lanes, MyCircuit,
        Tamper, BASE_DEGREE, LANE_ROWS, MAX_LANES,
    };

    use crate::data::graph_data_processing::Edge;
    use crate::graph_sql::g_sql3_obj::{gq3_derive, pack2};
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::Circuit;
    use halo2curves::pasta::Fp;

    use std::collections::HashSet;
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
    /// capacity of 11, which at 4 rows per lane really spans 3 lanes and leaves
    /// the LAST lane 3 rows tall instead of 4.
    ///
    /// That short last lane is why every negative direction below is also a test
    /// of the shortened boundary: a lane that stopped being fully constrained
    /// past its capacity would still satisfy `mock_three_lanes`, and only these
    /// would notice. `Tamper::PadRow` in particular breaks the last row of the
    /// released capacity, which is the last LIVE row of that short lane.
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

    /// The live rows of the c lanes must TILE the released capacity: every lane
    /// but the last full, the last stopping exactly at the capacity, and the sum
    /// equal to it. Anything else either drops a released row or leaves lane
    /// quantization behind.
    #[test]
    fn live_rows_tile_the_capacity() {
        for lane_rows in [1usize, 2, 3, 4, 7, 260_800] {
            for capacity in [1usize, 2, 5, 11, 12, 13, 277_610] {
                let c = lanes_for(capacity, lane_rows);
                let live: Vec<usize> = (0..c)
                    .map(|l| lane_live_rows(l, c, lane_rows, capacity))
                    .collect();
                assert_eq!(
                    live.iter().sum::<usize>(),
                    capacity,
                    "lane_rows={lane_rows} capacity={capacity}"
                );
                for (l, &n) in live.iter().enumerate() {
                    assert!(n >= 1 && n <= lane_rows, "lane {l} has {n} live rows");
                    if l + 1 < c {
                        assert_eq!(n, lane_rows, "lane {l} is not full");
                    }
                }
            }
        }

        // the measured lastfm release the bench times: 2 lanes, and the second
        // one holds 6.4% of a lane
        assert_eq!(lane_live_rows(0, 2, 260_800, 277_610), 260_800);
        assert_eq!(lane_live_rows(1, 2, 260_800, 277_610), 16_810);
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

        // one lane costs a fixed number of advice columns, lookup arguments,
        // shuffles and permutation columns, and nothing else
        assert_eq!(
            step[1], 0,
            "fixed columns must not grow with the lane count"
        );
        assert_eq!(step[2], 0, "selectors must not grow with the lane count");
        assert_eq!(step[6], 0, "degree must not grow with the lane count");
        assert!(step[0] > 0 && step[3] > 0, "a lane must cost something");
        // Exactly one shuffle per lane: that lane's Conservation Check
        // (condition (7)). Per-lane is what makes it a refinement of the union
        // statement rather than something a prover can split, so this MUST grow
        // with the lane count; what must stay true is that it grows by the same
        // amount for every lane, which the window loop below checks.
        assert_eq!(
            step[4], 1,
            "a lane must add exactly its own Conservation shuffle"
        );
        // run_sum and cln_run, the two channel totals copied out of a lane; every
        // other per-lane column is read by gates and lookups only and stays out
        // of the permutation argument
        assert_eq!(step[5], 2, "a lane must add exactly two permutation columns");

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
        let cases: [(usize, usize, usize); 7] = [
            (2, 0, 3), // 6 real rows, no pad: lane 2 is entirely real and full
            (1, 0, 6), // one row per lane, q_sum never fires
            (4, 5, 3), // 11 released rows over 3 lanes of 4: last lane 3 tall
            (3, 1, 3), // 7 released rows over 3 lanes of 3: last lane 1 tall
            (5, 0, 2), // real rows straddle the lane-0/lane-1 boundary
            (4, 7, 4), // 13 released over 4 lanes of 4: last lane is ONE pad row
            (16, 5, 1), // one lane, live on 11 of its 16 rows
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

    /// Cost probe. Every One-Pass OBJ constraint in this file is a degree-1 to
    /// degree-3 gate, a shuffle, or a lookup whose input and table degrees sum to
    /// at most 5, so none of them may raise `cs.degree()` above the 7 the
    /// pre-existing "msg member" lookup already costs: a rise would double every
    /// FFT of the proof, on every lane.
    #[test]
    fn test_max_gate_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        for c in [1usize, 3, 8] {
            set_config_lanes(c);
            let mut cs = ConstraintSystem::<Fp>::default();
            let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
            println!(
                "lanes={} advice={} fixed={} sel={} deg={} gates={} polys={} lookups={} \
                 shuffles={} perm_cols={}",
                c,
                cs.num_advice_columns(),
                cs.num_fixed_columns(),
                cs.num_selectors(),
                cs.degree(),
                cs.gates().len(),
                cs.gates()
                    .iter()
                    .map(|g| g.polynomials().len())
                    .sum::<usize>(),
                cs.lookups().len(),
                cs.shuffles().len(),
                cs.permutation().get_columns().len(),
            );
            assert!(
                cs.degree() <= 7,
                "the maximum gate degree rose to {} at {} lanes, so a soundness \
                 patch is costing more than it should",
                cs.degree(),
                c
            );
        }
    }

    /// The lanes the clean Bag1 rows of `synthetic_edges` fall into at a given
    /// lane geometry, plus the number of Bag2 separator keys that no Bag1 wedge
    /// can match. Used to keep the cross-lane OBJ test below non-vacuous.
    fn clean_lane_spread(lane_rows: usize) -> (HashSet<usize>, usize) {
        let derived = gq3_derive(&synthetic_edges());
        let keys3: HashSet<u64> = derived
            .t3
            .iter()
            .map(|&(c, a, _, _)| pack2(a, c))
            .collect();
        let keys12: HashSet<u64> = derived
            .t12
            .iter()
            .filter(|&&(a, b, c, _, _, _, _)| a < b && b < c)
            .map(|&(a, _, c, _, _, _, _)| pack2(a, c))
            .collect();

        let mut lanes = HashSet::new();
        for (i, &(a, b, c, _, _, _, _)) in derived.t12.iter().enumerate() {
            if a < b && b < c && keys3.contains(&pack2(a, c)) {
                lanes.insert(i / lane_rows);
            }
        }
        (lanes, keys3.difference(&keys12).count())
    }

    /// THE cross-lane One-Pass OBJ test.
    ///
    /// Conditions (7), (9) and (10) all have to hold over the UNION of the lanes,
    /// so the geometries below are chosen to put the clean Bag1 rows in more than
    /// one lane: a single-lane run cannot tell a union statement from a per-lane
    /// one. Both negative directions are checked, and each of them is the one the
    /// other cannot see:
    ///
    ///  * one joinable Bag2 tuple hidden in the residual side, with both bags
    ///    re-reduced around it. Conservation still holds, the two clean sides
    ///    still agree on the separator key, and the published COUNT(*) is still
    ///    the honest one. Only condition (10) sees it, and only because its two
    ///    root sums are accumulated ACROSS the lanes before they are compared: a
    ///    per-lane equality would let the prover park the missing contribution in
    ///    a lane that still balances.
    ///  * every real tuple declared clean. Conservation holds and both channels of
    ///    condition (10) then carry the same multiplicity on every row, so its
    ///    equality holds for free. Only condition (9) sees it, through the
    ///    direction whose table is the union of the lanes' clean key columns.
    #[test]
    fn mock_obj_conditions_across_lanes() {
        // (lane_rows, bag1_pad_extra, expected lanes). Both geometries are narrow
        // enough that the 3 clean wedges of `synthetic_edges` CANNOT all fall into
        // one lane whatever order the host-side derivation emits them in: 3 clean
        // rows out of 6 real ones cannot fit in a single lane of 2 rows. A wider
        // lane would sometimes hold all of them and the test would silently stop
        // testing the cross-lane claim. The first case also has no padding at all
        // and one row per lane, so some lanes hold a clean row and others hold
        // none.
        let cases: [(usize, usize, usize); 2] = [(1, 0, 6), (2, 5, 6)];
        for (lane_rows, bag1_pad_extra, want_c) in cases {
            let c = lanes_for(6 + bag1_pad_extra, lane_rows);
            assert_eq!(c, want_c);

            let (lanes_hit, dangling3) = clean_lane_spread(lane_rows);
            assert!(
                lanes_hit.len() >= 2,
                "the clean Bag1 rows all sit in lane(s) {:?} at lane_rows={}, so this \
                 geometry cannot show that the conditions hold across lanes",
                lanes_hit,
                lane_rows
            );
            assert!(
                dangling3 > 0,
                "no dangling Bag2 tuple, so the all-clean partition really is \
                 pairwise consistent and the second direction would pass vacuously"
            );

            let build = |tamper| {
                set_config_lanes(c);
                MyCircuit::<Fp> {
                    edges: synthetic_edges(),
                    bag1_pad_extra,
                    bag2_pad_extra: 3,
                    lane_rows,
                    num_lanes: c,
                    tamper,
                    _marker: PhantomData,
                }
            };
            let public = vec![vec![Fp::from(3u64)]];

            let prover = MockProver::run(11, &build(Tamper::None), public.clone()).unwrap();
            prover.assert_satisfied();

            // condition (10), over the union of the lanes
            let hidden = MockProver::run(11, &build(Tamper::HideCleanTuple), public.clone())
                .unwrap();
            let failures = hidden
                .verify()
                .expect_err("condition (10) accepted a hidden joinable tuple");
            assert!(
                failures
                    .iter()
                    .any(|f| format!("{:?}", f).contains("cardinality preservation")),
                "the circuit rejected the hidden tuple, but not through the \
                 Cardinality Preservation Check: {:?}",
                failures
            );

            // condition (9), the direction whose table is the union of the lanes
            let all_clean =
                MockProver::run(11, &build(Tamper::MarkAllClean), public.clone()).unwrap();
            let failures = all_clean
                .verify()
                .expect_err("the all-clean partition was accepted");
            assert!(
                failures
                    .iter()
                    .any(|f| format!("{:?}", f).contains("pw:")),
                "the circuit rejected the all-clean partition, but not through a \
                 Pairwise Consistency check: {:?}",
                failures
            );
        }
    }

    /// Condition (7) must bite inside EVERY lane. Demoting the last clean row of
    /// each lane's partition group leaves the flag boolean and non-increasing and
    /// leaves the bag's own indicator honest, so the lane's Conservation shuffle
    /// is the only thing that can notice.
    #[test]
    fn mock_reject_tampered_partition_flag() {
        let lane_rows = 2;
        let bag1_pad_extra = 5;
        let c = lanes_for(6 + bag1_pad_extra, lane_rows);
        let (lanes_hit, _) = clean_lane_spread(lane_rows);
        assert!(lanes_hit.len() >= 2);

        set_config_lanes(c);
        let circuit = MyCircuit::<Fp> {
            edges: synthetic_edges(),
            bag1_pad_extra,
            bag2_pad_extra: 3,
            lane_rows,
            num_lanes: c,
            tamper: Tamper::PartitionFlag,
            _marker: PhantomData,
        };
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(3u64)]]).unwrap();
        let failures = prover
            .verify()
            .expect_err("a partition that does not conserve the bag was accepted");
        assert!(
            failures
                .iter()
                .any(|f| format!("{:?}", f).contains("bag1 conservation")),
            "rejected, but not through a lane's Conservation shuffle: {:?}",
            failures
        );
    }

    /// Condition (9)'s union direction must reject a host bit that points at the
    /// WRONG lane, not just a missing one: that is the difference between "some
    /// lane holds this key" and "the prover may name any lane it likes".
    #[test]
    fn mock_reject_wrong_host_lane() {
        let lane_rows = 2;
        let bag1_pad_extra = 5;
        let c = lanes_for(6 + bag1_pad_extra, lane_rows);
        let (lanes_hit, _) = clean_lane_spread(lane_rows);
        assert!(lanes_hit.len() >= 2);

        set_config_lanes(c);
        let circuit = MyCircuit::<Fp> {
            edges: synthetic_edges(),
            bag1_pad_extra,
            bag2_pad_extra: 3,
            lane_rows,
            num_lanes: c,
            tamper: Tamper::HostWrongLane,
            _marker: PhantomData,
        };
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(3u64)]]).unwrap();
        let failures = prover
            .verify()
            .expect_err("a clean Bag2 tuple hosted by the wrong lane was accepted");
        assert!(
            failures
                .iter()
                .any(|f| format!("{:?}", f).contains("pw:")),
            "rejected, but not through a Pairwise Consistency check: {:?}",
            failures
        );
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
        // `lane_span` is what the lanes COULD hold; the assigned rows are the
        // released capacity, since the last lane stops there.
        println!(
            "[gq3 dp-lanes] bag1_true={} n12={} lanes={} lane_rows={} lane_span={} \
             assigned_rows={}",
            p.true_size[0],
            p.capacity[0],
            p.lanes[0],
            p.lane_rows,
            p.lanes[0] * p.lane_rows,
            p.capacity[0]
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
