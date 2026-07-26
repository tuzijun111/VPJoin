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
//!     the C<D comparator), the prover's clean/residual partition of that
//!     block, the occurrence id, and a complete per-key CHILD STAGE of the
//!     Cardinality Preservation Check, which turns the lane's block of Bag2
//!     rows into a lane-local key-indexed table of TWO per-key sums.
//!   * A Bag1 lane replicates the T12 column group (the eight t12 columns, the
//!     A<B and B<C comparators), its own clean/residual partition, its
//!     occurrence id, one complete PARENT PROBE per Bag2 lane, and a lane-local
//!     prefix sum of each of the two channels.
//!
//! Lanes of a kind are identical by construction and all of them are assigned
//! to the full `lane_rows` rows, so the assigned structure is a function of the
//! PUBLIC released capacities only. It never depends on the true bag sizes. A
//! cheaper "overflow lane" that only carries padding would leak exactly the
//! quantity DP is paying to hide, so there is no such thing here.
//!
//! WHY LANE-LOCAL AGGREGATION IS SOUND
//! -----------------------------------
//! The message a Bag2 row sends is `sigma(pack2(A,C)) = COUNT(paths C->D->A
//! with C<D)`, a pure COUNT. Counts ADD across a partition, so splitting Bag2
//! block-wise over c2 lanes and grouping each block independently is exactly
//! correct: lane l's table holds `count_l(key)`, and
//!
//!     sum_l count_l(key) = count(key).
//!
//! A Bag1 row therefore probes all c2 tables and sums the c2 retrieved values.
//! Keys missing from a given lane's table are proven missing by that lane's gap
//! bracket and contribute 0, exactly as in the single-group circuit. No
//! ordering, no sentinel and no carry crosses a Bag2 lane boundary: the Bag2
//! side needs ZERO stitching for the aggregation itself.
//!
//! ONE-PASS OBJ ACROSS LANES
//! -------------------------
//! `g_sql4_obj` enforces three conditions of the One-Pass OBJ over a
//! prover-supplied partition of each bag into a clean part and a residual part.
//! A condition enforced PER LANE in a way that lets a prover split a violation
//! across two lanes is not enforced at all, so each one is realized here over
//! the UNION of the lanes:
//!
//!   (7) Conservation, `R == R^c U R^r`.
//!       The partition is carried as a boolean indicator column on the bag rows
//!       themselves (`t12_cflag` / `t34_cflag`), and the bit both channels of
//!       (10) actually read is the folded `ceff = predicate * cflag`. With the
//!       partition represented as an indicator ON `R`, `R^c = {r : ceff(r)=1}`
//!       and `R^r = {r : predicate(r)=1, ceff(r)=0}` are a partition of the
//!       predicate-passing rows BY CONSTRUCTION, per row and therefore over the
//!       union of the lanes: there is no second column group that could
//!       disagree with the bag, and no cross-lane split is expressible.
//!       `g_sql4_obj` instead lays the partition out as a separate
//!       [clean | residual | pad] column group and shuffles it against the bag.
//!       That form is equivalent, but it needs one selector per SECTION, whose
//!       enabled rows are the true clean and residual counts, i.e. it bakes the
//!       sensitive quantity this file exists to hide into the verifying key.
//!       See the note on that at `q_t12_row`.
//!   (9) Pairwise Consistency, `pi_K(R_i^c) == pi_K(R_j^c)` on the one bag tree
//!       edge, `K = pack2(A,C)`.
//!       A lookup's TABLE is a single set of column expressions, so it cannot be
//!       the union of c2 lane columns: membership in a laned relation has to be
//!       resolved arithmetically, by combining one per-lane answer per lane.
//!       That is what the clean channel already does. Every Bag1 row fetches
//!       `s_cln` from EVERY Bag2 lane, each fetch certified by that lane's
//!       membership lookup or by its gap bracket, so `sum_l s_cln_l(key)` is the
//!       number of clean Bag2 tuples with that key over ALL of Bag2. The gate
//!       `cp: root multiplicities and clean-key consistency` then requires a
//!       clean Bag1 row to have a nonzero total, which is exactly
//!       `pi_K(t12^c) subset of pi_K(t34^c)` over the union.
//!       The MIRROR inclusion is NOT enforced; see "WHAT IS NOT ENFORCED".
//!  (10) Cardinality Preservation, `|R^c join| == |R join|`.
//!       This is the one that must not be per lane. Each Bag1 lane accumulates
//!       its own `sum_all` and `sum_cln` over its own rows, both from
//!       multiplicities that already sum over all c2 Bag2 lanes; the c1 pairs of
//!       lane totals are then copied into the totals stage, added with two
//!       degree-2 accumulators, and compared by a SINGLE equality at row c1-1.
//!       A prover moving a hidden tuple's contribution into another lane changes
//!       one global total and nothing rebalances it.
//!
//! `sum_all` is also the ANSWER. `g_sql4_obj` propagates the per-key COUNT twice
//! (once through an `AggSumByKey` for the aggregate and once through the check's
//! own child table) and needs a gate to equate them; here the aggregate IS the
//! input channel of the check, `mu_all = pred * sum_l s_all_l`, so the certified
//! cardinality is the exposed COUNT by construction and there is nothing to
//! equate. That also makes this circuit CHEAPER than a laned `AggSumByKey` plus
//! a laned check would be.
//!
//! OCCURRENCE DISTINCTNESS ACROSS LANES
//! ------------------------------------
//! The four r1/r2/r3/r4 membership lookups say only that a bag row is SOME
//! valid wedge occurrence. Every argument downstream is blind to a DUPLICATE:
//! conservation is per row, (9) compares key sets, and (10) is a difference of
//! two channels, so copying a clean Bag2 tuple into a pad slot inflates
//! `s_all` and `s_cln` equally and inflates the COUNT with nothing rejecting.
//! In a laned layout the copy does not even have to stay in the same lane, so a
//! per-lane distinctness argument would miss it.
//!
//! Each bag row therefore carries `oid = pack2(the two source edge ids)` when
//! its predicate holds and PAD when it does not, and the bag rows are required
//! to be laid out in GLOBAL oid order: strictly increasing while below PAD, then
//! PAD for ever. Two uniform gates per lane say that over adjacent rows inside a
//! lane, and a `c-1` row boundary stage says it over each lane seam, with the
//! seam cells brought in by copy constraints (the only cells besides the two
//! lane sums that ever leave a lane). Concatenating the lanes in lane order, the
//! whole sequence is strictly increasing until PAD, so all non-PAD oids are
//! pairwise distinct ACROSS lanes, and the honest edge ids fit 32-bit lanes so
//! a passing row's oid is always below PAD.
//!
//! Note what this does and does not give. It rules out DUPLICATION, hence any
//! over-count. It cannot rule out OMISSION: `g_sql4_obj` pins the number of
//! predicate-passing rows through its partition sections, which is sound there
//! only because the Revealing-Join-Size regime publishes the true bag size. Here
//! the true bag size is precisely what the DP release hides, so no in-circuit
//! row count may depend on it, and bag completeness rests on the DP padding
//! scheme, which is a prover-side step this circuit deliberately does not
//! verify.
//!
//! WHAT IS NOT ENFORCED
//! --------------------
//! Condition (8) of the One-Pass OBJ (Non-Membership) is deliberately not
//! enforced, as in `g_sql4_obj`; with the partition carried as an indicator it
//! is vacuous anyway.
//!
//! The mirror half of (9), `pi_K(t34^c) subset of pi_K(t12^c)`, is not
//! enforced. Resolving it would need the whole message machinery mirrored: a
//! per-Bag1-lane clean-key child stage plus one probe per (Bag2 lane, Bag1 lane)
//! pair, i.e. a SECOND `c1 * c2` term, which roughly doubles the circuit that
//! the header cost note is already apologetic about. The exact residual freedom
//! it leaves the prover is: a KEPT Bag2 tuple whose separator key carries no
//! predicate-passing Bag1 tuple may be marked clean. Such a tuple's key is
//! multiplied by no Bag1 row, so it enters neither channel of (10) and no other
//! constrained quantity; it cannot move the COUNT, it only means `R^c` is not
//! fully reduced on the Bag2 end.
//!
//! COST NOTE (read before scaling this up)
//! ---------------------------------------
//! Probing c2 lane-local tables costs c1 * c2 probe replicas, so the Bag1 side
//! grows QUADRATICALLY in the lane counts, not linearly. That is inherent to
//! resolving a PRIVATE key against a table of private size: with c2 separate
//! tables a lookup argument can only address one of them at a time. It is fine
//! when one side collapses to a single lane (facebook and wiki at eps = 0.1
//! give c1 = c2 = 1) and it is roughly break-even against the k+2 jump at
//! lastfm eps = 0.1 (c1 = 4, c2 = 3). Getting a genuinely linear GQ4 needs a
//! different join strategy, namely replacing the message table by an in-circuit
//! sort-merge of the two bags, which is a different circuit rather than a
//! laned version of this one.
//!
//! WHAT STAYS SINGLE COPY
//! ----------------------
//! Everything whose height is fixed by the PUBLIC edge count: the base Edge
//! relation (`e_src`, `e_dst`, `e_eid`) and the two indexed views of it
//! (`in_by_dst`, `out_by_src`), which are the table side of every r1/r2/r3/r4
//! membership lookup. Their height tracks |E|, which is public in every privacy
//! regime. Two shuffles tie both views to the base relation, so the four bag
//! lookups read ONE edge relation instead of two independently invented ones,
//! and `q_view_tbl` keeps each view's sentinel row out of all four tables.
//!
//! WHERE THE STITCHES ARE
//! ----------------------
//! Three, all of them small and all of them cross-lane BY DESIGN:
//!   * the two channels of (10): c1 pairs of copy constraints into the totals
//!     stage, two degree-2 accumulators over c1 rows, one equality at row c1-1,
//!     and `out` at row c1-1 into the instance;
//!   * the global occurrence ordering: 2*(c1-1) and 2*(c2-1) copy constraints
//!     into the two boundary stages;
//!   * nothing else. A Bag2 lane still exports nothing by copy: its two per-key
//!     sums reach Bag1 through lookups.
//!
//! MEASURED STRUCTURE (see `tests::test_max_gate_degree`)
//! -----------------------------------------------------
//!   c1=1 c2=1 : advice 207  fixed 3  sel 51  lookups 110  shuf  6  perm 50
//!   c1=2 c2=1 : advice 276  fixed 3  sel 51  lookups 154  shuf  6  perm 53
//!   c1=1 c2=2 : advice 300  fixed 3  sel 68  lookups 162  shuf  8  perm 70
//!   c1=4 c2=3 : advice 738  fixed 3  sel 85  lookups 454  shuf 10  perm 99
//! so one more Bag1 lane costs 69 advice / 3 permutation columns / 44 lookup
//! arguments and nothing else, one more Bag2 lane costs 93 advice / 20
//! permutation columns / 17 selectors / 2 shuffles / 52 lookups, and each probe
//! pair costs 23 advice and 18 lookups and nothing else.
//!
//! Before the OBJ conditions were added this file measured 141 / 580 advice and
//! 31 / 58 permutation columns at (1,1) / (4,3), so the whole gate -- the two
//! partitions, the clean channel, condition (9), the global occurrence
//! ordering, the base Edge relation and its two shuffles -- costs about 27% more
//! advice at the production cell. Replacing the old `AggSumByKey` + `MapLookup`
//! pair by the check's own child stage and parent probe is what keeps it that
//! cheap: the second channel rides along on machinery the aggregate needed
//! anyway.
//!
//! cs.degree() is 7 for every configuration, the same bound `g_sql4_obj` pays
//! for its own membership lookup (2 + 3 + 2). `lanes_are_structural_replicas`
//! locks the shape down as EXACTLY BILINEAR in (c1, c2), which is the privacy
//! property that every lane of a kind costs the same. Per Bag1 lane the circuit
//! adds no fixed column, no selector and no shuffle, and exactly three
//! permutation columns (`sum_all`, `sum_cln`, `t12_oid`); per Bag2 lane it adds
//! no fixed column; per probe pair it adds no fixed column, no selector, no
//! shuffle and no permutation column.
//!
//! Row budget per lane, at circuit degree k:
//!   lane_rows = 2^k - PREAMBLE_ROWS - BLINDING_SLACK
//! where PREAMBLE_ROWS covers the u8 range-table `load` regions (each is 256
//! fixed rows in a region of its own, and the floor planner is free to place
//! them ahead of the witness region) and BLINDING_SLACK covers the blinding
//! rows halo2 reserves at the bottom of every advice column.

use halo2_proofs::plonk::Expression;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};
use crate::circuits::card_preserve::{
    assign_cp_agg, build_cp_stage, configure_cp_agg, cp_gap_witness, wire_cp_edge, CpAggConfig,
    CpJoinConfig, CpStage,
};
use crate::data::graph_data_processing::Edge;

// One shared definition of the PAD / packing conventions, the two chips whose
// height is public, and the host-side witness derivation (from g_sql4_obj).
use super::g_sql4_obj::{
    gq4_derive, pack2, Field, IndexedViewChip, IndexedViewConfig, NUM_BYTES, PACK_SHIFT, PAD_U64,
};

use std::collections::{BTreeMap, HashSet};
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

/// Lane count for a released bag capacity at a caller-supplied base degree,
/// with the structural cap enforced. Panics on an infeasible capacity; see
/// [`try_lanes_for_capacity`].
pub fn lanes_for_capacity(capacity: usize, base_degree: u32) -> usize {
    try_lanes_for_capacity(capacity, base_degree).unwrap_or_else(|e| panic!("{}", e))
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
    /// Bump one Bag1 lane row's input-channel multiplicity by 1 (the root
    /// multiplicity gate and that lane's own prefix-sum gates must reject).
    LaneMuAll,
    /// Bump one Bag1 lane total by 1 (the copy constraint into the totals
    /// stage and the totals accumulator must both reject).
    LaneTotal,
    /// Break the Bag1 pad-row convention on the last rounding-up row (the
    /// "bag1 real + key" gate and the r1 membership lookup must reject).
    Bag1PadRow,
    /// Break the Bag2 pad-row convention on the last rounding-up row (the
    /// r3 membership lookup must reject).
    Bag2PadRow,
    /// Zero the retrieved input-channel value of every probe EXCEPT probe 0,
    /// while leaving the root multiplicity at its honest (all-lane) value. This
    /// is the direct negative test for the cross-lane message sum: if the root
    /// gate ever stopped summing all `c2` probes, or if a probe's membership
    /// lookup were dropped, this witness would verify. It must not.
    DropProbeVal,
    /// Move one joinable Bag1 tuple to the residual side and re-reduce Bag2
    /// around it, so conditions (7) and (9) still hold and only the Cardinality
    /// Preservation Check can see the hidden tuple. The check compares GLOBAL
    /// totals, so it sees it in whichever lane the tuple lives.
    HideOneCleanTuple,
    /// Skip the semijoin reduction and declare every predicate-passing tuple of
    /// both bags clean, leaving the residual side empty. Both channels of (10)
    /// then agree on every row, so only Pairwise Consistency can see the
    /// dangling Bag1 tuples.
    MarkAllClean,
    /// Copy the first kept Bag2 occurrence over the FIRST row of the LAST Bag2
    /// lane. Both copies pass all four membership lookups and both channels of
    /// (10) move by the same amount, and every lane's own occurrence block stays
    /// sorted, so the ONLY constraint that can see the duplicate is the lane
    /// seam of the global ordering. A per-lane ordering would accept it.
    DuplicateOccurrenceAcrossLanes,
}

// ---------------------------------------------------------------------------
// Bag rows, in the order the global occurrence ordering requires.
// ---------------------------------------------------------------------------

/// One bag row as the circuit sees it. Bag1: `(A,B,C,i_r1,j_r2,r1_eid,r2_eid,
/// real)`. Bag2: `(C,D,A,i_r3,j_r4,r3_eid,r4_eid,real)`.
pub(crate) type BagRow = [u64; 8];

/// Bag1's per-row predicate bit, `real * [A<B] * [B<C]`.
pub(crate) fn bag1_pred(r: &BagRow) -> u64 {
    (r[7] == 1 && r[0] < r[1] && r[1] < r[2]) as u64
}

/// Bag2's per-row predicate bit, `real * [C<D]`.
pub(crate) fn bag2_keep(r: &BagRow) -> u64 {
    (r[7] == 1 && r[0] < r[1]) as u64
}

/// The occurrence id of a bag row: `pack2` of the two source edge ids when the
/// row's predicate holds, PAD when it does not.
pub(crate) fn bag_oid(r: &BagRow, keep: u64) -> u64 {
    if keep == 1 {
        pack2(r[5], r[6])
    } else {
        PAD_U64
    }
}

/// Bag1 over the whole released capacity, in the global occurrence-id order the
/// distinctness argument requires: predicate-passing rows first, ordered by
/// occurrence id, then the real rows that fail the predicate, then padding.
pub(crate) fn bag1_rows(
    t12: &[(u64, u64, u64, u64, u64, u64, u64)],
    capacity: usize,
) -> Vec<BagRow> {
    let mut rows: Vec<BagRow> = t12
        .iter()
        .map(|&(a, b, c, i1, j2, e1, e2)| [a, b, c, i1, j2, e1, e2, 1])
        .collect();
    rows.truncate(capacity);
    rows.resize(capacity, [0u64; 8]);
    rows.sort_by_key(|r| bag_oid(r, bag1_pred(r)));
    rows
}

/// Bag2 over the whole released capacity, same ordering convention.
pub(crate) fn bag2_rows(
    t34: &[(u64, u64, u64, u64, u64, u64, u64)],
    capacity: usize,
) -> Vec<BagRow> {
    let mut rows: Vec<BagRow> = t34
        .iter()
        .map(|&(c, d, a, i3, j4, e3, e4)| [c, d, a, i3, j4, e3, e4, 1])
        .collect();
    rows.truncate(capacity);
    rows.resize(capacity, [0u64; 8]);
    rows.sort_by_key(|r| bag_oid(r, bag2_keep(r)));
    rows
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// The lane-seam link of one bag's global occurrence ordering. Height `c-1`:
/// row l compares the last oid of lane l against the first oid of lane l+1.
#[derive(Clone, Debug)]
pub struct OidBoundaryConfig<F: Field + Ord> {
    lo: Column<Advice>,
    hi: Column<Advice>,
    iz_pad: IsZeroConfig<F>,
    lt: LtConfig<F, NUM_BYTES>,
    q: Selector,
}

/// One Bag2 lane: the T34 column group, the prover's partition of this lane's
/// block, its occurrence id, and the CHILD stage of the single bag tree edge,
/// which publishes both per-key sums for this block.
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

    // condition (7): the predicate bit, the separator key, the prover's clean
    // indicator, and the folded `keep * cflag` both channels read
    t34_keep: Column<Advice>,
    t34_key: Column<Advice>,
    t34_cflag: Column<Advice>,
    t34_ceff: Column<Advice>,

    // occurrence id and its within-lane ordering
    t34_oid: Column<Advice>,
    iz_oid_pad: IsZeroConfig<F>,
    lt_oid: LtConfig<F, NUM_BYTES>,

    // child side of the bag tree edge: per key, (sum keep, sum keep*cflag)
    cp: CpAggConfig<F, NUM_BYTES>,
}

/// One Bag1 lane: the T12 column group, its own partition, its occurrence id,
/// one parent probe per Bag2 lane, and a LANE-LOCAL prefix sum of each channel.
/// `out` is NOT replicated; it lives once in the cross-lane totals stage.
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

    // ordering predicates A<B and B<C
    lt_ab: LtConfig<F, NUM_BYTES>,
    lt_bc: LtConfig<F, NUM_BYTES>,

    // separator key, folded predicate bit, and condition (7)'s two indicators
    t12_key: Column<Advice>,
    t12_pred: Column<Advice>,
    t12_cflag: Column<Advice>,
    t12_ceff: Column<Advice>,

    // occurrence id and its within-lane ordering
    t12_oid: Column<Advice>,
    iz_oid_pad: IsZeroConfig<F>,
    lt_oid: LtConfig<F, NUM_BYTES>,

    // membership-or-gap probe of the child table, ONE PER BAG2 LANE. Counts add
    // across a partition, so the row's two multiplicities are the sums of the
    // c2 retrieved values.
    probes: Vec<CpJoinConfig<F, NUM_BYTES>>,

    // the two root multiplicities and their lane-local prefix sums. `mu_all` is
    // both the answer's per-row contribution and the input channel of (10).
    mu_all: Column<Advice>,
    mu_cln: Column<Advice>,
    sum_all: Column<Advice>,
    sum_cln: Column<Advice>,

    // is the clean multiplicity fetched from ALL Bag2 lanes zero? condition (9)
    iz_cln: IsZeroConfig<F>,
}

#[derive(Clone, Debug)]
pub struct Gq4DpConfig<F: Field + Ord> {
    instance: Column<Instance>,

    // ---------------- fixed-height sections (shared, never laned) ----------
    // The base Edge relation, and the two indexed views projected from it. The
    // views are the table side of every r1/r2/r3/r4 membership lookup; their
    // height tracks |E|, which is public in every regime.
    e_eid: Column<Advice>,
    e_src: Column<Advice>,
    e_dst: Column<Advice>,
    in_by_dst: IndexedViewConfig<F>,  // key=dst, val=src
    out_by_src: IndexedViewConfig<F>, // key=src, val=dst

    /// Table-side gate of the four bag lookups, enabled on exactly the n_base
    /// real rows of both views. See `configure_bag2_lane` for why.
    q_view_tbl: Selector,

    /// Both views are the base relation's (src, dst, eid) multiset.
    perm_edge_out: PermAnyConfig,
    perm_edge_in: PermAnyConfig,

    /// One u8 range table for EVERY lane Lt chip, of either kind. `load` costs
    /// one 256-row region per distinct u8 column, and that must not grow with
    /// the lane counts.
    lane_u8: Column<Fixed>,

    // ---------------- Bag2 lanes ------------------------------------------
    // All Bag2 lanes are active on the SAME rows 0..lane_rows, so one selector
    // of each kind serves every lane; only the columns replicate. (The
    // per-lane child stages keep their own selectors, since `assign_cp_agg`
    // enables them itself.)
    bag2_lanes: Vec<Bag2LaneConfig<F>>,
    q_t34_lookup: Selector, // r3/r4 membership lookups (complex)
    q_t34_flag: Selector,
    q_cd: Selector,
    q_t34_row: Selector,

    // ---------------- Bag1 lanes ------------------------------------------
    bag1_lanes: Vec<Bag1LaneConfig<F>>,
    q_t12_lookup: Selector, // r1/r2 membership lookups (complex)
    q_t12_key: Selector,
    q_t12_row: Selector,
    q_order12: Selector,
    // The two probe selectors are not repeated here: every probe carries them,
    // and `assign_probe` enables them through the probe it is assigning.
    q_mu: Selector,   // the two root multiplicities, and condition (9)
    q_sum0: Selector, // lane-local prefix sums, row 0 of every Bag1 lane
    q_sum: Selector,  // lane-local prefix sums, rows 1..lane_rows

    // ---------------- global occurrence ordering ---------------------------
    // Shared by both kinds of lane: they are live on the same rows.
    q_oid_step: Selector,
    /// `[Bag1, Bag2]`: the lane-seam link of each bag's ordering.
    oid_bnd: Vec<OidBoundaryConfig<F>>,

    // ---------------- cross-lane totals ------------------------------------
    // Row l holds Bag1 lane l's two channel totals, by copy constraint. This is
    // where condition (10) is compared and where the answer comes from.
    lane_tot: Column<Advice>,
    cln_tot: Column<Advice>,
    tot_run: Column<Advice>,
    cln_run: Column<Advice>,
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
    /// The two gates that make one lane's occurrence-id column strictly
    /// increasing while it is below PAD and PAD for ever afterwards. Uniform
    /// over `0..lane_rows-1`: no selector row depends on the witness.
    fn configure_oid_order(
        meta: &mut ConstraintSystem<F>,
        lane_u8: Column<Fixed>,
        q_step: Selector,
        oid: Column<Advice>,
    ) -> (IsZeroConfig<F>, LtConfig<F, NUM_BYTES>) {
        let aux = meta.advice_column();
        let iz_pad = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_step),
            |m| m.query_advice(oid, Rotation::cur()) - Expression::Constant(F::from(PAD_U64)),
            aux,
        );
        let lt = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            lane_u8,
            |m| m.query_selector(q_step),
            |m| m.query_advice(oid, Rotation::cur()),
            |m| m.query_advice(oid, Rotation::next()),
        );

        let izc = iz_pad.clone();
        meta.create_gate(
            "occ: ids strictly increase until PAD, then stay PAD",
            move |m| {
                let q = m.query_selector(q_step);
                let one = Expression::Constant(F::ONE);
                let pad = Expression::Constant(F::from(PAD_U64));
                let is_pad = izc.expr();
                vec![
                    q.clone() * (one.clone() - is_pad.clone()) * (one - lt.is_lt(m, None)),
                    q * is_pad * (m.query_advice(oid, Rotation::next()) - pad),
                ]
            },
        );

        (iz_pad, lt)
    }

    /// The same two statements over one lane seam, on a stage of `c-1` rows
    /// whose two cells are copied in from the two lanes it joins. Without this
    /// the ordering would only be per lane and a prover could park a duplicate
    /// occurrence in another lane.
    fn configure_oid_boundary(
        meta: &mut ConstraintSystem<F>,
        lane_u8: Column<Fixed>,
    ) -> OidBoundaryConfig<F> {
        let lo = meta.advice_column();
        let hi = meta.advice_column();
        meta.enable_equality(lo);
        meta.enable_equality(hi);
        let q = meta.selector();

        let aux = meta.advice_column();
        let iz_pad = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q),
            |m| m.query_advice(lo, Rotation::cur()) - Expression::Constant(F::from(PAD_U64)),
            aux,
        );
        let lt = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            lane_u8,
            |m| m.query_selector(q),
            |m| m.query_advice(lo, Rotation::cur()),
            |m| m.query_advice(hi, Rotation::cur()),
        );

        let izc = iz_pad.clone();
        meta.create_gate("occ: the ordering carries across the lane seam", move |m| {
            let qs = m.query_selector(q);
            let one = Expression::Constant(F::ONE);
            let pad = Expression::Constant(F::from(PAD_U64));
            let is_pad = izc.expr();
            vec![
                qs.clone() * (one.clone() - is_pad.clone()) * (one - lt.is_lt(m, None)),
                qs * is_pad * (m.query_advice(hi, Rotation::cur()) - pad),
            ]
        });

        OidBoundaryConfig {
            lo,
            hi,
            iz_pad,
            lt,
            q,
        }
    }

    /// The parent side of one (Bag1 lane, Bag2 lane) probe.
    ///
    /// This is `card_preserve::configure_cp_join` with two changes a replicated
    /// caller needs. The two selectors are supplied rather than allocated, since
    /// all c1*c2 probes are live on the same rows and a selector per probe would
    /// make the selector count carry a c1*c2 term. And the five columns are kept
    /// OUT of the permutation argument: halo2 charges for every column in it
    /// whether or not a copy constraint touches it, and nothing copies a probe
    /// cell.
    fn configure_probe(
        meta: &mut ConstraintSystem<F>,
        lane_u8: Column<Fixed>,
        q_lookup: Selector,
        q_lookup_complex: Selector,
        key_col: Column<Advice>,
    ) -> CpJoinConfig<F, NUM_BYTES> {
        let in_tbl = meta.advice_column();
        let low = meta.advice_column();
        let high = meta.advice_column();
        let s_all = meta.advice_column();
        let s_cln = meta.advice_column();

        let lt_low = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            lane_u8,
            |m| {
                let q = m.query_selector(q_lookup);
                let inx = m.query_advice(in_tbl, Rotation::cur());
                q * (Expression::Constant(F::ONE) - inx)
            },
            |m| m.query_advice(low, Rotation::cur()),
            |m| m.query_advice(key_col, Rotation::cur()),
        );
        let lt_high = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            lane_u8,
            |m| {
                let q = m.query_selector(q_lookup);
                let inx = m.query_advice(in_tbl, Rotation::cur());
                q * (Expression::Constant(F::ONE) - inx)
            },
            |m| m.query_advice(key_col, Rotation::cur()),
            |m| m.query_advice(high, Rotation::cur()),
        );

        meta.create_gate("cp: membership flag, gap and zero default", move |m| {
            let q = m.query_selector(q_lookup);
            let inx = m.query_advice(in_tbl, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let not_in = one.clone() - inx.clone();
            let low_ok = lt_low.is_lt(m, None);
            let high_ok = lt_high.is_lt(m, None);
            vec![
                q.clone() * inx.clone() * (one.clone() - inx),
                q.clone() * not_in.clone() * (one.clone() - low_ok),
                q.clone() * not_in.clone() * (one - high_ok),
                q.clone() * not_in.clone() * m.query_advice(s_all, Rotation::cur()),
                q * not_in * m.query_advice(s_cln, Rotation::cur()),
            ]
        });

        CpJoinConfig {
            in_tbl,
            low,
            high,
            s_all,
            s_cln,
            q_lookup,
            q_lookup_complex,
            lt_low,
            lt_high,
        }
    }

    /// One full structural replica of the Bag2 column group, its partition and
    /// its child stage. Every Bag2 lane gets the same gates and the same
    /// lookups; only the columns differ.
    #[allow(clippy::too_many_arguments)]
    fn configure_bag2_lane(
        meta: &mut ConstraintSystem<F>,
        in_by_dst: &IndexedViewConfig<F>,
        out_by_src: &IndexedViewConfig<F>,
        lane_u8: Column<Fixed>,
        q_view_tbl: Selector,
        q_t34_lookup: Selector,
        q_t34_flag: Selector,
        q_cd: Selector,
        q_t34_row: Selector,
        q_oid_step: Selector,
    ) -> Bag2LaneConfig<F> {
        let t34_c = meta.advice_column();
        let t34_d = meta.advice_column();
        let t34_a = meta.advice_column();
        let t34_i_r3 = meta.advice_column();
        let t34_j_r4 = meta.advice_column();
        let t34_r3_eid = meta.advice_column();
        let t34_r4_eid = meta.advice_column();
        let t34_real = meta.advice_column();
        // NOTE: no `enable_equality` on the tuple columns. halo2 charges for
        // every column in the permutation argument whether or not a copy
        // constraint ever touches it, so a per-lane column that is only read by
        // gates and lookups must stay out of it. The only Bag2 lane cells that
        // are ever copied are the two ends of `t34_oid`, for the lane seam.

        // The table side of the four bag lookups is gated by `q_view_tbl`, over
        // exactly the n_base real rows of both views.
        //
        // Each `IndexedView` carries a sentinel row at `n_base` for the
        // Rotation::next() of its own sortedness gate. That row is outside the
        // view's shuffle, outside `q_idx0` and outside `q_idx`, so `sorted_val`,
        // `sorted_eid` and `idx` there are free advice and only `sorted_key` is
        // touched at all (bounded above the largest real key). With an ungated
        // table side that sentinel is a LIVE table row, so a prover can mint an
        // edge the relation does not contain: set out_by_src's sentinel to
        // (key = D, idx = 0, val = A0, eid = anything) with D above every real
        // src, and a Bag2 row (C0, D, A0) then passes "bag2 r4 from out_by_src"
        // for an r4 = D->A0 edge that exists nowhere.
        //
        // On the ungated rows every table expression reads 0, so the tuple
        // (0,0,0,0) is in the table, which is what the padded bag rows look up
        // anyway, and it is a real row of both views regardless: base row 0 is
        // the (0,0,0) dummy edge with idx 0. The table side goes from degree 1
        // to degree 2, so these lookups go from 2+2+1 = 5 to 2+2+2 = 6, still
        // under the 2+3+2 = 7 the check's own membership lookup costs, and
        // cs.degree() does not move.
        meta.lookup_any("bag2 r3 from in_by_dst", |m| {
            let q = m.query_selector(q_t34_lookup);
            let t = m.query_selector(q_view_tbl);
            vec![
                (
                    q.clone() * m.query_advice(t34_d, Rotation::cur()),
                    t.clone() * m.query_advice(in_by_dst.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t34_i_r3, Rotation::cur()),
                    t.clone() * m.query_advice(in_by_dst.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t34_c, Rotation::cur()),
                    t.clone() * m.query_advice(in_by_dst.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t34_r3_eid, Rotation::cur()),
                    t * m.query_advice(in_by_dst.sorted_eid, Rotation::cur()),
                ),
            ]
        });
        // r4 via OutBySrc: key=D, idx=j_r4 -> val=A, eid=r4_eid  (r4 is D->A).
        // This is also the closing-edge check: `val` is t34_a, so the edge
        // D->A that closes the cycle is proven to exist by the same lookup
        // that produces A.
        meta.lookup_any("bag2 r4 from out_by_src", |m| {
            let q = m.query_selector(q_t34_lookup);
            let t = m.query_selector(q_view_tbl);
            vec![
                (
                    q.clone() * m.query_advice(t34_d, Rotation::cur()),
                    t.clone() * m.query_advice(out_by_src.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t34_j_r4, Rotation::cur()),
                    t.clone() * m.query_advice(out_by_src.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t34_a, Rotation::cur()),
                    t.clone() * m.query_advice(out_by_src.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t34_r4_eid, Rotation::cur()),
                    t * m.query_advice(out_by_src.sorted_eid, Rotation::cur()),
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

        // ---- condition (7) on this lane's block of Bag2 ----
        // `keep` is the predicate bit, `key` the separator key on the kept rows
        // and PAD elsewhere (a PAD key never reaches the child table), `cflag`
        // the prover's clean indicator and `ceff = keep * cflag` the bit the
        // clean channel reads. `g_sql4_obj` had no booleanity gate on the
        // indicator at all; without it a fractional "indicator" would scale the
        // clean channel freely. And `ceff` is what forces the indicator to 0
        // wherever the predicate fails, so a row that is not in R cannot be in
        // R^c.
        let t34_keep = meta.advice_column();
        let t34_key = meta.advice_column();
        let t34_cflag = meta.advice_column();
        let t34_ceff = meta.advice_column();

        meta.create_gate("bag2 keep, separator key and clean partition", |m| {
            let q = m.query_selector(q_t34_row);
            let one = Expression::Constant(F::ONE);
            let pad = Expression::Constant(F::from(PAD_U64));

            let real = m.query_advice(t34_real, Rotation::cur());
            let cd = lt_cd.is_lt(m, None);
            let keep = m.query_advice(t34_keep, Rotation::cur());
            let key = m.query_advice(t34_key, Rotation::cur());
            let cf = m.query_advice(t34_cflag, Rotation::cur());
            let ce = m.query_advice(t34_ceff, Rotation::cur());

            let key_expr = m.query_advice(t34_a, Rotation::cur())
                * Expression::Constant(F::from(PACK_SHIFT))
                + m.query_advice(t34_c, Rotation::cur());

            vec![
                q.clone() * (keep.clone() - real * cd),
                q.clone()
                    * (key - (keep.clone() * key_expr + (one.clone() - keep.clone()) * pad)),
                q.clone() * cf.clone() * (one - cf.clone()),
                q * (ce - keep * cf),
            ]
        });

        // ---- occurrence id ----
        let t34_oid = meta.advice_column();
        meta.enable_equality(t34_oid);
        meta.create_gate("occ: bag2 id = pack2(edge ids), PAD when not kept", |m| {
            let q = m.query_selector(q_t34_row);
            let one = Expression::Constant(F::ONE);
            let pad = Expression::Constant(F::from(PAD_U64));
            let keep = m.query_advice(t34_keep, Rotation::cur());
            let id = m.query_advice(t34_r3_eid, Rotation::cur())
                * Expression::Constant(F::from(PACK_SHIFT))
                + m.query_advice(t34_r4_eid, Rotation::cur());
            vec![
                q * (m.query_advice(t34_oid, Rotation::cur())
                    - (keep.clone() * id + (one - keep) * pad)),
            ]
        });
        let (iz_oid_pad, lt_oid) = Self::configure_oid_order(meta, lane_u8, q_oid_step, t34_oid);

        // ---- child side of the bag tree edge, both channels ----
        // Bag2 is a leaf of the bag tree, so a row's input-channel multiplicity
        // is its predicate bit and its clean-channel multiplicity is that bit
        // times the clean indicator. Both already sit in columns this lane
        // carries, so the second channel costs only the stage's own columns and
        // the stage groups them by the separator key in one pass.
        let cp = configure_cp_agg::<F, NUM_BYTES>(
            meta, lane_u8, t34_key, t34_keep, t34_ceff, PAD_U64,
        );

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
            t34_keep,
            t34_key,
            t34_cflag,
            t34_ceff,
            t34_oid,
            iz_oid_pad,
            lt_oid,
            cp,
        }
    }

    /// One full structural replica of the Bag1 column group, carrying one probe
    /// per Bag2 lane and the two root channels. Every Bag1 lane gets the same
    /// gates and the same lookups; only the columns differ.
    #[allow(clippy::too_many_arguments)]
    fn configure_bag1_lane(
        meta: &mut ConstraintSystem<F>,
        in_by_dst: &IndexedViewConfig<F>,
        out_by_src: &IndexedViewConfig<F>,
        bag2_lanes: &[Bag2LaneConfig<F>],
        lane_u8: Column<Fixed>,
        q_view_tbl: Selector,
        sel: &Bag1Selectors,
    ) -> Bag1LaneConfig<F> {
        let Bag1Selectors {
            q_t12_lookup,
            q_t12_key,
            q_t12_row,
            q_order12,
            q_probe,
            q_probe_complex,
            q_mu,
            q_sum0,
            q_sum,
            q_oid_step,
        } = *sel;

        let t12_a = meta.advice_column();
        let t12_b = meta.advice_column();
        let t12_c = meta.advice_column();
        let t12_i_r1 = meta.advice_column();
        let t12_j_r2 = meta.advice_column();
        let t12_r1_eid = meta.advice_column();
        let t12_r2_eid = meta.advice_column();
        let t12_real = meta.advice_column();
        // NOTE: no `enable_equality` here either; see the Bag2 lane. The Bag1
        // lane cells that are copied are the two ends of `t12_oid` and the last
        // `sum_all` / `sum_cln`.

        // r1 via InByDst: key=B, idx=i_r1 -> val=A, eid=r1_eid. Table side gated
        // by `q_view_tbl`, for the reason spelled out in `configure_bag2_lane`.
        meta.lookup_any("bag1 r1 from in_by_dst", |m| {
            let q = m.query_selector(q_t12_lookup);
            let t = m.query_selector(q_view_tbl);
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
            let t = m.query_selector(q_view_tbl);
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

        // The separator key every probe asks about. On a padding row the tuple
        // is all zero, so the key is 0, which is the dummy row of every child
        // table and carries both sums 0.
        let t12_key = meta.advice_column();
        meta.create_gate("bag1 real + key", |m| {
            let q = m.query_selector(q_t12_key);
            let one = Expression::Constant(F::ONE);
            let real = m.query_advice(t12_real, Rotation::cur());
            let key_expr = m.query_advice(t12_a, Rotation::cur())
                * Expression::Constant(F::from(PACK_SHIFT))
                + m.query_advice(t12_c, Rotation::cur());
            vec![
                q.clone() * real.clone() * (one - real),
                q * (m.query_advice(t12_key, Rotation::cur()) - key_expr),
            ]
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

        // ---- condition (7) on this lane's block of Bag1 ----
        // `pred` folds the three-factor predicate into a column so that every
        // gate below stays at degree <= 4; `cflag` is the prover's clean
        // indicator, gated boolean, and `ceff = pred * cflag` is the bit the
        // clean channel reads, which is 0 wherever the predicate fails.
        let t12_pred = meta.advice_column();
        let t12_cflag = meta.advice_column();
        let t12_ceff = meta.advice_column();
        meta.create_gate("bag1 pred and clean partition", |m| {
            let q = m.query_selector(q_t12_row);
            let one = Expression::Constant(F::ONE);
            let real = m.query_advice(t12_real, Rotation::cur());
            let ab = lt_ab.is_lt(m, None);
            let bc = lt_bc.is_lt(m, None);
            let pred = m.query_advice(t12_pred, Rotation::cur());
            let cf = m.query_advice(t12_cflag, Rotation::cur());
            let ce = m.query_advice(t12_ceff, Rotation::cur());
            vec![
                q.clone() * (pred.clone() - real * ab * bc),
                q.clone() * cf.clone() * (one - cf.clone()),
                q * (ce - pred * cf),
            ]
        });

        // ---- occurrence id ----
        let t12_oid = meta.advice_column();
        meta.enable_equality(t12_oid);
        meta.create_gate(
            "occ: bag1 id = pack2(edge ids), PAD when the predicate fails",
            |m| {
                let q = m.query_selector(q_t12_row);
                let one = Expression::Constant(F::ONE);
                let pad = Expression::Constant(F::from(PAD_U64));
                let pred = m.query_advice(t12_pred, Rotation::cur());
                let id = m.query_advice(t12_r1_eid, Rotation::cur())
                    * Expression::Constant(F::from(PACK_SHIFT))
                    + m.query_advice(t12_r2_eid, Rotation::cur());
                vec![
                    q * (m.query_advice(t12_oid, Rotation::cur())
                        - (pred.clone() * id + (one - pred) * pad)),
                ]
            },
        );
        let (iz_oid_pad, lt_oid) = Self::configure_oid_order(meta, lane_u8, q_oid_step, t12_oid);

        // ---- one probe per Bag2 lane ----
        // Each probe is a complete replica pointed at that lane's child table;
        // they share the two probe selectors and the u8 column because every
        // probe is live on every row.
        let probes: Vec<CpJoinConfig<F, NUM_BYTES>> = bag2_lanes
            .iter()
            .map(|b2| {
                let j = Self::configure_probe(meta, lane_u8, q_probe, q_probe_complex, t12_key);
                wire_cp_edge(meta, &j, &b2.cp, t12_key);
                j
            })
            .collect();

        // ---- the two root multiplicities, and condition (9) ----
        let mu_all = meta.advice_column();
        let mu_cln = meta.advice_column();
        let sum_all = meta.advice_column();
        let sum_cln = meta.advice_column();
        meta.enable_equality(sum_all);
        meta.enable_equality(sum_cln);

        // Is the clean multiplicity of this row's key, summed over ALL Bag2
        // lanes, zero? A per-lane test would be meaningless: a key legitimately
        // lives in only some of the lanes.
        let aux_cln = meta.advice_column();
        let probes_iz = probes.clone();
        let iz_cln = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_mu),
            move |m| {
                probes_iz.iter().fold(Expression::Constant(F::ZERO), |acc, p| {
                    acc + m.query_advice(p.s_cln, Rotation::cur())
                })
            },
            aux_cln,
        );

        let probes_mu = probes.clone();
        let izc = iz_cln.clone();
        meta.create_gate(
            "cp: root multiplicities and clean-key consistency",
            move |m| {
                let q = m.query_selector(q_mu);
                let mut s_all = Expression::Constant(F::ZERO);
                let mut s_cln = Expression::Constant(F::ZERO);
                for p in probes_mu.iter() {
                    s_all = s_all + m.query_advice(p.s_all, Rotation::cur());
                    s_cln = s_cln + m.query_advice(p.s_cln, Rotation::cur());
                }
                let pred = m.query_advice(t12_pred, Rotation::cur());
                let ceff = m.query_advice(t12_ceff, Rotation::cur());
                vec![
                    // the input channel, which is also this row's contribution
                    // to the COUNT
                    q.clone() * (m.query_advice(mu_all, Rotation::cur()) - pred * s_all),
                    // the clean channel
                    q.clone() * (m.query_advice(mu_cln, Rotation::cur()) - ceff.clone() * s_cln),
                    // condition (9), the direction pi_K(t12^c) subset of
                    // pi_K(t34^c): a Bag1 tuple left on the clean side must have
                    // at least one clean Bag2 partner SOMEWHERE in Bag2
                    q * ceff * izc.expr(),
                ]
            },
        );

        // Lane-local prefix sums of both channels: `q_sum0` fires at THIS
        // lane's row 0, so no accumulator ever crosses a lane boundary. The two
        // lane totals leave through the totals stage, which is where the single
        // cross-lane comparison of condition (10) happens.
        meta.create_gate("cp: lane sums first row", |m| {
            let q = m.query_selector(q_sum0);
            vec![
                q.clone()
                    * (m.query_advice(sum_all, Rotation::cur())
                        - m.query_advice(mu_all, Rotation::cur())),
                q * (m.query_advice(sum_cln, Rotation::cur())
                    - m.query_advice(mu_cln, Rotation::cur())),
            ]
        });
        meta.create_gate("cp: lane sums accumulate", |m| {
            let q = m.query_selector(q_sum);
            vec![
                q.clone()
                    * (m.query_advice(sum_all, Rotation::cur())
                        - (m.query_advice(sum_all, Rotation::prev())
                            + m.query_advice(mu_all, Rotation::cur()))),
                q * (m.query_advice(sum_cln, Rotation::cur())
                    - (m.query_advice(sum_cln, Rotation::prev())
                        + m.query_advice(mu_cln, Rotation::cur()))),
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
            lt_ab,
            lt_bc,
            t12_key,
            t12_pred,
            t12_cflag,
            t12_ceff,
            t12_oid,
            iz_oid_pad,
            lt_oid,
            probes,
            mu_all,
            mu_cln,
            sum_all,
            sum_cln,
            iz_cln,
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
        let e_eid = meta.advice_column();
        let e_src = meta.advice_column();
        let e_dst = meta.advice_column();
        for c in [e_eid, e_src, e_dst] {
            meta.enable_equality(c);
        }

        let in_by_dst = IndexedViewChip::<F>::configure(meta);
        let out_by_src = IndexedViewChip::<F>::configure(meta);
        let q_view_tbl = meta.complex_selector();

        // -------- Conservation of the edge relation --------
        // The two views were free advice tied only to their own sorted view,
        // with nothing relating them to each other, so the four bag lookups
        // proved membership in two INDEPENDENT prover-invented relations: r1/r3
        // could come from one edge list and r2/r4 from another. Two shuffles
        // fix that. `in_by_dst` is sorted (dst, src, eid) and `out_by_src` is
        // sorted (src, dst, eid), so both are compared against the base
        // relation's (src, dst, eid) with the key/val columns swapped on the
        // in-side. The comparison is against the SORTED columns, which are the
        // ones the four bag lookups actually read; each view's own shuffle
        // already ties those to its input columns, so this is the same statement
        // and it needs one fewer column group.
        let perm_edge_out = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            PermAnyChip::configure(
                meta,
                q1,
                q2,
                vec![e_src, e_dst, e_eid],
                vec![
                    out_by_src.sorted_key,
                    out_by_src.sorted_val,
                    out_by_src.sorted_eid,
                ],
            )
        };
        let perm_edge_in = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            PermAnyChip::configure(
                meta,
                q1,
                q2,
                vec![e_src, e_dst, e_eid],
                vec![
                    in_by_dst.sorted_val,
                    in_by_dst.sorted_key,
                    in_by_dst.sorted_eid,
                ],
            )
        };

        let lane_u8 = meta.fixed_column();

        // The global occurrence ordering is shared machinery: both kinds of lane
        // are live on rows 0..lane_rows, so one step selector drives all of them.
        let q_oid_step = meta.selector();

        // ---------------- Bag2 lanes ----------------
        let q_t34_lookup = meta.complex_selector();
        let q_t34_flag = meta.selector();
        let q_cd = meta.selector();
        let q_t34_row = meta.selector();

        let bag2_lanes: Vec<Bag2LaneConfig<F>> = (0..num_bag2_lanes)
            .map(|_| {
                Self::configure_bag2_lane(
                    meta,
                    &in_by_dst,
                    &out_by_src,
                    lane_u8,
                    q_view_tbl,
                    q_t34_lookup,
                    q_t34_flag,
                    q_cd,
                    q_t34_row,
                    q_oid_step,
                )
            })
            .collect();

        // ---------------- Bag1 lanes ----------------
        // Shared selectors: every Bag1 lane is active on the same rows, so one
        // selector of each kind drives all c1 replicas and all c1*c2 probes.
        //
        // A NOTE ON WHY NO SELECTOR ROW MAY DEPEND ON THE WITNESS. Selectors are
        // fixed columns, so where they are enabled is part of the verifying key.
        // `g_sql4_obj` realizes condition (7) as a shuffle against a
        // [clean | residual | pad] column group and needs one selector per
        // section, i.e. the true clean and residual counts of each bag are in
        // its vk. That is legitimate there, where the Revealing-Join-Size regime
        // publishes the bag size anyway, and it is exactly what this circuit may
        // not do: those counts are what the DP release pays to hide. Every
        // selector below is enabled on a row range determined by `lane_rows`,
        // `c1` and `c2`, all of which are public.
        let sel = Bag1Selectors {
            q_t12_lookup: meta.complex_selector(),
            q_t12_key: meta.selector(),
            q_t12_row: meta.selector(),
            q_order12: meta.selector(),
            q_probe: meta.selector(),
            q_probe_complex: meta.complex_selector(),
            q_mu: meta.selector(),
            q_sum0: meta.selector(),
            q_sum: meta.selector(),
            q_oid_step,
        };

        let bag1_lanes: Vec<Bag1LaneConfig<F>> = (0..num_bag1_lanes)
            .map(|_| {
                Self::configure_bag1_lane(
                    meta,
                    &in_by_dst,
                    &out_by_src,
                    &bag2_lanes,
                    lane_u8,
                    q_view_tbl,
                    &sel,
                )
            })
            .collect();

        // ---------------- lane seams of the occurrence ordering -------------
        let oid_bnd = vec![
            Self::configure_oid_boundary(meta, lane_u8),
            Self::configure_oid_boundary(meta, lane_u8),
        ];

        // ---------------- cross-lane totals ----------------
        // Row l holds Bag1 lane l's two channel totals, copied in from that
        // lane's last `sum_all` / `sum_cln`; `tot_run` and `cln_run` add them
        // with degree-2 accumulators. A flat degree-(1+c1) sum gate would blow
        // past cs.degree() = 7 for c1 > 6.
        let lane_tot = meta.advice_column();
        let cln_tot = meta.advice_column();
        let tot_run = meta.advice_column();
        let cln_run = meta.advice_column();
        for c in [lane_tot, cln_tot, tot_run, cln_run] {
            meta.enable_equality(c);
        }
        let q_tot0 = meta.selector();
        let q_tot = meta.selector();

        meta.create_gate("lane totals first", |m| {
            let q = m.query_selector(q_tot0);
            vec![
                q.clone()
                    * (m.query_advice(tot_run, Rotation::cur())
                        - m.query_advice(lane_tot, Rotation::cur())),
                q * (m.query_advice(cln_run, Rotation::cur())
                    - m.query_advice(cln_tot, Rotation::cur())),
            ]
        });
        meta.create_gate("lane totals accu", |m| {
            let q = m.query_selector(q_tot);
            vec![
                q.clone()
                    * (m.query_advice(tot_run, Rotation::cur())
                        - (m.query_advice(tot_run, Rotation::prev())
                            + m.query_advice(lane_tot, Rotation::cur()))),
                q * (m.query_advice(cln_run, Rotation::cur())
                    - (m.query_advice(cln_run, Rotation::prev())
                        + m.query_advice(cln_tot, Rotation::cur()))),
            ]
        });

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

        // ---- condition (10), over the UNION of the Bag1 lanes ----
        // ONE equality, between the two GLOBAL totals at row c1-1. Accumulating
        // per lane and comparing per lane would let a prover move a hidden
        // tuple's contribution into a lane whose own equality still balances;
        // there is only one equality here and it sees every lane.
        meta.create_gate("cp: cardinality preservation", |m| {
            let q = m.query_selector(q_out);
            vec![
                q * (m.query_advice(tot_run, Rotation::cur())
                    - m.query_advice(cln_run, Rotation::cur())),
            ]
        });

        Gq4DpConfig {
            instance,
            e_eid,
            e_src,
            e_dst,
            in_by_dst,
            out_by_src,
            q_view_tbl,
            perm_edge_out,
            perm_edge_in,
            lane_u8,
            bag2_lanes,
            q_t34_lookup,
            q_t34_flag,
            q_cd,
            q_t34_row,
            bag1_lanes,
            q_t12_lookup: sel.q_t12_lookup,
            q_t12_key: sel.q_t12_key,
            q_t12_row: sel.q_t12_row,
            q_order12: sel.q_order12,
            q_mu: sel.q_mu,
            q_sum0: sel.q_sum0,
            q_sum: sel.q_sum,
            q_oid_step,
            oid_bnd,
            lane_tot,
            cln_tot,
            tot_run,
            cln_run,
            q_tot0,
            q_tot,
            out,
            q_out,
        }
    }
}

/// The selectors every Bag1 lane shares. Bundled only to keep
/// `configure_bag1_lane` under a readable argument count.
#[derive(Clone, Copy, Debug)]
struct Bag1Selectors {
    q_t12_lookup: Selector,
    q_t12_key: Selector,
    q_t12_row: Selector,
    q_order12: Selector,
    q_probe: Selector,
    q_probe_complex: Selector,
    q_mu: Selector,
    q_sum0: Selector,
    q_sum: Selector,
    q_oid_step: Selector,
}

/// What one Bag1 lane exports to the cross-lane stages.
struct Bag1LaneOut<F: Field + Ord> {
    /// first and last cell of the lane's occurrence-id column
    oid_ends: (AssignedCell<F, F>, AssignedCell<F, F>),
    /// last cell of the two channel prefix sums
    sum_all: AssignedCell<F, F>,
    sum_cln: AssignedCell<F, F>,
    /// the two lane totals
    total_all: u64,
    total_cln: u64,
}

impl<F: Field + Ord> Gq4DpChip<F> {
    /// One probe's witness. Same content as `card_preserve::assign_cp_join`,
    /// written out here so `Tamper::DropProbeVal` can zero the fetched input
    /// channel without disturbing anything else, and so the probe selectors can
    /// be the shared ones.
    fn assign_probe(
        region: &mut Region<'_, F>,
        j: &CpJoinConfig<F, NUM_BYTES>,
        keys: &[u64],
        stage: &CpStage,
        drop_s_all: bool,
    ) -> Result<Vec<(u64, u64)>, Error> {
        let lt_low = LtChip::<F, NUM_BYTES>::construct(j.lt_low);
        let lt_high = LtChip::<F, NUM_BYTES>::construct(j.lt_high);
        let mut fetched = Vec::with_capacity(keys.len());

        for (r, &k) in keys.iter().enumerate() {
            let (inx, low, high) = cp_gap_witness(&stage.keys, k, PAD_U64);
            let (s_all, s_cln) = if inx == 1 {
                stage.map.get(&k).copied().unwrap_or((0, 0))
            } else {
                (0, 0)
            };

            j.q_lookup.enable(region, r)?;
            j.q_lookup_complex.enable(region, r)?;

            region.assign_advice(|| "cp in_tbl", j.in_tbl, r, || Value::known(F::from(inx)))?;
            region.assign_advice(|| "cp low", j.low, r, || Value::known(F::from(low)))?;
            region.assign_advice(|| "cp high", j.high, r, || Value::known(F::from(high)))?;
            let written = if drop_s_all { 0 } else { s_all };
            region.assign_advice(|| "cp s_all", j.s_all, r, || Value::known(F::from(written)))?;
            region.assign_advice(|| "cp s_cln", j.s_cln, r, || Value::known(F::from(s_cln)))?;

            // the Lt chips are gated by (1 - in_tbl), but every cell is written:
            // an unconstrained row costs nothing and an unassigned cell would be
            // an error under the real prover
            lt_low.assign(
                region,
                r,
                Value::known(F::from(low)),
                Value::known(F::from(k)),
            )?;
            lt_high.assign(
                region,
                r,
                Value::known(F::from(k)),
                Value::known(F::from(high)),
            )?;

            fetched.push((s_all, s_cln));
        }

        Ok(fetched)
    }

    /// Links the two ends of consecutive lanes' occurrence-id columns into the
    /// seam stage, so the ordering is GLOBAL rather than per lane.
    fn assign_oid_boundary(
        region: &mut Region<'_, F>,
        bnd: &OidBoundaryConfig<F>,
        ends: &[(AssignedCell<F, F>, AssignedCell<F, F>)],
        vals: &[(u64, u64)],
    ) -> Result<(), Error> {
        let iz = IsZeroChip::construct(bnd.iz_pad.clone());
        let lt = LtChip::<F, NUM_BYTES>::construct(bnd.lt);
        let pad = F::from(PAD_U64);

        for l in 0..ends.len().saturating_sub(1) {
            let lo_val = vals[l].1;
            let hi_val = vals[l + 1].0;

            bnd.q.enable(region, l)?;
            let lo_cell = region.assign_advice(
                || "oid seam lo",
                bnd.lo,
                l,
                || Value::known(F::from(lo_val)),
            )?;
            let hi_cell = region.assign_advice(
                || "oid seam hi",
                bnd.hi,
                l,
                || Value::known(F::from(hi_val)),
            )?;
            region.constrain_equal(ends[l].1.cell(), lo_cell.cell())?;
            region.constrain_equal(ends[l + 1].0.cell(), hi_cell.cell())?;

            iz.assign(region, l, Value::known(F::from(lo_val) - pad))?;
            lt.assign(
                region,
                l,
                Value::known(F::from(lo_val)),
                Value::known(F::from(hi_val)),
            )?;
        }

        Ok(())
    }

    /// Assign one Bag2 lane's block of the padded Bag2 pipeline, its partition,
    /// its occurrence id and its child stage.
    ///
    /// `base` is the global index of this lane's row 0, so lane l covers the
    /// padded rows `[l*lane_rows, (l+1)*lane_rows)`. Rows past the true bag are
    /// padding: all-zero tuple, real 0, hence keep 0, key PAD and oid PAD. View
    /// row 0 is (0,0,0) with idx 0, so a padding row satisfies both membership
    /// lookups.
    ///
    /// Returns the first and last cell of this lane's occurrence-id column, for
    /// the seam stage.
    #[allow(clippy::too_many_arguments)]
    fn assign_bag2_lane(
        lane: &Bag2LaneConfig<F>,
        region: &mut Region<'_, F>,
        lane_rows: usize,
        base: usize,
        rows: &[BagRow],
        keep: &[u64],
        key: &[u64],
        cflag: &[u64],
        ceff: &[u64],
        oid: &[u64],
        stage: &CpStage,
    ) -> Result<(AssignedCell<F, F>, AssignedCell<F, F>), Error> {
        let lt_cd_chip = LtChip::<F, NUM_BYTES>::construct(lane.lt_cd);
        let lt_oid_chip = LtChip::<F, NUM_BYTES>::construct(lane.lt_oid);
        let iz_oid_chip = IsZeroChip::construct(lane.iz_oid_pad.clone());
        let pad = F::from(PAD_U64);

        let mut first: Option<AssignedCell<F, F>> = None;
        let mut last: Option<AssignedCell<F, F>> = None;

        for r in 0..lane_rows {
            let i = base + r;
            let row = rows[i];

            for (col, v) in [
                (lane.t34_c, row[0]),
                (lane.t34_d, row[1]),
                (lane.t34_a, row[2]),
                (lane.t34_i_r3, row[3]),
                (lane.t34_j_r4, row[4]),
                (lane.t34_r3_eid, row[5]),
                (lane.t34_r4_eid, row[6]),
                (lane.t34_real, row[7]),
            ] {
                region.assign_advice(|| "t34", col, r, || Value::known(F::from(v)))?;
            }

            // LT witness (the constraint is gated by real)
            lt_cd_chip.assign(
                region,
                r,
                Value::known(F::from(row[0])),
                Value::known(F::from(row[1])),
            )?;

            for (col, v) in [
                (lane.t34_keep, keep[i]),
                (lane.t34_key, key[i]),
                (lane.t34_cflag, cflag[i]),
                (lane.t34_ceff, ceff[i]),
            ] {
                region.assign_advice(|| "t34 partition", col, r, || Value::known(F::from(v)))?;
            }

            let cell = region.assign_advice(
                || "t34_oid",
                lane.t34_oid,
                r,
                || Value::known(F::from(oid[i])),
            )?;
            if r == 0 {
                first = Some(cell.clone());
            }
            last = Some(cell);

            iz_oid_chip.assign(region, r, Value::known(F::from(oid[i]) - pad))?;
            let nxt = if r + 1 < lane_rows {
                oid[i + 1]
            } else {
                PAD_U64
            };
            lt_oid_chip.assign(
                region,
                r,
                Value::known(F::from(oid[i])),
                Value::known(F::from(nxt)),
            )?;
        }

        // Lane-local child stage: the sorted view, the group boundaries, the two
        // running sums, the emitted per-key rows and the key-indexed table all
        // live inside this lane's own columns, with the lane's own first/last
        // guards. Nothing crosses a Bag2 lane boundary here.
        let cp_rows: Vec<[u64; 3]> = (0..lane_rows)
            .map(|r| [key[base + r], keep[base + r], ceff[base + r]])
            .collect();
        assign_cp_agg(region, &lane.cp, &cp_rows, stage)?;

        Ok((
            first.expect("lane_rows > 0 guarantees a first row"),
            last.expect("lane_rows > 0 guarantees a last row"),
        ))
    }

    /// Assign one Bag1 lane's block of the padded Bag1 pipeline, its partition,
    /// its occurrence id, its c2 probes and its two lane-local prefix sums.
    #[allow(clippy::too_many_arguments)]
    fn assign_bag1_lane(
        lane: &Bag1LaneConfig<F>,
        region: &mut Region<'_, F>,
        lane_rows: usize,
        base: usize,
        rows: &[BagRow],
        pred: &[u64],
        key: &[u64],
        cflag: &[u64],
        ceff: &[u64],
        oid: &[u64],
        stages: &[CpStage],
        tamper: Tamper,
    ) -> Result<Bag1LaneOut<F>, Error> {
        let lt_ab_chip = LtChip::<F, NUM_BYTES>::construct(lane.lt_ab);
        let lt_bc_chip = LtChip::<F, NUM_BYTES>::construct(lane.lt_bc);
        let lt_oid_chip = LtChip::<F, NUM_BYTES>::construct(lane.lt_oid);
        let iz_oid_chip = IsZeroChip::construct(lane.iz_oid_pad.clone());
        let iz_cln_chip = IsZeroChip::construct(lane.iz_cln.clone());
        let pad = F::from(PAD_U64);

        let mut first: Option<AssignedCell<F, F>> = None;
        let mut last: Option<AssignedCell<F, F>> = None;

        for r in 0..lane_rows {
            let i = base + r;
            let row = rows[i];

            for (col, v) in [
                (lane.t12_a, row[0]),
                (lane.t12_b, row[1]),
                (lane.t12_c, row[2]),
                (lane.t12_i_r1, row[3]),
                (lane.t12_j_r2, row[4]),
                (lane.t12_r1_eid, row[5]),
                (lane.t12_r2_eid, row[6]),
                (lane.t12_real, row[7]),
            ] {
                region.assign_advice(|| "t12", col, r, || Value::known(F::from(v)))?;
            }

            // order witnesses (the constraints are gated by real)
            lt_ab_chip.assign(
                region,
                r,
                Value::known(F::from(row[0])),
                Value::known(F::from(row[1])),
            )?;
            lt_bc_chip.assign(
                region,
                r,
                Value::known(F::from(row[1])),
                Value::known(F::from(row[2])),
            )?;

            for (col, v) in [
                (lane.t12_key, key[i]),
                (lane.t12_pred, pred[i]),
                (lane.t12_cflag, cflag[i]),
                (lane.t12_ceff, ceff[i]),
            ] {
                region.assign_advice(|| "t12 partition", col, r, || Value::known(F::from(v)))?;
            }

            let cell = region.assign_advice(
                || "t12_oid",
                lane.t12_oid,
                r,
                || Value::known(F::from(oid[i])),
            )?;
            if r == 0 {
                first = Some(cell.clone());
            }
            last = Some(cell);

            iz_oid_chip.assign(region, r, Value::known(F::from(oid[i]) - pad))?;
            let nxt = if r + 1 < lane_rows {
                oid[i + 1]
            } else {
                PAD_U64
            };
            lt_oid_chip.assign(
                region,
                r,
                Value::known(F::from(oid[i])),
                Value::known(F::from(nxt)),
            )?;
        }

        // ---- the c2 probes, then the two root multiplicities ----
        // Counts add across a partition, so this row's two multiplicities are
        // the sums of the c2 fetched pairs. Every fetch is certified by that
        // lane's membership lookup or by its gap bracket, which is what makes
        // the sums statements about the WHOLE of Bag2.
        let keys: Vec<u64> = (0..lane_rows).map(|r| key[base + r]).collect();
        let mut s_all = vec![0u64; lane_rows];
        let mut s_cln = vec![0u64; lane_rows];
        for (p_idx, (probe, stage)) in lane.probes.iter().zip(stages.iter()).enumerate() {
            let drop = tamper == Tamper::DropProbeVal && p_idx > 0;
            let fetched = Self::assign_probe(region, probe, &keys, stage, drop)?;
            for r in 0..lane_rows {
                s_all[r] += fetched[r].0;
                s_cln[r] += fetched[r].1;
            }
        }

        let mut acc_all: u64 = 0;
        let mut acc_cln: u64 = 0;
        let mut sum_all_cell: Option<AssignedCell<F, F>> = None;
        let mut sum_cln_cell: Option<AssignedCell<F, F>> = None;

        for r in 0..lane_rows {
            let i = base + r;
            let mu_all = pred[i] * s_all[r];
            let mu_cln = ceff[i] * s_cln[r];

            // Tamper::LaneMuAll perturbs the cell only; the prefix sum keeps its
            // honest value, so the root multiplicity gate and the sum gate both
            // fail.
            let mu_all_written = if tamper == Tamper::LaneMuAll && base == 0 && r == 0 {
                mu_all + 1
            } else {
                mu_all
            };
            region.assign_advice(
                || "mu_all",
                lane.mu_all,
                r,
                || Value::known(F::from(mu_all_written)),
            )?;
            region.assign_advice(|| "mu_cln", lane.mu_cln, r, || Value::known(F::from(mu_cln)))?;
            iz_cln_chip.assign(region, r, Value::known(F::from(s_cln[r])))?;

            // `wrapping_add` is a host-side accumulator only. The gates add in
            // the field, so a u64 wrap here would make the prefix sum disagree
            // with its gate and the circuit would REJECT rather than accept a
            // wrong COUNT: a completeness limit past 2^64 join occurrences, not
            // a soundness hole.
            acc_all = if r == 0 {
                mu_all
            } else {
                acc_all.wrapping_add(mu_all)
            };
            acc_cln = if r == 0 {
                mu_cln
            } else {
                acc_cln.wrapping_add(mu_cln)
            };
            sum_all_cell = Some(region.assign_advice(
                || "sum_all",
                lane.sum_all,
                r,
                || Value::known(F::from(acc_all)),
            )?);
            sum_cln_cell = Some(region.assign_advice(
                || "sum_cln",
                lane.sum_cln,
                r,
                || Value::known(F::from(acc_cln)),
            )?);
        }

        Ok(Bag1LaneOut {
            oid_ends: (
                first.expect("lane_rows > 0 guarantees a first row"),
                last.expect("lane_rows > 0 guarantees a last row"),
            ),
            sum_all: sum_all_cell.expect("lane_rows > 0"),
            sum_cln: sum_cln_cell.expect("lane_rows > 0"),
            total_all: acc_all,
            total_cln: acc_cln,
        })
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
        // every lane chip of either kind, the two child stages' comparators, the
        // probes and both occurrence orderings share `lane_u8`, so this is 3
        // load regions for any lane counts.
        in_view_chip.load(layouter)?;
        out_view_chip.load(layouter)?;
        debug_assert_eq!(cfg.bag2_lanes[0].lt_cd.u8, cfg.lane_u8);
        debug_assert_eq!(cfg.bag2_lanes[0].cp.lt_key_cur_next.u8, cfg.lane_u8);
        LtChip::<F, NUM_BYTES>::construct(cfg.bag2_lanes[0].lt_cd).load(layouter)?;

        layouter.assign_region(
            || "cycle4_ordered dp-lanes witness",
            |mut region| {
                // -------------------
                // Host-side derivation, shared verbatim with g_sql4_obj
                // -------------------
                let derived = gq4_derive(edges);
                let n_base = derived.in_rows.len();
                // `pack2` of two edge ids is injective, and every real
                // occurrence id stays strictly below PAD, only while both ids
                // fit a 32-bit lane.
                assert!(
                    (n_base as u64) < PACK_SHIFT,
                    "the occurrence id packs two edge ids into 32-bit lanes"
                );

                // -------- the base Edge relation and its two views --------
                // `out_rows[i]` is exactly the base tuple (src, dst, eid), and
                // `in_rows[i]` the same tuple with key and val swapped, so the
                // base columns are written straight from the derivation.
                for i in 0..n_base {
                    cfg.q_view_tbl.enable(&mut region, i)?;
                    cfg.perm_edge_out.q_perm1.enable(&mut region, i)?;
                    cfg.perm_edge_out.q_perm2.enable(&mut region, i)?;
                    cfg.perm_edge_in.q_perm1.enable(&mut region, i)?;
                    cfg.perm_edge_in.q_perm2.enable(&mut region, i)?;
                    let (src, dst, eid) = derived.out_rows[i];
                    region.assign_advice(|| "e_src", cfg.e_src, i, || Value::known(F::from(src)))?;
                    region.assign_advice(|| "e_dst", cfg.e_dst, i, || Value::known(F::from(dst)))?;
                    region.assign_advice(|| "e_eid", cfg.e_eid, i, || Value::known(F::from(eid)))?;
                }

                in_view_chip.assign(&mut region, n_base, &derived.in_rows)?;
                out_view_chip.assign(&mut region, n_base, &derived.out_rows)?;

                // -------------------
                // Capacities. Both pipelines are laned, so both can be the one
                // that does not fit.
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

                // -------------------
                // Both bags, in the GLOBAL occurrence-id order the distinctness
                // argument requires. Lane l is the block
                // [l*lane_rows, (l+1)*lane_rows) of these vectors.
                // -------------------
                let mut rows34 = bag2_rows(&derived.t34, cap2);
                let mut rows12 = bag1_rows(&derived.t12, cap1);

                // Test hook: copy the bag's FIRST kept occurrence over the FIRST
                // row of the LAST Bag2 lane. Both copies pass all four
                // membership lookups and both channels of (10) move by the same
                // amount, so the only thing that can see the duplicate is the
                // occurrence ordering. The copy goes to the first row of a later
                // lane on purpose: every lane's own block stays sorted, so a
                // PER-LANE ordering would accept this witness and only the seam
                // gate rejects it.
                if tamper == Tamper::DuplicateOccurrenceAcrossLanes {
                    assert!(
                        c2 >= 2,
                        "the cross-lane duplicate needs at least two Bag2 lanes"
                    );
                    let src = rows34[0];
                    assert_eq!(bag2_keep(&src), 1, "the duplicated row must be kept");
                    rows34[(c2 - 1) * lane_rows] = src;
                }

                let keep34: Vec<u64> = rows34.iter().map(bag2_keep).collect();
                let key34: Vec<u64> = (0..cap2)
                    .map(|i| {
                        if keep34[i] == 1 {
                            pack2(rows34[i][2], rows34[i][0])
                        } else {
                            PAD_U64
                        }
                    })
                    .collect();
                let oid34: Vec<u64> = (0..cap2)
                    .map(|i| bag_oid(&rows34[i], keep34[i]))
                    .collect();

                let pred12: Vec<u64> = rows12.iter().map(bag1_pred).collect();
                // The probe key is pack2(A,C) on every row; a padding row's
                // tuple is all zero, so its key is 0, the dummy row of every
                // child table.
                let key12: Vec<u64> = (0..cap1)
                    .map(|i| pack2(rows12[i][0], rows12[i][2]))
                    .collect();
                let oid12: Vec<u64> = (0..cap1)
                    .map(|i| bag_oid(&rows12[i], pred12[i]))
                    .collect();

                // -------------------
                // Condition (7): the prover's partition of each bag. The bag
                // tree has two nodes, so the semijoin reduction is exact: a Bag1
                // tuple is clean iff its predicate holds and its separator key
                // carries at least one kept Bag2 tuple, and a Bag2 tuple is
                // clean iff it is kept and its key is the key of a clean Bag1
                // tuple. The per-key counts are taken over the WHOLE of Bag2,
                // i.e. over all lanes, which is what makes the clean side a
                // property of the union.
                // -------------------
                let mut global34: BTreeMap<u64, u64> = BTreeMap::new();
                for i in 0..cap2 {
                    if keep34[i] == 1 {
                        *global34.entry(key34[i]).or_default() += 1;
                    }
                }

                let mut cln12: Vec<u64> = (0..cap1)
                    .map(|i| {
                        (pred12[i] == 1 && *global34.get(&key12[i]).unwrap_or(&0) > 0) as u64
                    })
                    .collect();
                if tamper == Tamper::HideOneCleanTuple {
                    let h = (0..cap1)
                        .find(|&i| cln12[i] == 1)
                        .expect("the slice has no clean Bag1 tuple to hide");
                    cln12[h] = 0;
                }
                if tamper == Tamper::MarkAllClean {
                    cln12 = pred12.clone();
                }

                let clean_keys12: HashSet<u64> = (0..cap1)
                    .filter(|&i| cln12[i] == 1)
                    .map(|i| key12[i])
                    .collect();
                let cln34: Vec<u64> = if tamper == Tamper::MarkAllClean {
                    keep34.clone()
                } else {
                    (0..cap2)
                        .map(|i| (keep34[i] == 1 && clean_keys12.contains(&key34[i])) as u64)
                        .collect()
                };

                let ceff12: Vec<u64> = (0..cap1).map(|i| pred12[i] * cln12[i]).collect();
                let ceff34: Vec<u64> = (0..cap2).map(|i| keep34[i] * cln34[i]).collect();

                // Cell-level test hooks, applied AFTER the per-row values are
                // derived, so exactly the intended constraint is the one that
                // breaks.
                if tamper == Tamper::Bag1PadRow {
                    rows12[cap1 - 1][0] = 7;
                }
                if tamper == Tamper::Bag2PadRow {
                    rows34[cap2 - 1][0] = 7;
                }

                // -------------------
                // Bag2: shared selectors, then the c2 lanes.
                // -------------------
                for r in 0..lane_rows {
                    cfg.q_t34_lookup.enable(&mut region, r)?;
                    cfg.q_t34_flag.enable(&mut region, r)?;
                    cfg.q_cd.enable(&mut region, r)?;
                    cfg.q_t34_row.enable(&mut region, r)?;
                }
                // the within-lane step of both bags' occurrence orderings
                for r in 0..lane_rows.saturating_sub(1) {
                    cfg.q_oid_step.enable(&mut region, r)?;
                }

                let mut stages: Vec<CpStage> = Vec::with_capacity(c2);
                for l in 0..c2 {
                    let base = l * lane_rows;
                    let rows: Vec<[u64; 3]> = (0..lane_rows)
                        .map(|r| [key34[base + r], keep34[base + r], ceff34[base + r]])
                        .collect();
                    stages.push(build_cp_stage(&rows, PAD_U64));
                }
                debug_assert_eq!(
                    stages
                        .iter()
                        .map(|s| s.map.values().map(|v| v.0).sum::<u64>())
                        .sum::<u64>(),
                    keep34.iter().sum::<u64>(),
                    "the lane-local child tables must partition the kept Bag2 multiset"
                );

                let mut oid34_ends: Vec<(AssignedCell<F, F>, AssignedCell<F, F>)> =
                    Vec::with_capacity(c2);
                for (l, lane) in cfg.bag2_lanes.iter().enumerate() {
                    oid34_ends.push(Self::assign_bag2_lane(
                        lane,
                        &mut region,
                        lane_rows,
                        l * lane_rows,
                        &rows34,
                        &keep34,
                        &key34,
                        &cln34,
                        &ceff34,
                        &oid34,
                        &stages[l],
                    )?);
                }
                let oid34_vals: Vec<(u64, u64)> = (0..c2)
                    .map(|l| (oid34[l * lane_rows], oid34[(l + 1) * lane_rows - 1]))
                    .collect();
                Self::assign_oid_boundary(
                    &mut region,
                    &cfg.oid_bnd[1],
                    &oid34_ends,
                    &oid34_vals,
                )?;

                // -------------------
                // Bag1: shared selectors, then the c1 lanes.
                // -------------------
                for r in 0..lane_rows {
                    cfg.q_t12_lookup.enable(&mut region, r)?;
                    cfg.q_t12_key.enable(&mut region, r)?;
                    cfg.q_t12_row.enable(&mut region, r)?;
                    cfg.q_order12.enable(&mut region, r)?;
                    cfg.q_mu.enable(&mut region, r)?;
                    if r == 0 {
                        cfg.q_sum0.enable(&mut region, r)?;
                    } else {
                        cfg.q_sum.enable(&mut region, r)?;
                    }
                }

                let mut lanes_out: Vec<Bag1LaneOut<F>> = Vec::with_capacity(c1);
                for (l, lane) in cfg.bag1_lanes.iter().enumerate() {
                    lanes_out.push(Self::assign_bag1_lane(
                        lane,
                        &mut region,
                        lane_rows,
                        l * lane_rows,
                        &rows12,
                        &pred12,
                        &key12,
                        &cln12,
                        &ceff12,
                        &oid12,
                        &stages,
                        tamper,
                    )?);
                }
                let oid12_ends: Vec<(AssignedCell<F, F>, AssignedCell<F, F>)> = lanes_out
                    .iter()
                    .map(|o| o.oid_ends.clone())
                    .collect();
                let oid12_vals: Vec<(u64, u64)> = (0..c1)
                    .map(|l| (oid12[l * lane_rows], oid12[(l + 1) * lane_rows - 1]))
                    .collect();
                Self::assign_oid_boundary(
                    &mut region,
                    &cfg.oid_bnd[0],
                    &oid12_ends,
                    &oid12_vals,
                )?;

                // -------------------
                // The cross-lane totals stage: c1 pairs of copy constraints, two
                // degree-2 accumulators, ONE equality between the global totals
                // (condition (10)) and `out` for the instance.
                // -------------------
                let mut acc_all: u64 = 0;
                let mut acc_cln: u64 = 0;
                for l in 0..c1 {
                    let tot_all = lanes_out[l].total_all;
                    let tot_cln = lanes_out[l].total_cln;
                    let written = if tamper == Tamper::LaneTotal && l == 0 {
                        tot_all + 1
                    } else {
                        tot_all
                    };
                    let tot_cell = region.assign_advice(
                        || "lane_tot",
                        cfg.lane_tot,
                        l,
                        || Value::known(F::from(written)),
                    )?;
                    region.constrain_equal(lanes_out[l].sum_all.cell(), tot_cell.cell())?;
                    let cln_cell = region.assign_advice(
                        || "cln_tot",
                        cfg.cln_tot,
                        l,
                        || Value::known(F::from(tot_cln)),
                    )?;
                    region.constrain_equal(lanes_out[l].sum_cln.cell(), cln_cell.cell())?;

                    if l == 0 {
                        cfg.q_tot0.enable(&mut region, l)?;
                        acc_all = tot_all;
                        acc_cln = tot_cln;
                    } else {
                        cfg.q_tot.enable(&mut region, l)?;
                        acc_all = acc_all.wrapping_add(tot_all);
                        acc_cln = acc_cln.wrapping_add(tot_cln);
                    }
                    region.assign_advice(
                        || "tot_run",
                        cfg.tot_run,
                        l,
                        || Value::known(F::from(acc_all)),
                    )?;
                    region.assign_advice(
                        || "cln_run",
                        cfg.cln_run,
                        l,
                        || Value::known(F::from(acc_cln)),
                    )?;
                }
                debug_assert!(
                    tamper != Tamper::None || acc_all == acc_cln,
                    "cardinality preservation: |R^c join| != |R join|"
                );

                let out_row = c1 - 1;
                cfg.q_out.enable(&mut region, out_row)?;
                let out_cell = region.assign_advice(
                    || "out",
                    cfg.out,
                    out_row,
                    || Value::known(F::from(acc_all)),
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

/// Geometry the two released capacities imply, or an explanation of why this
/// configuration cannot be built.
///
/// `Err` covers every way a row can be rejected BEFORE proving: the lane cap on
/// EITHER bag, and the structural fit of the tallest column group at the pinned
/// degree. All three are properties of the request, so a sweep reports them and
/// continues.
fn try_dp_lane_setup(
    dataset: &str,
    privacy: crate::bench_queries::Privacy,
) -> Result<Gq4DpSetup, String> {
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
    // BOTH bags are laned, so both can be the one that does not fit; name the
    // bag in the message so a skipped row says which release was too big.
    let c1 = try_lanes_for_capacity(n12, k).map_err(|e| format!("bag1: {}", e))?;
    let c2 = try_lanes_for_capacity(n34, k).map_err(|e| format!("bag2: {}", e))?;

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
        query: "gq4".to_string(),
        dataset: dataset.to_string(),
        k,
        lane_rows,
        lanes: vec![c1, c2],
        capacity: vec![n12, n34],
        true_size: vec![stats.bag1_size as usize, stats.bag2_size as usize],
        pads: vec![bag1_pad_extra, bag2_pad_extra],
    };
    Ok(Gq4DpSetup {
        edges,
        bag1_pad_extra,
        bag2_pad_extra,
        plan,
    })
}

/// [`try_dp_lane_setup`] for callers that treat an infeasible release as fatal.
fn dp_lane_setup(dataset: &str, privacy: crate::bench_queries::Privacy) -> Gq4DpSetup {
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
        bag1_pred, bag2_keep, bag2_rows, lane_rows_for, lanes_for, lanes_for_capacity,
        set_config_lanes, MyCircuit, Tamper, BASE_DEGREE, LANE_ROWS, MAX_LANES,
    };

    use crate::data::graph_data_processing::Edge;
    use halo2_proofs::dev::{MockProver, VerifyFailure};
    use halo2_proofs::plonk::Circuit;
    use halo2curves::pasta::Fp;

    use std::marker::PhantomData;
    use std::time::Instant;

    /// Twelve directed edges over eight nodes: 0->1->3->d->0 for each
    /// d in {4,5,6,7,8}, so there are exactly FIVE ordered 4-cycles, all
    /// sharing the separator pair (A,C) = (0,3).
    ///
    /// The shape is chosen so the message cannot live in one lane. Both bags
    /// materialize 11 rows, and the single separator key that carries the answer
    /// has multiplicity 5 in Bag2. With `LANE_ROWS_SMALL` = 4 those five rows
    /// cannot all land in one Bag2 lane, so `mock_three_lanes` only reaches the
    /// right answer if the Bag1 probes really do SUM the values retrieved from
    /// every Bag2 lane's child table. See
    /// `heavy_key_spans_multiple_bag2_lanes`, which pins that down.
    ///
    /// The shape also has DANGLING tuples on both ends of the bag tree edge --
    /// 5 predicate-passing Bag1 rows whose key occurs in no Bag2 tuple, and 6
    /// kept Bag2 rows whose key occurs in no predicate-passing Bag1 row -- which
    /// is what makes `mock_reject_mark_all_clean` non-vacuous.
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

    /// Runs the three-lane circuit under MockProver and returns the failures.
    fn three_lane_failures(tamper: Tamper) -> Vec<VerifyFailure> {
        let (circuit, c1, c2) = three_lane_circuit(tamper);
        set_config_lanes(c1, c2);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]).unwrap();
        prover
            .verify()
            .expect_err("the tampered witness must not verify")
    }

    /// Names the distinct constraints a negative direction failed on. A new
    /// constraint can make a tampered witness fail for a NEW reason, which would
    /// silently destroy the evidence that the check it was written for works, so
    /// each direction below asserts on the name and this prints the full set.
    fn report_failing_constraints(label: &str, failures: &[VerifyFailure]) {
        let mut names: Vec<String> = failures
            .iter()
            .map(|f| match f {
                VerifyFailure::Lookup { name, .. } => format!("lookup {}", name),
                VerifyFailure::Shuffle { name, .. } => format!("shuffle {}", name),
                VerifyFailure::ConstraintNotSatisfied { constraint, .. } => {
                    format!("gate {}", constraint)
                }
                other => format!("{:?}", other),
            })
            .collect();
        names.sort();
        names.dedup();
        for n in names.iter() {
            println!("[gq4 dp] {} failing constraint: {}", label, n);
        }
    }

    fn failed_on(failures: &[VerifyFailure], needle: &str) -> bool {
        failures
            .iter()
            .any(|f| format!("{:?}", f).contains(needle))
    }

    /// The separator key that carries the whole answer has multiplicity 5 in
    /// Bag2 while a lane only holds 4 rows, so its count is necessarily split
    /// over at least two Bag2 lanes. This is what makes `mock_three_lanes` a
    /// real test of the cross-lane message sum rather than of a single table.
    ///
    /// The lane a Bag2 row falls into is decided by the GLOBAL occurrence-id
    /// order the distinctness argument requires, so this test asks
    /// `bag2_rows` for the layout instead of assuming the derivation order.
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

        let capacity = 3 * LANE_ROWS_SMALL;
        let rows = bag2_rows(&derived.t34, capacity);
        let mut per_lane: BTreeMap<usize, u64> = BTreeMap::new();
        for (i, r) in rows.iter().enumerate() {
            if bag2_keep(r) == 1 && crate::graph_sql::g_sql4_obj::pack2(r[2], r[0]) == heavy {
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

    /// The synthetic slice must really have dangling tuples on BOTH ends of the
    /// bag tree edge, or `mock_reject_mark_all_clean` would be vacuous: with no
    /// dangling tuple the all-clean partition IS the reduced instance and
    /// condition (9) would rightly accept it.
    #[test]
    fn synthetic_slice_has_dangling_tuples() {
        use std::collections::HashSet;

        let edges = synthetic_edges();
        let derived = crate::graph_sql::g_sql4_obj::gq4_derive(&edges);
        let pack2 = crate::graph_sql::g_sql4_obj::pack2;

        let kept34_keys: HashSet<u64> = derived
            .t34
            .iter()
            .filter(|&&(c, d, ..)| c < d)
            .map(|&(c, _d, a, ..)| pack2(a, c))
            .collect();
        let pred12_keys: HashSet<u64> = derived
            .t12
            .iter()
            .filter(|&&(a, b, c, ..)| a < b && b < c)
            .map(|&(a, _b, c, ..)| pack2(a, c))
            .collect();
        let dangling12 = derived
            .t12
            .iter()
            .filter(|&&(a, b, c, ..)| a < b && b < c && !kept34_keys.contains(&pack2(a, c)))
            .count();
        let dangling34 = derived
            .t34
            .iter()
            .filter(|&&(c, d, a, ..)| c < d && !pred12_keys.contains(&pack2(a, c)))
            .count();
        println!(
            "[gq4 dp] dangling: t12={} t34={}",
            dangling12, dangling34
        );
        assert!(
            dangling12 > 0,
            "no dangling Bag1 tuple: the all-clean direction would be vacuous"
        );
    }

    /// Every predicate-passing bag row must have a DISTINCT occurrence id, or
    /// the honest witness could not satisfy the global ordering in the first
    /// place. `pack2(r1_eid, r2_eid)` is a function of the occurrence, so this
    /// is a property of the derivation, checked here so a failure points at the
    /// derivation rather than at a gate.
    #[test]
    fn occurrence_ids_are_distinct() {
        use std::collections::HashSet;

        let edges = synthetic_edges();
        let derived = crate::graph_sql::g_sql4_obj::gq4_derive(&edges);

        let ids12: Vec<u64> = derived
            .t12
            .iter()
            .map(|&(a, b, c, _i, _j, e1, e2)| {
                let r = [a, b, c, 0, 0, e1, e2, 1];
                (bag1_pred(&r), crate::graph_sql::g_sql4_obj::pack2(e1, e2))
            })
            .filter(|&(p, _)| p == 1)
            .map(|(_, id)| id)
            .collect();
        assert_eq!(
            ids12.len(),
            ids12.iter().copied().collect::<HashSet<u64>>().len(),
            "two Bag1 occurrences share an id"
        );

        let ids34: Vec<u64> = derived
            .t34
            .iter()
            .filter(|&&(c, d, ..)| c < d)
            .map(|&(_c, _d, _a, _i, _j, e3, e4)| crate::graph_sql::g_sql4_obj::pack2(e3, e4))
            .collect();
        assert_eq!(
            ids34.len(),
            ids34.iter().copied().collect::<HashSet<u64>>().len(),
            "two Bag2 occurrences share an id"
        );
    }

    /// Cost and degree probe, permanent. `cs.degree()` drives the FFT size of
    /// every polynomial in the proof, so a rise here costs far more than any
    /// individual gate saves. The bound is the one `g_sql4_obj` already pays: a
    /// membership lookup with a degree-3 input and a degree-2 table side, i.e.
    /// 2 + 3 + 2 = 7.
    #[test]
    fn test_max_gate_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        for (c1, c2) in [(1usize, 1usize), (2, 1), (1, 2), (4, 3)] {
            set_config_lanes(c1, c2);
            let mut cs = ConstraintSystem::<Fp>::default();
            let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
            println!("c1={} c2={} cs.degree() = {}", c1, c2, cs.degree());
            println!(
                "  advice={} fixed={} instance={} selectors={} gates={} lookups={} \
                 shuffles={} perm_cols={}",
                cs.num_advice_columns(),
                cs.num_fixed_columns(),
                cs.num_instance_columns(),
                cs.num_selectors(),
                cs.gates().len(),
                cs.lookups().len(),
                cs.shuffles().len(),
                cs.permutation().get_columns().len(),
            );
            assert!(
                cs.degree() <= 7,
                "c1={} c2={}: the maximum gate degree rose to {}, which doubles every FFT",
                c1,
                c2,
                cs.degree()
            );
        }
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
            (2, "selectors"),
            (4, "shuffles"),
            (5, "permutation columns"),
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
        // Exactly three cells leave a Bag1 lane by copy: the two ends of its
        // occurrence-id column, for the lane seam of the global ordering, and
        // the last cell of each of the two channel prefix sums, for the totals
        // stage where condition (10) is compared. A Bag2 lane exports nothing by
        // copy except the two ends of ITS occurrence-id column; the rest of the
        // permutation cost a Bag2 lane pays is internal to its child stage.
        assert_eq!(
            d1[5], 3,
            "a Bag1 lane must add exactly three permutation columns \
             (t12_oid, sum_all, sum_cln)"
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
    /// row, and `lane_rows` = 1, where `q_sum` never fires, every lane's prefix
    /// sum is just its one row and EVERY step of the occurrence ordering is a
    /// lane seam.
    #[test]
    fn mock_lane_geometries() {
        // (lane_rows, bag1_pad_extra, bag2_pad_extra, expected c1, expected c2)
        let cases: [(usize, usize, usize, usize, usize); 6] = [
            (11, 0, 0, 1, 1),  // exactly one full lane each, no pad anywhere
            (6, 0, 0, 2, 2),   // real rows straddle the lane-0/lane-1 boundary
            (3, 1, 1, 4, 4),   // 12 released rows over 12: one rounding-up pad
            (5, 3, 0, 3, 3),   // 14 released rows over 15 vs 11 over 15
            (2, 0, 0, 6, 6),   // the last lane holds a single real row
            (1, 0, 0, 11, 11), // one row per lane: every step is a seam
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
    /// A lane occupies `lane_rows + 1` region rows: the child stage writes a
    /// sentinel row at `lane_rows` for its `Rotation::next()` comparisons.
    #[test]
    fn structure_fits_base_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        for (k, c1, c2) in [(18u32, 4usize, 3usize), (22, 1, 1), (18, 1, 1)] {
            set_config_lanes(c1, c2);
            let mut cs = ConstraintSystem::<Fp>::default();
            let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);

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
    /// lanes, each Bag2 lane builds its own child table, each Bag1 lane probes
    /// all three of them, the occurrence ordering runs across all six lane
    /// seams, and the totals stage adds the three pairs of Bag1 lane totals and
    /// compares the two global sums once.
    #[test]
    fn mock_three_lanes() {
        let (circuit, c1, c2) = three_lane_circuit(Tamper::None);
        set_config_lanes(c1, c2);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]).unwrap();
        prover.assert_satisfied();
    }

    /// Corrupting one Bag1 lane row's input-channel multiplicity must break the
    /// root multiplicity gate and that lane's prefix sum.
    #[test]
    fn mock_reject_tampered_lane_mu_all() {
        let failures = three_lane_failures(Tamper::LaneMuAll);
        report_failing_constraints("lane mu_all", &failures);
        assert!(
            failed_on(&failures, "cp: root multiplicities")
                || failed_on(&failures, "cp: lane sums"),
            "expected the root or prefix-sum gate to reject: {:?}",
            failures
        );
    }

    /// Corrupting one Bag1 lane total must break the copy constraint out of the
    /// lane and the totals accumulator.
    #[test]
    fn mock_reject_tampered_lane_total() {
        let failures = three_lane_failures(Tamper::LaneTotal);
        report_failing_constraints("lane total", &failures);
    }

    /// Breaking the Bag1 pad convention on the last rounding-up row must break
    /// the "bag1 real + key" gate and the r1 membership lookup.
    #[test]
    fn mock_reject_tampered_bag1_pad_row() {
        let failures = three_lane_failures(Tamper::Bag1PadRow);
        report_failing_constraints("bag1 pad row", &failures);
        assert!(
            failed_on(&failures, "bag1 real + key") || failed_on(&failures, "bag1 r1"),
            "expected the key gate or the r1 lookup to reject: {:?}",
            failures
        );
    }

    /// Breaking the Bag2 pad convention on the last rounding-up row must break
    /// the r3 membership lookup into the shared indexed view.
    #[test]
    fn mock_reject_tampered_bag2_pad_row() {
        let failures = three_lane_failures(Tamper::Bag2PadRow);
        report_failing_constraints("bag2 pad row", &failures);
        assert!(
            failed_on(&failures, "bag2 r3"),
            "expected the r3 lookup to reject: {:?}",
            failures
        );
    }

    /// Zeroing the value retrieved from every Bag2 lane past the first, while
    /// leaving the root multiplicity honest, must be rejected. This is the
    /// permanent negative test for the cross-lane message sum: it fails only
    /// because the root gate adds all `c2` probe values and each probe's
    /// membership lookup is really enforced.
    #[test]
    fn mock_reject_dropped_probe_val() {
        let failures = three_lane_failures(Tamper::DropProbeVal);
        report_failing_constraints("dropped probe val", &failures);
        assert!(
            failed_on(&failures, "cp: root multiplicities")
                || failed_on(&failures, "cp: key and its two sums"),
            "expected the root gate or a probe membership lookup to reject: {:?}",
            failures
        );
    }

    /// CONDITION (10), and the reason its equality is over the GLOBAL totals.
    /// One joinable Bag1 tuple is moved to the residual side and Bag2 is
    /// re-reduced around it, so condition (7) still holds, condition (9) still
    /// holds (the hidden tuple is no longer clean, so it asks nothing of the
    /// Bag2 side), every membership lookup still passes and the occurrence
    /// ordering is untouched. Only the Cardinality Preservation Check can see
    /// it, and it sees it whichever lane the tuple lives in, because the
    /// equality compares the two totals accumulated over ALL Bag1 lanes.
    #[test]
    fn mock_reject_hidden_clean_tuple() {
        let failures = three_lane_failures(Tamper::HideOneCleanTuple);
        report_failing_constraints("hidden clean tuple", &failures);
        assert!(
            failed_on(&failures, "cardinality preservation"),
            "the circuit rejected, but not through the Cardinality Preservation \
             Check: {:?}",
            failures
        );
    }

    /// CONDITION (9), the direction this circuit enforces. Every
    /// predicate-passing tuple of both bags is declared clean and the residual
    /// side is left empty, so both Conservation and both channels of condition
    /// (10) agree on every row and only Pairwise Consistency can see the five
    /// dangling Bag1 tuples of the slice. It must reject through the clean-key
    /// consistency constraint, which reads the clean multiplicity summed over
    /// ALL Bag2 lanes.
    #[test]
    fn mock_reject_mark_all_clean() {
        let failures = three_lane_failures(Tamper::MarkAllClean);
        report_failing_constraints("mark all clean", &failures);
        assert!(
            failed_on(&failures, "clean-key consistency"),
            "the circuit rejected, but not through condition (9): {:?}",
            failures
        );
    }

    /// OCCURRENCE DISTINCTNESS ACROSS LANES. The bag's first kept occurrence is
    /// copied over the FIRST row of the LAST Bag2 lane. Both copies pass all
    /// four membership lookups, both are internally consistent, both channels of
    /// condition (10) move by the same amount, and -- the point of the test --
    /// every lane's own occurrence block is still sorted, so a PER-LANE ordering
    /// would accept this witness and the COUNT would be inflated. It must reject
    /// through the LANE SEAM of the global ordering.
    #[test]
    fn mock_reject_duplicate_occurrence_across_lanes() {
        let failures = three_lane_failures(Tamper::DuplicateOccurrenceAcrossLanes);
        report_failing_constraints("cross-lane duplicate", &failures);
        assert!(
            failed_on(&failures, "carries across the lane seam"),
            "the circuit rejected, but not through the lane seam of the occurrence \
             ordering: {:?}",
            failures
        );
    }

    /// c1 != c2. The two released capacities are independent, and the real
    /// lastfm cell is c1 = 4, c2 = 3, so the symmetric case the other tests use
    /// must not be the only one exercised: `c1 * c2` probe replicas, `c2` child
    /// tables per Bag1 lane and `c1` rows in the totals stage all have to line
    /// up.
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
    /// row at all. Its child stage then sees an all-PAD input, emits nothing,
    /// and its table degenerates to the single (0,0,0) dummy row followed by PAD.
    /// Every Bag1 row must still be able to prove absence in it, which is the
    /// one gap bracket (0, PAD) that lane offers. The lane's occurrence-id
    /// column is all PAD, which is what the "once PAD, always PAD" half of the
    /// ordering is for.
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

    /// c1 = c2 = 1 degenerates to a single-group layout: this circuit and
    /// `g_sql4_obj` must accept the same input and expose the same COUNT(*).
    /// They are no longer byte-identical -- this file realizes condition (7) as
    /// an indicator on the bag rows rather than as a shuffled partition group,
    /// because the section selectors that form needs would put the true clean
    /// and residual counts in the verifying key -- so this checks the answer and
    /// mutual satisfiability, not the shape.
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
