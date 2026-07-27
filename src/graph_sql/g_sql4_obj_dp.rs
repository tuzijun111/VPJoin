//! Multi-lane DP-padding variant of the GQ4 4-cycle circuit (`g_sql4_obj`).
//!
//! WHY THIS FILE EXISTS
//! --------------------
//! GQ4 materializes its bag at a capacity that, under differential privacy, is a
//! NOISY RELEASE rather than the true bag size. The released capacity is public,
//! but it is not a power of two, and a single column group must be padded up to
//! the next power of two. On lastfm the true bag holds 232,943 rows and
//! Revealing-Join-Size proves at k = 18; the DP release at
//! (eps,delta) = (0.1,1e-5) is 512,521 rows, which alone would push the circuit
//! to k = 19. At eps = 0.01 the release is ~32x larger and the degree would
//! reach k = 24. Every step doubles the prover's work even though the capacity
//! grew by far less.
//!
//! (Those are the releases the default `VPJOIN_DP_SEED` produces now that
//! `bench_queries::graph_pads` spends ONE capacity release on GQ4 instead of
//! two: the two bags are one relation, hence one private cardinality, so the
//! budget splits 2 ways rather than 3 and the noise shrinks. The two-bag
//! version of this file quoted 788,220 and 621,070 from the 3-way split.)
//!
//! This circuit keeps the degree PINNED at the Revealing-Join-Size value and
//! hosts the released capacity in
//!
//!     c = ceil(capacity / lane_rows)
//!
//! parallel column-group LANES, so the cost grows with the released capacity
//! instead of doubling at each power of two. Same trick, same conventions, as
//! `sql::q5_obj_dp` does for TPC-H Q5 and `g_sql3_obj_dp` does for the GQ3
//! triangle.
//!
//! WHAT A LANE IS
//! --------------
//! GQ4's separator (A,C) splits the 4-cycle into the path A->B->C and the path
//! C->D->A, and those are THE SAME RELATION read with different column roles:
//! with `W = {(x,y,z) : x->y and y->z are edges, x<y}`, Bag 1 is W as (A,B,C)
//! and Bag 2 is W as (C,D,A). The role-1 reading adds [y<z] on top of W's own
//! x<y and the role-2 reading adds nothing, which together is exactly
//! A<B<C<D. So there is ONE pad-driven pipeline and ONE kind of lane, and
//! `g_sql4_obj` materializes W once for the same reason.
//!
//! A lane is a full structural replica of everything that pipeline needs:
//!
//!   * the eight W columns and the two comparators [x<y] and [y<z];
//!   * the ROLE-2 group (keep bit, separator key `pack2(z,x)`, the prover's
//!     clean indicator and the folded `ceff2`), and a complete per-key CHILD
//!     STAGE of the Cardinality Preservation Check, which turns the lane's block
//!     of W rows into a lane-local key-indexed table of TWO per-key sums;
//!   * the ROLE-1 group (predicate bit, separator key `pack2(x,z)`, its own
//!     clean indicator, the folded `ceff1` and `ck1 = ceff1 * key1`, the table
//!     side of the mirror half of (9)), one complete PARENT PROBE per lane, c
//!     boolean `host` bits naming the lane that holds this row's clean role-2
//!     key, and a lane-local prefix sum of each of the two channels;
//!   * the occurrence id of the row.
//!
//! The aggregate key is `pack2(z,x)`, REVERSED against the role-1 probe key
//! `pack2(x,z)`: the same two cells of the same row packed in opposite orders.
//! That reversal is the whole mechanism, so the two keys are two columns and
//! cannot merge.
//!
//! Lanes are identical by construction: every one carries the same columns, the
//! same gates, the same lookups and the same child stage. Every lane but the
//! last is assigned to the full `lane_rows` rows, and the last one stops at the
//! layout capacity (see [`lane_live_rows`]): the caller's released capacity
//! when [`MyCircuit::released_capacity`] is `Some`, the full lane span when it
//! is `None`. Either way the assigned structure is a function of PUBLIC numbers
//! only. It never depends on the true bag size. A cheaper "overflow lane" that
//! only carries padding would leak exactly the quantity DP is paying to hide,
//! so there is no such thing here: the last lane is shorter, not weaker, and
//! the rows it drops are the ones the LANE COUNT rounded up to, never the DP
//! pad.
//!
//! WHY LANE-LOCAL AGGREGATION IS SOUND
//! -----------------------------------
//! The message a role-2 row sends is `sigma(pack2(A,C)) = COUNT(paths C->D->A
//! with C<D)`, a pure COUNT. Counts ADD across a partition, so splitting W
//! block-wise over c lanes and grouping each block independently is exactly
//! correct: lane l's table holds `count_l(key)`, and
//!
//!     sum_l count_l(key) = count(key).
//!
//! A role-1 row therefore probes all c tables and sums the c retrieved values.
//! Keys missing from a given lane's table are proven missing by that lane's gap
//! bracket and contribute 0, exactly as in the single-group circuit. No
//! ordering, no sentinel and no carry crosses a lane boundary on the child side:
//! the aggregation itself needs ZERO stitching.
//!
//! ONE-PASS OBJ ACROSS LANES
//! -------------------------
//! `g_sql4_obj` enforces three conditions of the One-Pass OBJ over a
//! prover-supplied partition of each ROLE into a clean part and a residual part.
//! The two partitions stay separate even though there is one tuple group: a row
//! can be role-2 clean and role-1 dangling, so the two clean SETS differ, and
//! merging them would make (9) compare a set with itself and collapse the two
//! channels of (10) into one.
//!
//! A condition enforced PER LANE in a way that lets a prover split a violation
//! across two lanes is not enforced at all, so each one is realized here over
//! the UNION of the lanes:
//!
//!   (7) Conservation, `R == R^c U R^r`.
//!       Each partition is carried as a boolean indicator column on the W rows
//!       themselves (`cflag1` / `cflag2`), and the bit both channels of (10)
//!       actually read is the folded `ceff = predicate * cflag`. With the
//!       partition represented as an indicator ON `R`, `R^c = {r : ceff(r)=1}`
//!       and `R^r = {r : predicate(r)=1, ceff(r)=0}` are a partition of the
//!       predicate-passing rows BY CONSTRUCTION, per row and therefore over the
//!       union of the lanes: there is no second column group that could
//!       disagree with the bag, and no cross-lane split is expressible.
//!       `g_sql4_obj` instead lays each partition out as a separate
//!       [clean | residual | pad] column group and shuffles it against the bag.
//!       That form is equivalent, but it needs one selector per SECTION, whose
//!       enabled rows are the true clean and residual counts, i.e. it bakes the
//!       sensitive quantity this file exists to hide into the verifying key.
//!       See the note on that at the selector block of `configure`.
//!   (9) Pairwise Consistency, `pi_K(R_1^c) == pi_K(R_2^c)` on the one bag tree
//!       edge, `K = pack2(A,C)`. BOTH directions are enforced, by two different
//!       mechanisms, because both sides are laned and a lookup's TABLE is a
//!       single set of column expressions, so it cannot be the union of c lane
//!       columns.
//!         * `pi_K(R_1^c) subset of pi_K(R_2^c)` rides on the clean channel.
//!           Every role-1 row fetches `s_cln` from EVERY lane, each fetch
//!           certified by that lane's membership lookup or by its gap bracket,
//!           so `sum_l s_cln_l(key)` is the number of clean role-2 tuples with
//!           that key over ALL of W. The gate `cp: root multiplicities and
//!           clean-key consistency` then requires a clean role-1 row to have a
//!           nonzero total, which is the containment over the union.
//!         * the MIRROR, `pi_K(R_2^c) subset of pi_K(R_1^c)`, is resolved by
//!           NAMING the hosting lane, the trick `g_sql3_obj_dp` uses for its own
//!           union direction. Each lane carries c boolean `host` bits per row,
//!           gated to sum to `ceff2`, so a clean role-2 row names exactly one
//!           lane and a non-clean row names none; one width-1 lookup per
//!           (row lane, named lane) pair then checks the named claim against
//!           that lane's clean role-1 keys. Cost `2c + c^2` advice and `c^2`
//!           lookups, i.e. P += 2c + 4c^2, against a second per-lane child stage
//!           plus a second c^2 probe grid, which is the only other way to
//!           resolve it and would cost roughly two thirds of the whole circuit
//!           again.
//!  (10) Cardinality Preservation, `|R^c join| == |R join|`.
//!       This is the one that must not be per lane. Each lane accumulates its
//!       own `sum_all` and `sum_cln` over its own rows, both from
//!       multiplicities that already sum over all c child stages; the c pairs of
//!       lane totals are then copied into the totals stage, added with two
//!       degree-2 accumulators, and compared by a SINGLE equality at row c-1.
//!       A prover moving a hidden tuple's contribution into another lane changes
//!       one global total and nothing rebalances it.
//!
//! `sum_all` is also the ANSWER. `g_sql4_obj` propagates the per-key COUNT twice
//! (once through an `AggSumByKey` for the aggregate and once through the check's
//! own child table) and needs a gate to equate them; here the aggregate IS the
//! input channel of the check, `mu_all = pred1 * sum_l s_all_l`, so the
//! certified cardinality is the exposed COUNT by construction and there is
//! nothing to equate. That also makes this circuit CHEAPER than a laned
//! `AggSumByKey` plus a laned check would be.
//!
//! OCCURRENCE DISTINCTNESS ACROSS LANES
//! ------------------------------------
//! The two r_in/r_out membership lookups say only that a W row is SOME valid
//! wedge occurrence. Every argument downstream is blind to a DUPLICATE:
//! conservation is per row, (9) compares key sets, and (10) is a difference of
//! two channels, so copying a clean role-2 tuple into a pad slot inflates
//! `s_all` and `s_cln` equally and inflates the COUNT with nothing rejecting.
//! In a laned layout the copy does not even have to stay in the same lane, so a
//! per-lane distinctness argument would miss it.
//!
//! Each W row therefore carries `oid = pack2(the two source edge ids)` when its
//! ROLE-2 keep bit holds and PAD when it does not, and the rows are required to
//! be laid out in GLOBAL oid order: strictly increasing while below PAD, then
//! PAD for ever. Two uniform gates per lane say that over adjacent rows inside a
//! lane, and a `c-1` row boundary stage says it over each lane seam, with the
//! seam cells brought in by copy constraints (the only cells besides the two
//! lane sums that ever leave a lane). Concatenating the lanes in lane order, the
//! whole sequence is strictly increasing until PAD, so all non-PAD oids are
//! pairwise distinct ACROSS lanes, and the honest edge ids fit 32-bit lanes so
//! a kept row's oid is always below PAD.
//!
//! ONE argument does both roles, over the ROLE-2 keep bit, which is W's own x<y
//! filter. Role 1's rows are exactly the role-2 rows on which a genuine Lt
//! certifies [y<z], decided per row by the `role 1 pred = keep * [y<z]` gate, so
//! distinctness of the role-2 occurrences already gives it for the role-1
//! subset. Do not weaken that gate without restoring a second ordering.
//!
//! Note what this does and does not give. It rules out DUPLICATION, hence any
//! over-count. It cannot rule out OMISSION: `g_sql4_obj` pins the number of
//! predicate-passing rows through its partition sections, which is sound there
//! only because the Revealing-Join-Size regime publishes the true bag size. Here
//! the true bag size is precisely what the DP release hides, so no in-circuit
//! row count may depend on it. Bag completeness therefore rests on NOTHING, in
//! or out of circuit: the release publishes an upper-bound capacity and
//! certifies nothing about the prover filling it. What this circuit proves is
//! COUNT = |join(S)| for a prover-chosen S contained in W, i.e. COUNT is at most
//! the true count, where `g_sql4_obj` proves equality because its partition
//! sections put the kept-row count in the vk. Any claim that the padding scheme
//! closes this is wrong.
//!
//! WHAT IS NOT ENFORCED
//! --------------------
//! Condition (8) of the One-Pass OBJ (Non-Membership) is deliberately not
//! enforced, as in `g_sql4_obj`; with the partition carried as an indicator it
//! is vacuous anyway.
//!
//! COST NOTE (read before scaling this up)
//! ---------------------------------------
//! Probing c lane-local tables from c lanes costs c * c probe replicas, so the
//! circuit is QUADRATIC in the single lane count, not linear. That is inherent
//! to resolving a PRIVATE key against a table of private size: with c separate
//! tables a lookup argument can only address one of them at a time. Fitted to
//! the measured table below,
//!
//!   advice   = 56 + 90c + 24c^2
//!   lookups  = 24 + 42c + 19c^2
//!   shuffles = 4 + 2c
//!
//! so with `P = advice + 3*lookups + shuffles` as a rough proxy for the
//! prover's column work, P = 431 at c = 1 and 892 at c = 2. The `c^2` terms are
//! 24 advice and 19 lookups per (row lane, named lane) pair: 23 advice and 18
//! lookups for the probe replica, 1 advice for the `host` bit and 1 lookup for
//! the mirror direction of (9).
//!
//! The quadratic term CANNOT be removed by sharing. A `cp_agg` stage owns ONE
//! input column pair, bounded by the 2^k rows of the domain, while W spans
//! c * lane_rows rows, so each lane must keep its own partial table and every
//! row needs all c partials. Reducing the c partials to one global table first
//! would break even only at c = 2 and would release a NEW private statistic, the
//! number of distinct separator keys, which is exactly what a lane layout exists
//! to avoid. Getting a genuinely linear GQ4 needs a different join strategy,
//! namely replacing the message table by an in-circuit sort-merge of the two
//! readings, which is a different circuit rather than a laned version of this
//! one.
//!
//! At the production cells this is comfortable: eps = 0.1 gives c = 2 on lastfm
//! (P 892 against the 543 an unlaned k+1 would pay, for half the domain) and
//! c = 1 on facebook and wiki. At eps = 0.01 lastfm needs c = 63, i.e. 3,969
//! probe replicas and about 101,000 advice columns, which is not a circuit
//! anyone should build: that cell wants the sort-merge, not lanes.
//!
//! WHAT STAYS SINGLE COPY
//! ----------------------
//! Everything whose height is fixed by the PUBLIC edge count: the base Edge
//! relation (`e_src`, `e_dst`, `e_eid`) and the two indexed views of it
//! (`in_by_dst`, `out_by_src`), which are the table side of every r_in/r_out
//! membership lookup. Their height tracks |E|, which is public in every privacy
//! regime. Two shuffles tie both views to the base relation, so the lane
//! lookups read ONE edge relation instead of two independently invented ones,
//! and `q_view_tbl` keeps each view's sentinel row out of both tables.
//!
//! WHERE THE STITCHES ARE
//! ----------------------
//! Three, all of them small and all of them cross-lane BY DESIGN:
//!   * the two channels of (10): c pairs of copy constraints into the totals
//!     stage, two degree-2 accumulators over c rows, one equality at row c-1,
//!     and `out` at row c-1 into the instance;
//!   * the global occurrence ordering: 2*(c-1) copy constraints into the ONE
//!     boundary stage;
//!   * nothing else. A lane's child stage still exports nothing by copy: its two
//!     per-key sums reach the other lanes through lookups, and the mirror of (9)
//!     travels through a lookup as well.
//!
//! MEASURED STRUCTURE (see `tests::test_max_gate_degree`)
//! -----------------------------------------------------
//!   c=1 : advice 170  fixed 3  sel 44  lookups  85  shuf  6  perm  47  P  431
//!   c=2 : advice 332  fixed 3  sel 61  lookups 184  shuf  8  perm  69  P  892
//!   c=3 : advice 542  fixed 3  sel 78  lookups 321  shuf 10  perm  91  P 1515
//!   c=4 : advice 800  fixed 3  sel 95  lookups 496  shuf 12  perm 113  P 2300
//! so one more lane costs 90 advice / 17 selectors / 2 shuffles / 22 permutation
//! columns / 42 lookup arguments plus its share of the quadratic term, and one
//! more probe pair costs 24 advice and 19 lookups and NOTHING else: no fixed
//! column, no selector, no shuffle and no permutation column. Of a lane's 22
//! permutation columns only three carry a cell out of the lane (`oid`,
//! `sum_all`, `sum_cln`); the rest are internal to its child stage. The
//! selectors that do grow are the child stage's own, which `assign_cp_agg`
//! enables itself.
//!
//! Collapsing the two laned bags into this one saved 12 advice / 8 lookups of
//! shared preamble (the second seam stage) and, per lane, 28 advice / 18 lookups
//! (the tuple columns, the [x<y] comparator, the occurrence id and its
//! comparator, and the two membership lookups). The newly enforced mirror half
//! of (9) then spends 1 advice per lane (`ck1`) and 1 advice + 1 lookup per
//! pair (the host bit and its lookup), i.e. P += c + 4c^2. Net: P went
//! 543 -> 431 at c = 1, 1072 -> 892 at c = 2, 1755 -> 1515 at c = 3 and
//! 2592 -> 2300 at c = 4.
//!
//! cs.degree() is 7 for every configuration, the same bound `g_sql4_obj` pays
//! for its own membership lookup (2 + 3 + 2); the mirror lookup of (9) is
//! deliberately shaped to sit at that same 2 + 3 + 2.
//! `lanes_are_structural_replicas` locks the shape down as EXACTLY QUADRATIC in
//! c, which is the privacy property that every lane costs the same. Per lane the
//! circuit adds no fixed column; per probe pair it adds no fixed column, no
//! selector, no shuffle and no permutation column.
//!
//! Row budget per lane, at circuit degree k:
//!   lane_rows = 2^k - PREAMBLE_ROWS - BLINDING_SLACK
//! where PREAMBLE_ROWS covers the u8 range-table `load` regions (each is 256
//! fixed rows in a region of its own, and the floor planner is free to place
//! them ahead of the witness region) and BLINDING_SLACK covers the blinding
//! rows halo2 reserves at the bottom of every advice column. That is the height
//! a lane MAY have; with a released capacity supplied the last lane takes only
//! what that capacity leaves it, so the assigned rows total `capacity` rather
//! than `c * lane_rows`.
//!
//! WHAT THE VERIFYING KEY DISCLOSES
//! -------------------------------
//! Selectors are fixed columns, so every row range a lane is gated on is
//! committed at keygen and is visible to the verifier. The field that decides
//! those ranges is [`MyCircuit::released_capacity`], and it is a STRUCTURAL
//! input the caller supplies. It is never reconstructed from the witness: a
//! capacity computed as `w.len() + pad_extra` would put the true bag size in
//! the verifying key wherever the pad is a public constant.
//!
//!   * `Some(cap)`: the last lane stops at `cap`, so the vk pins `cap` (and
//!     with it the lane count). SOUND ONLY WHERE `cap` IS A GENUINE RELEASE,
//!     i.e. a number the verifier is told anyway. Under `Privacy::Dp` it is the
//!     DP release, whose whole point is to be published; under `Privacy::Rjs`
//!     it is the true bag size, which that regime reveals by definition.
//!   * `None`: every lane is filled to `lane_rows` and the vk pins only the
//!     lane count `c = ceil(cap / lane_rows)`. This is what `Privacy::Legacy`
//!     must use: its pad is a public CONSTANT, so `cap = true_size + constant`
//!     and a `cap` in the vk would pin the true bag size, the one statistic the
//!     padding exists to hide.
//!
//! `None` is the default, so a caller that says nothing gets the layout this
//! file had before the short last lane existed. The policy that maps a privacy
//! regime onto the two lives in [`crate::dp_lane::vk_released_capacity`].
//!
//! What the vk discloses about the single-copy stages is unchanged and is not a
//! function of this field: their height tracks |E|, public in every regime,
//! which is why they are not laned (see WHAT STAYS SINGLE COPY above).

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

/// Public structural cap on the lane count. GQ4 releases at eps = 0.01 run
/// into the dozens of lanes (gq4-lastfm needs 63 at k = 18), so this is
/// deliberately far above q5_obj_dp's cap of 16. Note that the circuit pays
/// c * c probes, so a config near this cap is enormous: see the COST NOTE.
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
/// Every lane but the last is full. The LAST one stops at `capacity` rather
/// than at `c * lane_rows`, so the rows the lane count rounds up to never
/// exist: they are not assigned, no selector covers them, no table side reads
/// them and the last lane's child stage never groups them. A caller that passes
/// `capacity = c * lane_rows` gets the full layout back, and that is what
/// [`MyCircuit::released_capacity`] `= None` does.
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
/// file may be a function of the true bag size, for the reason spelled out at
/// the selector block in `configure`.
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
// caller stamps the PUBLIC lane count here before keygen / MockProver runs;
// `synthesize` asserts the resulting config matches the circuit.
// Thread-local so parallel tests with different lane counts cannot race.
thread_local! {
    static CONFIG_LANES: std::cell::Cell<usize> = const { std::cell::Cell::new(1) };
}

/// Stamp the lane count the NEXT `configure` call on this thread lays out.
/// ONE count: both readings of the 4-cycle live on the same lanes, because they
/// are the same relation.
pub fn set_config_lanes(lanes: usize) {
    assert!(
        (1..=MAX_LANES).contains(&lanes),
        "lane count {} outside 1..={}",
        lanes,
        MAX_LANES
    );
    CONFIG_LANES.with(|l| l.set(lanes));
}

/// MockProver fault injection used by the negative tests in this file.
/// Production callers leave this at `None`; it only perturbs the witness,
/// never the constraint system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tamper {
    #[default]
    None,
    /// Bump one lane row's input-channel multiplicity by 1 (the root
    /// multiplicity gate and that lane's own prefix-sum gates must reject).
    LaneMuAll,
    /// Bump one lane total by 1 (the copy constraint into the totals stage and
    /// the totals accumulator must both reject).
    LaneTotal,
    /// Break the pad-row convention on the last rounding-up row in a way the
    /// ROLE-1 key gate sees: it sets `x`, which the role-1 key is packed from,
    /// so the "role 1 real + key" gate and the r_in membership lookup must both
    /// reject.
    Bag1PadRow,
    /// Break the pad-row convention on the last rounding-up row in a way ONLY
    /// the shared indexed views see: it sets `y`, the join column, which no
    /// per-row gate constrains on a pad row (`real = 0` gates both comparators
    /// and both keys out), so the two membership lookups are the only thing that
    /// can reject.
    Bag2PadRow,
    /// Zero the retrieved input-channel value of every probe EXCEPT probe 0,
    /// while leaving the root multiplicity at its honest (all-lane) value. This
    /// is the direct negative test for the cross-lane message sum: if the root
    /// gate ever stopped summing all `c` probes, or if a probe's membership
    /// lookup were dropped, this witness would verify. It must not.
    DropProbeVal,
    /// Move one joinable role-1 tuple to the residual side and re-reduce role 2
    /// around it, so conditions (7) and (9) still hold and only the Cardinality
    /// Preservation Check can see the hidden tuple. The check compares GLOBAL
    /// totals, so it sees it in whichever lane the tuple lives.
    HideOneCleanTuple,
    /// Skip the semijoin reduction and declare every predicate-passing tuple of
    /// both roles clean, leaving the residual side empty. Both channels of (10)
    /// then agree on every row, so only Pairwise Consistency can see the
    /// dangling role-1 tuples.
    MarkAllClean,
    /// Copy the first kept occurrence over the FIRST row of the LAST lane. Both
    /// copies pass both membership lookups and both channels of (10) move by the
    /// same amount, and every lane's own occurrence block stays sorted, so the
    /// ONLY constraint that can see the duplicate is the lane seam of the global
    /// ordering. A per-lane ordering would accept it.
    DuplicateOccurrenceAcrossLanes,
    /// Point one clean role-2 row's host bit at a lane that does NOT hold a
    /// clean role-1 row with its key. Exactly one bit is still set, so the
    /// "hosted by exactly one lane" gate still passes and only the mirror
    /// lookup of condition (9) can see it. Without this the host bits would be
    /// free advice and the mirror direction would be vacuous.
    MisnameHostLane,
}

// ---------------------------------------------------------------------------
// W rows, in the order the global occurrence ordering requires.
// ---------------------------------------------------------------------------

/// One row of the shared relation W as the circuit sees it:
/// `(x,y,z,i,j,e1,e2,real)`. Read as (A,B,C) it is the path A->B->C (role 1),
/// read as (C,D,A) it is the path C->D->A (role 2).
pub(crate) type BagRow = [u64; 8];

/// The ROLE-1 predicate bit, `real * [x<y] * [y<z]`, i.e. `[A<B] * [B<C]`.
/// Structurally this is `role2_keep * [y<z]`, which is the form the circuit
/// gate uses and the form the single occurrence ordering relies on.
pub(crate) fn role1_pred(r: &BagRow) -> u64 {
    (r[7] == 1 && r[0] < r[1] && r[1] < r[2]) as u64
}

/// The ROLE-2 keep bit, `real * [x<y]`, i.e. `[C<D]`. This is W's own filter,
/// so it holds on every real row and fails on every pad row.
pub(crate) fn role2_keep(r: &BagRow) -> u64 {
    (r[7] == 1 && r[0] < r[1]) as u64
}

/// The occurrence id of a W row: `pack2` of the two source edge ids when the
/// ROLE-2 keep bit holds, PAD when it does not.
pub(crate) fn bag_oid(r: &BagRow, keep: u64) -> u64 {
    if keep == 1 {
        pack2(r[5], r[6])
    } else {
        PAD_U64
    }
}

/// W over the whole released capacity, in the global occurrence-id order the
/// distinctness argument requires.
///
/// The sort key is the ROLE-2 occurrence id, and the role-2 keep bit is W's own
/// x<y filter, so EVERY real row carries a real id and the PAD tail is exactly
/// the pad rows. Role 1 needs no ordering of its own: its rows are the role-2
/// rows that also satisfy [y<z], a subset determined per row by the
/// "role 1 pred = keep * [y<z]" gate, so distinctness of the role-2 occurrences
/// carries over to them.
pub(crate) fn w_rows(
    w: &[(u64, u64, u64, u64, u64, u64, u64)],
    capacity: usize,
) -> Vec<BagRow> {
    let mut rows: Vec<BagRow> = w
        .iter()
        .map(|&(x, y, z, i, j, e1, e2)| [x, y, z, i, j, e1, e2, 1])
        .collect();
    rows.truncate(capacity);
    rows.resize(capacity, [0u64; 8]);
    rows.sort_by_key(|r| bag_oid(r, role2_keep(r)));
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

/// The part of a lane that pass 1 of `configure_w_lane` lays out: the tuple
/// columns, both column ROLES of them, the occurrence id and the child stage.
/// A lane's probes cannot be built yet, because they must reference EVERY
/// lane's child stage, so they come in pass 2.
#[derive(Clone, Debug)]
struct WLaneRows<F: Field + Ord> {
    // W rows: x->y->z with x<y, plus the source-edge ids and per-group indices.
    // (A,B,C) is the role-1 reading and (C,D,A) the role-2 reading of the SAME
    // eight columns.
    w_x: Column<Advice>,
    w_y: Column<Advice>,
    w_z: Column<Advice>,
    w_i: Column<Advice>,
    w_j: Column<Advice>,
    w_e1: Column<Advice>,
    w_e2: Column<Advice>,
    w_real: Column<Advice>,

    // [x<y], which is role 1's [A<B] and role 2's [C<D] at once, and [y<z],
    // which is role 1's [B<C]. One comparator each, not one per role.
    lt_xy: LtConfig<F, NUM_BYTES>,
    lt_yz: LtConfig<F, NUM_BYTES>,

    // ROLE 2 and condition (7) on it: the keep bit, the separator key
    // pack2(z,x) (PAD on a row that is not kept, and a PAD key never reaches
    // the child table), the prover's clean indicator, the folded `keep * cflag`
    // both channels read, and `ceff2 * key2` for the mirror half of (9).
    keep2: Column<Advice>,
    key2: Column<Advice>,
    cflag2: Column<Advice>,
    ceff2: Column<Advice>,

    // ROLE 1 and condition (7) on it: the separator key pack2(x,z) REVERSED
    // against role 2's, the predicate bit `keep2 * [y<z]`, this role's own
    // clean indicator, the folded `pred * cflag`, and `ceff1 * key1`, which is
    // the table side of the mirror half of (9).
    key1: Column<Advice>,
    pred1: Column<Advice>,
    cflag1: Column<Advice>,
    ceff1: Column<Advice>,
    ck1: Column<Advice>,

    // occurrence id and its within-lane ordering, over the ROLE-2 keep bit
    oid: Column<Advice>,
    iz_oid_pad: IsZeroConfig<F>,
    lt_oid: LtConfig<F, NUM_BYTES>,

    // child side of the bag tree edge over the ROLE-2 reading: per key,
    // (sum keep2, sum ceff2)
    cp: CpAggConfig<F, NUM_BYTES>,
}

/// One lane: everything in [`WLaneRows`], plus one parent probe per lane, the
/// host bits of the mirror half of (9), and a LANE-LOCAL prefix sum of each
/// channel. `out` is NOT replicated; it lives once in the cross-lane totals
/// stage.
#[derive(Clone, Debug)]
pub struct WLaneConfig<F: Field + Ord> {
    w_x: Column<Advice>,
    w_y: Column<Advice>,
    w_z: Column<Advice>,
    w_i: Column<Advice>,
    w_j: Column<Advice>,
    w_e1: Column<Advice>,
    w_e2: Column<Advice>,
    w_real: Column<Advice>,

    lt_xy: LtConfig<F, NUM_BYTES>,
    lt_yz: LtConfig<F, NUM_BYTES>,

    keep2: Column<Advice>,
    key2: Column<Advice>,
    cflag2: Column<Advice>,
    ceff2: Column<Advice>,

    key1: Column<Advice>,
    pred1: Column<Advice>,
    cflag1: Column<Advice>,
    ceff1: Column<Advice>,
    ck1: Column<Advice>,

    oid: Column<Advice>,
    iz_oid_pad: IsZeroConfig<F>,
    lt_oid: LtConfig<F, NUM_BYTES>,

    cp: CpAggConfig<F, NUM_BYTES>,

    // membership-or-gap probe of the child table, ONE PER LANE, including this
    // lane's own. Counts add across a partition, so the row's two
    // multiplicities are the sums of the c retrieved values.
    probes: Vec<CpJoinConfig<F, NUM_BYTES>>,

    // condition (9), mirror direction: `host[l]` is the prover's claim that
    // lane l holds a clean role-1 row with this row's role-2 key. Boolean, and
    // gated to sum to `ceff2`, so a non-clean row names no lane.
    host: Vec<Column<Advice>>,

    // the two root multiplicities and their lane-local prefix sums. `mu_all` is
    // both the answer's per-row contribution and the input channel of (10).
    mu_all: Column<Advice>,
    mu_cln: Column<Advice>,
    sum_all: Column<Advice>,
    sum_cln: Column<Advice>,

    // is the clean multiplicity fetched from ALL lanes zero? condition (9)
    iz_cln: IsZeroConfig<F>,
}

#[derive(Clone, Debug)]
pub struct Gq4DpConfig<F: Field + Ord> {
    instance: Column<Instance>,

    // ---------------- fixed-height sections (shared, never laned) ----------
    // The base Edge relation, and the two indexed views projected from it. The
    // views are the table side of every r_in/r_out membership lookup; their
    // height tracks |E|, which is public in every regime.
    e_eid: Column<Advice>,
    e_src: Column<Advice>,
    e_dst: Column<Advice>,
    in_by_dst: IndexedViewConfig<F>,  // key=dst, val=src
    out_by_src: IndexedViewConfig<F>, // key=src, val=dst

    /// Table-side gate of the two lane lookups, enabled on exactly the n_base
    /// real rows of both views. See `configure_w_lane_rows` for why.
    q_view_tbl: Selector,

    /// Both views are the base relation's (src, dst, eid) multiset.
    perm_edge_out: PermAnyConfig,
    perm_edge_in: PermAnyConfig,

    /// One u8 range table for EVERY lane Lt chip. `load` costs one 256-row
    /// region per distinct u8 column, and that must not grow with the lane
    /// count.
    lane_u8: Column<Fixed>,

    // ---------------- the lanes -------------------------------------------
    // Lanes 0..c-1 are active on the SAME rows 0..lane_rows, so one selector of
    // each kind serves all of them and all of their probes; only the columns
    // replicate. Lane c-1 stops at the released capacity and reads `last`.
    // (The per-lane child stages keep their own selectors, since
    // `assign_cp_agg` enables them itself, so they follow their lane's height
    // with nothing to do here.)
    lanes: Vec<WLaneConfig<F>>,
    /// Rows 0..lane_rows, every lane but the last. `q_w_lookup` is complex and
    /// enabled on exactly a lane's rows: it gates the INPUT side of the two
    /// membership lookups and BOTH sides of the mirror half of (9), whose table
    /// side must cover exactly the rows of the lane that OWNS the column, or the
    /// rows past that lane would be free advice minting table entries.
    /// The two probe selectors are in here too: every probe carries them, and
    /// `assign_probe` enables them through the probe it is assigning.
    full: WSelectors,
    /// Rows 0..lane_live_rows(c-1, ..), the last lane.
    last: WSelectors,

    // ---------------- global occurrence ordering ---------------------------
    /// The lane-seam link of the ONE ordering. Both roles ride on it.
    oid_bnd: OidBoundaryConfig<F>,

    // ---------------- cross-lane totals ------------------------------------
    // Row l holds lane l's two channel totals, by copy constraint. This is
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

    /// The parent side of one (row lane, child lane) probe.
    ///
    /// This is `card_preserve::configure_cp_join` with two changes a replicated
    /// caller needs. The two selectors are supplied rather than allocated, since
    /// all c*c probes are live on the same rows and a selector per probe would
    /// make the selector count carry a c^2 term. And the five columns are kept
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

    /// PASS 1 of one lane: the tuple columns, the two membership lookups, the
    /// two comparators, both column ROLES with their partitions, the occurrence
    /// id and the child stage of the bag tree edge.
    ///
    /// It stops there because a lane's probes must reference EVERY lane's child
    /// stage, including the ones not built yet, so the probes and everything
    /// that reads them are pass 2 (`configure_w_lane_probes`).
    ///
    /// Every lane gets the same gates and the same lookups; only the columns
    /// differ.
    fn configure_w_lane_rows(
        meta: &mut ConstraintSystem<F>,
        in_by_dst: &IndexedViewConfig<F>,
        out_by_src: &IndexedViewConfig<F>,
        lane_u8: Column<Fixed>,
        sel: &WSelectors,
    ) -> WLaneRows<F> {
        let WSelectors {
            q_view_tbl,
            q_w_lookup,
            q_w_row,
            q_oid_step,
            ..
        } = *sel;

        let w_x = meta.advice_column();
        let w_y = meta.advice_column();
        let w_z = meta.advice_column();
        let w_i = meta.advice_column();
        let w_j = meta.advice_column();
        let w_e1 = meta.advice_column();
        let w_e2 = meta.advice_column();
        let w_real = meta.advice_column();
        // NOTE: no `enable_equality` on the tuple columns. halo2 charges for
        // every column in the permutation argument whether or not a copy
        // constraint ever touches it, so a per-lane column that is only read by
        // gates and lookups must stay out of it. The only lane cells that are
        // ever copied are the two ends of `oid`, for the lane seam, and the last
        // cell of each of the two channel prefix sums, for the totals stage.

        // The table side of the two membership lookups is gated by
        // `q_view_tbl`, over exactly the n_base real rows of both views.
        //
        // Each `IndexedView` carries a sentinel row at `n_base` for the
        // Rotation::next() of its own sortedness gate. That row is outside the
        // view's shuffle, outside `q_idx0` and outside `q_idx`, so `sorted_val`,
        // `sorted_eid` and `idx` there are free advice and only `sorted_key` is
        // touched at all (bounded above the largest real key). With an ungated
        // table side that sentinel is a LIVE table row, so a prover can mint an
        // edge the relation does not contain: set out_by_src's sentinel to
        // (key = D, idx = 0, val = A0, eid = anything) with D above every real
        // src, and a W row (x = C0, y = D, z = A0) then passes
        // "W r_out from out_by_src" for an r_out = D->A0 edge that exists
        // nowhere.
        //
        // On the ungated rows every table expression reads 0, so the tuple
        // (0,0,0,0) is in the table, which is what the padded W rows look up
        // anyway, and it is a real row of both views regardless: base row 0 is
        // the (0,0,0) dummy edge with idx 0. The table side goes from degree 1
        // to degree 2, so these lookups go from 2+2+1 = 5 to 2+2+2 = 6, still
        // under the 2+3+2 = 7 the check's own membership lookup costs, and
        // cs.degree() does not move.
        //
        // TWO lookups, not four. In W coordinates the role-2 pair is literally
        // the role-1 pair: role 2's incoming edge is (y, i) -> (x, e1) against
        // in_by_dst and its outgoing edge is (y, j) -> (z, e2) against
        // out_by_src, the same cells of the same row against the same two
        // views. Reading the row as (C,D,A) instead of (A,B,C) renames the
        // columns and changes nothing a lookup can see. The `r_out` lookup is
        // also the closing-edge check of the cycle: `val` is `w_z`, so the edge
        // that closes the 4-cycle in the role-2 reading is proven to exist by
        // the same lookup that produces its endpoint.
        //
        // r_in via InByDst: key=y, idx=i -> val=x, eid=e1
        meta.lookup_any("W r_in from in_by_dst", |m| {
            let q = m.query_selector(q_w_lookup);
            let t = m.query_selector(q_view_tbl);
            vec![
                (
                    q.clone() * m.query_advice(w_y, Rotation::cur()),
                    t.clone() * m.query_advice(in_by_dst.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(w_i, Rotation::cur()),
                    t.clone() * m.query_advice(in_by_dst.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(w_x, Rotation::cur()),
                    t.clone() * m.query_advice(in_by_dst.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(w_e1, Rotation::cur()),
                    t * m.query_advice(in_by_dst.sorted_eid, Rotation::cur()),
                ),
            ]
        });
        // r_out via OutBySrc: key=y, idx=j -> val=z, eid=e2
        meta.lookup_any("W r_out from out_by_src", |m| {
            let q = m.query_selector(q_w_lookup);
            let t = m.query_selector(q_view_tbl);
            vec![
                (
                    q.clone() * m.query_advice(w_y, Rotation::cur()),
                    t.clone() * m.query_advice(out_by_src.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(w_j, Rotation::cur()),
                    t.clone() * m.query_advice(out_by_src.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(w_z, Rotation::cur()),
                    t.clone() * m.query_advice(out_by_src.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(w_e2, Rotation::cur()),
                    t * m.query_advice(out_by_src.sorted_eid, Rotation::cur()),
                ),
            ]
        });

        // The two order checks of the row, enabled only if real=1. `lt_xy`
        // compares (x,y), which is role 1's [A<B] and role 2's [C<D] at once,
        // and `lt_yz` compares (y,z), which is role 1's [B<C]. Two chips, not
        // three: the separate role-2 comparator had operands (t34_c, t34_d) =
        // (x, y) and so was the same predicate on the same cells as `lt_xy`.
        let lt_xy = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            lane_u8,
            |m| m.query_selector(q_w_row) * m.query_advice(w_real, Rotation::cur()),
            |m| m.query_advice(w_x, Rotation::cur()),
            |m| m.query_advice(w_y, Rotation::cur()),
        );
        let lt_yz = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            lane_u8,
            |m| m.query_selector(q_w_row) * m.query_advice(w_real, Rotation::cur()),
            |m| m.query_advice(w_y, Rotation::cur()),
            |m| m.query_advice(w_z, Rotation::cur()),
        );

        // ---- ROLE 2 and condition (7) on it ----
        // `keep2` is the role-2 predicate bit, which is W's own x<y filter;
        // `key2` the separator key pack2(A,C) = pack2(z,x) on the kept rows and
        // PAD elsewhere (a PAD key never reaches the child table); `cflag2` the
        // prover's clean indicator and `ceff2 = keep2 * cflag2` the bit the
        // clean channel reads. `g_sql4_obj` had no booleanity gate on the
        // indicator at all; without it a fractional "indicator" would scale the
        // clean channel freely. And `ceff2` is what forces the indicator to 0
        // wherever the predicate fails, so a row that is not in R cannot be in
        // R^c. `ck1 = ceff1 * key1` is the table side of the mirror half of (9),
        // folded into a column so that the lookup input stays at degree 3.
        //
        // The keep bit is real * [x<y], NOT the role-1 predicate. Aggregating
        // over the role-1 predicate would make the message count only rows that
        // also satisfy [y<z], i.e. it would silently require D<A on the C->D->A
        // path and change the answer.
        let keep2 = meta.advice_column();
        let key2 = meta.advice_column();
        let cflag2 = meta.advice_column();
        let ceff2 = meta.advice_column();

        meta.create_gate("role 2 keep, separator key and clean partition", |m| {
            let q = m.query_selector(q_w_row);
            let one = Expression::Constant(F::ONE);
            let pad = Expression::Constant(F::from(PAD_U64));

            let real = m.query_advice(w_real, Rotation::cur());
            let xy = lt_xy.is_lt(m, None);
            let keep = m.query_advice(keep2, Rotation::cur());
            let key = m.query_advice(key2, Rotation::cur());
            let cf = m.query_advice(cflag2, Rotation::cur());
            let ce = m.query_advice(ceff2, Rotation::cur());

            // pack2(A,C) = pack2(z,x), REVERSED against role 1's pack2(x,z) on
            // the SAME two cells of the SAME row. That reversal is the whole
            // mechanism: M(p,q) = |{w in W : x = p, z = q}| is aggregated on
            // pack2(z,x) and probed on pack2(x,z).
            let key_expr = m.query_advice(w_z, Rotation::cur())
                * Expression::Constant(F::from(PACK_SHIFT))
                + m.query_advice(w_x, Rotation::cur());

            vec![
                q.clone() * (keep.clone() - real * xy),
                q.clone()
                    * (key.clone() - (keep.clone() * key_expr + (one.clone() - keep.clone()) * pad)),
                q.clone() * cf.clone() * (one - cf.clone()),
                q * (ce.clone() - keep * cf),
            ]
        });

        // ---- ROLE 1 and condition (7) on it ----
        // `key1` is the probe key pack2(A,C) = pack2(x,z) FORWARD, pinned on
        // every row (a padding row's tuple is all zero, so its key is 0, which
        // is the dummy row of every child table and carries both sums 0).
        // `pred1` folds the three-factor predicate into a column so that every
        // gate below stays at degree <= 4, and is stated STRUCTURALLY as
        // `keep2 * [y<z]` on top of the role-2 bit rather than as
        // real * [x<y] * [y<z]. Beyond dropping a degree, that form is what
        // licenses the SINGLE occurrence-distinctness argument: role 1's rows
        // are exactly the role-2 rows that also pass a genuine Lt on (y,z),
        // decided per row. Do not weaken this gate without restoring a second
        // ordering.
        let key1 = meta.advice_column();
        let pred1 = meta.advice_column();
        let cflag1 = meta.advice_column();
        let ceff1 = meta.advice_column();
        let ck1 = meta.advice_column();

        meta.create_gate("role 1 real + key", |m| {
            let q = m.query_selector(q_w_row);
            let one = Expression::Constant(F::ONE);
            let real = m.query_advice(w_real, Rotation::cur());
            let key_expr = m.query_advice(w_x, Rotation::cur())
                * Expression::Constant(F::from(PACK_SHIFT))
                + m.query_advice(w_z, Rotation::cur());
            vec![
                q.clone() * real.clone() * (one - real),
                q * (m.query_advice(key1, Rotation::cur()) - key_expr),
            ]
        });

        meta.create_gate("role 1 pred = keep * [y<z], and clean partition", |m| {
            let q = m.query_selector(q_w_row);
            let one = Expression::Constant(F::ONE);
            let keep = m.query_advice(keep2, Rotation::cur());
            let yz = lt_yz.is_lt(m, None);
            let pred = m.query_advice(pred1, Rotation::cur());
            let cf = m.query_advice(cflag1, Rotation::cur());
            let ce = m.query_advice(ceff1, Rotation::cur());
            vec![
                q.clone() * (pred.clone() - keep * yz),
                q.clone() * cf.clone() * (one - cf.clone()),
                q.clone() * (ce.clone() - pred * cf),
                // the TABLE side of the mirror half of (9): the clean role-1
                // keys of this lane, 0 on every other row
                q * (m.query_advice(ck1, Rotation::cur())
                    - ce * m.query_advice(key1, Rotation::cur())),
            ]
        });

        // ---- occurrence id, over the ROLE-2 keep bit ----
        let oid = meta.advice_column();
        meta.enable_equality(oid);
        meta.create_gate("occ: id = pack2(edge ids), PAD when not kept", |m| {
            let q = m.query_selector(q_w_row);
            let one = Expression::Constant(F::ONE);
            let pad = Expression::Constant(F::from(PAD_U64));
            let keep = m.query_advice(keep2, Rotation::cur());
            let id = m.query_advice(w_e1, Rotation::cur())
                * Expression::Constant(F::from(PACK_SHIFT))
                + m.query_advice(w_e2, Rotation::cur());
            vec![
                q * (m.query_advice(oid, Rotation::cur())
                    - (keep.clone() * id + (one - keep) * pad)),
            ]
        });
        let (iz_oid_pad, lt_oid) = Self::configure_oid_order(meta, lane_u8, q_oid_step, oid);

        // ---- child side of the bag tree edge, both channels ----
        // The role-2 reading is a leaf of the bag tree, so a row's input-channel
        // multiplicity is its keep bit and its clean-channel multiplicity is
        // that bit times the clean indicator. Both already sit in columns this
        // lane carries, so the second channel costs only the stage's own columns
        // and the stage groups them by the separator key in one pass.
        let cp = configure_cp_agg::<F, NUM_BYTES>(meta, lane_u8, key2, keep2, ceff2, PAD_U64);

        WLaneRows {
            w_x,
            w_y,
            w_z,
            w_i,
            w_j,
            w_e1,
            w_e2,
            w_real,
            lt_xy,
            lt_yz,
            keep2,
            key2,
            cflag2,
            ceff2,
            key1,
            pred1,
            cflag1,
            ceff1,
            ck1,
            oid,
            iz_oid_pad,
            lt_oid,
            cp,
        }
    }

    /// PASS 2 of one lane: one parent probe per lane, the host bits and lookup
    /// that close the mirror half of condition (9), the two root multiplicities
    /// and the lane-local prefix sums.
    ///
    /// `lane_cps` and `lane_ck1` are indexed by lane and must cover EVERY lane,
    /// this one included: a role-1 row's key can live in any lane's child table,
    /// and a clean role-2 row can be hosted by any lane. `lane_ck1` carries each
    /// clean-key column together with the row selector of the lane that OWNS it,
    /// which is what gates that column as a table side.
    fn configure_w_lane_probes(
        meta: &mut ConstraintSystem<F>,
        rows: WLaneRows<F>,
        lane_cps: &[CpAggConfig<F, NUM_BYTES>],
        lane_ck1: &[(Column<Advice>, Selector)],
        lane_u8: Column<Fixed>,
        sel: &WSelectors,
    ) -> WLaneConfig<F> {
        let WSelectors {
            q_w_lookup,
            q_probe,
            q_probe_complex,
            q_mu,
            q_sum0,
            q_sum,
            ..
        } = *sel;
        let WLaneRows {
            w_x,
            w_y,
            w_z,
            w_i,
            w_j,
            w_e1,
            w_e2,
            w_real,
            lt_xy,
            lt_yz,
            keep2,
            key2,
            cflag2,
            ceff2,
            key1,
            pred1,
            cflag1,
            ceff1,
            ck1,
            oid,
            iz_oid_pad,
            lt_oid,
            cp,
        } = rows;

        // ---- one probe per lane ----
        // Each probe is a complete replica pointed at that lane's child table;
        // they share the two probe selectors and the u8 column because every
        // probe is live on every row. This is the c^2 term, and it cannot be
        // shared: a `cp_agg` stage owns ONE input column pair bounded by the
        // rows of the domain, while W spans c * lane_rows rows.
        let probes: Vec<CpJoinConfig<F, NUM_BYTES>> = lane_cps
            .iter()
            .map(|child| {
                let j = Self::configure_probe(meta, lane_u8, q_probe, q_probe_complex, key1);
                wire_cp_edge(meta, &j, child, key1);
                j
            })
            .collect();

        // ---- condition (9), the MIRROR direction ----
        // pi_K(R_2^c) subset of pi_K(R_1^c), whose table is the union of the c
        // per-lane clean role-1 key columns. A lookup cannot say "in lane 1 OR
        // in lane 2", and asking every lane to contain every clean role-2 key
        // would reject every honest witness. So the prover NAMES the hosting
        // lane with a boolean bit per lane, the gate below forces those bits to
        // sum to `ceff2` (not to 1: a row that is not role-2 clean must name NO
        // lane, so its input is zeroed), and one width-1 lookup
        // per named lane checks the claim against that lane's own clean role-1
        // keys.
        //
        // BOTH SIDES ARE GATED, which is the part that has to be right. The
        // table side is `q_l * ck1_l` where `q_l` is the complex row selector of
        // the lane that OWNS `ck1_l`, enabled on exactly that lane's live rows:
        // without it the rows past the lane's capacity would sit outside every
        // gate, so `ck1` there would be free advice and a prover could mint any
        // clean key it liked. It is the owner's selector and NOT this lane's,
        // because the last lane stops at the released capacity while the rest
        // run to `lane_rows`, and gating a short lane's column by a full lane's
        // selector would reopen exactly that hole. On the gated-off rows both
        // sides read 0, so 0 is in the table and the rows cost nothing; a real
        // key is pack2 of two shifted node ids, hence at least PACK_SHIFT + 1,
        // so the containment is over the real clean keys only. Input degree 3
        // and table degree 2 put this lookup at the same 2 + 3 + 2 = 7 the
        // check's own membership lookup already pays.
        let host: Vec<Column<Advice>> = lane_cps.iter().map(|_| meta.advice_column()).collect();
        {
            let hosts = host.clone();
            meta.create_gate(
                "pw: every clean role-2 tuple is hosted by exactly one lane",
                move |m| {
                    let q = m.query_selector(q_mu);
                    let one = Expression::Constant(F::ONE);
                    let mut polys = Vec::with_capacity(hosts.len() + 1);
                    let mut sum = Expression::Constant(F::ZERO);
                    for h in hosts.iter() {
                        let hq = m.query_advice(*h, Rotation::cur());
                        polys.push(q.clone() * hq.clone() * (one.clone() - hq.clone()));
                        sum = sum + hq;
                    }
                    polys.push(q * (sum - m.query_advice(ceff2, Rotation::cur())));
                    polys
                },
            );
        }
        for (h, (tbl, q_tbl)) in host.iter().zip(lane_ck1.iter()) {
            let (h, tbl, q_tbl) = (*h, *tbl, *q_tbl);
            meta.lookup_any("pw: role2^c key in role1^c (lane host)", move |m| {
                let gate = m.query_selector(q_w_lookup) * m.query_advice(h, Rotation::cur());
                // `key2`, not `ceff2 * key2`: the host gate forces
                // `sum_l host[l] = ceff2` with every bit boolean, so a SET host
                // bit already implies `ceff2 = 1`, and a clear one zeroes the
                // product. Folding `ceff2` in again would cost one advice
                // column and one degree-3 gate per lane for nothing.
                vec![(
                    gate * m.query_advice(key2, Rotation::cur()),
                    m.query_selector(q_tbl) * m.query_advice(tbl, Rotation::cur()),
                )]
            });
        }

        // ---- the two root multiplicities, and condition (9) direction 1 ----
        let mu_all = meta.advice_column();
        let mu_cln = meta.advice_column();
        let sum_all = meta.advice_column();
        let sum_cln = meta.advice_column();
        meta.enable_equality(sum_all);
        meta.enable_equality(sum_cln);

        // Is the clean multiplicity of this row's key, summed over ALL lanes,
        // zero? A per-lane test would be meaningless: a key legitimately lives
        // in only some of the lanes.
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
                let pred = m.query_advice(pred1, Rotation::cur());
                let ceff = m.query_advice(ceff1, Rotation::cur());
                vec![
                    // the input channel, which is also this row's contribution
                    // to the COUNT
                    q.clone() * (m.query_advice(mu_all, Rotation::cur()) - pred * s_all),
                    // the clean channel
                    q.clone() * (m.query_advice(mu_cln, Rotation::cur()) - ceff.clone() * s_cln),
                    // condition (9), the direction pi_K(R_1^c) subset of
                    // pi_K(R_2^c): a role-1 tuple left on the clean side must
                    // have at least one clean role-2 partner SOMEWHERE in W
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

        WLaneConfig {
            w_x,
            w_y,
            w_z,
            w_i,
            w_j,
            w_e1,
            w_e2,
            w_real,
            lt_xy,
            lt_yz,
            keep2,
            key2,
            cflag2,
            ceff2,
            key1,
            pred1,
            cflag1,
            ceff1,
            ck1,
            oid,
            iz_oid_pad,
            lt_oid,
            cp,
            probes,
            host,
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
        // with nothing relating them to each other, so the lane lookups proved
        // membership in two INDEPENDENT prover-invented relations: `r_in` could
        // come from one edge list and `r_out` from another. Two shuffles fix
        // that. `in_by_dst` is sorted (dst, src, eid) and `out_by_src` is
        // sorted (src, dst, eid), so both are compared against the base
        // relation's (src, dst, eid) with the key/val columns swapped on the
        // in-side. The comparison is against the SORTED columns, which are the
        // ones the lane lookups actually read; each view's own shuffle already
        // ties those to its input columns, so this is the same statement and it
        // needs one fewer column group.
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

        // ---------------- the lanes ----------------
        // Shared selectors: every lane is active on the same rows, so one
        // selector of each kind drives all c replicas and all c*c probes.
        //
        // A NOTE ON WHY NO SELECTOR ROW MAY DEPEND ON THE WITNESS. Selectors are
        // fixed columns, so where they are enabled is part of the verifying key.
        // `g_sql4_obj` realizes condition (7) as a shuffle against a
        // [clean | residual | pad] column group and needs one selector per
        // section, i.e. the true clean and residual counts of each role are in
        // its vk. That is legitimate there, where the Revealing-Join-Size regime
        // publishes the bag size anyway, and it is exactly what this circuit may
        // not do: those counts are what the DP release pays to hide. Every
        // selector below is enabled on a row range determined by `lane_rows`,
        // `c` and [`MyCircuit::released_capacity`], none of which is read off
        // the witness. The capacity in particular is a STRUCTURAL input, and
        // stopping the last lane at it is sound only where the caller supplied
        // a genuine release: that is the value the harness prints and the one
        // that already fixes `c`, so a selector pattern stopping there reveals
        // nothing the lane count did not. Where there is no such release the
        // caller passes `None` and every lane is filled instead.
        let full = WSelectors {
            q_view_tbl,
            q_w_lookup: meta.complex_selector(),
            q_w_row: meta.selector(),
            q_probe: meta.selector(),
            q_probe_complex: meta.complex_selector(),
            q_mu: meta.selector(),
            q_sum0: meta.selector(),
            q_sum: meta.selector(),
            // The global occurrence ordering is shared machinery: every lane but
            // the last is live on rows 0..lane_rows, so one step selector drives
            // all of them.
            q_oid_step: meta.selector(),
        };

        // The last lane MAY stop short of `lane_rows`, so it cannot share the
        // ones above: a selector is a fixed column and turns its gate on for
        // every lane at once. Four more columns cover its four distinct row
        // ranges, whatever c is, and `q_sum0` stays shared because row 0 is live
        // in every lane. These four are laid out unconditionally, so the
        // constraint system does not move with `released_capacity`; under
        // `None` they simply cover all `lane_rows` rows and the layout is the
        // full one.
        let q_last_row = meta.selector();
        let q_last_complex = meta.complex_selector();
        let q_last_sum = meta.selector();
        let q_last_step = meta.selector();
        let last = WSelectors {
            q_view_tbl,
            q_w_lookup: q_last_complex,
            q_w_row: q_last_row,
            q_probe: q_last_row,
            q_probe_complex: q_last_complex,
            q_mu: q_last_row,
            q_sum0: full.q_sum0,
            q_sum: q_last_sum,
            q_oid_step: q_last_step,
        };
        let sel_of = |l: usize| if l + 1 == num_lanes { &last } else { &full };

        // TWO PASSES, because a lane's probes must reference EVERY lane's child
        // stage: pass 1 lays out the rows of all c lanes and their child stages,
        // pass 2 adds each lane's c probes, its host bits and its merge path.
        let stage1: Vec<WLaneRows<F>> = (0..num_lanes)
            .map(|l| {
                Self::configure_w_lane_rows(meta, &in_by_dst, &out_by_src, lane_u8, sel_of(l))
            })
            .collect();
        let lane_cps: Vec<CpAggConfig<F, NUM_BYTES>> =
            stage1.iter().map(|s| s.cp.clone()).collect();
        // The mirror half of (9) reads lane `l`'s `ck1` as a TABLE, so its table
        // side has to be gated by lane `l`'s OWN row selector and not by the
        // probing lane's: with the two ranges now different, gating a short
        // lane's table by a full lane's selector would leave the rows past the
        // short lane's capacity inside the table and outside every gate, i.e.
        // free advice minting clean keys. Carry the owner's selector alongside
        // the column so the pairing cannot come apart.
        let lane_ck1: Vec<(Column<Advice>, Selector)> = stage1
            .iter()
            .enumerate()
            .map(|(l, s)| (s.ck1, sel_of(l).q_w_lookup))
            .collect();
        let lanes: Vec<WLaneConfig<F>> = stage1
            .into_iter()
            .enumerate()
            .map(|(l, rows)| {
                Self::configure_w_lane_probes(
                    meta,
                    rows,
                    &lane_cps,
                    &lane_ck1,
                    lane_u8,
                    sel_of(l),
                )
            })
            .collect();

        // ---------------- lane seams of the occurrence ordering -------------
        // ONE stage: there is one occurrence ordering, over the role-2 keep bit,
        // and role 1's rows are a per-row-determined subset of role 2's.
        let oid_bnd = Self::configure_oid_boundary(meta, lane_u8);

        // ---------------- cross-lane totals ----------------
        // Row l holds lane l's two channel totals, copied in from that lane's
        // last `sum_all` / `sum_cln`; `tot_run` and `cln_run` add them with
        // degree-2 accumulators. A flat degree-(1+c) sum gate would blow past
        // cs.degree() = 7 for c > 6.
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

        // ---- condition (10), over the UNION of the lanes ----
        // ONE equality, between the two GLOBAL totals at row c-1. Accumulating
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
            lanes,
            full,
            last,
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

/// The row selectors one lane's machinery reads. Bundled to keep the two passes
/// of `configure_w_lane` under a readable argument count, and because there are
/// now TWO sets: a selector is one fixed column, so it is on or off for every
/// lane at once, and the last lane is live on fewer rows than the rest.
///
/// The `full` set serves lanes `0..c-1` over all `lane_rows` rows. The `last`
/// set serves lane `c-1` over its [`lane_live_rows`] rows only, and folds the
/// eight laned entries down to four columns, one per distinct ROW RANGE, which
/// is all a selector is: `0..live` for every gate, `0..live` for every lookup
/// (complex, since only a complex selector may be read by one), `1..live` for
/// the two prefix sums and `0..live-1` for the occurrence-order step, which
/// reads `Rotation::next()`. `q_view_tbl` is not laned at all: it gates the two
/// shared indexed views, whose height tracks |E|.
#[derive(Clone, Copy, Debug)]
struct WSelectors {
    q_view_tbl: Selector,
    q_w_lookup: Selector,
    q_w_row: Selector,
    q_probe: Selector,
    q_probe_complex: Selector,
    q_mu: Selector,
    q_sum0: Selector,
    q_sum: Selector,
    q_oid_step: Selector,
}

/// The per-row witness of the WHOLE released capacity, in the global
/// occurrence order. Every vector has `c * lane_rows` entries and is indexed by
/// the GLOBAL row index; lane l reads the block
/// `[l*lane_rows, (l+1)*lane_rows)`. Bundled so that `assign_w_lane` takes one
/// argument instead of twelve slices, all of which have to agree on their
/// indexing.
struct WWitness {
    rows: Vec<BagRow>,
    // role 2, the (C,D,A) reading: keep bit, aggregate key pack2(z,x), clean
    // indicator, folded keep*cflag, and ceff2*key2
    keep2: Vec<u64>,
    key2: Vec<u64>,
    cflag2: Vec<u64>,
    ceff2: Vec<u64>,
    // role 1, the (A,B,C) reading: probe key pack2(x,z), predicate bit, clean
    // indicator, folded pred*cflag, and ceff1*key1
    key1: Vec<u64>,
    pred1: Vec<u64>,
    cflag1: Vec<u64>,
    ceff1: Vec<u64>,
    ck1: Vec<u64>,
    /// occurrence id, over the role-2 keep bit
    oid: Vec<u64>,
    /// the lane the prover names as holding a clean role-1 row with this row's
    /// role-2 key, meaningful only where `ceff2` is 1
    host: Vec<usize>,
}

/// What one lane exports to the cross-lane stages.
struct WLaneOut<F: Field + Ord> {
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

    /// Assign one lane's block of the padded W pipeline: the tuple, both column
    /// roles with their partitions, the occurrence id, the child stage, the c
    /// probes, the host bits and the two lane-local prefix sums.
    ///
    /// `base` is the global index of this lane's row 0 and `live_rows` its
    /// [`lane_live_rows`], so lane l covers the padded rows
    /// `[l*lane_rows, l*lane_rows + live_rows)`: a full lane_rows except in the
    /// last lane, which stops at the released capacity. Rows past the released
    /// capacity's real part are padding: all-zero tuple, real 0, hence keep2 0,
    /// key2 PAD, pred1 0 and oid PAD. View row 0 is (0,0,0) with idx 0, so a
    /// padding row satisfies both membership lookups.
    #[allow(clippy::too_many_arguments)]
    fn assign_w_lane(
        lane: &WLaneConfig<F>,
        region: &mut Region<'_, F>,
        lane_idx: usize,
        live_rows: usize,
        base: usize,
        w: &WWitness,
        stages: &[CpStage],
        tamper: Tamper,
    ) -> Result<WLaneOut<F>, Error> {
        let lt_xy_chip = LtChip::<F, NUM_BYTES>::construct(lane.lt_xy);
        let lt_yz_chip = LtChip::<F, NUM_BYTES>::construct(lane.lt_yz);
        let lt_oid_chip = LtChip::<F, NUM_BYTES>::construct(lane.lt_oid);
        let iz_oid_chip = IsZeroChip::construct(lane.iz_oid_pad.clone());
        let iz_cln_chip = IsZeroChip::construct(lane.iz_cln.clone());
        let pad = F::from(PAD_U64);

        let mut first: Option<AssignedCell<F, F>> = None;
        let mut last: Option<AssignedCell<F, F>> = None;

        for r in 0..live_rows {
            let i = base + r;
            let row = w.rows[i];

            for (col, v) in [
                (lane.w_x, row[0]),
                (lane.w_y, row[1]),
                (lane.w_z, row[2]),
                (lane.w_i, row[3]),
                (lane.w_j, row[4]),
                (lane.w_e1, row[5]),
                (lane.w_e2, row[6]),
                (lane.w_real, row[7]),
            ] {
                region.assign_advice(|| "w", col, r, || Value::known(F::from(v)))?;
            }

            // order witnesses (both constraints are gated by real)
            lt_xy_chip.assign(
                region,
                r,
                Value::known(F::from(row[0])),
                Value::known(F::from(row[1])),
            )?;
            lt_yz_chip.assign(
                region,
                r,
                Value::known(F::from(row[1])),
                Value::known(F::from(row[2])),
            )?;

            for (col, v) in [
                (lane.keep2, w.keep2[i]),
                (lane.key2, w.key2[i]),
                (lane.cflag2, w.cflag2[i]),
                (lane.ceff2, w.ceff2[i]),
                (lane.key1, w.key1[i]),
                (lane.pred1, w.pred1[i]),
                (lane.cflag1, w.cflag1[i]),
                (lane.ceff1, w.ceff1[i]),
                (lane.ck1, w.ck1[i]),
            ] {
                region.assign_advice(|| "w roles", col, r, || Value::known(F::from(v)))?;
            }

            // The host bits of the mirror half of (9): exactly one is 1 on a
            // clean role-2 row, and none is on any other row, which is what the
            // "hosted by exactly one lane" gate reads as `sum = ceff2`.
            for (l, h) in lane.host.iter().enumerate() {
                let bit = (w.ceff2[i] == 1 && w.host[i] == l) as u64;
                region.assign_advice(|| "pw host", *h, r, || Value::known(F::from(bit)))?;
            }

            let cell = region.assign_advice(
                || "w_oid",
                lane.oid,
                r,
                || Value::known(F::from(w.oid[i])),
            )?;
            if r == 0 {
                first = Some(cell.clone());
            }
            last = Some(cell);

            iz_oid_chip.assign(region, r, Value::known(F::from(w.oid[i]) - pad))?;
            // The step gate is off at the lane's last live row, so the witness
            // there needs no next row: PAD stands in for one that does not
            // exist. The ordering carries on into the next lane through the seam
            // stage, which reads this lane's last cell and that lane's first.
            let nxt = if r + 1 < live_rows {
                w.oid[i + 1]
            } else {
                PAD_U64
            };
            lt_oid_chip.assign(
                region,
                r,
                Value::known(F::from(w.oid[i])),
                Value::known(F::from(nxt)),
            )?;
        }

        // Lane-local child stage: the sorted view, the group boundaries, the two
        // running sums, the emitted per-key rows and the key-indexed table all
        // live inside this lane's own columns, with the lane's own first/last
        // guards. Nothing crosses a lane boundary here.
        // It is exactly `live_rows` tall, so in the last lane it groups the
        // released capacity and nothing beyond it. `assign_cp_agg` enables its
        // own selectors over the rows it is handed, including the PAD sentinel
        // at `live_rows` its `Rotation::next()` comparisons need, so the stage
        // follows the lane's height with nothing to do here.
        let cp_rows: Vec<[u64; 3]> = (0..live_rows)
            .map(|r| [w.key2[base + r], w.keep2[base + r], w.ceff2[base + r]])
            .collect();
        assign_cp_agg(region, &lane.cp, &cp_rows, &stages[lane_idx])?;

        // ---- the c probes, then the two root multiplicities ----
        // Counts add across a partition, so this row's two multiplicities are
        // the sums of the c fetched pairs. Every fetch is certified by that
        // lane's membership lookup or by its gap bracket, which is what makes
        // the sums statements about the WHOLE of W.
        let keys: Vec<u64> = (0..live_rows).map(|r| w.key1[base + r]).collect();
        let mut s_all = vec![0u64; live_rows];
        let mut s_cln = vec![0u64; live_rows];
        assert_eq!(
            lane.probes.len(),
            stages.len(),
            "every lane probes every lane's child table, so the two must have the \
             same length or a lane's rows would go uncounted"
        );
        for (p_idx, (probe, stage)) in lane.probes.iter().zip(stages.iter()).enumerate() {
            let drop = tamper == Tamper::DropProbeVal && p_idx > 0;
            let fetched = Self::assign_probe(region, probe, &keys, stage, drop)?;
            for r in 0..live_rows {
                s_all[r] += fetched[r].0;
                s_cln[r] += fetched[r].1;
            }
        }

        let mut acc_all: u64 = 0;
        let mut acc_cln: u64 = 0;
        let mut sum_all_cell: Option<AssignedCell<F, F>> = None;
        let mut sum_cln_cell: Option<AssignedCell<F, F>> = None;

        for r in 0..live_rows {
            let i = base + r;
            let mu_all = w.pred1[i] * s_all[r];
            let mu_cln = w.ceff1[i] * s_cln[r];

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

        // The two ends the seam stage reads and the drain into the totals stage
        // are the cells at rows 0 and `live_rows - 1`, which in the last lane is
        // now the last row of the RELEASED capacity rather than the last row of
        // the lane's full height.
        Ok(WLaneOut {
            oid_ends: (
                first.expect("live_rows > 0 guarantees a first row"),
                last.expect("live_rows > 0 guarantees a last row"),
            ),
            sum_all: sum_all_cell.expect("live_rows > 0"),
            sum_cln: sum_cln_cell.expect("live_rows > 0"),
            total_all: acc_all,
            total_cln: acc_cln,
        })
    }
    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        edges: &[Edge],
        pad_extra: usize,
        lane_rows: usize,
        released_capacity: Option<usize>,
        tamper: Tamper,
    ) -> Result<AssignedCell<F, F>, Error> {
        let cfg = self.cfg.clone();
        let c = cfg.lanes.len();
        assert!(lane_rows > 0, "lane_rows must be positive");

        // -------------------
        // The capacity the LAYOUT is cut to. Structural input, settled here,
        // before a single row of witness is derived: selectors are fixed
        // columns, so this number is committed at keygen and lands in the
        // verifying key. `Some(cap)` is the caller's public release and the
        // last lane stops at it; `None` fills every lane, which discloses only
        // the lane count. See the module header for which regime may use which.
        // -------------------
        let cap = released_capacity.unwrap_or(c * lane_rows);
        // The live rows must TILE the lanes exactly. `lane_live_rows` re-checks
        // it for every lane; calling it here first makes a capacity that does
        // not fit the geometry fail on the caller's own numbers rather than
        // deep inside the region.
        let live = |l: usize| lane_live_rows(l, c, lane_rows, cap);
        let live_last = live(c - 1);

        let in_view_chip = IndexedViewChip::<F>::construct(cfg.in_by_dst.clone());
        let out_view_chip = IndexedViewChip::<F>::construct(cfg.out_by_src.clone());

        // Load all LT tables used. The two shared views keep a u8 column each;
        // every lane's two comparators, every child stage's comparators, the
        // probes and the occurrence ordering share `lane_u8`, so this is 3 load
        // regions for any lane count.
        in_view_chip.load(layouter)?;
        out_view_chip.load(layouter)?;
        debug_assert_eq!(cfg.lanes[0].lt_xy.u8, cfg.lane_u8);
        debug_assert_eq!(cfg.lanes[0].cp.lt_key_cur_next.u8, cfg.lane_u8);
        LtChip::<F, NUM_BYTES>::construct(cfg.lanes[0].lt_xy).load(layouter)?;

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
                // Capacity. ONE released capacity, because the two readings are
                // one relation and `bench_queries::graph_pads` releases its
                // cardinality once.
                // -------------------
                let real = derived.w.len();
                // The capacity this WITNESS implies. It is checked against the
                // structural one and never used in its place, which is the
                // whole point: `real` is the private statistic.
                //
                //   * `Some(cap)`: exact equality. A drift either drops real
                //     rows past the release (unsound: rows outside the last
                //     lane's live range are not assigned and not gated) or pins
                //     a capacity the release never announced (a disclosure).
                //     Either way it is a panic here, not a silent layout.
                //   * `None`: the layout runs to the full lane span, so the
                //     implied capacity only has to FIT it.
                let implied = std::cmp::max(real + pad_extra, 1);
                match released_capacity {
                    Some(released) => assert_eq!(
                        implied, released,
                        "this witness implies a capacity of {} rows ({} real + {} pad) but \
                         the caller released {}",
                        implied, real, pad_extra, released
                    ),
                    None => assert!(
                        c * lane_rows >= implied,
                        "released capacity {} exceeds {} lanes of {} rows",
                        implied,
                        c,
                        lane_rows
                    ),
                }

                // -------------------
                // W over the whole released capacity, in the GLOBAL
                // occurrence-id order the distinctness argument requires. Lane l
                // is the block [l*lane_rows, l*lane_rows + live(l)) of every
                // vector below, and the c blocks tile [0, cap) exactly.
                // -------------------
                let mut rows = w_rows(&derived.w, cap);

                // Test hook: copy the FIRST kept occurrence over the FIRST row
                // of the LAST lane. Both copies pass both membership lookups and
                // both channels of (10) move by the same amount, so the only
                // thing that can see the duplicate is the occurrence ordering.
                // The copy goes to the first row of a later lane on purpose:
                // every lane's own block stays sorted, so a PER-LANE ordering
                // would accept this witness and only the seam gate rejects it.
                if tamper == Tamper::DuplicateOccurrenceAcrossLanes {
                    assert!(c >= 2, "the cross-lane duplicate needs at least two lanes");
                    let src = rows[0];
                    assert_eq!(role2_keep(&src), 1, "the duplicated row must be kept");
                    rows[(c - 1) * lane_rows] = src;
                }

                // ---- the ROLE-2 reading, (C,D,A) = (x,y,z) ----
                // The keep bit is W's own x<y filter, the aggregate key is
                // pack2(A,C) = pack2(z,x), and the occurrence id rides on the
                // keep bit.
                let keep2: Vec<u64> = rows.iter().map(role2_keep).collect();
                let key2: Vec<u64> = (0..cap)
                    .map(|i| {
                        if keep2[i] == 1 {
                            pack2(rows[i][2], rows[i][0])
                        } else {
                            PAD_U64
                        }
                    })
                    .collect();
                let oid: Vec<u64> = (0..cap).map(|i| bag_oid(&rows[i], keep2[i])).collect();

                // ---- the ROLE-1 reading, (A,B,C) = (x,y,z) ----
                // The probe key is pack2(A,C) = pack2(x,z), REVERSED against the
                // aggregate key, and pinned on every row; a padding row's tuple
                // is all zero, so its key is 0, the dummy row of every child
                // table.
                let pred1: Vec<u64> = rows.iter().map(role1_pred).collect();
                let key1: Vec<u64> = (0..cap).map(|i| pack2(rows[i][0], rows[i][2])).collect();

                // -------------------
                // Condition (7): the prover's partition of each ROLE. The bag
                // tree has two nodes, so the semijoin reduction is exact: a
                // role-1 tuple is clean iff its predicate holds and its
                // separator key carries at least one kept role-2 tuple, and a
                // role-2 tuple is clean iff it is kept and its key is the key of
                // a clean role-1 tuple. The per-key counts are taken over the
                // WHOLE of W, i.e. over all lanes, which is what makes the clean
                // side a property of the union.
                // -------------------
                let mut global2: BTreeMap<u64, u64> = BTreeMap::new();
                for i in 0..cap {
                    if keep2[i] == 1 {
                        *global2.entry(key2[i]).or_default() += 1;
                    }
                }

                let mut cln1: Vec<u64> = (0..cap)
                    .map(|i| (pred1[i] == 1 && *global2.get(&key1[i]).unwrap_or(&0) > 0) as u64)
                    .collect();
                if tamper == Tamper::HideOneCleanTuple {
                    let h = (0..cap)
                        .find(|&i| cln1[i] == 1)
                        .expect("the slice has no clean role-1 tuple to hide");
                    cln1[h] = 0;
                }
                if tamper == Tamper::MarkAllClean {
                    cln1 = pred1.clone();
                }

                let clean_keys1: HashSet<u64> = (0..cap)
                    .filter(|&i| cln1[i] == 1)
                    .map(|i| key1[i])
                    .collect();
                let cln2: Vec<u64> = if tamper == Tamper::MarkAllClean {
                    keep2.clone()
                } else {
                    (0..cap)
                        .map(|i| (keep2[i] == 1 && clean_keys1.contains(&key2[i])) as u64)
                        .collect()
                };

                let ceff1: Vec<u64> = (0..cap).map(|i| pred1[i] * cln1[i]).collect();
                let ceff2: Vec<u64> = (0..cap).map(|i| keep2[i] * cln2[i]).collect();
                let ck1: Vec<u64> = (0..cap).map(|i| ceff1[i] * key1[i]).collect();

                // ---- condition (9), the mirror direction: name the host lane
                // of every clean role-2 row ----
                // Lane l hosts the key iff it holds a clean role-1 row with that
                // key. By construction of `cln2` such a lane exists for every
                // clean role-2 row, so the honest witness always names one. Under
                // `MarkAllClean` it may not, and the fallback below names lane 0
                // so that the LOOKUP is what rejects (rather than the
                // "hosted by exactly one lane" gate, which would hide which
                // statement actually failed).
                let clean1_by_lane: Vec<HashSet<u64>> = (0..c)
                    .map(|l| {
                        let base = l * lane_rows;
                        (base..(base + live(l)))
                            .filter(|&i| ceff1[i] == 1)
                            .map(|i| key1[i])
                            .collect()
                    })
                    .collect();
                let mut host: Vec<usize> = (0..cap)
                    .map(|i| {
                        if ceff2[i] != 1 {
                            return 0;
                        }
                        (0..c)
                            .find(|&l| clean1_by_lane[l].contains(&key2[i]))
                            .unwrap_or(0)
                    })
                    .collect();
                // Test hook: name a lane that does not hold the key. Exactly one
                // bit is still set, so only the mirror lookup can see it.
                if tamper == Tamper::MisnameHostLane {
                    // At c == 1 the single lane necessarily holds every clean
                    // role-1 key, so there is no wrong lane to name. The mirror
                    // constraint is still live there: `Tamper::MarkAllClean`
                    // fires it at c == 1. Same precondition as
                    // `Tamper::DuplicateOccurrenceAcrossLanes`.
                    assert!(c >= 2, "misnaming a host lane needs at least two lanes");
                    let (i, wrong) = (0..cap)
                        .filter(|&i| ceff2[i] == 1)
                        .find_map(|i| {
                            (0..c)
                                .find(|&l| !clean1_by_lane[l].contains(&key2[i]))
                                .map(|l| (i, l))
                        })
                        .expect("every lane holds every clean key: nothing to misname");
                    host[i] = wrong;
                }

                // Cell-level test hooks, applied AFTER the per-row values are
                // derived, so exactly the intended constraint is the one that
                // breaks. Both hit the LAST row of the released capacity, which
                // is the last live row of the last lane and one row of ONE
                // column group now, so they differ in WHICH cell they break: `x`
                // is packed into the role-1 key, `y` is only ever read by the
                // two membership lookups.
                if tamper == Tamper::Bag1PadRow {
                    rows[cap - 1][0] = 7;
                }
                if tamper == Tamper::Bag2PadRow {
                    rows[cap - 1][1] = 7;
                }

                let w = WWitness {
                    rows,
                    keep2,
                    key2,
                    cflag2: cln2,
                    ceff2,
                    key1,
                    pred1,
                    cflag1: cln1,
                    ceff1,
                    ck1,
                    oid,
                    host,
                };

                // -------------------
                // Two selector sets, one row range each, then the c lanes.
                // `full` covers all `lane_rows` rows and fires for every lane but
                // the last; `last` covers the live rows of the last lane, which
                // stop at the RELEASED capacity. Both ranges are functions of the
                // capacity, `c` and `lane_rows`, all public.
                //
                // `q_w_lookup` and `q_w_row` MUST cover the same rows, in EACH
                // set. `q_w_row` is what pins `ck1 = ceff1 * key1`, and
                // `q_w_lookup` is what gates that column as the table side of the
                // mirror half of (9): a row inside the lookup's table but outside
                // the pinning gate would be free advice minting a clean key. They
                // are enabled together here so the coupling is visible in one
                // place, and the mirror lookup takes its table gate from the lane
                // that OWNS the column so the two cannot drift apart.
                //
                // Row 0 is live in every lane, so `q_sum0` stays shared and is
                // enabled once, outside both loops. The probe selectors are not
                // here: `assign_probe` enables them over the rows of the lane it
                // is assigning, through that lane's own probe config.
                //
                // `live_last` is the one computed at the top of `assign`, from
                // the structural capacity alone.
                // -------------------
                cfg.full.q_sum0.enable(&mut region, 0)?;
                if c > 1 {
                    for r in 0..lane_rows {
                        cfg.full.q_w_lookup.enable(&mut region, r)?;
                        cfg.full.q_w_row.enable(&mut region, r)?;
                        cfg.full.q_mu.enable(&mut region, r)?;
                        if r > 0 {
                            cfg.full.q_sum.enable(&mut region, r)?;
                        }
                        // the within-lane step of the occurrence ordering
                        if r + 1 < lane_rows {
                            cfg.full.q_oid_step.enable(&mut region, r)?;
                        }
                    }
                }
                for r in 0..live_last {
                    cfg.last.q_w_lookup.enable(&mut region, r)?;
                    cfg.last.q_w_row.enable(&mut region, r)?;
                    cfg.last.q_mu.enable(&mut region, r)?;
                    if r > 0 {
                        cfg.last.q_sum.enable(&mut region, r)?;
                    }
                    if r + 1 < live_last {
                        cfg.last.q_oid_step.enable(&mut region, r)?;
                    }
                }

                // The c child tables, built before any lane is assigned: every
                // lane's probes read every one of them. Each one covers its own
                // lane's live rows, so together they still partition [0, cap).
                let mut stages: Vec<CpStage> = Vec::with_capacity(c);
                for l in 0..c {
                    let base = l * lane_rows;
                    let cp_rows: Vec<[u64; 3]> = (0..live(l))
                        .map(|r| [w.key2[base + r], w.keep2[base + r], w.ceff2[base + r]])
                        .collect();
                    stages.push(build_cp_stage(&cp_rows, PAD_U64));
                }
                debug_assert_eq!(
                    stages
                        .iter()
                        .map(|s| s.map.values().map(|v| v.0).sum::<u64>())
                        .sum::<u64>(),
                    w.keep2.iter().sum::<u64>(),
                    "the lane-local child tables must partition the kept role-2 multiset"
                );

                let mut lanes_out: Vec<WLaneOut<F>> = Vec::with_capacity(c);
                for (l, lane) in cfg.lanes.iter().enumerate() {
                    lanes_out.push(Self::assign_w_lane(
                        lane,
                        &mut region,
                        l,
                        live(l),
                        l * lane_rows,
                        &w,
                        &stages,
                        tamper,
                    )?);
                }

                // -------------------
                // The ONE lane seam of the occurrence ordering. Sorting W by the
                // role-2 occurrence id put every real row before every pad row,
                // so this single stage carries the ordering of both readings.
                // -------------------
                let oid_ends: Vec<(AssignedCell<F, F>, AssignedCell<F, F>)> = lanes_out
                    .iter()
                    .map(|o| o.oid_ends.clone())
                    .collect();
                let oid_vals: Vec<(u64, u64)> = (0..c)
                    .map(|l| {
                        let base = l * lane_rows;
                        (w.oid[base], w.oid[base + live(l) - 1])
                    })
                    .collect();
                Self::assign_oid_boundary(&mut region, &cfg.oid_bnd, &oid_ends, &oid_vals)?;

                // -------------------
                // The cross-lane totals stage: c pairs of copy constraints, two
                // degree-2 accumulators, ONE equality between the global totals
                // (condition (10)) and `out` for the instance.
                // -------------------
                let mut acc_all: u64 = 0;
                let mut acc_cln: u64 = 0;
                for l in 0..c {
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

                let out_row = c - 1;
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
/// Wrapper circuit. `lane_rows`, `num_lanes` and `released_capacity` are
/// circuit STRUCTURE: they decide which rows are assigned and gated, so they
/// land in the verifying key and none of them may be derived from the witness.
pub struct MyCircuit<F: Field + Ord> {
    pub edges: Vec<Edge>,
    pub pad_extra: usize,
    pub lane_rows: usize,
    pub num_lanes: usize,
    /// The RELEASED capacity of W, when the caller is supplying a genuine
    /// public release.
    ///
    ///   * `Some(cap)`: the last lane stops at `cap`, so the assigned rows
    ///     total `cap` rather than `num_lanes * lane_rows`, and `cap` is what
    ///     the verifying key pins. Only pass it where `cap` is public already:
    ///     `Privacy::Dp` (the DP release) and `Privacy::Rjs` (the true size,
    ///     which that regime reveals by definition).
    ///   * `None`: every lane is filled to `lane_rows`, exactly as before the
    ///     short last lane existed, and the vk pins only the lane count. This
    ///     is the behaviour-preserving default and the one `Privacy::Legacy`
    ///     must use, since its pad is a public constant.
    ///
    /// The check that the witness really implies this capacity is in `assign`.
    pub released_capacity: Option<usize>,
    pub tamper: Tamper,
    pub _marker: PhantomData<F>,
}

impl<F: Field + Ord> Default for MyCircuit<F> {
    fn default() -> Self {
        Self {
            edges: vec![],
            pad_extra: 0,
            lane_rows: LANE_ROWS,
            num_lanes: 1,
            // behaviour-preserving: full lanes, lane count only in the vk
            released_capacity: None,
            tamper: Tamper::None,
            _marker: PhantomData,
        }
    }
}

impl<F: Field + Ord> Circuit<F> for MyCircuit<F> {
    type Config = Gq4DpConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        // Keep the whole lane geometry: it is circuit STRUCTURE, not witness,
        // and dropping the capacity here is what broke keygen at c > 1 (the
        // layout collapsed to a single row, which no lane count but 1 tiles).
        //
        // `pad_extra` is set to the release rather than to 0 on purpose. W
        // itself is gone, so the honest reading of this circuit is an EMPTY
        // relation padded to the whole released capacity: 0 real rows plus
        // `cap` pad rows. That is what the drift check in `assign` compares
        // against, so it stays exact instead of having to special-case a
        // witness-free circuit.
        Self {
            lane_rows: self.lane_rows,
            num_lanes: self.num_lanes,
            released_capacity: self.released_capacity,
            pad_extra: self.released_capacity.unwrap_or(0),
            ..Self::default()
        }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        Gq4DpChip::<F>::configure(meta)
    }

    fn synthesize(&self, cfg: Self::Config, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        assert_eq!(
            cfg.lanes.len(),
            self.num_lanes,
            "configured lane count does not match the circuit: call \
             set_config_lanes(num_lanes) before keygen / MockProver"
        );
        let chip = Gq4DpChip::<F>::construct(cfg);
        let out = chip.assign(
            &mut layouter,
            &self.edges,
            self.pad_extra,
            self.lane_rows,
            self.released_capacity,
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

/// Geometry the released capacity implies, plus the edges it was derived from.
/// No SRS is read and no key is built here.
struct Gq4DpSetup {
    edges: Vec<Edge>,
    pad_extra: usize,
    /// The capacity this regime may pin in the verifying key, or `None` where
    /// it may not. See [`crate::dp_lane::vk_released_capacity`].
    released_capacity: Option<usize>,
    plan: crate::dp_lane::DpLanePlan,
}

/// Geometry the released capacity implies, or an explanation of why this
/// configuration cannot be built.
///
/// `Err` covers every way a row can be rejected BEFORE proving: the lane cap,
/// and the structural fit of the tallest column group at the pinned degree.
/// Both are properties of the request, so a sweep reports them and continues.
fn try_dp_lane_setup(
    dataset: &str,
    privacy: crate::bench_queries::Privacy,
) -> Result<Gq4DpSetup, String> {
    let edges = crate::bench_queries::load_graph(dataset);
    // ONE released capacity: the two column roles are one relation, so their
    // cardinality is one statistic and `graph_pads` returns it twice (the pair
    // shape is what GQ3 and Q5 need). Reading `.0` is what `bench_queries`
    // itself does for gq4.
    let pad_extra = crate::bench_queries::graph_pads("gq4", dataset, &edges, privacy).0;

    // The degree is PINNED at the Revealing-Join-Size value; the DP release is
    // absorbed by lanes, not by a bigger domain.
    let k = crate::bench_queries::degree_for("gq4", dataset, crate::bench_queries::Privacy::Rjs);
    let lane_rows = lane_rows_for(k);

    let stats = crate::bench_queries::bag_stats("gq4", &edges);
    let n = stats.bag1_size as usize + pad_extra;
    let c = try_lanes_for_capacity(n, k)?;

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
    //
    // The cross-lane stages are the other two candidates and both are tiny: the
    // totals stage is c rows and the ONE seam stage is c-1, so `max(lane_rows,
    // c)` covers them. The second seam term the two-bag version omitted is now
    // genuinely absent.
    let tallest = (edges.len() + 2).max(lane_rows).max(c);
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
        lanes: vec![c],
        capacity: vec![n],
        true_size: vec![stats.bag1_size as usize],
        pads: vec![pad_extra],
    };
    // POLICY, applied here and nowhere else in this file: whether the circuit
    // may stop its last lane at the released capacity, which is the same thing
    // as whether it may pin that capacity in the verifying key.
    //   * Privacy::Dp  -> Some(n). The release is public, so the vk discloses
    //                     nothing the harness did not already print.
    //   * Privacy::Rjs -> Some(n), which equals the true bag size. That regime
    //                     reveals the join size by definition.
    //   * Privacy::Legacy -> None. Its pad is a public CONSTANT, so a capacity
    //                     in the vk would pin the true bag size; fall back to
    //                     the full-lane layout, which discloses only the lane
    //                     count.
    let released_capacity = crate::dp_lane::vk_released_capacity(privacy, n);

    Ok(Gq4DpSetup {
        edges,
        pad_extra,
        released_capacity,
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
    let cnt = crate::bench_queries::count_gq4(&setup.edges);

    set_config_lanes(c);
    let build = || MyCircuit::<Fp> {
        edges: setup.edges.clone(),
        pad_extra: setup.pad_extra,
        lane_rows: setup.plan.lane_rows,
        num_lanes: c,
        // `Some` only where the capacity is a genuine public release; see the
        // policy comment in `try_dp_lane_setup`.
        released_capacity: setup.released_capacity,
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
        lane_live_rows, lane_rows_for, lanes_for, lanes_for_capacity, role1_pred, role2_keep,
        set_config_lanes, w_rows, MyCircuit, Tamper, BASE_DEGREE, LANE_ROWS, MAX_LANES,
    };

    use crate::data::graph_data_processing::Edge;
    use halo2_proofs::dev::{MockProver, VerifyFailure};
    use halo2_proofs::plonk::{keygen_vk, Circuit};
    use halo2curves::pasta::Fp;

    use std::marker::PhantomData;
    use std::time::Instant;

    /// Twelve directed edges over eight nodes: 0->1->3->d->0 for each
    /// d in {4,5,6,7,8}, so there are exactly FIVE ordered 4-cycles, all
    /// sharing the separator pair (A,C) = (0,3).
    ///
    /// The shape is chosen so the message cannot live in one lane. W
    /// materializes 11 rows, and the single separator key that carries the answer
    /// has multiplicity 5 in the ROLE-2 reading. With `LANE_ROWS_SMALL` = 4 those
    /// five rows cannot all land in one lane, so `mock_three_lanes` only reaches
    /// the right answer if the role-1 probes really do SUM the values retrieved
    /// from every lane's child table. See `heavy_key_spans_multiple_bag2_lanes`,
    /// which pins that down.
    ///
    /// The shape also has DANGLING tuples on both ends of the bag tree edge --
    /// 5 predicate-passing role-1 rows whose key occurs in no kept role-2 tuple,
    /// and 6 kept role-2 rows whose key occurs in no predicate-passing role-1
    /// row -- which is what makes `mock_reject_mark_all_clean` non-vacuous and
    /// what the mirror half of condition (9) has to accept on the honest
    /// witness.
    fn synthetic_edges() -> Vec<Edge> {
        let mut edges = vec![Edge { src: 0, dst: 1 }, Edge { src: 1, dst: 3 }];
        for d in 4..9u64 {
            edges.push(Edge { src: 3, dst: d });
            edges.push(Edge { src: d, dst: 0 });
        }
        edges
    }

    /// True size of W on [`synthetic_edges`].
    const SYNTH_BAG_ROWS: usize = 11;
    /// Rows per lane in the small tests.
    const LANE_ROWS_SMALL: usize = 4;
    /// COUNT(*) on [`synthetic_edges`].
    const SYNTH_CNT: u64 = 5;

    /// W holds 11 rows, so 1 pad row gives a released capacity of 12, which at 4
    /// rows per lane spans exactly 3 lanes and leaves the last row of the
    /// pipeline as a rounding-up pad.
    fn three_lane_circuit(tamper: Tamper) -> (MyCircuit<Fp>, usize) {
        let edges = synthetic_edges();
        let lane_rows = LANE_ROWS_SMALL;
        let pad_extra = 1;
        let c = lanes_for(SYNTH_BAG_ROWS + pad_extra, lane_rows);
        assert_eq!(c, 3);

        let circuit = MyCircuit::<Fp> {
            edges,
            pad_extra,
            lane_rows,
            num_lanes: c,
            released_capacity: released(pad_extra),
            tamper,
            _marker: PhantomData,
        };
        (circuit, c)
    }

    /// The released capacity a pad implies here, handed to the circuit as a
    /// GENUINE release. This is the `Privacy::Dp` / `Privacy::Rjs` arm: the
    /// number is public, so the last lane may stop at it and the vk may pin it.
    /// The `None` arm (`Privacy::Legacy`) is exercised by
    /// `mock_some_and_none_at_the_same_geometry` and
    /// `full_lanes_keep_the_capacity_out_of_the_key`.
    fn released(pad_extra: usize) -> Option<usize> {
        Some(SYNTH_BAG_ROWS + pad_extra)
    }

    /// Runs the three-lane circuit under MockProver and returns the failures.
    fn three_lane_failures(tamper: Tamper) -> Vec<VerifyFailure> {
        let (circuit, c) = three_lane_circuit(tamper);
        set_config_lanes(c);
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

    /// The separator key that carries the whole answer has multiplicity 5 in the
    /// ROLE-2 reading while a lane only holds 4 rows, so its count is
    /// necessarily split over at least two lanes. This is what makes
    /// `mock_three_lanes` a real test of the cross-lane message sum rather than
    /// of a single table.
    ///
    /// The lane a row falls into is decided by the GLOBAL occurrence-id order the
    /// distinctness argument requires, so this test asks `w_rows` for the layout
    /// instead of assuming the derivation order.
    #[test]
    fn heavy_key_spans_multiple_bag2_lanes() {
        use std::collections::BTreeMap;

        let edges = synthetic_edges();
        let derived = crate::graph_sql::g_sql4_obj::gq4_derive(&edges);
        // ONE materialized relation, read in two column roles, so one length
        // assertion covers both readings.
        assert_eq!(derived.w.len(), SYNTH_BAG_ROWS);

        // the heaviest message key, and the lanes its role-2 rows fall into
        let (&heavy, &mult) = derived
            .msg_map
            .iter()
            .max_by_key(|(_, v)| **v)
            .expect("non-empty message map");
        assert_eq!(mult, 5, "the five 4-cycles all share one separator key");

        let capacity = 3 * LANE_ROWS_SMALL;
        let rows = w_rows(&derived.w, capacity);
        let mut per_lane: BTreeMap<usize, u64> = BTreeMap::new();
        for (i, r) in rows.iter().enumerate() {
            if role2_keep(r) == 1 && crate::graph_sql::g_sql4_obj::pack2(r[2], r[0]) == heavy {
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

        // Both readings are the same `w` rows: role 1 reads them as (A,B,C) and
        // needs the extra [B<C], role 2 reads them as (C,D,A) and needs nothing
        // extra since `w` is pre-filtered to x<y.
        let kept2_keys: HashSet<u64> = derived
            .w
            .iter()
            .map(|&(x, _y, z, ..)| pack2(z, x))
            .collect();
        let pred1_keys: HashSet<u64> = derived
            .w
            .iter()
            .filter(|&&(_x, y, z, ..)| y < z)
            .map(|&(x, _y, z, ..)| pack2(x, z))
            .collect();
        let dangling1 = derived
            .w
            .iter()
            .filter(|&&(x, y, z, ..)| y < z && !kept2_keys.contains(&pack2(x, z)))
            .count();
        let dangling2 = derived
            .w
            .iter()
            .filter(|&&(x, _y, z, ..)| !pred1_keys.contains(&pack2(z, x)))
            .count();
        println!(
            "[gq4 dp] dangling: role1={} role2={}",
            dangling1, dangling2
        );
        assert!(
            dangling1 > 0,
            "no dangling role-1 tuple: the all-clean direction would be vacuous"
        );
    }

    /// Every kept row must have a DISTINCT occurrence id, or the honest witness
    /// could not satisfy the global ordering in the first place.
    /// `pack2(e1, e2)` is a function of the occurrence, so this is a property of
    /// the derivation, checked here so a failure points at the derivation rather
    /// than at a gate.
    ///
    /// ONE ordering covers both roles, so this checks the ROLE-2 keep bit, which
    /// is W's own x<y filter and therefore holds on every real row; the role-1
    /// ids are a subset of these.
    #[test]
    fn occurrence_ids_are_distinct() {
        use std::collections::HashSet;

        let edges = synthetic_edges();
        let derived = crate::graph_sql::g_sql4_obj::gq4_derive(&edges);

        let ids2: Vec<u64> = derived
            .w
            .iter()
            .map(|&(x, y, z, _i, _j, e1, e2)| [x, y, z, 0, 0, e1, e2, 1])
            .filter(|r| role2_keep(r) == 1)
            .map(|r| crate::graph_sql::g_sql4_obj::pack2(r[5], r[6]))
            .collect();
        assert_eq!(ids2.len(), derived.w.len(), "W is pre-filtered to x<y");
        assert_eq!(
            ids2.len(),
            ids2.iter().copied().collect::<HashSet<u64>>().len(),
            "two role-2 occurrences share an id"
        );

        let ids1: Vec<u64> = derived
            .w
            .iter()
            .map(|&(x, y, z, _i, _j, e1, e2)| [x, y, z, 0, 0, e1, e2, 1])
            .filter(|r| role1_pred(r) == 1)
            .map(|r| crate::graph_sql::g_sql4_obj::pack2(r[5], r[6]))
            .collect();
        assert!(
            ids1.len() < ids2.len(),
            "role 1 must be a proper subset here, or the slice is degenerate"
        );
        assert_eq!(
            ids1.len(),
            ids1.iter().copied().collect::<HashSet<u64>>().len(),
            "two role-1 occurrences share an id"
        );
    }

    /// Cost and degree probe, permanent. `cs.degree()` drives the FFT size of
    /// every polynomial in the proof, so a rise here costs far more than any
    /// individual gate saves. The bound is the one `g_sql4_obj` already pays: a
    /// membership lookup with a degree-3 input and a degree-2 table side, i.e.
    /// 2 + 3 + 2 = 7.
    ///
    /// `P` is the crude column-work proxy the COST NOTE quotes,
    /// advice + 3*lookups + shuffles.
    #[test]
    fn test_max_gate_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        for c in [1usize, 2, 3, 4] {
            set_config_lanes(c);
            let mut cs = ConstraintSystem::<Fp>::default();
            let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
            let p = cs.num_advice_columns() + 3 * cs.lookups().len() + cs.shuffles().len();
            println!("c={} cs.degree() = {}", c, cs.degree());
            println!(
                "  advice={} fixed={} instance={} selectors={} gates={} lookups={} \
                 shuffles={} perm_cols={} P={}",
                cs.num_advice_columns(),
                cs.num_fixed_columns(),
                cs.num_instance_columns(),
                cs.num_selectors(),
                cs.gates().len(),
                cs.lookups().len(),
                cs.shuffles().len(),
                cs.permutation().get_columns().len(),
                p,
            );
            assert!(
                cs.degree() <= 7,
                "c={}: the maximum gate degree rose to {}, which doubles every FFT",
                c,
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

        // The releases the default VPJOIN_DP_SEED produces, as `graph_pads`
        // reports them for ONE capacity release (the two column roles are one
        // private cardinality). These assertions are arithmetic on the released
        // numbers, so a seed change moves the inputs, not the property.
        //
        // lastfm at eps=0.1 needs 2 lanes at k=18 (it would otherwise need
        // k=19); the two-bag version quoted 788,220 and 621,070 from a 3-way
        // budget split and got 4 and 3.
        assert_eq!(lanes_for_capacity(512_521, 18), 2);
        assert_eq!(lanes_for_capacity(788_220, 18), 4);
        // facebook and wiki collapse to a single lane at k=22
        assert_eq!(lanes_for_capacity(3_284_609, 22), 1);
        assert_eq!(lanes_for_capacity(2_767_286, 22), 1);
        // at eps=0.01 every dataset stays under the structural cap, lastfm only
        // just: 63 lanes is 3,969 probe replicas, which the COST NOTE says is
        // past the point where lanes are the right tool.
        assert_eq!(lanes_for_capacity(16_336_389, 18), 63);
        assert!(lanes_for_capacity(16_336_389, 18) <= MAX_LANES);
        assert_eq!(lanes_for_capacity(18_253_740, 22), 5);
        assert_eq!(lanes_for_capacity(17_190_338, 22), 5);
    }

    /// The live rows of the c lanes must TILE the released capacity: every lane
    /// but the last full, the last stopping exactly at the capacity, and the sum
    /// equal to it. Anything else either drops a released row or leaves lane
    /// quantization behind.
    #[test]
    fn live_rows_tile_the_capacity() {
        for lane_rows in [1usize, 2, 3, 5, 11, 260_800] {
            for capacity in [1usize, 2, 11, 12, 14, 16, 512_521] {
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

        // `released_capacity = None` passes `c * lane_rows`, which must give
        // every lane back its full height whatever the lane count is
        for c in [1usize, 2, 7] {
            for l in 0..c {
                assert_eq!(lane_live_rows(l, c, 4, c * 4), 4);
            }
        }
    }

    /// `keygen_vk` at the degree the mock tests use, as its PINNED
    /// representation. `PinnedVerificationKey` carries the domain, the whole
    /// constraint system, the permutation and every fixed commitment, i.e.
    /// everything the verifier holds, and its `Debug` string is exactly what
    /// halo2 hashes into the key's `transcript_repr`. Comparing those strings
    /// is comparing the keys; this fork exposes no cheaper equality.
    fn pinned_vk(circuit: &MyCircuit<Fp>) -> String {
        use halo2_proofs::poly::commitment::ParamsProver;
        use halo2_proofs::poly::ipa::commitment::ParamsIPA;
        use halo2curves::pasta::EqAffine;

        let params: ParamsIPA<EqAffine> = ParamsIPA::new(11);
        let vk = keygen_vk(&params, circuit).expect("keygen_vk must not fail");
        format!("{:?}", vk.pinned())
    }

    /// DEFECT 2. halo2's structural guarantee is that a verifying key cannot
    /// depend on witness data, and `without_witnesses()` is where a circuit
    /// states it: keygen has to go through on a circuit carrying no W at all.
    /// A capacity reconstructed from the witness collapsed to `max(0 + 0, 1)`
    /// there, which no lane count but 1 tiles, so this panicked at every c > 1.
    ///
    /// The second half is the property worth protecting: with the capacity kept
    /// as a structural field, the witness-free key is a function of
    /// `(lane_rows, num_lanes, released_capacity)` and of nothing else, so two
    /// completely different graphs give the SAME key.
    ///
    /// WHAT THIS CANNOT ASSERT, AND WHY. The witness-free key is NOT equal to
    /// the key of the witness-bearing circuit, and that gap is not the
    /// capacity's doing: it is exactly as wide with `released_capacity = None`,
    /// where the field plays no part in any row range at all. It comes from the
    /// single-copy stages, whose height tracks |E| (the base Edge relation and
    /// its two indexed views), plus halo2's selector compression, which folds
    /// selectors into fixed columns using the enable pattern it observes: an
    /// empty edge list changes `num_fixed_columns` and the compressed gate
    /// expressions. |E| is public in every regime, which is why those stages
    /// are not laned. `run_dp_lanes` accordingly keygens on the witness-bearing
    /// circuit, as it always has. What the released capacity itself contributes
    /// to the key is isolated in `full_lanes_keep_the_capacity_out_of_the_key`.
    #[test]
    fn without_witnesses_keygens_at_every_lane_count() {
        let other_edges: Vec<Edge> = [(0u64, 1u64), (1, 2), (2, 3), (3, 0), (0, 2)]
            .into_iter()
            .map(|(src, dst)| Edge { src, dst })
            .collect();

        // (lane_rows, pad, expected lanes): one lane, and the three-lane
        // geometry with a short last lane
        for (lane_rows, pad_extra, want_c) in [(16usize, 1usize, 1usize), (5, 3, 3)] {
            let c = lanes_for(SYNTH_BAG_ROWS + pad_extra, lane_rows);
            assert_eq!(c, want_c);
            set_config_lanes(c);

            let circuit = MyCircuit::<Fp> {
                edges: synthetic_edges(),
                pad_extra,
                lane_rows,
                num_lanes: c,
                released_capacity: released(pad_extra),
                tamper: Tamper::None,
                _marker: PhantomData,
            };
            // the defect-2 fix: this is what panicked at c > 1
            let empty = <MyCircuit<Fp> as Circuit<Fp>>::without_witnesses(&circuit);
            assert_eq!(empty.lane_rows, lane_rows);
            assert_eq!(empty.num_lanes, c);
            assert_eq!(empty.released_capacity, released(pad_extra));
            let vk_empty = pinned_vk(&empty);

            // a different graph, the same structural fields: same key
            let other = MyCircuit::<Fp> {
                edges: other_edges.clone(),
                pad_extra,
                lane_rows,
                num_lanes: c,
                released_capacity: released(pad_extra),
                tamper: Tamper::None,
                _marker: PhantomData,
            };
            set_config_lanes(c);
            let vk_other_empty =
                pinned_vk(&<MyCircuit<Fp> as Circuit<Fp>>::without_witnesses(&other));
            // HONEST ABOUT WHAT THIS PROVES. `without_witnesses` discards the
            // edges by construction, so this equality cannot fail and is not
            // evidence that the key is witness-independent. It is a guard on
            // `without_witnesses` ITSELF: if a later edit made it carry any
            // part of the witness through, the two keys would diverge here.
            // The property that the capacity stays out of the key is tested
            // by `full_lanes_keep_the_capacity_out_of_the_key`, which varies
            // a real witness quantity at a fixed lane count.
            assert_eq!(
                vk_empty, vk_other_empty,
                "lane_rows={lane_rows} c={c}: without_witnesses carried witness data through"
            );

            // The same run with `None`: the capacity field is preserved as
            // `None` and keygen still goes through, on the full-lane layout.
            set_config_lanes(c);
            let full = MyCircuit::<Fp> {
                edges: synthetic_edges(),
                pad_extra,
                lane_rows,
                num_lanes: c,
                released_capacity: None,
                tamper: Tamper::None,
                _marker: PhantomData,
            };
            let empty_full = <MyCircuit<Fp> as Circuit<Fp>>::without_witnesses(&full);
            assert_eq!(empty_full.released_capacity, None);
            assert_eq!(empty_full.pad_extra, 0);
            let vk_empty_full = pinned_vk(&empty_full);

            // The gap documented above, pinned so the diagnosis is checkable
            // rather than asserted in prose: it is there under `Some` and under
            // `None` alike, so nothing about it is the capacity's. If a later
            // change ever makes the single-copy stages structural too, both of
            // these become equalities and this is the test to strengthen.
            set_config_lanes(c);
            assert_ne!(pinned_vk(&circuit), vk_empty);
            set_config_lanes(c);
            assert_ne!(pinned_vk(&full), vk_empty_full);
        }
    }

    /// DEFECT 1. Under `Privacy::Legacy` the pad is a public CONSTANT, so a
    /// capacity in the verifying key would pin `true_size = capacity - pad`.
    /// That regime therefore passes `None`, and this is what `None` buys: two
    /// pads that land on the same lane count give BYTE-IDENTICAL keys, so the
    /// key cannot be inverted for the bag size. With `Some` the two keys
    /// differ, which is the point of a genuine release: the capacity is public
    /// and the last lane may stop at it.
    #[test]
    fn full_lanes_keep_the_capacity_out_of_the_key() {
        let lane_rows = 5;
        // 14 and 15 released rows both need exactly 3 lanes of 5
        let (pad_a, pad_b) = (3usize, 4usize);
        let c = lanes_for(SYNTH_BAG_ROWS + pad_a, lane_rows);
        assert_eq!(c, lanes_for(SYNTH_BAG_ROWS + pad_b, lane_rows));
        assert_eq!(c, 3);

        let build = |pad: usize, released_capacity: Option<usize>| MyCircuit::<Fp> {
            edges: synthetic_edges(),
            pad_extra: pad,
            lane_rows,
            num_lanes: c,
            released_capacity,
            tamper: Tamper::None,
            _marker: PhantomData,
        };

        set_config_lanes(c);
        let none_a = pinned_vk(&build(pad_a, None));
        set_config_lanes(c);
        let none_b = pinned_vk(&build(pad_b, None));
        assert_eq!(
            none_a, none_b,
            "with `None` the verifying key must not move with the pad, or it pins \
             the true bag size wherever the pad is a public constant"
        );

        set_config_lanes(c);
        let some_a = pinned_vk(&build(pad_a, released(pad_a)));
        set_config_lanes(c);
        let some_b = pinned_vk(&build(pad_b, released(pad_b)));
        assert_ne!(
            some_a, some_b,
            "a released capacity IS pinned by the key: two different releases must \
             give two different keys"
        );
        assert_ne!(
            some_a, none_a,
            "the short last lane must be visible in the key at all, or the `Some` \
             arm is not doing anything"
        );
        // 15 released rows fill all three lanes, so `Some(15)` and `None` are
        // the same layout and must give the same key.
        set_config_lanes(c);
        assert_eq!(
            pinned_vk(&build(pad_b, Some(c * lane_rows))),
            none_b,
            "a release that exactly fills the lanes is the full-lane layout"
        );
    }

    /// The `Some` / `None` pair at ONE geometry: both must prove, both must
    /// reject a wrong COUNT(*), and they must differ in exactly the way the
    /// field promises, namely how far the last lane runs.
    #[test]
    fn mock_some_and_none_at_the_same_geometry() {
        let lane_rows = 5;
        let pad_extra = 3;
        let c = lanes_for(SYNTH_BAG_ROWS + pad_extra, lane_rows);
        assert_eq!(c, 3);
        // 14 released rows over 3 lanes of 5: the two arms really do differ
        assert_eq!(lane_live_rows(c - 1, c, lane_rows, 14), 4);
        assert_eq!(lane_live_rows(c - 1, c, lane_rows, c * lane_rows), 5);

        for released_capacity in [released(pad_extra), None] {
            let build = |tamper| {
                set_config_lanes(c);
                MyCircuit::<Fp> {
                    edges: synthetic_edges(),
                    pad_extra,
                    lane_rows,
                    num_lanes: c,
                    released_capacity,
                    tamper,
                    _marker: PhantomData,
                }
            };
            let public = vec![vec![Fp::from(SYNTH_CNT)]];

            let prover = MockProver::run(11, &build(Tamper::None), public.clone()).unwrap();
            assert_eq!(
                prover.verify(),
                Ok(()),
                "released_capacity={released_capacity:?} must prove"
            );

            let wrong =
                MockProver::run(11, &build(Tamper::None), vec![vec![Fp::from(SYNTH_CNT + 1)]])
                    .unwrap();
            assert!(
                wrong.verify().is_err(),
                "released_capacity={released_capacity:?} accepted a wrong COUNT(*)"
            );

            // Both pad-row hooks break the LAST live row of the last lane,
            // which is row 13 under `Some(14)` and row 14 under `None`. Both
            // must be inside the gated range, or the arm has an unconstrained
            // tail.
            for tamper in [Tamper::Bag1PadRow, Tamper::Bag2PadRow] {
                let tampered = MockProver::run(11, &build(tamper), public.clone()).unwrap();
                assert!(
                    tampered.verify().is_err(),
                    "released_capacity={released_capacity:?} left its last row \
                     unconstrained under {tamper:?}"
                );
            }
        }
    }

    /// A released capacity that does not TILE the lanes is a caller error and
    /// must abort, not silently re-cut the layout: too many rows would leave
    /// the overflow ungated, too few would leave a whole lane with no live row.
    #[test]
    #[should_panic(expected = "is not hosted by exactly 3 lanes of 5 rows")]
    fn released_capacity_above_the_lane_span_panics() {
        let lane_rows = 5;
        let pad_extra = 5; // 11 + 5 = 16 rows, which needs 4 lanes of 5
        let c = 3;
        set_config_lanes(c);
        let circuit = MyCircuit::<Fp> {
            edges: synthetic_edges(),
            pad_extra,
            lane_rows,
            num_lanes: c,
            released_capacity: released(pad_extra),
            tamper: Tamper::None,
            _marker: PhantomData,
        };
        let _ = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]);
    }

    #[test]
    #[should_panic(expected = "is not hosted by exactly 4 lanes of 5 rows")]
    fn released_capacity_below_the_lane_span_panics() {
        let lane_rows = 5;
        let pad_extra = 3; // 11 + 3 = 14 rows, which 3 lanes of 5 host
        let c = 4;
        set_config_lanes(c);
        let circuit = MyCircuit::<Fp> {
            edges: synthetic_edges(),
            pad_extra,
            lane_rows,
            num_lanes: c,
            released_capacity: released(pad_extra),
            tamper: Tamper::None,
            _marker: PhantomData,
        };
        let _ = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]);
    }

    /// A release that does not match the witness is a panic, not a layout. Too
    /// small drops real rows past the last lane's live range, where nothing
    /// gates them; too large pins a capacity the release never announced.
    #[test]
    #[should_panic(expected = "but the caller released 15")]
    fn released_capacity_that_drifts_from_the_witness_panics() {
        let lane_rows = 5;
        let pad_extra = 3;
        let c = lanes_for(SYNTH_BAG_ROWS + pad_extra, lane_rows);
        set_config_lanes(c);
        let circuit = MyCircuit::<Fp> {
            edges: synthetic_edges(),
            pad_extra,
            lane_rows,
            num_lanes: c,
            // the witness implies 14; claim 15, which still tiles the lanes
            released_capacity: Some(15),
            tamper: Tamper::None,
            _marker: PhantomData,
        };
        let _ = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]);
    }

    /// PRIVACY AUDIT. Every lane must be a FULL structural replica: no lane may
    /// be cheaper than another, or the constraint system would leak where the
    /// real rows stop, which is exactly what the DP release pays to hide. With
    /// one kind of lane and one probe per (row lane, child lane) pair that is
    /// equivalent to the circuit shape being EXACTLY QUADRATIC in c -- constant,
    /// plus a fixed cost per lane, plus a fixed cost per probe pair. Fit that
    /// model on c = 1, 2, 3 and check it off the fit.
    ///
    /// The same fit also pins down the sharing that keeps the preamble bounded:
    /// fixed columns must not grow with c at all (one shared u8 range table),
    /// the shuffle count must be exactly 4 + 2c (the four preamble shuffles,
    /// plus the two a lane's child stage owns), and the quadratic coefficient
    /// must be zero for everything except advice and lookups, since a probe
    /// pair adds no fixed column, no selector, no shuffle and no permutation
    /// column.
    ///
    /// Selectors DO carry a linear term: `assign_cp_agg` enables its own
    /// selectors, so each lane's child stage brings its own. That is uniform
    /// across lanes, which is what the audit is about.
    #[test]
    fn lanes_are_structural_replicas() {
        use halo2_proofs::plonk::ConstraintSystem;

        const N: usize = 7;
        const ADVICE: usize = 0;
        const FIXED: usize = 1;
        const SEL: usize = 2;
        const LOOKUPS: usize = 3;
        const SHUF: usize = 4;
        const PERM: usize = 5;
        const DEGREE: usize = 6;

        fn shape(c: usize) -> [i64; N] {
            set_config_lanes(c);
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

        let s1 = shape(1);
        let s2 = shape(2);
        let s3 = shape(3);

        // shape(c) = a + b*c + q*c^2, from the three cells
        let mut a = [0i64; N];
        let mut b = [0i64; N];
        let mut q = [0i64; N];
        for i in 0..N {
            // second difference is 2q
            assert_eq!(
                (s3[i] - 2 * s2[i] + s1[i]) % 2,
                0,
                "component {i} is not quadratic in c"
            );
            q[i] = (s3[i] - 2 * s2[i] + s1[i]) / 2;
            b[i] = s2[i] - s1[i] - 3 * q[i];
            a[i] = s1[i] - b[i] - q[i];
        }

        for i in [FIXED, DEGREE] {
            assert_eq!(b[i], 0, "component {i} must not grow with the lane count");
            assert_eq!(q[i], 0, "component {i} must not carry a c^2 term");
        }
        assert_eq!(
            a[DEGREE] + b[DEGREE] + q[DEGREE],
            7,
            "degree must stay at g_sql4_obj's 7"
        );
        assert_eq!(a[FIXED], 3, "three fixed columns: two view u8 tables and one shared lane u8");
        // A probe pair is a pure advice-plus-lookup replica: it shares the two
        // probe selectors with every other probe, holds no range table of its
        // own, owns no shuffle, and nothing copies a probe cell.
        for i in [FIXED, SEL, SHUF, PERM, DEGREE] {
            assert_eq!(
                q[i], 0,
                "component {i} must not carry a c^2 term (probes share them)"
            );
        }
        assert!(
            q[ADVICE] > 0 && q[LOOKUPS] > 0,
            "a probe pair must cost advice and lookups"
        );
        assert!(b[ADVICE] > 0, "a lane must cost");
        // Shuffles: the two view sorts and the two edge-conservation shuffles,
        // plus the two a lane's child stage owns. Nothing else may shuffle, and
        // in particular no cross-lane column group may appear.
        assert_eq!(
            (a[SHUF], b[SHUF], q[SHUF]),
            (4, 2, 0),
            "the shuffle count must be exactly 4 + 2c"
        );
        // Of a lane's permutation columns exactly three carry a cell OUT of the
        // lane: the occurrence-id column, for the lane seam of the global
        // ordering, and the two channel prefix sums, for the totals stage where
        // condition (10) is compared. The rest are internal to its child stage,
        // which shuffles its own sorted view and emitted table.
        assert!(
            b[PERM] >= 3,
            "a lane must export its oid ends and its two channel totals"
        );

        // Now the audit proper: the model must hold EXACTLY off the fit. Any
        // lane that were cheaper than its siblings, or any structure keyed to
        // the true bag size, would break it.
        for c in 1..=5usize {
            let got = shape(c);
            for i in 0..N {
                let want = a[i] + b[i] * c as i64 + q[i] * (c * c) as i64;
                assert_eq!(
                    got[i], want,
                    "c={c}: component {i} is {} but a full replica costs {}",
                    got[i], want
                );
            }
        }
    }

    /// The COUNT(*) must be invariant to the lane geometry, and the block-wise
    /// split must place every real row in exactly one lane at the right offset.
    /// The geometries below cover: no rounding-up pad at all (the last lane is
    /// entirely real), real rows straddling a lane boundary, an odd rounding-up
    /// tail, a lane holding a single real row, and `lane_rows` = 1, where
    /// `q_sum` never fires, every lane's prefix sum is just its one row and
    /// EVERY step of the occurrence ordering is a lane seam.
    #[test]
    fn mock_lane_geometries() {
        // (lane_rows, pad_extra, expected c)
        let cases: [(usize, usize, usize); 7] = [
            (11, 0, 1), // exactly one full lane, no pad anywhere
            (6, 0, 2),  // real rows straddle the boundary; last lane 5 of 6 tall
            (3, 1, 4),  // 12 released rows over 4 lanes of 3: every lane full
            (5, 3, 3),  // 14 released rows over 3 lanes of 5: last lane 4 tall
            (2, 0, 6),  // the last lane holds a single real row
            (4, 6, 5),  // 17 released over 5 lanes of 4: last lane is ONE pad row
            (1, 0, 11), // one row per lane: every step is a seam
        ];
        for (lane_rows, pad_extra, want_c) in cases {
            let c = lanes_for(SYNTH_BAG_ROWS + pad_extra, lane_rows);
            assert_eq!(c, want_c, "lane_rows={lane_rows}");

            set_config_lanes(c);
            let circuit = MyCircuit::<Fp> {
                edges: synthetic_edges(),
                pad_extra,
                lane_rows,
                num_lanes: c,
                released_capacity: released(pad_extra),
                tamper: Tamper::None,
                _marker: PhantomData,
            };
            let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]).unwrap();
            assert_eq!(prover.verify(), Ok(()), "lane_rows={lane_rows} c={c}");

            // ... and the same geometry must reject a wrong public count
            let prover =
                MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT + 1)]]).unwrap();
            assert!(
                prover.verify().is_err(),
                "lane_rows={lane_rows} c={c} accepted a wrong COUNT(*)"
            );
        }
    }

    /// `LANE_ROWS` is arithmetic on a CONSERVATIVE guess at the preamble and
    /// the blinding rows. Get it wrong and nothing fails until a multi-hour
    /// real proof aborts, so check it against the constraint system halo2
    /// actually builds, at the production cells:
    ///   * lastfm  k = 18, c = 2 (the cell lanes exist for), plus c = 4 as
    ///     headroom for a noisier release
    ///   * fb/wiki k = 22, c = 1
    /// A lane occupies `lane_rows + 1` region rows: the child stage writes a
    /// sentinel row at `lane_rows` for its `Rotation::next()` comparisons.
    #[test]
    fn structure_fits_base_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        for (k, c) in [(18u32, 2usize), (18, 4), (22, 1), (18, 1)] {
            set_config_lanes(c);
            let mut cs = ConstraintSystem::<Fp>::default();
            let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);

            assert!(
                cs.degree() <= 7,
                "k={} c={}: degree {} exceeds g_sql4_obj's 7",
                k,
                c,
                cs.degree()
            );

            let lane_rows = lane_rows_for(k);
            let needed = lane_rows + 1 + cs.blinding_factors() + 1;
            assert!(
                needed <= 1usize << k,
                "k={} c={}: a lane needs {} rows (lane_rows {} + sentinel + {} blinding \
                 + 1) but 2^k = {}",
                k,
                c,
                needed,
                lane_rows,
                cs.blinding_factors(),
                1usize << k
            );
        }
    }

    /// c = 3 lanes at k = 11: the padded pipeline really spans three lanes, each
    /// lane builds its own child table, each lane probes all three of them, the
    /// occurrence ordering runs across both lane seams, the mirror half of
    /// condition (9) has to find a hosting lane for every clean role-2 row, and
    /// the totals stage adds the three pairs of lane totals and compares the two
    /// global sums once.
    #[test]
    fn mock_three_lanes() {
        let (circuit, c) = three_lane_circuit(Tamper::None);
        set_config_lanes(c);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]).unwrap();
        prover.assert_satisfied();
    }

    /// Corrupting one lane row's input-channel multiplicity must break the root
    /// multiplicity gate and that lane's prefix sum.
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

    /// Corrupting one lane total must break the copy constraint out of the lane
    /// and the totals accumulator.
    #[test]
    fn mock_reject_tampered_lane_total() {
        let failures = three_lane_failures(Tamper::LaneTotal);
        report_failing_constraints("lane total", &failures);
    }

    /// Breaking the pad convention on the last rounding-up row, on the cell the
    /// ROLE-1 key is packed from, must break the "role 1 real + key" gate and
    /// the r_in membership lookup.
    #[test]
    fn mock_reject_tampered_bag1_pad_row() {
        let failures = three_lane_failures(Tamper::Bag1PadRow);
        report_failing_constraints("pad row, role-1 key cell", &failures);
        assert!(
            failed_on(&failures, "role 1 real + key") || failed_on(&failures, "W r_in"),
            "expected the key gate or the r_in lookup to reject: {:?}",
            failures
        );
    }

    /// Breaking the pad convention on the join column `y`, which no per-row gate
    /// constrains on a pad row, must break the membership lookups into the
    /// shared indexed views. This is the direction that shows the two lookups
    /// are load-bearing on the pad rows too, now that both roles read one group.
    #[test]
    fn mock_reject_tampered_bag2_pad_row() {
        let failures = three_lane_failures(Tamper::Bag2PadRow);
        report_failing_constraints("pad row, view cell", &failures);
        assert!(
            failed_on(&failures, "W r_in") || failed_on(&failures, "W r_out"),
            "expected a membership lookup to reject: {:?}",
            failures
        );
    }

    /// Zeroing the value retrieved from every lane past the first, while leaving
    /// the root multiplicity honest, must be rejected. This is the permanent
    /// negative test for the cross-lane message sum: it fails only because the
    /// root gate adds all `c` probe values and each probe's membership lookup is
    /// really enforced.
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
    /// One joinable role-1 tuple is moved to the residual side and role 2 is
    /// re-reduced around it, so condition (7) still holds, condition (9) still
    /// holds (the hidden tuple is no longer clean, so it asks nothing of the
    /// role-2 side, and the mirror direction only loses hosts it does not need),
    /// every membership lookup still passes and the occurrence ordering is
    /// untouched. Only the Cardinality Preservation Check can see it, and it
    /// sees it whichever lane the tuple lives in, because the equality compares
    /// the two totals accumulated over ALL lanes.
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

    /// CONDITION (9). Every predicate-passing tuple of both roles is declared
    /// clean and the residual side is left empty, so both Conservation and both
    /// channels of condition (10) agree on every row and only Pairwise
    /// Consistency can see the five dangling role-1 tuples of the slice. It must
    /// reject through the clean-key consistency constraint, which reads the
    /// clean multiplicity summed over ALL lanes.
    ///
    /// The six dangling role-2 tuples the all-clean partition also creates are
    /// what the MIRROR direction sees, so its lookup fires here too; the
    /// assertion below names the direction this test was written for.
    #[test]
    fn mock_reject_mark_all_clean() {
        let failures = three_lane_failures(Tamper::MarkAllClean);
        report_failing_constraints("mark all clean", &failures);
        assert!(
            failed_on(&failures, "clean-key consistency"),
            "the circuit rejected, but not through condition (9): {:?}",
            failures
        );
        assert!(
            failed_on(&failures, "pw: role2^c key in role1^c"),
            "the mirror half of condition (9) did not see the dangling role-2 \
             tuples: {:?}",
            failures
        );
    }

    /// CONDITION (9), the MIRROR direction, and the reason its host bits are not
    /// free advice. One clean role-2 row is made to name a lane that does not
    /// hold a clean role-1 row with its key. Exactly one host bit is still set,
    /// so the "hosted by exactly one lane" gate still passes; conditions (7) and
    /// (10) never read the host bits at all, and the direction-1 half reads the
    /// clean channel rather than these columns. The mirror lookup is the only
    /// thing that can reject, and if it could not, a prover could satisfy the
    /// mirror direction for an arbitrary clean role-2 side.
    #[test]
    fn mock_reject_misnamed_host_lane() {
        let failures = three_lane_failures(Tamper::MisnameHostLane);
        report_failing_constraints("misnamed host lane", &failures);
        assert!(
            failed_on(&failures, "pw: role2^c key in role1^c"),
            "expected the mirror lookup of condition (9) to reject: {:?}",
            failures
        );
    }

    /// OCCURRENCE DISTINCTNESS ACROSS LANES. The first kept occurrence is copied
    /// over the FIRST row of the LAST lane. Both copies pass both membership
    /// lookups, both are internally consistent, both channels of condition (10)
    /// move by the same amount, and -- the point of the test -- every lane's own
    /// occurrence block is still sorted, so a PER-LANE ordering would accept
    /// this witness and the COUNT would be inflated. It must reject through the
    /// LANE SEAM of the global ordering.
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

    /// A released capacity large enough that the LAST lane holds no real row at
    /// all. Its child stage then sees an all-PAD input, emits nothing, and its
    /// table degenerates to the single (0,0,0) dummy row followed by PAD. Every
    /// role-1 row must still be able to prove absence in it, which is the one
    /// gap bracket (0, PAD) that lane offers, and no clean role-2 row may name
    /// it as a host. The lane's occurrence-id column is all PAD, which is what
    /// the "once PAD, always PAD" half of the ordering is for.
    #[test]
    fn mock_all_pad_bag2_lane() {
        let lane_rows = LANE_ROWS_SMALL; // 4
        let pad_extra = 5; // n = 16 -> 4 lanes, the last all pad
        let c = lanes_for(SYNTH_BAG_ROWS + pad_extra, lane_rows);
        assert_eq!(c, 4);
        // lane 3 covers padded rows 12..16, and W only has 11 real rows
        assert!(3 * lane_rows >= SYNTH_BAG_ROWS);

        set_config_lanes(c);
        let circuit = MyCircuit::<Fp> {
            edges: synthetic_edges(),
            pad_extra,
            lane_rows,
            num_lanes: c,
            released_capacity: released(pad_extra),
            tamper: Tamper::None,
            _marker: PhantomData,
        };
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(SYNTH_CNT)]]).unwrap();
        prover.assert_satisfied();
    }

    /// THE SHORT LAST LANE. `three_lane_circuit` releases 12 rows over 3 lanes
    /// of 4, so its last lane is full and none of the negative directions above
    /// says anything about a lane that stops early. This geometry releases 14
    /// rows over 3 lanes of 5, so lane 2 is live on 4 rows of 5 and the rows
    /// past the capacity are neither assigned nor gated.
    ///
    /// A lane that quietly stopped being CONSTRAINED at its capacity, rather
    /// than merely stopping there, would still satisfy the honest witness, so
    /// only the negative directions can see it. Each one below lands in that
    /// short lane: the two pad-row hooks break the last row of the released
    /// capacity, which is its last live row, and the duplicate occurrence is
    /// copied over its FIRST row.
    #[test]
    fn mock_short_last_lane_still_rejects() {
        let lane_rows = 5;
        let pad_extra = 3;
        let c = lanes_for(SYNTH_BAG_ROWS + pad_extra, lane_rows);
        assert_eq!(c, 3);
        assert_eq!(
            lane_live_rows(c - 1, c, lane_rows, SYNTH_BAG_ROWS + pad_extra),
            4,
            "this geometry is only a test of the short last lane if it is short"
        );

        let build = |tamper| {
            set_config_lanes(c);
            MyCircuit::<Fp> {
                edges: synthetic_edges(),
                pad_extra,
                lane_rows,
                num_lanes: c,
                released_capacity: released(pad_extra),
                tamper,
                _marker: PhantomData,
            }
        };
        let public = vec![vec![Fp::from(SYNTH_CNT)]];

        let prover = MockProver::run(11, &build(Tamper::None), public.clone()).unwrap();
        prover.assert_satisfied();

        // (tamper, the constraint this direction must reject through)
        let cases: [(Tamper, &str); 6] = [
            (Tamper::Bag1PadRow, "role 1 real + key"),
            (Tamper::Bag2PadRow, "W r_"),
            (
                Tamper::DuplicateOccurrenceAcrossLanes,
                "carries across the lane seam",
            ),
            (Tamper::MisnameHostLane, "pw: role2^c key in role1^c"),
            (Tamper::HideOneCleanTuple, "cardinality preservation"),
            (Tamper::MarkAllClean, "clean-key consistency"),
        ];
        for (tamper, needle) in cases {
            let failures = MockProver::run(11, &build(tamper), public.clone())
                .unwrap()
                .verify()
                .err()
                .unwrap_or_else(|| panic!("{:?} verified against a short last lane", tamper));
            report_failing_constraints(&format!("short last lane, {:?}", tamper), &failures);
            assert!(
                failed_on(&failures, needle),
                "{:?} rejected, but not through `{}`: {:?}",
                tamper,
                needle,
                failures
            );
        }
    }

    /// c = 1 degenerates to a single-group layout: this circuit and
    /// `g_sql4_obj` must accept the same input and expose the same COUNT(*).
    /// They are no longer byte-identical -- this file realizes condition (7) as
    /// an indicator on the W rows rather than as a shuffled partition group,
    /// because the section selectors that form needs would put the true clean
    /// and residual counts in the verifying key -- so this checks the answer and
    /// mutual satisfiability, not the shape.
    #[test]
    fn mock_single_lane_matches_g_sql4_obj() {
        let edges = synthetic_edges();
        let public: Vec<Fp> = vec![Fp::from(SYNTH_CNT)];

        let baseline = crate::graph_sql::g_sql4_obj::MyCircuit::<Fp> {
            edges: edges.clone(),
            pad_extra: 1,
            _marker: PhantomData,
        };
        let prover = MockProver::run(11, &baseline, vec![public.clone()]).unwrap();
        prover.assert_satisfied();

        let lane_rows = 16; // n = 11 + 1 = 12 fits one lane
        let c = lanes_for(SYNTH_BAG_ROWS + 1, lane_rows);
        assert_eq!(c, 1);
        set_config_lanes(c);
        let laned = MyCircuit::<Fp> {
            edges,
            pad_extra: 1,
            lane_rows,
            num_lanes: c,
            released_capacity: released(1),
            tamper: Tamper::None,
            _marker: PhantomData,
        };
        let prover = MockProver::run(11, &laned, vec![public]).unwrap();
        prover.assert_satisfied();
    }

    /// Real IPA proving at the Revealing-Join-Size degree with the DP release
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
            "[gq4 dp-lanes] dataset={} privacy={} pad={}",
            dataset,
            privacy.label(),
            p.pads[0]
        );
        // `lane_span` is what the lanes COULD hold; the assigned rows are the
        // released capacity, since the last lane stops there.
        println!(
            "[gq4 dp-lanes] true={} capacity={} lanes={} | lane_rows={} \
             lane_span={} assigned_rows={} | probe replicas={}",
            p.true_size[0],
            p.capacity[0],
            p.lanes[0],
            p.lane_rows,
            p.lanes[0] * p.lane_rows,
            p.capacity[0],
            p.lanes[0] * p.lanes[0]
        );
        println!("Proof written to: {}", proof_path);
        println!(
            "[gq4 dp-lanes] lanes={} keygen {:.2}s prove {:.2}s verify {:.2}s \
             total prove+verify {:?}",
            p.lanes[0],
            run.keygen_s,
            run.prove_mean(),
            run.verify_s,
            t_total.elapsed()
        );
    }
}
