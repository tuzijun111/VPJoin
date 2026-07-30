use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::lessthan_or_equal_generic::{
    LtEqGenericChip, LtEqGenericConfig, LtEqGenericInstruction,
};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};
use crate::circuits::conserve_idx::{
    assign_conserve, assign_row_index, configure_conserve, configure_row_index, ConserveConfig,
    RowIndexConfig,
};
use crate::circuits::card_preserve::{
    assign_cp_agg, assign_cp_join, assign_cp_root, build_cp_stage, configure_cp_agg,
    configure_cp_join, configure_cp_root, wire_cp_edge, CpAggConfig, CpJoinConfig, CpRootConfig,
};

use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use std::collections::HashSet;
use std::marker::PhantomData;

// One shared definition of the PAD/sentinel discipline (from q5_obj).
use super::q5_obj::{
    q5_derive, Q5Chip, Q5Derived, NUM_BYTES, PAD_REV, PAD_U64, SCALE, SHIFT_NATION,
};

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

/// Degree Q5's DP lane circuit is proved at. PINNED: the released capacity is
/// absorbed by lanes, not by a bigger domain.
///
/// The value is the degree the SINGLE-COLUMN Q5 circuit takes at the sweep's
/// reference budget: `degree_for("q5", ..)` sizes the domain from the released
/// LS capacity, which at eps = 0.1 gives k = 17 (`results/simpl_q5.csv`). So at
/// that budget a lane row and the baseline row it is compared against share a
/// domain size and the comparison isolates the lane layout rather than the
/// degree. At smaller eps the single-column circuit escalates (18 at 0.05, 19
/// at 0.02, 20 at 0.01) while this circuit stays at 17 and spends the release
/// on lanes instead, which is the whole point of the comparison.
pub const DP_LANE_K: u32 = 17;

/// Rows the u8 range-table `load` regions may claim ahead of the witness
/// region: seven u8 tables of 256 fixed rows each.
///
/// The reserve stays at seven columns' worth even though the repaired circuit
/// loads only four. It is a conservative margin, not a count:
/// `SimpleFloorPlanner` starts a region at the first row free in the columns
/// that region touches, and each `load` touches only its own fixed column, so
/// all of them begin at row 0 and none of them displaces the witness region.
pub const CHIP_LOAD_ROWS: usize = 7 * 256;

/// Blinding rows halo2 keeps at the bottom of every advice column.
pub const BLINDING_SLACK: usize = 64;

/// Usable rows per lane at [`DP_LANE_K`]: a lane fills the domain left over
/// after the range-table loads and the blinding reserve, the same rule
/// `g_sql3_obj_dp` and `g_sql4_obj_dp` derive their `LANE_ROWS` by. Lanes are
/// parallel COLUMN groups inside one 2^k circuit, so a lane shorter than the
/// domain would pay for rows it never fills.
pub const LANE_ROWS: usize = (1usize << DP_LANE_K) - CHIP_LOAD_ROWS - BLINDING_SLACK;

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
    /// Move one joinable LS tuple to the residual side and re-reduce the CO and
    /// NR bags around it, so the partition still passes Conservation and
    /// Pairwise Consistency and ONLY condition (10) can catch it. This is
    /// exactly the cheat a residual-side-only argument misses, so the negative
    /// test using it is what shows the Cardinality Preservation Check is not
    /// vacuous. Multi-lane on purpose: the hidden tuple's contribution is
    /// removed from one lane, and the check has to notice at the global root sum.
    HideOneCleanTuple,
    /// Skip the semijoin reduction entirely and declare every tuple that passes
    /// its predicate clean, so all three residual sections are empty.
    /// Conservation still holds and both channels of condition (10) then agree
    /// row by row, so this is the escape only Pairwise Consistency can close,
    /// and it is the non-vacuity witness for condition (9).
    MarkAllClean,
    /// Keep the honest partition but write `cflag_ls = 1` on one RESIDUAL
    /// lineitem row, i.e. claim on the input side that one more occurrence went
    /// to `LS^c` than the partition side carries. Only the LS Conservation
    /// shuffle, which now carries the indicator as its fifth column, can see
    /// this, so it is the non-vacuity witness for condition (7).
    CleanFlagLie,
    /// Break the canonical pad tuple on the very last lane row, which is past
    /// `|LS^c|` and so outside every copy constraint. Before the repair those
    /// rows were free advice that `perm_lsort` still carried into the lane
    /// aggregation, the drain and the reported revenue.
    LanePadRow,
    /// Set the boundary cell one row past every lane's sorted view to that
    /// lane's LAST sorted nation key, and derive the whole rest of the lane
    /// witness consistently from it. Every gate then agrees: the group-boundary
    /// detector reports "same group" on the last row, the emit gate FORCES the
    /// sentinel there, the drain shuffle carries one fewer real partial and the
    /// merge is built from it, so the lane's last group silently loses its tail
    /// contribution and the reported revenue is wrong. Before the repair that
    /// cell was free advice and this was accepted; now the boundary gate is the
    /// only thing that rejects it.
    LaneSentinel,
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
    // rows of THIS lane past the global clean prefix: the canonical PAD tuple.
    // The boundary is lane-dependent (lane l is all-pad once l*lane_rows is past
    // |LS^c|), so unlike every other lane selector this one cannot be shared.
    q_ls_pad: Selector,
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
    // one per-proof choice instead of one per-row choice
    q_cond_eu: Selector,
    q_cond_dt: Selector,

    // ---------------- bag materialization: NR ----------------
    q_nr_join: Selector,
    q_nr_pred: Selector,
    q_region_tbl: Selector, // TABLE side of the nation->region lookup
    q_nr_pad: Selector,     // rows [nation.len(), nr_total): keep == 0
    nr_rname: Column<Advice>,
    nr_keep: Column<Advice>,
    // ---------------- (1) Conservation Check ----------------
    // R^_i == R^_i^c U+ R^_i^r over the INDEXED cluster relation, one
    // permutation each, laid out ONCE and never per lane. A relation at a DP
    // capacity has its padding rows in the residual part, indicator zero.
    row_idx: RowIndexConfig,
    cons_nr: ConserveConfig,
    cons_co: ConserveConfig,
    cons_ls: ConserveConfig,

    cflag_nr: Column<Advice>, // the selector bit c per NR row
    nr_pair: Vec<Column<Advice>>,
    q_row_nr: Selector, // rows [0, nr_total): the NR relation's whole capacity
    iz_nr: IsZeroConfig<F>,

    // ---------------- bag materialization: CO ----------------
    q_oc_join: Selector,
    q_cust_tbl: Selector, // TABLE side of the orders->customer lookup
    q_co_ge: Selector,
    q_co_lt: Selector,
    q_co_and: Selector,
    q_co_pad: Selector, // rows [orders.len(), co_total): keep == 0
    co_ge_ok: Column<Advice>,
    co_lt_ok: Column<Advice>,
    co_keep: Column<Advice>,
    cflag_co: Column<Advice>, // clean indicator per order row
    co_nk: Column<Advice>,
    co_pair: Vec<Column<Advice>>,
    co_pkey: Column<Advice>, // co_pair[0]*SHIFT_NATION + co_pair[1]
    q_row_co: Selector, // rows [0, co_total): the CO relation's whole capacity
    lteq_start_le_odate: LtEqGenericConfig<F, NUM_BYTES>,
    lt_odate_lt_end: LtConfig<F, NUM_BYTES>,

    // ---------------- bag materialization: LS ----------------
    q_ls_join: Selector,
    q_supp_tbl: Selector, // TABLE side of the lineitem->supplier lookup
    ls_mat: Vec<Column<Advice>>,
    cflag_ls: Column<Advice>, // clean indicator per LS row
    ls_pkey: Column<Advice>,  // ls_mat[0]*SHIFT_NATION + ls_mat[1]

    // ---------------- LS partition: disjoin side ----------------
    ls_disjoin: Vec<Column<Advice>>,
    // 5 cols: the tuple plus the clean flag. Rows [0, |LS^c|) are copy-equal,
    // in GLOBAL pipeline order, to the lanes' clean `ls_join` cells, which is
    // what makes this the union-of-lanes view of `LS^c`.
    ls_part_pad: Vec<Column<Advice>>,
    perm_ls: PermAnyConfig,

    // -------- (2) Pairwise Consistency --------
    // Both sides of every lookup read a COMMITTED key column gated by that
    // relation's selector bit, so no clean-prefix extent reaches the fixed
    // columns. `q_row_nr` / `q_row_co` above and `q_ls_join` are the ranges.

    // -------- condition (10), Cardinality Preservation --------
    // |R^c join| == |R join| over the cluster tree rooted at LS, accumulated
    // over the whole root relation and compared ONCE.
    cp_agg_co: CpAggConfig<F, NUM_BYTES>,
    cp_agg_nr: CpAggConfig<F, NUM_BYTES>,
    cp_join_co: CpJoinConfig<F, NUM_BYTES>,
    cp_join_nr: CpJoinConfig<F, NUM_BYTES>,
    cp_root: CpRootConfig,
    q_cp_mu: Selector,

    // The LS union-of-lanes compaction, which is NOT an OBJ condition: it is
    // the padding layer that tethers the lanes' clean `ls_join` cells to one
    // column group in global pipeline order. See the note in `configure`.
    q_cln_flag: Vec<Selector>, // rows of LS^c inside ls_part_pad: flag == 1
    q_res_flag: Vec<Selector>, // rows of LS^r: flag == 0
    q_pad_flag: Vec<Selector>, // pad tail: the whole row is the PAD tuple

    // ---------------- lanes (shared selectors, per-lane columns) ----------
    // All lanes are active on the SAME rows 0..lane_rows, so one selector of
    // each kind serves every lane; only the columns replicate.
    lanes: Vec<LaneConfig<F>>,
    q_lane_line: Selector,
    q_lane_first: Selector,
    q_lane_accu: Selector,
    q_lane_sentinel: Selector, // row lane_rows of every lane's ls_sorted
    q_drain: Selector,         // LHS gate of the per-lane drain shuffles (complex)

    // ---------------- cross-lane merge ----------------
    merge_nk: Column<Advice>,
    merge_sum: Column<Advice>,
    merge_real: Column<Advice>, // 1 on drained group rows, 0 on sentinels
    q_merge: Selector,
    msort: Vec<Column<Advice>>, // merge sorted by nk, sentinels last
    perm_merge: PermAnyConfig,
    // msort[0] is nondecreasing: this is the LAST grouping before the answer,
    // so unlike the lane-local sorts its order has to be proved
    q_sort_m: Selector,
    q_m_sentinel: Selector, // row m_total of msort, read by iz_m_same_next
    lt_m_cur_next: LtConfig<F, NUM_BYTES>,
    iz_m_key_eq: IsZeroConfig<F>,

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
    // degree-1 stand-in for `1 - iz_m_same_next.expr()`, so the name lookup can
    // afford to gate its TABLE side without raising the circuit's degree
    m_res_is_last: Column<Advice>,

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

        // -------- query parameters are constant down the column --------
        // The three parameter columns are plain advice read at Rotation::cur
        // inside the per-row predicate chips. Without a cross-row tie each row
        // carries its OWN window: `co_keep = ge * lt` accepts an out-of-window
        // order by widening `cond_start`/`cond_end` on that row alone, and
        // `nr_keep` accepts a non-European nation by setting `cond_europe` to
        // that row's `nr_rname`. The circuit would then certify the answer of no
        // single Q5 instance. These two gates turn the per-row prover choice
        // into one per-proof choice.
        let q_cond_eu = meta.selector();
        let q_cond_dt = meta.selector();
        meta.create_gate("cond_europe is constant", |m| {
            let q = m.query_selector(q_cond_eu);
            vec![
                q * (m.query_advice(cond_europe, Rotation::cur())
                    - m.query_advice(cond_europe, Rotation::next())),
            ]
        });
        meta.create_gate("date window is constant", |m| {
            let q = m.query_selector(q_cond_dt);
            vec![
                q.clone()
                    * (m.query_advice(cond_start, Rotation::cur())
                        - m.query_advice(cond_start, Rotation::next())),
                q * (m.query_advice(cond_end, Rotation::cur())
                    - m.query_advice(cond_end, Rotation::next())),
            ]
        });

        // ---------------- NR materialization (as q5_obj) ----------------
        let q_nr_join = meta.complex_selector();
        let q_nr_pred = meta.selector();
        // The TABLE side of a `lookup_any` is 0 on rows where its selector is
        // off, so gating it with a selector enabled over exactly the dimension
        // table's real rows is what makes the lookup relation the ASSIGNED
        // prefix instead of the whole (mostly free-advice) column.
        let q_region_tbl = meta.complex_selector();
        // The NR permutation runs over nr_total >= nation.len() rows, so the
        // link gate below is live on rows that carry no nation. There `nr_keep`
        // is free advice, and keep = 1 injects an arbitrary triple straight into
        // nr_filt_pad and, through the Conservation shuffle, into nr_out_pad.
        let q_nr_pad = meta.selector();

        let nr_rname = meta.advice_column();
        let nr_keep = meta.advice_column();
        let cflag_nr = meta.advice_column();

        let nr_pair = vec![meta.advice_column(), meta.advice_column()];
        // One column wider than before: the last column of each side carries the
        // clean indicator, so the Conservation Check binds it.
        // The NR relation occupies its whole capacity, rows [0, nr_total): the
        // real nations first, then padding rows whose predicate bit `q_nr_pad`
        // pins to 0. That capacity is the DP padding layer and is unchanged
        // here; what is gone is the `[clean | residual | pad]` group and the
        // Conservation shuffle that used to bind the indicator to it.
        let q_row_nr = meta.complex_selector();

        meta.lookup_any("nation_region_join", |m| {
            let q = m.query_selector(q_nr_join);
            let qt = m.query_selector(q_region_tbl);
            vec![
                (
                    q.clone() * m.query_advice(nation[2], Rotation::cur()),
                    qt.clone() * m.query_advice(region_file[0], Rotation::cur()),
                ),
                (
                    q * m.query_advice(nr_rname, Rotation::cur()),
                    qt * m.query_advice(region_file[1], Rotation::cur()),
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

        // The clean indicator is a BIT. Nothing else says so: the link gate
        // bool-checks `nr_keep` only, `q_cln_flag`/`q_res_flag` pin the flag on
        // the clean and residual sections of the partition side but not on its
        // pad tail, and `card_preserve` puts no range check on v_cln. A value
        // such as 1 + 1/k parked in that tail inflates the clean channel of
        // condition (10) by a fraction, which is exactly the compensation the
        // check exists to forbid.
        // ---------- (1) Selector Check on NR ----------
        // Under `q_row_nr`, so it covers the capacity tail as well as the real
        // nations. The predicate half is what the link gate used to give
        // structurally; on a capacity row `nr_keep` is pinned to 0, so it also
        // confines the selection to the real rows, which is the paper's cluster
        // adjustment.
        meta.create_gate("NR selector is a bit and implies its predicate", |m| {
            let q = m.query_selector(q_row_nr);
            let c = m.query_advice(cflag_nr, Rotation::cur());
            let b = m.query_advice(nr_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone() * c.clone() * (one.clone() - c.clone()),
                q * c * (one - b),
            ]
        });

        // NR padding rows carry no nation, so their predicate bit is 0 and the
        // link gate then forces the whole nr_filt_pad row to the PAD tuple.
        meta.create_gate("NR pad row keeps nothing", |m| {
            let q = m.query_selector(q_nr_pad);
            vec![q * m.query_advice(nr_keep, Rotation::cur())]
        });

        // ---------------- CO materialization (as q5_obj) ----------------
        let q_oc_join = meta.complex_selector();
        let q_cust_tbl = meta.complex_selector();
        let q_co_ge = meta.selector();
        let q_co_lt = meta.selector();
        let q_co_and = meta.selector();
        let q_co_pad = meta.selector();

        let co_ge_ok = meta.advice_column();
        let co_lt_ok = meta.advice_column();
        let co_keep = meta.advice_column();
        let cflag_co = meta.advice_column();

        let co_nk = meta.advice_column();
        let co_pair = vec![meta.advice_column(), meta.advice_column()];
        let co_pkey = meta.advice_column();
        // As on the NR side: the CO relation occupies its whole capacity, rows
        // [0, co_total), and that capacity is the DP padding layer, untouched.
        let q_row_co = meta.complex_selector();

        meta.lookup_any("orders_customer_join", |m| {
            let q = m.query_selector(q_oc_join);
            let qt = m.query_selector(q_cust_tbl);
            vec![
                (
                    q.clone() * m.query_advice(orders[1], Rotation::cur()),
                    qt.clone() * m.query_advice(customer[0], Rotation::cur()),
                ),
                (
                    q * m.query_advice(co_nk, Rotation::cur()),
                    qt * m.query_advice(customer[1], Rotation::cur()),
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

        // same two patches as on the NR side: the clean indicator is a bit, and
        // a CO padding row (one past orders.len(), still inside the Conservation
        // permutation) keeps nothing, so its co_filt_pad row is the PAD tuple.
        // ---------- (1) Selector Check on CO ----------
        meta.create_gate("CO selector is a bit and implies its predicate", |m| {
            let q = m.query_selector(q_row_co);
            let c = m.query_advice(cflag_co, Rotation::cur());
            let b = m.query_advice(co_keep, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone() * c.clone() * (one.clone() - c.clone()),
                q * c * (one - b),
            ]
        });
        meta.create_gate("CO pad row keeps nothing", |m| {
            let q = m.query_selector(q_co_pad);
            vec![q * m.query_advice(co_keep, Rotation::cur())]
        });

        // ---------------- LS materialization (as q5_obj) ----------------
        let q_ls_join = meta.complex_selector();
        let q_supp_tbl = meta.complex_selector();
        let ls_mat = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        meta.lookup_any("lineitem_supplier_join", |m| {
            let q = m.query_selector(q_ls_join);
            let qt = m.query_selector(q_supp_tbl);
            vec![
                (
                    q.clone() * m.query_advice(lineitem[1], Rotation::cur()),
                    qt.clone() * m.query_advice(supplier[0], Rotation::cur()),
                ),
                (
                    q * m.query_advice(ls_mat[1], Rotation::cur()),
                    qt * m.query_advice(supplier[1], Rotation::cur()),
                ),
            ]
        });

        let cflag_ls = meta.advice_column();
        let ls_pkey = meta.advice_column();

        // `cflag_ls` is the factor of the root clean multiplicity, so state its
        // booleanity directly rather than argue it from the partition layout.
        meta.create_gate("cflag_ls is a bit", |m| {
            let q = m.query_selector(q_ls_join);
            let c = m.query_advice(cflag_ls, Rotation::cur());
            vec![q * c.clone() * (Expression::Constant(F::ONE) - c)]
        });

        // the same packed key on the parent side of the LS -> CO edge
        meta.create_gate("ls_pkey = okey*SHIFT_NATION + nk", |m| {
            let q = m.query_selector(q_ls_join);
            let ok = m.query_advice(ls_mat[0], Rotation::cur());
            let nk = m.query_advice(ls_mat[1], Rotation::cur());
            let pk = m.query_advice(ls_pkey, Rotation::cur());
            vec![q * (pk - (ok * Expression::Constant(F::from(SHIFT_NATION)) + nk))]
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

        // ---------------- LS partition permutation, condition (7) ----------
        // ls_join is laned; ls_disjoin and ls_part_pad stay ONE column group.
        // That is the whole reason condition (7) survives the lane split: the
        // partition side of the shuffle is a single group of |lineitem| rows
        // whose clean prefix is copy-linked, in global pipeline order, to the
        // lanes' ls_join cells, so the multiset the shuffle fixes is the UNION
        // over lanes and not one lane at a time.
        let ls_disjoin = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        // One column wider than before: the clean indicator rides along, so the
        // LS Conservation Check binds it.
        let ls_part_pad = (0..5).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for &c in ls_disjoin.iter().chain(ls_part_pad.iter()) {
            meta.enable_equality(c);
        }

        let perm_ls = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            let mut ls_in = ls_mat.clone();
            ls_in.push(cflag_ls);
            PermAnyChip::configure(meta, q1, q2, ls_in, ls_part_pad.clone())
        };

        // -------- the LS union-of-lanes compaction, NOT an OBJ condition ------
        // `ls_part_pad` is laid out as [clean rows | residual rows | pad rows]
        // and its clean prefix is copy-constrained, in GLOBAL pipeline order,
        // to the lanes' clean `ls_join` cells. That is the multi-lane design
        // rule this file rests on: every condition attaches to a column group
        // laid out ONCE, and the lane rows are tethered to it at both ends.
        //
        // This group is not part of the OBJ itself. All three conditions read
        // the committed rows through `cflag_ls`, which
        // has its own booleanity gate. What the shuffle and these pins still
        // buy is the PADDING LAYER: they certify that the clean prefix holds
        // exactly the selected LS tuples, which is what lets the lanes cover
        // |LS^c| + ls_pad_extra rows rather than every committed lineitem row,
        // and it is what the released capacity is spent on. NR and CO have no
        // lane pipeline, so their groups are gone entirely.
        let q_cln_flag = (0..1).map(|_| meta.selector()).collect::<Vec<_>>();
        let q_res_flag = (0..1).map(|_| meta.selector()).collect::<Vec<_>>();
        let q_pad_flag = (0..1).map(|_| meta.selector()).collect::<Vec<_>>();

        {
            let cols = ls_part_pad.clone();
            let flag_col = *ls_part_pad.last().unwrap();
            let q_c = q_cln_flag[0];
            let q_r = q_res_flag[0];
            let q_p = q_pad_flag[0];
            meta.create_gate("LS compaction: flag on each section", move |m| {
                let qc = m.query_selector(q_c);
                let qr = m.query_selector(q_r);
                let qp = m.query_selector(q_p);
                let f = m.query_advice(flag_col, Rotation::cur());
                let mut cs = vec![qc * (f.clone() - Expression::Constant(F::ONE)), qr * f];
                let last = cols.len() - 1;
                for (j, &c) in cols.iter().enumerate() {
                    let v = m.query_advice(c, Rotation::cur());
                    let want = if j == last {
                        Expression::Constant(F::ZERO)
                    } else {
                        Expression::Constant(F::from(PAD_U64))
                    };
                    cs.push(qp.clone() * (v - want));
                }
                cs
            });
        }

        // ---------------- (1) Conservation Check ----------------
        // One permutation per cluster relation, over column groups laid out
        // ONCE rather than per lane, which is the design rule this file rests
        // on. The indices are distinct, so the indexed relation is a set even
        // though the relation is a bag, and the single permutation rules out an
        // occurrence being fabricated, lost, duplicated or counted on both
        // sides: no Non-Membership Check.
        //
        // LS carries a second permutation, `perm_ls`, which is NOT this
        // condition: it is the union-of-lanes compaction that tethers the lanes
        // to one group in global pipeline order and lets them cover
        // |LS^c| + ls_pad_extra rows.
        let row_idx = configure_row_index::<F>(meta);
        let cons_nr = configure_conserve::<F>(meta, &row_idx, &nr_pair, cflag_nr);
        let cons_co = configure_conserve::<F>(meta, &row_idx, &co_pair, cflag_co);
        let cons_ls = configure_conserve::<F>(meta, &row_idx, &ls_mat, cflag_ls);

        // ---------------- (2) Pairwise Consistency ----------------
        // Two mutual Membership Checks per edge of the cluster tree, each
        // looking one relation's key column up directly in the adjacent
        // relation's. Both sides read the COMMITTED columns gated by that
        // relation's selector bit:
        //
        //     q_row * c(t) * (key + 1),
        //
        // so a deselected row, a capacity padding row and a row past the
        // relation all read 0, 0 is in every table, and the containment is over
        // the selected keys only. The shift by one stops a real key of 0 from
        // colliding with that gated-off 0.
        //
        // This is also what makes the condition lane-agnostic. Reading it off
        // the clean prefix of `ls_part_pad` worked only because that group is
        // the union of the lanes in global pipeline order, and the gating
        // selector's extent was |LS^c|, a private length baked into a fixed
        // column. Here the LS side is the committed lineitem row, which no lane
        // touches, and the three extents are |lineitem|, nr_total and co_total,
        // all public.
        let mut pw_edge = |name: &'static str,
                           q_in: Selector,
                           c_in: Column<Advice>,
                           k_in: Column<Advice>,
                           q_tb: Selector,
                           c_tb: Column<Advice>,
                           k_tb: Column<Advice>| {
            meta.lookup_any(name, move |m| {
                let one = Expression::Constant(F::ONE);
                let lhs = m.query_selector(q_in)
                    * m.query_advice(c_in, Rotation::cur())
                    * (m.query_advice(k_in, Rotation::cur()) + one.clone());
                let rhs = m.query_selector(q_tb)
                    * m.query_advice(c_tb, Rotation::cur())
                    * (m.query_advice(k_tb, Rotation::cur()) + one);
                vec![(lhs, rhs)]
            });
        };

        // edge (LS, CO) on the packed (o_orderkey, nationkey_shift) key
        pw_edge("pw: LS^c pkey in CO^c pkey",
            q_ls_join, cflag_ls, ls_pkey, q_row_co, cflag_co, co_pkey);
        pw_edge("pw: CO^c pkey in LS^c pkey",
            q_row_co, cflag_co, co_pkey, q_ls_join, cflag_ls, ls_pkey);

        // edge (LS, NR) on nationkey_shift
        pw_edge("pw: LS^c nationkey in NR^c nationkey",
            q_ls_join, cflag_ls, ls_mat[1], q_row_nr, cflag_nr, nr_pair[0]);
        pw_edge("pw: NR^c nationkey in LS^c nationkey",
            q_row_nr, cflag_nr, nr_pair[0], q_ls_join, cflag_ls, ls_mat[1]);

        // ------------- condition (10), Cardinality Preservation -------------
        // One fixed column serves every Lt chip of the check (and the merge
        // sortedness ladder below), so the whole thing costs one u8 range table.
        let cp_u8 = meta.fixed_column();

        // Children of the root. Both are leaves, so their two multiplicity
        // columns are columns the circuit already has: the predicate bit is the
        // input channel and the BOUND indicator (keep * c, the last column of the
        // filt_pad side of the Conservation Check) is the clean one.
        let cp_agg_co = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            co_pkey,
            co_keep,
            cflag_co,
            PAD_U64,
        );
        let cp_agg_nr = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            nr_pair[0],
            nr_keep,
            cflag_nr,
            PAD_U64,
        );

        // Parent side, on the rows of LS (one per lineitem row). These columns
        // are NOT laned: the root traversal runs over the whole root relation in
        // base-relation order, so both channels are accumulated globally and the
        // single equality below compares two totals over the union of the lanes.
        let cp_join_co = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, ls_pkey);
        let cp_join_nr = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, ls_mat[1]);
        wire_cp_edge(meta, &cp_join_co, &cp_agg_co, ls_pkey);
        wire_cp_edge(meta, &cp_join_nr, &cp_agg_nr, ls_mat[1]);

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
        let q_lane_sentinel = meta.selector();
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

            // Rows of this lane past the global clean prefix. `ls_part_pad`'s
            // copy constraints stop at |LS^c| and the Pairwise Consistency
            // selectors stop there too, so these rows are outside every other
            // argument -- yet `perm_lsort` carries whatever sits in them into
            // ls_sorted, hence into line_rev, run_sum, res_pad, the drain
            // shuffle, the merge sum and the reported revenue. As free advice
            // they are a direct forgery of the query answer, and at the
            // benchmarked release most lane rows ARE pad rows. Pinning them to
            // the canonical pad tuple [0, PAD, 0, 0] is what makes them inert;
            // gating them out of `perm_lsort` instead would change what the
            // lane pipeline proves.
            let q_ls_pad = meta.selector();
            {
                let cols = ls_join.clone();
                meta.create_gate("lane ls_join pad tail is PAD", move |m| {
                    let q = m.query_selector(q_ls_pad);
                    let want = [F::ZERO, F::from(PAD_U64), F::ZERO, F::ZERO];
                    cols.iter()
                        .zip(want.iter())
                        .map(|(&c, &w)| {
                            q.clone()
                                * (m.query_advice(c, Rotation::cur()) - Expression::Constant(w))
                        })
                        .collect::<Vec<_>>()
                });
            }

            // The sentinel row `lane_rows` of this lane's ls_sorted, which
            // `iz_same_next` reads at row lane_rows-1 to close the last group.
            // Unpinned, setting it equal to the last sorted nationkey makes
            // iz_same_next report "same group" on the last real row; the lane
            // emit gate then FORCES res_pad to the SENTINEL there and that
            // group's partial revenue silently disappears from the merge, so a
            // whole nation loses its tail contribution. The pinned value is 0,
            // not PAD: real nationkeys are stored shifted by +1 and pad rows are
            // keyed at PAD_U64, so 0 differs from every key that can appear at
            // row lane_rows-1 and closes BOTH a real last group and a trailing
            // PAD-keyed group (whose emitted sum is then 0 = the sentinel).
            {
                let nk_col = ls_sorted[1];
                meta.create_gate("lane ls_sorted group-by sentinel is 0", move |m| {
                    let q = m.query_selector(q_lane_sentinel);
                    vec![q * m.query_advice(nk_col, Rotation::cur())]
                });
            }

            // Lane-local sort by nation key. Deliberately NOT proved sorted,
            // and that is sound here: the run_sum ladder splits the lane's rows
            // into maximal runs of equal nationkey and emits each run's total at
            // its last row, so whatever the order, the emitted partials for a
            // nation sum to exactly the sum of that nation's rows in this lane,
            // and the merge stage below re-groups and re-SUMS them per nation.
            // A mis-ordered lane can only emit MORE partials, never fewer and
            // never smaller ones, and more than SEG of them makes the drain
            // shuffle unsatisfiable. The merge stage's own sorted view is the
            // last grouping before the answer, so THAT one is proved.
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
                q_ls_pad,
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

        // The merge grouping finds group boundaries by comparing msort[0] with
        // its neighbours, and it is the LAST grouping before the answer: nothing
        // downstream re-sums. `perm_merge` ties msort to the merge region as a
        // MULTISET and says nothing about ORDER, so without this ladder a prover
        // lays one nationkey out as two non-adjacent runs, each run looks like
        // its own group, and the answer carries that nation TWICE with its
        // revenue split between the two rows -- a wrong answer that every other
        // constraint in this file accepts. Same shape as `configure_cp_agg`'s
        // "cp: sorted key nondecreasing", sharing the same u8 range table, and
        // the merge region is only c * SEG <= 512 rows so it is nearly free.
        let q_sort_m = meta.selector();
        let aux_m_key_eq = meta.advice_column();
        let iz_m_key_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort_m),
            |m| {
                m.query_advice(msort[0], Rotation::next())
                    - m.query_advice(msort[0], Rotation::cur())
            },
            aux_m_key_eq,
        );
        let lt_m_cur_next = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            cp_u8,
            |m| m.query_selector(q_sort_m),
            |m| m.query_advice(msort[0], Rotation::cur()),
            |m| m.query_advice(msort[0], Rotation::next()),
        );
        meta.create_gate("sorted merge nk is nondecreasing", |m| {
            let q = m.query_selector(q_sort_m);
            let le = lt_m_cur_next.is_lt(m, None) + iz_m_key_eq.expr();
            vec![q * (le - Expression::Constant(F::ONE))]
        });

        // Row m_total of msort, which `iz_m_same_next` reads at row m_total-1.
        // `q_sort_m` stops at the pair (m_total-2, m_total-1) and `perm_merge`
        // covers rows 0..m_total, so nothing else touches it. Unpinned, setting
        // it equal to the last nationkey in msort makes the final emit gate write
        // PAD instead of that group's revenue and the corresponding nation
        // vanishes from the answer, with the name lookup going vacuous with it.
        //
        // The pinned value is PAD_U64 rather than 0, which follows from the
        // name-attachment lookup below: its table is the committed NR pair
        // gated by `cflag_nr`, and a capacity row
        // contributes the all-zero tuple rather than (PAD, PAD), so a trailing
        // run of PAD-keyed merge sentinels has nothing to attach to and must
        // never close. Every real nationkey is stored shifted by +1, so a real
        // group still ends the moment the next row carries a different key.
        let q_m_sentinel = meta.selector();
        meta.create_gate("msort group-by sentinel is PAD", |m| {
            let q = m.query_selector(q_m_sentinel);
            vec![
                q * (m.query_advice(msort[0], Rotation::cur())
                    - Expression::Constant(F::from(PAD_U64))),
            ]
        });

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

        // `m_res_is_last` is `1 - iz_m_same_next.expr()` copied into an advice
        // cell on every row the merge grouping covers. It exists purely for
        // degree: the lookup below needs a gated TABLE side, and with the
        // degree-2 IsZero expression still on the input side that lookup would
        // cost 2 + 4 + 2 = 8 and push the whole circuit's degree up. Through this
        // column the input side is degree 3 and the lookup stays at 7, which is
        // what the Cardinality Preservation Check's own gap lookup already costs.
        let m_res_is_last = meta.advice_column();
        meta.create_gate("m_res_is_last = is_last", |m| {
            let q = m.query_selector(q_m_line);
            let one = Expression::Constant(F::ONE);
            let is_last = one - iz_m_same_next.expr();
            vec![q * (m.query_advice(m_res_is_last, Rotation::cur()) - is_last)]
        });

        // attach (nk, name) by looking the merged group up in the SELECTED NR
        // rows, ONCE for all lanes. The table side is the committed pair gated
        // by `cflag_nr`, so a deselected nation, a capacity padding row and a
        // row past nr_total all contribute the all-zero tuple, which is what a
        // gated-off input row reads; the shift by one keeps that dummy away
        // from any real pair.
        //
        // Gating by the bit is what makes this tight. Gating instead by a
        // permutation selector over the assigned rows of an `nr_out_pad` group
        // would put every RESIDUAL nation in the table too, and a merged group
        // could then be labelled with the name of a nation the reduction had
        // dropped.
        let (m_res_nk, m_res_nm) = (m_res_pad[0], m_res_pad[1]);
        let nr_pair_l = nr_pair.clone();
        meta.lookup_any("attach name from NR (merge)", move |m| {
            let one = Expression::Constant(F::ONE);
            let gate = m.query_selector(q_m_res_lookup)
                * m.query_advice(m_res_is_last, Rotation::cur());
            let q_tbl = m.query_selector(q_row_nr) * m.query_advice(cflag_nr, Rotation::cur());

            vec![
                (
                    gate.clone() * (m.query_advice(m_res_nk, Rotation::cur()) + one.clone()),
                    q_tbl.clone() * (m.query_advice(nr_pair_l[0], Rotation::cur()) + one.clone()),
                ),
                (
                    gate * (m.query_advice(m_res_nm, Rotation::cur()) + one.clone()),
                    q_tbl * (m.query_advice(nr_pair_l[1], Rotation::cur()) + one),
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
            q_cond_eu,
            q_cond_dt,

            q_nr_join,
            q_nr_pred,
            q_region_tbl,
            q_nr_pad,
            nr_rname,
            nr_keep,
            row_idx,
            cons_nr,
            cons_co,
            cons_ls,
            cflag_nr,
            nr_pair,
            q_row_nr,
            iz_nr,

            q_oc_join,
            q_cust_tbl,
            q_co_ge,
            q_co_lt,
            q_co_and,
            q_co_pad,
            co_ge_ok,
            co_lt_ok,
            co_keep,
            cflag_co,
            co_nk,
            co_pair,
            co_pkey,
            q_row_co,
            lteq_start_le_odate,
            lt_odate_lt_end,

            q_ls_join,
            q_supp_tbl,
            ls_mat,
            cflag_ls,
            ls_pkey,

            ls_disjoin,
            ls_part_pad,
            perm_ls,


            cp_agg_co,
            cp_agg_nr,
            cp_join_co,
            cp_join_nr,
            cp_root,
            q_cp_mu,
            q_cln_flag,
            q_res_flag,
            q_pad_flag,

            lanes,
            q_lane_line,
            q_lane_first,
            q_lane_accu,
            q_lane_sentinel,
            q_drain,

            merge_nk,
            merge_sum,
            merge_real,
            q_merge,
            msort,
            perm_merge,
            q_sort_m,
            q_m_sentinel,
            lt_m_cur_next,
            iz_m_key_eq,

            q_m_line,
            q_m_first,
            q_m_accu,
            m_run_sum,
            iz_m_same_prev,
            iz_m_same_next,
            m_res_pad,
            q_m_res_lookup,
            m_res_is_last,

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
        // chips. Four u8-range loads now, not seven: the Cardinality
        // Preservation Check shares ONE fixed column across all of its Lt chips
        // and the merge sortedness ladder, where the deleted residual-side gap
        // argument used four columns of its own. `CHIP_LOAD_ROWS` below is left
        // at the old 7 * 256 as a conservative row budget.
        let iz_nr_chip = IsZeroChip::construct(self.config.iz_nr.clone());

        let lteq_ge_chip =
            LtEqGenericChip::<F, NUM_BYTES>::construct(self.config.lteq_start_le_odate.clone());
        lteq_ge_chip.load(layouter)?;

        let lt_end_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_odate_lt_end.clone());
        lt_end_chip.load(layouter)?;

        // Every Lt chip of the Cardinality Preservation Check shares one u8 fixed
        // column, and the merge sortedness ladder shares it too, so a single load
        // covers all of them. This replaces the four gap chips (and their four
        // separate u8 loads) of the deleted residual-side argument.
        LtChip::<F, NUM_BYTES>::construct(self.config.cp_agg_co.lt_key_cur_next).load(layouter)?;

        let lt_m_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_m_cur_next);
        let iz_m_key_eq_chip = IsZeroChip::construct(self.config.iz_m_key_eq.clone());

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

        // ---------------- clean/residual witness over the cluster tree --------
        // The honest clean instance is the fully reduced one: an LS tuple is
        // clean iff it joins both bags, a CO tuple iff its packed key occurs in a
        // clean LS tuple, an NR tuple iff its nationkey does. That is a fixed
        // point of the semijoin reduction, so Conservation and Pairwise
        // Consistency both hold on it and condition (10) holds with equality.
        //
        // Nothing here is lane-aware: the partition is a property of the
        // relations, and only the aggregation over LS^c is laned.
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
        if tamper == Tamper::HideOneCleanTuple {
            let i = ls_cln_b
                .iter()
                .position(|&b| b == 1)
                .expect("HideOneCleanTuple needs at least one clean LS tuple");
            ls_cln_b[i] = 0;
        }

        // test hook only: no reduction at all. Every tuple that passes its own
        // predicate is declared clean, so all three residual sections come out
        // empty and both channels of condition (10) agree row by row. Only
        // condition (9) can reject this.
        let all_clean = tamper == Tamper::MarkAllClean;
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

        // What the `cflag_ls` COLUMN carries. Honestly this is `ls_cln_b`; the
        // CleanFlagLie hook makes the input side of the LS Conservation shuffle
        // disagree with its partition side by exactly one flag.
        let mut ls_cflag_written = ls_cln_b.clone();
        if tamper == Tamper::CleanFlagLie {
            let i = ls_cflag_written
                .iter()
                .position(|&b| b == 0)
                .expect("CleanFlagLie needs at least one residual LS tuple");
            ls_cflag_written[i] = 1;
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
        if tamper == Tamper::LanePadRow {
            assert!(
                join_len < cap,
                "LanePadRow tamper needs at least one pad row"
            );
            ls_join_full[cap - 1] = vec![0u64, PAD_U64, 12_345u64, 0u64];
        }

        struct LaneWit {
            sorted: Vec<Vec<u64>>,
            line_rev: Vec<u64>,
            run_sum: Vec<u64>,
            res_pad: Vec<[u64; 2]>,
            drained: Vec<[u64; 2]>,
            /// value written one row past the sorted view; honestly 0, and
            /// `q_lane_sentinel` pins it there
            sentinel_nk: u64,
        }

        let mut lane_wit: Vec<LaneWit> = Vec::with_capacity(c);
        for l in 0..c {
            let mut sorted: Vec<Vec<u64>> =
                ls_join_full[l * lane_rows..(l + 1) * lane_rows].to_vec();
            sorted.sort_by_key(|r| r[1]); // by nationkey_shift (PAD_U64 last)

            let sentinel_nk = if tamper == Tamper::LaneSentinel {
                sorted[lane_rows - 1][1]
            } else {
                0
            };

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
                    sentinel_nk
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
                sentinel_nk,
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

            // the pinned sentinel at row m_total carries PAD_U64, so a
            // trailing run of PAD-keyed merge sentinels never closes and never
            // emits
            let next_nk = if i + 1 < m_total {
                msort_u64[i + 1][0]
            } else {
                PAD_U64
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
                // TABLE side of the orders->customer lookup: exactly the assigned
                // customer rows
                for i in 0..customer.len() {
                    self.config.q_cust_tbl.enable(&mut region, i)?;
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
                // the date window is one per-proof choice, not one per row
                for i in 0..orders.len().saturating_sub(1) {
                    self.config.q_cond_dt.enable(&mut region, i)?;
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
                // TABLE side of the lineitem->supplier lookup
                for i in 0..supplier.len() {
                    self.config.q_supp_tbl.enable(&mut region, i)?;
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
                // TABLE side of the nation->region lookup
                for i in 0..region_file.len() {
                    self.config.q_region_tbl.enable(&mut region, i)?;
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

                // the EUROPE hash is one per-proof choice, not one per row
                for i in 0..nation.len().saturating_sub(1) {
                    self.config.q_cond_eu.enable(&mut region, i)?;
                }

                // NR extra padding rows (as q5_obj)
                for i in nation.len()..nr_total {
                    // q_nr_* not enabled, but the link gate IS live here, so the
                    // predicate bit has to be pinned to 0
                    self.config.q_nr_pad.enable(&mut region, i)?;
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

                // The NR relation spans its whole capacity. The selector marks
                // those rows and nothing else, so the Selector Check covers the
                // padding tail too and the two lookups that read NR are gated
                // by the bit rather than by a clean-prefix extent.
                for i in 0..nr_total {
                    self.config.q_row_nr.enable(&mut region, i)?;
                }

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
                    // q_co_* not enabled, but the link gate IS live here, so the
                    // predicate bit has to be pinned to 0
                    self.config.q_co_pad.enable(&mut region, i)?;
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
                    self.config.q_row_co.enable(&mut region, i)?;
                }

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
                        || Value::known(F::from(ls_cflag_written[i])),
                    )?;
                }

                // ---------- LS join lanes ----------
                // Lane l hosts global pipeline rows [l*lane_rows,
                // (l+1)*lane_rows); cells of REAL join rows are collected in
                // global order for the block-wise part_pad linkage below, which
                // is what makes the union of the lanes the partition side of the
                // LS Conservation Check.
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
                        } else {
                            // past |LS^c|: outside the copy constraints, so pinned
                            // to the canonical pad tuple instead
                            lane.q_ls_pad.enable(&mut region, r)?;
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

                // The LS compaction's three sections. These are rows of
                // ls_part_pad, the ONE global group, so the flag count is a
                // count over the union of the lanes. This is the padding layer,
                // not an OBJ condition: it is what tethers the lanes to a
                // column group laid out once and lets them cover |LS^c| +
                // ls_pad_extra rows.
                for i in 0..ls_join_u64.len() {
                    self.config.q_cln_flag[0].enable(&mut region, i)?;
                }
                for i in ls_join_u64.len()..(ls_join_u64.len() + ls_dis_u64.len()) {
                    self.config.q_res_flag[0].enable(&mut region, i)?;
                }
                for i in (ls_join_u64.len() + ls_dis_u64.len())..lineitem.len() {
                    self.config.q_pad_flag[0].enable(&mut region, i)?;
                }

                // Pairwise Consistency needs no assignment of its own: its four
                // lookups read the committed key columns over the ranges
                // `q_ls_join`, `q_row_co` and `q_row_nr` already cover, gated by
                // the three selector bits.

                // ===================== (1) CONSERVATION CHECK =====================
                assign_row_index(
                    &mut region,
                    &self.config.row_idx,
                    nr_total.max(co_total).max(lineitem.len()),
                )?;
                // NR and CO span their DP CAPACITY, not their input length: the
                // loops above fill rows [n, *_total) with (PAD, PAD) and a zero
                // indicator, and the conserved relation is that whole extent.
                // So the witness handed over has to carry the same padding.
                // Slicing the unpadded witness to `*_total` instead panics the
                // moment a released capacity exceeds the input size, which for
                // CO is every budget the sweep uses (eps=0.01 releases 39,060
                // rows on top of |orders| = 15,000).
                let pad_pair = |src: &[Vec<u64>], n: usize, total: usize| -> Vec<Vec<u64>> {
                    (0..total)
                        .map(|i| {
                            if i < n {
                                src[i].clone()
                            } else {
                                vec![PAD_U64, PAD_U64]
                            }
                        })
                        .collect()
                };
                let pad_flag = |src: &[u64], n: usize, total: usize| -> Vec<u64> {
                    (0..total).map(|i| if i < n { src[i] } else { 0 }).collect()
                };
                assign_conserve(
                    &mut region,
                    &self.config.cons_nr,
                    &pad_pair(&nr_pair_u64, nation.len(), nr_total),
                    &pad_flag(&cln_nr, nation.len(), nr_total),
                )?;
                assign_conserve(
                    &mut region,
                    &self.config.cons_co,
                    &pad_pair(&co_pair_u64, orders.len(), co_total),
                    &pad_flag(&cln_co, orders.len(), co_total),
                )?;
                assign_conserve(
                    &mut region,
                    &self.config.cons_ls,
                    &ls_mat_u64[..lineitem.len()],
                    &ls_cln_b[..lineitem.len()],
                )?;

                // ============ CARDINALITY PRESERVATION CHECK, condition (10) ===
                // Over the cluster tree rooted at LS: both multiplicity channels
                // are propagated from the two child bags to LS, accumulated over
                // ALL lineitem rows, and the two totals compared once. Nothing
                // here is per lane.
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

                assign_cp_agg(
                    &mut region,
                    &self.config.cp_agg_co,
                    &cp_rows_co,
                    &cp_stage_co,
                )?;
                assign_cp_agg(
                    &mut region,
                    &self.config.cp_agg_nr,
                    &cp_rows_nr,
                    &cp_stage_nr,
                )?;

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

                let cp_mu: Vec<(u64, u64)> = (0..lineitem.len())
                    .map(|i| {
                        (
                            fetched_co[i].0 * fetched_nr[i].0,
                            ls_cflag_written[i] * fetched_co[i].1 * fetched_nr[i].1,
                        )
                    })
                    .collect();
                for i in 0..lineitem.len() {
                    self.config.q_cp_mu.enable(&mut region, i)?;
                }
                let (cp_all, cp_cln) = assign_cp_root(&mut region, &self.config.cp_root, &cp_mu)?;
                if tamper == Tamper::None {
                    debug_assert_eq!(
                        cp_all, cp_cln,
                        "cardinality preservation: |R^c join| != |R join|"
                    );
                }

                // ---------- shared lane selectors ----------
                for i in 0..lane_rows {
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
                // every lane's ls_sorted sentinel sits at the same row, so one
                // selector pins all of them (the gate is per-lane)
                self.config.q_lane_sentinel.enable(&mut region, lane_rows)?;

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
                    // sentinel row for same_next, pinned by
                    // "lane ls_sorted group-by sentinel is 0"
                    for j in 0..4 {
                        let v = if j == 1 { w.sentinel_nk } else { 0 };
                        region.assign_advice(
                            || "ls_sorted_lane_sentinel",
                            lane.ls_sorted[j],
                            lane_rows,
                            || Value::known(F::from(v)),
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
                            w.sentinel_nk
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
                // sentinel row for merge same_next, pinned by
                // "msort group-by sentinel is PAD"
                for j in 0..2 {
                    let v = if j == 0 { PAD_U64 } else { 0u64 };
                    region.assign_advice(
                        || "msort_sentinel",
                        self.config.msort[j],
                        m_total,
                        || Value::known(F::from(v)),
                    )?;
                }
                self.config.q_m_sentinel.enable(&mut region, m_total)?;

                for i in 1..m_total {
                    let diff = F::from(msort_u64[i][0]) - F::from(msort_u64[i - 1][0]);
                    iz_m_same_prev_chip.assign(&mut region, i, Value::known(diff))?;
                }
                for i in 0..m_total {
                    let next_nk = if i + 1 < m_total {
                        msort_u64[i + 1][0]
                    } else {
                        PAD_U64
                    };
                    let diff = F::from(next_nk) - F::from(msort_u64[i][0]);
                    iz_m_same_next_chip.assign(&mut region, i, Value::known(diff))?;
                    // the degree-1 copy of is_last that the name lookup reads
                    region.assign_advice(
                        || "m_res_is_last",
                        self.config.m_res_is_last,
                        i,
                        || Value::known(F::from((next_nk != msort_u64[i][0]) as u64)),
                    )?;
                }

                // msort[0] nondecreasing over the m_total real rows. Row m_total
                // is the pinned sentinel, so the ladder stops at the pair
                // (m_total-2, m_total-1).
                for i in 0..m_total.saturating_sub(1) {
                    self.config.q_sort_m.enable(&mut region, i)?;
                    let cur = msort_u64[i][0];
                    let next = msort_u64[i + 1][0];
                    lt_m_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(cur)),
                        Value::known(F::from(next)),
                    )?;
                    iz_m_key_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(next) - F::from(cur)),
                    )?;
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

// The lane geometry ([`DP_LANE_K`], [`CHIP_LOAD_ROWS`], [`BLINDING_SLACK`] and
// the [`LANE_ROWS`] derived from them) is declared at the top of this file,
// next to the lane-count helpers that read it.

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

    // k is FIXED at DP_LANE_K: every column group must fit under the 7 u8-range
    // chip loads plus blinding slack.
    //
    // A lane region is `LANE_ROWS + 1` rows (the group-by writes a sentinel at
    // `LANE_ROWS` for its `Rotation::next()` comparisons), but `LANE_ROWS` is
    // already `2^k - CHIP_LOAD_ROWS - BLINDING_SLACK`, and the sentinel plus
    // halo2's real blinding factors sit well inside the 64-row BLINDING_SLACK
    // reserve. Counting the sentinel here on TOP of the full reserve would
    // overcount by exactly one row and reject every legal configuration, so
    // this check uses `LANE_ROWS`, matching g_sql3_obj_dp / g_sql4_obj_dp. The
    // authoritative fit check against the constraint system halo2 actually
    // builds is `tests::structure_fits_base_degree`.
    let tallest = lineitem
        .len() // ls_mat / ls_part_pad, the tallest FIXED group
        .max(orders.len() + co_pad_extra)
        .max(nation.len() + nr_pad_extra)
        .max(LANE_ROWS) // laned pipeline
        .max(c * SEG + 1); // merge region + sentinel row
    if CHIP_LOAD_ROWS + tallest + BLINDING_SLACK > 1usize << DP_LANE_K {
        return Err(format!(
            "tallest column group ({} rows) does not fit k={} ({} rows, of which {} go to \
             the u8 range-table loads and {} to blinding)",
            tallest,
            DP_LANE_K,
            1usize << DP_LANE_K,
            CHIP_LOAD_ROWS,
            BLINDING_SLACK
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

/// Real IPA proving at [`DP_LANE_K`] with the DP release hosted in lanes.
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
        ipa::{commitment::IPACommitmentScheme, multiopen::ProverIPA, strategy::SingleStrategy},
        VerificationStrategy,
    };
    use halo2_proofs::transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    };
    use halo2curves::pasta::{EqAffine, Fp};
    use std::time::Instant;

    assert!(reps >= 1, "reps must be at least 1");
    let (circuit, plan) = dp_lane_setup(privacy);

    // Shared loader: reads the persisted SRS, and generates and persists a
    // degree that is not shipped rather than aborting the sweep on it.
    let params = crate::bench_queries::params_for(DP_LANE_K);

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
    use super::{
        lanes_for, set_config_lanes, MyCircuit, Tamper, BLINDING_SLACK, CHIP_LOAD_ROWS, DP_LANE_K,
        LANE_ROWS, MAX_LANES, SEG,
    };

    use halo2_proofs::dev::{MockProver, VerifyFailure};
    use halo2_proofs::plonk::Circuit;
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

    /// One synthetic circuit at an explicit lane geometry.
    ///
    /// `c` is passed in rather than derived, because two of the negative hooks
    /// change `|LS^c|` and therefore the released length `n`; the lane count is
    /// circuit STRUCTURE and must be stamped before `configure` either way.
    fn lane_circuit(
        tamper: Tamper,
        lane_rows: usize,
        ls_pad_extra: usize,
        c: usize,
    ) -> MyCircuit<Fp> {
        let (customer, orders, lineitem, supplier, nation, region, eur, start_ts, end_ts) =
            synthetic_tables();
        MyCircuit::<Fp> {
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
        }
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

        (lane_circuit(tamper, lane_rows, ls_pad_extra, c), c)
    }

    /// The maximum gate degree of this circuit, at the WIDEST lane count the
    /// structural cap allows. Every soundness patch in this file is written to
    /// fit under the ceiling the Cardinality Preservation Check's own lookups
    /// already set, because a degree rise doubles every FFT of the prover and
    /// at 16 lanes that is the dominant cost of the whole DP sweep.
    ///
    /// The lane count does not change the degree (the lanes replicate columns,
    /// not gate shapes), which this probe also records by measuring both ends.
    #[test]
    fn test_max_gate_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        for c in [1usize, MAX_LANES] {
            set_config_lanes(c);
            let mut cs = ConstraintSystem::<Fp>::default();
            let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
            let degree = cs.degree();
            println!("lanes={} cs.degree() = {}", c, degree);
            println!(
                "  advice={} fixed={} instance={} selectors={} gates={} polys={} lookups={} shuffles={}",
                cs.num_advice_columns(),
                cs.num_fixed_columns(),
                cs.num_instance_columns(),
                cs.num_selectors(),
                cs.gates().len(),
                cs.gates()
                    .iter()
                    .map(|g| g.polynomials().len())
                    .sum::<usize>(),
                cs.lookups().len(),
                cs.shuffles().len(),
            );
            // 8, one above the 7 of `q5_obj_dp.rs`: gating both sides of a
            // Pairwise Consistency lookup by the selector COLUMN rather than
            // by a selector RANGE costs one degree on each side. It buys no
            // FFT, which is the number that matters here, since the lane count
            // is chosen against a PINNED domain: halo2 sizes the extended
            // domain at the next power of two above degree - 1, so 7, 8 and 9
            // all run on an 8x domain and only 10 doubles it.
            assert!(
                degree <= 9,
                "the maximum gate degree rose to {} at {} lanes, which would \
                 double every FFT the prover runs",
                degree,
                c
            );
        }
        set_config_lanes(1);
    }

    /// The lane geometry really fits [`DP_LANE_K`], checked against the
    /// constraint system halo2 builds rather than against the arithmetic
    /// `try_dp_lane_setup` reasons with.
    ///
    /// `LANE_ROWS` fills the domain (`2^k - CHIP_LOAD_ROWS - BLINDING_SLACK`),
    /// so the sentinel row a lane writes at `LANE_ROWS` and halo2's own
    /// unusable rows have to come out of the 64-row reserve. Both ends of the
    /// lane cap are probed: lanes replicate columns, not rows, so the lane
    /// count must not change the answer.
    #[test]
    fn structure_fits_base_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        for c in [1usize, MAX_LANES] {
            set_config_lanes(c);
            let mut cs = ConstraintSystem::<Fp>::default();
            let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
            let blinding = cs.blinding_factors();

            // The reserve LANE_ROWS was subtracted with has to cover what halo2
            // actually reserves, otherwise a full lane runs into unusable rows.
            assert!(
                blinding + 1 <= BLINDING_SLACK,
                "lanes={}: halo2 keeps {} blinding factors (+1 unusable row), \
                 above the {}-row BLINDING_SLACK that sizes LANE_ROWS",
                c,
                blinding,
                BLINDING_SLACK
            );

            // A lane region is LANE_ROWS + 1 rows: the group-by writes a
            // sentinel at LANE_ROWS for its Rotation::next() comparisons.
            let needed = CHIP_LOAD_ROWS + LANE_ROWS + 1 + blinding + 1;
            assert!(
                needed <= 1usize << DP_LANE_K,
                "lanes={}: {} reserved rows + a lane ({} rows + sentinel) + {} blinding \
                 + 1 needs {} rows, but 2^{} = {}",
                c,
                CHIP_LOAD_ROWS,
                LANE_ROWS,
                blinding,
                needed,
                DP_LANE_K,
                1usize << DP_LANE_K
            );

            // The merge region is the other laned column group: SEG rows per
            // lane plus its own sentinel.
            let merge = CHIP_LOAD_ROWS + c * SEG + 1 + blinding + 1;
            assert!(
                merge <= 1usize << DP_LANE_K,
                "lanes={}: the merge region needs {} rows, but 2^{} = {}",
                c,
                merge,
                DP_LANE_K,
                1usize << DP_LANE_K
            );
        }
        set_config_lanes(1);
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

    /// Condition (10), Cardinality Preservation, ACROSS LANES.
    ///
    /// One joinable LS tuple is moved to the residual side and the CO and NR
    /// bags are re-reduced around it, so Conservation and Pairwise Consistency
    /// both still hold and the only thing left to notice is that the clean join
    /// lost an occurrence. The hidden tuple's contribution disappears from lane
    /// 0 alone, which is exactly the shape a PER-LANE equality would let a
    /// prover rebalance; the root sum here is accumulated over all of lineitem
    /// and compared once, so it cannot be.
    #[test]
    fn mock_reject_hidden_clean_tuple() {
        // |LS^c| drops to 5, so n = 11 and the 3 four-row lanes still hold it
        let circuit = lane_circuit(Tamper::HideOneCleanTuple, 4, 6, 3);
        set_config_lanes(3);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(1u64)]]).unwrap();
        let failures = prover
            .verify()
            .expect_err("condition (10) accepted a hidden joinable tuple");
        // `all`, not `any`: this direction must reject through the single
        // root-sum equality of the Cardinality Preservation Check and through
        // NOTHING else. A new constraint that made this witness fail for some
        // other reason would silently destroy the evidence that condition (10)
        // is doing the work, and an `any` assertion would not notice.
        assert!(
            !failures.is_empty()
                && failures
                    .iter()
                    .all(|f| format!("{:?}", f).contains("cardinality preservation")),
            "the circuit rejected, but not (only) through the Cardinality \
             Preservation Check: {:?}",
            failures
        );
    }

    /// Condition (9), Pairwise Consistency, ACROSS LANES.
    ///
    /// No reduction at all: every tuple that passes its own predicate is
    /// declared clean, so all three residual sections are empty, Conservation
    /// holds and both channels of condition (10) agree row by row. The dangling
    /// clean LS tuples land in lane 1 and lane 2 (global pipeline rows 6, 7 and
    /// 8 with four-row lanes), so a per-lane version of this check could be
    /// satisfied lane by lane; the LS side of all four lookups is `ls_part_pad`,
    /// the union of the lanes, so one lookup sees all of them.
    #[test]
    fn mock_reject_unreduced_clean_instance() {
        // |LS^c| rises to 9, so n = 15 and the release needs a 4th lane
        let circuit = lane_circuit(Tamper::MarkAllClean, 4, 6, 4);
        set_config_lanes(4);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(1u64)]]).unwrap();
        let failures = prover
            .verify()
            .expect_err("condition (9) accepted an unreduced clean instance");
        // "attach name from NR_out (merge)" fires alongside the Pairwise
        // Consistency lookups, because a non-European group has no
        // (nationkey, name) row to be labelled from; that is a downstream
        // consequence of the same unreduced witness, not a substitute for
        // condition (9), so the assertion still demands a "pw: " lookup.
        assert!(
            failures.iter().any(|f| matches!(
                f,
                VerifyFailure::Lookup { name, .. } if name.starts_with("pw: ")
            )),
            "the circuit rejected, but not through a Pairwise Consistency \
             lookup: {:?}",
            failures
        );
    }

    /// Condition (7), Conservation, and specifically the clean indicator riding
    /// as the fifth column of the LS shuffle: claiming one extra clean
    /// occurrence on the INPUT side while the partition side keeps the honest
    /// sections must break the multiset equality. Without this binding the
    /// `HideOneCleanTuple` direction above would be escapable by simply lying
    /// about `cflag_ls`.
    #[test]
    fn mock_reject_clean_flag_lie() {
        let (circuit, c) = three_lane_circuit(Tamper::CleanFlagLie);
        set_config_lanes(c);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(1u64)]]).unwrap();
        let failures = prover
            .verify()
            .expect_err("condition (7) accepted an unbound clean indicator");
        assert!(
            failures
                .iter()
                .any(|f| matches!(f, VerifyFailure::Shuffle { .. })),
            "the clean-indicator lie did not break a Conservation shuffle: {:?}",
            failures
        );
    }

    /// The lane padding tail is past |LS^c| and so outside every copy
    /// constraint, yet `perm_lsort` carries it into the lane aggregation and on
    /// into the reported revenue. Breaking the canonical pad tuple there must
    /// be rejected.
    #[test]
    fn mock_reject_tampered_lane_pad_row() {
        let (circuit, c) = three_lane_circuit(Tamper::LanePadRow);
        set_config_lanes(c);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(1u64)]]).unwrap();
        assert!(
            prover.verify().is_err(),
            "a lane pad row outside the canonical pad tuple must not verify"
        );
    }

    /// The cell one row past each lane's sorted view. The whole rest of the lane
    /// witness is derived consistently from the tampered value, so this is the
    /// real attack and not just an inconsistent assignment: lane 0's sorted view
    /// is [nk1, nk1, nk2, nk2], the boundary cell claims nk2 again, the last row
    /// stops being a group end and nation nk2 loses its 12000 from lane 0's
    /// partial. Everything else -- the emit gate, the drain shuffle, the merge --
    /// agrees with it, so the boundary gate is the ONLY thing that can reject.
    #[test]
    fn mock_reject_tampered_lane_sentinel() {
        let (circuit, c) = three_lane_circuit(Tamper::LaneSentinel);
        set_config_lanes(c);
        let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(1u64)]]).unwrap();
        let failures = prover
            .verify()
            .expect_err("an unpinned sorted-view boundary cell was accepted");
        assert!(
            !failures.is_empty()
                && failures
                    .iter()
                    .all(|f| format!("{:?}", f).contains("group-by sentinel is 0")),
            "the circuit rejected, but not (only) through the sorted-view \
             boundary gate: {:?}",
            failures
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

    /// Real k = 17 IPA proving over full TPC-H with the DP release hosted in
    /// lanes. Hours on the production tables, so it is ignored by default.
    /// Run it explicitly, for example:
    ///
    ///   VPJOIN_PRIVACY=dp VPJOIN_EPS=0.1 VPJOIN_DELTA=1e-5 cargo test \
    ///     --release sql::q5_obj_dp::tests::test_dp_lanes -- --ignored --nocapture
    ///
    /// The same code path is what `cargo run --bin dp_lane_bench -- q5` drives.
    #[test]
    #[ignore = "real k=17 IPA proving over full TPC-H; run explicitly"]
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

        let proof_path = crate::paths::proof_file("proof_q5_dp_lanes_new");
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
