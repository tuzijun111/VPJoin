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
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

// pub(crate) so the multi-lane DP variant (`q5_obj_dp.rs`) shares one
// definition of the PAD/sentinel discipline instead of copying it.
pub(crate) const NUM_BYTES: usize = 7;
pub(crate) const MAX_SENTINEL: u64 = (1u64 << (8 * NUM_BYTES)) - 1;
pub(crate) const PAD_U64: u64 = MAX_SENTINEL;

pub(crate) const SCALE: u64 = 1000;
pub(crate) const PAD_REV: u64 = 0;

// pack (orderkey, nationkey_shift)
pub(crate) const SHIFT_NATION: u64 = 1u64 << 8; // nationkey_shift <= 25+1 fits

/// Test hook, off in every benchmark path: when set, the prover moves one
/// joinable LS tuple to the residual side and re-reduces the CO and NR bags
/// around it, so the partition still passes Conservation, Non-Membership and
/// Pairwise Consistency and only condition (4) can catch it. This is exactly
/// the cheat a residual-side-only argument misses, so the negative test in this
/// module is what shows the Cardinality Preservation Check is not vacuous.
thread_local! {
    static HIDE_ONE_CLEAN_TUPLE_TL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// THREAD-LOCAL: `cargo test` runs modules in parallel threads while synthesis
/// is single-threaded, so a process-wide flag corrupts every other circuit being
/// assigned at that moment. That is not hypothetical -- as an `AtomicBool` this
/// made the q5 witness-binding tests pass in isolation and fail in the full
/// suite. Same fix as `conserve_idx::set_misplace_one_occurrence`.
pub fn set_hide_one_clean_tuple(on: bool) {
    HIDE_ONE_CLEAN_TUPLE_TL.with(|c| c.set(on));
}

fn hide_one_clean_tuple() -> bool {
    HIDE_ONE_CLEAN_TUPLE_TL.with(|c| c.get())
}

/// Test hook, off in every benchmark path: when set, the prover skips the
/// semijoin reduction entirely and declares every tuple that passes its
/// predicate clean, so all three residual sections are empty. Conservation still
/// holds and both channels of condition (4) then agree row by row, so this is the
/// escape that only Pairwise Consistency can close, and the negative direction
/// for it in this module is what shows condition (3) is doing work.
thread_local! {
    static MARK_ALL_CLEAN_TL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// THREAD-LOCAL: `cargo test` runs modules in parallel threads while synthesis
/// is single-threaded, so a process-wide flag corrupts every other circuit being
/// assigned at that moment. That is not hypothetical -- as an `AtomicBool` this
/// made the q5 witness-binding tests pass in isolation and fail in the full
/// suite. Same fix as `conserve_idx::set_misplace_one_occurrence`.
pub fn set_mark_all_clean(on: bool) {
    MARK_ALL_CLEAN_TL.with(|c| c.set(on));
}

fn mark_all_clean() -> bool {
    MARK_ALL_CLEAN_TL.with(|c| c.get())
}

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

#[derive(Clone, Debug)]
pub struct Q5Config<F: Field + Ord> {
    // ---------------- base tables ----------------
    // customer: [c_custkey, c_nationkey_shift]
    customer: Vec<Column<Advice>>,
    // orders: [o_orderdate_ts, o_custkey, o_orderkey]
    orders: Vec<Column<Advice>>,
    // lineitem: [l_orderkey, l_suppkey, l_ext, l_disc] (scaled)
    lineitem: Vec<Column<Advice>>,
    // supplier: [s_suppkey, s_nationkey_shift]
    supplier: Vec<Column<Advice>>,
    // nation: [n_nationkey_shift, n_name_hash, n_regionkey_shift]
    nation: Vec<Column<Advice>>,
    // region: [r_regionkey_shift, r_name_hash]
    region_file: Vec<Column<Advice>>,

    // ---------------- conditions ----------------
    cond_europe: Column<Advice>,
    cond_start: Column<Advice>,
    cond_end: Column<Advice>,
    // one per-proof choice instead of one per-row choice: see
    // "query parameters are constant down the column" in `configure`
    q_cond_eu: Selector,
    q_cond_dt: Selector,

    // ---------------- bag materialization: NR ----------------
    q_nr_join: Selector,              // enable nation->region tuple lookup
    q_nr_pred: Selector,              // enable isZero (r_name == EUROPE)
    q_region_tbl: Selector,           // TABLE side of the nation->region lookup
    q_nr_pad: Selector,               // rows [nation.len(), nr_total): keep == 0
    nr_rname: Column<Advice>,         // looked-up region name hash for nation row
    nr_keep: Column<Advice>,          // boolean
    // ---------------- (1) Conservation Check ----------------
    // R^_i == R^_i^c U+ R^_i^r over the INDEXED cluster relation, one
    // permutation each. A relation laid out at a DP capacity has its padding
    // rows in the residual part, since their indicator is zero.
    row_idx: RowIndexConfig,
    cons_nr: ConserveConfig,
    cons_co: ConserveConfig,
    cons_ls: ConserveConfig,

    cflag_nr: Column<Advice>,     // the selector bit c per NR row
    nr_pair: Vec<Column<Advice>>, // [nk_shift, n_name_hash] per nation row
    iz_nr: IsZeroConfig<F>,

    // ---------------- bag materialization: CO ----------------
    q_oc_join: Selector,  // orders->customer tuple lookup
    q_cust_tbl: Selector, // TABLE side of the orders->customer lookup
    q_co_ge: Selector,    // start <= odate (LtEqGeneric)
    q_co_lt: Selector,    // odate < end (LtChip)
    q_co_and: Selector,   // keep = ge*lt
    q_co_pad: Selector,   // rows [orders.len(), co_total): keep == 0
    co_ge_ok: Column<Advice>,
    co_lt_ok: Column<Advice>,
    co_keep: Column<Advice>,      // boolean
    cflag_co: Column<Advice>,     // the selector bit c per CO row
    co_nk: Column<Advice>,        // looked-up nationkey_shift for order row
    co_pair: Vec<Column<Advice>>, // [okey, nk_shift] per order row
    co_pkey: Column<Advice>,      // co_pair[0]*SHIFT_NATION + co_pair[1]
    lteq_start_le_odate: LtEqGenericConfig<F, NUM_BYTES>,
    lt_odate_lt_end: LtConfig<F, NUM_BYTES>,

    // ---------------- bag materialization: LS ----------------
    q_ls_join: Selector,         // lineitem->supplier tuple lookup
    q_supp_tbl: Selector,        // TABLE side of the lineitem->supplier lookup
    ls_mat: Vec<Column<Advice>>, // [okey, nk_shift, ext, disc] per lineitem row
    cflag_ls: Column<Advice>,    // clean indicator per LS row
    ls_pkey: Column<Advice>,     // ls_mat[0]*SHIFT_NATION + ls_mat[1]

    // ---------------- LS partition: join/disjoin ----------------
    ls_join: Vec<Column<Advice>>,
    ls_disjoin: Vec<Column<Advice>>,
    ls_part_pad: Vec<Column<Advice>>, // 5 cols: the tuple plus the clean flag
    perm_ls: PermAnyConfig,
    // rows [|LS^c|, n) of ls_join: the aggregation tail, pinned to the canonical
    // PAD tuple because `perm_lsort` carries those rows into the group-by
    q_ls_pad: Selector,

    // -------- (1) Selector Check / (2) Pairwise Consistency --------
    // One complex selector per cluster relation over ITS WHOLE CAPACITY: NR is
    // laid out over nr_total rows, CO over co_total, LS over |lineitem|. It
    // carries the booleanity and predicate gates of the Selector Check and
    // gates both sides of the four Pairwise Consistency lookups, so no
    // data-dependent row range reaches the fixed columns.
    q_row_nr: Selector, // rows [0, nr_total)
    q_row_co: Selector, // rows [0, co_total)

    // ---------------- Cardinality Preservation Check (condition (4)) ----------------
    // |R^c join| == |R join| over the cluster tree rooted at LS
    cp_agg_co: CpAggConfig<F, NUM_BYTES>, // child CO, keyed by the packed key
    cp_agg_nr: CpAggConfig<F, NUM_BYTES>, // child NR, keyed by nk_shift
    cp_join_co: CpJoinConfig<F, NUM_BYTES>, // LS -> CO
    cp_join_nr: CpJoinConfig<F, NUM_BYTES>, // LS -> NR
    cp_root: CpRootConfig,
    q_cp_mu: Selector, // the two root product gates

    // The LS compaction, which is NOT an OBJ condition: it is the padding
    // layer that lets the group-by run over |LS^c| + ls_pad_extra rows instead
    // of over every committed lineitem row. See the note in `configure`.
    q_cln_flag: Vec<Selector>, // rows of LS^c inside ls_part_pad: flag == 1
    q_res_flag: Vec<Selector>, // rows of LS^r: flag == 0
    q_pad_flag: Vec<Selector>, // pad tail: the whole row is the PAD tuple

    // ---------------- aggregation over contributing LS ----------------
    ls_sorted: Vec<Column<Advice>>,
    perm_lsort: PermAnyConfig,
    q_ls_sentinel: Selector, // row n of ls_sorted, read by iz_same_next at n-1

    // ls_sorted[1] (the GROUP BY column) is nondecreasing
    q_sort_ls: Selector,
    lt_nk_cur_next: LtConfig<F, NUM_BYTES>,
    iz_nk_eq: IsZeroConfig<F>,

    q_line: Selector,
    q_first: Selector,
    q_accu: Selector,

    line_rev: Column<Advice>,
    run_sum: Column<Advice>,
    iz_same_prev: IsZeroConfig<F>,
    iz_same_next: IsZeroConfig<F>,

    // emitted padded result aligned with ls_sorted rows:
    // [nationkey_shift, n_name_hash, revenue]
    res_pad: Vec<Column<Advice>>,

    // attach (nk, name) via lookup into nr_out_pad
    q_res_lookup: Selector,
    // a degree-1 stand-in for `1 - iz_same_next.expr()`, so the name lookup can
    // afford to gate its TABLE side without raising the circuit's degree
    res_is_last: Column<Advice>,

    // ORDER BY revenue DESC
    res_sorted: Vec<Column<Advice>>,
    perm_res: PermAnyConfig,
    q_sort_res: Selector,
    lteq_rev_next_le_cur: LtEqGenericConfig<F, NUM_BYTES>,

    // public
    instance: Column<Instance>,
    instance_test: Column<Advice>,
}

#[derive(Clone, Debug)]
pub struct Q5Chip<F: Field + Ord> {
    config: Q5Config<F>,
}

impl<F: Field + Ord> Q5Chip<F> {
    pub fn construct(config: Q5Config<F>) -> Self {
        Self { config }
    }

    // pub(crate): reused by the multi-lane DP variant (`q5_obj_dp.rs`).
    pub(crate) fn assign_table_u64(
        region: &mut Region<'_, F>,
        tag: &'static str,
        cols: &[Column<Advice>],
        rows: &[Vec<u64>],
    ) -> Result<Vec<Vec<AssignedCell<F, F>>>, Error> {
        let mut out: Vec<Vec<AssignedCell<F, F>>> = Vec::with_capacity(rows.len());
        for (i, r) in rows.iter().enumerate() {
            let mut row_cells = Vec::with_capacity(cols.len());
            for (j, &v) in r.iter().enumerate() {
                let cell = region.assign_advice(|| tag, cols[j], i, || Value::known(F::from(v)))?;
                row_cells.push(cell);
            }
            out.push(row_cells);
        }
        Ok(out)
    }

    pub(crate) fn assign_table_f(
        region: &mut Region<'_, F>,
        tag: &'static str,
        cols: &[Column<Advice>],
        rows: &[Vec<F>],
    ) -> Result<(), Error> {
        for (i, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                region.assign_advice(|| tag, cols[j], i, || Value::known(v))?;
            }
        }
        Ok(())
    }

    pub(crate) fn assign_part_pad_and_link(
        region: &mut Region<'_, F>,
        tag: &'static str,
        part_cols: &[Column<Advice>],
        part_rows: &[Vec<F>],
        join_cells: &[Vec<AssignedCell<F, F>>],
        dis_cells: &[Vec<AssignedCell<F, F>>],
    ) -> Result<(), Error> {
        let join_len = join_cells.len();
        let dis_len = dis_cells.len();

        for (i, r) in part_rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                let part_cell =
                    region.assign_advice(|| tag, part_cols[j], i, || Value::known(v))?;

                // The last column of part_rows is the clean indicator, which has
                // no counterpart in the join/disjoin tables: it is pinned by
                // q_cln_flag / q_res_flag instead, so it is skipped here.
                if i < join_len {
                    if j < join_cells[i].len() {
                        region.constrain_equal(part_cell.cell(), join_cells[i][j].cell())?;
                    }
                } else if i < join_len + dis_len {
                    if j < dis_cells[i - join_len].len() {
                        region
                            .constrain_equal(part_cell.cell(), dis_cells[i - join_len][j].cell())?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> Q5Config<F> {
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

        // These sixteen columns ARE the committed input columns, in the order
        // `bench_queries::TpchInput::columns` publishes them for Q5 (customer 2,
        // orders 3, lineitem 4, supplier 2, nation 3, region 2). Equality-enable
        // them so a bound wrapper can copy-constrain the binding's data columns
        // to THESE cells (`inline_bind::tie_columns`). `enable_equality` is
        // idempotent, so re-enabling elsewhere is safe.
        //
        // NOTE for whoever writes that tie: five of the sixteen are NOT stored
        // verbatim. `assign_with_input_cells` writes `c_nationkey + 1`,
        // `s_nationkey + 1`, `n_nationkey + 1`, `n_regionkey + 1` and
        // `r_regionkey + 1`, because 0 is reserved as the "no match" sentinel of
        // the three tuple lookups and TPC-H really has nationkey 0. See the
        // doc comment on `assign_with_input_cells`.
        for c in customer
            .iter()
            .chain(orders.iter())
            .chain(lineitem.iter())
            .chain(supplier.iter())
            .chain(nation.iter())
            .chain(region_file.iter())
        {
            meta.enable_equality(*c);
        }

        // conditions
        let cond_europe = meta.advice_column();
        let cond_start = meta.advice_column();
        let cond_end = meta.advice_column();

        // -------- query parameters are constant down the column --------
        // The three parameter columns are plain advice, read only at
        // Rotation::cur inside the per-row predicate chips. Without a cross-row
        // tie each row carries its OWN window: `co_keep = ge * lt` accepts an
        // out-of-window order by widening `cond_start`/`cond_end` on that row
        // alone, and `nr_keep` accepts a non-European nation by setting
        // `cond_europe` to that row's `nr_rname`. The circuit would then certify
        // the answer of no single Q5 instance.
        //
        // These two gates compare Rotation::cur with Rotation::next over every
        // row the parameter is read on but the last, which turns a per-row
        // prover choice into one per-proof choice. Binding the value to a public
        // input needs an instance-vector change and is out of this file's scope;
        // see the report.
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

        // ---------------- NR materialization ----------------
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
        // The NR relation occupies its whole capacity, rows [0, nr_total): the
        // real nations first, then padding rows whose predicate bit `q_nr_pad`
        // pins to 0. That capacity is the DP padding layer and is unchanged
        // here; what is gone is the `[clean | residual | pad]` group and the
        // Conservation shuffle that used to bind the indicator to it.
        let q_row_nr = meta.complex_selector();

        // nation->region tuple lookup: (n_regionkey_shift, nr_rname) in region(r_regionkey_shift, r_name_hash)
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

        // isZero: nr_rname == EUROPE
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

        // nr_pair is just nation columns (shifted key + name hash)
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
        // condition (4) by a fraction, which is exactly the compensation the
        // check exists to forbid.
        // ---------- (1) Selector Check on NR ----------
        // Under `q_row_nr`, so it covers the capacity tail as well as the real
        // nations. Booleanity matters because the clean channel of (3)
        // multiplies by this bit: a value such as 1 + 1/k would inflate that
        // channel by exactly the fraction the check exists to forbid, and
        // `card_preserve` puts no range check on v_cln. The predicate half is
        // what `q5_obj.rs` got structurally from the link gate padding the
        // indicator away wherever `nr_keep` was 0; on a capacity row `nr_keep`
        // is pinned to 0, so it also confines the selection to the real rows,
        // which is the cluster adjustment.
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

        // ---------------- CO materialization ----------------
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

        // orders->customer tuple lookup: (o_custkey, co_nk) in customer(c_custkey, c_nationkey_shift)
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

        // co_pair = [o_orderkey, co_nk]
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

        // start <= odate (LtEqGeneric)
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

        // odate < end (LtChip)
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

        // co_keep = ge*lt
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
        meta.create_gate("cflag_co is a bit", |m| {
            let q = m.query_selector(q_co_and);
            let c = m.query_advice(cflag_co, Rotation::cur());
            vec![q * c.clone() * (Expression::Constant(F::ONE) - c)]
        });
        meta.create_gate("CO pad row keeps nothing", |m| {
            let q = m.query_selector(q_co_pad);
            vec![q * m.query_advice(co_keep, Rotation::cur())]
        });

        // ---------------- LS materialization (lineitem ⋈ supplier) ----------------
        let q_ls_join = meta.complex_selector();
        let q_supp_tbl = meta.complex_selector();
        let ls_mat = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        // tuple lookup (l_suppkey, ls_mat.nk) in supplier(s_suppkey, s_nationkey_shift)
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

        // `cflag_ls` is already forced boolean indirectly, because the LS
        // partition is padded only to lineitem.len() and so has no pad tail for a
        // non-boolean value to hide in. State it directly anyway: it is the
        // factor of the root clean multiplicity, and one degree-3 gate is cheaper
        // than the argument.
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

        // ls_mat copies orderkey/ext/disc from lineitem
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
        let ls_join = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let ls_disjoin = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        // One column wider than in q5_obj.rs: the clean indicator rides along,
        // so the LS Conservation Check binds it.
        let ls_part_pad = (0..5).map(|_| meta.advice_column()).collect::<Vec<_>>();

        for &c in ls_join
            .iter()
            .chain(ls_disjoin.iter())
            .chain(ls_part_pad.iter())
        {
            meta.enable_equality(c);
        }

        let perm_ls = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            let mut ls_in = ls_mat.clone();
            ls_in.push(cflag_ls);
            PermAnyChip::configure(meta, q1, q2, ls_in, ls_part_pad.clone())
        };

        // -------- partition side of the clean indicator: 1 on R^c rows, 0 on R^r --------
        // The LS compaction is laid out as [clean rows | residual rows | pad
        // rows], so one selector per section pins its flag column. This is the
        // one place a `[clean | residual | pad]` group survives, and it is NOT
        // an OBJ condition any more: the gate's three conditions all read the
        // committed rows through `cflag_ls`, which has its own booleanity gate.
        // What the shuffle and these pins buy is the PADDING LAYER: they certify
        // that `ls_join`'s prefix holds exactly the selected LS tuples, which is
        // what lets the group-by, its sortedness ladder, the emitted result and
        // both ORDER BY passes run over |LS^c| + ls_pad_extra rows instead of
        // over every committed lineitem row. Dropping it would make Q5 fully
        // oblivious at the cost of an aggregation as tall as `lineitem`, and
        // would take the DP capacity knob with it.
        //
        // The pad tail needs the same pin the clean and residual sections get:
        // the shuffle fixes only the MULTISET, so without it a surplus flag = 1
        // row could sit past the residual section, outside the range
        // `q_cln_flag` covers.
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
                // pad tuple: PAD_U64 in every key/value column, 0 in the flag
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
        // One permutation per cluster relation, between the indexed relation
        // and the concatenation of its two parts. The indices are distinct, so
        // the indexed relation is a set even though the relation is a bag, and
        // the single permutation rules out an occurrence being fabricated,
        // lost, duplicated or counted on both sides: no Non-Membership Check.
        //
        // LS carries a second permutation, `perm_ls`, which is NOT this
        // condition: it is the compaction that certifies `ls_join`'s prefix and
        // lets the aggregation run over |LS^c| + ls_pad_extra rows. The two are
        // kept apart on purpose, the OBJ condition from the padding layer.
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
        // `q5_obj.rs` read these off the clean prefixes of the three partition
        // groups, gated by selectors whose extent was |NR^c|, |CO^c| and
        // |LS^c|. Those extents are baked into fixed columns at keygen, so they
        // leaked the clean/residual ratio the paper claims never to reveal, and
        // a malicious circuit generator could publish a vk with all three empty
        // and make the condition vacuous. Here the gating factor is the
        // committed bit and the three selectors mark nothing but each
        // relation's capacity, which is public.
        let q_pw_ls = q_ls_join;

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
            q_pw_ls, cflag_ls, ls_pkey, q_row_co, cflag_co, co_pkey);
        pw_edge("pw: CO^c pkey in LS^c pkey",
            q_row_co, cflag_co, co_pkey, q_pw_ls, cflag_ls, ls_pkey);

        // edge (LS, NR) on nationkey_shift
        pw_edge("pw: LS^c nationkey in NR^c nationkey",
            q_pw_ls, cflag_ls, ls_mat[1], q_row_nr, cflag_nr, nr_pair[0]);
        pw_edge("pw: NR^c nationkey in LS^c nationkey",
            q_row_nr, cflag_nr, nr_pair[0], q_pw_ls, cflag_ls, ls_mat[1]);

        // ---------------- Cardinality Preservation Check (condition (4)) ----------------
        // One fixed column serves every Lt chip of the check, so the whole check
        // costs a single u8 range table.
        let cp_u8 = meta.fixed_column();

        // Children of the root. Both are leaves, so their two multiplicity
        // columns are columns the circuit already has: the predicate bit is the
        // input channel and the bound indicator (keep * c, i.e. the last column
        // of the filt_pad side of the Conservation Check) is the clean one.
        let cp_agg_co = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            co_pkey, // packed (o_orderkey, nk_shift)
            co_keep,
            cflag_co,
            PAD_U64,
        );
        let cp_agg_nr = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            nr_pair[0], // nk_shift
            nr_keep,
            cflag_nr,
            PAD_U64,
        );

        // Parent side, on the rows of LS (one per lineitem row).
        let cp_join_co = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, ls_pkey);
        let cp_join_nr = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, ls_mat[1]);
        wire_cp_edge(meta, &cp_join_co, &cp_agg_co, ls_pkey);
        wire_cp_edge(meta, &cp_join_nr, &cp_agg_nr, ls_mat[1]);

        // Root multiplicities and the single equality that compares the two join
        // cardinalities.
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

        // ---------------- aggregation ----------------
        let ls_sorted = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let perm_lsort = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            // NOTE: still permute ls_join <-> ls_sorted, but you may enable more rows in witness
            // if you pass ls_pad_extra; those additional rows must be padded consistently on both sides.
            PermAnyChip::configure(meta, q1, q2, ls_join.clone(), ls_sorted.clone())
        };

        // Rows [|LS^c|, n) of ls_join exist only to give `perm_lsort` a left-hand
        // side when ls_pad_extra > 0. They are outside every other argument: the
        // copy constraints of `assign_part_pad_and_link` stop at |LS^c|, and both
        // sides of the four Pairwise Consistency lookups are gated by q_pw_ls,
        // which stops there too. Yet the shuffle carries whatever sits in them
        // into ls_sorted, hence into line_rev, run_sum and the reported revenue,
        // so as free advice they are a direct forgery of the query answer: the
        // benchmarked Legacy configuration has 59452 of them. Pinning them to the
        // canonical pad tuple [0, PAD, 0, 0] is what makes them inert -- gating
        // them out of the shuffle instead would change what Conservation proves.
        let q_ls_pad = meta.selector();
        {
            let cols = ls_join.clone();
            meta.create_gate("ls_join aggregation tail is PAD", move |m| {
                let q = m.query_selector(q_ls_pad);
                let want = [F::ZERO, F::from(PAD_U64), F::ZERO, F::ZERO];
                cols.iter()
                    .zip(want.iter())
                    .map(|(&c, &w)| {
                        q.clone() * (m.query_advice(c, Rotation::cur()) - Expression::Constant(w))
                    })
                    .collect::<Vec<_>>()
            });
        }

        // The sentinel row n of ls_sorted, which `iz_same_next` reads at row
        // n-1 to close the last group. `q_sort_ls` stops at the pair (n-2, n-1)
        // and `perm_lsort` covers rows 0..n-1, so nothing else touches row n.
        // Unpinned, setting it equal to the last (largest) sorted nationkey makes
        // iz_same_next report "same group" on the last real row; `emit_res_pad`
        // then FORCES res_pad[n-1] to PAD and the highest-nationkey group vanishes
        // from the answer, with the name lookup going vacuous along with it. This
        // is the hole `card_preserve`'s "cp: sorted view sentinel is PAD" closes
        // for the aggregation views it owns; Q5's own view needs its own gate.
        //
        // The pinned value is PAD_U64, where `q5_obj.rs` pinned 0. The change
        // follows the name-attachment lookup below. There the table was the pad
        // tail of `nr_out_pad`, which carried the (PAD, PAD) tuple, so a
        // trailing PAD-keyed group could close, emit (PAD, PAD, 0) and match.
        // Here the table is the committed NR pair gated by `cflag_nr`, and a
        // capacity row contributes the all-zero tuple, not (PAD, PAD), so a
        // PAD-keyed group has nothing to attach to and must never close.
        //
        // PAD_U64 does that and nothing else: every real nationkey is stored
        // shifted by +1, so a real group still ends the moment the next row
        // carries a different key or the first PAD row, and only a run of PAD
        // rows runs into the sentinel unclosed. Those rows then take the
        // not_last branch of `emit_res_pad`, which is the canonical pad result
        // triple.
        let q_ls_sentinel = meta.selector();
        meta.create_gate("ls_sorted group-by sentinel is PAD", |m| {
            let q = m.query_selector(q_ls_sentinel);
            vec![
                q * (m.query_advice(ls_sorted[1], Rotation::cur())
                    - Expression::Constant(F::from(PAD_U64))),
            ]
        });

        // The group-by below finds group boundaries by comparing ls_sorted[1]
        // with its neighbours, which is only sound if that column really is
        // sorted. `perm_lsort` ties ls_sorted to ls_join as a MULTISET and says
        // nothing about ORDER, so without this gate a prover could lay one
        // nationkey out as two non-adjacent runs; each run would look like its
        // own group and emit its own partial revenue. Same shape as
        // `configure_cp_agg`'s "cp: sorted key nondecreasing", and it shares the
        // same u8 range table.
        let q_sort_ls = meta.selector();
        let aux_nk_eq = meta.advice_column();
        let iz_nk_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort_ls),
            |m| {
                m.query_advice(ls_sorted[1], Rotation::next())
                    - m.query_advice(ls_sorted[1], Rotation::cur())
            },
            aux_nk_eq,
        );
        let lt_nk_cur_next = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            cp_u8,
            |m| m.query_selector(q_sort_ls),
            |m| m.query_advice(ls_sorted[1], Rotation::cur()),
            |m| m.query_advice(ls_sorted[1], Rotation::next()),
        );
        meta.create_gate("sorted ls nk is nondecreasing", |m| {
            let q = m.query_selector(q_sort_ls);
            let le = lt_nk_cur_next.is_lt(m, None) + iz_nk_eq.expr();
            vec![q * (le - Expression::Constant(F::ONE))]
        });

        let q_line = meta.selector();
        let q_first = meta.selector();
        let q_accu = meta.selector();

        let line_rev = meta.advice_column();
        let run_sum = meta.advice_column();

        let aux_same_prev = meta.advice_column();
        let aux_same_next = meta.advice_column();
        let iz_same_prev = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_accu),
            |m| {
                m.query_advice(ls_sorted[1], Rotation::cur())
                    - m.query_advice(ls_sorted[1], Rotation::prev())
            },
            aux_same_prev,
        );
        let iz_same_next = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_line),
            |m| {
                m.query_advice(ls_sorted[1], Rotation::next())
                    - m.query_advice(ls_sorted[1], Rotation::cur())
            },
            aux_same_next,
        );

        // line_rev = ext*(SCALE-disc)
        meta.create_gate("line_rev", |m| {
            let q = m.query_selector(q_line);
            let ext = m.query_advice(ls_sorted[2], Rotation::cur());
            let disc = m.query_advice(ls_sorted[3], Rotation::cur());
            let lr = m.query_advice(line_rev, Rotation::cur());
            let scale = Expression::Constant(F::from(SCALE));
            vec![q * (lr - ext * (scale - disc))]
        });
        meta.create_gate("run_sum_first", |m| {
            let q = m.query_selector(q_first);
            let rs = m.query_advice(run_sum, Rotation::cur());
            let lr = m.query_advice(line_rev, Rotation::cur());
            vec![q * (rs - lr)]
        });
        meta.create_gate("run_sum_accu", |m| {
            let q = m.query_selector(q_accu);
            let same = iz_same_prev.expr();
            let rs_cur = m.query_advice(run_sum, Rotation::cur());
            let rs_prev = m.query_advice(run_sum, Rotation::prev());
            let lr = m.query_advice(line_rev, Rotation::cur());
            vec![q * (rs_cur - (same * rs_prev + lr))]
        });

        let res_pad = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let q_res_lookup = meta.complex_selector();

        // emit group row only at last row of each nation group
        meta.create_gate("emit_res_pad", |m| {
            let q = m.query_selector(q_line);
            let one = Expression::Constant(F::ONE);
            let is_last = one.clone() - iz_same_next.expr();
            let not_last = one.clone() - is_last.clone();

            let nk = m.query_advice(ls_sorted[1], Rotation::cur());
            let rs = m.query_advice(run_sum, Rotation::cur());

            let out_nk = m.query_advice(res_pad[0], Rotation::cur());
            let out_nm = m.query_advice(res_pad[1], Rotation::cur());
            let out_rev = m.query_advice(res_pad[2], Rotation::cur());

            let pad_nk = Expression::Constant(F::from(PAD_U64));
            let pad_nm = Expression::Constant(F::from(PAD_U64));
            let pad_rev = Expression::Constant(F::from(PAD_REV));

            vec![
                q.clone() * (out_nk - (is_last.clone() * nk + not_last.clone() * pad_nk)),
                q.clone() * (out_rev - (is_last.clone() * rs + not_last.clone() * pad_rev)),
                q * not_last * (out_nm - pad_nm),
            ]
        });

        // `res_is_last` is `1 - iz_same_next.expr()` copied into an advice cell on
        // every row the group-by covers. It exists purely for degree: the lookup
        // below needs a gated TABLE side, and with the degree-2 IsZero expression
        // still on the input side that lookup would cost 2 + 4 + 2 = 8 and push the
        // whole circuit's degree up. Through this column the input side is degree 3
        // and the lookup stays at 7, which is what the Cardinality Preservation
        // Check's own gap lookup already costs.
        let res_is_last = meta.advice_column();
        meta.create_gate("res_is_last = is_last", |m| {
            let q = m.query_selector(q_line);
            let one = Expression::Constant(F::ONE);
            let is_last = one - iz_same_next.expr();
            vec![q * (m.query_advice(res_is_last, Rotation::cur()) - is_last)]
        });

        // attach (nk, name) by looking the emitted group up in the SELECTED NR
        // rows. The table side is the committed pair gated by `cflag_nr`, so a
        // deselected nation, a capacity padding row and a row past nr_total all
        // contribute the all-zero tuple, which is what a gated-off input row
        // reads; the shift by one keeps that dummy away from any real pair.
        //
        // `q5_obj.rs` gated this table with `perm_nr.q_perm2` over the assigned
        // rows of `nr_out_pad`, which put every RESIDUAL nation in the table as
        // well: a group could be labelled with the name of a nation the
        // reduction had dropped. Gating by the bit is strictly tighter.
        let (res_nk, res_nm) = (res_pad[0], res_pad[1]);
        let nr_pair_l = nr_pair.clone();
        meta.lookup_any("attach name from NR", move |m| {
            let one = Expression::Constant(F::ONE);
            let gate = m.query_selector(q_res_lookup)
                * m.query_advice(res_is_last, Rotation::cur());
            let q_tbl = m.query_selector(q_row_nr) * m.query_advice(cflag_nr, Rotation::cur());

            vec![
                (
                    gate.clone() * (m.query_advice(res_nk, Rotation::cur()) + one.clone()),
                    q_tbl.clone() * (m.query_advice(nr_pair_l[0], Rotation::cur()) + one.clone()),
                ),
                (
                    gate * (m.query_advice(res_nm, Rotation::cur()) + one.clone()),
                    q_tbl * (m.query_advice(nr_pair_l[1], Rotation::cur()) + one),
                ),
            ]
        });

        // ---------------- ORDER BY revenue DESC ----------------
        let res_sorted = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let perm_res = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            PermAnyChip::configure(meta, q1, q2, res_pad.clone(), res_sorted.clone())
        };
        let q_sort_res = meta.selector();

        let lteq_rev_next_le_cur = LtEqGenericChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| vec![m.query_advice(res_sorted[2], Rotation::next())],
            |m| vec![m.query_advice(res_sorted[2], Rotation::cur())],
        );
        meta.create_gate("ORDER BY revenue DESC", |m| {
            let q = m.query_selector(q_sort_res);
            vec![q * (lteq_rev_next_le_cur.is_lt(m, None) - Expression::Constant(F::ONE))]
        });

        Q5Config {
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

            ls_join,
            ls_disjoin,
            ls_part_pad,
            perm_ls,
            q_ls_pad,


            cp_agg_co,
            cp_agg_nr,
            cp_join_co,
            cp_join_nr,
            cp_root,
            q_cp_mu,
            q_cln_flag,
            q_res_flag,
            q_pad_flag,

            ls_sorted,
            perm_lsort,
            q_ls_sentinel,

            q_sort_ls,
            lt_nk_cur_next,
            iz_nk_eq,

            q_line,
            q_first,
            q_accu,

            line_rev,
            run_sum,
            iz_same_prev,
            iz_same_next,

            res_pad,
            q_res_lookup,
            res_is_last,

            res_sorted,
            perm_res,
            q_sort_res,
            lteq_rev_next_le_cur,

            instance,
            instance_test,
        }
    }

    /// The baseline entry point, kept so the plain circuit and `q5_obj_dp` are
    /// unchanged. A thin wrapper over [`Self::assign_with_input_cells`], which
    /// additionally hands back the cells of the committed input columns.
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
    ) -> Result<AssignedCell<F, F>, Error> {
        self.assign_with_input_cells(
            layouter,
            customer,
            orders,
            lineitem,
            supplier,
            nation,
            region_file,
            europe_hash,
            start_ts,
            end_ts,
            nr_pad_extra,
            co_pad_extra,
            ls_pad_extra,
        )
        .map(|(out, _)| out)
    }

    /// Same assignment, but also returns the cells holding the sixteen
    /// committed input columns, in the order `bench_queries::TpchInput::columns`
    /// publishes them for Q5:
    ///
    /// ```text
    ///    0.. 2  customer  [c_custkey, c_nationkey]
    ///    2.. 5  orders    [o_orderdate_ts, o_custkey, o_orderkey]
    ///    5.. 9  lineitem  [l_orderkey, l_suppkey, l_extendedprice*1000, l_discount*1000]
    ///    9..11  supplier  [s_suppkey, s_nationkey]
    ///   11..14  nation    [n_nationkey, n_name_hash, n_regionkey]
    ///   14..16  region    [r_regionkey, r_name_hash]
    /// ```
    ///
    /// `out[j][i]` is this proof's own cell for committed column `j` at row `i`.
    /// The base relations are never permuted here (only the derived `ls_sorted`
    /// and `res_sorted` views are, in their own columns), so row `i` of the
    /// commitment is row `i` of the witness.
    ///
    /// ENCODING, which the tie has to respect. Eleven columns hold the committed
    /// value VERBATIM: 0 (c_custkey), 2..5 (all of orders), 5..9 (all of
    /// lineitem), 9 (s_suppkey), 12 (n_name_hash) and 15 (r_name_hash). The
    /// other FIVE hold the committed value PLUS ONE:
    ///
    /// ```text
    ///    1  c_nationkey  ->  c_nationkey + 1
    ///   10  s_nationkey  ->  s_nationkey + 1
    ///   11  n_nationkey  ->  n_nationkey + 1
    ///   13  n_regionkey  ->  n_regionkey + 1
    ///   14  r_regionkey  ->  r_regionkey + 1
    /// ```
    ///
    /// The shift is load-bearing and cannot be dropped: `q5_derive` maps a
    /// customer (resp. supplier) with no matching row to nationkey 0 and a
    /// nation with no matching region to region 0, and TPC-H has a real
    /// nationkey 0 (ALGERIA) and a real regionkey 0 (AFRICA), so 0 has to stay
    /// free as the "no match" sentinel. A bound wrapper therefore cannot
    /// equality-tie those five columns to the binding's data columns directly;
    /// see the `+1` adapter gate in `q5_bound.rs`.
    ///
    /// The binding zero-extends every committed column to the height of the
    /// tallest relation, so the short relations are extended here with explicit
    /// zero cells on those rows. No selector covers them, so no gate, lookup or
    /// permutation of the query reads them; they exist only so that every row
    /// the binding accumulates is a row this proof witnesses.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn assign_with_input_cells(
        &self,
        layouter: &mut impl Layouter<F>,
        // base inputs
        customer: Vec<Vec<u64>>, // [custkey, nationkey] (unshifted in file)
        orders: Vec<Vec<u64>>,   // [odate_ts, custkey, orderkey]
        lineitem: Vec<Vec<u64>>, // [orderkey, suppkey, ext_scaled, disc_scaled]
        supplier: Vec<Vec<u64>>, // [suppkey, nationkey] (unshifted)
        nation: Vec<Vec<u64>>,   // [nationkey, name_hash, regionkey] (unshifted)
        region_file: Vec<Vec<u64>>, // [regionkey, name_hash] (unshifted)
        europe_hash: u64,
        start_ts: u64,
        end_ts: u64,
        // NEW: padding knobs
        nr_pad_extra: usize,
        co_pad_extra: usize,
        ls_pad_extra: usize,
    ) -> Result<(AssignedCell<F, F>, Vec<Vec<AssignedCell<F, F>>>), Error> {
        // chips
        let iz_nr_chip = IsZeroChip::construct(self.config.iz_nr.clone());

        let lteq_ge_chip =
            LtEqGenericChip::<F, NUM_BYTES>::construct(self.config.lteq_start_le_odate.clone());
        lteq_ge_chip.load(layouter)?;

        let lt_end_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_odate_lt_end.clone());
        lt_end_chip.load(layouter)?;

        // Every Lt chip of the Cardinality Preservation Check shares one u8 fixed
        // column, so a single load covers the whole check. This replaces the four
        // gap chips of the deleted residual-side argument.
        LtChip::<F, NUM_BYTES>::construct(self.config.cp_agg_co.lt_key_cur_next).load(layouter)?;

        let iz_same_prev_chip = IsZeroChip::construct(self.config.iz_same_prev.clone());
        let iz_same_next_chip = IsZeroChip::construct(self.config.iz_same_next.clone());

        // sortedness of the GROUP BY column; shares the cp u8 table loaded above,
        // so it needs no load of its own.
        let lt_nk_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_nk_cur_next);
        let iz_nk_eq_chip = IsZeroChip::construct(self.config.iz_nk_eq.clone());

        let lteq_rev_chip =
            LtEqGenericChip::<F, NUM_BYTES>::construct(self.config.lteq_rev_next_le_cur.clone());
        lteq_rev_chip.load(layouter)?;

        // helpers
        fn to_field_rows<FF: Field + Ord>(u: &[Vec<u64>]) -> Vec<Vec<FF>> {
            u.iter()
                .map(|r| r.iter().map(|&x| FF::from(x)).collect())
                .collect()
        }
        fn pad_filter_u64(rows: &[Vec<u64>], keep: &[bool], pad: &[u64]) -> Vec<Vec<u64>> {
            rows.iter()
                .zip(keep.iter())
                .map(|(r, &k)| if k { r.clone() } else { pad.to_vec() })
                .collect()
        }
        fn pad_out_u64(filtered: &[Vec<u64>], total: usize, pad: &[u64]) -> Vec<Vec<u64>> {
            let mut out: Vec<Vec<u64>> = Vec::with_capacity(total);
            out.extend_from_slice(filtered);
            while out.len() < total {
                out.push(pad.to_vec());
            }
            out
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

        // Derived intermediates come from ONE shared definition (see
        // `q5_derive`), which `bench_queries::q5_pads` also uses to size the
        // DP capacity for Bag 2 {L,S} -- so the two can never drift.
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

        // ---------------- clean/residual witness over the cluster tree ----------------
        // The honest clean instance is the fully reduced one: an LS tuple is clean
        // iff it joins both bags, a CO tuple iff its packed key occurs in a clean
        // LS tuple, an NR tuple iff its nationkey does. That is a fixed point of
        // the semijoin reduction, so Conservation, Non-Membership and Pairwise
        // Consistency all hold on it and condition (4) holds with equality.
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
        let tamper = hide_one_clean_tuple();
        if tamper {
            if let Some(i) = ls_cln_b.iter().position(|&b| b == 1) {
                ls_cln_b[i] = 0;
            }
        }

        // test hook only: no reduction at all. Every tuple that passes its own
        // predicate is declared clean, so all three residual sections come out
        // empty and both channels of condition (4) agree row by row. Only
        // condition (3) can reject this.
        let all_clean = mark_all_clean();
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

        // condition (3) needs no key vectors any more: the four Pairwise
        // Consistency lookups run between the clean key columns themselves.

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

        // ---------- aggregation over ls_join (with optional extra padding rows) ----------
        // Use pad rows that sort to the end by nk=PAD_U64 and have ext=disc=0.
        let join_len = ls_join_u64.len();
        let n = join_len.saturating_add(ls_pad_extra).max(1);

        let mut ls_join_ext = ls_join_u64.clone();
        while ls_join_ext.len() < n {
            // pad row: [okey=0, nk=PAD_U64, ext=0, disc=0]
            ls_join_ext.push(vec![0u64, PAD_U64, 0u64, 0u64]);
        }

        let mut ls_sorted_u64 = ls_join_ext.clone();
        ls_sorted_u64.sort_by_key(|r| r[1]); // by nationkey_shift (PAD_U64 goes last)

        let mut line_rev_u64 = vec![0u64; n];
        let mut run_sum_u64 = vec![0u64; n];
        let mut res_pad_u64: Vec<[u64; 3]> = vec![[PAD_U64, PAD_U64, PAD_REV]; n];

        // NOTE (completeness limit, deliberately not "fixed" in the circuit): the
        // accumulator is a u128 and is written out with `as u64`, so at a scale
        // factor where one nation's revenue exceeds 2^64 the honest prover would
        // truncate here and the `run_sum` gates would reject its own witness. That
        // is a host-side witness-generation ceiling, not a soundness hole: no
        // truncated witness is ACCEPTED, and the gates below are field-exact. The
        // matching in-circuit ceiling is the 7-byte decomposition the ORDER BY
        // LtEqGeneric imposes on res_sorted[2]. Turning `line_rev` and `run_sum`
        // into range-checked values would cost a range chip per aggregation row
        // and buys nothing at the scale factors this file is used at.
        let mut acc: u128 = 0;
        let mut prev_nk: Option<u64> = None;
        for i in 0..n {
            let nk = ls_sorted_u64[i][1];
            let ext = ls_sorted_u64[i][2] as u128;
            let disc = ls_sorted_u64[i][3] as u128;
            let lr = ext * ((SCALE as u128) - disc);
            line_rev_u64[i] = lr as u64;

            if prev_nk == Some(nk) {
                acc += lr;
            } else {
                acc = lr;
            }
            run_sum_u64[i] = acc as u64;

            // the pinned sentinel at row n carries PAD_U64, so a trailing run
            // of PAD-keyed aggregation rows never closes and never emits
            let next_nk = if i + 1 < n {
                ls_sorted_u64[i + 1][1]
            } else {
                PAD_U64
            };

            if next_nk != nk && nk != PAD_U64 {
                let nm = *nk_to_name.get(&nk).unwrap_or(&0);
                res_pad_u64[i] = [nk, nm, run_sum_u64[i]];
            }
            prev_nk = Some(nk);
        }

        // sort result by revenue desc
        let mut groups: Vec<[u64; 3]> = res_pad_u64
            .iter()
            .copied()
            .filter(|r| r[0] != PAD_U64)
            .collect();
        groups.sort_by(|a, b| b[2].cmp(&a[2]));
        let mut res_sorted_u64: Vec<[u64; 3]> = vec![];
        res_sorted_u64.extend(groups);
        while res_sorted_u64.len() < n {
            res_sorted_u64.push([PAD_U64, PAD_U64, PAD_REV]);
        }

        // ---------- assign region ----------
        layouter.assign_region(
            || "q5 witness",
            |mut region| {
                // The cells of the sixteen committed input columns, in the order
                // `TpchInput::columns` publishes them; see the doc comment for
                // the offsets and for which five carry the +1 key encoding.
                let rows_max = customer
                    .len()
                    .max(orders.len())
                    .max(lineitem.len())
                    .max(supplier.len())
                    .max(nation.len())
                    .max(region_file.len());
                let mut input_cells: Vec<Vec<AssignedCell<F, F>>> =
                    (0..16).map(|_| Vec::with_capacity(rows_max)).collect();

                // base tables
                for i in 0..customer.len() {
                    for j in 0..2 {
                        // shift nationkey inside the assigned customer table (keep base single-table input)
                        let v = if j == 1 {
                            customer[i][j] + 1
                        } else {
                            customer[i][j]
                        };
                        let cell = region.assign_advice(
                            || "customer",
                            self.config.customer[j],
                            i,
                            || Value::known(F::from(v)),
                        )?;
                        input_cells[j].push(cell);
                    }
                }
                // TABLE side of the orders->customer lookup: exactly the assigned
                // customer rows
                for i in 0..customer.len() {
                    self.config.q_cust_tbl.enable(&mut region, i)?;
                }
                for i in 0..orders.len() {
                    for j in 0..3 {
                        let cell = region.assign_advice(
                            || "orders",
                            self.config.orders[j],
                            i,
                            || Value::known(F::from(orders[i][j])),
                        )?;
                        input_cells[2 + j].push(cell);
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
                        let cell = region.assign_advice(
                            || "lineitem",
                            self.config.lineitem[j],
                            i,
                            || Value::known(F::from(lineitem[i][j])),
                        )?;
                        input_cells[5 + j].push(cell);
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
                        let cell = region.assign_advice(
                            || "supplier",
                            self.config.supplier[j],
                            i,
                            || Value::known(F::from(v)),
                        )?;
                        input_cells[9 + j].push(cell);
                    }
                }
                for i in 0..nation.len() {
                    // nation: shift n_nationkey and n_regionkey
                    let c_nk = region.assign_advice(
                        || "nation_nk",
                        self.config.nation[0],
                        i,
                        || Value::known(F::from(nation[i][0] + 1)),
                    )?;
                    let c_nm = region.assign_advice(
                        || "nation_nm",
                        self.config.nation[1],
                        i,
                        || Value::known(F::from(nation[i][1])),
                    )?;
                    let c_rk = region.assign_advice(
                        || "nation_rk",
                        self.config.nation[2],
                        i,
                        || Value::known(F::from(nation[i][2] + 1)),
                    )?;
                    input_cells[11].push(c_nk);
                    input_cells[12].push(c_nm);
                    input_cells[13].push(c_rk);
                }
                // TABLE side of the nation->region lookup
                for i in 0..region_file.len() {
                    self.config.q_region_tbl.enable(&mut region, i)?;
                }
                for i in 0..region_file.len() {
                    let c_rk = region.assign_advice(
                        || "region_rk",
                        self.config.region_file[0],
                        i,
                        || Value::known(F::from(region_file[i][0] + 1)),
                    )?;
                    let c_nm = region.assign_advice(
                        || "region_nm",
                        self.config.region_file[1],
                        i,
                        || Value::known(F::from(region_file[i][1])),
                    )?;
                    input_cells[14].push(c_rk);
                    input_cells[15].push(c_nm);
                }

                // The binding zero-extends every committed column to the height
                // of the tallest relation, so a shorter relation needs a cell of
                // THIS circuit on those rows too, or the extra rows would be
                // untied -- exactly the gap the tie exists to close. No selector
                // covers them (`q_cust_tbl`, `q_oc_join`/`q_co_*`, `q_ls_join`,
                // `q_supp_tbl`, `q_nr_join`/`q_nr_pred` and `q_region_tbl` all
                // stop at their own relation's length), so nothing else in the
                // circuit reads them.
                for (base, len, off) in [
                    (&self.config.customer, customer.len(), 0usize),
                    (&self.config.orders, orders.len(), 2),
                    (&self.config.lineitem, lineitem.len(), 5),
                    (&self.config.supplier, supplier.len(), 9),
                    (&self.config.nation, nation.len(), 11),
                    (&self.config.region_file, region_file.len(), 14),
                ] {
                    for i in len..rows_max {
                        for (j, col) in base.iter().enumerate() {
                            let cell = region.assign_advice(
                                || "committed column zero-extension",
                                *col,
                                i,
                                || Value::known(F::ZERO),
                            )?;
                            input_cells[off + j].push(cell);
                        }
                    }
                }

                // ---------- NR materialization assignments (real nation rows) ----------
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

                // ---------- NR extra padding rows (no join/pred selectors), but must assign cols used by link gate ----------
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
                // padding tail too and the two lookups that read NR are gated by
                // the bit rather than by a clean-prefix extent.
                for i in 0..nr_total {
                    self.config.q_row_nr.enable(&mut region, i)?;
                }

                // ---------- CO materialization assignments (real order rows) ----------
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

                    // predicate chips
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

                // ---------- CO extra padding rows (no join/pred selectors), but must assign cols used by link gate ----------
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

                // ---------- LS materialization assignments ----------
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
                        || Value::known(F::from(ls_cln_b[i])),
                    )?;
                }

                // ---------- LS join/disjoin witnesses + partition permutation ----------
                let ls_join_cells = Self::assign_table_u64(
                    &mut region,
                    "ls_join",
                    &self.config.ls_join,
                    &ls_join_u64, // REAL join rows only (for part_pad linking)
                )?;
                let ls_dis_cells = Self::assign_table_u64(
                    &mut region,
                    "ls_disjoin",
                    &self.config.ls_disjoin,
                    &ls_dis_u64,
                )?;

                // NEW: if ls_pad_extra > 0, we also need to assign extra padding rows into ls_join
                // so that perm_lsort (enabled on n rows) has cells on the LHS.
                for i in ls_join_u64.len()..n {
                    // pad row: [okey=0, nk=PAD_U64, ext=0, disc=0], pinned by
                    // "ls_join aggregation tail is PAD" because perm_lsort carries
                    // these rows into the group-by
                    self.config.q_ls_pad.enable(&mut region, i)?;
                    region.assign_advice(
                        || "ls_join_pad_okey",
                        self.config.ls_join[0],
                        i,
                        || Value::known(F::ZERO),
                    )?;
                    region.assign_advice(
                        || "ls_join_pad_nk",
                        self.config.ls_join[1],
                        i,
                        || Value::known(F::from(PAD_U64)),
                    )?;
                    region.assign_advice(
                        || "ls_join_pad_ext",
                        self.config.ls_join[2],
                        i,
                        || Value::known(F::ZERO),
                    )?;
                    region.assign_advice(
                        || "ls_join_pad_disc",
                        self.config.ls_join[3],
                        i,
                        || Value::known(F::ZERO),
                    )?;
                }

                for i in 0..lineitem.len() {
                    self.config.perm_ls.q_perm1.enable(&mut region, i)?;
                    self.config.perm_ls.q_perm2.enable(&mut region, i)?;
                }
                Self::assign_part_pad_and_link(
                    &mut region,
                    "ls_part_pad",
                    &self.config.ls_part_pad,
                    &ls_part_pad_f,
                    &ls_join_cells,
                    &ls_dis_cells,
                )?;
                // partition side: 1 on the clean LS rows, 0 on the residual ones
                for i in 0..ls_join_u64.len() {
                    self.config.q_cln_flag[0].enable(&mut region, i)?;
                }
                for i in ls_join_u64.len()..(ls_join_u64.len() + ls_dis_u64.len()) {
                    self.config.q_res_flag[0].enable(&mut region, i)?;
                }
                // the LS partition is padded only to lineitem.len(), which the
                // clean and residual sections already fill exactly, so this range
                // is empty. Enabled for uniformity with the two child bags, and so
                // that it stays pinned if the padding ever grows.
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
                // indicator, and the conserved relation is that whole extent, so
                // the witness handed over must carry the same padding. Slicing
                // the UNPADDED witness to `*_total` instead panics the moment a
                // released capacity exceeds the input size, which under dp it
                // always does -- at eps=0.1 the CO release is 4,036 rows on top
                // of |orders| = 15,000, and `vpjoin full q5` died on exactly
                // that. Same defect, same fix, as `q5_obj_dp.rs`; `rjs` hid it
                // because its capacities are zero.
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

                // ===================== CARDINALITY PRESERVATION CHECK =====================
                // condition (4) of the One-Pass OBJ over the cluster tree rooted
                // at LS: both multiplicity channels are propagated from the two
                // child bags to LS and their root sums compared.
                //
                // Both children are leaves, so a child row's input-channel
                // multiplicity is its predicate bit and its clean-channel
                // multiplicity is that bit times the clean indicator.
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

                // parent side, on the rows of LS (one per lineitem row)
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

                // root multiplicities and the equality between the two sums
                let cp_mu: Vec<(u64, u64)> = (0..lineitem.len())
                    .map(|i| {
                        (
                            fetched_co[i].0 * fetched_nr[i].0,
                            ls_cln_b[i] * fetched_co[i].1 * fetched_nr[i].1,
                        )
                    })
                    .collect();
                for i in 0..lineitem.len() {
                    self.config.q_cp_mu.enable(&mut region, i)?;
                }
                let (cp_all, cp_cln) = assign_cp_root(&mut region, &self.config.cp_root, &cp_mu)?;
                if !tamper && !all_clean {
                    debug_assert_eq!(
                        cp_all, cp_cln,
                        "cardinality preservation: |R^c join| != |R join|"
                    );
                }

                // ---------- ls_join -> ls_sorted permutation ----------
                for i in 0..n {
                    self.config.perm_lsort.q_perm1.enable(&mut region, i)?;
                    self.config.perm_lsort.q_perm2.enable(&mut region, i)?;
                }

                for i in 0..n {
                    for j in 0..4 {
                        region.assign_advice(
                            || "ls_sorted",
                            self.config.ls_sorted[j],
                            i,
                            || Value::known(F::from(ls_sorted_u64[i][j])),
                        )?;
                    }
                }
                // sentinel row for same_next, pinned by
                // "ls_sorted group-by sentinel is PAD"
                for j in 0..4 {
                    let v = if j == 1 { PAD_U64 } else { 0u64 };
                    region.assign_advice(
                        || "ls_sorted_sentinel",
                        self.config.ls_sorted[j],
                        n,
                        || Value::known(F::from(v)),
                    )?;
                }
                self.config.q_ls_sentinel.enable(&mut region, n)?;

                // enable line/accu/selectors and assign helpers
                if n > 0 {
                    self.config.q_line.enable(&mut region, 0)?;
                    self.config.q_first.enable(&mut region, 0)?;
                    self.config.q_res_lookup.enable(&mut region, 0)?;
                }
                for i in 0..n {
                    self.config.q_line.enable(&mut region, i)?;
                    self.config.q_res_lookup.enable(&mut region, i)?;
                }
                for i in 1..n {
                    self.config.q_accu.enable(&mut region, i)?;
                }

                // assign line_rev/run_sum/res_pad/res_sorted
                for i in 0..n {
                    region.assign_advice(
                        || "line_rev",
                        self.config.line_rev,
                        i,
                        || Value::known(F::from(line_rev_u64[i])),
                    )?;
                    region.assign_advice(
                        || "run_sum",
                        self.config.run_sum,
                        i,
                        || Value::known(F::from(run_sum_u64[i])),
                    )?;

                    let rp = res_pad_u64[i];
                    for j in 0..3 {
                        region.assign_advice(
                            || "res_pad",
                            self.config.res_pad[j],
                            i,
                            || Value::known(F::from(rp[j])),
                        )?;
                    }
                    let rs = res_sorted_u64[i];
                    for j in 0..3 {
                        region.assign_advice(
                            || "res_sorted",
                            self.config.res_sorted[j],
                            i,
                            || Value::known(F::from(rs[j])),
                        )?;
                    }
                }

                // same_prev / same_next
                for i in 1..n {
                    let diff = F::from(ls_sorted_u64[i][1]) - F::from(ls_sorted_u64[i - 1][1]);
                    iz_same_prev_chip.assign(&mut region, i, Value::known(diff))?;
                }
                for i in 0..n {
                    let next_nk = if i + 1 < n {
                        ls_sorted_u64[i + 1][1]
                    } else {
                        // the pinned sentinel at row n
                        PAD_U64
                    };
                    let diff = F::from(next_nk) - F::from(ls_sorted_u64[i][1]);
                    iz_same_next_chip.assign(&mut region, i, Value::known(diff))?;
                    // the degree-1 copy of is_last that the name lookup reads
                    region.assign_advice(
                        || "res_is_last",
                        self.config.res_is_last,
                        i,
                        || Value::known(F::from((next_nk != ls_sorted_u64[i][1]) as u64)),
                    )?;
                }

                // ls_sorted[1] nondecreasing over the n real rows. Row n is the
                // all-zero sentinel that `iz_same_next` uses to close the last
                // group, so the ladder stops at the pair (n-2, n-1).
                for i in 0..n.saturating_sub(1) {
                    self.config.q_sort_ls.enable(&mut region, i)?;
                    let cur = ls_sorted_u64[i][1];
                    let next = ls_sorted_u64[i + 1][1];
                    lt_nk_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(cur)),
                        Value::known(F::from(next)),
                    )?;
                    iz_nk_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(next) - F::from(cur)),
                    )?;
                }

                // res_pad <-> res_sorted permutation and ORDER BY
                for i in 0..n {
                    self.config.perm_res.q_perm1.enable(&mut region, i)?;
                    self.config.perm_res.q_perm2.enable(&mut region, i)?;
                }
                for i in 0..n.saturating_sub(1) {
                    self.config.q_sort_res.enable(&mut region, i)?;
                    lteq_rev_chip.assign(
                        &mut region,
                        i,
                        &[F::from(res_sorted_u64[i + 1][2])],
                        &[F::from(res_sorted_u64[i][2])],
                    )?;
                }

                // public output
                let out = region.assign_advice(
                    || "instance_test",
                    self.config.instance_test,
                    0,
                    || Value::known(F::from(1u64)),
                )?;
                Ok((out, input_cells))
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

/// Everything Q5's witness generation derives from the base tables before
/// any circuit region is touched.
///
/// Extracted from `assign` VERBATIM so that the bag sizes needed for DP
/// capacity release (`bench_queries::q5_pads`) come from the SAME code the
/// circuit witnesses, and cannot drift from it.
pub struct Q5Derived {
    pub nr_rname_u64: Vec<u64>,
    pub nr_keep_b: Vec<bool>,
    pub nr_pair_u64: Vec<Vec<u64>>,
    pub nr_filtered: Vec<Vec<u64>>,
    pub nk_to_name: HashMap<u64, u64>,
    pub co_nk_u64: Vec<u64>,
    pub co_ge_b: Vec<bool>,
    pub co_lt_b: Vec<bool>,
    pub co_keep_b: Vec<bool>,
    pub co_pair_u64: Vec<Vec<u64>>,
    pub co_filtered: Vec<Vec<u64>>,
    pub ls_mat_u64: Vec<Vec<u64>>,
    pub ls_join_u64: Vec<Vec<u64>>,
    pub ls_dis_u64: Vec<Vec<u64>>,
    pub nr_filt_pad_u64_ext: Vec<Vec<u64>>,
    pub co_filt_pad_u64_ext: Vec<Vec<u64>>,
    pub nr_total: usize,
    pub co_total: usize,
    pub nr_out_pad_u64: Vec<Vec<u64>>,
    pub co_out_pad_u64: Vec<Vec<u64>>,
}

/// Derive Q5's intermediates from the base tables (pure; no circuit access).
///
/// Bag 1 {O,C} is `co_*` and is sized `orders.len() + co_pad_extra`; Bag 2
/// {L,S} is `ls_join_u64` and is sized `ls_join_u64.len() + ls_pad_extra` by
/// the caller. `nr_*` is the N join R dimension filter (one row per nation),
/// not a bag.
#[allow(clippy::too_many_arguments)]
pub fn q5_derive(
    customer: &[Vec<u64>],
    orders: &[Vec<u64>],
    lineitem: &[Vec<u64>],
    supplier: &[Vec<u64>],
    nation: &[Vec<u64>],
    region_file: &[Vec<u64>],
    europe_hash: u64,
    start_ts: u64,
    end_ts: u64,
    nr_pad_extra: usize,
    co_pad_extra: usize,
) -> Q5Derived {
    // NOTE: positional, not compacting -- a filtered-out row is REPLACED by a
    // PAD row at the same index, preserving order and length. Copied verbatim
    // from `assign`; an earlier hand-written "compact then pad" version broke
    // every downstream constraint.
    fn pad_filter_u64(rows: &[Vec<u64>], keep: &[bool], pad: &[u64]) -> Vec<Vec<u64>> {
        rows.iter()
            .zip(keep.iter())
            .map(|(r, &k)| if k { r.clone() } else { pad.to_vec() })
            .collect()
    }
    fn pad_out_u64(filtered: &[Vec<u64>], total: usize, pad: &[u64]) -> Vec<Vec<u64>> {
        let mut out: Vec<Vec<u64>> = Vec::with_capacity(total);
        out.extend_from_slice(filtered);
        while out.len() < total {
            out.push(pad.to_vec());
        }
        out
    }

    // ---------- build shifted maps ----------
    // shift nationkey+1, regionkey+1 to keep 0 as sentinel
    let mut cust_to_nk: HashMap<u64, u64> = HashMap::new();
    for r in customer.iter() {
        cust_to_nk.insert(r[0], r[1] + 1);
    }
    let mut supp_to_nk: HashMap<u64, u64> = HashMap::new();
    for r in supplier.iter() {
        supp_to_nk.insert(r[0], r[1] + 1);
    }
    let mut reg_to_name: HashMap<u64, u64> = HashMap::new();
    for r in region_file.iter() {
        reg_to_name.insert(r[0] + 1, r[1]); // regionkey_shift
    }

    // ---------- NR derivation (nation ⋈ region, filter EUROPE) ----------
    let mut nr_rname_u64 = vec![0u64; nation.len()];
    let mut nr_keep_b = vec![false; nation.len()];
    let mut nr_pair_u64: Vec<Vec<u64>> = vec![vec![0, 0]; nation.len()];

    for i in 0..nation.len() {
        let nk_shift = nation[i][0] + 1;
        let nm_hash = nation[i][1];
        let rk_shift = nation[i][2] + 1;
        let rname = *reg_to_name.get(&rk_shift).unwrap_or(&0);
        nr_rname_u64[i] = rname;
        nr_keep_b[i] = rname == europe_hash;
        nr_pair_u64[i] = vec![nk_shift, nm_hash];
    }
    let nr_filtered: Vec<Vec<u64>> = nr_pair_u64
        .iter()
        .cloned()
        .zip(nr_keep_b.iter())
        .filter(|(_, &k)| k)
        .map(|(r, _)| r)
        .collect();

    let pad2 = vec![PAD_U64; 2];
    let nr_filt_pad_u64 = pad_filter_u64(&nr_pair_u64, &nr_keep_b, &pad2);

    // NEW: allow NR to be padded beyond nation.len()
    let nr_total = nation.len().saturating_add(nr_pad_extra).max(1);
    let mut nr_filt_pad_u64_ext = nr_filt_pad_u64.clone();
    while nr_filt_pad_u64_ext.len() < nr_total {
        nr_filt_pad_u64_ext.push(pad2.clone());
    }
    let nr_out_pad_u64 = pad_out_u64(&nr_filtered, nr_total, &pad2);

    // map nk_shift -> name_hash for filtered NR
    let mut nk_to_name: HashMap<u64, u64> = HashMap::new();
    for r in nr_filtered.iter() {
        nk_to_name.insert(r[0], r[1]);
    }

    // ---------- CO derivation (orders ⋈ customer, filter by date range) ----------
    let mut co_nk_u64 = vec![0u64; orders.len()];
    let mut co_ge_b = vec![false; orders.len()];
    let mut co_lt_b = vec![false; orders.len()];
    let mut co_keep_b = vec![false; orders.len()];
    let mut co_pair_u64: Vec<Vec<u64>> = vec![vec![0, 0]; orders.len()];

    for i in 0..orders.len() {
        let odate = orders[i][0];
        let cust = orders[i][1];
        let okey = orders[i][2];

        let nk_shift = *cust_to_nk.get(&cust).unwrap_or(&0);
        co_nk_u64[i] = nk_shift;

        co_ge_b[i] = start_ts <= odate;
        co_lt_b[i] = odate < end_ts;
        co_keep_b[i] = co_ge_b[i] && co_lt_b[i];

        co_pair_u64[i] = vec![okey, nk_shift];
    }

    let co_filtered: Vec<Vec<u64>> = co_pair_u64
        .iter()
        .cloned()
        .zip(co_keep_b.iter())
        .filter(|(_, &k)| k)
        .map(|(r, _)| r)
        .collect();

    let co_filt_pad_u64 = pad_filter_u64(&co_pair_u64, &co_keep_b, &pad2);

    // NEW: allow CO to be padded beyond orders.len()
    let co_total = orders.len().saturating_add(co_pad_extra).max(1);
    let mut co_filt_pad_u64_ext = co_filt_pad_u64.clone();
    while co_filt_pad_u64_ext.len() < co_total {
        co_filt_pad_u64_ext.push(pad2.clone());
    }
    let co_out_pad_u64 = pad_out_u64(&co_filtered, co_total, &pad2);

    // build key sets for join
    let co_set: HashSet<u64> = co_filtered
        .iter()
        .map(|r| r[0] * SHIFT_NATION + r[1])
        .collect();

    let nr_set: HashSet<u64> = nr_filtered.iter().map(|r| r[0]).collect();

    // ---------- LS materialization (lineitem ⋈ supplier) ----------
    let mut ls_mat_u64: Vec<Vec<u64>> = vec![vec![0, 0, 0, 0]; lineitem.len()];
    for i in 0..lineitem.len() {
        let okey = lineitem[i][0];
        let supp = lineitem[i][1];
        let ext = lineitem[i][2];
        let disc = lineitem[i][3];
        let nk_shift = *supp_to_nk.get(&supp).unwrap_or(&0);
        ls_mat_u64[i] = vec![okey, nk_shift, ext, disc];
    }

    // ---------- partition LS into join/disjoin (relative to CO and NR) ----------
    let mut ls_join_u64 = vec![];
    let mut ls_dis_u64 = vec![];
    for r in ls_mat_u64.iter() {
        let packed = r[0] * SHIFT_NATION + r[1];
        let ok = co_set.contains(&packed) && nr_set.contains(&r[1]);
        if ok {
            ls_join_u64.push(r.clone());
        } else {
            ls_dis_u64.push(r.clone());
        }
    }

    Q5Derived {
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

    // NEW: padding knobs (analogy to bag1_pad_extra / bag2_pad_extra)
    pub nr_pad_extra: usize,
    pub co_pad_extra: usize,
    pub ls_pad_extra: usize,

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
            _marker: PhantomData,
        }
    }
}

impl<F: Field + Ord> Circuit<F> for MyCircuit<F> {
    type Config = Q5Config<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        Q5Chip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = Q5Chip::construct(config);

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
        )?;

        chip.expose_public(&mut layouter, out_cell, 0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::MyCircuit;

    use crate::data::data_processing;
    use chrono::{DateTime, NaiveDate, Utc};
    use halo2_proofs::dev::{MockProver, VerifyFailure};
    use halo2curves::pasta::{vesta, EqAffine, Fp};
    use rand::rngs::OsRng;

    use halo2_proofs::{
        plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit},
        poly::{
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

    use halo2_proofs::poly::commitment::Params;
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
        let mut result = 0u64;
        for (i, c) in s.chars().enumerate() {
            result += (i as u64 + 1) * (c as u64);
        }
        result
    }

    fn scale_by_1000(x: f64) -> u64 {
        (1000.0 * x) as u64
    }

    fn date_to_timestamp(date_str: &str) -> u64 {
        match NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
            Ok(date) => {
                let datetime: DateTime<Utc> = DateTime::<Utc>::from_utc(date.and_hms(0, 0, 0), Utc);
                datetime.timestamp() as u64
            }
            Err(_) => 0,
        }
    }

    /// The maximum gate degree of this circuit. Every soundness patch below is
    /// written to fit under the ceiling the four Pairwise Consistency lookups
    /// already set, because a degree rise doubles every FFT of the prover and
    /// would cost far more than any of those patches.
    #[test]
    fn test_max_gate_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        let mut cs = ConstraintSystem::<Fp>::default();
        let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
        let degree = cs.degree();
        println!("cs.degree() = {}", degree);
        println!(
            "advice={} fixed={} instance={} selectors={} gates={} polys={} lookups={} shuffles={}",
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
        // 8, one above the 7 of `q5_obj.rs`: gating both sides of a Pairwise
        // Consistency lookup by the selector COLUMN rather than by a selector
        // RANGE costs one degree on each side. It buys no FFT, which is the
        // number that matters: halo2 sizes the extended domain at the next
        // power of two above degree - 1, so 7, 8 and 9 all run on an 8x domain
        // and only 10 doubles it.
        assert!(
            degree <= 9,
            "the maximum gate degree rose to {}, which would double every FFT \
             the prover runs",
            degree
        );
    }

    #[test]
    #[ignore = "inherited heavy end-to-end proof; the fast check is test_cardinality_preservation"]
    fn test_1() {
        // ---------------- paths ----------------
        let customer_file_path = &crate::paths::data_file("customer.tbl");
        let orders_file_path = &crate::paths::data_file("orders.tbl");
        let lineitem_file_path = &crate::paths::data_file("lineitem.tbl");
        let supplier_file_path = &crate::paths::data_file("supplier.tbl");
        let nation_file_path = &crate::paths::data_file("nation.tbl");
        let region_file_path = &crate::paths::data_file("region.cvs"); // keep your repo spelling

        // customer: [c_custkey, c_nationkey]
        let mut customer: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::customer_read_records_from_file(customer_file_path) {
            customer = records
                .iter()
                .map(|r| vec![r.c_custkey, r.c_nationkey])
                .collect();
        }

        // orders: [o_orderdate_ts, o_custkey, o_orderkey]
        let mut orders: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::orders_read_records_from_file(orders_file_path) {
            orders = records
                .iter()
                .map(|r| vec![date_to_timestamp(&r.o_orderdate), r.o_custkey, r.o_orderkey])
                .collect();
        }

        // lineitem: [l_orderkey, l_suppkey, l_extendedprice_scaled, l_discount_scaled]
        let mut lineitem: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::lineitem_read_records_from_file(lineitem_file_path) {
            lineitem = records
                .iter()
                .map(|r| {
                    vec![
                        r.l_orderkey,
                        r.l_suppkey,
                        scale_by_1000(r.l_extendedprice),
                        scale_by_1000(r.l_discount),
                    ]
                })
                .collect();
        }

        // supplier: [s_suppkey, s_nationkey]
        let mut supplier: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::supplier_read_records_from_file(supplier_file_path) {
            supplier = records
                .iter()
                .map(|r| vec![r.s_suppkey, r.s_nationkey])
                .collect();
        }

        // nation: [n_nationkey, n_name_hash, n_regionkey]
        let mut nation: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::nation_read_records_from_file(nation_file_path) {
            nation = records
                .iter()
                .map(|r| vec![r.n_nationkey, string_to_u64(&r.n_name), r.n_regionkey])
                .collect();
        }

        // region: [r_regionkey, r_name_hash]
        let mut region: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::region_read_records_from_cvs(region_file_path) {
            region = records
                .iter()
                .map(|r| vec![r.r_regionkey, string_to_u64(&r.r_name)])
                .collect();
        }

        // ---------------- query params ----------------
        let europe_hash = string_to_u64("EUROPE");
        let start_ts = date_to_timestamp("1997-01-01");
        let end_ts = date_to_timestamp("1998-01-01"); // strict < end

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
        let (nr_pad_extra, co_pad_extra, ls_pad_extra) = crate::bench_queries::q5_pads(privacy);
        println!(
            "[q5 test] privacy={} pads: nr={} co={} ls={}",
            privacy.label(),
            nr_pad_extra,
            co_pad_extra,
            ls_pad_extra
        );

        let circuit = MyCircuit::<Fp> {
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
            _marker: PhantomData,
        };

        let public_input: Vec<Fp> = vec![Fp::from(1u64)];

        let k = crate::bench_queries::degree_for("q5", "tpch-60K", privacy);

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
            let proof_path = &crate::paths::proof_file("proof_q5_obj");
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
    }

    /// Fast correctness check of the Cardinality Preservation Check: a truncated
    /// slice of the dataset under MockProver, which verifies every gate, shuffle
    /// and lookup of the circuit without paying for a real proof.
    /// A released NR/CO capacity LARGER than the input table -- the only regime
    /// `dp` ever produces, and the one no other q5 test covers.
    #[test]
    fn dp_capacity_beyond_the_input_tables() {
        let k = 14;

        // customer, supplier, nation and region are taken whole because the three
        // tuple lookups of the bag materialization require every referenced key to
        // be present in its dimension table. orders and lineitem are truncated.
        //
        // On this slice every bag really does split, so no section of any
        // partition is vacuous: NR is 4 clean / 1 residual / 20 pad rows, CO is
        // 13 clean / 601 residual / 1386 pad rows, LS is 13 clean / 7987
        // residual, and both sides of condition (4) count 13 join results.
        const N_ORD: usize = 2000;
        const N_LINE: usize = 8000;

        let mut customer: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::customer_read_records_from_file(
            &crate::paths::data_file("customer.tbl"),
        ) {
            customer = records
                .iter()
                .map(|r| vec![r.c_custkey, r.c_nationkey])
                .collect();
        }

        let mut orders: Vec<Vec<u64>> = vec![];
        if let Ok(records) =
            data_processing::orders_read_records_from_file(&crate::paths::data_file("orders.tbl"))
        {
            orders = records
                .iter()
                .take(N_ORD)
                .map(|r| vec![date_to_timestamp(&r.o_orderdate), r.o_custkey, r.o_orderkey])
                .collect();
        }

        let mut lineitem: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::lineitem_read_records_from_file(
            &crate::paths::data_file("lineitem.tbl"),
        ) {
            lineitem = records
                .iter()
                .take(N_LINE)
                .map(|r| {
                    vec![
                        r.l_orderkey,
                        r.l_suppkey,
                        scale_by_1000(r.l_extendedprice),
                        scale_by_1000(r.l_discount),
                    ]
                })
                .collect();
        }

        let mut supplier: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::supplier_read_records_from_file(
            &crate::paths::data_file("supplier.tbl"),
        ) {
            supplier = records
                .iter()
                .map(|r| vec![r.s_suppkey, r.s_nationkey])
                .collect();
        }

        let mut nation: Vec<Vec<u64>> = vec![];
        if let Ok(records) =
            data_processing::nation_read_records_from_file(&crate::paths::data_file("nation.tbl"))
        {
            nation = records
                .iter()
                .map(|r| vec![r.n_nationkey, string_to_u64(&r.n_name), r.n_regionkey])
                .collect();
        }

        let mut region: Vec<Vec<u64>> = vec![];
        if let Ok(records) =
            data_processing::region_read_records_from_cvs(&crate::paths::data_file("region.cvs"))
        {
            region = records
                .iter()
                .map(|r| vec![r.r_regionkey, string_to_u64(&r.r_name)])
                .collect();
        }

        assert!(
            !customer.is_empty()
                && !orders.is_empty()
                && !lineitem.is_empty()
                && !supplier.is_empty()
                && !nation.is_empty()
                && !region.is_empty(),
            "dataset files not found under {}",
            crate::paths::data_file("customer.tbl")
        );

        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            supplier,
            nation,
            region,
            europe_hash: string_to_u64("EUROPE"),
            start_ts: date_to_timestamp("1996-01-01"),
            end_ts: date_to_timestamp("1998-01-01"),
            // BOTH strictly greater than 0, so nr_total > |nation| and
            // co_total > |orders|: the regime `dp` always produces and the one
            // every other q5 test misses, since they all run at rjs where the
            // capacities are zero. Slicing the unpadded witness to the capacity
            // panicked here -- `vpjoin full q5` died on exactly this.
            nr_pad_extra: 7,
            co_pad_extra: 20,
            ls_pad_extra: 30,
            _marker: PhantomData,
        };

        let prover = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        prover.assert_satisfied();
    }
    #[test]
    fn test_cardinality_preservation() {
        let k = 14;

        // customer, supplier, nation and region are taken whole because the three
        // tuple lookups of the bag materialization require every referenced key to
        // be present in its dimension table. orders and lineitem are truncated.
        //
        // On this slice every bag really does split, so no section of any
        // partition is vacuous: NR is 4 clean / 1 residual / 20 pad rows, CO is
        // 13 clean / 601 residual / 1386 pad rows, LS is 13 clean / 7987
        // residual, and both sides of condition (4) count 13 join results.
        const N_ORD: usize = 2000;
        const N_LINE: usize = 8000;

        let mut customer: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::customer_read_records_from_file(
            &crate::paths::data_file("customer.tbl"),
        ) {
            customer = records
                .iter()
                .map(|r| vec![r.c_custkey, r.c_nationkey])
                .collect();
        }

        let mut orders: Vec<Vec<u64>> = vec![];
        if let Ok(records) =
            data_processing::orders_read_records_from_file(&crate::paths::data_file("orders.tbl"))
        {
            orders = records
                .iter()
                .take(N_ORD)
                .map(|r| vec![date_to_timestamp(&r.o_orderdate), r.o_custkey, r.o_orderkey])
                .collect();
        }

        let mut lineitem: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::lineitem_read_records_from_file(
            &crate::paths::data_file("lineitem.tbl"),
        ) {
            lineitem = records
                .iter()
                .take(N_LINE)
                .map(|r| {
                    vec![
                        r.l_orderkey,
                        r.l_suppkey,
                        scale_by_1000(r.l_extendedprice),
                        scale_by_1000(r.l_discount),
                    ]
                })
                .collect();
        }

        let mut supplier: Vec<Vec<u64>> = vec![];
        if let Ok(records) = data_processing::supplier_read_records_from_file(
            &crate::paths::data_file("supplier.tbl"),
        ) {
            supplier = records
                .iter()
                .map(|r| vec![r.s_suppkey, r.s_nationkey])
                .collect();
        }

        let mut nation: Vec<Vec<u64>> = vec![];
        if let Ok(records) =
            data_processing::nation_read_records_from_file(&crate::paths::data_file("nation.tbl"))
        {
            nation = records
                .iter()
                .map(|r| vec![r.n_nationkey, string_to_u64(&r.n_name), r.n_regionkey])
                .collect();
        }

        let mut region: Vec<Vec<u64>> = vec![];
        if let Ok(records) =
            data_processing::region_read_records_from_cvs(&crate::paths::data_file("region.cvs"))
        {
            region = records
                .iter()
                .map(|r| vec![r.r_regionkey, string_to_u64(&r.r_name)])
                .collect();
        }

        assert!(
            !customer.is_empty()
                && !orders.is_empty()
                && !lineitem.is_empty()
                && !supplier.is_empty()
                && !nation.is_empty()
                && !region.is_empty(),
            "dataset files not found under {}",
            crate::paths::data_file("customer.tbl")
        );

        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            supplier,
            nation,
            region,
            europe_hash: string_to_u64("EUROPE"),
            start_ts: date_to_timestamp("1996-01-01"),
            end_ts: date_to_timestamp("1998-01-01"),
            nr_pad_extra: 0,
            co_pad_extra: 0,
            ls_pad_extra: 0,
            _marker: PhantomData,
        };

        let prover = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        prover.assert_satisfied();

        // Negative direction: the same witness with one joinable LS tuple hidden
        // in the residual side, and the CO and NR bags re-reduced around it so
        // that Conservation, Non-Membership and Pairwise Consistency all still
        // hold. Only condition (4) can see this, so the circuit must now reject.
        super::set_hide_one_clean_tuple(true);
        let tampered = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        let verdict = tampered.verify();
        super::set_hide_one_clean_tuple(false);

        let failures = verdict.expect_err("condition (4) accepted a hidden joinable tuple");
        // `all`, not `any`: this direction must reject through the single root-sum
        // equality of the Cardinality Preservation Check and through NOTHING else.
        // A new constraint that made this witness fail for some other reason would
        // silently destroy the evidence that condition (4) is doing the work, and
        // an `any` assertion would not notice.
        assert!(
            !failures.is_empty()
                && failures
                    .iter()
                    .all(|f| format!("{:?}", f).contains("cardinality preservation")),
            "the circuit rejected, but not (only) through the Cardinality Preservation Check: {:?}",
            failures
        );

        // Third direction: no reduction at all, every tuple that passes its own
        // predicate declared clean and all three residual sections empty.
        // Conservation holds and condition (4) is satisfied for free, since both
        // channels then agree row by row, so this is the escape that condition (3)
        // exists to close. The two child bags now carry dangling tuples as well,
        // which is what the mirror direction of each edge catches.
        super::set_mark_all_clean(true);
        let unreduced = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        let verdict = unreduced.verify();
        super::set_mark_all_clean(false);

        // Three of the four Pairwise Consistency lookups fire here: LS^c now
        // carries lineitems whose packed key is in no kept order and whose nation
        // is not European, and CO^c carries orders matching no clean LS tuple.
        // "attach name from NR_out" fires alongside them, because a non-European
        // group has no (nationkey, name) row to be labelled from; that is a
        // downstream consequence of the same unreduced witness, not a substitute
        // for condition (3), so the assertion below still demands a "pw: " lookup.
        let failures = verdict.expect_err("condition (3) accepted an unreduced clean instance");
        assert_eq!(
            failures
                .iter()
                .filter(|f| matches!(
                    f,
                    VerifyFailure::Lookup { name, .. } if name.starts_with("pw: ")
                ))
                .count()
                .min(1),
            1,
            "the circuit rejected, but not through a Pairwise Consistency lookup: {:?}",
            failures
        );
    }
} // end mod tests

// nation:   25
// part:     2000
// customer: 1500
// orders:   15000
// lineitem: 60175
// partsupp: 8000
// supplier: 100
// region:   5
