//! TPC-H Query 5 proof (base tables only) with internal bag materialization.
//!
//! Proves:
//! 1) NR = nation ⋈ region, filter r_name == EUROPE  -> nr_out_pad(nk_shift, n_name_hash)
//! 2) CO = orders ⋈ customer, filter start<=odate<end -> co_out_pad(okey, nk_shift)
//! 3) LS = lineitem ⋈ supplier -> ls_mat(okey, nk_shift, ext, disc)
//! 4) Join LS with CO (packed (okey,nk)) and with NR (nk)
//! 5) GROUP BY nk -> SUM(ext*(1-disc))  (scaled by 1000)
//! 6) attach n_name_hash from NR
//! 7) ORDER BY revenue DESC
//!
//! IMPORTANT: we shift nationkey and regionkey by +1 in preprocessing to keep 0 as sentinel.
//!
//! UPDATE (padding extras):
//! - You can now pass padding extras for NR and CO materialized bags, and for the LS-join aggregation tail:
//!   * nr_pad_extra: extends NR permutation length by extra dummy PAD rows
//!   * co_pad_extra: extends CO permutation length by extra dummy PAD rows
//!   * ls_pad_extra: extends the LS-join aggregation/sort length by extra dummy PAD rows
//!
//! This is analogous to the triangle example where Bag sizes can be padded beyond the true witness size.

use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::lessthan_or_equal_generic::{
    LtEqGenericChip, LtEqGenericConfig, LtEqGenericInstruction,
};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};

use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

const NUM_BYTES: usize = 7;
const MAX_SENTINEL: u64 = (1u64 << (8 * NUM_BYTES)) - 1;
const PAD_U64: u64 = MAX_SENTINEL;

const SCALE: u64 = 1000;
const PAD_REV: u64 = 0;

// pack (orderkey, nationkey_shift)
const SHIFT_NATION: u64 = 1u64 << 8; // nationkey_shift <= 25+1 fits

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

    // ---------------- bag materialization: NR ----------------
    q_nr_join: Selector,              // enable nation->region tuple lookup
    q_nr_pred: Selector,              // enable isZero (r_name == EUROPE)
    nr_rname: Column<Advice>,         // looked-up region name hash for nation row
    nr_keep: Column<Advice>,          // boolean
    nr_pair: Vec<Column<Advice>>,     // [nk_shift, n_name_hash] per nation row
    nr_filt_pad: Vec<Column<Advice>>, // keep? nr_pair : PAD
    nr_out_pad: Vec<Column<Advice>>,  // filtered rows padded (and can be extended by extra padding)
    perm_nr: PermAnyConfig,
    iz_nr: IsZeroConfig<F>,

    // ---------------- bag materialization: CO ----------------
    q_oc_join: Selector, // orders->customer tuple lookup
    q_co_ge: Selector,   // start <= odate (LtEqGeneric)
    q_co_lt: Selector,   // odate < end (LtChip)
    q_co_and: Selector,  // keep = ge*lt
    co_ge_ok: Column<Advice>,
    co_lt_ok: Column<Advice>,
    co_keep: Column<Advice>,      // boolean
    co_nk: Column<Advice>,        // looked-up nationkey_shift for order row
    co_pair: Vec<Column<Advice>>, // [okey, nk_shift] per order row
    co_filt_pad: Vec<Column<Advice>>,
    co_out_pad: Vec<Column<Advice>>,
    perm_co: PermAnyConfig,
    lteq_start_le_odate: LtEqGenericConfig<F, NUM_BYTES>,
    lt_odate_lt_end: LtConfig<F, NUM_BYTES>,

    // ---------------- bag materialization: LS ----------------
    q_ls_join: Selector,         // lineitem->supplier tuple lookup
    ls_mat: Vec<Column<Advice>>, // [okey, nk_shift, ext, disc] per lineitem row

    // ---------------- LS partition: join/disjoin ----------------
    ls_join: Vec<Column<Advice>>,
    ls_disjoin: Vec<Column<Advice>>,
    ls_part_pad: Vec<Column<Advice>>,
    perm_ls: PermAnyConfig,

    // ---------------- membership + gap proof on ls_disjoin ----------------
    q_flagged_lookup: Selector,
    q_lookup_complex: Selector,
    flags_in: Vec<Column<Advice>>,       // [in_co, in_nr]
    range_low_high: Vec<Column<Advice>>, // [co_low,co_high,nr_low,nr_high]

    lt_co_gap_low: LtConfig<F, NUM_BYTES>,
    lt_co_gap_high: LtConfig<F, NUM_BYTES>,
    lt_nr_gap_low: LtConfig<F, NUM_BYTES>,
    lt_nr_gap_high: LtConfig<F, NUM_BYTES>,

    co_key: Column<Advice>,
    co_key_next: Column<Advice>,
    nr_key: Column<Advice>,
    nr_key_next: Column<Advice>,

    q_join_member: Selector, // enable membership lookups for ls_join rows

    // ---------------- aggregation over contributing LS ----------------
    ls_sorted: Vec<Column<Advice>>,
    perm_lsort: PermAnyConfig,

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

    fn assign_table_u64(
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

    fn assign_table_f(
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

    fn assign_part_pad_and_link(
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

                if i < join_len {
                    region.constrain_equal(part_cell.cell(), join_cells[i][j].cell())?;
                } else if i < join_len + dis_len {
                    region.constrain_equal(part_cell.cell(), dis_cells[i - join_len][j].cell())?;
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

        // conditions
        let cond_europe = meta.advice_column();
        let cond_start = meta.advice_column();
        let cond_end = meta.advice_column();

        // ---------------- NR materialization ----------------
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

        // nation->region tuple lookup: (n_regionkey_shift, nr_rname) in region(r_regionkey_shift, r_name_hash)
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

        // link nr_filt_pad = keep? nr_pair : PAD
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

        // ---------------- CO materialization ----------------
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

        // orders->customer tuple lookup: (o_custkey, co_nk) in customer(c_custkey, c_nationkey_shift)
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

        // co_pair = [o_orderkey, co_nk]
        meta.create_gate("co_pair copies orderkey and looked nk", |m| {
            let q = m.query_selector(q_co_and);
            let ok = m.query_advice(orders[2], Rotation::cur());
            let nk = m.query_advice(co_nk, Rotation::cur());
            let p0 = m.query_advice(co_pair[0], Rotation::cur());
            let p1 = m.query_advice(co_pair[1], Rotation::cur());
            vec![q.clone() * (p0 - ok), q * (p1 - nk)]
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

        // link co_filt_pad = keep? co_pair : PAD
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

        // ---------------- LS materialization (lineitem ⋈ supplier) ----------------
        let q_ls_join = meta.complex_selector();
        let ls_mat = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        // tuple lookup (l_suppkey, ls_mat.nk) in supplier(s_suppkey, s_nationkey_shift)
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
        let ls_part_pad = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

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
            PermAnyChip::configure(meta, q1, q2, ls_mat.clone(), ls_part_pad.clone())
        };

        // ---------------- membership + gap proof ----------------
        let q_flagged_lookup = meta.selector();
        let q_lookup_complex = meta.complex_selector();
        let q_join_member = meta.complex_selector();

        let flags_in = vec![meta.advice_column(), meta.advice_column()]; // in_co, in_nr
        let range_low_high = vec![
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        // lt chips gated by (1-in_*)
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

        // key tables
        let co_key = meta.advice_column();
        let co_key_next = meta.advice_column();
        let nr_key = meta.advice_column();
        let nr_key_next = meta.advice_column();

        // membership lookups on ls_disjoin
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

        // disjoin emptiness constraints
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
                q.clone() * in_co.clone() * in_nr.clone(), // can't be both in
                q.clone() * (one.clone() - in_co.clone()) * (one.clone() - co_low_ok),
                q.clone() * (one.clone() - in_co.clone()) * (one.clone() - co_high_ok),
                q.clone() * (one.clone() - in_nr.clone()) * (one.clone() - nr_low_ok),
                q * (one.clone() - in_nr) * (one - nr_high_ok),
            ]
        });

        // join rows must be members of both sets
        meta.lookup_any("CO member (ls_join)", |m| {
            let q = m.query_selector(q_join_member);
            let key = (m.query_advice(ls_join[0], Rotation::cur())
                * Expression::Constant(F::from(SHIFT_NATION)))
                + m.query_advice(ls_join[1], Rotation::cur());
            vec![(q * key, m.query_advice(co_key, Rotation::cur()))]
        });
        meta.lookup_any("NR member (ls_join)", |m| {
            let q = m.query_selector(q_join_member);
            let nk = m.query_advice(ls_join[1], Rotation::cur());
            vec![(q * nk, m.query_advice(nr_key, Rotation::cur()))]
        });

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

        // attach (nk,name) via lookup into nr_out_pad (tuple lookup)
        meta.lookup_any("attach name from NR_out", |m| {
            let q_in = m.query_selector(q_res_lookup);
            let one = Expression::Constant(F::ONE);
            let is_last = one - iz_same_next.expr();
            let gate = q_in * is_last;

            vec![
                (
                    gate.clone() * m.query_advice(res_pad[0], Rotation::cur()),
                    m.query_advice(nr_out_pad[0], Rotation::cur()),
                ),
                (
                    gate * m.query_advice(res_pad[1], Rotation::cur()),
                    m.query_advice(nr_out_pad[1], Rotation::cur()),
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

            ls_join,
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

            q_join_member,

            ls_sorted,
            perm_lsort,

            q_line,
            q_first,
            q_accu,

            line_rev,
            run_sum,
            iz_same_prev,
            iz_same_next,

            res_pad,
            q_res_lookup,

            res_sorted,
            perm_res,
            q_sort_res,
            lteq_rev_next_le_cur,

            instance,
            instance_test,
        }
    }

    pub fn assign(
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
    ) -> Result<AssignedCell<F, F>, Error> {
        // chips
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

        let iz_same_prev_chip = IsZeroChip::construct(self.config.iz_same_prev.clone());
        let iz_same_next_chip = IsZeroChip::construct(self.config.iz_same_next.clone());

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

            let next_nk = if i + 1 < n {
                ls_sorted_u64[i + 1][1]
            } else {
                0
            };

            // IMPORTANT: don't emit for PAD_U64 groups; keep res_pad as PAD rows
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
                // base tables
                for i in 0..customer.len() {
                    for j in 0..2 {
                        // shift nationkey inside the assigned customer table (keep base single-table input)
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
                    // nation: shift n_nationkey and n_regionkey
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

                    iz_nr_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(nr_rname_u64[i]) - F::from(europe_hash)),
                    )?;
                }

                // ---------- NR extra padding rows (no join/pred selectors), but must assign cols used by link gate ----------
                for i in nation.len()..nr_total {
                    // q_nr_* not enabled
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

                // enable permutation for NR and assign filt/out (extended)
                for i in 0..nr_total {
                    self.config.perm_nr.q_perm1.enable(&mut region, i)?;
                    self.config.perm_nr.q_perm2.enable(&mut region, i)?;
                }
                Self::assign_table_f(
                    &mut region,
                    "nr_filt_pad",
                    &self.config.nr_filt_pad,
                    &to_field_rows::<F>(&nr_filt_pad_u64_ext),
                )?;
                Self::assign_table_f(
                    &mut region,
                    "nr_out_pad",
                    &self.config.nr_out_pad,
                    &to_field_rows::<F>(&nr_out_pad_u64),
                )?;

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
                    // q_co_* not enabled
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
                Self::assign_table_f(
                    &mut region,
                    "co_filt_pad",
                    &self.config.co_filt_pad,
                    &to_field_rows::<F>(&co_filt_pad_u64_ext),
                )?;
                Self::assign_table_f(
                    &mut region,
                    "co_out_pad",
                    &self.config.co_out_pad,
                    &to_field_rows::<F>(&co_out_pad_u64),
                )?;

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
                    // pad row: [okey=0, nk=PAD_U64, ext=0, disc=0]
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

                // ---------- key tables (row0 dummy) ----------
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

                // ---------- disjoin emptiness witnesses ----------
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

                // join member selector (ONLY real join rows)
                for i in 0..ls_join_u64.len() {
                    self.config.q_join_member.enable(&mut region, i)?;
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
                // sentinel row for same_next
                for j in 0..4 {
                    region.assign_advice(
                        || "ls_sorted_sentinel",
                        self.config.ls_sorted[j],
                        n,
                        || Value::known(F::from(0u64)),
                    )?;
                }

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
                        0u64
                    };
                    let diff = F::from(next_nk) - F::from(ls_sorted_u64[i][1]);
                    iz_same_next_chip.assign(&mut region, i, Value::known(diff))?;
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
    use halo2_proofs::dev::MockProver;
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

    #[test]
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

        // NEW: padding knobs (set to 0 if you don't need)
        let nr_pad_extra = 0usize;
        let co_pad_extra = (32 * 89) as usize;
        let ls_pad_extra = (668 * 89) as usize;

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

        let k = 16;

        // let test = true;
        let test = false;

        if test {
            let prover = MockProver::run(k, &circuit, vec![public_input]).unwrap();
            prover.assert_satisfied();
        } else {
            let proof_path = &crate::paths::proof_file("proof_q5_obj_dp");
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
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
