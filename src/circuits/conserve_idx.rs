//! Condition (1) of the revised One-Pass OBJ: the indexed Conservation Check.
//!
//! The gate certifies a prover-supplied clean/residual split of every relation.
//! Following `letter.tex` (subsec:output_certification and subsubsec:opt_semijoin),
//! the split is stated over the INDEXED relation
//!
//!   R^_i = { (l, t_l, c(l)) : l in [|R_i|] },
//!
//! where `l` is the committed row position and `c(l)` is the indicator the
//! prover sets on that row, and the condition is
//!
//!   R^_i  ==  R^_i^c  U+  R^_i^r          (as multisets)
//!
//! realized by a single permutation argument per relation between R^_i and the
//! concatenation of the two parts, treated as one virtual column.
//!
//! Why one permutation suffices, and why there is no Non-Membership Check here:
//! the indices are distinct, so R^_i is a SET even when R_i is a bag. A
//! permutation onto a group of the same height therefore places each occurrence
//! on exactly one side by itself: none can be fabricated, dropped, duplicated,
//! or counted in both parts. That is the whole of question (i) bar the
//! predicate, and it is where the One-Pass gate earns its saving over the
//! value-level split, which needs a disjointness argument to keep two equal
//! tuples apart.
//!
//! Carrying `c` inside the conserved entry is what ties the partition to the
//! indicator column the other two conditions read. Matching `(l, t_l, c(l))`
//! against an entry tagged 1 or 0 forces `c(l) = 1` exactly on the occurrences
//! placed in the clean part, so the Pairwise Consistency lookups, the
//! multiplicity propagation and the final join enumeration, all of which read
//! the split through `c` on the committed rows, cannot be shown two different
//! partitions.
//!
//! There is no pad section. The split is over ALL committed rows, so the two
//! parts fill exactly |R_i| rows; a relation laid out at a DP capacity simply
//! has its padding rows in the residual part, since their indicator is zero.
//!
//! One deviation from the text, for a reason internal to halo2: the paper
//! carries `l` in a FIXED column, which makes the indices distinct by
//! construction and free. `PermAnyChip` shuffles advice columns only, so `l`
//! lives in an advice column pinned by [`configure_row_index`] to 0 at row 0 and
//! to `prev + 1` afterwards. That is as strong as a fixed column (the values are
//! forced, not chosen) at one advice column and two degree-2 gates for the whole
//! circuit, shared by every relation, since each relation occupies rows
//! `[0, |R_i|)` of the same region.

use std::sync::atomic::{AtomicBool, Ordering};

use halo2_proofs::halo2curves::ff::PrimeField;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

/// The shared committed row index `l`, pinned to the row number.
#[derive(Clone, Copy, Debug)]
pub struct RowIndexConfig {
    pub col: Column<Advice>,
    q_zero: Selector,
    q_step: Selector,
}

/// Allocates the index column and pins it. One per circuit: every relation
/// starts at row 0, so they all read the same column over their own prefix.
pub fn configure_row_index<F: Field + Ord>(meta: &mut ConstraintSystem<F>) -> RowIndexConfig {
    let col = meta.advice_column();
    meta.enable_equality(col);
    let q_zero = meta.selector();
    let q_step = meta.selector();

    meta.create_gate("row index starts at 0", move |m| {
        let q = m.query_selector(q_zero);
        vec![q * m.query_advice(col, Rotation::cur())]
    });
    meta.create_gate("row index increments by 1", move |m| {
        let q = m.query_selector(q_step);
        vec![
            q * (m.query_advice(col, Rotation::next())
                - m.query_advice(col, Rotation::cur())
                - Expression::Constant(F::ONE)),
        ]
    });

    RowIndexConfig {
        col,
        q_zero,
        q_step,
    }
}

/// Assigns the index column over `n` rows. Call once, with the height of the
/// tallest relation the circuit conserves.
pub fn assign_row_index<F: Field + Ord>(
    region: &mut Region<'_, F>,
    cfg: &RowIndexConfig,
    n: usize,
) -> Result<(), Error> {
    for i in 0..n {
        region.assign_advice(
            || "row index",
            cfg.col,
            i,
            || Value::known(F::from(i as u64)),
        )?;
    }
    if n > 0 {
        cfg.q_zero.enable(region, 0)?;
    }
    for i in 0..n.saturating_sub(1) {
        cfg.q_step.enable(region, i)?;
    }
    Ok(())
}

/// Test hook, off in every benchmark path: when set, [`assign_conserve`] moves
/// one occurrence across the clean/residual boundary WITHOUT changing the flag
/// pattern, so the partition side still reads `1` over its first block and `0`
/// after it while holding a residual occurrence tagged clean. Only the
/// permutation argument can see that, which is what shows the Conservation
/// Check is wired and not vacuous.
pub static MISPLACE_ONE_OCCURRENCE: AtomicBool = AtomicBool::new(false);

/// One relation's Conservation Check.
#[derive(Clone, Debug)]
pub struct ConserveConfig {
    /// `(l, tuple..., flag)` laid out as `[clean rows | residual rows]`
    pub part: Vec<Column<Advice>>,
    pub perm: PermAnyConfig,
    /// every row of the group: the flag is boolean
    pub q_flag: Selector,
    /// rows 0..n-1: the flag is non-increasing
    pub q_mono: Selector,
}

/// Configures the Conservation Check of one relation.
///
/// `base` is the relation's committed tuple columns and `cflag` its indicator
/// column, both on the relation's own rows. The conserved entry is
/// `(row_index, base..., cflag)`; the partition side is a fresh group of the
/// same width.
pub fn configure_conserve<F: Field + Ord>(
    meta: &mut ConstraintSystem<F>,
    row_idx: &RowIndexConfig,
    base: &[Column<Advice>],
    cflag: Column<Advice>,
) -> ConserveConfig {
    let q1 = meta.complex_selector();
    let q2 = meta.complex_selector();
    let q_flag = meta.selector();
    let q_mono = meta.selector();
    configure_conserve_with(meta, row_idx, base, cflag, q1, q2, q_flag, q_mono)
}

/// Same, but with the four selectors supplied by the caller.
///
/// A replicated (laned) relation needs this: every lane is active on the same
/// rows, so one selector of each kind serves all of them and only the columns
/// replicate. Allocating fresh selectors per lane would make the selector count
/// grow with the lane count, which the lane geometry must not do.
#[allow(clippy::too_many_arguments)]
pub fn configure_conserve_with<F: Field + Ord>(
    meta: &mut ConstraintSystem<F>,
    row_idx: &RowIndexConfig,
    base: &[Column<Advice>],
    cflag: Column<Advice>,
    q1: Selector,
    q2: Selector,
    q_flag: Selector,
    q_mono: Selector,
) -> ConserveConfig {
    configure_conserve_inner(meta, row_idx, base, cflag, q1, q2, q_flag, q_mono, true)
}

/// Same again, but the permutation is raw `meta.shuffle` with no
/// `enable_equality` on either side.
///
/// A replicated relation needs this too: `PermAnyChip` enables equality on
/// every column it touches, which puts them all in the permutation argument, so
/// a laned relation would add a permutation column per bag attribute per lane.
/// The multiset equality does not need copy constraints, only the shuffle, so a
/// lane pays two permutation columns (its two channel totals) and nothing else.
#[allow(clippy::too_many_arguments)]
pub fn configure_conserve_laned<F: Field + Ord>(
    meta: &mut ConstraintSystem<F>,
    row_idx: &RowIndexConfig,
    base: &[Column<Advice>],
    cflag: Column<Advice>,
    q1: Selector,
    q2: Selector,
    q_flag: Selector,
    q_mono: Selector,
) -> ConserveConfig {
    configure_conserve_inner(meta, row_idx, base, cflag, q1, q2, q_flag, q_mono, false)
}

#[allow(clippy::too_many_arguments)]
fn configure_conserve_inner<F: Field + Ord>(
    meta: &mut ConstraintSystem<F>,
    row_idx: &RowIndexConfig,
    base: &[Column<Advice>],
    cflag: Column<Advice>,
    q1: Selector,
    q2: Selector,
    q_flag: Selector,
    q_mono: Selector,
    copy_constrainable: bool,
) -> ConserveConfig {
    let width = base.len() + 2;
    let part = (0..width).map(|_| meta.advice_column()).collect::<Vec<_>>();

    let mut input = Vec::with_capacity(width);
    input.push(row_idx.col);
    input.extend_from_slice(base);
    input.push(cflag);

    let perm = if copy_constrainable {
        PermAnyChip::configure(meta, q1, q2, input, part.clone())
    } else {
        let ins = input.clone();
        let tbl = part.clone();
        meta.shuffle("conservation (laned)", move |m| {
            let qi = m.query_selector(q1);
            let qo = m.query_selector(q2);
            ins.iter()
                .zip(tbl.iter())
                .map(|(i, t)| {
                    (
                        qi.clone() * m.query_advice(*i, Rotation::cur()),
                        qo.clone() * m.query_advice(*t, Rotation::cur()),
                    )
                })
                .collect::<Vec<_>>()
        });
        PermAnyConfig::bare(q1, q2)
    };

    // The flag on the partition side is boolean and NON-INCREASING, which is
    // what makes the group `[1...1 0...0]`: a clean block followed by a
    // residual one. Without it a prover could keep the conserved multiset and
    // interleave the two, so the block a downstream reader takes for the clean
    // part would not be it.
    //
    // Deliberately NOT two selectors covering `[0, n_cln)` and `[n_cln, n)`.
    // Those ranges are a function of the private clean count, and a Selector
    // becomes a fixed column at keygen, so pinning the sections that way would
    // publish |R_i^c| in the verifying key -- exactly the split size the
    // obliviousness argument says is hidden inside the |R_i| rows. Booleanity
    // plus monotonicity says the same thing about the flag column with two
    // selectors whose extent is |R_i| alone.
    {
        let flag = *part.last().unwrap();
        meta.create_gate("conservation: flag is a bit, non-increasing", move |m| {
            let qf = m.query_selector(q_flag);
            let qm = m.query_selector(q_mono);
            let one = Expression::Constant(F::ONE);
            let f = m.query_advice(flag, Rotation::cur());
            let f_next = m.query_advice(flag, Rotation::next());
            vec![
                qf * f.clone() * (one.clone() - f.clone()),
                qm * f_next * (one - f),
            ]
        });
    }

    ConserveConfig {
        part,
        perm,
        q_flag,
        q_mono,
    }
}

/// Assigns one relation's Conservation Check.
///
/// `rows[i]` is the committed tuple at row `i`, in the same column order as the
/// `base` slice passed to [`configure_conserve`], and `cflag[i]` its indicator.
/// The caller has already assigned both; this writes only the partition side.
pub fn assign_conserve<F: Field + Ord>(
    region: &mut Region<'_, F>,
    cfg: &ConserveConfig,
    rows: &[Vec<u64>],
    cflag: &[u64],
) -> Result<(), Error> {
    let n = rows.len();
    assert_eq!(cflag.len(), n, "one indicator per committed row");
    if n == 0 {
        return Ok(());
    }

    // [clean entries | residual entries], each carrying its own committed index
    let mut part_rows: Vec<Vec<u64>> = Vec::with_capacity(n);
    for want in [1u64, 0u64] {
        for (i, r) in rows.iter().enumerate() {
            if cflag[i] == want {
                let mut e = Vec::with_capacity(r.len() + 2);
                e.push(i as u64);
                e.extend_from_slice(r);
                e.push(want);
                part_rows.push(e);
            }
        }
    }
    debug_assert_eq!(part_rows.len(), n, "the split must cover every row exactly");
    let n_cln = cflag.iter().filter(|&&f| f == 1).count();

    // test hook only: swap the last clean entry with the first residual one and
    // leave the two flags where they were, so the blocks keep their shape and
    // only the conserved multiset changes
    if MISPLACE_ONE_OCCURRENCE.load(Ordering::Relaxed) && n_cln > 0 && n_cln < n {
        let last = part_rows[n_cln - 1].clone();
        let first = part_rows[n_cln].clone();
        let w = last.len() - 1;
        part_rows[n_cln - 1] = first;
        part_rows[n_cln] = last;
        part_rows[n_cln - 1][w] = 1;
        part_rows[n_cln][w] = 0;
    }

    for (i, e) in part_rows.iter().enumerate() {
        for (j, &v) in e.iter().enumerate() {
            region.assign_advice(
                || "conservation partition",
                cfg.part[j],
                i,
                || Value::known(F::from(v)),
            )?;
        }
        cfg.perm.q_perm1.enable(region, i)?;
        cfg.perm.q_perm2.enable(region, i)?;
    }
    // The gate queries the flag at Rotation::next() on every row it covers, so
    // row n needs a value even though the monotonicity polynomial is gated off
    // there. Zero is the honest continuation of a non-increasing column and the
    // permutation does not reach this row.
    region.assign_advice(
        || "conservation flag sentinel",
        *cfg.part.last().unwrap(),
        n,
        || Value::known(F::ZERO),
    )?;

    for i in 0..n {
        cfg.q_flag.enable(region, i)?;
    }
    for i in 0..n.saturating_sub(1) {
        cfg.q_mono.enable(region, i)?;
    }
    Ok(())
}
