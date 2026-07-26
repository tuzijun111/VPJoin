//! Cardinality Preservation Check: condition (4) of the One-Pass OBJ.
//!
//! The One-Pass OBJ certifies a prover-supplied partition of every input
//! relation into a clean part `R_i^c` and a residual part `R_i^r` with four
//! conditions:
//!
//!   (1) `R_i == R_i^c U R_i^r`                     Conservation Check
//!   (2) `R_i^c /\ R_i^r == {}`                     Non-Membership Check
//!   (3) `pi_K(R_i^c) == pi_K(R_j^c)` per tree edge  Pairwise Consistency Check
//!   (4) `|R_1^c |X| ... |X| R_k^c| == |R_1 |X| ... |X| R_k|`
//!
//! Conditions (1)-(3) are local to a relation or to a tree edge and are
//! realized in each query circuit as before. Condition (4) is what this module
//! implements. It replaces the residual-side condition used previously, which
//! asked only that a semijoin reduction over the residual relations empty at
//! the root: that detects a join witness lying entirely on the residual side
//! but not one that mixes clean and residual tuples, since a tuple wrongly
//! moved into `R_i^r` whose join partners stay clean leaves no trace on the
//! residual side at all.
//!
//! Instead of traversing the residual side, (4) counts. The join of the clean
//! relations is always contained in the join of the inputs, so requiring the
//! two joins to have equal cardinality forces them to be the same multiset,
//! and no join occurrence, hence no participating tuple, can be missing.
//!
//! Both cardinalities come from one traversal of the join tree that carries
//! two multiplicities per tuple instead of the one propagated for aggregation:
//!
//!   mu_all(t_i) = pred_i(t) * prod_{j child of i} sigma_all_j(t_i[K_ij])
//!   mu_cln(t_i) = c_i(t) * pred_i(t) * prod_{j child of i} sigma_cln_j(t_i[K_ij])
//!
//! where `sigma_x_j(v) = sum over child tuples with key v of mu_x_j`, `pred_i`
//! is the in-relation predicate bit the circuit already evaluates per tuple,
//! and `c_i(t)` is an indicator of membership in `R_i^c`. Leaf tuples carry
//! `mu_all = pred` and `mu_cln = c * pred`. The two root sums are the two
//! sides of (4) and a single equality constraint compares them.
//!
//! Both channels run over the *same* rows, so they share the sorted view, the
//! group boundaries, the key lookup and the traversal; the second channel
//! costs only its own column, running sum and product.
//!
//! One structural point about the channel over the inputs: pairwise
//! consistency guarantees on the clean side that every parent key has a
//! matching child, but the dangling tuples of `R` break that guarantee. The
//! key-indexed table therefore carries a dummy row `(0, 0, 0)` and a parent
//! whose key occurs in no child tuple proves the absence with a gap witness
//! (`low < key < high` for two keys adjacent in the table) and takes
//! `sigma = 0`. This is the `sigma_j(v) = 0` default.
//!
//! Layout per tree edge `(R_i parent, R_j child)`:
//!
//!   child side  (`CpAggConfig`, one per edge, over the rows of `R_j`)
//!     v_all, v_cln          per-child-row multiplicities, in `R_j` row order
//!     sorted[key,all,cln]   permutation of `(K_ij, v_all, v_cln)`, key sorted
//!     run_all, run_cln      per-group running sums over the sorted view
//!     emit[3]               one `(key, sum_all, sum_cln)` per group, else PAD
//!     tbl[3], tbl_key_next  permutation of `emit`, keys sorted, PAD rows last
//!     map[3], map_key_next  `tbl` prefixed with the dummy row `(0, 0, 0)`
//!
//!   parent side (`CpJoinConfig`, one per edge, over the rows of `R_i`)
//!     in_tbl                is the parent key present in the child table?
//!     low, high             gap witness when it is not
//!     s_all, s_cln          the two fetched sigma values, zero when absent
//!
//!   root        (`CpRootConfig`, once, over the rows of `R_root`)
//!     mu_all, mu_cln        the two root multiplicities
//!     sum_all, sum_cln      running sums, compared once at the last row
//!
//! Keys must be nonzero: key `0` is reserved for the dummy table row. Query
//! circuits whose datasets can contain a zero key shift all keys by a constant
//! before they reach this gadget, which preserves equality joins.

use std::collections::HashMap;

use halo2_proofs::halo2curves::ff::PrimeField;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

// ---------------------------------------------------------------------------
// Child side of one edge: group the child rows by the join key and publish a
// key-indexed table of the two per-key sums.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CpAggConfig<F: Field + Ord, const N_BYTES: usize> {
    // per-child-row multiplicities, in the child relation's own row order
    pub v_all: Column<Advice>,
    pub v_cln: Column<Advice>,

    // sorted view of (key, v_all, v_cln)
    pub sorted: [Column<Advice>; 3],
    pub perm_sort: PermAnyConfig,

    pub q_sort: Selector,
    pub q_sentinel: Selector,
    pub lt_key_cur_next: LtConfig<F, N_BYTES>,
    pub iz_key_eq: IsZeroConfig<F>,

    // group-by over the sorted key column
    pub q_first: Selector,
    pub q_accu: Selector,
    pub q_emit: Selector,
    pub run_all: Column<Advice>,
    pub run_cln: Column<Advice>,
    pub iz_same_prev: IsZeroConfig<F>,
    pub iz_same_next: IsZeroConfig<F>,

    // one emitted (key, sum_all, sum_cln) per group, PAD elsewhere
    pub emit: [Column<Advice>; 3],

    // key-indexed table: a permutation of `emit` with the keys sorted
    pub tbl: [Column<Advice>; 3],
    pub tbl_key_next: Column<Advice>,
    pub perm_tbl: PermAnyConfig,
    pub q_tbl_sort: Selector,
    pub lt_tbl_cur_next: LtConfig<F, N_BYTES>,
    pub iz_tbl_eq: IsZeroConfig<F>,
    pub q_tbl_shift: Selector,
    pub q_tbl_last: Selector,

    // the same table with the dummy row (0, 0, 0) in front, used by the lookups
    pub map: [Column<Advice>; 3],
    pub map_key_next: Column<Advice>,
    pub q_map_tbl: Selector,
    pub q_map_first: Selector,
    pub q_map_link: Selector,
    pub q_map_shift: Selector,
    pub q_map_last: Selector,

    pub pad_key: u64,
}

/// Configures the child side of one tree edge.
///
/// `key_col` is the child relation's join-key column on its own base rows, and
/// `v_all` / `v_cln` are the two multiplicity columns on those same rows. The
/// caller owns those three columns and their values: at a leaf they are the
/// predicate bit and the clean indicator times the predicate bit, which a
/// query circuit usually already carries as columns, and at an internal node
/// they are fresh columns tied by the caller's product gate to the sums
/// fetched on that node's own child edges.
///
/// `u8_col` is a fixed column shared by every Lt chip of the gadget, so a
/// circuit with many edges pays one `load` for all of them.
pub fn configure_cp_agg<F: Field + Ord, const N_BYTES: usize>(
    meta: &mut ConstraintSystem<F>,
    u8_col: Column<Fixed>,
    key_col: Column<Advice>,
    v_all: Column<Advice>,
    v_cln: Column<Advice>,
    pad_key: u64,
) -> CpAggConfig<F, N_BYTES> {
    meta.enable_equality(v_all);
    meta.enable_equality(v_cln);

    let sorted = [
        meta.advice_column(),
        meta.advice_column(),
        meta.advice_column(),
    ];
    for c in sorted {
        meta.enable_equality(c);
    }

    // (key, v_all, v_cln) on the base rows is a permutation of the sorted view
    let q_perm_in = meta.complex_selector();
    let q_perm_out = meta.complex_selector();
    let perm_sort = PermAnyChip::configure(
        meta,
        q_perm_in,
        q_perm_out,
        vec![key_col, v_all, v_cln],
        sorted.to_vec(),
    );

    // the sorted key column is nondecreasing (row n is a PAD sentinel)
    let q_sort = meta.selector();
    let aux_key_eq = meta.advice_column();
    let iz_key_eq = IsZeroChip::configure(
        meta,
        |m| m.query_selector(q_sort),
        |m| m.query_advice(sorted[0], Rotation::next()) - m.query_advice(sorted[0], Rotation::cur()),
        aux_key_eq,
    );
    let lt_key_cur_next = LtChip::<F, N_BYTES>::configure_with_u8(
        meta,
        u8_col,
        |m| m.query_selector(q_sort),
        |m| m.query_advice(sorted[0], Rotation::cur()),
        |m| m.query_advice(sorted[0], Rotation::next()),
    );
    meta.create_gate("cp: sorted key nondecreasing", |m| {
        let q = m.query_selector(q_sort);
        let le = lt_key_cur_next.is_lt(m, None) + iz_key_eq.expr();
        vec![q * (le - Expression::Constant(F::ONE))]
    });

    // The group-boundary detector on the last real row reads the sentinel cell at
    // row n, and nondecreasing alone does not pin that cell. A prover may set it
    // equal to the last real key; then `iz_same_next` reports "same group" on the
    // last real row, the emit gate below writes PAD instead of that group's sums,
    // and the highest-key group disappears from the table the parent looks into.
    // A parent carrying that key would then certify its ABSENCE with a gap
    // witness that really does hold in the forged table and take sigma = 0,
    // lowering the input-side count; combined with a clean tuple hidden on the
    // other side, that is a way to satisfy the equality with an incomplete clean
    // instance.
    //
    // Pinning the sentinel to PAD closes it. Note this is deliberately NOT a
    // strict-increase requirement on the last comparison: a stage whose rows
    // include padding keyed at PAD (the bag stages of the cyclic queries do)
    // legitimately ends with real rows at PAD, and those must NOT be emitted as
    // a group. With the sentinel pinned, every group whose key is below PAD is
    // recognised as a group end, and a PAD-keyed padding group is correctly
    // skipped. It costs one selector and one degree-1 gate: no advice column,
    // no Lt chip and no lookup.
    let q_sentinel = meta.selector();
    meta.create_gate("cp: sorted view sentinel is PAD", |m| {
        let q = m.query_selector(q_sentinel);
        vec![q * (m.query_advice(sorted[0], Rotation::cur()) - Expression::Constant(F::from(pad_key)))]
    });

    // group-by over the sorted key column, one running sum per channel
    let q_first = meta.selector();
    let q_accu = meta.selector();
    let q_emit = meta.selector();

    let run_all = meta.advice_column();
    let run_cln = meta.advice_column();
    meta.enable_equality(run_all);
    meta.enable_equality(run_cln);

    let aux_same_prev = meta.advice_column();
    let iz_same_prev = IsZeroChip::configure(
        meta,
        |m| m.query_selector(q_accu),
        |m| m.query_advice(sorted[0], Rotation::cur()) - m.query_advice(sorted[0], Rotation::prev()),
        aux_same_prev,
    );

    let aux_same_next = meta.advice_column();
    let iz_same_next = IsZeroChip::configure(
        meta,
        |m| m.query_selector(q_emit),
        |m| m.query_advice(sorted[0], Rotation::next()) - m.query_advice(sorted[0], Rotation::cur()),
        aux_same_next,
    );

    meta.create_gate("cp: run_sum first row", |m| {
        let q = m.query_selector(q_first);
        vec![
            q.clone()
                * (m.query_advice(run_all, Rotation::cur())
                    - m.query_advice(sorted[1], Rotation::cur())),
            q * (m.query_advice(run_cln, Rotation::cur())
                - m.query_advice(sorted[2], Rotation::cur())),
        ]
    });

    meta.create_gate("cp: run_sum accumulate", |m| {
        let q = m.query_selector(q_accu);
        let same = iz_same_prev.expr(); // 1 when the previous row is in this group
        vec![
            q.clone()
                * (m.query_advice(run_all, Rotation::cur())
                    - (same.clone() * m.query_advice(run_all, Rotation::prev())
                        + m.query_advice(sorted[1], Rotation::cur()))),
            q * (m.query_advice(run_cln, Rotation::cur())
                - (same * m.query_advice(run_cln, Rotation::prev())
                    + m.query_advice(sorted[2], Rotation::cur()))),
        ]
    });

    let emit = [
        meta.advice_column(),
        meta.advice_column(),
        meta.advice_column(),
    ];
    for c in emit {
        meta.enable_equality(c);
    }

    meta.create_gate("cp: emit group sums at the last row of the group", |m| {
        let q = m.query_selector(q_emit);
        let one = Expression::Constant(F::ONE);
        let is_last = one.clone() - iz_same_next.expr();
        let not_last = one - is_last.clone();

        let cur_key = m.query_advice(sorted[0], Rotation::cur());
        let cur_all = m.query_advice(run_all, Rotation::cur());
        let cur_cln = m.query_advice(run_cln, Rotation::cur());

        let out_k = m.query_advice(emit[0], Rotation::cur());
        let out_all = m.query_advice(emit[1], Rotation::cur());
        let out_cln = m.query_advice(emit[2], Rotation::cur());

        let pad_k = Expression::Constant(F::from(pad_key));

        vec![
            q.clone() * (out_k - (is_last.clone() * cur_key + not_last * pad_k)),
            q.clone() * (out_all - is_last.clone() * cur_all),
            q * (out_cln - is_last * cur_cln),
        ]
    });

    // the key-indexed table: a permutation of `emit`, keys sorted
    let tbl = [
        meta.advice_column(),
        meta.advice_column(),
        meta.advice_column(),
    ];
    let tbl_key_next = meta.advice_column();
    for c in tbl {
        meta.enable_equality(c);
    }
    meta.enable_equality(tbl_key_next);

    let q_tbl_in = meta.complex_selector();
    let q_tbl_out = meta.complex_selector();
    let perm_tbl = PermAnyChip::configure(meta, q_tbl_in, q_tbl_out, emit.to_vec(), tbl.to_vec());

    let q_tbl_sort = meta.selector();
    let aux_tbl_eq = meta.advice_column();
    let iz_tbl_eq = IsZeroChip::configure(
        meta,
        |m| m.query_selector(q_tbl_sort),
        |m| m.query_advice(tbl[0], Rotation::next()) - m.query_advice(tbl[0], Rotation::cur()),
        aux_tbl_eq,
    );
    let lt_tbl_cur_next = LtChip::<F, N_BYTES>::configure_with_u8(
        meta,
        u8_col,
        |m| m.query_selector(q_tbl_sort),
        |m| m.query_advice(tbl[0], Rotation::cur()),
        |m| m.query_advice(tbl[0], Rotation::next()),
    );
    meta.create_gate("cp: table keys nondecreasing", |m| {
        let q = m.query_selector(q_tbl_sort);
        let le = lt_tbl_cur_next.is_lt(m, None) + iz_tbl_eq.expr();
        vec![q * (le - Expression::Constant(F::ONE))]
    });

    let q_tbl_shift = meta.selector();
    meta.create_gate("cp: tbl_key_next = next(tbl_key)", |m| {
        let q = m.query_selector(q_tbl_shift);
        vec![
            q * (m.query_advice(tbl_key_next, Rotation::cur())
                - m.query_advice(tbl[0], Rotation::next())),
        ]
    });

    let q_tbl_last = meta.selector();
    meta.create_gate("cp: tbl_key_next of the last row is PAD", |m| {
        let q = m.query_selector(q_tbl_last);
        vec![
            q * (m.query_advice(tbl_key_next, Rotation::cur())
                - Expression::Constant(F::from(pad_key))),
        ]
    });

    // the same table with the dummy row (0, 0, 0) in front
    let map = [
        meta.advice_column(),
        meta.advice_column(),
        meta.advice_column(),
    ];
    let map_key_next = meta.advice_column();
    for c in map {
        meta.enable_equality(c);
    }
    meta.enable_equality(map_key_next);

    let q_map_tbl = meta.complex_selector();
    let q_map_first = meta.selector();
    let q_map_link = meta.selector();
    let q_map_shift = meta.selector();
    let q_map_last = meta.selector();

    meta.create_gate("cp: map row 0 is the dummy (0, 0, 0)", |m| {
        let q = m.query_selector(q_map_first);
        vec![
            q.clone() * m.query_advice(map[0], Rotation::cur()),
            q.clone() * m.query_advice(map[1], Rotation::cur()),
            q * m.query_advice(map[2], Rotation::cur()),
        ]
    });

    meta.create_gate("cp: map[i+1] = tbl[i]", |m| {
        let q = m.query_selector(q_map_link);
        vec![
            q.clone()
                * (m.query_advice(map[0], Rotation::next()) - m.query_advice(tbl[0], Rotation::cur())),
            q.clone()
                * (m.query_advice(map[1], Rotation::next()) - m.query_advice(tbl[1], Rotation::cur())),
            q * (m.query_advice(map[2], Rotation::next()) - m.query_advice(tbl[2], Rotation::cur())),
        ]
    });

    meta.create_gate("cp: map_key_next = next(map_key)", |m| {
        let q = m.query_selector(q_map_shift);
        vec![
            q * (m.query_advice(map_key_next, Rotation::cur())
                - m.query_advice(map[0], Rotation::next())),
        ]
    });

    meta.create_gate("cp: map_key_next of the last row is PAD", |m| {
        let q = m.query_selector(q_map_last);
        vec![
            q * (m.query_advice(map_key_next, Rotation::cur())
                - Expression::Constant(F::from(pad_key))),
        ]
    });

    CpAggConfig {
        v_all,
        v_cln,
        sorted,
        perm_sort,
        q_sort,
        q_sentinel,
        lt_key_cur_next,
        iz_key_eq,
        q_first,
        q_accu,
        q_emit,
        run_all,
        run_cln,
        iz_same_prev,
        iz_same_next,
        emit,
        tbl,
        tbl_key_next,
        perm_tbl,
        q_tbl_sort,
        lt_tbl_cur_next,
        iz_tbl_eq,
        q_tbl_shift,
        q_tbl_last,
        map,
        map_key_next,
        q_map_tbl,
        q_map_first,
        q_map_link,
        q_map_shift,
        q_map_last,
        pad_key,
    }
}

// ---------------------------------------------------------------------------
// Parent side of one edge: fetch the two per-key sums for the parent's key,
// or certify that the key occurs in no child tuple and take zero.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CpJoinConfig<F: Field + Ord, const N_BYTES: usize> {
    pub in_tbl: Column<Advice>,
    pub low: Column<Advice>,
    pub high: Column<Advice>,
    pub s_all: Column<Advice>,
    pub s_cln: Column<Advice>,

    pub q_lookup: Selector,
    pub q_lookup_complex: Selector,

    pub lt_low: LtConfig<F, N_BYTES>,
    pub lt_high: LtConfig<F, N_BYTES>,
}

/// Configures the parent side of one tree edge. `key_col` is the parent's
/// join-key column for this edge, on the parent's base rows.
pub fn configure_cp_join<F: Field + Ord, const N_BYTES: usize>(
    meta: &mut ConstraintSystem<F>,
    u8_col: Column<Fixed>,
    key_col: Column<Advice>,
) -> CpJoinConfig<F, N_BYTES> {
    let in_tbl = meta.advice_column();
    let low = meta.advice_column();
    let high = meta.advice_column();
    let s_all = meta.advice_column();
    let s_cln = meta.advice_column();
    for c in [in_tbl, low, high, s_all, s_cln] {
        meta.enable_equality(c);
    }

    let q_lookup = meta.selector();
    let q_lookup_complex = meta.complex_selector();

    let lt_low = LtChip::<F, N_BYTES>::configure_with_u8(
        meta,
        u8_col,
        |m| {
            let q = m.query_selector(q_lookup);
            let inx = m.query_advice(in_tbl, Rotation::cur());
            q * (Expression::Constant(F::ONE) - inx)
        },
        |m| m.query_advice(low, Rotation::cur()),
        |m| m.query_advice(key_col, Rotation::cur()),
    );
    let lt_high = LtChip::<F, N_BYTES>::configure_with_u8(
        meta,
        u8_col,
        |m| {
            let q = m.query_selector(q_lookup);
            let inx = m.query_advice(in_tbl, Rotation::cur());
            q * (Expression::Constant(F::ONE) - inx)
        },
        |m| m.query_advice(key_col, Rotation::cur()),
        |m| m.query_advice(high, Rotation::cur()),
    );

    meta.create_gate("cp: membership flag, gap and zero default", |m| {
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

/// Wires the two lookup arguments that connect one parent side to one child
/// side. Call once per tree edge, after both sides have been configured.
///
/// `parent_key_col` must be the same column passed to `configure_cp_join`.
pub fn wire_cp_edge<F: Field + Ord, const N_BYTES: usize>(
    meta: &mut ConstraintSystem<F>,
    join: &CpJoinConfig<F, N_BYTES>,
    agg: &CpAggConfig<F, N_BYTES>,
    parent_key_col: Column<Advice>,
) {
    let j = join.clone();
    let t = agg.clone();

    // absent key: (low, high) must be a pair of keys adjacent in the table
    meta.lookup_any("cp: gap pair in the child table", move |m| {
        let q_in = m.query_selector(j.q_lookup_complex);
        let inx = m.query_advice(j.in_tbl, Rotation::cur());
        let gate = q_in * (Expression::Constant(F::ONE) - inx);

        let low = m.query_advice(j.low, Rotation::cur());
        let high = m.query_advice(j.high, Rotation::cur());

        let q_tbl = m.query_selector(t.q_map_tbl);
        let key = m.query_advice(t.map[0], Rotation::cur());
        let keyn = m.query_advice(t.map_key_next, Rotation::cur());

        vec![
            (gate.clone() * low, q_tbl.clone() * key),
            (gate * high, q_tbl * keyn),
        ]
    });

    let j = join.clone();
    let t = agg.clone();

    // present key: (key, s_all, s_cln) must be a row of the table. When the
    // flag is 0 the gate above forces both sums to 0 and the queried tuple is
    // (0, 0, 0), which is the dummy row.
    meta.lookup_any("cp: key and its two sums in the child table", move |m| {
        let q_in = m.query_selector(j.q_lookup_complex);
        let inx = m.query_advice(j.in_tbl, Rotation::cur());

        let key = m.query_advice(parent_key_col, Rotation::cur());
        let s_all = m.query_advice(j.s_all, Rotation::cur());
        let s_cln = m.query_advice(j.s_cln, Rotation::cur());

        let q_tbl = m.query_selector(t.q_map_tbl);
        let tk = m.query_advice(t.map[0], Rotation::cur());
        let t_all = m.query_advice(t.map[1], Rotation::cur());
        let t_cln = m.query_advice(t.map[2], Rotation::cur());

        vec![
            (q_in.clone() * inx * key, q_tbl.clone() * tk),
            (q_in.clone() * s_all, q_tbl.clone() * t_all),
            (q_in * s_cln, q_tbl * t_cln),
        ]
    });
}

// ---------------------------------------------------------------------------
// Key edges: the specialization for an edge whose child holds at most one
// tuple per join-key value.
// ---------------------------------------------------------------------------
//
// On such an edge every sigma is 0 or 1, and two things collapse.
//
// The per-key aggregation disappears. With at most one child tuple per key,
// `sigma_j(v)` is that tuple's own multiplicity, so the sorted view, the group
// boundaries, the running sums, the emitted pairs and the emit-to-table
// permutation of `configure_cp_agg` all reduce to one sorted, key-indexed copy
// of the child's rows, which is the map table the parent already looks into.
//
// The clean channel disappears too, which is the larger saving and is worth
// stating as a lemma, since it is what makes this a specialization rather than
// an assumption:
//
//   On a tree all of whose edges are key edges, `mu_cln(t) = c(t) * pred(t)`
//   at every tuple, so the clean root sum is just the number of clean root
//   tuples and no clean multiplicity has to be propagated at all.
//
// Proof, by induction from the leaves. At a leaf `mu_cln = c * pred` by
// definition. At an internal node, take a tuple `t` with `c(t) = 1`. Condition
// (9), Pairwise Consistency, gives `pi_K(t) in pi_K(R_j^c)` on each child edge,
// so some clean child tuple carries that key; because the child holds at most
// one tuple per key, it is the only one, and it lies in `R_j^c`, hence its
// indicator is 1 and, since the clean side is built from tuples that pass their
// predicate, its predicate bit is 1 as well. By induction its `mu_cln` is 1, so
// `sigma_cln_j(pi_K(t)) = 1` for every child and the product contributes
// nothing. For `c(t) = 0` the whole product is multiplied by zero anyway. []
//
// So a key edge needs no `v_cln`, no `sigma_cln` and no third map column, and
// the root's clean multiplicity is read straight off the bound indicator.
//
// The precondition is not assumed: the map's key column is constrained to be
// STRICTLY increasing, which certifies that no two child rows share a key. If a
// query circuit uses this on an edge whose child does have repeated keys, the
// honest prover simply cannot satisfy it, so the specialization cannot be
// applied where it does not hold.
//
// What does NOT collapse is absence certification on the parent side: a parent
// whose key occurs in no child tuple must still exhibit a gap witness, or a
// prover could under-report `sigma_all`, lower the input-side count and match it
// against an incomplete clean side. `CpJoinConfig` is therefore reused
// unchanged, and `s_cln` simply goes unread.

#[derive(Clone, Debug)]
pub struct CpKeyConfig<F: Field + Ord, const N_BYTES: usize> {
    /// the child's input-channel multiplicity, on the child's own base rows;
    /// caller-owned, exactly as for `configure_cp_agg`
    pub v_all: Column<Advice>,

    /// `(key, mu_all)` sorted by key, with the dummy row `(0, 0)` in front, so
    /// this is both the sorted view and the lookup table
    pub map: [Column<Advice>; 2],
    pub map_key_next: Column<Advice>,
    pub perm_map: PermAnyConfig,

    pub q_sort: Selector,
    pub lt_key_cur_next: LtConfig<F, N_BYTES>,

    pub q_map_tbl: Selector,
    pub q_map_first: Selector,
    pub q_map_shift: Selector,
    pub q_map_last: Selector,

    pub pad_key: u64,
}

/// Configures the child side of a key edge. `key_col` and `v_all` are the
/// child's join-key column and input-channel multiplicity on its base rows.
pub fn configure_cp_key<F: Field + Ord, const N_BYTES: usize>(
    meta: &mut ConstraintSystem<F>,
    u8_col: Column<Fixed>,
    key_col: Column<Advice>,
    v_all: Column<Advice>,
    pad_key: u64,
) -> CpKeyConfig<F, N_BYTES> {
    meta.enable_equality(v_all);

    let map = [meta.advice_column(), meta.advice_column()];
    let map_key_next = meta.advice_column();
    for c in map {
        meta.enable_equality(c);
    }
    meta.enable_equality(map_key_next);

    // rows 1..=n of the map are a permutation of the child's (key, mu_all)
    let q_perm_in = meta.complex_selector();
    let q_perm_out = meta.complex_selector();
    let perm_map = PermAnyChip::configure(
        meta,
        q_perm_in,
        q_perm_out,
        vec![key_col, v_all],
        map.to_vec(),
    );

    // STRICTLY increasing keys: this both orders the table for the gap witness
    // and certifies that the child holds at most one tuple per key, which is
    // the precondition of the whole specialization
    let q_sort = meta.selector();
    let lt_key_cur_next = LtChip::<F, N_BYTES>::configure_with_u8(
        meta,
        u8_col,
        |m| m.query_selector(q_sort),
        |m| m.query_advice(map[0], Rotation::cur()),
        |m| m.query_advice(map[0], Rotation::next()),
    );
    meta.create_gate("cp key: map keys strictly increasing", |m| {
        let q = m.query_selector(q_sort);
        vec![q * (lt_key_cur_next.is_lt(m, None) - Expression::Constant(F::ONE))]
    });

    let q_map_tbl = meta.complex_selector();
    let q_map_first = meta.selector();
    let q_map_shift = meta.selector();
    let q_map_last = meta.selector();

    meta.create_gate("cp key: map row 0 is the dummy (0, 0)", |m| {
        let q = m.query_selector(q_map_first);
        vec![
            q.clone() * m.query_advice(map[0], Rotation::cur()),
            q * m.query_advice(map[1], Rotation::cur()),
        ]
    });

    meta.create_gate("cp key: map_key_next = next(map_key)", |m| {
        let q = m.query_selector(q_map_shift);
        vec![
            q * (m.query_advice(map_key_next, Rotation::cur())
                - m.query_advice(map[0], Rotation::next())),
        ]
    });

    meta.create_gate("cp key: map_key_next of the last row is PAD", |m| {
        let q = m.query_selector(q_map_last);
        vec![
            q * (m.query_advice(map_key_next, Rotation::cur())
                - Expression::Constant(F::from(pad_key))),
        ]
    });

    CpKeyConfig {
        v_all,
        map,
        map_key_next,
        perm_map,
        q_sort,
        lt_key_cur_next,
        q_map_tbl,
        q_map_first,
        q_map_shift,
        q_map_last,
        pad_key,
    }
}

/// Wires one key edge's two lookup arguments. `join` is the ordinary parent
/// side; only its `s_all` output is read, since a key edge propagates no clean
/// multiplicity.
pub fn wire_cp_key_edge<F: Field + Ord, const N_BYTES: usize>(
    meta: &mut ConstraintSystem<F>,
    join: &CpJoinConfig<F, N_BYTES>,
    key: &CpKeyConfig<F, N_BYTES>,
    parent_key_col: Column<Advice>,
) {
    let j = join.clone();
    let t = key.clone();

    meta.lookup_any("cp key: gap pair in the child table", move |m| {
        let q_in = m.query_selector(j.q_lookup_complex);
        let inx = m.query_advice(j.in_tbl, Rotation::cur());
        let gate = q_in * (Expression::Constant(F::ONE) - inx);

        let q_tbl = m.query_selector(t.q_map_tbl);
        vec![
            (
                gate.clone() * m.query_advice(j.low, Rotation::cur()),
                q_tbl.clone() * m.query_advice(t.map[0], Rotation::cur()),
            ),
            (
                gate * m.query_advice(j.high, Rotation::cur()),
                q_tbl * m.query_advice(t.map_key_next, Rotation::cur()),
            ),
        ]
    });

    let j = join.clone();
    let t = key.clone();

    meta.lookup_any("cp key: key and its multiplicity in the child table", move |m| {
        let q_in = m.query_selector(j.q_lookup_complex);
        let inx = m.query_advice(j.in_tbl, Rotation::cur());
        let q_tbl = m.query_selector(t.q_map_tbl);
        vec![
            (
                q_in.clone() * inx * m.query_advice(parent_key_col, Rotation::cur()),
                q_tbl.clone() * m.query_advice(t.map[0], Rotation::cur()),
            ),
            (
                q_in * m.query_advice(j.s_all, Rotation::cur()),
                q_tbl * m.query_advice(t.map[1], Rotation::cur()),
            ),
        ]
    });
}

/// Host side of one key-edge child stage.
#[derive(Clone, Debug, Default)]
pub struct CpKeyStage {
    /// `(key, mu_all)` sorted by key; the map occupies rows `1..=n`.
    pub sorted: Vec<[u64; 2]>,
    /// `key_next[i]` for map row `i`, so index 0 is the dummy row's next key.
    pub key_next: Vec<u64>,
    /// `key -> mu_all`, for the parent side.
    pub map: HashMap<u64, u64>,
    /// the map's key set including `0` and PAD, for gap witnesses.
    pub keys: Vec<u64>,
}

/// Builds a key-edge child stage from `rows[i] = [key, mu_all]`. Panics if two
/// rows share a key, since the circuit's strict-increase check would reject
/// that witness anyway and failing here names the problem.
pub fn build_cp_key_stage(rows: &[[u64; 2]], pad_key: u64) -> CpKeyStage {
    let mut sorted = rows.to_vec();
    sorted.sort_by_key(|r| r[0]);
    for w in sorted.windows(2) {
        assert!(
            w[0][0] != w[1][0],
            "cp key edge: the child holds two rows with key {}, so this edge is \
             not a key edge and needs the general configure_cp_agg stage",
            w[0][0]
        );
    }

    let n = sorted.len();
    // map row 0 is the dummy, rows 1..=n hold `sorted`
    let mut key_next = vec![pad_key; n + 1];
    for i in 0..=n {
        key_next[i] = if i < n { sorted[i][0] } else { pad_key };
    }

    let mut map: HashMap<u64, u64> = HashMap::new();
    for r in sorted.iter() {
        map.insert(r[0], r[1]);
    }

    let mut keys: Vec<u64> = sorted.iter().map(|r| r[0]).collect();
    keys.push(0);
    keys.push(pad_key);
    keys.sort();
    keys.dedup();

    CpKeyStage {
        sorted,
        key_next,
        map,
        keys,
    }
}

/// Assigns one key-edge child stage. The caller has already assigned the key
/// column and `v_all` on the child's `rows.len()` base rows.
pub fn assign_cp_key<F: Field + Ord, const N_BYTES: usize>(
    region: &mut Region<'_, F>,
    a: &CpKeyConfig<F, N_BYTES>,
    rows: &[[u64; 2]],
    w: &CpKeyStage,
) -> Result<(), Error> {
    let n = rows.len();
    if n == 0 {
        return Ok(());
    }
    let pad = a.pad_key;

    // the dummy row, then the sorted child rows
    region.assign_advice(|| "cp key map dummy k", a.map[0], 0, || Value::known(F::ZERO))?;
    region.assign_advice(|| "cp key map dummy v", a.map[1], 0, || Value::known(F::ZERO))?;
    for i in 0..n {
        region.assign_advice(
            || "cp key map k",
            a.map[0],
            i + 1,
            || Value::known(F::from(w.sorted[i][0])),
        )?;
        region.assign_advice(
            || "cp key map v",
            a.map[1],
            i + 1,
            || Value::known(F::from(w.sorted[i][1])),
        )?;
    }
    // a PAD sentinel one past the table, so the strict-increase gate at the
    // last real row has a Rotation::next() to read
    region.assign_advice(
        || "cp key map k sentinel",
        a.map[0],
        n + 1,
        || Value::known(F::from(pad)),
    )?;

    for i in 0..=n {
        region.assign_advice(
            || "cp key map_key_next",
            a.map_key_next,
            i,
            || Value::known(F::from(w.key_next[i])),
        )?;
    }

    for i in 0..=n {
        a.q_map_tbl.enable(region, i)?;
    }
    a.q_map_first.enable(region, 0)?;
    for i in 0..n {
        a.q_map_shift.enable(region, i)?;
    }
    a.q_map_last.enable(region, n)?;

    // the permutation runs over the child's base rows against map rows 1..=n
    for i in 0..n {
        a.perm_map.q_perm1.enable(region, i)?;
        a.perm_map.q_perm2.enable(region, i + 1)?;
    }

    // strict increase over the whole table, dummy row included
    let lt = LtChip::<F, N_BYTES>::construct(a.lt_key_cur_next);
    for i in 0..=n {
        a.q_sort.enable(region, i)?;
        let cur = if i == 0 { 0 } else { w.sorted[i - 1][0] };
        let next = if i < n { w.sorted[i][0] } else { pad };
        lt.assign(
            region,
            i,
            Value::known(F::from(cur)),
            Value::known(F::from(next)),
        )?;
    }

    Ok(())
}

/// Assigns one key-edge parent side and returns the fetched `mu_all` bit per
/// parent row. `s_cln` is written as 0 on every row, since a key edge
/// propagates no clean multiplicity and the gate reads only `s_all`.
pub fn assign_cp_key_join<F: Field + Ord, const N_BYTES: usize>(
    region: &mut Region<'_, F>,
    j: &CpJoinConfig<F, N_BYTES>,
    parent_keys: &[u64],
    child: &CpKeyStage,
    pad_key: u64,
) -> Result<Vec<u64>, Error> {
    let lt_low = LtChip::<F, N_BYTES>::construct(j.lt_low);
    let lt_high = LtChip::<F, N_BYTES>::construct(j.lt_high);

    let mut fetched = Vec::with_capacity(parent_keys.len());
    for (i, &key) in parent_keys.iter().enumerate() {
        let (inx, low, high) = cp_gap_witness(&child.keys, key, pad_key);
        let s_all = if inx == 1 {
            child.map.get(&key).copied().unwrap_or(0)
        } else {
            0
        };

        j.q_lookup.enable(region, i)?;
        j.q_lookup_complex.enable(region, i)?;

        region.assign_advice(|| "cp key in_tbl", j.in_tbl, i, || Value::known(F::from(inx)))?;
        region.assign_advice(|| "cp key low", j.low, i, || Value::known(F::from(low)))?;
        region.assign_advice(|| "cp key high", j.high, i, || Value::known(F::from(high)))?;
        region.assign_advice(|| "cp key s_all", j.s_all, i, || Value::known(F::from(s_all)))?;
        region.assign_advice(|| "cp key s_cln", j.s_cln, i, || Value::known(F::ZERO))?;

        lt_low.assign(
            region,
            i,
            Value::known(F::from(low)),
            Value::known(F::from(key)),
        )?;
        lt_high.assign(
            region,
            i,
            Value::known(F::from(key)),
            Value::known(F::from(high)),
        )?;

        fetched.push(s_all);
    }
    Ok(fetched)
}

// ---------------------------------------------------------------------------
// Root: accumulate both channels over the root relation and compare them once.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CpRootConfig {
    pub mu_all: Column<Advice>,
    pub mu_cln: Column<Advice>,
    pub sum_all: Column<Advice>,
    pub sum_cln: Column<Advice>,
    pub q_first: Selector,
    pub q_accu: Selector,
    pub q_equal: Selector,
}

/// Configures the root accumulator. The caller wires `mu_all` and `mu_cln`
/// with its own product gate over the root relation's rows.
pub fn configure_cp_root<F: Field + Ord>(meta: &mut ConstraintSystem<F>) -> CpRootConfig {
    let mu_all = meta.advice_column();
    let mu_cln = meta.advice_column();
    let sum_all = meta.advice_column();
    let sum_cln = meta.advice_column();
    for c in [mu_all, mu_cln, sum_all, sum_cln] {
        meta.enable_equality(c);
    }

    let q_first = meta.selector();
    let q_accu = meta.selector();
    let q_equal = meta.selector();

    meta.create_gate("cp: root sums first row", |m| {
        let q = m.query_selector(q_first);
        vec![
            q.clone()
                * (m.query_advice(sum_all, Rotation::cur()) - m.query_advice(mu_all, Rotation::cur())),
            q * (m.query_advice(sum_cln, Rotation::cur()) - m.query_advice(mu_cln, Rotation::cur())),
        ]
    });

    meta.create_gate("cp: root sums accumulate", |m| {
        let q = m.query_selector(q_accu);
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

    // condition (4): the two join cardinalities agree
    meta.create_gate("cp: cardinality preservation", |m| {
        let q = m.query_selector(q_equal);
        vec![
            q * (m.query_advice(sum_all, Rotation::cur())
                - m.query_advice(sum_cln, Rotation::cur())),
        ]
    });

    CpRootConfig {
        mu_all,
        mu_cln,
        sum_all,
        sum_cln,
        q_first,
        q_accu,
        q_equal,
    }
}

// ---------------------------------------------------------------------------
// Host side: the witnesses of one child stage.
// ---------------------------------------------------------------------------

/// Everything the prover computes for the child side of one edge. All of it is
/// derived from `rows`, so nothing here is a free choice.
#[derive(Clone, Debug, Default)]
pub struct CpStage {
    /// `(key, v_all, v_cln)` sorted by key; same length as `rows`.
    pub sorted: Vec<[u64; 3]>,
    /// per-group running sums over `sorted`.
    pub run_all: Vec<u64>,
    pub run_cln: Vec<u64>,
    /// one `(key, sum_all, sum_cln)` at the last row of each group, else PAD.
    pub emit: Vec<[u64; 3]>,
    /// `emit` with the PAD rows moved to the back and the keys sorted.
    pub tbl: Vec<[u64; 3]>,
    /// `key_next[i] == tbl[i + 1].key`, PAD at the last row.
    pub key_next: Vec<u64>,
    /// `key -> (sum_all, sum_cln)`, for the parent side.
    pub map: HashMap<u64, (u64, u64)>,
    /// the map's key set including `0` and PAD, for gap witnesses.
    pub keys: Vec<u64>,
}

/// Builds a child stage from `rows[i] = [key, v_all, v_cln]` given in the
/// child relation's own row order.
pub fn build_cp_stage(rows: &[[u64; 3]], pad_key: u64) -> CpStage {
    let n = rows.len();

    let mut sorted = rows.to_vec();
    sorted.sort_by_key(|r| r[0]);

    let mut run_all = vec![0u64; n];
    let mut run_cln = vec![0u64; n];
    let mut acc_all: u128 = 0;
    let mut acc_cln: u128 = 0;
    let mut prev: Option<u64> = None;
    for i in 0..n {
        let k = sorted[i][0];
        if prev == Some(k) {
            acc_all += sorted[i][1] as u128;
            acc_cln += sorted[i][2] as u128;
        } else {
            acc_all = sorted[i][1] as u128;
            acc_cln = sorted[i][2] as u128;
        }
        run_all[i] = acc_all as u64;
        run_cln[i] = acc_cln as u64;
        prev = Some(k);
    }

    let mut emit = vec![[pad_key, 0u64, 0u64]; n];
    for i in 0..n {
        let cur = sorted[i][0];
        let next = if i + 1 < n { sorted[i + 1][0] } else { pad_key };
        if next != cur {
            emit[i] = [cur, run_all[i], run_cln[i]];
        }
    }

    let mut tbl: Vec<[u64; 3]> = emit.iter().copied().filter(|r| r[0] != pad_key).collect();
    tbl.sort_by_key(|r| r[0]);
    while tbl.len() < n {
        tbl.push([pad_key, 0, 0]);
    }
    tbl.truncate(n);

    let mut key_next = vec![pad_key; n];
    for i in 0..n {
        key_next[i] = if i + 1 < n { tbl[i + 1][0] } else { pad_key };
    }

    let mut map: HashMap<u64, (u64, u64)> = HashMap::new();
    for r in tbl.iter() {
        if r[0] != pad_key {
            map.insert(r[0], (r[1], r[2]));
        }
    }

    let mut keys: Vec<u64> = tbl.iter().map(|r| r[0]).filter(|&k| k != pad_key).collect();
    keys.push(0);
    keys.push(pad_key);
    keys.sort();
    keys.dedup();

    CpStage {
        sorted,
        run_all,
        run_cln,
        emit,
        tbl,
        key_next,
        map,
        keys,
    }
}

/// `(in_tbl, low, high)` for one parent key against a child stage's key set.
pub fn cp_gap_witness(keys: &[u64], x: u64, pad_key: u64) -> (u64, u64, u64) {
    match keys.binary_search(&x) {
        Ok(_) => (1, 0, pad_key),
        Err(idx) => {
            let high = keys[idx];
            let low = if idx == 0 { 0 } else { keys[idx - 1] };
            (0, low, high)
        }
    }
}

// ---------------------------------------------------------------------------
// Assignment.
// ---------------------------------------------------------------------------

/// Assigns one child stage over a child relation of `n` rows. The caller has
/// already assigned the key column and the two multiplicity columns on those
/// rows, since it owns them; `rows` is only used for its length here, and `w`
/// is the stage `build_cp_stage` derived from the same rows.
pub fn assign_cp_agg<F: Field + Ord, const N_BYTES: usize>(
    region: &mut Region<'_, F>,
    a: &CpAggConfig<F, N_BYTES>,
    rows: &[[u64; 3]],
    w: &CpStage,
) -> Result<(), Error> {
    let n = rows.len();
    if n == 0 {
        return Ok(());
    }
    let pad = a.pad_key;

    // the sorted view, plus a PAD sentinel row at n for the Rotation::next()
    // queries of the sortedness and emit gates
    for i in 0..n {
        for j in 0..3 {
            region.assign_advice(
                || "cp sorted",
                a.sorted[j],
                i,
                || Value::known(F::from(w.sorted[i][j])),
            )?;
        }
    }
    region.assign_advice(|| "cp sorted key sentinel", a.sorted[0], n, || Value::known(F::from(pad)))?;
    region.assign_advice(|| "cp sorted all sentinel", a.sorted[1], n, || Value::known(F::ZERO))?;
    region.assign_advice(|| "cp sorted cln sentinel", a.sorted[2], n, || Value::known(F::ZERO))?;

    for i in 0..n {
        region.assign_advice(
            || "cp run_all",
            a.run_all,
            i,
            || Value::known(F::from(w.run_all[i])),
        )?;
        region.assign_advice(
            || "cp run_cln",
            a.run_cln,
            i,
            || Value::known(F::from(w.run_cln[i])),
        )?;
        for j in 0..3 {
            region.assign_advice(
                || "cp emit",
                a.emit[j],
                i,
                || Value::known(F::from(w.emit[i][j])),
            )?;
            region.assign_advice(
                || "cp tbl",
                a.tbl[j],
                i,
                || Value::known(F::from(w.tbl[i][j])),
            )?;
        }
        region.assign_advice(
            || "cp tbl_key_next",
            a.tbl_key_next,
            i,
            || Value::known(F::from(w.key_next[i])),
        )?;
    }

    // the map table: row 0 is the dummy, rows 1..=n copy the table
    for j in 0..3 {
        region.assign_advice(|| "cp map dummy", a.map[j], 0, || Value::known(F::ZERO))?;
    }
    for i in 0..n {
        for j in 0..3 {
            region.assign_advice(
                || "cp map",
                a.map[j],
                i + 1,
                || Value::known(F::from(w.tbl[i][j])),
            )?;
        }
    }
    for r in 0..n {
        region.assign_advice(
            || "cp map_key_next",
            a.map_key_next,
            r,
            || Value::known(F::from(w.tbl[r][0])),
        )?;
    }
    region.assign_advice(
        || "cp map_key_next last",
        a.map_key_next,
        n,
        || Value::known(F::from(pad)),
    )?;

    // selectors
    for i in 0..=n {
        a.q_map_tbl.enable(region, i)?;
    }
    a.q_map_first.enable(region, 0)?;
    for i in 0..n {
        a.q_map_link.enable(region, i)?;
        a.q_map_shift.enable(region, i)?;
    }
    a.q_map_last.enable(region, n)?;

    for i in 0..n {
        a.perm_sort.q_perm1.enable(region, i)?;
        a.perm_sort.q_perm2.enable(region, i)?;
        a.perm_tbl.q_perm1.enable(region, i)?;
        a.perm_tbl.q_perm2.enable(region, i)?;
        a.q_sort.enable(region, i)?;
        a.q_emit.enable(region, i)?;
    }
    // pin the sentinel, so the final group is recognised as a group end
    a.q_sentinel.enable(region, n)?;
    for i in 0..n.saturating_sub(1) {
        a.q_tbl_sort.enable(region, i)?;
        a.q_tbl_shift.enable(region, i)?;
    }
    a.q_tbl_last.enable(region, n - 1)?;
    a.q_first.enable(region, 0)?;
    for i in 1..n {
        a.q_accu.enable(region, i)?;
    }

    // helper chips
    let lt_key = LtChip::<F, N_BYTES>::construct(a.lt_key_cur_next.clone());
    let lt_tbl = LtChip::<F, N_BYTES>::construct(a.lt_tbl_cur_next.clone());
    let iz_key = IsZeroChip::construct(a.iz_key_eq.clone());
    let iz_tbl = IsZeroChip::construct(a.iz_tbl_eq.clone());
    let iz_prev = IsZeroChip::construct(a.iz_same_prev.clone());
    let iz_next = IsZeroChip::construct(a.iz_same_next.clone());

    for i in 0..n {
        let cur = w.sorted[i][0];
        let next = if i + 1 < n { w.sorted[i + 1][0] } else { pad };
        lt_key.assign(
            region,
            i,
            Value::known(F::from(cur)),
            Value::known(F::from(next)),
        )?;
        iz_key.assign(region, i, Value::known(F::from(next) - F::from(cur)))?;
        iz_next.assign(region, i, Value::known(F::from(next) - F::from(cur)))?;
    }
    for i in 1..n {
        let diff = F::from(w.sorted[i][0]) - F::from(w.sorted[i - 1][0]);
        iz_prev.assign(region, i, Value::known(diff))?;
    }
    for i in 0..n.saturating_sub(1) {
        lt_tbl.assign(
            region,
            i,
            Value::known(F::from(w.tbl[i][0])),
            Value::known(F::from(w.tbl[i + 1][0])),
        )?;
        let diff = F::from(w.tbl[i + 1][0]) - F::from(w.tbl[i][0]);
        iz_tbl.assign(region, i, Value::known(diff))?;
    }

    Ok(())
}

/// Assigns one parent side and returns the fetched `(s_all, s_cln)` per parent
/// row, so the caller can form its product gates from the same values.
pub fn assign_cp_join<F: Field + Ord, const N_BYTES: usize>(
    region: &mut Region<'_, F>,
    j: &CpJoinConfig<F, N_BYTES>,
    parent_keys: &[u64],
    child: &CpStage,
    pad_key: u64,
) -> Result<Vec<(u64, u64)>, Error> {
    let lt_low = LtChip::<F, N_BYTES>::construct(j.lt_low.clone());
    let lt_high = LtChip::<F, N_BYTES>::construct(j.lt_high.clone());

    let mut fetched = Vec::with_capacity(parent_keys.len());

    for (i, &key) in parent_keys.iter().enumerate() {
        let (inx, low, high) = cp_gap_witness(&child.keys, key, pad_key);
        let (s_all, s_cln) = if inx == 1 {
            child.map.get(&key).copied().unwrap_or((0, 0))
        } else {
            (0, 0)
        };

        j.q_lookup.enable(region, i)?;
        j.q_lookup_complex.enable(region, i)?;

        region.assign_advice(|| "cp in_tbl", j.in_tbl, i, || Value::known(F::from(inx)))?;
        region.assign_advice(|| "cp low", j.low, i, || Value::known(F::from(low)))?;
        region.assign_advice(|| "cp high", j.high, i, || Value::known(F::from(high)))?;
        region.assign_advice(|| "cp s_all", j.s_all, i, || Value::known(F::from(s_all)))?;
        region.assign_advice(|| "cp s_cln", j.s_cln, i, || Value::known(F::from(s_cln)))?;

        // the Lt chips are gated by (1 - in_tbl), but their witnesses are
        // assigned on every row: an unconstrained row costs nothing and an
        // unassigned cell would be an error under the real prover
        lt_low.assign(
            region,
            i,
            Value::known(F::from(low)),
            Value::known(F::from(key)),
        )?;
        lt_high.assign(
            region,
            i,
            Value::known(F::from(key)),
            Value::known(F::from(high)),
        )?;

        fetched.push((s_all, s_cln));
    }

    Ok(fetched)
}

/// Assigns the root accumulator over `mu[i] = (mu_all, mu_cln)` and returns
/// the two final sums, i.e. the two sides of condition (4).
pub fn assign_cp_root<F: Field + Ord>(
    region: &mut Region<'_, F>,
    r: &CpRootConfig,
    mu: &[(u64, u64)],
) -> Result<(u64, u64), Error> {
    let n = mu.len();
    if n == 0 {
        return Ok((0, 0));
    }

    let mut acc_all: u128 = 0;
    let mut acc_cln: u128 = 0;
    for i in 0..n {
        acc_all += mu[i].0 as u128;
        acc_cln += mu[i].1 as u128;
        region.assign_advice(|| "cp mu_all", r.mu_all, i, || Value::known(F::from(mu[i].0)))?;
        region.assign_advice(|| "cp mu_cln", r.mu_cln, i, || Value::known(F::from(mu[i].1)))?;
        region.assign_advice(
            || "cp sum_all",
            r.sum_all,
            i,
            || Value::known(F::from(acc_all as u64)),
        )?;
        region.assign_advice(
            || "cp sum_cln",
            r.sum_cln,
            i,
            || Value::known(F::from(acc_cln as u64)),
        )?;
    }

    r.q_first.enable(region, 0)?;
    for i in 1..n {
        r.q_accu.enable(region, i)?;
    }
    r.q_equal.enable(region, n - 1)?;

    Ok((acc_all as u64, acc_cln as u64))
}

/// Loads the shared u8 range table once. Call from `assign` before the main
/// region, exactly like the other Lt chips of a query circuit.
pub fn load_cp_u8<F: Field + Ord, const N_BYTES: usize>(
    layouter: &mut impl Layouter<F>,
    any_lt: &LtConfig<F, N_BYTES>,
) -> Result<(), Error> {
    LtChip::<F, N_BYTES>::construct(*any_lt).load(layouter)
}
