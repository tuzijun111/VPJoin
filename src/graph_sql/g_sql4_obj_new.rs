use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::circuits::conserve_idx::{
    assign_conserve, assign_row_index, configure_conserve, configure_row_index, ConserveConfig,
    RowIndexConfig,
};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};
use crate::circuits::card_preserve::{
    assign_cp_agg, assign_cp_join, assign_cp_root, build_cp_stage, configure_cp_agg,
    configure_cp_join, configure_cp_root, wire_cp_edge, CpAggConfig, CpJoinConfig, CpRootConfig,
};

// ✅ Use the dataset Edge type directly (no conversion needed).
use crate::data::graph_data_processing::Edge;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

// `pub(crate)` so the multi-lane DP variant `g_sql4_obj_dp` inherits exactly
// the same PAD / packing conventions instead of restating them.
pub(crate) const NUM_BYTES: usize = 8;
pub(crate) const PAD_U64: u64 = u64::MAX;

// shift node IDs by +1 so 0 can be reserved for dummy row
const SHIFT_ID: u64 = 1;

// pack2(hi, lo) using 32-bit lanes (assumes hi,lo < 2^32)
const PACK_BITS: u32 = 32;
pub(crate) const PACK_SHIFT: u64 = 1u64 << PACK_BITS;
pub(crate) fn pack2(hi: u64, lo: u64) -> u64 {
    hi * PACK_SHIFT + lo
}

/// Test hook, off in every benchmark path: when set, the prover moves one
/// joinable role-1 tuple to the residual side and re-reduces role 2 around it, so
/// the partition still passes Conservation, Non-Membership and Pairwise
/// Consistency and only condition (4) can catch it. This is exactly the cheat
/// a residual-side-only argument misses, so the negative test in this module is
/// what shows the Cardinality Preservation Check is not vacuous.
pub static HIDE_ONE_CLEAN_TUPLE: AtomicBool = AtomicBool::new(false);

/// Test hook, off in every benchmark path: when set, the prover skips the
/// semijoin reduction entirely and declares every real tuple clean in both
/// roles, leaving the residual side empty. Conservation still holds and both
/// channels of condition (4) then agree on every row, so this is exactly the
/// escape that Pairwise Consistency has to close.
pub static MARK_ALL_CLEAN: AtomicBool = AtomicBool::new(false);

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

/// Assigns one column group of a Conservation Check, `rows[i][j]` into column
/// `cols[j]` at row `i`.
fn assign_perm_group<F: Field + Ord>(
    region: &mut Region<'_, F>,
    name: &'static str,
    cols: &[Column<Advice>],
    rows: &[Vec<u64>],
) -> Result<(), Error> {
    for (i, r) in rows.iter().enumerate() {
        for (j, &col) in cols.iter().enumerate() {
            region.assign_advice(|| name, col, i, || Value::known(F::from(r[j])))?;
        }
    }
    Ok(())
}

/// Splits one bag into the two sides of its Conservation Check.
///
/// `tuples[i]` is the bag tuple on base row `i`, `keep[i]` its predicate bit
/// and `cln[i]` its clean indicator. Returns the input side (the tuple when the
/// predicate holds, PAD otherwise, with the indicator as the last column), the
/// partition side laid out as [clean rows | residual rows | pad rows], and the
/// length of the first two sections.
fn split_partition(
    tuples: &[[u64; 5]],
    keep: &[u64],
    cln: &[u64],
    n: usize,
    pad: &[u64; 6],
) -> (Vec<Vec<u64>>, Vec<Vec<u64>>, usize, usize) {
    let mut filt: Vec<Vec<u64>> = Vec::with_capacity(n);
    let mut clean: Vec<Vec<u64>> = vec![];
    let mut resid: Vec<Vec<u64>> = vec![];

    for i in 0..n {
        if keep[i] == 1 {
            let mut row = tuples[i].to_vec();
            row.push(cln[i]);
            filt.push(row.clone());
            if cln[i] == 1 {
                clean.push(row);
            } else {
                resid.push(row);
            }
        } else {
            filt.push(pad.to_vec());
        }
    }

    let (n_cln, n_res) = (clean.len(), resid.len());
    let mut part = clean;
    part.extend(resid);
    while part.len() < n {
        part.push(pad.to_vec());
    }
    (filt, part, n_cln, n_res)
}

/// Assigns one bag's occurrence-id column, its sorted view, and the strict
/// increase over the `m` rows whose predicate holds.
///
/// `oid[i]` is `pack2(edge ids)` on the rows that pass the predicate and PAD on
/// the rest, so sorting by value puts the `m` real ids in rows `0..m` and PAD
/// afterwards. One PAD row past the end of the bag gives the last comparison a
/// `Rotation::next()` to read even when every row passes.
#[allow(clippy::too_many_arguments)]
fn assign_occurrence_ids<F: Field + Ord>(
    region: &mut Region<'_, F>,
    oid_col: Column<Advice>,
    sid_col: Column<Advice>,
    perm: &PermAnyConfig,
    q_sort: Selector,
    q_pad: Selector,
    lt: &LtConfig<F, NUM_BYTES>,
    oid: &[u64],
    m: usize,
) -> Result<(), Error> {
    let n = oid.len();
    debug_assert!(m <= n);

    let mut sorted = oid.to_vec();
    sorted.sort_unstable();
    debug_assert_eq!(
        oid.iter().filter(|&&x| x != PAD_U64).count(),
        m,
        "the number of rows passing the predicate must be n_cln + n_res"
    );

    let lt_chip = LtChip::<F, NUM_BYTES>::construct(*lt);
    for i in 0..n {
        region.assign_advice(|| "occ id", oid_col, i, || Value::known(F::from(oid[i])))?;
        region.assign_advice(
            || "occ id sorted",
            sid_col,
            i,
            || Value::known(F::from(sorted[i])),
        )?;
        perm.q_perm1.enable(region, i)?;
        perm.q_perm2.enable(region, i)?;

        let cur = sorted[i];
        let next = if i + 1 < n { sorted[i + 1] } else { PAD_U64 };
        lt_chip.assign(
            region,
            i,
            Value::known(F::from(cur)),
            Value::known(F::from(next)),
        )?;
    }
    region.assign_advice(
        || "occ id sentinel",
        sid_col,
        n,
        || Value::known(F::from(PAD_U64)),
    )?;

    for i in 0..m {
        q_sort.enable(region, i)?;
    }
    for i in m..=n {
        q_pad.enable(region, i)?;
    }
    Ok(())
}

/// ------------------------------
/// Host-side witness derivation, shared by this circuit and its multi-lane
/// DP variant `g_sql4_obj_dp`.  Pure function of the edge list: it knows
/// nothing about padding, degrees or lanes.
/// ------------------------------
pub(crate) struct Gq4Derived {
    /// InByDst input rows (key=dst, val=src, eid); row 0 is the (0,0,0) dummy.
    pub in_rows: Vec<(u64, u64, u64)>,
    /// OutBySrc input rows (key=src, val=dst, eid); row 0 is the (0,0,0) dummy.
    pub out_rows: Vec<(u64, u64, u64)>,
    /// The ONE materialized bag, W = {(x,y,z) : x->y and y->z are edges, x<y},
    /// as (x,y,z,i,j,e_in,e_out): `i` indexes x->y inside `in_by_dst`'s group
    /// for key y, `j` indexes y->z inside `out_by_src`'s group for key y.
    ///
    /// The separator (A,C) of the 4-cycle splits it into the path A->B->C and
    /// the path C->D->A, and those two paths are THE SAME RELATION read with
    /// different column roles: Bag1 is W read as (A,B,C) and Bag2 is W read as
    /// (C,D,A). The two loops this replaced were literally the same computation
    /// over the same two maps with the same filter, so `t34` was a
    /// byte-for-byte second copy of `t12` (and relied, unstated, on two
    /// successive `in_groups.iter()` passes yielding the same order). Role 1
    /// adds y<z on top of W's own x<y and role 2 adds nothing, which together
    /// is exactly A<B<C<D.
    pub w: Vec<(u64, u64, u64, u64, u64, u64, u64)>,
    /// Message table over the role-2 reading: msg_key = pack2(A,C) = pack2(z,x)
    /// -> COUNT(W rows with that (z,x)). The key is REVERSED relative to the
    /// role-1 probe key pack2(x,z), which is the whole of what makes one
    /// relation serve as both bags.
    pub msg_map: BTreeMap<u64, u64>,
    /// Sorted, deduped msg keys plus the 0 and PAD sentinels, for gap witnesses.
    pub keys: Vec<u64>,
}

pub(crate) fn gq4_derive(edges: &[Edge]) -> Gq4Derived {
    // -------------------
    // Build (eid,src,dst) with dummy row0
    // -------------------
    let n_base = edges.len() + 1;
    let mut base: Vec<(u64, u64, u64)> = Vec::with_capacity(n_base); // (eid,src,dst)
    base.push((0, 0, 0));
    for (i, e) in edges.iter().enumerate() {
        let eid = (i + 1) as u64;
        base.push((eid, (e.src as u64) + SHIFT_ID, (e.dst as u64) + SHIFT_ID));
    }

    // -------------------
    // Build view inputs from base rows
    // -------------------
    let mut in_rows: Vec<(u64, u64, u64)> = Vec::with_capacity(n_base);
    let mut out_rows: Vec<(u64, u64, u64)> = Vec::with_capacity(n_base);
    for (eid, src, dst) in base.iter().copied() {
        in_rows.push((dst, src, eid)); // key=dst, val=src
        out_rows.push((src, dst, eid)); // key=src, val=dst
    }

    // -------------------
    // Host-side grouping to compute indices (must match sorting by (key,eid))
    // -------------------
    let mut in_groups: HashMap<u64, Vec<(u64 /*eid*/, u64 /*src*/)>> = HashMap::new();
    let mut out_groups: HashMap<u64, Vec<(u64 /*eid*/, u64 /*dst*/)>> = HashMap::new();
    for (eid, src, dst) in base.iter().copied() {
        if eid == 0 {
            continue;
        }
        in_groups.entry(dst).or_default().push((eid, src));
        out_groups.entry(src).or_default().push((eid, dst));
    }
    for v in in_groups.values_mut() {
        v.sort_by_key(|(eid, _)| *eid);
    }
    for v in out_groups.values_mut() {
        v.sort_by_key(|(eid, _)| *eid);
    }

    // -------------------
    // Materialize the ONE bag: W = r_in(x->y) |x| r_out(y->z) on y, filtered to
    // x<y.  row = (x,y,z,i,j,e_in,e_out)
    //
    // Read as (A,B,C) this is Bag1, the path A->B->C; read as (C,D,A) it is
    // Bag2, the path C->D->A. The two used to be materialized by two separate
    // loops over the same `in_groups`/`out_groups` with the same filter, which
    // produced identical rows at twice the cost.
    // -------------------
    let mut w: Vec<(u64, u64, u64, u64, u64, u64, u64)> = vec![];
    for (&y, incoming) in in_groups.iter() {
        if let Some(outgoing) = out_groups.get(&y) {
            for (i, (e_in, x)) in incoming.iter().enumerate() {
                // EARLY FILTER: x < y, mirroring `g_sql3_obj`.  In the role-1
                // reading it is [A<B], which the "contrib gate" multiplies in,
                // so a row with A >= B contributes exactly zero; in the role-2
                // reading it is [C<D], and the "msg input from W" gate emits
                // (in_key, in_val) = (PAD, 0) when it fails, so such a row
                // contributes nothing to the message map either. Materializing
                // it only pays for capacity. Dropping it here is why GQ4 fits
                // the same domain as GQ3 on the directed graph (wiki:
                // 4,542,805 -> 2,255,867 rows, i.e. k=23 -> k=22).
                //
                // Skip BEFORE the inner loop so `i` keeps its meaning as the
                // index into `incoming`: it is looked up against
                // `in_by_dst.idx`, so it must stay the original enumerate index.
                if *x >= y {
                    continue;
                }
                for (j, (e_out, z)) in outgoing.iter().enumerate() {
                    w.push((*x, y, *z, i as u64, j as u64, *e_in, *e_out));
                }
            }
        }
    }

    // -------------------
    // Message map (host-side), over the ROLE-2 reading of W: a row
    // (x,y,z) = (C,D,A) contributes to msg_key = pack2(A,C) = pack2(z,x), with
    // the key REVERSED relative to the role-1 probe key pack2(x,z).
    // msg_val = COUNT of W rows with C<D, and every real row of W already
    // satisfies x<y, so the filter is a debug_assert rather than a test.
    // -------------------
    let mut msg_map: BTreeMap<u64, u64> = BTreeMap::new();
    for (x, y, z, _i, _j, _e_in, _e_out) in w.iter().copied() {
        debug_assert!(x < y, "W is materialized pre-filtered to x<y");
        let key = pack2(z, x);
        *msg_map.entry(key).or_default() += 1;
    }

    // Sorted key list for gap witnesses (host-side).
    // IMPORTANT: must include 0 and PAD
    let mut keys: Vec<u64> = msg_map.keys().copied().collect();
    keys.push(0);
    keys.push(PAD_U64);
    keys.sort();
    keys.dedup();

    Gq4Derived {
        in_rows,
        out_rows,
        w,
        msg_map,
        keys,
    }
}

/// ------------------------------
/// AggSumByKey: group-by SUM(val) over key (used for counting)
/// Produces:
///  - out table: sorted unique keys with sums (padded to n)
///  - map table: (map_key,map_val,map_key_next) for membership+gap proofs
/// ------------------------------
#[derive(Clone, Debug)]
pub struct AggSumByKeyConfig<F: Field + Ord> {
    pub in_key: Column<Advice>,
    pub in_val: Column<Advice>,

    sorted_key: Column<Advice>,
    sorted_val: Column<Advice>,
    perm_sort: PermAnyConfig,

    q_sort: Selector,
    q_sentinel: Selector,
    lt_key: LtConfig<F, NUM_BYTES>,
    iz_eq_key: IsZeroConfig<F>,

    q_first: Selector,
    q_accu: Selector,
    run_sum: Column<Advice>,
    iz_same_prev: IsZeroConfig<F>,
    iz_same_next: IsZeroConfig<F>,

    q_emit: Selector,
    emit_key: Column<Advice>,
    emit_sum: Column<Advice>,

    out_key: Column<Advice>,
    out_sum: Column<Advice>,
    perm_out: PermAnyConfig,

    q_out_sort: Selector,
    lt_out_key: LtConfig<F, NUM_BYTES>,
    iz_out_eq: IsZeroConfig<F>,

    // map table for membership+gap proofs
    pub map_key: Column<Advice>,
    pub map_val: Column<Advice>,
    pub map_key_next: Column<Advice>,

    // IMPORTANT: used on RHS in lookup_any => must be complex_selector()
    pub q_map_tbl: Selector,

    q_map_first: Selector,
    q_map_link: Selector,
    q_map_shift: Selector,
    q_map_last: Selector,
}

#[derive(Clone, Debug)]
pub struct AggSumByKeyChip<F: Field + Ord> {
    cfg: AggSumByKeyConfig<F>,
}
impl<F: Field + Ord> AggSumByKeyChip<F> {
    pub fn construct(cfg: AggSumByKeyConfig<F>) -> Self {
        Self { cfg }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> AggSumByKeyConfig<F> {
        let u8_key = meta.fixed_column();
        let u8_out_key = meta.fixed_column();
        Self::configure_with_u8(meta, u8_key, u8_out_key)
    }

    /// Same replica as [`AggSumByKeyChip::configure`], but with the two u8
    /// range columns supplied by the caller.
    ///
    /// `g_sql4_obj_dp` builds one aggregator replica per Bag2 lane.  `load`
    /// writes 256 fixed rows through a region of its own, so a fresh u8 column
    /// per lane would make the range-table preamble grow with the lane count.
    /// Sharing one column keeps it constant.  It does not change the number of
    /// lookup arguments, which are per diff-byte column.
    pub fn configure_with_u8(
        meta: &mut ConstraintSystem<F>,
        u8_key: Column<Fixed>,
        u8_out_key: Column<Fixed>,
    ) -> AggSumByKeyConfig<F> {
        let in_key = meta.advice_column();
        let in_val = meta.advice_column();
        meta.enable_equality(in_key);
        meta.enable_equality(in_val);

        let sorted_key = meta.advice_column();
        let sorted_val = meta.advice_column();
        meta.enable_equality(sorted_key);
        meta.enable_equality(sorted_val);

        let q1 = meta.complex_selector();
        let q2 = meta.complex_selector();
        let perm_sort = PermAnyChip::configure(
            meta,
            q1,
            q2,
            vec![in_key, in_val],
            vec![sorted_key, sorted_val],
        );

        // sorted_key nondecreasing
        let q_sort = meta.selector();
        let lt_key = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            u8_key,
            |m| m.query_selector(q_sort),
            |m| m.query_advice(sorted_key, Rotation::cur()),
            |m| m.query_advice(sorted_key, Rotation::next()),
        );
        let aux_eq = meta.advice_column();
        let iz_eq_key = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort),
            |m| {
                m.query_advice(sorted_key, Rotation::next())
                    - m.query_advice(sorted_key, Rotation::cur())
            },
            aux_eq,
        );
        meta.create_gate("agg sorted key nondecreasing", |m| {
            let q = m.query_selector(q_sort);
            let le = lt_key.is_lt(m, None) + iz_eq_key.expr();
            vec![q * (le - Expression::Constant(F::ONE))]
        });

        // The group-boundary detector `iz_same_next` below reads `sorted_key` at
        // row n through Rotation::next() on the last real row, and the
        // nondecreasing gate above does not pin that cell: row n is outside the
        // `perm_sort` shuffle (its two selectors are enabled only on 0..n-1) and
        // no other expression reads it. A prover could set it equal to the
        // largest real key; `iz_same_next` would then report "same group" on the
        // last real row, the emit gate would write (PAD, 0) instead of that
        // group's key and sum, and the highest-key group would vanish from the
        // map table. Every probing row carrying that key could then certify its
        // ABSENCE with a gap witness that genuinely holds in the forged table
        // and take msg_val = 0, so the COUNT would silently undercount.
        //
        // Pinning the sentinel to PAD closes it, and is deliberately NOT a
        // strict-increase requirement on the last comparison: the aggregator's
        // input includes the PAD-keyed rows of the non-kept tuples, so the
        // sorted view legitimately ends at PAD and that group must not be
        // emitted. Same reasoning and same shape as
        // `cp: sorted view sentinel is PAD` in `crate::circuits::card_preserve`.
        // One selector and one degree-1 gate: no advice column, no Lt chip.
        //
        // `sorted_val` at row n stays free advice on purpose: nothing reads it
        // (`q_first` is row 0 only and `q_accu` is 1..n-1), so it cannot move
        // any constrained value.
        let q_sentinel = meta.selector();
        meta.create_gate("agg sorted view sentinel is PAD", |m| {
            let q = m.query_selector(q_sentinel);
            vec![
                q * (m.query_advice(sorted_key, Rotation::cur())
                    - Expression::Constant(F::from(PAD_U64))),
            ]
        });

        // run sum
        let run_sum = meta.advice_column();
        meta.enable_equality(run_sum);
        let q_first = meta.selector();
        let q_accu = meta.selector();
        let q_emit = meta.selector();

        let aux_same_prev = meta.advice_column();
        let iz_same_prev = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_accu),
            |m| {
                m.query_advice(sorted_key, Rotation::cur())
                    - m.query_advice(sorted_key, Rotation::prev())
            },
            aux_same_prev,
        );
        let aux_same_next = meta.advice_column();
        let iz_same_next = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_emit),
            |m| {
                m.query_advice(sorted_key, Rotation::next())
                    - m.query_advice(sorted_key, Rotation::cur())
            },
            aux_same_next,
        );

        meta.create_gate("agg run_first", |m| {
            let q = m.query_selector(q_first);
            vec![
                q * (m.query_advice(run_sum, Rotation::cur())
                    - m.query_advice(sorted_val, Rotation::cur())),
            ]
        });
        meta.create_gate("agg run_accu", |m| {
            let q = m.query_selector(q_accu);
            let same = iz_same_prev.expr();
            let rs_cur = m.query_advice(run_sum, Rotation::cur());
            let rs_prev = m.query_advice(run_sum, Rotation::prev());
            let v = m.query_advice(sorted_val, Rotation::cur());
            vec![q * (rs_cur - (same * rs_prev + v))]
        });

        // emit last-of-group
        let emit_key = meta.advice_column();
        let emit_sum = meta.advice_column();
        meta.enable_equality(emit_key);
        meta.enable_equality(emit_sum);

        meta.create_gate("agg emit last-of-group", |m| {
            let q = m.query_selector(q_emit);
            let one = Expression::Constant(F::ONE);
            let is_last = one.clone() - iz_same_next.expr();
            let not_last = one - is_last.clone();

            let k = m.query_advice(sorted_key, Rotation::cur());
            let s = m.query_advice(run_sum, Rotation::cur());

            let outk = m.query_advice(emit_key, Rotation::cur());
            let outs = m.query_advice(emit_sum, Rotation::cur());

            vec![
                q.clone()
                    * (outk
                        - (is_last.clone() * k
                            + not_last.clone() * Expression::Constant(F::from(PAD_U64)))),
                q * (outs - (is_last * s + not_last * Expression::Constant(F::ZERO))),
            ]
        });

        // compact emit -> out (permute)
        let out_key = meta.advice_column();
        let out_sum = meta.advice_column();
        meta.enable_equality(out_key);
        meta.enable_equality(out_sum);

        let p1 = meta.complex_selector();
        let p2 = meta.complex_selector();
        let perm_out = PermAnyChip::configure(
            meta,
            p1,
            p2,
            vec![emit_key, emit_sum],
            vec![out_key, out_sum],
        );

        // out nondecreasing
        let q_out_sort = meta.selector();
        let lt_out_key = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            u8_out_key,
            |m| m.query_selector(q_out_sort),
            |m| m.query_advice(out_key, Rotation::cur()),
            |m| m.query_advice(out_key, Rotation::next()),
        );
        let aux_oeq = meta.advice_column();
        let iz_out_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_out_sort),
            |m| {
                m.query_advice(out_key, Rotation::next()) - m.query_advice(out_key, Rotation::cur())
            },
            aux_oeq,
        );
        meta.create_gate("agg out nondecreasing", |m| {
            let q = m.query_selector(q_out_sort);
            let le = lt_out_key.is_lt(m, None) + iz_out_eq.expr();
            vec![q * (le - Expression::Constant(F::ONE))]
        });

        // map table
        let map_key = meta.advice_column();
        let map_val = meta.advice_column();
        let map_key_next = meta.advice_column();
        meta.enable_equality(map_key);
        meta.enable_equality(map_val);
        meta.enable_equality(map_key_next);

        // RHS gating selector for lookups (must be complex)
        let q_map_tbl = meta.complex_selector();

        let q_map_first = meta.selector();
        let q_map_link = meta.selector();
        let q_map_shift = meta.selector();
        let q_map_last = meta.selector();

        meta.create_gate("map row0 key/val = 0", |m| {
            let q = m.query_selector(q_map_first);
            vec![
                q.clone() * m.query_advice(map_key, Rotation::cur()),
                q * m.query_advice(map_val, Rotation::cur()),
            ]
        });

        // at row i: map_key[i+1] == out_key[i] and map_val[i+1] == out_sum[i]
        meta.create_gate("map link out", |m| {
            let q = m.query_selector(q_map_link);
            vec![
                q.clone()
                    * (m.query_advice(map_key, Rotation::next())
                        - m.query_advice(out_key, Rotation::cur())),
                q * (m.query_advice(map_val, Rotation::next())
                    - m.query_advice(out_sum, Rotation::cur())),
            ]
        });

        // at row r: map_key_next[r] == map_key[r+1]
        meta.create_gate("map key_next = next(map_key)", |m| {
            let q = m.query_selector(q_map_shift);
            vec![
                q * (m.query_advice(map_key_next, Rotation::cur())
                    - m.query_advice(map_key, Rotation::next())),
            ]
        });

        meta.create_gate("map last key_next=PAD", |m| {
            let q = m.query_selector(q_map_last);
            vec![
                q * (m.query_advice(map_key_next, Rotation::cur())
                    - Expression::Constant(F::from(PAD_U64))),
            ]
        });

        AggSumByKeyConfig {
            in_key,
            in_val,
            sorted_key,
            sorted_val,
            perm_sort,
            q_sort,
            q_sentinel,
            lt_key,
            iz_eq_key,
            q_first,
            q_accu,
            run_sum,
            iz_same_prev,
            iz_same_next,
            q_emit,
            emit_key,
            emit_sum,
            out_key,
            out_sum,
            perm_out,
            q_out_sort,
            lt_out_key,
            iz_out_eq,
            map_key,
            map_val,
            map_key_next,
            q_map_tbl,
            q_map_first,
            q_map_link,
            q_map_shift,
            q_map_last,
        }
    }

    /// Assign full agg + map. Input rows length n (fixed). Adds one sentinel row at n for next-comparisons.
    /// Returns the emitted (key,sum) pairs (without padding).
    pub fn assign(
        &self,
        region: &mut Region<'_, F>,
        n: usize,
        rows: &[(u64, u64)],
    ) -> Result<Vec<(u64, u64)>, Error> {
        let cfg = &self.cfg;

        let lt_key_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_key.clone());
        let lt_out_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_out_key.clone());
        let iz_eq_chip = IsZeroChip::construct(cfg.iz_eq_key.clone());
        let iz_same_prev_chip = IsZeroChip::construct(cfg.iz_same_prev.clone());
        let iz_same_next_chip = IsZeroChip::construct(cfg.iz_same_next.clone());
        let iz_out_eq_chip = IsZeroChip::construct(cfg.iz_out_eq.clone());

        // perm selectors (in->sorted)
        for i in 0..n {
            cfg.perm_sort.q_perm1.enable(region, i)?;
            cfg.perm_sort.q_perm2.enable(region, i)?;
        }

        // sorted witness
        let mut sorted = rows.to_vec();
        sorted.sort_by_key(|(k, _)| *k);
        let mut sorted_ext = sorted.clone();
        sorted_ext.push((PAD_U64, 0)); // sentinel for i+1

        for i in 0..n {
            region.assign_advice(
                || "in_key",
                cfg.in_key,
                i,
                || Value::known(F::from(rows[i].0)),
            )?;
            region.assign_advice(
                || "in_val",
                cfg.in_val,
                i,
                || Value::known(F::from(rows[i].1)),
            )?;
            region.assign_advice(
                || "sorted_key",
                cfg.sorted_key,
                i,
                || Value::known(F::from(sorted_ext[i].0)),
            )?;
            region.assign_advice(
                || "sorted_val",
                cfg.sorted_val,
                i,
                || Value::known(F::from(sorted_ext[i].1)),
            )?;
        }
        // sentinel row n
        region.assign_advice(
            || "sorted_key_s",
            cfg.sorted_key,
            n,
            || Value::known(F::from(sorted_ext[n].0)),
        )?;
        region.assign_advice(
            || "sorted_val_s",
            cfg.sorted_val,
            n,
            || Value::known(F::from(sorted_ext[n].1)),
        )?;
        // pin it, so the last real group is recognised as a group end
        cfg.q_sentinel.enable(region, n)?;

        for i in 0..n {
            cfg.q_sort.enable(region, i)?;
            iz_eq_chip.assign(
                region,
                i,
                Value::known(F::from(sorted_ext[i + 1].0) - F::from(sorted_ext[i].0)),
            )?;
            lt_key_chip.assign(
                region,
                i,
                Value::known(F::from(sorted_ext[i].0)),
                Value::known(F::from(sorted_ext[i + 1].0)),
            )?;
        }

        if n > 0 {
            cfg.q_first.enable(region, 0)?;
        }
        for i in 1..n {
            cfg.q_accu.enable(region, i)?;
        }
        for i in 0..n {
            cfg.q_emit.enable(region, i)?;
        }

        // run sums
        let mut run_sum_u64 = vec![0u64; n];
        let mut acc: u128 = 0;
        for i in 0..n {
            let (k, v) = sorted[i];
            if i == 0 || sorted[i - 1].0 != k {
                acc = v as u128;
            } else {
                acc += v as u128;
            }
            run_sum_u64[i] = acc as u64;
            region.assign_advice(
                || "run_sum",
                cfg.run_sum,
                i,
                || Value::known(F::from(run_sum_u64[i])),
            )?;

            if i > 0 {
                iz_same_prev_chip.assign(
                    region,
                    i,
                    Value::known(F::from(sorted[i].0) - F::from(sorted[i - 1].0)),
                )?;
            }
            iz_same_next_chip.assign(
                region,
                i,
                Value::known(F::from(sorted_ext[i + 1].0) - F::from(sorted_ext[i].0)),
            )?;
        }

        // emit last-of-group
        let mut emitted: Vec<(u64, u64)> = vec![];
        for i in 0..n {
            let is_last = sorted_ext[i].0 != sorted_ext[i + 1].0;
            let ek = if is_last { sorted_ext[i].0 } else { PAD_U64 };
            let es = if is_last { run_sum_u64[i] } else { 0 };
            region.assign_advice(|| "emit_key", cfg.emit_key, i, || Value::known(F::from(ek)))?;
            region.assign_advice(|| "emit_sum", cfg.emit_sum, i, || Value::known(F::from(es)))?;
            if is_last && ek != PAD_U64 {
                emitted.push((ek, es));
            }
        }

        // out table = emitted padded to n
        let mut out = emitted.clone();
        while out.len() < n {
            out.push((PAD_U64, 0));
        }

        for i in 0..n {
            cfg.perm_out.q_perm1.enable(region, i)?;
            cfg.perm_out.q_perm2.enable(region, i)?;
        }
        for i in 0..n {
            region.assign_advice(
                || "out_key",
                cfg.out_key,
                i,
                || Value::known(F::from(out[i].0)),
            )?;
            region.assign_advice(
                || "out_sum",
                cfg.out_sum,
                i,
                || Value::known(F::from(out[i].1)),
            )?;
        }
        // The `out` table's own sentinel at row n needs no pin, unlike the sorted
        // view's above: `q_out_sort` runs over 0..n-1, so the comparisons at
        // i = 0..n-2 already order the n real rows among themselves, and row n
        // enters only the last comparison, where a free value merely makes that
        // one comparison trivial. Nothing else reads it: the map table copies
        // out[0..n-1], so the gap witnesses read the ordered part only.
        region.assign_advice(
            || "out_key_s",
            cfg.out_key,
            n,
            || Value::known(F::from(PAD_U64)),
        )?;
        region.assign_advice(|| "out_sum_s", cfg.out_sum, n, || Value::known(F::ZERO))?;

        // out sort checks
        let mut out_ext = out.clone();
        out_ext.push((PAD_U64, 0));
        for i in 0..n {
            cfg.q_out_sort.enable(region, i)?;
            iz_out_eq_chip.assign(
                region,
                i,
                Value::known(F::from(out_ext[i + 1].0) - F::from(out_ext[i].0)),
            )?;
            lt_out_chip.assign(
                region,
                i,
                Value::known(F::from(out_ext[i].0)),
                Value::known(F::from(out_ext[i + 1].0)),
            )?;
        }

        // map table rows: 0..n (total n+1)
        // row0 dummy: key=0,val=0, key_next = key(row1)
        cfg.q_map_tbl.enable(region, 0)?;
        cfg.q_map_first.enable(region, 0)?;
        region.assign_advice(|| "map_key0", cfg.map_key, 0, || Value::known(F::ZERO))?;
        region.assign_advice(|| "map_val0", cfg.map_val, 0, || Value::known(F::ZERO))?;
        let first_key = out.get(0).map(|x| x.0).unwrap_or(PAD_U64);
        region.assign_advice(
            || "map_kn0",
            cfg.map_key_next,
            0,
            || Value::known(F::from(first_key)),
        )?;
        cfg.q_map_shift.enable(region, 0)?;

        // rows 1..n copy out[0..n-1]
        for i in 0..n {
            cfg.q_map_tbl.enable(region, i + 1)?;
            cfg.q_map_link.enable(region, i)?;
            region.assign_advice(
                || "map_key",
                cfg.map_key,
                i + 1,
                || Value::known(F::from(out[i].0)),
            )?;
            region.assign_advice(
                || "map_val",
                cfg.map_val,
                i + 1,
                || Value::known(F::from(out[i].1)),
            )?;
        }

        // key_next chain for rows 1..n-1  (and row0 already)
        for r in 1..n {
            cfg.q_map_shift.enable(region, r)?;
            let nextk = out.get(r).map(|x| x.0).unwrap_or(PAD_U64);
            region.assign_advice(
                || "map_kn",
                cfg.map_key_next,
                r,
                || Value::known(F::from(nextk)),
            )?;
        }

        // last row r=n: key_next = PAD
        cfg.q_map_tbl.enable(region, n)?;
        cfg.q_map_last.enable(region, n)?;
        region.assign_advice(
            || "map_kn_last",
            cfg.map_key_next,
            n,
            || Value::known(F::from(PAD_U64)),
        )?;

        Ok(emitted)
    }
}

/// ------------------------------
/// Indexed view: (key,val,eid) sorted by (key,eid), plus idx within key-group.
/// Used to prove join rows by lookup: (key, idx) -> (val, eid)
/// ------------------------------
#[derive(Clone, Debug)]
pub struct IndexedViewConfig<F: Field + Ord> {
    in_key: Column<Advice>,
    in_val: Column<Advice>,
    in_eid: Column<Advice>,

    pub sorted_key: Column<Advice>,
    pub sorted_val: Column<Advice>,
    pub sorted_eid: Column<Advice>,
    perm: PermAnyConfig,

    q_sort: Selector,
    lt_key: LtConfig<F, NUM_BYTES>,
    iz_eq_key: IsZeroConfig<F>,

    pub idx: Column<Advice>,
    q_idx0: Selector,
    q_idx: Selector,
    iz_same_prev: IsZeroConfig<F>,
}

#[derive(Clone, Debug)]
pub struct IndexedViewChip<F: Field + Ord> {
    cfg: IndexedViewConfig<F>,
}
impl<F: Field + Ord> IndexedViewChip<F> {
    pub fn construct(cfg: IndexedViewConfig<F>) -> Self {
        Self { cfg }
    }

    /// Load the u8 range table backing the sortedness comparison.  Needed by
    /// external users of the chip (`pone_baseline`), which cannot reach the
    /// private `lt_key` field to load it themselves.
    pub fn load(&self, layouter: &mut impl Layouter<F>) -> Result<(), Error> {
        LtChip::<F, NUM_BYTES>::construct(self.cfg.lt_key.clone()).load(layouter)
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> IndexedViewConfig<F> {
        let in_key = meta.advice_column();
        let in_val = meta.advice_column();
        let in_eid = meta.advice_column();
        for c in [in_key, in_val, in_eid] {
            meta.enable_equality(c);
        }

        let sorted_key = meta.advice_column();
        let sorted_val = meta.advice_column();
        let sorted_eid = meta.advice_column();
        for c in [sorted_key, sorted_val, sorted_eid] {
            meta.enable_equality(c);
        }

        let q1 = meta.complex_selector();
        let q2 = meta.complex_selector();
        let perm = PermAnyChip::configure(
            meta,
            q1,
            q2,
            vec![in_key, in_val, in_eid],
            vec![sorted_key, sorted_val, sorted_eid],
        );

        let q_sort = meta.selector();
        let lt_key = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort),
            |m| m.query_advice(sorted_key, Rotation::cur()),
            |m| m.query_advice(sorted_key, Rotation::next()),
        );
        let aux = meta.advice_column();
        let iz_eq_key = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort),
            |m| {
                m.query_advice(sorted_key, Rotation::next())
                    - m.query_advice(sorted_key, Rotation::cur())
            },
            aux,
        );
        meta.create_gate("view key nondecreasing", |m| {
            let q = m.query_selector(q_sort);
            let le = lt_key.is_lt(m, None) + iz_eq_key.expr();
            vec![q * (le - Expression::Constant(F::ONE))]
        });

        let idx = meta.advice_column();
        meta.enable_equality(idx);
        let q_idx0 = meta.selector();
        let q_idx = meta.selector();

        let aux_same = meta.advice_column();
        let iz_same_prev = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_idx),
            |m| {
                m.query_advice(sorted_key, Rotation::cur())
                    - m.query_advice(sorted_key, Rotation::prev())
            },
            aux_same,
        );

        meta.create_gate("idx0=0", |m| {
            let q = m.query_selector(q_idx0);
            vec![q * m.query_advice(idx, Rotation::cur())]
        });
        meta.create_gate("idx recurrence", |m| {
            let q = m.query_selector(q_idx);
            let same = iz_same_prev.expr();
            let idx_cur = m.query_advice(idx, Rotation::cur());
            let idx_prev = m.query_advice(idx, Rotation::prev());
            vec![q * (idx_cur - (same * (idx_prev + Expression::Constant(F::ONE))))]
        });

        IndexedViewConfig {
            in_key,
            in_val,
            in_eid,
            sorted_key,
            sorted_val,
            sorted_eid,
            perm,
            q_sort,
            lt_key,
            iz_eq_key,
            idx,
            q_idx0,
            q_idx,
            iz_same_prev,
        }
    }

    /// Assign view for n input tuples (key,val,eid). Adds a sentinel row at n for next-comparisons.
    pub fn assign(
        &self,
        region: &mut Region<'_, F>,
        n: usize,
        in_rows: &[(u64, u64, u64)],
    ) -> Result<(), Error> {
        let cfg = &self.cfg;

        let lt_key_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_key.clone());
        let iz_eq_chip = IsZeroChip::construct(cfg.iz_eq_key.clone());
        let iz_same_prev_chip = IsZeroChip::construct(cfg.iz_same_prev.clone());

        for i in 0..n {
            cfg.perm.q_perm1.enable(region, i)?;
            cfg.perm.q_perm2.enable(region, i)?;
        }

        for i in 0..n {
            region.assign_advice(
                || "in_key",
                cfg.in_key,
                i,
                || Value::known(F::from(in_rows[i].0)),
            )?;
            region.assign_advice(
                || "in_val",
                cfg.in_val,
                i,
                || Value::known(F::from(in_rows[i].1)),
            )?;
            region.assign_advice(
                || "in_eid",
                cfg.in_eid,
                i,
                || Value::known(F::from(in_rows[i].2)),
            )?;
        }

        let mut sorted = in_rows.to_vec();
        sorted.sort_by_key(|(k, _, eid)| (*k, *eid));
        let mut sorted_ext = sorted.clone();
        sorted_ext.push((PAD_U64, 0, 0));

        for i in 0..n {
            region.assign_advice(
                || "sorted_key",
                cfg.sorted_key,
                i,
                || Value::known(F::from(sorted_ext[i].0)),
            )?;
            region.assign_advice(
                || "sorted_val",
                cfg.sorted_val,
                i,
                || Value::known(F::from(sorted_ext[i].1)),
            )?;
            region.assign_advice(
                || "sorted_eid",
                cfg.sorted_eid,
                i,
                || Value::known(F::from(sorted_ext[i].2)),
            )?;
        }
        region.assign_advice(
            || "sorted_key_s",
            cfg.sorted_key,
            n,
            || Value::known(F::from(sorted_ext[n].0)),
        )?;
        region.assign_advice(
            || "sorted_val_s",
            cfg.sorted_val,
            n,
            || Value::known(F::from(sorted_ext[n].1)),
        )?;
        region.assign_advice(
            || "sorted_eid_s",
            cfg.sorted_eid,
            n,
            || Value::known(F::from(sorted_ext[n].2)),
        )?;

        for i in 0..n {
            cfg.q_sort.enable(region, i)?;
            iz_eq_chip.assign(
                region,
                i,
                Value::known(F::from(sorted_ext[i + 1].0) - F::from(sorted_ext[i].0)),
            )?;
            lt_key_chip.assign(
                region,
                i,
                Value::known(F::from(sorted_ext[i].0)),
                Value::known(F::from(sorted_ext[i + 1].0)),
            )?;
        }

        // idx within key-group
        let mut idx_u64 = vec![0u64; n];
        for i in 0..n {
            if i == 0 {
                idx_u64[i] = 0;
            } else {
                idx_u64[i] = if sorted[i].0 == sorted[i - 1].0 {
                    idx_u64[i - 1] + 1
                } else {
                    0
                };
            }
            region.assign_advice(|| "idx", cfg.idx, i, || Value::known(F::from(idx_u64[i])))?;
        }
        region.assign_advice(|| "idx_s", cfg.idx, n, || Value::known(F::ZERO))?;

        if n > 0 {
            cfg.q_idx0.enable(region, 0)?;
        }
        for i in 1..n {
            cfg.q_idx.enable(region, i)?;
            iz_same_prev_chip.assign(
                region,
                i,
                Value::known(F::from(sorted[i].0) - F::from(sorted[i - 1].0)),
            )?;
        }

        Ok(())
    }
}

/// ------------------------------
/// MapLookup: membership+gap lookup into AggSumByKey map table
/// - If in_set=1: (key,val) must appear in map tables
/// - If in_set=0: prove a gap (low < key < high) where (low,high) is consecutive in map_key/key_next
/// ------------------------------
#[derive(Clone, Debug)]
pub struct MapLookupConfig<F: Field + Ord> {
    pub q_flag: Selector,
    pub q_complex: Selector,

    pub key: Column<Advice>,
    pub in_set: Column<Advice>,
    pub low: Column<Advice>,
    pub high: Column<Advice>,
    pub val: Column<Advice>,

    pub lt_low: LtConfig<F, NUM_BYTES>,
    pub lt_high: LtConfig<F, NUM_BYTES>,

    map_key: Column<Advice>,
    map_val: Column<Advice>,
    map_key_next: Column<Advice>,
    q_tbl: Selector, // complex selector from agg table
}

#[derive(Clone, Debug)]
pub struct MapLookupChip<F: Field + Ord> {
    cfg: MapLookupConfig<F>,
}
impl<F: Field + Ord> MapLookupChip<F> {
    pub fn construct(cfg: MapLookupConfig<F>) -> Self {
        Self { cfg }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        map_key: Column<Advice>,
        map_val: Column<Advice>,
        map_key_next: Column<Advice>,
        q_tbl: Selector, // must be complex
    ) -> MapLookupConfig<F> {
        let q_flag = meta.selector();
        let q_complex = meta.complex_selector();
        let u8_low = meta.fixed_column();
        let u8_high = meta.fixed_column();
        Self::configure_with(
            meta,
            map_key,
            map_val,
            map_key_next,
            q_tbl,
            q_flag,
            q_complex,
            u8_low,
            u8_high,
            true,
        )
    }

    /// Same replica as [`MapLookupChip::configure`], but with the two probe
    /// selectors and the two u8 range columns supplied by the caller.
    ///
    /// `g_sql4_obj_dp` builds one MapLookup replica per (Bag1 lane, Bag2 lane)
    /// pair.  All of them are active on the same rows, so they share one
    /// `q_flag` / `q_complex` pair, and they share one u8 column so the number
    /// of 256-row `load` regions does not grow with the lane counts.
    ///
    /// `probe_equality` puts the five probe columns into the permutation
    /// argument.  halo2 charges for every column in the permutation whether or
    /// not a copy constraint ever touches it, and NOTHING copies a probe cell,
    /// so a replicated caller passes `false`: otherwise the permutation cost
    /// would grow as `5 * c1 * c2` for nothing.  [`MapLookupChip::configure`]
    /// passes `true` to keep the single-group circuits byte-identical.
    #[allow(clippy::too_many_arguments)]
    pub fn configure_with(
        meta: &mut ConstraintSystem<F>,
        map_key: Column<Advice>,
        map_val: Column<Advice>,
        map_key_next: Column<Advice>,
        q_tbl: Selector, // must be complex
        q_flag: Selector,
        q_complex: Selector, // must be complex
        u8_low: Column<Fixed>,
        u8_high: Column<Fixed>,
        probe_equality: bool,
    ) -> MapLookupConfig<F> {
        let key = meta.advice_column();
        let in_set = meta.advice_column();
        let low = meta.advice_column();
        let high = meta.advice_column();
        let val = meta.advice_column();
        if probe_equality {
            for c in [key, in_set, low, high, val] {
                meta.enable_equality(c);
            }
        }

        let lt_low = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            u8_low,
            |m| {
                let q = m.query_selector(q_flag);
                let inside = m.query_advice(in_set, Rotation::cur());
                q * (Expression::Constant(F::ONE) - inside)
            },
            |m| m.query_advice(low, Rotation::cur()),
            |m| m.query_advice(key, Rotation::cur()),
        );
        let lt_high = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            u8_high,
            |m| {
                let q = m.query_selector(q_flag);
                let inside = m.query_advice(in_set, Rotation::cur());
                q * (Expression::Constant(F::ONE) - inside)
            },
            |m| m.query_advice(key, Rotation::cur()),
            |m| m.query_advice(high, Rotation::cur()),
        );

        // membership: (key,val) must appear in table when in_set=1
        meta.lookup_any("msg member", |m| {
            let q = m.query_selector(q_complex);
            let inside = m.query_advice(in_set, Rotation::cur());
            let gate = q.clone() * inside;

            let tk = m.query_selector(q_tbl) * m.query_advice(map_key, Rotation::cur());
            let tv = m.query_selector(q_tbl) * m.query_advice(map_val, Rotation::cur());

            vec![
                (gate.clone() * m.query_advice(key, Rotation::cur()), tk),
                (gate * m.query_advice(val, Rotation::cur()), tv),
            ]
        });

        // gap: (low,high) must appear as consecutive (map_key, map_key_next) when in_set=0
        meta.lookup_any("msg gap", |m| {
            let q = m.query_selector(q_complex);
            let inside = m.query_advice(in_set, Rotation::cur());
            let gate = q * (Expression::Constant(F::ONE) - inside);

            let tk = m.query_selector(q_tbl) * m.query_advice(map_key, Rotation::cur());
            let tkn = m.query_selector(q_tbl) * m.query_advice(map_key_next, Rotation::cur());

            vec![
                (gate.clone() * m.query_advice(low, Rotation::cur()), tk),
                (gate * m.query_advice(high, Rotation::cur()), tkn),
            ]
        });

        meta.create_gate("msg lookup correctness", |m| {
            let q = m.query_selector(q_flag);
            let inside = m.query_advice(in_set, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let low_ok = lt_low.is_lt(m, None);
            let high_ok = lt_high.is_lt(m, None);

            vec![
                q.clone() * inside.clone() * (one.clone() - inside.clone()),
                q.clone() * (one.clone() - inside.clone()) * (one.clone() - low_ok),
                q.clone() * (one.clone() - inside.clone()) * (one.clone() - high_ok),
                // if missing => val=0
                q * (one - inside) * m.query_advice(val, Rotation::cur()),
            ]
        });

        MapLookupConfig {
            q_flag,
            q_complex,
            key,
            in_set,
            low,
            high,
            val,
            lt_low,
            lt_high,
            map_key,
            map_val,
            map_key_next,
            q_tbl,
        }
    }
}

/// ------------------------------
/// Main circuit config (4-cycle)
/// ------------------------------
#[derive(Clone, Debug)]
pub struct Cycle4OrderedConfig<F: Field + Ord> {
    instance: Column<Instance>,

    // base Edge table: (eid, src, dst) includes dummy row0 = (0,0,0)
    e_eid: Column<Advice>,
    e_src: Column<Advice>,
    e_dst: Column<Advice>,

    // indexed views
    in_by_dst: IndexedViewConfig<F>,
    out_by_src: IndexedViewConfig<F>,

    // Table-side gate of the two bag lookups, enabled on exactly the n_base
    // real rows of both views. See "W r_in from in_by_dst" for why.
    q_view_tbl: Selector,

    // Conservation of the edge relation: both indexed views are the same
    // (src, dst, eid) multiset as the base Edge table.
    perm_edge_out: PermAnyConfig,
    perm_edge_in: PermAnyConfig,

    // The ONE materialized bag W: x->y->z with x<y. Read as (A,B,C) it is the
    // path A->B->C, read as (C,D,A) it is the path C->D->A, and both readings
    // live on the same row of the same eight columns.
    w_x: Column<Advice>,
    w_y: Column<Advice>,
    w_z: Column<Advice>,
    w_i: Column<Advice>,
    w_j: Column<Advice>,
    w_e1: Column<Advice>,
    w_e2: Column<Advice>,
    w_real: Column<Advice>,
    q_w_lookup: Selector,
    q_w_key: Selector,
    q_w_msg_in: Selector,

    // message aggregator over the role-2 reading: msg_key = pack2(A,C) =
    // pack2(z,x), msg_val = count
    agg_msg: AggSumByKeyConfig<F>,

    // message lookup per row, on the role-1 probe key pack2(A,C) = pack2(x,z)
    msg_lookup: MapLookupConfig<F>,

    // the two order checks on the row: [x<y], which is role 1's [A<B] and
    // role 2's [C<D] at once, and [y<z], which is role 1's [B<C]
    q_order: Selector,
    lt_ab: LtConfig<F, NUM_BYTES>,
    lt_bc: LtConfig<F, NUM_BYTES>,

    // contribution and sum
    contrib: Column<Advice>,
    q_contrib: Selector,

    run_sum: Column<Advice>,
    q_sum0: Selector,
    q_sum: Selector,

    out: Column<Advice>,
    q_out: Selector,

    // ---------------- Cardinality Preservation Check ----------------
    // condition (4): |R^c join| == |R join| over the one bag tree edge
    // role 1 -> role 2
    cp_agg_2: CpAggConfig<F, NUM_BYTES>, // child = role 2, keyed by pack2(z,x)
    cp_join_2: CpJoinConfig<F, NUM_BYTES>, // parent = role 1 -> child
    cp_root: CpRootConfig,
    q_cp_mu: Selector, // the two root product gates

    // the role-1 predicate bit, folded into a column so that every gate of the
    // check stays at degree <= 4. Structurally it is keep * [y<z], with
    // keep = w_real * [x<y] the role-2 bit, so role 1's rows are a per-row
    // determined SUBSET of role 2's.
    w_pred: Column<Advice>,
    q_w_pred: Selector,

    // clean indicator per role, and the Conservation Check that binds it. The
    // two indicators genuinely differ: a row can be role-2 clean and role-1
    // dangling, so the two partitions cannot be merged (and one row order could
    // not put both clean sets in a prefix anyway).
    w_cflag1: Column<Advice>,
    w_cflag2: Column<Advice>,

    // ---------------- (1) Conservation Check ----------------
    // R^_i == R^_i^c U+ R^_i^r over the INDEXED bag. W is read in two roles, so
    // its rows are conserved twice, once per role, each against that role's own
    // indicator. Padding rows sit in the residual part, their indicator zero.
    row_idx: RowIndexConfig,
    cons_role1: ConserveConfig,
    cons_role2: ConserveConfig,

    // ---------------- (2) Pairwise Consistency ----------------
    // pi_K(R_1^c) == pi_K(R_2^c) on the one bag tree edge. Every vector below
    // is indexed [role1, role2]; the two keys are the same two cells packed in
    // OPPOSITE orders, each MASKED by that role's selector bit.
    pw_key: Vec<Column<Advice>>, // c * pack2(A, C) on the bag rows
    q_pw_key: Vec<Selector>,     // pins pw_key on every bag row
    // Enabled over the bag's whole capacity: the clean row set the containments
    // range over is carried by the masked key, not by the selector's extent.
    q_pw_in: Vec<Selector>,

    // ---------------- Occurrence distinctness ----------------
    // ONE argument now, over the role-2 keep bit: `w_oid` is the occurrence id
    // on the bag rows, `oid_sorted` its sorted view, and the two selectors pin
    // the strict increase over R^c U R^r and the PAD tail. The role-1 copy is
    // gone because role 1's rows are the role-2 rows that also satisfy [y<z],
    // pinned per row by the `w_pred` gate.
    w_oid: Column<Advice>,
    oid_sorted: Column<Advice>,
    perm_oid: PermAnyConfig,
    q_oid_sort: Selector,
    q_oid_pad: Selector,
    lt_oid: LtConfig<F, NUM_BYTES>,
}

#[derive(Clone, Debug)]
pub struct Cycle4OrderedChip<F: Field + Ord> {
    cfg: Cycle4OrderedConfig<F>,
}
impl<F: Field + Ord> Cycle4OrderedChip<F> {
    pub fn construct(cfg: Cycle4OrderedConfig<F>) -> Self {
        Self { cfg }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> Cycle4OrderedConfig<F> {
        let instance = meta.instance_column();
        meta.enable_equality(instance);

        let e_eid = meta.advice_column();
        let e_src = meta.advice_column();
        let e_dst = meta.advice_column();
        for c in [e_eid, e_src, e_dst] {
            meta.enable_equality(c);
        }

        // views
        let in_by_dst = IndexedViewChip::<F>::configure(meta);
        let out_by_src = IndexedViewChip::<F>::configure(meta);

        // The ONE tuple column group. The separator (A,C) splits GQ4 into the
        // path A->B->C and the path C->D->A, and those are the same relation
        // W = {(x,y,z) : x->y, y->z edges, x<y} read with different column
        // roles, so the circuit materializes W once and uses it twice. Pad rows
        // are all-zero with w_real = 0.
        let w_x = meta.advice_column();
        let w_y = meta.advice_column();
        let w_z = meta.advice_column();
        let w_i = meta.advice_column();
        let w_j = meta.advice_column();
        let w_e1 = meta.advice_column();
        let w_e2 = meta.advice_column();
        let w_real = meta.advice_column();
        for c in [w_x, w_y, w_z, w_i, w_j, w_e1, w_e2, w_real] {
            meta.enable_equality(c);
        }
        let q_w_lookup = meta.complex_selector();
        let q_w_key = meta.selector();
        let q_w_msg_in = meta.selector();

        // Table-side gate of the two bag lookups below.
        //
        // Each `IndexedView` carries a sentinel row at `n_base` for the
        // Rotation::next() of its own sortedness gate. That row is outside the
        // view's `perm` (both its selectors stop at n_base-1), outside `q_idx0`
        // (row 0) and outside `q_idx` (1..n_base-1), so `sorted_val`,
        // `sorted_eid` and `idx` there are free advice, and only `sorted_key` is
        // touched at all (by "view key nondecreasing" at i = n_base-1, which
        // merely bounds it above the largest real key). With an ungated table
        // side that sentinel is a LIVE table row, so a prover can mint an edge
        // that the relation does not contain: set out_by_src's sentinel to
        // (key = D, idx = 0, val = A0, eid = anything) with D above every real
        // src, and a W row (x = C0, y = D, z = A0) then passes
        // "W r_out from out_by_src" for an edge D->A0 that exists nowhere.
        // Swapping it in for a clean row of another clean key keeps both
        // partition section lengths, so the honest vk still verifies.
        //
        // Gating the table side over exactly the n_base real rows removes the
        // sentinel from both tables. On the ungated rows every table
        // expression reads 0, so the tuple (0,0,0,0) is in the table, which is
        // what the padded bag rows (all-zero, real = 0) look up anyway, and it
        // is a real row of both views regardless: base row 0 is the (0,0,0)
        // dummy edge with idx 0. The table side goes from degree 1 to degree 2,
        // so these two lookups go from 2+2+1 = 5 to 2+2+2 = 6, still under the
        // 2+3+2 = 7 that the Cardinality Preservation Check's own membership
        // lookup already costs: cs.degree() does not move.
        let q_view_tbl = meta.complex_selector();

        // The TWO membership lookups of W. There used to be four, two per bag,
        // but in W coordinates the Bag2 pair is literally the Bag1 pair: r3 is
        // (y, i) -> (x, e1) against in_by_dst and r4 is (y, j) -> (z, e2)
        // against out_by_src, the same cells of the same row against the same
        // two views. Reading the row as (C,D,A) instead of (A,B,C) renames the
        // columns and changes nothing that a lookup can see.
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

        // -------- Conservation of the edge relation (condition (7) for R) -----
        // `e_eid`/`e_src`/`e_dst` were dead columns and the two views' input
        // columns were free advice tied only to their own sorted view, with
        // nothing relating the views to each other, so the bag lookups
        // above proved membership in two INDEPENDENT prover-invented relations:
        // r_in could come from one edge list and r_out from another. Two
        // shuffles fix that. `in_by_dst` holds (dst, src, eid) and `out_by_src`
        // holds (src, dst, eid), so both are compared against the base table's
        // (src, dst, eid) with the two key/val columns swapped on the in-side.
        // The base table becomes the single edge relation both views project,
        // and the anchor a public commitment would bind to.
        let perm_edge_out = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            PermAnyChip::configure(
                meta,
                q1,
                q2,
                vec![e_src, e_dst, e_eid],
                vec![out_by_src.in_key, out_by_src.in_val, out_by_src.in_eid],
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
                vec![in_by_dst.in_val, in_by_dst.in_key, in_by_dst.in_eid],
            )
        };

        // The two order checks of the row, enabled only if real=1. `lt_ab`
        // compares (x,y), which is role 1's [A<B] and role 2's [C<D] at once,
        // and `lt_bc` compares (y,z), which is role 1's [B<C]. The deleted
        // `lt_cd` chip had operands (t34_c, t34_d) = (x, y) and so was the same
        // predicate on the same cells as `lt_ab`.
        let q_order = meta.selector();
        let lt_ab = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_order) * m.query_advice(w_real, Rotation::cur()),
            |m| m.query_advice(w_x, Rotation::cur()),
            |m| m.query_advice(w_y, Rotation::cur()),
        );
        let lt_bc = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_order) * m.query_advice(w_real, Rotation::cur()),
            |m| m.query_advice(w_y, Rotation::cur()),
            |m| m.query_advice(w_z, Rotation::cur()),
        );

        // Message aggregator
        let agg_msg = AggSumByKeyChip::<F>::configure(meta);

        // Tie agg_msg inputs to the ROLE-2 reading (C,D,A) = (x,y,z):
        // keep = w_real * [x<y]
        // in_key = keep ? pack2(A,C) = pack2(z,x) : PAD
        // in_val = keep
        //
        // The key is REVERSED relative to the role-1 probe key pack2(x,z) that
        // "role 1 real + key" pins on the SAME two cells of the SAME row. That
        // reversal is the whole mechanism: M(p,q) = |{w in W : x = p, z = q}| is
        // aggregated on pack2(z,x) and probed on pack2(x,z), so
        // answer = sum over w in W with y<z of M(z,x).
        //
        // The keep bit is real * [x<y], NOT the role-1 predicate. Aggregating
        // over the role-1 predicate would make the message count only rows that
        // also satisfy [y<z], i.e. it would silently require D<A on the
        // C->D->A path and change the answer.
        meta.create_gate("msg input from W (role 2)", |m| {
            let q = m.query_selector(q_w_msg_in);

            let z = m.query_advice(w_z, Rotation::cur());
            let x = m.query_advice(w_x, Rotation::cur());
            let key_expr = z * Expression::Constant(F::from(PACK_SHIFT)) + x;

            let real = m.query_advice(w_real, Rotation::cur());
            let xy = lt_ab.is_lt(m, None);
            let keep = real.clone() * xy;

            let one = Expression::Constant(F::ONE);
            let pad = Expression::Constant(F::from(PAD_U64));
            let selected_key = keep.clone() * key_expr + (one - keep.clone()) * pad;

            vec![
                q.clone() * (m.query_advice(agg_msg.in_key, Rotation::cur()) - selected_key),
                q * (m.query_advice(agg_msg.in_val, Rotation::cur()) - keep),
            ]
        });

        // The ROLE-1 predicate bit, folded into a column so that every gate of
        // the Cardinality Preservation Check below stays at degree <= 4, and
        // stated STRUCTURALLY as pred = keep * [y<z] on top of the role-2 bit
        // rather than as real * [x<y] * [y<z]. Beyond dropping the degree from 4
        // to 3, that form is what licenses the single occurrence-distinctness
        // argument: role 1's rows are exactly the role-2 rows that also pass a
        // genuine Lt on (y,z), decided per row, so distinctness of the role-2
        // occurrences carries over. Do not weaken this gate or the role-2
        // partition tail pin without restoring the second argument.
        let w_pred = meta.advice_column();
        meta.enable_equality(w_pred);
        let q_w_pred = meta.selector();
        meta.create_gate("role 1 pred = keep * [y<z]", |m| {
            let q = m.query_selector(q_w_pred);
            let keep = m.query_advice(agg_msg.in_val, Rotation::cur());
            let bc = lt_bc.is_lt(m, None);
            vec![q * (m.query_advice(w_pred, Rotation::cur()) - keep * bc)]
        });

        // Message lookup per row (table is agg_msg.map_* gated by agg_msg.q_map_tbl)
        let msg_lookup = MapLookupChip::<F>::configure(
            meta,
            agg_msg.map_key,
            agg_msg.map_val,
            agg_msg.map_key_next,
            agg_msg.q_map_tbl,
        );

        // w_real is boolean + tie msg_lookup.key to the ROLE-1 probe key,
        // pack2(A,C) = pack2(x,z) FORWARD. This reads the same two cells as the
        // aggregate's key gate above, on the same row, into a different advice
        // column: the two packings are different numbers, so the two columns
        // cannot merge. The booleanity polynomial here is the only one w_real
        // needs; the deleted "bag2 real boolean" gate was a second copy of it.
        meta.create_gate("role 1 real + key", |m| {
            let q = m.query_selector(q_w_key);
            let x = m.query_advice(w_x, Rotation::cur());
            let z = m.query_advice(w_z, Rotation::cur());
            let key_expr = x * Expression::Constant(F::from(PACK_SHIFT)) + z;

            let real = m.query_advice(w_real, Rotation::cur());
            let one = Expression::Constant(F::ONE);

            vec![
                q.clone() * real.clone() * (one.clone() - real.clone()),
                q * (m.query_advice(msg_lookup.key, Rotation::cur()) - key_expr),
            ]
        });

        // contrib = w_pred * msg_val, i.e. the role-1 predicate times M(z,x).
        // Spelling it through the folded bit instead of real * msgv * ab * bc
        // drops this gate from degree 5 to 3; cs.degree() is set by the
        // membership lookups either way.
        let contrib = meta.advice_column();
        meta.enable_equality(contrib);
        let q_contrib = meta.selector();
        meta.create_gate("contrib gate", |m| {
            let q = m.query_selector(q_contrib);

            let pred = m.query_advice(w_pred, Rotation::cur());
            let msgv = m.query_advice(msg_lookup.val, Rotation::cur());

            let outc = m.query_advice(contrib, Rotation::cur());
            vec![q * (outc - pred * msgv)]
        });

        // prefix sum of contrib
        let run_sum = meta.advice_column();
        meta.enable_equality(run_sum);
        let q_sum0 = meta.selector();
        let q_sum = meta.selector();

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

        // out equals run_sum at the row where q_out is enabled
        let out = meta.advice_column();
        meta.enable_equality(out);
        let q_out = meta.selector();
        meta.create_gate("out = run_sum", |m| {
            let q = m.query_selector(q_out);
            vec![
                q * (m.query_advice(out, Rotation::cur())
                    - m.query_advice(run_sum, Rotation::cur())),
            ]
        });

        // ================= Cardinality Preservation Check (condition (4)) =================
        //
        // `w_pred`, the role-1 predicate bit the check needs on the parent side,
        // is allocated with the order chips above so that the "contrib gate" can
        // read it too.

        // Clean indicator per ROLE. The two are genuinely different bits: a row
        // is role-2 clean iff it is kept and its pack2(z,x) is the key of a
        // role-1 clean row, and role-1 clean iff its predicate holds and its
        // pack2(x,z) carries a kept row. A row can be role-2 clean and role-1
        // dangling, so the two clean SETS differ and no single row order puts
        // both of them in a prefix: this is why the two partitions below stay
        // separate even though there is now only one tuple group.
        let w_cflag1 = meta.advice_column();
        let w_cflag2 = meta.advice_column();

        // ---------------- (1) Conservation Check ----------------
        // One permutation per role, between the indexed bag and the
        // concatenation of its two parts. The indices are distinct, so the
        // indexed bag is a set even though W holds duplicate wedges, and the
        // single permutation rules out an occurrence being fabricated, lost,
        // duplicated or counted on both sides: no Non-Membership Check.
        let row_idx = configure_row_index::<F>(meta);
        let cons_role1 = configure_conserve::<F>(
            meta,
            &row_idx,
            &[w_x, w_y, w_z, w_i, w_j, w_e1, w_e2],
            w_cflag1,
        );
        let cons_role2 = configure_conserve::<F>(
            meta,
            &row_idx,
            &[w_x, w_y, w_z, w_i, w_j, w_e1, w_e2],
            w_cflag2,
        );
        for c in [w_cflag1, w_cflag2] {
            meta.enable_equality(c);
        }

        // The indicator had no booleanity gate anywhere: only `enable_equality`
        // and the link gate below, which multiplies it by the predicate bit.
        // Two degree-3 constraints under a selector the circuit already enables
        // on every row.
        //
        // The predicate half of the Selector Check comes with it. `g_sql4_obj.rs`
        // got that structurally: the link gate below padded the indicator away
        // wherever the role's keep bit was 0, so `filt[5]` was `keep * c` and
        // the propagation read that product. With no filtered view to pad, the
        // implication is a constraint, and it is what makes the input channel of
        // the Cardinality Preservation Check count the join of the
        // predicate-filtered relations rather than of the raw bag. Role 1's bit
        // is `w_pred`, role 2's is `agg_msg.in_val` (already forced to
        // w_real * [x<y] by the "msg input from W" gate), so a selected row is
        // in particular a REAL row: this is the cluster adjustment, a padding
        // row contributes to neither channel.
        meta.create_gate("selector is a bit and implies its predicate", |m| {
            let one = Expression::Constant(F::ONE);
            let q = m.query_selector(q_w_pred);
            let c1 = m.query_advice(w_cflag1, Rotation::cur());
            let c2 = m.query_advice(w_cflag2, Rotation::cur());
            let b1 = m.query_advice(w_pred, Rotation::cur());
            let b2 = m.query_advice(agg_msg.in_val, Rotation::cur());
            vec![
                q.clone() * c1.clone() * (one.clone() - c1.clone()),
                q.clone() * c2.clone() * (one.clone() - c2.clone()),
                q.clone() * c1 * (one.clone() - b1),
                q * c2 * (one - b2),
            ]
        });

        // ================= Pairwise Consistency (condition (3)) =================
        // pi_K(R_i^c) == pi_K(R_j^c) on every join tree edge. Here the bag tree
        // has one edge, role 1 -- role 2 on the packed separator key pack2(A,C),
        // so the condition is two mutual Membership Checks between the clean
        // sections of the two partitions.
        //
        // Both sides read the BAG's own columns now, masked by that role's
        // selector bit. Which rows the containments range over is carried by
        // the mask: `c * pack2(A, C)` is the packed key on a selected row and 0
        // on every other one, and the lookup selectors cover the bag's whole
        // capacity. `g_sql4_obj.rs` read the clean prefix of a materialized
        // partition group instead, which needed a Conservation Check per role
        // to mean pi_K(R^c) and baked |R^c| into the fixed columns.
        //
        // The two roles read the same tuple, so the shared separator key
        // pack2(A,C) is pack2(x,z) on the role-1 side and pack2(z,x) on the
        // role-2 side: the SAME two columns in OPPOSITE order. Folding the
        // product into one column per role keeps both lookups at the degree of
        // the circuit's other membership arguments.
        let pw_key = (0..2).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let q_pw_key = (0..2).map(|_| meta.selector()).collect::<Vec<_>>();
        for (idx, (a_col, c_col, flag)) in
            [(w_x, w_z, w_cflag1), (w_z, w_x, w_cflag2)].iter().copied().enumerate()
        {
            let q = q_pw_key[idx];
            let key_col = pw_key[idx];
            meta.create_gate("pw: masked separator key = c * pack2(A,C)", move |m| {
                let q = m.query_selector(q);
                let a = m.query_advice(a_col, Rotation::cur());
                let c = m.query_advice(c_col, Rotation::cur());
                let packed = a * Expression::Constant(F::from(PACK_SHIFT)) + c;
                vec![
                    q * (m.query_advice(key_col, Rotation::cur())
                        - m.query_advice(flag, Rotation::cur()) * packed),
                ]
            });
        }

        // The two containments run directly between `pw_key[0]` and `pw_key[1]`.
        // The earlier round routed each of them through an intermediate advice
        // column holding [0] ++ uniq(keys of the other role's clean section), but
        // nothing in the circuit bound that column to the set it claimed to
        // enumerate: a prover could put role 1's clean keys in the table role 1
        // looks into and role 2's in the table role 2 looks into, and both lookups
        // would pass for an arbitrary partition, which made condition (3) vacuous.
        // Looking the two key columns up in each other leaves no free advice, and
        // costs one advice column and one complex selector per direction less.
        //
        // The selectors must be complex: a simple selector may not appear in a
        // lookup expression, so the q_cln_flag pair of the Conservation Check
        // cannot be reused here. Each one now gates both the input side of its
        // own direction and the table side of the other.
        let q_pw_in = (0..2).map(|_| meta.complex_selector()).collect::<Vec<_>>();

        // On every row where a selector is off both sides of its lookup read as
        // 0, so 0 is always in the table and the gated-off rows cost nothing. The
        // real keys are pack2(A,C) = A*2^32 + C over node ids shifted by SHIFT_ID,
        // hence at least 2^32 + 1, so the containment is over the real keys.
        let mut pw_edge = |name: &'static str,
                           q_in: Selector,
                           in_col: Column<Advice>,
                           q_t: Selector,
                           tbl_col: Column<Advice>| {
            meta.lookup_any(name, move |m| {
                let lhs = m.query_selector(q_in) * m.query_advice(in_col, Rotation::cur());
                let rhs = m.query_selector(q_t) * m.query_advice(tbl_col, Rotation::cur());
                vec![(lhs, rhs)]
            });
        };

        // pi_K(R_1^c) subset of pi_K(R_2^c): a role-1 tuple left on the clean side
        // with no clean role-2 partner is rejected here. The mirror direction makes
        // the two key sets equal rather than merely nested.
        pw_edge(
            "pw: role1^c key in role2^c",
            q_pw_in[0],
            pw_key[0],
            q_pw_in[1],
            pw_key[1],
        );
        pw_edge(
            "pw: role2^c key in role1^c",
            q_pw_in[1],
            pw_key[1],
            q_pw_in[0],
            pw_key[0],
        );

        // -------- the two channels over the single bag tree edge --------
        // One fixed column serves every Lt chip of the check, so the whole
        // check costs a single u8 range table.
        let cp_u8 = meta.fixed_column();

        // Child side, over the ROLE-2 reading. Role 2 is a leaf of the bag tree,
        // so its two multiplicities are the two per-tuple bits themselves: the
        // predicate bit `agg_msg.in_val` for the input channel and the bound
        // indicator `keep * c2` for the clean channel. The separator key is
        // `agg_msg.in_key`, already forced to pack2(A,C) = pack2(z,x) on the kept
        // rows and to PAD elsewhere, and a PAD key never reaches the emitted
        // table.
        let cp_agg_2 = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            agg_msg.in_key,
            agg_msg.in_val,
            w_cflag2,
            PAD_U64,
        );

        // Parent side, over the ROLE-1 reading, on the probe key that the
        // "role 1 real + key" gate ties to pack2(A,C) = pack2(x,z).
        let cp_join_2 = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, msg_lookup.key);
        wire_cp_edge(meta, &cp_join_2, &cp_agg_2, msg_lookup.key);

        // Root multiplicities and the single equality that compares the two
        // join cardinalities. `mu_all` reproduces `contrib`, so the input side
        // of condition (4) is the certified COUNT itself.
        let cp_root = configure_cp_root::<F>(meta);
        let q_cp_mu = meta.selector();
        {
            let s_all = cp_join_2.s_all;
            let s_cln = cp_join_2.s_cln;
            let cln_1 = w_cflag1;
            let mu_all = cp_root.mu_all;
            let mu_cln = cp_root.mu_cln;
            meta.create_gate("cp: root multiplicities over role 1", move |m| {
                let q = m.query_selector(q_cp_mu);
                let all = m.query_advice(mu_all, Rotation::cur())
                    - m.query_advice(w_pred, Rotation::cur())
                        * m.query_advice(s_all, Rotation::cur());
                let cln = m.query_advice(mu_cln, Rotation::cur())
                    - m.query_advice(cln_1, Rotation::cur())
                        * m.query_advice(s_cln, Rotation::cur());
                vec![q.clone() * all, q * cln]
            });

            // The module doc's claim that "mu_all reproduces contrib" was an
            // ASSUMPTION, not a constraint. `msg_lookup.val` is fetched from
            // the AggSumByKey map and `s_all` from the table `configure_cp_agg`
            // builds; the two are independent computations of the same per-key
            // COUNT over the same (in_key, in_val) pair, and the only statement
            // that they agree was a host-side `debug_assert_eq!` inside
            // `assign`, which is compiled out in release and never executed by a
            // hostile prover. So `contrib = pred * msgv`, the number the circuit
            // outputs, and `mu_all = pred * s_all`, the number condition (10)
            // certifies, could differ freely: condition (10) certified a
            // quantity unrelated to the answer.
            //
            // One degree-2 equality per row ties them. With it,
            // contrib = pred * msgv = pred * s_all = mu_all on every row, so
            // `run_sum` and `sum_all` are two accumulations of the same values
            // and the certified cardinality IS the exposed COUNT. `q_cp_mu` is
            // already enabled on every row, so this costs no selector.
            let msg_val = msg_lookup.val;
            meta.create_gate("cp: answer channel equals certified channel", move |m| {
                let q = m.query_selector(q_cp_mu);
                vec![
                    q * (m.query_advice(msg_val, Rotation::cur())
                        - m.query_advice(s_all, Rotation::cur())),
                ]
            });
        }

        // ============= occurrence distinctness of W =============
        // The bag was pinned only ROW BY ROW: the two lookups above say that a
        // row is SOME valid wedge occurrence, and nothing said that two rows are
        // different occurrences or that the bag holds all of them.
        // Every argument downstream is blind to that. Conservation is a multiset
        // equality, so it accepts a duplicate as readily as the original;
        // Pairwise Consistency compares KEY SETS, so a duplicate of a key that
        // already has a partner is invisible; and condition (10) is a difference
        // of two channels, so any transformation that moves both channels
        // equally passes. Concretely: overwrite one role-2 clean row with a
        // byte-copy of a role-2 clean row of a DIFFERENT clean key. Both keys keep
        // a clean occurrence, so both "pw: " lookups still pass; the clean and
        // residual section lengths do not move, so the honest vk still verifies;
        // and both cp channels move by the same amount, so `sum_all == sum_cln`
        // holds. Only the answer changes. This is the root cause the
        // partition-tail and sentinel fixes above do NOT reach.
        //
        // What closes it is distinctness plus a pinned count. Give every bag row
        // an occurrence id, `pack2` of the two edge ids the row joins. That id is
        // a FUNCTION of the occurrence: the view's `idx` recurrence makes
        // (key, idx) unique, so (y, i, j) determines (e1, e2).
        // Injectivity in the other direction is not needed for soundness -- a
        // collision would only make the constraint harder to satisfy -- it is a
        // completeness requirement, and honest edge ids are below n_base, hence
        // below the 32-bit lane. Sort the ids in a shuffled view, require STRICT
        // increase over the first `m = n_cln + n_res` rows and pin the rest to
        // PAD. Then:
        //
        //   * the tail pin of the role-2 partition already forces the number of
        //     rows that pass the role-2 keep bit to be exactly `m` (a kept row's
        //     tuple cannot be the PAD tuple, since x < y rules out x = y = PAD),
        //     so the sorted view holds exactly `m` real ids;
        //   * PAD cannot appear inside the increasing prefix, because the
        //     comparison at row m-1 reads the pinned row m and an 8-byte Lt can
        //     only certify PAD < PAD by a zero difference, which it rejects;
        //   * strict increase makes those `m` ids pairwise distinct: each step
        //     adds a difference in [1, 2^64], and m * 2^64 is nowhere near the
        //     field order, so the chain cannot wrap round onto itself;
        //   * a kept row cannot smuggle its id into the PAD tail either, since
        //     the two counts above force the number of PAD ids to be exactly
        //     n - m, so no kept row's id may equal PAD.
        //
        // So the `m` rows that pass the keep bit are `m` DISTINCT valid
        // occurrences, and since the honest W has exactly `m` of them, they
        // are all of them. W is now exactly the x<y-filtered wedge join of the
        // edge relation in `e_*`, which is what makes the two channels of
        // condition (10) count the right thing rather than merely count
        // consistently.
        //
        // ONE argument, over the ROLE-2 keep bit `agg_msg.in_val`, does both
        // roles. The role-1 copy of it, over `w_pred` and the same two edge id
        // columns, would be a second argument about the same rows: role 1's rows
        // are the role-2 rows on which a genuine Lt certifies [y<z], decided per
        // row by the "role 1 pred = keep * [y<z]" gate, so distinctness of the
        // role-2 occurrences already gives distinctness of the role-1 subset, and
        // the role-1 count is pinned by its own partition tail. That is why this
        // deletion is licensed ONLY while that gate and the role-2 partition tail
        // pin both stay. Saves two advice columns, one shuffle, one Lt chip
        // (9 advice + 8 lookups) and two selectors.
        //
        // Cost: two advice columns, one width-1 shuffle and one Lt chip, sharing
        // the check's u8 column. Highest new gate degree is 3.
        let w_oid = meta.advice_column();
        let oid_sorted = meta.advice_column();
        meta.enable_equality(w_oid);
        meta.enable_equality(oid_sorted);

        // The role-2 bit is `agg_msg.in_val`, which the "msg input from W
        // (role 2)" gate forces to w_real * [x<y], and the occurrence is
        // (w_e1, w_e2). `q_w_msg_in` is already enabled on every row, so the id
        // gate costs no selector.
        {
            let keep = agg_msg.in_val;
            meta.create_gate(
                "occ: id = pack2(edge ids), PAD when the keep bit fails",
                move |m| {
                    let q = m.query_selector(q_w_msg_in);
                    let k = m.query_advice(keep, Rotation::cur());
                    let one = Expression::Constant(F::ONE);
                    let id = m.query_advice(w_e1, Rotation::cur())
                        * Expression::Constant(F::from(PACK_SHIFT))
                        + m.query_advice(w_e2, Rotation::cur());
                    let pad = Expression::Constant(F::from(PAD_U64));
                    vec![
                        q * (m.query_advice(w_oid, Rotation::cur())
                            - (k.clone() * id + (one - k) * pad)),
                    ]
                },
            );
        }

        let perm_oid = {
            let q1 = meta.complex_selector();
            let q2 = meta.complex_selector();
            PermAnyChip::configure(meta, q1, q2, vec![w_oid], vec![oid_sorted])
        };

        let q_oid_sort = meta.selector();
        let lt_oid = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            cp_u8,
            |m| m.query_selector(q_oid_sort),
            |m| m.query_advice(oid_sorted, Rotation::cur()),
            |m| m.query_advice(oid_sorted, Rotation::next()),
        );
        meta.create_gate("occ: ids strictly increase over R^c U R^r", move |m| {
            let q = m.query_selector(q_oid_sort);
            vec![q * (lt_oid.is_lt(m, None) - Expression::Constant(F::ONE))]
        });

        let q_oid_pad = meta.selector();
        meta.create_gate("occ: id tail is PAD", move |m| {
            let q = m.query_selector(q_oid_pad);
            vec![
                q * (m.query_advice(oid_sorted, Rotation::cur())
                    - Expression::Constant(F::from(PAD_U64))),
            ]
        });

        Cycle4OrderedConfig {
            instance,
            e_eid,
            e_src,
            e_dst,
            in_by_dst,
            out_by_src,
            q_view_tbl,
            perm_edge_out,
            perm_edge_in,
            w_x,
            w_y,
            w_z,
            w_i,
            w_j,
            w_e1,
            w_e2,
            w_real,
            q_w_lookup,
            q_w_key,
            q_w_msg_in,
            agg_msg,
            msg_lookup,
            q_order,
            lt_ab,
            lt_bc,
            contrib,
            q_contrib,
            run_sum,
            q_sum0,
            q_sum,
            out,
            q_out,
            cp_agg_2,
            cp_join_2,
            cp_root,
            q_cp_mu,
            w_pred,
            q_w_pred,
            w_cflag1,
            w_cflag2,
            row_idx,
            cons_role1,
            cons_role2,
            pw_key,
            q_pw_key,
            q_pw_in,
            w_oid,
            oid_sorted,
            perm_oid,
            q_oid_sort,
            q_oid_pad,
            lt_oid,
        }
    }

    /// `pad_extra` is a SINGLE knob: there is one materialized bag, so the two
    /// pad counts this used to take were always two names for the same number.
    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        edges: Vec<Edge>,
        pad_extra: usize,
    ) -> Result<AssignedCell<F, F>, Error> {
        let cfg = self.cfg.clone();

        // Load all LT tables used
        LtChip::<F, NUM_BYTES>::construct(cfg.in_by_dst.lt_key.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.out_by_src.lt_key.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.agg_msg.lt_key.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.agg_msg.lt_out_key.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.msg_lookup.lt_low.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.msg_lookup.lt_high.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.lt_ab.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.lt_bc.clone()).load(layouter)?;
        // Every Lt chip of the Cardinality Preservation Check shares one u8
        // fixed column, so a single load covers the whole check.
        LtChip::<F, NUM_BYTES>::construct(cfg.cp_agg_2.lt_key_cur_next).load(layouter)?;

        let in_view_chip = IndexedViewChip::<F>::construct(cfg.in_by_dst.clone());
        let out_view_chip = IndexedViewChip::<F>::construct(cfg.out_by_src.clone());
        let agg_msg_chip = AggSumByKeyChip::<F>::construct(cfg.agg_msg.clone());

        layouter.assign_region(
            || "cycle4_ordered witness",
            |mut region| {
                // -------------------
                // Base table with dummy row0 = (0,0,0)
                // -------------------
                let n_base = edges.len() + 1;
                for i in 0..n_base {
                    // the two bag lookups read the two views' rows 0..n_base-1
                    // as their table, and nothing else
                    cfg.q_view_tbl.enable(&mut region, i)?;
                    // both views are the base table's (src, dst, eid) multiset
                    cfg.perm_edge_out.q_perm1.enable(&mut region, i)?;
                    cfg.perm_edge_out.q_perm2.enable(&mut region, i)?;
                    cfg.perm_edge_in.q_perm1.enable(&mut region, i)?;
                    cfg.perm_edge_in.q_perm2.enable(&mut region, i)?;
                    if i == 0 {
                        region.assign_advice(|| "eid0", cfg.e_eid, 0, || Value::known(F::ZERO))?;
                        region.assign_advice(|| "src0", cfg.e_src, 0, || Value::known(F::ZERO))?;
                        region.assign_advice(|| "dst0", cfg.e_dst, 0, || Value::known(F::ZERO))?;
                    } else {
                        let e = &edges[i - 1];
                        let eid = i as u64;
                        let src = (e.src as u64) + SHIFT_ID;
                        let dst = (e.dst as u64) + SHIFT_ID;
                        region.assign_advice(
                            || "eid",
                            cfg.e_eid,
                            i,
                            || Value::known(F::from(eid)),
                        )?;
                        region.assign_advice(
                            || "src",
                            cfg.e_src,
                            i,
                            || Value::known(F::from(src)),
                        )?;
                        region.assign_advice(
                            || "dst",
                            cfg.e_dst,
                            i,
                            || Value::known(F::from(dst)),
                        )?;
                    }
                }

                // -------------------
                // Host-side derivation: view inputs, the one bag, message map.
                // Shared verbatim with `g_sql4_obj_dp`.
                // -------------------
                let derived = gq4_derive(&edges);
                let Gq4Derived {
                    in_rows,
                    out_rows,
                    w,
                    msg_map,
                    keys,
                } = derived;

                // Assign indexed views
                in_view_chip.assign(&mut region, n_base, &in_rows)?;
                out_view_chip.assign(&mut region, n_base, &out_rows)?;

                // ============ clean/residual partition, once per role ============
                // The Cardinality Preservation Check needs two sides to
                // compare, and this circuit partitions nothing, so the
                // partition is built here. The bag tree has two nodes, so the
                // semijoin reduction is exact: a tuple is clean in its role iff
                // that role's predicate holds and it extends to a full 4-cycle.
                //
                // ONE capacity knob: there is one materialized bag, so the two
                // pad counts this used to take were two names for one number.
                let real_w = w.len();
                let n = std::cmp::max(real_w + pad_extra, 1);

                // Role 2, the reading (C,D,A) = (x,y,z): keep = real * [x<y],
                // key = pack2(A,C) = pack2(z,x) REVERSED on the kept rows and
                // PAD elsewhere, exactly what the "msg input from W (role 2)"
                // gate forces on (agg_msg.in_key, agg_msg.in_val).
                let mut keep2: Vec<u64> = vec![0; n];
                let mut key2: Vec<u64> = vec![PAD_U64; n];
                for i in 0..real_w {
                    let (x, y, z, _i, _j, _e1, _e2) = w[i];
                    if x < y {
                        keep2[i] = 1;
                        key2[i] = pack2(z, x);
                    }
                }

                // Role 1, the reading (A,B,C) = (x,y,z): pred = keep * [y<z];
                // the probe key is pack2(A,C) = pack2(x,z) FORWARD on the real
                // rows and 0 on the pad rows.
                let mut pred1: Vec<u64> = vec![0; n];
                let mut key1: Vec<u64> = vec![0; n];
                for i in 0..real_w {
                    let (x, y, z, _i, _j, _e1, _e2) = w[i];
                    if keep2[i] == 1 && y < z {
                        pred1[i] = 1;
                    }
                    key1[i] = pack2(x, z);
                }

                // A role-1 tuple is clean iff its predicate holds and its
                // separator key carries at least one kept role-2 tuple; a role-2
                // tuple is clean iff it is kept and its key is the key of a
                // clean role-1 tuple. The two clean SETS therefore differ, which
                // is why the two partitions cannot be merged.
                let mut cln1: Vec<u64> = (0..n)
                    .map(|i| (pred1[i] == 1 && *msg_map.get(&key1[i]).unwrap_or(&0) > 0) as u64)
                    .collect();

                // test hook only: hide one joinable role-1 tuple in the residual
                // side. `cln2` is recomputed from the reduced `cln1` just
                // below, so the two partitions stay partitions and conditions
                // (1)-(3) still hold: only condition (4) can see this.
                let tamper = HIDE_ONE_CLEAN_TUPLE.load(Ordering::Relaxed);
                if tamper {
                    if let Some(h) = (0..n).find(|&i| cln1[i] == 1) {
                        cln1[h] = 0;
                    }
                }

                // test hook only: skip the reduction and declare every real
                // tuple of both roles clean, which leaves the residual side
                // empty. Conservation still holds and both channels of
                // condition (4) then agree on every row, so only Pairwise
                // Consistency can see the dangling tuples.
                let mark_all_clean = MARK_ALL_CLEAN.load(Ordering::Relaxed);
                if mark_all_clean {
                    cln1 = pred1.clone();
                }

                let clean_keys1: HashSet<u64> =
                    (0..n).filter(|&i| cln1[i] == 1).map(|i| key1[i]).collect();
                let cln2: Vec<u64> = if mark_all_clean {
                    keep2.clone()
                } else {
                    (0..n)
                        .map(|i| (keep2[i] == 1 && clean_keys1.contains(&key2[i])) as u64)
                        .collect()
                };

                // Both sides of both Conservation Checks. The five tuple columns
                // are shared: only the keep bit and the indicator differ.
                let pad_bag: [u64; 6] = [PAD_U64, PAD_U64, PAD_U64, PAD_U64, PAD_U64, 0];
                let tuples: Vec<[u64; 5]> = (0..n)
                    .map(|i| {
                        if i < real_w {
                            let (x, y, z, _i, _j, e1, e2) = w[i];
                            [x, y, z, e1, e2]
                        } else {
                            [0, 0, 0, 0, 0]
                        }
                    })
                    .collect();
                let (filt_2, part_2, n_cln2, n_res2) =
                    split_partition(&tuples, &keep2, &cln2, n, &pad_bag);
                let (filt_1, part_1, n_cln1, n_res1) =
                    split_partition(&tuples, &pred1, &cln1, n, &pad_bag);

                // -------------------
                // ONE pass over the bag rows. Both roles read the same row, so
                // the tuple columns, the two order witnesses, the aggregate
                // input, the probe, the folded role-1 bit, both indicators, the
                // contribution and the running sum are all assigned together.
                // -------------------
                // `keys`, the sorted gap-witness key list (with the 0 and PAD
                // sentinels), comes from `gq4_derive` above.

                let lt_ab_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_ab.clone());
                let lt_bc_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_bc.clone());
                let lt_low_chip = LtChip::<F, NUM_BYTES>::construct(cfg.msg_lookup.lt_low.clone());
                let lt_high_chip =
                    LtChip::<F, NUM_BYTES>::construct(cfg.msg_lookup.lt_high.clone());

                let mut agg_in: Vec<(u64, u64)> = vec![(PAD_U64, 0); n]; // default PAD bucket
                let mut running: u64 = 0;

                for i in 0..n {
                    cfg.q_w_lookup.enable(&mut region, i)?;
                    cfg.q_w_key.enable(&mut region, i)?;
                    cfg.q_w_msg_in.enable(&mut region, i)?;
                    cfg.msg_lookup.q_flag.enable(&mut region, i)?;
                    cfg.msg_lookup.q_complex.enable(&mut region, i)?;
                    cfg.q_order.enable(&mut region, i)?;
                    cfg.q_contrib.enable(&mut region, i)?;

                    let (x, y, z, i_in, j_out, e1, e2, real) = if i < real_w {
                        let (x, y, z, i_in, j_out, e1, e2) = w[i];
                        (x, y, z, i_in, j_out, e1, e2, 1u64)
                    } else {
                        (0, 0, 0, 0, 0, 0, 0, 0u64)
                    };

                    region.assign_advice(|| "w_x", cfg.w_x, i, || Value::known(F::from(x)))?;
                    region.assign_advice(|| "w_y", cfg.w_y, i, || Value::known(F::from(y)))?;
                    region.assign_advice(|| "w_z", cfg.w_z, i, || Value::known(F::from(z)))?;
                    region.assign_advice(|| "w_i", cfg.w_i, i, || Value::known(F::from(i_in)))?;
                    region.assign_advice(|| "w_j", cfg.w_j, i, || Value::known(F::from(j_out)))?;
                    region.assign_advice(|| "w_e1", cfg.w_e1, i, || Value::known(F::from(e1)))?;
                    region.assign_advice(|| "w_e2", cfg.w_e2, i, || Value::known(F::from(e2)))?;
                    region.assign_advice(
                        || "w_real",
                        cfg.w_real,
                        i,
                        || Value::known(F::from(real)),
                    )?;

                    // order witnesses (constraints disabled when real=0)
                    lt_ab_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(x)),
                        Value::known(F::from(y)),
                    )?;
                    lt_bc_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(y)),
                        Value::known(F::from(z)),
                    )?;

                    // role-2 aggregate input, on the REVERSED key
                    agg_in[i] = (key2[i], keep2[i]);

                    // role-1 message lookup witness, on the FORWARD key
                    let key = if real == 1 { key1[i] } else { 0 };
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
                                // pos is always < keys.len() because PAD_U64 is included and key < PAD_U64 here
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
                        cfg.msg_lookup.key,
                        i,
                        || Value::known(F::from(key)),
                    )?;
                    region.assign_advice(
                        || "msg_in_set",
                        cfg.msg_lookup.in_set,
                        i,
                        || Value::known(F::from(inside)),
                    )?;
                    region.assign_advice(
                        || "msg_low",
                        cfg.msg_lookup.low,
                        i,
                        || Value::known(F::from(low)),
                    )?;
                    region.assign_advice(
                        || "msg_high",
                        cfg.msg_lookup.high,
                        i,
                        || Value::known(F::from(high)),
                    )?;
                    region.assign_advice(
                        || "msg_val",
                        cfg.msg_lookup.val,
                        i,
                        || Value::known(F::from(val)),
                    )?;

                    // gap LT witnesses (relevant when inside=0)
                    lt_low_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(low)),
                        Value::known(F::from(key)),
                    )?;
                    lt_high_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(key)),
                        Value::known(F::from(high)),
                    )?;

                    // contrib witness: the role-1 predicate times M(z,x)
                    let contrib_u64 = pred1[i] * val;

                    region.assign_advice(
                        || "contrib",
                        cfg.contrib,
                        i,
                        || Value::known(F::from(contrib_u64)),
                    )?;

                    // the folded role-1 bit and both selector bits of this row
                    cfg.q_w_pred.enable(&mut region, i)?;
                    region.assign_advice(
                        || "w_pred",
                        cfg.w_pred,
                        i,
                        || Value::known(F::from(pred1[i])),
                    )?;
                    region.assign_advice(
                        || "w_cflag1",
                        cfg.w_cflag1,
                        i,
                        || Value::known(F::from(cln1[i])),
                    )?;
                    region.assign_advice(
                        || "w_cflag2",
                        cfg.w_cflag2,
                        i,
                        || Value::known(F::from(cln2[i])),
                    )?;

                    // running sum
                    // `wrapping_add` is a host-side accumulator only. The "sum"
                    // gate adds in the field, so a u64 wrap here would make
                    // `run_sum` disagree with the gate and the circuit would
                    // REJECT rather than accept a wrong COUNT: a completeness
                    // limit past 2^64 join occurrences, not a soundness hole.
                    if i == 0 {
                        cfg.q_sum0.enable(&mut region, i)?;
                        running = contrib_u64;
                    } else {
                        cfg.q_sum.enable(&mut region, i)?;
                        running = running.wrapping_add(contrib_u64);
                    }
                    region.assign_advice(
                        || "run_sum",
                        cfg.run_sum,
                        i,
                        || Value::known(F::from(running)),
                    )?;
                }

                // Aggregate the role-2 rows into the msg map table (in-circuit).
                // This also writes agg_msg.in_key / in_val, which the
                // "msg input from W (role 2)" gate ties to the row.
                let _msg_emitted = agg_msg_chip.assign(&mut region, n, &agg_in)?;

                // ---- occurrence distinctness, over the role-2 keep bit ----
                // `pack2` of the two edge ids is injective only while both fit a
                // 32-bit lane, which also keeps every real id strictly below the
                // PAD id that marks the rows failing the keep bit. One argument
                // ---- (1) Conservation Check, one permutation per role ----
                {
                    let w_rows: Vec<Vec<u64>> = (0..n)
                        .map(|i| {
                            if i < real_w {
                                let (x, y, z, i_in, j_out, e1, e2) = w[i];
                                vec![x, y, z, i_in, j_out, e1, e2]
                            } else {
                                vec![0u64; 7]
                            }
                        })
                        .collect();
                    assign_row_index(&mut region, &cfg.row_idx, n)?;
                    assign_conserve(&mut region, &cfg.cons_role1, &w_rows, &cln1)?;
                    assign_conserve(&mut region, &cfg.cons_role2, &w_rows, &cln2)?;
                }

                // covers both roles: role 1's rows are the role-2 rows that also
                // pass the pinned [y<z] test.
                assert!(
                    (n_base as u64) < PACK_SHIFT,
                    "the occurrence id packs two edge ids into 32-bit lanes"
                );
                let oid: Vec<u64> = (0..n)
                    .map(|i| {
                        if keep2[i] == 1 {
                            pack2(w[i].5, w[i].6)
                        } else {
                            PAD_U64
                        }
                    })
                    .collect();
                assign_occurrence_ids(
                    &mut region,
                    cfg.w_oid,
                    cfg.oid_sorted,
                    &cfg.perm_oid,
                    cfg.q_oid_sort,
                    cfg.q_oid_pad,
                    &cfg.lt_oid,
                    &oid,
                    n_cln2 + n_res2,
                )?;

                // ---- child side of the bag tree edge, both channels ----
                // Role 2 is a leaf, so a row's input-channel multiplicity is its
                // keep bit and its clean-channel multiplicity is that bit times
                // the role-2 clean indicator. Both already sit in columns the
                // circuit carries, so the second channel costs one column and
                // the stage groups them by the reversed separator key in one
                // pass.
                let cp_rows_2: Vec<[u64; 3]> = (0..n)
                    .map(|i| [key2[i], keep2[i], keep2[i] * cln2[i]])
                    .collect();
                let cp_stage_2 = build_cp_stage(&cp_rows_2, PAD_U64);
                assign_cp_agg(&mut region, &cfg.cp_agg_2, &cp_rows_2, &cp_stage_2)?;

                // ============== (2) PAIRWISE CONSISTENCY ==============
                // The two key sets of the one bag tree edge, read off the BAG
                // rows masked by each role's selector bit and looked up in each
                // other. The shared key pack2(A,C) is pack2(x,z) in role 1 and
                // pack2(z,x) in role 2, the same two cells in opposite order,
                // exactly as the masked-key gate reads them. A deselected or
                // padding row masks to 0, which every table contains, so the
                // selectors cover the bag's whole capacity and no clean-prefix
                // extent reaches the fixed columns.
                let pw_keys: [Vec<u64>; 2] = [
                    (0..n)
                        .map(|i| if cln1[i] == 1 { pack2(w[i].0, w[i].2) } else { 0 })
                        .collect(),
                    (0..n)
                        .map(|i| if cln2[i] == 1 { pack2(w[i].2, w[i].0) } else { 0 })
                        .collect(),
                ];
                for r in 0..2 {
                    for (i, &k) in pw_keys[r].iter().enumerate() {
                        cfg.q_pw_key[r].enable(&mut region, i)?;
                        cfg.q_pw_in[r].enable(&mut region, i)?;
                        region.assign_advice(
                            || "pw_key",
                            cfg.pw_key[r],
                            i,
                            || Value::known(F::from(k)),
                        )?;
                    }
                }

                // ============== CARDINALITY PRESERVATION CHECK ==============
                // Parent side of the single bag tree edge: fetch both sigma
                // values for this row's role-1 separator key, or certify with a
                // gap witness that the key occurs in no role-2 tuple and take 0.
                let fetched =
                    assign_cp_join(&mut region, &cfg.cp_join_2, &key1, &cp_stage_2, PAD_U64)?;

                // Root multiplicities and the equality between the two sums.
                let cp_mu: Vec<(u64, u64)> = (0..n)
                    .map(|i| (pred1[i] * fetched[i].0, pred1[i] * cln1[i] * fetched[i].1))
                    .collect();
                for i in 0..n {
                    cfg.q_cp_mu.enable(&mut region, i)?;
                }
                let (cp_all, cp_cln) = assign_cp_root(&mut region, &cfg.cp_root, &cp_mu)?;
                if !tamper && !mark_all_clean {
                    debug_assert_eq!(
                        cp_all, cp_cln,
                        "cardinality preservation: |R^c join| != |R join|"
                    );
                    debug_assert_eq!(
                        cp_all, running,
                        "the input channel must reproduce the COUNT"
                    );
                }

                // enable q_out at the last sum row
                let out_row = n - 1;
                cfg.q_out.enable(&mut region, out_row)?;
                let out_cell = region.assign_advice(
                    || "out",
                    cfg.out,
                    out_row,
                    || Value::known(F::from(running)),
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

/// Wrapper circuit (uses dataset Edge directly)
pub struct MyCircuit<F: Field + Ord> {
    pub edges: Vec<Edge>,
    /// ONE capacity knob: the circuit materializes one bag, read in two column
    /// roles, so the two pad counts this used to carry were one number twice.
    pub pad_extra: usize,
    pub _marker: PhantomData<F>,
}
impl<F: Field + Ord> Default for MyCircuit<F> {
    fn default() -> Self {
        Self {
            edges: vec![],
            pad_extra: 0,
            _marker: PhantomData,
        }
    }
}

impl<F: Field + Ord> Circuit<F> for MyCircuit<F> {
    type Config = Cycle4OrderedConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        Cycle4OrderedChip::<F>::configure(meta)
    }

    fn synthesize(&self, cfg: Self::Config, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        let chip = Cycle4OrderedChip::<F>::construct(cfg);
        let out = chip.assign(&mut layouter, self.edges.clone(), self.pad_extra)?;
        chip.expose_public(&mut layouter, out, 0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::graph_data_processing::read_edges;
    use crate::data::graph_data_processing::read_edges_csv;
    use halo2_proofs::dev::{MockProver, VerifyFailure};

    use halo2_proofs::{
        plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit},
        poly::{
            commitment::Params,
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
    use halo2curves::pasta::{vesta, EqAffine, Fp};
    use rand::rngs::OsRng;
    use std::time::Instant;
    use std::{fs::File, io::Write, path::Path};

    fn generate_and_verify_proof<C: Circuit<Fp>>(
        circuit: C,
        public_input: &[Fp],
        proof_path: &str,
    ) {
        let params_path = &crate::paths::param_file(18);
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

    fn expected_cnt(edges: &[Edge]) -> u64 {
        use std::collections::HashMap;

        // multiplicity of each directed edge
        let mut cnt: HashMap<(u64, u64), u64> = HashMap::new();
        for e in edges {
            let s = e.src as u64;
            let d = e.dst as u64;
            *cnt.entry((s, d)).or_default() += 1;
        }

        // adjacency by src: src -> [(dst, mult)]
        let mut out: HashMap<u64, Vec<(u64, u64)>> = HashMap::new();
        for (&(s, d), &c) in cnt.iter() {
            out.entry(s).or_default().push((d, c));
        }

        let mut total: u128 = 0;

        // Enumerate A->B, B->C, C->D, D->A with A<B<C<D
        for (&a, outs_ab) in out.iter() {
            for &(b, cab) in outs_ab.iter() {
                if !(a < b) {
                    continue;
                }
                if let Some(outs_bc) = out.get(&b) {
                    for &(c, cbc) in outs_bc.iter() {
                        if !(b < c) {
                            continue;
                        }
                        if let Some(outs_cd) = out.get(&c) {
                            for &(d, ccd) in outs_cd.iter() {
                                if !(c < d) {
                                    continue;
                                }
                                if let Some(&cda) = cnt.get(&(d, a)) {
                                    total += (cab as u128)
                                        * (cbc as u128)
                                        * (ccd as u128)
                                        * (cda as u128);
                                }
                            }
                        }
                    }
                }
            }
        }
        total as u64
    }

    /// Returns: (src_node, src_max_cnt, dst_node, dst_max_cnt)
    fn max_src_dst_frequency(edges: &[Edge]) -> (u64, u64, u64, u64) {
        let mut src_cnt: HashMap<u64, u64> = HashMap::new();
        let mut dst_cnt: HashMap<u64, u64> = HashMap::new();

        for e in edges {
            let s = e.src as u64;
            let d = e.dst as u64;
            *src_cnt.entry(s).or_insert(0) += 1;
            *dst_cnt.entry(d).or_insert(0) += 1;
        }

        let (mut best_s, mut best_sc) = (0u64, 0u64);
        for (s, c) in src_cnt {
            if c > best_sc {
                best_sc = c;
                best_s = s;
            }
        }

        let (mut best_d, mut best_dc) = (0u64, 0u64);
        for (d, c) in dst_cnt {
            if c > best_dc {
                best_dc = c;
                best_d = d;
            }
        }

        (best_s, best_sc, best_d, best_dc)
    }

    #[test]
    #[ignore = "inherited heavy end-to-end proof; the fast check is test_cardinality_preservation"]
    fn test_max_fre() {
        let base_path = &crate::paths::graph_dir();
        // let mut edges = read_edges(&format!("{}/wiki/wiki_Vote.txt", base_path)).unwrap();
        // [test_max_fre] max src freq: node=2565 count=893
        // [test_max_fre] max dst freq: node=4037 count=457

        // let mut edges =
        //     read_edges(&format!("{}/facebook/facebook_combined.txt", base_path)).unwrap();
        // [test_max_fre] max src freq: node=107 count=1043
        // [test_max_fre] max dst freq: node=1888 count=251

        let mut edges =
            read_edges_csv(&format!("{}/last/lastfm_asia_edges.csv", base_path)).unwrap();
        // [test_max_fre] max src freq: node=524 count=164
        // [test_max_fre] max dst freq: node=7237 count=203

        let (s, sc, d, dc) = max_src_dst_frequency(&edges);

        println!("[test_max_fre] max src freq: node={} count={}", s, sc);
        println!("[test_max_fre] max dst freq: node={} count={}", d, dc);
    }

    #[test]
    #[ignore = "inherited heavy end-to-end proof; the fast check is test_cardinality_preservation"]
    fn test() {
        let dataset = std::env::var("VPJOIN_DATASET").unwrap_or_else(|_| "lastfm".into());
        let edges = crate::bench_queries::load_graph(&dataset);
        let cnt = expected_cnt(&edges);

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
        // `graph_pads` keeps its (usize, usize) signature for GQ3 and Q5; the
        // GQ4 consumer reads .0 only, since there is one bag to pad.
        let (pad_extra, _) = crate::bench_queries::graph_pads("gq4", &dataset, &edges, privacy);
        println!(
            "[gq4 test] dataset={} privacy={} pad={}",
            dataset,
            privacy.label(),
            pad_extra
        );

        let circuit = MyCircuit::<Fp> {
            edges,
            pad_extra,
            _marker: PhantomData,
        };

        let public_input = vec![Fp::from(cnt)];
        let k = crate::bench_queries::degree_for("gq4", &dataset, privacy);

        // With VPJOIN_MOCK=1 this checks every gate, lookup and shuffle at full
        // scale under MockProver, which does no cryptography at all, instead of
        // generating a real proof. That is the cheap way to confirm the circuit
        // still fits its degree and its domain on the whole dataset. Unset, it
        // measures a real keygen / prove / verify, which is the number the paper
        // reports.
        let test = std::env::var("VPJOIN_MOCK")
            .map(|v| v == "1")
            .unwrap_or(false);

        if test {
            let prover = MockProver::run(k, &circuit, vec![public_input]).unwrap();
            prover.assert_satisfied();
        } else {
            let proof_path = &crate::paths::proof_file("last_proof_q4_new");
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
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
            println!("[gq4 cp] {} failing constraint: {}", label, n);
        }
    }

    /// Cost and degree probe. `cs.degree()` drives the FFT size of every
    /// polynomial in the proof, so a rise here costs far more than any of the
    /// gates below save. The bound is the one the Cardinality Preservation
    /// Check already needs: its `cp: key and its two sums in the child table`
    /// lookup has a degree-3 input and a degree-2 table side, i.e. 2 + 3 + 2.
    #[test]
    fn test_max_gate_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        let mut cs = ConstraintSystem::<Fp>::default();
        let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
        println!("cs.degree() = {}", cs.degree());
        println!(
            "advice={} fixed={} instance={} selectors={} gates={} lookups={} shuffles={} perm_cols={}",
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
            "the maximum gate degree rose to {}, which doubles every FFT",
            cs.degree()
        );
    }

    /// A truncated slice of `wiki_Vote`, small enough for MockProver.
    ///
    /// `lastfm` and `facebook` list every undirected edge once with
    /// `src < dst`, so a directed 4-cycle with A<B<C<D would need the edge
    /// D->A with D > A and none exists: both datasets answer 0 and leave the
    /// clean instance empty, which would make this test vacuous. `wiki_Vote`
    /// is genuinely directed, so the slice below has real 4-cycles.
    fn cp_test_edges() -> Vec<Edge> {
        const N_EDGES: usize = 3000;
        let mut edges = read_edges(&crate::paths::graph_file("wiki/wiki_Vote.txt"))
            .expect("wiki_Vote.txt not found");
        edges.truncate(N_EDGES);
        edges
    }

    /// Fast correctness check of the Cardinality Preservation Check and of
    /// Pairwise Consistency: a truncated slice of the dataset under MockProver,
    /// which verifies every gate, shuffle and lookup of the circuit without
    /// paying for a real proof.
    #[test]
    fn test_cardinality_preservation() {
        let k = 15;
        let edges = cp_test_edges();
        let cnt = expected_cnt(&edges);
        println!("[gq4 cp] edges={} cnt={}", edges.len(), cnt);
        assert!(
            cnt > 0,
            "the slice has no 4-cycle, the test would be vacuous"
        );

        // The third direction below only bites if the slice really has dangling
        // bag tuples: with none of them the all-clean partition would be the
        // reduced instance and condition (3) would rightly accept it. Count
        // them on both ends of the bag tree edge, the same way `assign` does:
        // one relation W, read as (A,B,C) with the extra [y<z] for role 1 and as
        // (C,D,A) with nothing extra for role 2, on the two opposite packings of
        // the separator.
        let derived = gq4_derive(&edges);
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
            "[gq4 cp] |W|={} dangling: role1={} role2={}",
            derived.w.len(),
            dangling1,
            dangling2
        );
        assert!(
            dangling1 + dangling2 > 0,
            "the slice has no dangling bag tuple, the all-clean direction would be vacuous"
        );

        let circuit = MyCircuit::<Fp> {
            edges,
            pad_extra: 0,
            _marker: PhantomData,
        };

        let prover = MockProver::run(k, &circuit, vec![vec![Fp::from(cnt)]]).unwrap();
        prover.assert_satisfied();

        // Negative direction: the same witness with one joinable role-1 tuple
        // hidden in the residual side, and role 2 re-reduced around it so that
        // Conservation, Non-Membership and Pairwise Consistency all still hold.
        // Only condition (4) can see this, so the circuit must now reject.
        super::HIDE_ONE_CLEAN_TUPLE.store(true, Ordering::Relaxed);
        let tampered = MockProver::run(k, &circuit, vec![vec![Fp::from(cnt)]]).unwrap();
        let verdict = tampered.verify();
        super::HIDE_ONE_CLEAN_TUPLE.store(false, Ordering::Relaxed);

        // The tampered witness produces exactly one failure, the
        // "cp: cardinality preservation" gate at the last row: the two
        // Conservation Checks, the bag lookups and the membership + gap proof
        // at the separator are all still satisfied, so nothing but condition
        // (4) sees the hidden tuple.
        let failures = verdict.expect_err("condition (4) accepted a hidden joinable tuple");
        println!("[gq4 cp] tampered failures: {}", failures.len());
        report_failing_constraints("tampered", &failures);
        assert!(
            failures
                .iter()
                .any(|f| format!("{:?}", f).contains("cardinality preservation")),
            "the circuit rejected, but not through the Cardinality Preservation Check: {:?}",
            failures
        );

        // Third direction: the escape that Pairwise Consistency closes. Every
        // real bag tuple is declared clean and the residual side is left empty,
        // so both Conservation Checks still pass and the two channels of
        // condition (4) compute the same number on every row. Only condition
        // (3) can see the dangling tuples counted above, and it must, through a
        // "pw: " lookup.
        super::MARK_ALL_CLEAN.store(true, Ordering::Relaxed);
        let all_clean = MockProver::run(k, &circuit, vec![vec![Fp::from(cnt)]]).unwrap();
        let verdict = all_clean.verify();
        super::MARK_ALL_CLEAN.store(false, Ordering::Relaxed);

        let failures = verdict.expect_err("condition (3) accepted the all-clean partition");
        println!("[gq4 cp] all-clean failures: {}", failures.len());
        report_failing_constraints("all-clean", &failures);
        assert!(
            failures.iter().any(|f| matches!(
                f,
                VerifyFailure::Lookup { name, .. } if name.starts_with("pw: ")
            )),
            "the circuit rejected, but not through a Pairwise Consistency lookup: {:?}",
            failures
        );
    }
}
