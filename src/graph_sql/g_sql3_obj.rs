use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};
use crate::circuits::card_preserve::{
    assign_cp_agg, assign_cp_join, assign_cp_root, build_cp_stage, configure_cp_agg,
    configure_cp_join, configure_cp_root, wire_cp_edge, CpAggConfig, CpJoinConfig, CpRootConfig,
};

// ✅ Use the dataset Edge type directly.
use crate::data::graph_data_processing::Edge;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

// IMPORTANT: pack2(A,C) can exceed 5 bytes. Use 8 bytes to avoid LT gate failures.
// `pub(crate)` so the multi-lane DP variant `g_sql3_obj_dp` inherits exactly the
// same PAD / packing conventions instead of restating them.
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
/// joinable Bag2 tuple to the residual side and re-reduces the bags around it,
/// so the partition still passes Conservation and stays pairwise consistent on
/// the clean side, and only condition (4) can catch it. This is exactly the
/// cheat a residual-side-only argument misses, so the negative test in this
/// module is what shows the Cardinality Preservation Check is not vacuous.
pub static HIDE_ONE_CLEAN_TUPLE: AtomicBool = AtomicBool::new(false);

/// Test hook, off in every benchmark path: when set, the prover skips the
/// semijoin reduction entirely and declares every real tuple clean, so the
/// residual side of both bags holds only what the bag's own predicate already
/// dropped. Conservation still holds, and with the two channels then carrying
/// the same multiplicity on every row both root sums of condition (4) agree for
/// free. This is exactly the escape Pairwise Consistency has to close, so the
/// third direction of the test in this module is what shows condition (3) is
/// not vacuous.
pub static MARK_ALL_CLEAN: AtomicBool = AtomicBool::new(false);

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

/// ------------------------------
/// Host-side witness derivation, shared by this circuit and its multi-lane
/// DP variant `g_sql3_obj_dp`. Pure function of the edge list: it knows
/// nothing about padding, degrees or lanes.
/// ------------------------------
pub(crate) struct Gq3Derived {
    /// InByDst input rows (key=dst, val=src, eid); row 0 is the (0,0,0) dummy.
    pub in_rows: Vec<(u64, u64, u64)>,
    /// OutBySrc input rows (key=src, val=dst, eid); row 0 is the (0,0,0) dummy.
    pub out_rows: Vec<(u64, u64, u64)>,
    /// Bag1 wedges A->B->C as (A,B,C,i_r1,j_r2,r1_eid,r2_eid), pre-filtered to A<B.
    pub t12: Vec<(u64, u64, u64, u64, u64, u64, u64)>,
    /// Bag2 closing edges C->A as (C,A,j_r3,r3_eid).
    pub t3: Vec<(u64, u64, u64, u64)>,
    /// Message table: msg_key = pack2(A,C) -> COUNT(edges C->A).
    pub msg_map: BTreeMap<u64, u64>,
    /// Sorted, deduped msg keys plus the 0 and PAD sentinels, for gap witnesses.
    pub keys: Vec<u64>,
}

pub(crate) fn gq3_derive(edges: &[Edge]) -> Gq3Derived {
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
    // Build view inputs
    // InByDst: key=dst, val=src
    // OutBySrc: key=src, val=dst
    // -------------------
    let mut in_rows: Vec<(u64, u64, u64)> = Vec::with_capacity(n_base);
    let mut out_rows: Vec<(u64, u64, u64)> = Vec::with_capacity(n_base);
    for (eid, src, dst) in base.iter().copied() {
        in_rows.push((dst, src, eid));
        out_rows.push((src, dst, eid));
    }

    // -------------------
    // Host-side grouping by key with eid-sorted order (to match IndexedView sorting)
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
    // Bag1: wedges A->B->C from join on B
    // row = (A,B,C,i_r1,j_r2,r1_eid,r2_eid)
    // ✅ Filter early: only keep wedges with A < B  (i.e., r1.src < r2.src)
    // -------------------
    let mut t12: Vec<(u64, u64, u64, u64, u64, u64, u64)> = vec![];
    for (&b, incoming) in in_groups.iter() {
        if let Some(outgoing) = out_groups.get(&b) {
            for (i, (r1_eid, a)) in incoming.iter().enumerate() {
                // EARLY FILTER: r1.src < r2.src  (A < B)
                // Note: a and b are already SHIFT_ID-shifted, inequality is preserved.
                if *a >= b {
                    continue;
                }
                for (j, (r2_eid, c)) in outgoing.iter().enumerate() {
                    t12.push((*a, b, *c, i as u64, j as u64, *r1_eid, *r2_eid));
                }
            }
        }
    }

    // -------------------
    // Bag2: edges C->A (r3) from OutBySrc grouped by C
    // row = (C,A,j_r3,r3_eid)
    // -------------------
    let mut t3: Vec<(u64, u64, u64, u64)> = vec![];
    for (&c, outs) in out_groups.iter() {
        for (j, (r3_eid, a)) in outs.iter().enumerate() {
            // edge is c -> a
            t3.push((c, *a, j as u64, *r3_eid));
        }
    }

    // -------------------
    // Message map: msg_key = pack2(A,C), msg_val = count of edges C->A
    // -------------------
    let mut msg_map: BTreeMap<u64, u64> = BTreeMap::new();
    for (c, a, _j, _eid) in t3.iter().copied() {
        let key = pack2(a, c);
        *msg_map.entry(key).or_default() += 1;
    }

    // key list for gap witnesses
    let mut keys: Vec<u64> = msg_map.keys().copied().collect();
    keys.push(0);
    keys.push(PAD_U64);
    keys.sort();
    keys.dedup();

    Gq3Derived {
        in_rows,
        out_rows,
        t12,
        t3,
        msg_map,
        keys,
    }
}

/// ------------------------------
/// AggSumByKey: group-by SUM(val) over key
/// Produces:
///  - out table: sorted unique keys with sums (padded)
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
    /// pins the one-past-the-end cells of the two sorted views to PAD; enabled
    /// on row `n` only
    q_sentinel: Selector,
    pub(crate) lt_key: LtConfig<F, NUM_BYTES>,
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
    pub(crate) lt_out_key: LtConfig<F, NUM_BYTES>,
    iz_out_eq: IsZeroConfig<F>,

    pub map_key: Column<Advice>,
    pub map_val: Column<Advice>,
    pub map_key_next: Column<Advice>,

    // ✅ used on RHS in lookup_any => must be complex_selector()
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
        let lt_key = LtChip::<F, NUM_BYTES>::configure(
            meta,
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

        // The one-past-the-end cell of the sorted view, row `n`. It is READ: the
        // group-boundary detector `iz_same_next` on the last real row compares
        // against it, and nondecreasing alone does not pin it. Left free, a
        // prover sets it equal to the last real key, `iz_same_next` then reports
        // "same group" on the last real row, the emit gate below writes
        // (PAD, 0) instead of that group's sum, and the highest-key group
        // disappears from the message table the answer is read off. The
        // separate `cp_agg_msg` group-by over the same input columns has its own
        // sentinel pinned, so condition (10) keeps that group and still
        // balances; only the certified COUNT(*) comes out short.
        //
        // This is the same hole and the same fix as "cp: sorted view sentinel is
        // PAD" in `crate::circuits::card_preserve`, and for the reason given
        // there it pins the VALUE rather than requiring the last comparison to
        // be strict: the Bag2 rows include padding keyed at PAD, so the sorted
        // view legitimately ends with real rows at PAD and those must not be
        // emitted as a group.
        let q_sentinel = meta.selector();

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
        let lt_out_key = LtChip::<F, NUM_BYTES>::configure(
            meta,
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

        // Both one-past-the-end sentinels, pinned to PAD by two degree-1
        // constraints under one selector enabled on row `n` only. `sorted_key[n]`
        // is the load-bearing one (see above); `out_key[n]` is read only by the
        // last `q_out_sort` comparison and never reaches the map, but it costs
        // nothing to pin it in the same gate and it removes the last free cell
        // of the aggregator.
        meta.create_gate("agg sorted view sentinels are PAD", |m| {
            let q = m.query_selector(q_sentinel);
            let pad = Expression::Constant(F::from(PAD_U64));
            vec![
                q.clone() * (m.query_advice(sorted_key, Rotation::cur()) - pad.clone()),
                q * (m.query_advice(out_key, Rotation::cur()) - pad),
            ]
        });

        // map table
        let map_key = meta.advice_column();
        let map_val = meta.advice_column();
        let map_key_next = meta.advice_column();
        meta.enable_equality(map_key);
        meta.enable_equality(map_val);
        meta.enable_equality(map_key_next);

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

        for i in 0..n {
            cfg.perm_sort.q_perm1.enable(region, i)?;
            cfg.perm_sort.q_perm2.enable(region, i)?;
        }

        let mut sorted = rows.to_vec();
        sorted.sort_by_key(|(k, _)| *k);
        let mut sorted_ext = sorted.clone();
        sorted_ext.push((PAD_U64, 0)); // sentinel

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
        region.assign_advice(
            || "out_key_s",
            cfg.out_key,
            n,
            || Value::known(F::from(PAD_U64)),
        )?;
        region.assign_advice(|| "out_sum_s", cfg.out_sum, n, || Value::known(F::ZERO))?;

        // both sentinels of this aggregator live on row `n` and are pinned to PAD
        cfg.q_sentinel.enable(region, n)?;

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

        // map rows 0..n
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

    /// Table selector of every join lookup that reads this view. It is enabled
    /// on exactly the view's real rows [0, n_base) and must stay a complex
    /// selector, since a simple one may not appear in a lookup expression.
    ///
    /// Without it the four columns (sorted_key, idx, sorted_val, sorted_eid)
    /// are a live lookup table on EVERY row of the domain, including the
    /// one-past-the-end sentinel row n_base and every row above it, none of
    /// which any constraint of this chip touches (perm, q_sort, q_idx0 and
    /// q_idx all stop at n_base). That is an unlimited supply of forged view
    /// tuples: a prover picks any (key, idx, val, eid) it likes on a free row
    /// and a bag row then "proves" a join it never had, which is a full count
    /// inflation with no change to any fixed column.
    pub q_tbl: Selector,

    q_sort: Selector,
    pub(crate) lt_key: LtConfig<F, NUM_BYTES>,
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

        let q_tbl = meta.complex_selector();

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
            q_tbl,
            q_sort,
            lt_key,
            iz_eq_key,
            idx,
            q_idx0,
            q_idx,
            iz_same_prev,
        }
    }

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
            // table side of the join lookups: exactly the rows the permutation
            // and the sort/idx recurrences cover, so the sentinel row `n` and
            // every row above it are NOT table entries
            cfg.q_tbl.enable(region, i)?;
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
    q_tbl: Selector,
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
        q_tbl: Selector,
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
    /// `g_sql3_obj_dp` builds one MapLookup replica per lane. All lanes are
    /// active on the same rows, so they share one `q_flag` / `q_complex` pair,
    /// and they share one u8 column so the number of 256-row `load` regions
    /// does not grow with the lane count.
    ///
    /// `probe_equality` decides whether the five probe columns join the
    /// permutation argument. Nothing in this circuit family ever copies them,
    /// so the replicated (per-lane) call passes `false`: halo2 pays for every
    /// column that has equality enabled, whether or not a copy constraint
    /// actually touches it, and that cost would otherwise scale with the lane
    /// count. `configure` passes `true` to leave the single-group layout of
    /// `g_sql3_obj` byte-for-byte as it was.
    #[allow(clippy::too_many_arguments)]
    pub fn configure_with(
        meta: &mut ConstraintSystem<F>,
        map_key: Column<Advice>,
        map_val: Column<Advice>,
        map_key_next: Column<Advice>,
        q_tbl: Selector,
        q_flag: Selector,
        q_complex: Selector,
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
/// Main circuit config (Triangle, Path+Closer)
/// ------------------------------
#[derive(Clone, Debug)]
pub struct TrianglePathCloserConfig<F: Field + Ord> {
    instance: Column<Instance>,

    // indexed views of the edge multiset
    in_by_dst: IndexedViewConfig<F>,  // key=dst, val=src
    out_by_src: IndexedViewConfig<F>, // key=src, val=dst

    // Bag1 rows: A->B->C
    t12_a: Column<Advice>,
    t12_b: Column<Advice>,
    t12_c: Column<Advice>,
    t12_i_r1: Column<Advice>,
    t12_j_r2: Column<Advice>,
    t12_r1_eid: Column<Advice>,
    t12_r2_eid: Column<Advice>,
    t12_real: Column<Advice>,
    q_t12_lookup: Selector,
    q_t12_key: Selector,

    // Bag2 rows: edge C->A (closer)
    t3_c: Column<Advice>,
    t3_a: Column<Advice>,
    t3_j_r3: Column<Advice>,
    t3_r3_eid: Column<Advice>,
    t3_real: Column<Advice>,
    q_t3_lookup: Selector,
    q_t3_msg_in: Selector,

    // message: msg_key=pack2(A,C), msg_val=count
    agg_msg: AggSumByKeyConfig<F>,
    msg_lookup: MapLookupConfig<F>,

    // ordering checks in Bag1: A<B and B<C
    q_order: Selector,
    lt_ab: LtConfig<F, NUM_BYTES>,
    lt_bc: LtConfig<F, NUM_BYTES>,

    // contrib and sum
    contrib: Column<Advice>,
    q_contrib: Selector,

    run_sum: Column<Advice>,
    q_sum0: Selector,
    q_sum: Selector,

    out: Column<Advice>,
    q_out: Selector,

    // ---------------- clean/residual partition of the two bags ----------------
    // bound clean indicator per bag row: 1 only on a row that passes the bag's
    // own predicate and went to R^c
    cln12: Column<Advice>,
    cln3: Column<Advice>,
    q_cln12_bind: Selector,
    q_cln3_bind: Selector,

    // `real == [the row carries a tuple at all]`: the two indicators that pin
    // the real bit of a bag row to that row's own attributes, so a padding row
    // cannot be promoted to a real input-channel tuple and a real row cannot be
    // demoted while keeping its attributes. Node ids are SHIFT_ID-shifted, so a
    // real tuple has B != 0 (Bag1) and C != 0 (Bag2) and only padding is zero.
    iz_t12_pad: IsZeroConfig<F>,
    iz_t3_pad: IsZeroConfig<F>,

    // Conservation Check (condition (1)) per bag: the bag rows carrying the
    // indicator are a permutation of [clean rows | residual rows | pad rows]
    part12: Vec<Column<Advice>>, // 8: (A,B,C,i_r1,j_r2,r1_eid,r2_eid,flag)
    part3: Vec<Column<Advice>>,  // 5: (C,A,j_r3,r3_eid,flag)
    perm_bag1: PermAnyConfig,
    perm_bag2: PermAnyConfig,
    q_cln_flag: Vec<Selector>, // [bag1, bag2] rows of R^c: flag == 1
    q_res_flag: Vec<Selector>, // [bag1, bag2] rows of R^r: flag == 0

    // ---------------- Pairwise Consistency (condition (3)) ----------------
    // packed (A,C) separator key of each partition group, pinned to that
    // group's own attribute columns
    pk12: Column<Advice>,
    pk3: Column<Advice>,
    q_pk12: Selector,
    q_pk3: Selector,

    // one complex selector per group, enabled over exactly the clean block of
    // that group. Each one is the input selector of one direction and the table
    // selector of the other, so no free advice is left to forge.
    q_pw_in_12: Selector,
    q_pw_in_3: Selector,

    // ---------------- Cardinality Preservation Check ----------------
    // condition (4): |Bag1^c |X| Bag2^c| == |Bag1 |X| Bag2|
    cp_agg_msg: CpAggConfig<F, NUM_BYTES>, // child Bag2, keyed by pack2(A,C)
    cp_join_msg: CpJoinConfig<F, NUM_BYTES>, // parent side, on the Bag1 rows
    cp_root: CpRootConfig,
    q_cp_mu: Selector, // the two root product gates
}

#[derive(Clone, Debug)]
pub struct TrianglePathCloserChip<F: Field + Ord> {
    cfg: TrianglePathCloserConfig<F>,
}
impl<F: Field + Ord> TrianglePathCloserChip<F> {
    pub fn construct(cfg: TrianglePathCloserConfig<F>) -> Self {
        Self { cfg }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> TrianglePathCloserConfig<F> {
        let instance = meta.instance_column();
        meta.enable_equality(instance);

        // Views
        let in_by_dst = IndexedViewChip::<F>::configure(meta);
        let out_by_src = IndexedViewChip::<F>::configure(meta);

        // Bag1 columns
        let t12_a = meta.advice_column();
        let t12_b = meta.advice_column();
        let t12_c = meta.advice_column();
        let t12_i_r1 = meta.advice_column();
        let t12_j_r2 = meta.advice_column();
        let t12_r1_eid = meta.advice_column();
        let t12_r2_eid = meta.advice_column();
        let t12_real = meta.advice_column();
        for c in [
            t12_a, t12_b, t12_c, t12_i_r1, t12_j_r2, t12_r1_eid, t12_r2_eid, t12_real,
        ] {
            meta.enable_equality(c);
        }
        let q_t12_lookup = meta.complex_selector();
        let q_t12_key = meta.selector();

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

        // ---------------- clean/residual partition of the two bags ----------------
        // Neither bag was partitioned in g_sql3_obj.rs, and condition (4) needs
        // two sides to compare, so the partition is introduced here. `cln12` /
        // `cln3` are the bound indicators on the bag rows: the gates below pin
        // them boolean and below the bag's own predicate, and the Conservation
        // Check pins them to the partition itself.
        let cln12 = meta.advice_column();
        let cln3 = meta.advice_column();
        meta.enable_equality(cln12);
        meta.enable_equality(cln3);
        let q_cln12_bind = meta.selector();
        let q_cln3_bind = meta.selector();

        // The real bit is the input-channel multiplicity of a bag row, so it
        // decides on its own whether the row enters the message table and the
        // input channel of condition (10). Boolean is not enough: neither real
        // bit is inside its bag's Conservation shuffle (the shuffled tuple is
        // (attributes, indicator)), so on a padding row it is free advice. With
        // t3_real = 1 on a padding row the gate "msg input from bag2 edges"
        // publishes in_key = pack2(0,0) = 0, the reserved dummy key, and with
        // t3_real = 0 on a REAL row that row's key leaves the message map and
        // the cp table at once, which is a load-bearing step of a count
        // inflation.
        //
        // Both are closed by pinning the bit to the row's own attributes. Node
        // ids are SHIFT_ID-shifted, so B (Bag1) and C (Bag2) are nonzero on
        // every real tuple and zero on the all-zero padding tuple, and the join
        // lookups only accept a zero key against the (0,0,0) dummy view row,
        // whose val and eid are zero too. So `real == [attr != 0]` is exactly
        // "this row carries a tuple", and it costs one advice column and one
        // degree-3 constraint per bag rather than a wider shuffle.
        let aux_t3_pad = meta.advice_column();
        let iz_t3_pad = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_cln3_bind),
            |m| m.query_advice(t3_c, Rotation::cur()),
            aux_t3_pad,
        );
        let aux_t12_pad = meta.advice_column();
        let iz_t12_pad = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_cln12_bind),
            |m| m.query_advice(t12_b, Rotation::cur()),
            aux_t12_pad,
        );

        meta.create_gate("bag2 clean indicator", |m| {
            let q = m.query_selector(q_cln3_bind);
            let cln = m.query_advice(cln3, Rotation::cur());
            let real = m.query_advice(t3_real, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                // t3_real is a multiplicity of the input channel now, so it has
                // to be boolean; g_sql3_obj.rs only ever used it as a factor
                q.clone() * real.clone() * (one.clone() - real.clone()),
                q.clone() * cln.clone() * (one.clone() - cln.clone()),
                // clean implies real, i.e. cln3 == t3_real * c_3
                q.clone() * cln * (one.clone() - real.clone()),
                // real == [C != 0]: padding cannot be promoted to a real tuple
                // at the reserved key, and a real tuple cannot be demoted
                q * (real - (one - iz_t3_pad.expr())),
            ]
        });

        // Conservation Check per bag: the bag rows, carrying the bound
        // indicator, are a permutation of [clean rows | residual rows | pad
        // rows]. The flag is a constant 1 over the clean rows and 0 over the
        // residual rows, so the multiset equality forces the indicator on a bag
        // row to mark exactly the tuples that went to R^c. Without it a prover
        // could mark a residual row clean and inflate the clean channel.
        let part12 = (0..8).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let part3 = (0..5).map(|_| meta.advice_column()).collect::<Vec<_>>();

        let q_perm12_in = meta.complex_selector();
        let q_perm12_out = meta.complex_selector();
        let perm_bag1 = PermAnyChip::configure(
            meta,
            q_perm12_in,
            q_perm12_out,
            vec![
                t12_a, t12_b, t12_c, t12_i_r1, t12_j_r2, t12_r1_eid, t12_r2_eid, cln12,
            ],
            part12.clone(),
        );

        let q_perm3_in = meta.complex_selector();
        let q_perm3_out = meta.complex_selector();
        let perm_bag2 = PermAnyChip::configure(
            meta,
            q_perm3_in,
            q_perm3_out,
            vec![t3_c, t3_a, t3_j_r3, t3_r3_eid, cln3],
            part3.clone(),
        );

        let q_cln_flag = (0..2).map(|_| meta.selector()).collect::<Vec<_>>();
        let q_res_flag = (0..2).map(|_| meta.selector()).collect::<Vec<_>>();
        for (idx, part) in [part12.clone(), part3.clone()].iter().enumerate() {
            let flag_col = *part.last().unwrap();
            let q_c = q_cln_flag[idx];
            let q_r = q_res_flag[idx];
            meta.create_gate("clean indicator on the partition side", move |m| {
                let qc = m.query_selector(q_c);
                let qr = m.query_selector(q_r);
                let f = m.query_advice(flag_col, Rotation::cur());
                vec![qc * (f.clone() - Expression::Constant(F::ONE)), qr * f]
            });
        }

        // ---------------- Pairwise Consistency (condition (3)) ----------------
        // pi_K(Bag1^c) == pi_K(Bag2^c) on the single edge of the cluster tree,
        // the packed (A,C) separator, as two mutual Membership Checks.
        //
        // Both sides are read off the CLEAN block of the two partition groups,
        // rows [0, n_cln), never off the bag rows. A lookup over the bag rows
        // would only certify membership in the whole relation R_i, which is the
        // weaker statement the message table already makes. A group's tuple
        // columns are tied to its bag by that group's Conservation Check and the
        // flag gate above pins flag == 1 on rows [0, n_cln) and 0 after them, so
        // the group's own packed key restricted to that block is exactly
        // pi_K(R^c).
        //
        // The key is composite, so it is packed with the same pack2 the rest of
        // the file uses; the packed column is derived from the group's own A and
        // C columns by one degree-2 gate rather than packed a second time by
        // hand.
        let pk12 = meta.advice_column();
        let pk3 = meta.advice_column();
        let q_pk12 = meta.selector();
        let q_pk3 = meta.selector();
        {
            let p_a = part12[0];
            let p_c = part12[2];
            meta.create_gate("pw: bag1 partition packed key", move |m| {
                let q = m.query_selector(q_pk12);
                let key = m.query_advice(p_a, Rotation::cur())
                    * Expression::Constant(F::from(PACK_SHIFT))
                    + m.query_advice(p_c, Rotation::cur());
                vec![q * (m.query_advice(pk12, Rotation::cur()) - key)]
            });
        }
        {
            let p_c = part3[0];
            let p_a = part3[1];
            meta.create_gate("pw: bag2 partition packed key", move |m| {
                let q = m.query_selector(q_pk3);
                let key = m.query_advice(p_a, Rotation::cur())
                    * Expression::Constant(F::from(PACK_SHIFT))
                    + m.query_advice(p_c, Rotation::cur());
                vec![q * (m.query_advice(pk3, Rotation::cur()) - key)]
            });
        }

        // One complex selector per group, enabled over exactly the clean block
        // [0, n_cln) of that group. Each one is the input selector of one
        // direction and the table selector of the other, so the two containments
        // hold between the two clean key columns themselves. An earlier version
        // routed each direction through an intermediate advice column holding
        // the deduplicated key set, but nothing in the circuit bound those
        // columns to the relation they claimed to enumerate: setting each table
        // to the key column that looks into it satisfies both lookups for an
        // arbitrary partition, which made condition (3) vacuous. There is no
        // free advice left here, so there is nothing to forge. The selectors
        // must stay complex, since a simple selector may not appear in a lookup
        // expression, so they are fresh rather than the q_cln_flag pair.
        //
        // A lookup input is 0 on every row where its selector is off, and the
        // table side is 0 on those rows too, so 0 is always in the table and the
        // gated-off rows cost nothing. Node IDs are SHIFT_ID-shifted, so a real
        // packed key is at least PACK_SHIFT + 1 and the containment is over the
        // real keys only.
        let q_pw_in_12 = meta.complex_selector();
        let q_pw_in_3 = meta.complex_selector();

        // pi_K(Bag1^c) subset of pi_K(Bag2^c)
        meta.lookup_any("pw: bag1^c key in bag2^c", |m| {
            let lhs = m.query_selector(q_pw_in_12) * m.query_advice(pk12, Rotation::cur());
            let rhs = m.query_selector(q_pw_in_3) * m.query_advice(pk3, Rotation::cur());
            vec![(lhs, rhs)]
        });

        // pi_K(Bag2^c) subset of pi_K(Bag1^c)
        meta.lookup_any("pw: bag2^c key in bag1^c", |m| {
            let lhs = m.query_selector(q_pw_in_3) * m.query_advice(pk3, Rotation::cur());
            let rhs = m.query_selector(q_pw_in_12) * m.query_advice(pk12, Rotation::cur());
            vec![(lhs, rhs)]
        });

        // Bag1 lookups:
        //
        // Every table side below is multiplied by the view's own `q_tbl`, which
        // is enabled on exactly the rows [0, n_base) the view's permutation,
        // sort and idx recurrences cover. An ungated table side would make the
        // sentinel row and every unconstrained row above it a live table entry,
        // i.e. free advice a bag row could join against. A lookup input is 0 on
        // every row where its own selector is off, and a table row with `q_tbl`
        // off contributes 0 on every column, so the all-zero tuple stays in the
        // table and the gated-off input rows still cost nothing.
        //
        // r1 via InByDst: key=B, idx=i_r1 -> val=A, eid=r1_eid
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

        // Bag2 lookup (r3) via OutBySrc: key=C, idx=j_r3 -> val=A, eid=r3_eid
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

        // Message aggregator
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

        // Message lookup per Bag1 row
        let msg_lookup = MapLookupChip::<F>::configure(
            meta,
            agg_msg.map_key,
            agg_msg.map_val,
            agg_msg.map_key_next,
            agg_msg.q_map_tbl,
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
        let q_order = meta.selector();
        let lt_ab = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_order) * m.query_advice(t12_real, Rotation::cur()),
            |m| m.query_advice(t12_a, Rotation::cur()),
            |m| m.query_advice(t12_b, Rotation::cur()),
        );
        let lt_bc = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_order) * m.query_advice(t12_real, Rotation::cur()),
            |m| m.query_advice(t12_b, Rotation::cur()),
            |m| m.query_advice(t12_c, Rotation::cur()),
        );

        // Bag1 side of the indicator binding: clean implies the row passes the
        // bag's own predicate, i.e. cln12 == t12_real * [A<B] * [B<C] * c_1.
        // This is what the filt/pad link gate does in the TPC-H files, where a
        // row dropped by the predicate pads its indicator with 0.
        meta.create_gate("bag1 clean indicator", |m| {
            let q = m.query_selector(q_cln12_bind);
            let cln = m.query_advice(cln12, Rotation::cur());
            let real = m.query_advice(t12_real, Rotation::cur());
            let ab = lt_ab.is_lt(m, None);
            let bc = lt_bc.is_lt(m, None);
            let one = Expression::Constant(F::ONE);
            vec![
                q.clone() * cln.clone() * (one.clone() - cln.clone()),
                q.clone() * cln.clone() * (one.clone() - real.clone()),
                q.clone() * cln.clone() * (one.clone() - ab),
                q.clone() * cln * (one.clone() - bc),
                // real == [B != 0], the Bag1 half of the pin described above
                q * (real - (one - iz_t12_pad.expr())),
            ]
        });

        // contrib = t12_real * msg_val * lt_ab * lt_bc
        let contrib = meta.advice_column();
        meta.enable_equality(contrib);
        let q_contrib = meta.selector();
        meta.create_gate("contrib gate", |m| {
            let q = m.query_selector(q_contrib);

            let real = m.query_advice(t12_real, Rotation::cur());
            let msgv = m.query_advice(msg_lookup.val, Rotation::cur());
            let ab = lt_ab.is_lt(m, None);
            let bc = lt_bc.is_lt(m, None);

            let outc = m.query_advice(contrib, Rotation::cur());
            vec![q * (outc - real * msgv * ab * bc)]
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

        // out equals run_sum at q_out row
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

        // ---------------- Cardinality Preservation Check (condition (4)) ----------------
        // One fixed column serves every Lt chip of the check, so the whole check
        // costs a single u8 range table and a single 256-row load region.
        let cp_u8 = meta.fixed_column();

        // Child side, over the Bag2 rows. Bag2 is a leaf of the cluster tree, so
        // its two multiplicity columns are columns the circuit already has: the
        // separator key and the input-channel multiplicity are agg_msg.in_key and
        // agg_msg.in_val, both pinned to the Bag2 attributes by the gate "msg
        // input from bag2 edges", and the clean channel is the bound indicator.
        let cp_agg_msg = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            agg_msg.in_key, // t3_real ? pack2(A,C) : PAD
            agg_msg.in_val, // t3_real
            cln3,
            PAD_U64,
        );

        // Parent side, on the rows of Bag1, keyed by the same packed separator
        // the message lookup probes with. The membership + gap witness of
        // MapLookupChip stays where it is; this is its second, two-channel copy,
        // which is what carries the clean sum up to the root.
        let cp_join_msg = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, msg_lookup.key);
        wire_cp_edge(meta, &cp_join_msg, &cp_agg_msg, msg_lookup.key);

        // Root multiplicities and the single equality that compares the two join
        // cardinalities. mu_all is degree 5, the same as the existing contrib
        // gate, so the circuit's degree does not rise; mu_cln is degree 3
        // because the bound indicator already carries the predicate factors.
        let cp_root = configure_cp_root::<F>(meta);
        let q_cp_mu = meta.selector();
        {
            let s_all = cp_join_msg.s_all;
            let s_cln = cp_join_msg.s_cln;
            let mu_all = cp_root.mu_all;
            let mu_cln = cp_root.mu_cln;
            meta.create_gate("cp: root multiplicities over bag1", move |m| {
                let q = m.query_selector(q_cp_mu);
                let real = m.query_advice(t12_real, Rotation::cur());
                let ab = lt_ab.is_lt(m, None);
                let bc = lt_bc.is_lt(m, None);
                let all = m.query_advice(mu_all, Rotation::cur())
                    - real * ab * bc * m.query_advice(s_all, Rotation::cur());
                let cln = m.query_advice(mu_cln, Rotation::cur())
                    - m.query_advice(cln12, Rotation::cur())
                        * m.query_advice(s_cln, Rotation::cur());
                vec![q.clone() * all, q * cln]
            });
        }

        TrianglePathCloserConfig {
            instance,
            in_by_dst,
            out_by_src,
            t12_a,
            t12_b,
            t12_c,
            t12_i_r1,
            t12_j_r2,
            t12_r1_eid,
            t12_r2_eid,
            t12_real,
            q_t12_lookup,
            q_t12_key,
            t3_c,
            t3_a,
            t3_j_r3,
            t3_r3_eid,
            t3_real,
            q_t3_lookup,
            q_t3_msg_in,
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
            cln12,
            cln3,
            q_cln12_bind,
            q_cln3_bind,
            iz_t12_pad,
            iz_t3_pad,
            part12,
            part3,
            perm_bag1,
            perm_bag2,
            q_cln_flag,
            q_res_flag,
            pk12,
            pk3,
            q_pk12,
            q_pk3,
            q_pw_in_12,
            q_pw_in_3,
            cp_agg_msg,
            cp_join_msg,
            cp_root,
            q_cp_mu,
        }
    }

    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        edges: Vec<Edge>,
        bag1_pad_extra: usize,
        bag2_pad_extra: usize,
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
        LtChip::<F, NUM_BYTES>::construct(cfg.cp_agg_msg.lt_key_cur_next).load(layouter)?;

        let in_view_chip = IndexedViewChip::<F>::construct(cfg.in_by_dst.clone());
        let out_view_chip = IndexedViewChip::<F>::construct(cfg.out_by_src.clone());
        let agg_msg_chip = AggSumByKeyChip::<F>::construct(cfg.agg_msg.clone());

        layouter.assign_region(
            || "triangle_path_closer witness",
            |mut region| {
                // -------------------
                // Host-side derivation (views, Bag1, Bag2, message map)
                // -------------------
                let derived = gq3_derive(&edges);
                let Gq3Derived {
                    ref in_rows,
                    ref out_rows,
                    ref t12,
                    ref t3,
                    ref msg_map,
                    ref keys,
                } = derived;
                let n_base = in_rows.len();

                in_view_chip.assign(&mut region, n_base, in_rows)?;
                out_view_chip.assign(&mut region, n_base, out_rows)?;

                // -------------------
                // Row counts of the two bags. The partition witness below is
                // computed before either bag is assigned, so both counts and
                // both padding schemes are settled here, unchanged.
                // -------------------
                let real3 = t3.len();
                let n3 = std::cmp::max(real3 + bag2_pad_extra, 1);
                let real12 = t12.len();
                let n12 = std::cmp::max(real12 + bag1_pad_extra, 1);

                // ---------------- clean / residual partition of the two bags ----------------
                // The honest clean part of a relation is its fully reduced
                // instance: the tuples that extend to a full join result. A Bag1
                // wedge contributes only when it is real and A<B<C, and a Bag1
                // wedge and a Bag2 closing edge extend each other exactly when
                // they agree on the packed separator key, so the reduction of
                // this two-node cluster tree is the intersection of the two key
                // sets.
                //
                // On a two-node tree that single simultaneous pass is already a
                // fixed point of semijoin reduction, which is what condition (3)
                // needs: both clean key sets come out as keys12 /\ keys3. On a
                // deeper tree it would not be, and the reduction would have to
                // be iterated.
                let pred12: Vec<bool> = (0..n12)
                    .map(|i| i < real12 && t12[i].0 < t12[i].1 && t12[i].1 < t12[i].2)
                    .collect();
                let key12: Vec<u64> = (0..n12)
                    .map(|i| {
                        if i < real12 {
                            pack2(t12[i].0, t12[i].2)
                        } else {
                            0
                        }
                    })
                    .collect();

                let keys3: HashSet<u64> = t3.iter().map(|&(c, a, _, _)| pack2(a, c)).collect();
                let keys12: HashSet<u64> =
                    (0..n12).filter(|&i| pred12[i]).map(|i| key12[i]).collect();

                let mut cln3: Vec<u64> = (0..n3)
                    .map(|i| (i < real3 && keys12.contains(&pack2(t3[i].1, t3[i].0))) as u64)
                    .collect();
                let mut cln12: Vec<u64> = (0..n12)
                    .map(|i| (pred12[i] && keys3.contains(&key12[i])) as u64)
                    .collect();

                // test hook only: hide one joinable Bag2 tuple in the residual
                // side and re-reduce both bags around it, so Conservation still
                // holds and the clean sides stay pairwise consistent
                let tamper = HIDE_ONE_CLEAN_TUPLE.load(Ordering::Relaxed);
                let mark_all = MARK_ALL_CLEAN.load(Ordering::Relaxed);
                if tamper {
                    if let Some(hidden) = (0..n3).find(|&i| cln3[i] == 1) {
                        cln3[hidden] = 0;
                        // iterate the reduction until nothing moves, so the two
                        // clean key sets are still equal and condition (3) has
                        // nothing to say about this witness
                        loop {
                            let mut moved = false;
                            let live3: HashSet<u64> = (0..n3)
                                .filter(|&i| cln3[i] == 1)
                                .map(|i| pack2(t3[i].1, t3[i].0))
                                .collect();
                            for i in 0..n12 {
                                if cln12[i] == 1 && !live3.contains(&key12[i]) {
                                    cln12[i] = 0;
                                    moved = true;
                                }
                            }
                            let live12: HashSet<u64> = (0..n12)
                                .filter(|&i| cln12[i] == 1)
                                .map(|i| key12[i])
                                .collect();
                            for i in 0..n3 {
                                if cln3[i] == 1 && !live12.contains(&pack2(t3[i].1, t3[i].0)) {
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

                // test hook only: skip the reduction and declare every real tuple
                // clean. The indicator still has to satisfy its binding gate, so
                // a Bag1 wedge its own predicate drops stays out; everything the
                // predicate keeps goes to the clean side and the residual side
                // holds nothing else. Conservation still holds and both channels
                // of condition (4) then carry the same multiplicity on every row,
                // so only Pairwise Consistency can reject this.
                if mark_all {
                    for i in 0..n3 {
                        cln3[i] = (i < real3) as u64;
                    }
                    for i in 0..n12 {
                        cln12[i] = pred12[i] as u64;
                    }
                }

                // -------------------
                // Assign Bag2 + build agg inputs
                // -------------------
                let mut agg_in: Vec<(u64, u64)> = vec![(PAD_U64, 0); n3];

                // witnesses of the two `real == [attr != 0]` pins
                let iz_t3_pad_chip = IsZeroChip::construct(cfg.iz_t3_pad.clone());
                let iz_t12_pad_chip = IsZeroChip::construct(cfg.iz_t12_pad.clone());

                for i in 0..n3 {
                    cfg.q_t3_lookup.enable(&mut region, i)?;
                    cfg.q_t3_msg_in.enable(&mut region, i)?;

                    // clean indicator and the input side of Bag2's Conservation
                    // Check, both over the same rows as the bag itself
                    cfg.q_cln3_bind.enable(&mut region, i)?;
                    cfg.perm_bag2.q_perm1.enable(&mut region, i)?;
                    cfg.perm_bag2.q_perm2.enable(&mut region, i)?;
                    region.assign_advice(
                        || "cln3",
                        cfg.cln3,
                        i,
                        || Value::known(F::from(cln3[i])),
                    )?;
                    iz_t3_pad_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(if i < real3 { t3[i].0 } else { 0 })),
                    )?;

                    if i < real3 {
                        let (c, a, j_r3, r3_eid) = t3[i];

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

                // Aggregate msg table inside circuit
                let _emitted = agg_msg_chip.assign(&mut region, n3, &agg_in)?;

                // -------------------
                // Assign Bag1 + message lookup + ordering + sum
                // -------------------
                // Debug print, silenced: `assign` runs during keygen as well as
                // proving, so this fired several times per measured row.
                // println!("The length of n12 is: {}", t12.len());

                let lt_ab_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_ab.clone());
                let lt_bc_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_bc.clone());
                let lt_low_chip = LtChip::<F, NUM_BYTES>::construct(cfg.msg_lookup.lt_low.clone());
                let lt_high_chip =
                    LtChip::<F, NUM_BYTES>::construct(cfg.msg_lookup.lt_high.clone());

                let mut running: u64 = 0;

                for i in 0..n12 {
                    cfg.q_t12_lookup.enable(&mut region, i)?;
                    cfg.q_t12_key.enable(&mut region, i)?;
                    cfg.msg_lookup.q_flag.enable(&mut region, i)?;
                    cfg.msg_lookup.q_complex.enable(&mut region, i)?;
                    cfg.q_order.enable(&mut region, i)?;
                    cfg.q_contrib.enable(&mut region, i)?;

                    // clean indicator, the input side of Bag1's Conservation
                    // Check and the two root multiplicities of condition (4)
                    cfg.q_cln12_bind.enable(&mut region, i)?;
                    cfg.perm_bag1.q_perm1.enable(&mut region, i)?;
                    cfg.perm_bag1.q_perm2.enable(&mut region, i)?;
                    cfg.q_cp_mu.enable(&mut region, i)?;
                    region.assign_advice(
                        || "cln12",
                        cfg.cln12,
                        i,
                        || Value::known(F::from(cln12[i])),
                    )?;

                    let (a, b, c, i_r1, j_r2, r1_eid, r2_eid, real) = if i < real12 {
                        let (a, b, c, i_r1, j_r2, r1_eid, r2_eid) = t12[i];
                        (a, b, c, i_r1, j_r2, r1_eid, r2_eid, 1u64)
                    } else {
                        (0, 0, 0, 0, 0, 0, 0, 0u64)
                    };

                    region.assign_advice(|| "t12_a", cfg.t12_a, i, || Value::known(F::from(a)))?;
                    region.assign_advice(|| "t12_b", cfg.t12_b, i, || Value::known(F::from(b)))?;
                    region.assign_advice(|| "t12_c", cfg.t12_c, i, || Value::known(F::from(c)))?;
                    region.assign_advice(
                        || "t12_i_r1",
                        cfg.t12_i_r1,
                        i,
                        || Value::known(F::from(i_r1)),
                    )?;
                    region.assign_advice(
                        || "t12_j_r2",
                        cfg.t12_j_r2,
                        i,
                        || Value::known(F::from(j_r2)),
                    )?;
                    region.assign_advice(
                        || "t12_r1_eid",
                        cfg.t12_r1_eid,
                        i,
                        || Value::known(F::from(r1_eid)),
                    )?;
                    region.assign_advice(
                        || "t12_r2_eid",
                        cfg.t12_r2_eid,
                        i,
                        || Value::known(F::from(r2_eid)),
                    )?;
                    region.assign_advice(
                        || "t12_real",
                        cfg.t12_real,
                        i,
                        || Value::known(F::from(real)),
                    )?;
                    iz_t12_pad_chip.assign(&mut region, i, Value::known(F::from(b)))?;

                    // LT witnesses (constraints disabled when real=0)
                    lt_ab_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(a)),
                        Value::known(F::from(b)),
                    )?;
                    lt_bc_chip.assign(
                        &mut region,
                        i,
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

                    let ab = if a < b { 1u64 } else { 0u64 };
                    let bc = if b < c { 1u64 } else { 0u64 };
                    let contrib_u64 = real * val * ab * bc;

                    region.assign_advice(
                        || "contrib",
                        cfg.contrib,
                        i,
                        || Value::known(F::from(contrib_u64)),
                    )?;

                    if i == 0 {
                        cfg.q_sum0.enable(&mut region, i)?;
                        running = contrib_u64;
                    } else {
                        cfg.q_sum.enable(&mut region, i)?;
                        // Completeness only, not soundness: the "sum" gate is
                        // over F, so a prefix sum that overflows u64 here would
                        // simply fail to synthesize rather than be accepted. It
                        // takes more than 2^64 triangles to reach, so no honest
                        // witness can hit it.
                        running = running.wrapping_add(contrib_u64);
                    }
                    region.assign_advice(
                        || "run_sum",
                        cfg.run_sum,
                        i,
                        || Value::known(F::from(running)),
                    )?;
                }

                // ---- Conservation Check: the partition side of both bags ----
                // [clean rows | residual rows | pad rows]. The pad rows are the
                // all-zero tuple the bags already pad with, and SHIFT_ID keeps 0
                // out of the real tuples, so the two multisets agree exactly.
                let mut part12_rows: Vec<[u64; 8]> = Vec::with_capacity(n12);
                for i in 0..real12 {
                    if cln12[i] == 1 {
                        let (a, b, c, i_r1, j_r2, r1_eid, r2_eid) = t12[i];
                        part12_rows.push([a, b, c, i_r1, j_r2, r1_eid, r2_eid, 1]);
                    }
                }
                let n_cln12 = part12_rows.len();
                for i in 0..real12 {
                    if cln12[i] == 0 {
                        let (a, b, c, i_r1, j_r2, r1_eid, r2_eid) = t12[i];
                        part12_rows.push([a, b, c, i_r1, j_r2, r1_eid, r2_eid, 0]);
                    }
                }
                let n_res12 = part12_rows.len() - n_cln12;
                while part12_rows.len() < n12 {
                    part12_rows.push([0u64; 8]);
                }

                let mut part3_rows: Vec<[u64; 5]> = Vec::with_capacity(n3);
                for i in 0..real3 {
                    if cln3[i] == 1 {
                        let (c, a, j_r3, r3_eid) = t3[i];
                        part3_rows.push([c, a, j_r3, r3_eid, 1]);
                    }
                }
                let n_cln3 = part3_rows.len();
                for i in 0..real3 {
                    if cln3[i] == 0 {
                        let (c, a, j_r3, r3_eid) = t3[i];
                        part3_rows.push([c, a, j_r3, r3_eid, 0]);
                    }
                }
                let n_res3 = part3_rows.len() - n_cln3;
                while part3_rows.len() < n3 {
                    part3_rows.push([0u64; 5]);
                }

                for i in 0..n12 {
                    for j in 0..8 {
                        region.assign_advice(
                            || "part12",
                            cfg.part12[j],
                            i,
                            || Value::known(F::from(part12_rows[i][j])),
                        )?;
                    }
                }
                for i in 0..n3 {
                    for j in 0..5 {
                        region.assign_advice(
                            || "part3",
                            cfg.part3[j],
                            i,
                            || Value::known(F::from(part3_rows[i][j])),
                        )?;
                    }
                }

                // KNOWN LIMITATION, not a per-proof soundness hole. The four
                // block extents below are witness-derived and appear in the
                // circuit only as selector enable ranges, i.e. as fixed columns
                // materialized at keygen. Under the standard soundness game the
                // vk is fixed, so a prover cannot move or shrink them; what they
                // do cost is (i) a trusted-keygen dependency, since a vk
                // generated from tampered data certifies nothing, and (ii)
                // obliviousness, since n_cln12 and n_cln3 are exactly |Bag1^c|
                // and |Bag2^c| and so leak the clean/residual ratio the paper
                // claims never to reveal. Both need the extents to become
                // public inputs or padded to a data-independent bound, which is
                // an instance-vector change outside this file.
                for i in 0..n_cln12 {
                    cfg.q_cln_flag[0].enable(&mut region, i)?;
                }
                for i in n_cln12..(n_cln12 + n_res12) {
                    cfg.q_res_flag[0].enable(&mut region, i)?;
                }
                for i in 0..n_cln3 {
                    cfg.q_cln_flag[1].enable(&mut region, i)?;
                }
                for i in n_cln3..(n_cln3 + n_res3) {
                    cfg.q_res_flag[1].enable(&mut region, i)?;
                }

                // ===================== PAIRWISE CONSISTENCY =====================
                // condition (3): pi_K(Bag1^c) == pi_K(Bag2^c) on the packed (A,C)
                // separator, as two mutual Membership Checks over the clean block
                // of the two partition groups.
                //
                // The packed key of a group is derived from that group's own A
                // and C columns by the gates "pw: bag1/bag2 partition packed
                // key", enabled on every row of the group, so it is pinned on the
                // clean rows and defined everywhere else.
                for i in 0..n12 {
                    cfg.q_pk12.enable(&mut region, i)?;
                    region.assign_advice(
                        || "pk12",
                        cfg.pk12,
                        i,
                        || Value::known(F::from(pack2(part12_rows[i][0], part12_rows[i][2]))),
                    )?;
                }
                for i in 0..n3 {
                    cfg.q_pk3.enable(&mut region, i)?;
                    region.assign_advice(
                        || "pk3",
                        cfg.pk3,
                        i,
                        || Value::known(F::from(pack2(part3_rows[i][1], part3_rows[i][0]))),
                    )?;
                }

                // Rows [0, n_cln) of each group are exactly pi_K(R^c), because
                // the flag gate pins flag == 1 there and the group's tuple
                // columns are tied to the bag by its Conservation Check. One
                // selector per group serves as the input selector of its own
                // direction and as the table selector of the other, so the two
                // containments run directly between the two clean key columns.
                for i in 0..n_cln12 {
                    cfg.q_pw_in_12.enable(&mut region, i)?;
                }
                for i in 0..n_cln3 {
                    cfg.q_pw_in_3.enable(&mut region, i)?;
                }

                // ===================== CARDINALITY PRESERVATION CHECK =====================
                // condition (4) of the One-Pass OBJ: the two multiplicity
                // channels are propagated over the cluster tree and their root
                // sums compared.
                //
                // Bag2 is a leaf, so a Bag2 row's input-channel multiplicity is
                // its real bit and its clean-channel multiplicity is the bound
                // indicator. Both columns are already assigned above, and the
                // key column is the same agg_msg.in_key the message table sorts.
                let cp_rows: Vec<[u64; 3]> = (0..n3)
                    .map(|i| [agg_in[i].0, agg_in[i].1, cln3[i]])
                    .collect();
                let cp_stage = build_cp_stage(&cp_rows, PAD_U64);
                assign_cp_agg(&mut region, &cfg.cp_agg_msg, &cp_rows, &cp_stage)?;

                // parent side, on the rows of Bag1
                let fetched =
                    assign_cp_join(&mut region, &cfg.cp_join_msg, &key12, &cp_stage, PAD_U64)?;

                // root multiplicities and the equality between the two sums
                let cp_mu: Vec<(u64, u64)> = (0..n12)
                    .map(|i| ((pred12[i] as u64) * fetched[i].0, cln12[i] * fetched[i].1))
                    .collect();
                let (cp_all, cp_cln) = assign_cp_root(&mut region, &cfg.cp_root, &cp_mu)?;
                if !tamper && !mark_all {
                    debug_assert_eq!(
                        cp_all, cp_cln,
                        "cardinality preservation: |R^c join| != |R join|"
                    );
                }

                let out_row = n12 - 1;
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

/// Wrapper circuit
pub struct MyCircuit<F: Field + Ord> {
    pub edges: Vec<Edge>,
    pub bag1_pad_extra: usize,
    pub bag2_pad_extra: usize,
    pub _marker: PhantomData<F>,
}
impl<F: Field + Ord> Default for MyCircuit<F> {
    fn default() -> Self {
        Self {
            edges: vec![],
            bag1_pad_extra: 0,
            bag2_pad_extra: 0,
            _marker: PhantomData,
        }
    }
}

impl<F: Field + Ord> Circuit<F> for MyCircuit<F> {
    type Config = TrianglePathCloserConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        TrianglePathCloserChip::<F>::configure(meta)
    }

    fn synthesize(&self, cfg: Self::Config, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        let chip = TrianglePathCloserChip::<F>::construct(cfg);
        let out = chip.assign(
            &mut layouter,
            self.edges.clone(),
            self.bag1_pad_extra,
            self.bag2_pad_extra,
        )?;
        chip.expose_public(&mut layouter, out, 0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::{MockProver, VerifyFailure};
    use std::sync::atomic::Ordering;

    use halo2_proofs::{
        plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit},
        poly::{
            commitment::{Params, ParamsProver},
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

    fn generate_para(k: u32) {
        // Time to generate parameters
        // let params_time_start = Instant::now();

        // Note: Ensure `use halo2_proofs::poly::commitment::ParamsProver;` is in your imports for .new()
        let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(k);

        let params_path = &crate::paths::param_file(22);
        let mut fd = std::fs::File::create(params_path).unwrap();
        params.write(&mut fd).unwrap();

        // println!("Time to generate params {:?}", params_time);
    }

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

        // Enumerate A->B, B->C, C->A with A<B<C
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
                        if let Some(&cca) = cnt.get(&(c, a)) {
                            total += (cab as u128) * (cbc as u128) * (cca as u128);
                        }
                    }
                }
            }
        }

        total as u64
    }

    // #[test]
    // fn test1() {
    //     let k = 22;

    //     generate_para(k);
    // }

    #[test]
    #[ignore = "inherited heavy end-to-end proof; the fast check is test_cardinality_preservation"]
    fn test() {
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
        let edges = crate::bench_queries::load_graph(&dataset);

        let cnt = expected_cnt(&edges);

        let (bag1_pad_extra, bag2_pad_extra) =
            crate::bench_queries::graph_pads("gq3", &dataset, &edges, privacy);
        println!(
            "[gq3 test] dataset={} privacy={} pads: bag1={} bag2={}",
            dataset,
            privacy.label(),
            bag1_pad_extra,
            bag2_pad_extra
        );

        let circuit = MyCircuit::<Fp> {
            edges,
            bag1_pad_extra,
            bag2_pad_extra,
            _marker: PhantomData,
        };
        let public_input = vec![Fp::from(cnt)];
        let k = crate::bench_queries::degree_for("gq3", &dataset, privacy);

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
            let proof_path = &crate::paths::proof_file("last_proof_q3_dp");
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
    }

    /// Cost probe. The soundness patches of this file are all degree-1 or
    /// degree-2 gates and one table selector, so none of them may raise
    /// `cs.degree()`: a rise would double every FFT of the proof.
    #[test]
    fn test_max_gate_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        let mut cs = ConstraintSystem::<Fp>::default();
        let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
        println!(
            "advice={} fixed={} sel={} deg={} gates={} polys={} lookups={} shuffles={}",
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
        );
        assert!(
            cs.degree() <= 7,
            "the maximum gate degree rose to {}, so a soundness patch is costing \
             more than it should",
            cs.degree()
        );
    }

    /// Fast correctness check of the Cardinality Preservation Check: a truncated
    /// slice of a real graph under MockProver, which verifies every gate,
    /// shuffle and lookup of the circuit without paying for a real proof.
    ///
    /// The slice is taken from `wiki`, the one shipped dataset that is a
    /// directed graph: `lastfm` and `facebook` store every edge in one
    /// direction only, so they contain no closing edge C->A at all and the whole
    /// clean instance would be empty, which would make both directions of this
    /// test vacuous.
    #[test]
    fn test_cardinality_preservation() {
        const N_EDGES: usize = 3000;
        let k = 14;

        let mut edges = crate::bench_queries::load_graph("wiki");
        assert!(
            edges.len() >= N_EDGES,
            "graph dataset not found or too small: {} edges",
            edges.len()
        );
        edges.truncate(N_EDGES);

        let cnt = expected_cnt(&edges);
        assert!(
            cnt > 0,
            "the slice has no triangle, so the clean instance is empty and the test is vacuous"
        );

        // Non-vacuity of the third direction below: the slice must contain a
        // dangling tuple, i.e. the two unreduced key sets on the separator must
        // differ, otherwise the all-clean partition really is pairwise
        // consistent and condition (3) would be right to accept it.
        let derived = gq3_derive(&edges);
        let keys3: HashSet<u64> = derived.t3.iter().map(|&(c, a, _, _)| pack2(a, c)).collect();
        let keys12: HashSet<u64> = derived
            .t12
            .iter()
            .filter(|&&(a, b, c, _, _, _, _)| a < b && b < c)
            .map(|&(a, _, c, _, _, _, _)| pack2(a, c))
            .collect();
        let dangling12 = keys12.difference(&keys3).count();
        let dangling3 = keys3.difference(&keys12).count();
        println!(
            "[gq3 pw] separator keys: bag1={} bag2={} bag1-only={} bag2-only={}",
            keys12.len(),
            keys3.len(),
            dangling12,
            dangling3
        );
        assert!(
            dangling12 + dangling3 > 0,
            "the slice has no dangling tuple, so the all-clean partition is pairwise \
             consistent and the third direction would pass vacuously"
        );

        // a few oblivious pad rows on both bags, so the padding path of the new
        // columns is exercised too
        let circuit = MyCircuit::<Fp> {
            edges,
            bag1_pad_extra: 5,
            bag2_pad_extra: 3,
            _marker: PhantomData,
        };
        let public_input = vec![Fp::from(cnt)];

        let prover = MockProver::run(k, &circuit, vec![public_input.clone()]).unwrap();
        prover.assert_satisfied();

        // Negative direction: the same witness with one joinable Bag2 tuple
        // hidden in the residual side and both bags re-reduced around it, so
        // Conservation still holds and the two clean sides still agree on the
        // separator key. Only condition (4) can see this, so the circuit must
        // now reject.
        super::HIDE_ONE_CLEAN_TUPLE.store(true, Ordering::Relaxed);
        let tampered = MockProver::run(k, &circuit, vec![public_input.clone()]).unwrap();
        let verdict = tampered.verify();
        super::HIDE_ONE_CLEAN_TUPLE.store(false, Ordering::Relaxed);

        let failures = verdict.expect_err("condition (4) accepted a hidden joinable tuple");
        assert!(
            failures
                .iter()
                .any(|f| format!("{:?}", f).contains("cardinality preservation")),
            "the circuit rejected, but not through the Cardinality Preservation Check: {:?}",
            failures
        );

        // Third direction: the escape condition (3) closes. With every real tuple
        // declared clean the partition still conserves both bags, and both
        // channels of the Cardinality Preservation Check then compute the same
        // number on every Bag1 row, so sum_cln == sum_all holds for free. Nothing
        // but Pairwise Consistency notices that the clean side is not the reduced
        // instance, so the circuit must reject through a "pw: " lookup.
        super::MARK_ALL_CLEAN.store(true, Ordering::Relaxed);
        let all_clean = MockProver::run(k, &circuit, vec![public_input]).unwrap();
        let verdict = all_clean.verify();
        super::MARK_ALL_CLEAN.store(false, Ordering::Relaxed);

        let failures = verdict.expect_err("the all-clean partition was accepted");
        assert!(
            failures.iter().any(|f| matches!(
                f,
                VerifyFailure::Lookup { name, .. } if name.starts_with("pw: ")
            )),
            "the circuit rejected the all-clean partition, but not through a Pairwise \
             Consistency lookup: {:?}",
            failures
        );
    }
}
