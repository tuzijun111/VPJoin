
//! 3-cycle (triangle) COUNT(*) with ordering (a<b<c) using 2-bag decomposition:
//!
//! Query:
//!   SELECT COUNT(*) AS cnt
//!   FROM R1 r1
//!   JOIN R1 r2 ON r1.dst = r2.src
//!   JOIN R1 r3 ON r2.dst = r3.src AND r3.dst = r1.src
//!   WHERE r1.src < r2.src AND r2.src < r3.src;
//!
//! Variables:
//!   a = r1.src = r3.dst
//!   b = r1.dst = r2.src
//!   c = r2.dst = r3.src
//!
//! Bags (share the SAME r2 tuple identity):
//!   Bag1: {r1, r2} materialize J12 = r1 ⋈ r2 on b  -> rows (a,b,c,r2_eid)
//!   Bag2: {r2, r3} materialize J23 = r2 ⋈ r3 on c  -> rows (b,c,a,r2_eid)
//!   Separator is r2_eid (row identity for r2). This avoids squaring multiplicities when duplicates exist.
//!
//! DP/message from Bag2 to Bag1:
//!   msg(r2_eid, a) = # of Bag2 rows with that (r2_eid,a)  (i.e., #choices of r3 for fixed r2 and a)
//!   answer = Σ_{(a,b,c,r2_eid) in J12} msg(r2_eid,a) * [a<b] * [b<c]
//!
//! Important:
//! - We do NOT take J12/J23 as inputs; we materialize them inside the circuit using lookups into
//!   sorted+indexed edge tables (like the 5-cycle style).
//! - This file depends on your existing chips:
//!     crate::chips::is_zero::{IsZeroChip, IsZeroConfig}
//!     crate::chips::less_than::{LtChip, LtConfig, LtInstruction}
//!     crate::chips::permutation_any::{PermAnyChip, PermAnyConfig}

use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};

use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;

const NUM_BYTES: usize = 8;
const PAD_U64: u64 = u64::MAX;

// shift node IDs by +1 so 0 is reserved if you want it
const SHIFT_ID: u64 = 1;

// pack two shifted values into one u64 key (assumes each < 2^PACK_BITS)
const PACK_BITS: u32 = 21;
const PACK_SHIFT: u64 = 1u64 << PACK_BITS;
fn pack2(hi: u64, lo: u64) -> u64 {
    hi * PACK_SHIFT + lo
}

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub src: u64,
    pub dst: u64,
}

/// ------------------------------
/// AggSumByKey: group-by SUM(val) over key
/// (used for counting: feed val=1)
/// ------------------------------
#[derive(Clone, Debug)]
pub struct AggSumByKeyConfig<F: Field + Ord> {
    in_key: Column<Advice>,
    in_val: Column<Advice>,

    sorted_key: Column<Advice>,
    sorted_val: Column<Advice>,
    perm_sort: PermAnyConfig,

    q_sort: Selector,
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

    // map for membership+value lookup
    map_key: Column<Advice>,
    map_val: Column<Advice>,
    map_key_next: Column<Advice>,
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
            vec![q * (m.query_advice(run_sum, Rotation::cur()) - m.query_advice(sorted_val, Rotation::cur()))]
        });
        meta.create_gate("agg run_accu", |m| {
            let q = m.query_selector(q_accu);
            let same = iz_same_prev.expr();
            let rs_cur = m.query_advice(run_sum, Rotation::cur());
            let rs_prev = m.query_advice(run_sum, Rotation::prev());
            let v = m.query_advice(sorted_val, Rotation::cur());
            vec![q * (rs_cur - (same * rs_prev + v))]
        });

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
            |m| m.query_advice(out_key, Rotation::next()) - m.query_advice(out_key, Rotation::cur()),
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
                    * (m.query_advice(map_key, Rotation::next()) - m.query_advice(out_key, Rotation::cur())),
                q * (m.query_advice(map_val, Rotation::next()) - m.query_advice(out_sum, Rotation::cur())),
            ]
        });

        meta.create_gate("map key_next = next(map_key)", |m| {
            let q = m.query_selector(q_map_shift);
            vec![q * (m.query_advice(map_key_next, Rotation::cur()) - m.query_advice(map_key, Rotation::next()))]
        });

        meta.create_gate("map last key_next=PAD", |m| {
            let q = m.query_selector(q_map_last);
            vec![q * (m.query_advice(map_key_next, Rotation::cur()) - Expression::Constant(F::from(PAD_U64)))]
        });

        AggSumByKeyConfig {
            in_key,
            in_val,
            sorted_key,
            sorted_val,
            perm_sort,
            q_sort,
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
            q_map_first,
            q_map_link,
            q_map_shift,
            q_map_last,
        }
    }

    /// Assign full agg + map. Input rows length n.
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
            region.assign_advice(|| "in_key", cfg.in_key, i, || Value::known(F::from(rows[i].0)))?;
            region.assign_advice(|| "in_val", cfg.in_val, i, || Value::known(F::from(rows[i].1)))?;
            region.assign_advice(|| "sorted_key", cfg.sorted_key, i, || Value::known(F::from(sorted_ext[i].0)))?;
            region.assign_advice(|| "sorted_val", cfg.sorted_val, i, || Value::known(F::from(sorted_ext[i].1)))?;
        }
        // sentinel row n
        region.assign_advice(|| "sorted_key_s", cfg.sorted_key, n, || Value::known(F::from(sorted_ext[n].0)))?;
        region.assign_advice(|| "sorted_val_s", cfg.sorted_val, n, || Value::known(F::from(sorted_ext[n].1)))?;

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
            region.assign_advice(|| "run_sum", cfg.run_sum, i, || Value::known(F::from(run_sum_u64[i])))?;

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
            region.assign_advice(|| "out_key", cfg.out_key, i, || Value::known(F::from(out[i].0)))?;
            region.assign_advice(|| "out_sum", cfg.out_sum, i, || Value::known(F::from(out[i].1)))?;
        }
        region.assign_advice(|| "out_key_s", cfg.out_key, n, || Value::known(F::from(PAD_U64)))?;
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

        // map table: n+1 rows
        cfg.q_map_first.enable(region, 0)?;
        region.assign_advice(|| "map_key0", cfg.map_key, 0, || Value::known(F::ZERO))?;
        region.assign_advice(|| "map_val0", cfg.map_val, 0, || Value::known(F::ZERO))?;
        // key_next[0] = map_key[1] enforced by q_map_shift@0
        let first_key = out.get(0).map(|x| x.0).unwrap_or(PAD_U64);
        region.assign_advice(|| "map_kn0", cfg.map_key_next, 0, || Value::known(F::from(first_key)))?;
        cfg.q_map_shift.enable(region, 0)?;

        for i in 0..n {
            cfg.q_map_link.enable(region, i)?;
            region.assign_advice(|| "map_key", cfg.map_key, i + 1, || Value::known(F::from(out[i].0)))?;
            region.assign_advice(|| "map_val", cfg.map_val, i + 1, || Value::known(F::from(out[i].1)))?;

            let nextk = if i + 1 < n { out[i + 1].0 } else { PAD_U64 };
            region.assign_advice(|| "map_kn", cfg.map_key_next, i + 1, || Value::known(F::from(nextk)))?;
            cfg.q_map_shift.enable(region, i + 1)?; // constrains map_key_next[i+1] = map_key[i+2]
        }
        cfg.q_map_last.enable(region, n)?;

        Ok(emitted)
    }
}

/// ------------------------------
/// LookupValue in map (membership+gap), like your Q5/Q-cycle pattern:
/// - if in_set=1: (key,val) must be in (map_key,map_val)
/// - if in_set=0: (low,high) must be adjacent pair in (map_key,map_key_next) and val=0
/// ------------------------------
#[derive(Clone, Debug)]
pub struct MapLookupConfig<F: Field + Ord> {
    q_flag: Selector,
    q_complex: Selector,

    key: Column<Advice>,
    in_set: Column<Advice>,
    low: Column<Advice>,
    high: Column<Advice>,
    val: Column<Advice>,

    lt_low: LtConfig<F, NUM_BYTES>,
    lt_high: LtConfig<F, NUM_BYTES>,

    map_key: Column<Advice>,
    map_val: Column<Advice>,
    map_key_next: Column<Advice>,
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
    ) -> MapLookupConfig<F> {
        let q_flag = meta.selector();
        let q_complex = meta.complex_selector();

        let key = meta.advice_column();
        let in_set = meta.advice_column();
        let low = meta.advice_column();
        let high = meta.advice_column();
        let val = meta.advice_column();
        for c in [key, in_set, low, high, val] {
            meta.enable_equality(c);
        }

        let lt_low = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| {
                let q = m.query_selector(q_flag);
                let inside = m.query_advice(in_set, Rotation::cur());
                q * (Expression::Constant(F::ONE) - inside)
            },
            |m| m.query_advice(low, Rotation::cur()),
            |m| m.query_advice(key, Rotation::cur()),
        );
        let lt_high = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| {
                let q = m.query_selector(q_flag);
                let inside = m.query_advice(in_set, Rotation::cur());
                q * (Expression::Constant(F::ONE) - inside)
            },
            |m| m.query_advice(key, Rotation::cur()),
            |m| m.query_advice(high, Rotation::cur()),
        );

        meta.lookup_any("map member", |m| {
            let q = m.query_selector(q_complex);
            let inside = m.query_advice(in_set, Rotation::cur());
            vec![
                (
                    q.clone() * inside.clone() * m.query_advice(key, Rotation::cur()),
                    m.query_advice(map_key, Rotation::cur()),
                ),
                (
                    q * inside * m.query_advice(val, Rotation::cur()),
                    m.query_advice(map_val, Rotation::cur()),
                ),
            ]
        });

        meta.lookup_any("map gap", |m| {
            let q = m.query_selector(q_complex);
            let inside = m.query_advice(in_set, Rotation::cur());
            let gate = q * (Expression::Constant(F::ONE) - inside);
            vec![
                (
                    gate.clone() * m.query_advice(low, Rotation::cur()),
                    m.query_advice(map_key, Rotation::cur()),
                ),
                (
                    gate * m.query_advice(high, Rotation::cur()),
                    m.query_advice(map_key_next, Rotation::cur()),
                ),
            ]
        });

        meta.create_gate("map lookup correctness", |m| {
            let q = m.query_selector(q_flag);
            let inside = m.query_advice(in_set, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            let low_ok = lt_low.is_lt(m, None);
            let high_ok = lt_high.is_lt(m, None);

            vec![
                // boolean
                q.clone() * inside.clone() * (one.clone() - inside.clone()),
                // gap strictness when missing
                q.clone() * (one.clone() - inside.clone()) * (one.clone() - low_ok),
                q.clone() * (one.clone() - inside.clone()) * (one.clone() - high_ok),
                // if missing => val = 0
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
        }
    }
}

/// ------------------------------
/// Indexed edge table view:
/// Sort by `key` (and by eid to make deterministic), with idx within each key group.
/// We store columns: sorted_key, sorted_val, sorted_eid, idx
/// and use permutation to prove it's a reordering of base tuples.
/// ------------------------------
#[derive(Clone, Debug)]
pub struct IndexedViewConfig<F: Field + Ord> {
    // input tuple (key,val,eid) copied from base table
    in_key: Column<Advice>,
    in_val: Column<Advice>,
    in_eid: Column<Advice>,

    // sorted tuple
    sorted_key: Column<Advice>,
    sorted_val: Column<Advice>,
    sorted_eid: Column<Advice>,
    perm: PermAnyConfig,

    // key nondecreasing
    q_sort: Selector,
    lt_key: LtConfig<F, NUM_BYTES>,
    iz_eq_key: IsZeroConfig<F>,

    // idx within group
    idx: Column<Advice>,
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
            |m| m.query_advice(sorted_key, Rotation::next()) - m.query_advice(sorted_key, Rotation::cur()),
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
            |m| m.query_advice(sorted_key, Rotation::cur()) - m.query_advice(sorted_key, Rotation::prev()),
            aux_same,
        );
        meta.create_gate("idx0=0", |m| {
            let q = m.query_selector(q_idx0);
            vec![q * m.query_advice(idx, Rotation::cur())]
        });
        meta.create_gate("idx recurrence", |m| {
            let q = m.query_selector(q_idx);
            let same = iz_same_prev.expr(); // 1 if same key
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
}

/// ------------------------------
/// Main circuit config
/// ------------------------------
#[derive(Clone, Debug)]
pub struct Triangle2BagConfig<F: Field + Ord> {
    instance: Column<Instance>,

    // base table E: (eid, src, dst)
    e_eid: Column<Advice>,
    e_src: Column<Advice>,
    e_dst: Column<Advice>,

    // copy gates into views
    q_copy_in_by_dst: Selector,
    q_copy_out_by_src: Selector,

    // view1: InByDst keyed by dst: (key=dst, val=src, eid)
    in_by_dst: IndexedViewConfig<F>,
    // view2: OutBySrc keyed by src: (key=src, val=dst, eid)
    out_by_src: IndexedViewConfig<F>,

    // Bag1 join rows: (a,b,c,r2_eid) proven by lookups:
    //   r1: (a->b) comes from InByDst at key=b
    //   r2: (b->c) comes from OutBySrc at key=b (we keep its eid as r2_eid)
    t12_a: Column<Advice>,
    t12_b: Column<Advice>,
    t12_c: Column<Advice>,
    t12_i_r1: Column<Advice>,
    t12_j_r2: Column<Advice>,
    t12_r1_eid: Column<Advice>,
    t12_r2_eid: Column<Advice>,
    t12_pad: Column<Advice>,
    q_t12: Selector,
    q_t12_lookup: Selector,

    // Bag2 join rows: (b,c,a,r2_eid) proven by lookups:
    //   r2: (b->c) comes from InByDst at key=c (dst=c) giving src=b and eid=r2_eid
    //   r3: (c->a) comes from OutBySrc at key=c giving dst=a (eid=r3_eid not used in message)
    t23_b: Column<Advice>,
    t23_c: Column<Advice>,
    t23_a: Column<Advice>,
    t23_i_r2: Column<Advice>,
    t23_j_r3: Column<Advice>,
    t23_r2_eid: Column<Advice>,
    t23_r3_eid: Column<Advice>,
    t23_pad: Column<Advice>,
    q_t23: Selector,
    q_t23_lookup: Selector,

    // Message from Bag2: key=pack2(r2_eid, a), val=count
    agg_msg: AggSumByKeyConfig<F>,

    // Lookup msg for each Bag1 row
    msg_lookup: MapLookupConfig<F>,

    // ordering a<b and b<c (on node IDs, not eids)
    q_order: Selector,
    lt_ab: LtConfig<F, NUM_BYTES>,
    lt_bc: LtConfig<F, NUM_BYTES>,

    // contribution = msg_val * lt_ab * lt_bc  (0 if pad)
    contrib: Column<Advice>,
    q_contrib: Selector,

    // sum contrib
    run_sum: Column<Advice>,
    q_sum0: Selector,
    q_sum: Selector,

    out: Column<Advice>,
    q_out: Selector,
}

#[derive(Clone, Debug)]
pub struct Triangle2BagChip<F: Field + Ord> {
    cfg: Triangle2BagConfig<F>,
}
impl<F: Field + Ord> Triangle2BagChip<F> {
    pub fn construct(cfg: Triangle2BagConfig<F>) -> Self {
        Self { cfg }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> Triangle2BagConfig<F> {
        let instance = meta.instance_column();
        meta.enable_equality(instance);

        let e_eid = meta.advice_column();
        let e_src = meta.advice_column();
        let e_dst = meta.advice_column();
        for c in [e_eid, e_src, e_dst] {
            meta.enable_equality(c);
        }

        let q_copy_in_by_dst = meta.selector();
        let q_copy_out_by_src = meta.selector();

        // Views
        let in_by_dst = IndexedViewChip::<F>::configure(meta);
        let out_by_src = IndexedViewChip::<F>::configure(meta);

        // copy constraints base -> view inputs (same row)
        meta.create_gate("copy base -> in_by_dst inputs", |m| {
            let q = m.query_selector(q_copy_in_by_dst);
            vec![
                q.clone() * (m.query_advice(in_by_dst.in_key, Rotation::cur()) - m.query_advice(e_dst, Rotation::cur())),
                q.clone() * (m.query_advice(in_by_dst.in_val, Rotation::cur()) - m.query_advice(e_src, Rotation::cur())),
                q * (m.query_advice(in_by_dst.in_eid, Rotation::cur()) - m.query_advice(e_eid, Rotation::cur())),
            ]
        });
        meta.create_gate("copy base -> out_by_src inputs", |m| {
            let q = m.query_selector(q_copy_out_by_src);
            vec![
                q.clone() * (m.query_advice(out_by_src.in_key, Rotation::cur()) - m.query_advice(e_src, Rotation::cur())),
                q.clone() * (m.query_advice(out_by_src.in_val, Rotation::cur()) - m.query_advice(e_dst, Rotation::cur())),
                q * (m.query_advice(out_by_src.in_eid, Rotation::cur()) - m.query_advice(e_eid, Rotation::cur())),
            ]
        });

        // Bag1 columns
        let t12_a = meta.advice_column();
        let t12_b = meta.advice_column();
        let t12_c = meta.advice_column();
