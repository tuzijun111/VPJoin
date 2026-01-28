//! 3-cycle (triangle) COUNT(*) with ordering (a<b<c) using 2-bag decomposition.
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
//! Bags (share SAME r2 tuple identity via eid):
//!   Bag1: {r1, r2}  materialize J12 = r1 ⋈ r2 on b  -> rows (a,b,c,r1_eid,r2_eid, idx_r1_in, idx_r2_out)
//!   Bag2: {r2, r3}  materialize J23 = r2 ⋈ r3 on c  -> rows (b,c,a,r2_eid,r3_eid, idx_r2_in, idx_r3_out)
//!   Separator: r2_eid
//!
//! Message from Bag2:
//!   msg_key = pack2(r2_eid, a), msg_val = COUNT(*) over Bag2 rows grouped by msg_key
//!
//! Final:
//!   answer = Σ_{row in Bag1} msg_val(pack2(r2_eid,a)) * [a<b] * [b<c]
//!
//! Important implementation notes:
//! - We DO NOT take J12/J23 as inputs. We materialize them in witness from base edges,
//!   and prove each row corresponds to real edges by lookups into two indexed+sorted views:
//!     (1) InByDst:  key=dst, val=src, eid  (sorted by (dst,eid) with idx inside key-group)
//!     (2) OutBySrc: key=src, val=dst, eid  (sorted by (src,eid) with idx inside key-group)
//! - To safely gate lookups when rows are padded, we include a dummy edge (eid=0,src=0,dst=0)
//!   in the base table and therefore in both views.
//!
//! Depends on your existing chips:
//!   crate::chips::is_zero::{IsZeroChip, IsZeroConfig}
//!   crate::chips::less_than::{LtChip, LtConfig, LtInstruction}
//!   crate::chips::permutation_any::{PermAnyChip, PermAnyConfig}

use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};

use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;

const NUM_BYTES: usize = 8;
const PAD_U64: u64 = u64::MAX;

// shift node IDs by +1 so 0 can be reserved for dummy
const SHIFT_ID: u64 = 1;

// pack2(hi, lo) with 32-bit lanes: hi < 2^32 and lo < 2^32 (assumption for truncated tests)
const PACK_BITS: u32 = 32;
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
/// AggSumByKey: group-by SUM(val) over key (used for counting)
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

    // map table for membership+gap proofs
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

        // compact emit -> out
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

    /// Assign full agg + map. Input rows length n (fixed).
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
            // IMPORTANT: use sorted_ext for i+1
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

        // map table rows: 0..n  (total n+1)
        // row0 dummy: key=0,val=0, key_next = key(row1)
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
        cfg.q_map_shift.enable(region, 0)?; // map_key_next[0] == map_key[1]

        // rows 1..n copy out[0..n-1]
        for i in 0..n {
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

        // key_next chain for rows 1..n-1 (i.e., r=1..n-1)
        // Enable shift for r=1..n-1, because shift enforces map_key_next[r] == map_key[r+1].
        for r in 1..n {
            cfg.q_map_shift.enable(region, r)?;
            let nextk = out.get(r).map(|x| x.0).unwrap_or(PAD_U64); // map_key[r+1] == out[r]
            region.assign_advice(
                || "map_kn",
                cfg.map_key_next,
                r,
                || Value::known(F::from(nextk)),
            )?;
        }

        // last row r=n: key_next = PAD
        cfg.q_map_last.enable(region, n)?;
        region.assign_advice(
            || "map_kn_last",
            cfg.map_key_next,
            n,
            || Value::known(F::from(PAD_U64)),
        )?;

        Ok(emitted)
    }

    // expose for other gadgets
    pub fn map_key(&self) -> Column<Advice> {
        self.cfg.map_key
    }
    pub fn map_val(&self) -> Column<Advice> {
        self.cfg.map_val
    }
    pub fn map_key_next(&self) -> Column<Advice> {
        self.cfg.map_key_next
    }
}

/// ------------------------------
/// Indexed view: (key,val,eid) sorted by (key,eid), plus idx within key-group.
/// ------------------------------
#[derive(Clone, Debug)]
pub struct IndexedViewConfig<F: Field + Ord> {
    in_key: Column<Advice>,
    in_val: Column<Advice>,
    in_eid: Column<Advice>,

    sorted_key: Column<Advice>,
    sorted_val: Column<Advice>,
    sorted_eid: Column<Advice>,
    perm: PermAnyConfig,

    q_sort: Selector,
    lt_key: LtConfig<F, NUM_BYTES>,
    iz_eq_key: IsZeroConfig<F>,

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

        // perm selectors
        for i in 0..n {
            cfg.perm.q_perm1.enable(region, i)?;
            cfg.perm.q_perm2.enable(region, i)?;
        }

        // assign inputs
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

        // build sorted
        let mut sorted = in_rows.to_vec();
        sorted.sort_by_key(|(k, _, eid)| (*k, *eid));

        let mut sorted_ext = sorted.clone();
        sorted_ext.push((PAD_U64, 0, 0)); // sentinel

        // assign sorted + sentinel row
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

        // sort constraints rows 0..n-1 comparing to next (including sentinel)
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

        // idx computation + constraints
        // idx[0]=0, idx[i]=idx[i-1]+1 if same key else 0
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
        // dummy idx on sentinel row (unused)
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

    // expose columns for lookups
    pub fn sorted_key(&self) -> Column<Advice> {
        self.cfg.sorted_key
    }
    pub fn sorted_val(&self) -> Column<Advice> {
        self.cfg.sorted_val
    }
    pub fn sorted_eid(&self) -> Column<Advice> {
        self.cfg.sorted_eid
    }
    pub fn idx(&self) -> Column<Advice> {
        self.cfg.idx
    }
}

/// ------------------------------
/// MapLookup: membership+gap lookup into AggSumByKey map table
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

        meta.lookup_any("msg member", |m| {
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

        meta.lookup_any("msg gap", |m| {
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
        }
    }

    pub fn key_col(&self) -> Column<Advice> {
        self.cfg.key
    }
    pub fn val_col(&self) -> Column<Advice> {
        self.cfg.val
    }
}

/// ------------------------------
/// Main circuit config
/// ------------------------------
#[derive(Clone, Debug)]
pub struct Triangle2BagConfig<F: Field + Ord> {
    instance: Column<Instance>,

    // base table E: (eid, src, dst)  includes dummy row0 = (0,0,0)
    e_eid: Column<Advice>,
    e_src: Column<Advice>,
    e_dst: Column<Advice>,

    // views
    in_by_dst: IndexedViewConfig<F>,
    out_by_src: IndexedViewConfig<F>,

    // Bag1 rows (a,b,c + indices + eids)
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

    // Bag2 rows (b,c,a + indices + eids)
    t23_b: Column<Advice>,
    t23_c: Column<Advice>,
    t23_a: Column<Advice>,
    t23_i_r2: Column<Advice>,
    t23_j_r3: Column<Advice>,
    t23_r2_eid: Column<Advice>,
    t23_r3_eid: Column<Advice>,
    t23_real: Column<Advice>,
    q_t23_lookup: Selector,
    q_t23_msg_in: Selector,

    // message aggregator (key=pack2(r2_eid,a), val=t23_real)
    agg_msg: AggSumByKeyConfig<F>,

    // message lookup per Bag1 row
    msg_lookup: MapLookupConfig<F>,

    // ordering checks (a<b and b<c)
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

        // views
        let in_by_dst = IndexedViewChip::<F>::configure(meta);
        let out_by_src = IndexedViewChip::<F>::configure(meta);

        // We will assign view inputs directly equal to base values (no extra copy-gates needed).

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

        // Bag2 columns
        let t23_b = meta.advice_column();
        let t23_c = meta.advice_column();
        let t23_a = meta.advice_column();
        let t23_i_r2 = meta.advice_column();
        let t23_j_r3 = meta.advice_column();
        let t23_r2_eid = meta.advice_column();
        let t23_r3_eid = meta.advice_column();
        let t23_real = meta.advice_column();
        for c in [
            t23_b, t23_c, t23_a, t23_i_r2, t23_j_r3, t23_r2_eid, t23_r3_eid, t23_real,
        ] {
            meta.enable_equality(c);
        }
        let q_t23_lookup = meta.complex_selector();
        let q_t23_msg_in = meta.selector();

        // Join lookups (Bag1):
        // r1 via InByDst: key=b, idx=i_r1 -> val=a, eid=r1_eid
        meta.lookup_any("bag1 r1 from in_by_dst", |m| {
            let q = m.query_selector(q_t12_lookup);
            vec![
                (
                    q.clone() * m.query_advice(t12_b, Rotation::cur()),
                    m.query_advice(in_by_dst.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t12_i_r1, Rotation::cur()),
                    m.query_advice(in_by_dst.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t12_a, Rotation::cur()),
                    m.query_advice(in_by_dst.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t12_r1_eid, Rotation::cur()),
                    m.query_advice(in_by_dst.sorted_eid, Rotation::cur()),
                ),
            ]
        });
        // r2 via OutBySrc: key=b, idx=j_r2 -> val=c, eid=r2_eid
        meta.lookup_any("bag1 r2 from out_by_src", |m| {
            let q = m.query_selector(q_t12_lookup);
            vec![
                (
                    q.clone() * m.query_advice(t12_b, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t12_j_r2, Rotation::cur()),
                    m.query_advice(out_by_src.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t12_c, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t12_r2_eid, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_eid, Rotation::cur()),
                ),
            ]
        });

        // Join lookups (Bag2):
        // r2 via InByDst: key=c (dst=c), idx=i_r2 -> val=b, eid=r2_eid
        meta.lookup_any("bag2 r2 from in_by_dst", |m| {
            let q = m.query_selector(q_t23_lookup);
            vec![
                (
                    q.clone() * m.query_advice(t23_c, Rotation::cur()),
                    m.query_advice(in_by_dst.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t23_i_r2, Rotation::cur()),
                    m.query_advice(in_by_dst.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t23_b, Rotation::cur()),
                    m.query_advice(in_by_dst.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t23_r2_eid, Rotation::cur()),
                    m.query_advice(in_by_dst.sorted_eid, Rotation::cur()),
                ),
            ]
        });
        // r3 via OutBySrc: key=c (src=c), idx=j_r3 -> val=a, eid=r3_eid
        meta.lookup_any("bag2 r3 from out_by_src", |m| {
            let q = m.query_selector(q_t23_lookup);
            vec![
                (
                    q.clone() * m.query_advice(t23_c, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t23_j_r3, Rotation::cur()),
                    m.query_advice(out_by_src.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t23_a, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t23_r3_eid, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_eid, Rotation::cur()),
                ),
            ]
        });

        // Message aggregator
        let agg_msg = AggSumByKeyChip::<F>::configure(meta);

        // Tie agg_msg inputs to Bag2 rows:
        // in_key = pack2(r2_eid, a), in_val = t23_real
        meta.create_gate("msg input from bag2", |m| {
            let q = m.query_selector(q_t23_msg_in);
            let r2eid = m.query_advice(t23_r2_eid, Rotation::cur());
            let a = m.query_advice(t23_a, Rotation::cur());
            let key_expr = r2eid * Expression::Constant(F::from(PACK_SHIFT)) + a;
            vec![
                q.clone() * (m.query_advice(agg_msg.in_key, Rotation::cur()) - key_expr),
                q * (m.query_advice(agg_msg.in_val, Rotation::cur())
                    - m.query_advice(t23_real, Rotation::cur())),
            ]
        });

        // Message lookup per Bag1 row, using agg_msg map table
        let msg_lookup = MapLookupChip::<F>::configure(
            meta,
            agg_msg.map_key,
            agg_msg.map_val,
            agg_msg.map_key_next,
        );

        // Tie msg_lookup.key to Bag1 (r2_eid,a)
        // =======================
        // CONTINUE FROM:
        // meta.create_gate("bag1 key = pack2(r2_eid,a)", |m| {
        //   let q = m.query_selector(q_t12_key);
        //   let r ...
        // =======================

        meta.create_gate("bag1 key = pack2(r2_eid,a)", |m| {
            let q = m.query_selector(q_t12_key);
            let r2eid = m.query_advice(t12_r2_eid, Rotation::cur());
            let a = m.query_advice(t12_a, Rotation::cur());
            let key_expr = r2eid * Expression::Constant(F::from(PACK_SHIFT)) + a;

            let real = m.query_advice(t12_real, Rotation::cur());
            let one = Expression::Constant(F::ONE);

            vec![
                // boolean for real
                q.clone() * real.clone() * (one.clone() - real.clone()),
                // msg_lookup.key = pack2(r2_eid, a)   (even if real=0, we set a=r2eid=0 in witness)
                q * (m.query_advice(msg_lookup.key, Rotation::cur()) - key_expr),
            ]
        });

        // ordering checks (as filter bits): lt_ab = [a<b], lt_bc = [b<c]
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

        // out equals run_sum at the SAME row where q_out is enabled
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

        Triangle2BagConfig {
            instance,
            e_eid,
            e_src,
            e_dst,
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
            t23_b,
            t23_c,
            t23_a,
            t23_i_r2,
            t23_j_r3,
            t23_r2_eid,
            t23_r3_eid,
            t23_real,
            q_t23_lookup,
            q_t23_msg_in,
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
        }
    }

    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        edges: Vec<Edge>,
    ) -> Result<AssignedCell<F, F>, Error> {
        let cfg = self.cfg.clone();

        // Load all LT tables used in this circuit
        LtChip::<F, NUM_BYTES>::construct(cfg.in_by_dst.lt_key.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.out_by_src.lt_key.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.agg_msg.lt_key.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.agg_msg.lt_out_key.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.msg_lookup.lt_low.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.msg_lookup.lt_high.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.lt_ab.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.lt_bc.clone()).load(layouter)?;

        let in_view_chip = IndexedViewChip::<F>::construct(cfg.in_by_dst.clone());
        let out_view_chip = IndexedViewChip::<F>::construct(cfg.out_by_src.clone());
        let agg_msg_chip = AggSumByKeyChip::<F>::construct(cfg.agg_msg.clone());

        layouter.assign_region(
            || "triangle2bag witness",
            |mut region| {
                // -------------------
                // Base table with dummy row0 = (0,0,0)
                // -------------------
                let n_base = edges.len() + 1;
                for i in 0..n_base {
                    if i == 0 {
                        region.assign_advice(|| "eid0", cfg.e_eid, 0, || Value::known(F::ZERO))?;
                        region.assign_advice(|| "src0", cfg.e_src, 0, || Value::known(F::ZERO))?;
                        region.assign_advice(|| "dst0", cfg.e_dst, 0, || Value::known(F::ZERO))?;
                    } else {
                        let e = edges[i - 1];
                        let eid = i as u64;
                        let src = e.src + SHIFT_ID;
                        let dst = e.dst + SHIFT_ID;
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
                // Build view inputs from base rows
                // -------------------
                let mut in_rows: Vec<(u64, u64, u64)> = Vec::with_capacity(n_base);
                let mut out_rows: Vec<(u64, u64, u64)> = Vec::with_capacity(n_base);

                // base rows in host representation
                let mut base: Vec<(u64, u64, u64)> = Vec::with_capacity(n_base); // (eid,src,dst)
                base.push((0, 0, 0));
                for (i, e) in edges.iter().enumerate() {
                    let eid = (i + 1) as u64;
                    base.push((eid, e.src + SHIFT_ID, e.dst + SHIFT_ID));
                }

                for (eid, src, dst) in base.iter().copied() {
                    in_rows.push((dst, src, eid)); // key=dst, val=src
                    out_rows.push((src, dst, eid)); // key=src, val=dst
                }

                // Assign both indexed views
                in_view_chip.assign(&mut region, n_base, &in_rows)?;
                out_view_chip.assign(&mut region, n_base, &out_rows)?;

                // -------------------
                // Host-side grouped indices (match view sorting by (key,eid))
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

                let mut in_idx_by_eid: HashMap<u64, u64> = HashMap::new();
                for (dst, v) in in_groups.iter() {
                    for (idx, (eid, _src)) in v.iter().enumerate() {
                        let _ = dst;
                        in_idx_by_eid.insert(*eid, idx as u64);
                    }
                }
                let mut out_idx_by_eid: HashMap<u64, u64> = HashMap::new();
                for (src, v) in out_groups.iter() {
                    for (idx, (eid, _dst)) in v.iter().enumerate() {
                        let _ = src;
                        out_idx_by_eid.insert(*eid, idx as u64);
                    }
                }

                // -------------------
                // Materialize Bag1: J12 = r1(a->b) ⋈ r2(b->c) on b
                // row = (a,b,c,i_r1,j_r2,r1_eid,r2_eid)
                // -------------------
                let mut j12: Vec<(u64, u64, u64, u64, u64, u64, u64)> = vec![];
                for (&b, incoming) in in_groups.iter() {
                    if let Some(outgoing) = out_groups.get(&b) {
                        for (i, (r1_eid, a)) in incoming.iter().enumerate() {
                            for (j, (r2_eid, c)) in outgoing.iter().enumerate() {
                                j12.push((*a, b, *c, i as u64, j as u64, *r1_eid, *r2_eid));
                            }
                        }
                    }
                }

                // -------------------
                // Materialize Bag2: J23 = r2(b->c) ⋈ r3(c->a) on c
                // row = (b,c,a,i_r2,j_r3,r2_eid,r3_eid)
                // -------------------
                let mut j23: Vec<(u64, u64, u64, u64, u64, u64, u64)> = vec![];
                for (r2_eid, b, c) in base.iter().copied() {
                    if r2_eid == 0 {
                        continue;
                    }
                    // r2 is edge (b->c) so its dst=c => idx in in_by_dst group for key=c
                    let i_r2 = *in_idx_by_eid.get(&r2_eid).unwrap_or(&0);

                    // r3 from out_by_src at key=c gives edges (c->a)
                    if let Some(v3) = out_groups.get(&c) {
                        for (j, (r3_eid, a)) in v3.iter().enumerate() {
                            j23.push((b, c, *a, i_r2, j as u64, r2_eid, *r3_eid));
                        }
                    }
                }

                // -------------------
                // Message map: msg_key = pack2(r2_eid, a), msg_val = count
                // -------------------
                let mut msg_map: BTreeMap<u64, u64> = BTreeMap::new();
                for (_b, _c, a, _i, _j, r2_eid, _r3_eid) in j23.iter().copied() {
                    let key = pack2(r2_eid, a);
                    *msg_map.entry(key).or_default() += 1;
                }

                // -------------------
                // Assign Bag2 rows + tie agg_msg inputs
                // -------------------
                let n23 = std::cmp::max(j23.len(), 1);
                let mut agg_in: Vec<(u64, u64)> = vec![(0, 0); n23];

                for i in 0..n23 {
                    cfg.q_t23_lookup.enable(&mut region, i)?;
                    cfg.q_t23_msg_in.enable(&mut region, i)?;

                    if i < j23.len() {
                        let (b, c, a, i_r2, j_r3, r2_eid, r3_eid) = j23[i];

                        region.assign_advice(
                            || "t23_b",
                            cfg.t23_b,
                            i,
                            || Value::known(F::from(b)),
                        )?;
                        region.assign_advice(
                            || "t23_c",
                            cfg.t23_c,
                            i,
                            || Value::known(F::from(c)),
                        )?;
                        region.assign_advice(
                            || "t23_a",
                            cfg.t23_a,
                            i,
                            || Value::known(F::from(a)),
                        )?;
                        region.assign_advice(
                            || "t23_i_r2",
                            cfg.t23_i_r2,
                            i,
                            || Value::known(F::from(i_r2)),
                        )?;
                        region.assign_advice(
                            || "t23_j_r3",
                            cfg.t23_j_r3,
                            i,
                            || Value::known(F::from(j_r3)),
                        )?;
                        region.assign_advice(
                            || "t23_r2_eid",
                            cfg.t23_r2_eid,
                            i,
                            || Value::known(F::from(r2_eid)),
                        )?;
                        region.assign_advice(
                            || "t23_r3_eid",
                            cfg.t23_r3_eid,
                            i,
                            || Value::known(F::from(r3_eid)),
                        )?;
                        region.assign_advice(
                            || "t23_real",
                            cfg.t23_real,
                            i,
                            || Value::known(F::ONE),
                        )?;

                        let key = pack2(r2_eid, a);
                        agg_in[i] = (key, 1);
                    } else {
                        // dummy padded row: use dummy edge everywhere (key=0 idx=0 val=0 eid=0)
                        region.assign_advice(
                            || "t23_b0",
                            cfg.t23_b,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t23_c0",
                            cfg.t23_c,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t23_a0",
                            cfg.t23_a,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t23_i0",
                            cfg.t23_i_r2,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t23_j0",
                            cfg.t23_j_r3,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t23_r2eid0",
                            cfg.t23_r2_eid,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t23_r3eid0",
                            cfg.t23_r3_eid,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t23_real0",
                            cfg.t23_real,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        agg_in[i] = (0, 0);
                    }
                }

                // Aggregate Bag2 rows into msg map table
                let _msg_emitted = agg_msg_chip.assign(&mut region, n23, &agg_in)?;

                // -------------------
                // Assign Bag1 rows
                // -------------------
                let n12 = std::cmp::max(j12.len(), 1);

                // Prepare a sorted key list for gap proofs
                let mut keys: Vec<u64> = msg_map.keys().copied().collect();
                keys.push(0);
                keys.push(PAD_U64);
                keys.sort();
                keys.dedup();

                // LT chips for order
                let lt_ab_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_ab.clone());
                let lt_bc_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_bc.clone());

                // LT chips for msg lookup gap checks
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

                    let (a, b, c, i_r1, j_r2, r1_eid, r2_eid, real) = if i < j12.len() {
                        let (a, b, c, i_r1, j_r2, r1_eid, r2_eid) = j12[i];
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

                    // Assign LT witnesses (safe even when real=0; constraints are disabled by real)
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

                    // Message lookup witness
                    let key = if real == 1 { pack2(r2_eid, a) } else { 0 };
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

                    // gap (only used if inside=0)
                    let (low, high) = if inside == 0 {
                        match keys.binary_search(&key) {
                            Ok(_) => (0u64, PAD_U64), // shouldn't happen if inside==0
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

                    // gap LT witnesses (only relevant when inside=0)
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

                    // Contribution witness (must match gate)
                    let ab = if a < b { 1u64 } else { 0u64 };
                    let bc = if b < c { 1u64 } else { 0u64 };
                    let contrib_u64 = (real as u64) * val * ab * bc;

                    region.assign_advice(
                        || "contrib",
                        cfg.contrib,
                        i,
                        || Value::known(F::from(contrib_u64)),
                    )?;

                    // Sum witnesses
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

                // ✅ FIX: enable q_out at the LAST sum row, not row 0
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
pub struct MyCircuit<F: Field> {
    pub edges: Vec<Edge>,
    pub _m: PhantomData<F>,
}
impl<F: Field> Default for MyCircuit<F> {
    fn default() -> Self {
        Self {
            edges: vec![],
            _m: PhantomData,
        }
    }
}
// impl<F: Field> Circuit<F> for MyCircuit<F> {
//     type Config = Triangle2BagConfig<F>;
//     type FloorPlanner = SimpleFloorPlanner;

//     fn without_witnesses(&self) -> Self {
//         Self::default()
//     }

//     fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
//         Triangle2BagChip::<F>::configure(meta)
//     }

//     fn synthesize(&self, cfg: Self::Config, mut layouter: impl Layouter<F>) -> Result<(), Error> {
//         let chip = Triangle2BagChip::construct(cfg);
//         let out = chip.assign(&mut layouter, self.edges.clone())?;
//         chip.expose_public(&mut layouter, out, 0)?;
//         Ok(())
//     }
// }
impl<F: Field + Ord> Circuit<F> for MyCircuit<F> {
    type Config = Triangle2BagConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        Triangle2BagChip::<F>::configure(meta)
    }

    fn synthesize(&self, cfg: Self::Config, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        let chip = Triangle2BagChip::<F>::construct(cfg.clone());
        let out = chip.assign(&mut layouter, self.edges.clone())?;
        chip.expose_public(&mut layouter, out, 0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;
    use halo2curves::pasta::Fp;

    fn expected_cnt(edges: &[Edge]) -> u64 {
        use std::collections::HashMap;

        // count multiplicity of each directed edge
        let mut cnt: HashMap<(u64, u64), u64> = HashMap::new();
        for e in edges {
            *cnt.entry((e.src, e.dst)).or_default() += 1;
        }

        // adjacency by src
        let mut out: HashMap<u64, Vec<(u64, u64)>> = HashMap::new(); // src -> [(dst, mult)]
        for (&(s, d), &c) in cnt.iter() {
            out.entry(s).or_default().push((d, c));
        }

        let mut total: u128 = 0;
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
                        // need edge (c -> a)
                        if let Some(&cca) = cnt.get(&(c, a)) {
                            total += (cab as u128) * (cbc as u128) * (cca as u128);
                        }
                    }
                }
            }
        }
        total as u64
    }

    #[test]
    fn test_small_triangle() {
        // triangle 0->1->2->0 with a<b<c satisfied: 0<1<2
        let edges = vec![
            Edge { src: 0, dst: 1 },
            Edge { src: 1, dst: 2 },
            Edge { src: 2, dst: 0 },
        ];
        let cnt = expected_cnt(&edges);

        let circuit = MyCircuit::<Fp> {
            edges,
            _m: PhantomData,
        };
        let public_input = vec![Fp::from(cnt)];

        let k = 15;
        let prover = MockProver::run(k, &circuit, vec![public_input]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_random_like() {
        // simple deterministic "random-like" generator (xorshift)
        let mut x: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            x ^= x << 7;
            x ^= x >> 9;
            x ^= x << 8;
            x
        };

        let n_nodes = 25u64;
        let n_edges = 120usize;

        let mut edges: Vec<Edge> = Vec::with_capacity(n_edges);
        for _ in 0..n_edges {
            let s = (next() % n_nodes) as u64;
            let d = (next() % n_nodes) as u64;
            // allow self-loops; they won't pass a<b<c anyway
            edges.push(Edge { src: s, dst: d });
        }

        let cnt = expected_cnt(&edges);

        let circuit = MyCircuit::<Fp> {
            edges,
            _m: PhantomData,
        };
        let public_input = vec![Fp::from(cnt)];

        let k = 17;
        let prover = MockProver::run(k, &circuit, vec![public_input]).unwrap();
        prover.assert_satisfied();
    }
}
