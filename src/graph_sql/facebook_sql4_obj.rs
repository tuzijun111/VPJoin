//! 4-cycle COUNT(*) with strict src ordering A<B<C<D over a single Edge table.
//!
//! SQL:
//!   SELECT COUNT(*) AS cnt
//!   FROM Edge r1
//!   JOIN Edge r2 ON r1.dst = r2.src
//!   JOIN Edge r3 ON r2.dst = r3.src
//!   JOIN Edge r4 ON r3.dst = r4.src
//!   AND r4.dst = r1.src
//!   WHERE r1.src < r2.src
//!     AND r2.src < r3.src
//!     AND r3.src < r4.src;
//!
//! Variables:
//!   A = r1.src = r4.dst
//!   B = r1.dst = r2.src
//!   C = r2.dst = r3.src
//!   D = r3.dst = r4.src
//!
//! Directed 4-cycle: A -> B -> C -> D -> A with ordering A<B<C<D.
//!
//! Acyclic decomposition via diagonal cut on (A,C):
//!   Bag1 {r1,r2}: paths A->B->C
//!   Bag2 {r3,r4}: paths C->D->A
//!   Separator variables: (A,C)
//!
//! Message from Bag2 -> Bag1:
//!   msg_key = pack2(A, C)
//!   msg_val = COUNT(paths C->D->A that satisfy C<D) grouped by msg_key
//!
//! Final:
//!   answer = Σ_{row in Bag1} msg_val(pack2(A,C)) * [A<B] * [B<C]
//!
//! Key fix vs your earlier “mismatched types” problem:
//! - DO NOT define a local Edge struct here.
//! - Reuse the dataset type: crate::data::graph_data_processing::Edge
//!   so read_edges(...) returns exactly the same Edge type as the circuit.
//!
//! Padding knobs:
//! - bag1_pad_extra: pad Bag1 length to real_len + bag1_pad_extra
//! - bag2_pad_extra: pad Bag2 length to real_len + bag2_pad_extra
//!
//! Requires your existing chips:
//!   crate::chips::is_zero::{IsZeroChip, IsZeroConfig}
//!   crate::chips::less_than::{LtChip, LtConfig, LtInstruction}
//!   crate::chips::permutation_any::{PermAnyChip, PermAnyConfig}

use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};

// ✅ Use the dataset Edge type directly (no conversion needed).
use crate::data::graph_data_processing::Edge;

use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;

const NUM_BYTES: usize = 8;
const PAD_U64: u64 = u64::MAX;

// shift node IDs by +1 so 0 can be reserved for dummy row
const SHIFT_ID: u64 = 1;

// pack2(hi, lo) using 32-bit lanes (assumes hi,lo < 2^32)
const PACK_BITS: u32 = 32;
const PACK_SHIFT: u64 = 1u64 << PACK_BITS;
fn pack2(hi: u64, lo: u64) -> u64 {
    hi * PACK_SHIFT + lo
}

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

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

    // Bag1 (T12) rows: A->B->C
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

    // Bag2 (T34) rows: C->D->A
    t34_c: Column<Advice>,
    t34_d: Column<Advice>,
    t34_a: Column<Advice>,
    t34_i_r3: Column<Advice>,
    t34_j_r4: Column<Advice>,
    t34_r3_eid: Column<Advice>,
    t34_r4_eid: Column<Advice>,
    t34_real: Column<Advice>,
    q_t34_lookup: Selector,
    q_t34_flag: Selector,
    q_t34_msg_in: Selector,

    // ordering in bag2: C < D
    q_cd: Selector,
    lt_cd: LtConfig<F, NUM_BYTES>,

    // message aggregator: msg_key=pack2(A,C), msg_val=count
    agg_msg: AggSumByKeyConfig<F>,

    // message lookup per Bag1 row
    msg_lookup: MapLookupConfig<F>,

    // ordering checks for Bag1: A<B and B<C
    q_order12: Selector,
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
        let t34_c = meta.advice_column();
        let t34_d = meta.advice_column();
        let t34_a = meta.advice_column();
        let t34_i_r3 = meta.advice_column();
        let t34_j_r4 = meta.advice_column();
        let t34_r3_eid = meta.advice_column();
        let t34_r4_eid = meta.advice_column();
        let t34_real = meta.advice_column();
        for c in [
            t34_c, t34_d, t34_a, t34_i_r3, t34_j_r4, t34_r3_eid, t34_r4_eid, t34_real,
        ] {
            meta.enable_equality(c);
        }
        let q_t34_lookup = meta.complex_selector();
        let q_t34_flag = meta.selector();
        let q_t34_msg_in = meta.selector();

        // Bag1 lookups:
        // r1 via InByDst: key=B, idx=i_r1 -> val=A, eid=r1_eid
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
        // r2 via OutBySrc: key=B, idx=j_r2 -> val=C, eid=r2_eid
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

        // Bag2 lookups:
        // r3 via InByDst: key=D, idx=i_r3 -> val=C, eid=r3_eid  (since r3 is C->D)
        meta.lookup_any("bag2 r3 from in_by_dst", |m| {
            let q = m.query_selector(q_t34_lookup);
            vec![
                (
                    q.clone() * m.query_advice(t34_d, Rotation::cur()),
                    m.query_advice(in_by_dst.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t34_i_r3, Rotation::cur()),
                    m.query_advice(in_by_dst.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t34_c, Rotation::cur()),
                    m.query_advice(in_by_dst.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t34_r3_eid, Rotation::cur()),
                    m.query_advice(in_by_dst.sorted_eid, Rotation::cur()),
                ),
            ]
        });
        // r4 via OutBySrc: key=D, idx=j_r4 -> val=A, eid=r4_eid  (since r4 is D->A)
        meta.lookup_any("bag2 r4 from out_by_src", |m| {
            let q = m.query_selector(q_t34_lookup);
            vec![
                (
                    q.clone() * m.query_advice(t34_d, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_key, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t34_j_r4, Rotation::cur()),
                    m.query_advice(out_by_src.idx, Rotation::cur()),
                ),
                (
                    q.clone() * m.query_advice(t34_a, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_val, Rotation::cur()),
                ),
                (
                    q * m.query_advice(t34_r4_eid, Rotation::cur()),
                    m.query_advice(out_by_src.sorted_eid, Rotation::cur()),
                ),
            ]
        });

        // Bag2 real is boolean
        meta.create_gate("bag2 real boolean", |m| {
            let q = m.query_selector(q_t34_flag);
            let real = m.query_advice(t34_real, Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![q * real.clone() * (one - real)]
        });

        // ordering in Bag2: C < D, enabled only if real=1
        let q_cd = meta.selector();
        let lt_cd = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_cd) * m.query_advice(t34_real, Rotation::cur()),
            |m| m.query_advice(t34_c, Rotation::cur()),
            |m| m.query_advice(t34_d, Rotation::cur()),
        );

        // Message aggregator
        let agg_msg = AggSumByKeyChip::<F>::configure(meta);

        // Tie agg_msg inputs to Bag2 rows:
        // keep = t34_real * [C<D]
        // in_key = keep ? pack2(A,C) : PAD
        // in_val = keep
        meta.create_gate("msg input from bag2", |m| {
            let q = m.query_selector(q_t34_msg_in);

            let a = m.query_advice(t34_a, Rotation::cur());
            let c = m.query_advice(t34_c, Rotation::cur());
            let key_expr = a * Expression::Constant(F::from(PACK_SHIFT)) + c;

            let real = m.query_advice(t34_real, Rotation::cur());
            let cd = lt_cd.is_lt(m, None);
            let keep = real.clone() * cd;

            let one = Expression::Constant(F::ONE);
            let pad = Expression::Constant(F::from(PAD_U64));
            let selected_key = keep.clone() * key_expr + (one - keep.clone()) * pad;

            vec![
                q.clone() * (m.query_advice(agg_msg.in_key, Rotation::cur()) - selected_key),
                q * (m.query_advice(agg_msg.in_val, Rotation::cur()) - keep),
            ]
        });

        // Message lookup per Bag1 row (table is agg_msg.map_* gated by agg_msg.q_map_tbl)
        let msg_lookup = MapLookupChip::<F>::configure(
            meta,
            agg_msg.map_key,
            agg_msg.map_val,
            agg_msg.map_key_next,
            agg_msg.q_map_tbl,
        );

        // Bag1 real is boolean + tie msg_lookup.key = pack2(A,C)
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
        let q_order12 = meta.selector();
        let lt_ab = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_order12) * m.query_advice(t12_real, Rotation::cur()),
            |m| m.query_advice(t12_a, Rotation::cur()),
            |m| m.query_advice(t12_b, Rotation::cur()),
        );
        let lt_bc = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_order12) * m.query_advice(t12_real, Rotation::cur()),
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

        Cycle4OrderedConfig {
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
            t34_c,
            t34_d,
            t34_a,
            t34_i_r3,
            t34_j_r4,
            t34_r3_eid,
            t34_r4_eid,
            t34_real,
            q_t34_lookup,
            q_t34_flag,
            q_t34_msg_in,
            q_cd,
            lt_cd,
            agg_msg,
            msg_lookup,
            q_order12,
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
        LtChip::<F, NUM_BYTES>::construct(cfg.lt_cd.clone()).load(layouter)?;

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
                // Build view inputs from base rows
                // -------------------
                let mut base: Vec<(u64, u64, u64)> = Vec::with_capacity(n_base); // (eid,src,dst)
                base.push((0, 0, 0));
                for (i, e) in edges.iter().enumerate() {
                    let eid = (i + 1) as u64;
                    base.push((eid, (e.src as u64) + SHIFT_ID, (e.dst as u64) + SHIFT_ID));
                }

                let mut in_rows: Vec<(u64, u64, u64)> = Vec::with_capacity(n_base);
                let mut out_rows: Vec<(u64, u64, u64)> = Vec::with_capacity(n_base);
                for (eid, src, dst) in base.iter().copied() {
                    in_rows.push((dst, src, eid)); // key=dst, val=src
                    out_rows.push((src, dst, eid)); // key=src, val=dst
                }

                // Assign indexed views
                in_view_chip.assign(&mut region, n_base, &in_rows)?;
                out_view_chip.assign(&mut region, n_base, &out_rows)?;

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
                // Materialize Bag1: T12 = r1(A->B) ⋈ r2(B->C) on B
                // row = (A,B,C,i_r1,j_r2,r1_eid,r2_eid)
                // -------------------
                let mut t12: Vec<(u64, u64, u64, u64, u64, u64, u64)> = vec![];
                for (&b, incoming) in in_groups.iter() {
                    if let Some(outgoing) = out_groups.get(&b) {
                        for (i, (r1_eid, a)) in incoming.iter().enumerate() {
                            for (j, (r2_eid, c)) in outgoing.iter().enumerate() {
                                t12.push((*a, b, *c, i as u64, j as u64, *r1_eid, *r2_eid));
                            }
                        }
                    }
                }

                // -------------------
                // Materialize Bag2: T34 = r3(C->D) ⋈ r4(D->A) on D
                // row = (C,D,A,i_r3,j_r4,r3_eid,r4_eid)
                // -------------------
                let mut t34: Vec<(u64, u64, u64, u64, u64, u64, u64)> = vec![];
                for (&d, incoming) in in_groups.iter() {
                    if let Some(outgoing) = out_groups.get(&d) {
                        for (i, (r3_eid, c)) in incoming.iter().enumerate() {
                            for (j, (r4_eid, a)) in outgoing.iter().enumerate() {
                                t34.push((*c, d, *a, i as u64, j as u64, *r3_eid, *r4_eid));
                            }
                        }
                    }
                }

                // -------------------
                // Message map (host-side): msg_key = pack2(A,C), msg_val = count of Bag2 rows with C<D
                // -------------------
                let mut msg_map: BTreeMap<u64, u64> = BTreeMap::new();
                for (c, d, a, _i, _j, _r3_eid, _r4_eid) in t34.iter().copied() {
                    if c < d {
                        let key = pack2(a, c);
                        *msg_map.entry(key).or_default() += 1;
                    }
                }

                // -------------------
                // Assign Bag2 rows + build agg inputs (key,value)
                // -------------------
                let real34 = t34.len();
                let n34 = std::cmp::max(real34 + bag2_pad_extra, 1);
                println!("The length of n34 is: {}", t12.len());

                let mut agg_in: Vec<(u64, u64)> = vec![(PAD_U64, 0); n34]; // default PAD bucket

                let lt_cd_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_cd.clone());

                for i in 0..n34 {
                    cfg.q_t34_lookup.enable(&mut region, i)?;
                    cfg.q_t34_flag.enable(&mut region, i)?;
                    cfg.q_cd.enable(&mut region, i)?;
                    cfg.q_t34_msg_in.enable(&mut region, i)?;

                    if i < real34 {
                        let (c, d, a, i_r3, j_r4, r3_eid, r4_eid) = t34[i];

                        region.assign_advice(
                            || "t34_c",
                            cfg.t34_c,
                            i,
                            || Value::known(F::from(c)),
                        )?;
                        region.assign_advice(
                            || "t34_d",
                            cfg.t34_d,
                            i,
                            || Value::known(F::from(d)),
                        )?;
                        region.assign_advice(
                            || "t34_a",
                            cfg.t34_a,
                            i,
                            || Value::known(F::from(a)),
                        )?;
                        region.assign_advice(
                            || "t34_i_r3",
                            cfg.t34_i_r3,
                            i,
                            || Value::known(F::from(i_r3)),
                        )?;
                        region.assign_advice(
                            || "t34_j_r4",
                            cfg.t34_j_r4,
                            i,
                            || Value::known(F::from(j_r4)),
                        )?;
                        region.assign_advice(
                            || "t34_r3_eid",
                            cfg.t34_r3_eid,
                            i,
                            || Value::known(F::from(r3_eid)),
                        )?;
                        region.assign_advice(
                            || "t34_r4_eid",
                            cfg.t34_r4_eid,
                            i,
                            || Value::known(F::from(r4_eid)),
                        )?;
                        region.assign_advice(
                            || "t34_real",
                            cfg.t34_real,
                            i,
                            || Value::known(F::ONE),
                        )?;

                        // lt witness
                        lt_cd_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(c)),
                            Value::known(F::from(d)),
                        )?;

                        let keep = if c < d { 1u64 } else { 0u64 };
                        let key = if keep == 1 { pack2(a, c) } else { PAD_U64 };
                        agg_in[i] = (key, keep);
                    } else {
                        // padded row (dummy edge)
                        region.assign_advice(
                            || "t34_c0",
                            cfg.t34_c,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t34_d0",
                            cfg.t34_d,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t34_a0",
                            cfg.t34_a,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t34_i0",
                            cfg.t34_i_r3,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t34_j0",
                            cfg.t34_j_r4,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t34_r3eid0",
                            cfg.t34_r3_eid,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t34_r4eid0",
                            cfg.t34_r4_eid,
                            i,
                            || Value::known(F::ZERO),
                        )?;
                        region.assign_advice(
                            || "t34_real0",
                            cfg.t34_real,
                            i,
                            || Value::known(F::ZERO),
                        )?;

                        lt_cd_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::ZERO),
                            Value::known(F::ZERO),
                        )?;
                        agg_in[i] = (PAD_U64, 0);
                    }
                }

                // Aggregate Bag2 rows into msg map table (inside the circuit)
                let _msg_emitted = agg_msg_chip.assign(&mut region, n34, &agg_in)?;

                // -------------------
                // Assign Bag1 rows + lookup message + ordering + sum
                // -------------------
                let real12 = t12.len();
                let n12 = std::cmp::max(real12 + bag1_pad_extra, 1);
                println!("The length of n12 is: {}", t12.len());

                // Prepare sorted key list for gap witness (host-side)
                // IMPORTANT: must include 0 and PAD
                let mut keys: Vec<u64> = msg_map.keys().copied().collect();
                keys.push(0);
                keys.push(PAD_U64);
                keys.sort();
                keys.dedup();

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
                    cfg.q_order12.enable(&mut region, i)?;
                    cfg.q_contrib.enable(&mut region, i)?;

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

                    // order witnesses (constraints disabled when real=0)
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

                    // contrib witness
                    let ab = if a < b { 1u64 } else { 0u64 };
                    let bc = if b < c { 1u64 } else { 0u64 };
                    let contrib_u64 = real * val * ab * bc;

                    region.assign_advice(
                        || "contrib",
                        cfg.contrib,
                        i,
                        || Value::known(F::from(contrib_u64)),
                    )?;

                    // running sum
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

                // enable q_out at the last sum row
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

/// Wrapper circuit (uses dataset Edge directly)
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
    use crate::data::graph_data_processing::read_edges;
    use crate::data::graph_data_processing::read_edges_csv;
    use halo2_proofs::dev::MockProver;

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
        let params_path = "/home2/binbin/PoneglyphDB/src/proof/param18";
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
    fn test_max_fre() {
        let base_path = "/home2/binbin/PoneglyphDB/src/graph_data";
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
    fn test() {
        let base_path = "/home2/binbin/PoneglyphDB/src/graph_data";

        let mut edges = read_edges(&format!("{}/wiki/wiki_Vote.txt", base_path)).unwrap();

        // let mut edges =
        //     read_edges(&format!("{}/facebook/facebook_combined.txt", base_path)).unwrap();

        // let mut edges =
        //     read_edges_csv(&format!("{}/last/lastfm_asia_edges.csv", base_path)).unwrap();

        edges.truncate(10000);

        let cnt = expected_cnt(&edges);

        let circuit = MyCircuit::<Fp> {
            edges,
            bag1_pad_extra: 18067,
            bag2_pad_extra: 18067,
            _marker: PhantomData,
        };

        let public_input = vec![Fp::from(cnt)];
        let k = 18;

        // let test = true;
        let test = false;

        if test {
            let prover = MockProver::run(k, &circuit, vec![public_input]).unwrap();
            prover.assert_satisfied();
        } else {
            let proof_path = "/home2/binbin/PoneglyphDB/src/proof/last_proof_q4_dp";
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
    }
}
