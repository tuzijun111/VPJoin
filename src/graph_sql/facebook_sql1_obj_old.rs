// Acyclic query 1: 5-way path (chain).
// SELECT COUNT(*) AS cnt
// FROM R1 r1
// JOIN R2 r2 ON r1.dst = r2.src
// JOIN R3 r3 ON r2.dst = r3.src
// JOIN R4 r4 ON r3.dst = r4.src
// JOIN R5 r5 ON r4.dst = r5.src;

use halo2_proofs::halo2curves::ff::PrimeField;
use halo2_proofs::plonk::Expression;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};

use crate::data::graph_data_processing::Edge;

use std::collections::HashMap;
use std::marker::PhantomData;

const NUM_BYTES: usize = 5;
const MAX_SENTINEL: u64 = (1u64 << (8 * NUM_BYTES)) - 1; // 2^40-1
const PAD_KEY: u64 = MAX_SENTINEL; // pad key goes last in ASC
const PAD_VAL: u64 = 0;

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

#[derive(Clone, Debug)]
struct AggConfig<F: Field + Ord> {
    // Input triple: (src, dst, val)  -- src/dst are existing base columns, val is separate col
    val_in: Column<Advice>,

    // Sorted triple columns
    sorted: [Column<Advice>; 3],

    // perm: (src, dst, val_in) <-> sorted
    perm_sort: PermAnyConfig,

    // sortedness check on sorted[0] (src)
    q_sort: Selector,
    lt_src_cur_next: LtConfig<F, NUM_BYTES>,
    iz_src_eq: IsZeroConfig<F>,

    // group helpers on sorted src
    q_first: Selector,
    q_accu: Selector,
    q_emit: Selector,
    run_sum: Column<Advice>,
    iz_same_prev: IsZeroConfig<F>,
    iz_same_next: IsZeroConfig<F>,

    // emitted padded group pairs (key,val) over rows of sorted
    emit_pair: [Column<Advice>; 2],

    // table (key,val) padded, permuted from emit_pair
    tbl_pair: [Column<Advice>; 2],
    tbl_key_next: Column<Advice>,
    perm_tbl: PermAnyConfig,

    // prove tbl_pair keys are sorted and tbl_key_next is the "next" key
    q_tbl_sort: Selector,
    lt_tbl_key_cur_next: LtConfig<F, NUM_BYTES>,
    iz_tbl_key_eq: IsZeroConfig<F>,

    q_tbl_shift: Selector, // enforce tbl_key_next[i] == tbl_key[i+1]
    q_tbl_last: Selector,  // enforce tbl_key_next[last] == PAD_KEY

    // === NEW: map table used by joins (has dummy row 0) ===
    map_pair: [Column<Advice>; 2], // (key,val) with row0 = (0,0), rows 1.. = tbl_pair shifted
    map_key_next: Column<Advice>,  // next(key) for map_pair[0]

    q_map_tbl: Selector,   // gate table rows 0..=L
    q_map_first: Selector, // enforce map_pair[0] == (0,0)
    q_map_link: Selector,  // enforce map_pair[i+1] == tbl_pair[i]
    q_map_shift: Selector, // enforce map_key_next[i] == map_key[i+1]
    q_map_last: Selector,  // enforce map_key_next[last] == PAD_KEY
}

#[derive(Clone, Debug)]
struct JoinConfig<F: Field + Ord> {
    // per-row membership flag: dst exists in next table?
    in_next: Column<Advice>,
    // gap witnesses if in_next == 0
    low: Column<Advice>,
    high: Column<Advice>,

    // attached value from next table (if in_next==1 else 0)
    val: Column<Advice>,

    // selectors for lookups / logic
    q_lookup: Selector,
    q_lookup_complex: Selector,

    // LT checks for low < dst and dst < high when not in_next
    lt_low: LtConfig<F, NUM_BYTES>,
    lt_high: LtConfig<F, NUM_BYTES>,
}

#[derive(Clone, Debug)]
pub struct GraphJoin5Config<F: Field + Ord> {
    instance: Column<Instance>,

    // base edges R1..R5: [src, dst]
    r: [[Column<Advice>; 2]; 5],

    // join steps for R1..R4 (R5 has no join-to-next)
    join: [JoinConfig<F>; 4],

    // aggregators producing T5, T4, T3, T2  (4 tables total)
    // agg[0]=for R5 -> T5, agg[1]=R4->T4, agg[2]=R3->T3, agg[3]=R2->T2
    agg: [AggConfig<F>; 4],

    // final sum over R1 values
    q_sum_first: Selector,
    q_sum_accu: Selector,
    sum: Column<Advice>,

    out: Column<Advice>,
}

#[derive(Clone)]
pub struct GraphJoin5Circuit<F: Field + Ord> {
    pub r1: Vec<Edge>,
    pub r2: Vec<Edge>,
    pub r3: Vec<Edge>,
    pub r4: Vec<Edge>,
    pub r5: Vec<Edge>,
    pub _marker: PhantomData<F>,
}

impl<F: Field + Ord> Default for GraphJoin5Circuit<F> {
    fn default() -> Self {
        Self {
            r1: vec![],
            r2: vec![],
            r3: vec![],
            r4: vec![],
            r5: vec![],
            _marker: PhantomData,
        }
    }
}

pub struct GraphJoin5Chip<F: Field + Ord> {
    cfg: GraphJoin5Config<F>,
}

impl<F: Field + Ord> GraphJoin5Chip<F> {
    pub fn construct(cfg: GraphJoin5Config<F>) -> Self {
        Self { cfg }
    }

    // ---------- helpers ----------
    fn mk_perm(
        meta: &mut ConstraintSystem<F>,
        width: usize,
    ) -> (Vec<Column<Advice>>, Vec<Column<Advice>>, PermAnyConfig) {
        let q1 = meta.complex_selector();
        let q2 = meta.complex_selector();
        let mut a = vec![];
        let mut b = vec![];
        for _ in 0..width {
            a.push(meta.advice_column());
            b.push(meta.advice_column());
        }
        let perm = PermAnyChip::configure(meta, q1, q2, a.clone(), b.clone());
        (a, b, perm)
    }

    fn configure_agg(
        meta: &mut ConstraintSystem<F>,
        src_col: Column<Advice>,
        dst_col: Column<Advice>,
    ) -> AggConfig<F> {
        // input val column
        let val_in = meta.advice_column();
        meta.enable_equality(val_in);

        // sorted triple
        let sorted = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        for c in sorted {
            meta.enable_equality(c);
        }

        // permutation: (src, dst, val_in) <-> sorted
        let q_perm_in = meta.complex_selector();
        let q_perm_out = meta.complex_selector();
        let perm_sort = PermAnyChip::configure(
            meta,
            q_perm_in,
            q_perm_out,
            vec![src_col, dst_col, val_in],
            sorted.to_vec(),
        );

        // sortedness on sorted[0] <= sorted[0]_next
        let q_sort = meta.selector();
        let aux_src_eq = meta.advice_column();
        let iz_src_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort),
            |m| {
                m.query_advice(sorted[0], Rotation::next())
                    - m.query_advice(sorted[0], Rotation::cur())
            },
            aux_src_eq,
        );
        let lt_src_cur_next = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort),
            |m| m.query_advice(sorted[0], Rotation::cur()),
            |m| m.query_advice(sorted[0], Rotation::next()),
        );
        meta.create_gate("sorted src is nondecreasing", |m| {
            let q = m.query_selector(q_sort);
            let le = lt_src_cur_next.is_lt(m, None) + iz_src_eq.expr();
            vec![q * (le - Expression::Constant(F::ONE))]
        });

        // group-by on sorted src, sum sorted[2]
        let q_first = meta.selector();
        let q_accu = meta.selector();
        let q_emit = meta.selector();

        let run_sum = meta.advice_column();
        meta.enable_equality(run_sum);

        let aux_same_prev = meta.advice_column();
        let iz_same_prev = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_accu),
            |m| {
                m.query_advice(sorted[0], Rotation::cur())
                    - m.query_advice(sorted[0], Rotation::prev())
            },
            aux_same_prev,
        );

        let aux_same_next = meta.advice_column();
        let iz_same_next = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_emit),
            |m| {
                m.query_advice(sorted[0], Rotation::next())
                    - m.query_advice(sorted[0], Rotation::cur())
            },
            aux_same_next,
        );

        meta.create_gate("run_sum_first", |m| {
            let q = m.query_selector(q_first);
            let rs = m.query_advice(run_sum, Rotation::cur());
            let v = m.query_advice(sorted[2], Rotation::cur());
            vec![q * (rs - v)]
        });

        meta.create_gate("run_sum_accu", |m| {
            let q = m.query_selector(q_accu);
            let same = iz_same_prev.expr(); // 1 if same group
            let rs_cur = m.query_advice(run_sum, Rotation::cur());
            let rs_prev = m.query_advice(run_sum, Rotation::prev());
            let v = m.query_advice(sorted[2], Rotation::cur());
            vec![q * (rs_cur - (same * rs_prev + v))]
        });

        // emit padded (key,val)
        let emit_pair = [meta.advice_column(), meta.advice_column()];
        for c in emit_pair {
            meta.enable_equality(c);
        }

        meta.create_gate("emit group pair at last row of group", |m| {
            let q = m.query_selector(q_emit);
            let one = Expression::Constant(F::ONE);
            let same_next = iz_same_next.expr();
            let is_last = one.clone() - same_next;
            let not_last = one - is_last.clone();

            let cur_key = m.query_advice(sorted[0], Rotation::cur());
            let cur_sum = m.query_advice(run_sum, Rotation::cur());

            let out_k = m.query_advice(emit_pair[0], Rotation::cur());
            let out_v = m.query_advice(emit_pair[1], Rotation::cur());

            let pad_k = Expression::Constant(F::from(PAD_KEY));
            let pad_v = Expression::Constant(F::from(PAD_VAL));

            vec![
                q.clone() * (out_k - (is_last.clone() * cur_key + not_last.clone() * pad_k)),
                q * (out_v - (is_last * cur_sum + not_last * pad_v)),
            ]
        });

        // Build a padded table (key,val) that must be a permutation of emit_pair
        let tbl_pair = [meta.advice_column(), meta.advice_column()];
        let tbl_key_next = meta.advice_column();
        for c in tbl_pair {
            meta.enable_equality(c);
        }
        meta.enable_equality(tbl_key_next);

        let q_tbl_in = meta.complex_selector();
        let q_tbl_out = meta.complex_selector();
        let perm_tbl = PermAnyChip::configure(
            meta,
            q_tbl_in,
            q_tbl_out,
            emit_pair.to_vec(),
            tbl_pair.to_vec(),
        );

        // prove tbl_pair[0] sorted
        let q_tbl_sort = meta.selector();
        let aux_tbl_eq = meta.advice_column();
        let iz_tbl_key_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_tbl_sort),
            |m| {
                m.query_advice(tbl_pair[0], Rotation::next())
                    - m.query_advice(tbl_pair[0], Rotation::cur())
            },
            aux_tbl_eq,
        );
        let lt_tbl_key_cur_next = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_tbl_sort),
            |m| m.query_advice(tbl_pair[0], Rotation::cur()),
            |m| m.query_advice(tbl_pair[0], Rotation::next()),
        );
        meta.create_gate("tbl keys nondecreasing", |m| {
            let q = m.query_selector(q_tbl_sort);
            let le = lt_tbl_key_cur_next.is_lt(m, None) + iz_tbl_key_eq.expr();
            vec![q * (le - Expression::Constant(F::ONE))]
        });

        // enforce tbl_key_next[i] == tbl_key[i+1], last == PAD
        let q_tbl_shift = meta.selector();
        meta.create_gate("tbl_key_next = next(tbl_key)", |m| {
            let q = m.query_selector(q_tbl_shift);
            let kn = m.query_advice(tbl_key_next, Rotation::cur());
            let nextk = m.query_advice(tbl_pair[0], Rotation::next());
            vec![q * (kn - nextk)]
        });

        let q_tbl_last = meta.selector();
        meta.create_gate("tbl_key_next last = PAD", |m| {
            let q = m.query_selector(q_tbl_last);
            let kn = m.query_advice(tbl_key_next, Rotation::cur());
            vec![q * (kn - Expression::Constant(F::from(PAD_KEY)))]
        });

        // ---- NEW map table (dummy row 0) ----
        let map_pair = [meta.advice_column(), meta.advice_column()];
        let map_key_next = meta.advice_column();
        for c in map_pair {
            meta.enable_equality(c);
        }
        meta.enable_equality(map_key_next);

        let q_map_tbl = meta.complex_selector();
        let q_map_first = meta.selector();
        let q_map_link = meta.selector();
        let q_map_shift = meta.selector();
        let q_map_last = meta.selector();

        // row0: (0,0)
        meta.create_gate("map_first_is_zero", |m| {
            let q = m.query_selector(q_map_first);
            let k0 = m.query_advice(map_pair[0], Rotation::cur());
            let v0 = m.query_advice(map_pair[1], Rotation::cur());
            vec![q.clone() * k0, q * v0]
        });

        // link: map[i+1] == tbl[i]
        meta.create_gate("map_link_tbl_shift", |m| {
            let q = m.query_selector(q_map_link);
            let mk_next = m.query_advice(map_pair[0], Rotation::next());
            let mv_next = m.query_advice(map_pair[1], Rotation::next());
            let tk_cur = m.query_advice(tbl_pair[0], Rotation::cur());
            let tv_cur = m.query_advice(tbl_pair[1], Rotation::cur());
            vec![q.clone() * (mk_next - tk_cur), q * (mv_next - tv_cur)]
        });

        // map_key_next = next(map_key)
        meta.create_gate("map_key_next = next(map_key)", |m| {
            let q = m.query_selector(q_map_shift);
            let kn = m.query_advice(map_key_next, Rotation::cur());
            let nextk = m.query_advice(map_pair[0], Rotation::next());
            vec![q * (kn - nextk)]
        });

        // last: map_key_next[last] == PAD_KEY
        meta.create_gate("map_key_next last = PAD", |m| {
            let q = m.query_selector(q_map_last);
            let kn = m.query_advice(map_key_next, Rotation::cur());
            vec![q * (kn - Expression::Constant(F::from(PAD_KEY)))]
        });

        AggConfig {
            val_in,
            sorted,
            perm_sort,
            q_sort,
            lt_src_cur_next,
            iz_src_eq,

            q_first,
            q_accu,
            q_emit,
            run_sum,
            iz_same_prev,
            iz_same_next,

            emit_pair,
            tbl_pair,
            tbl_key_next,
            perm_tbl,

            q_tbl_sort,
            lt_tbl_key_cur_next,
            iz_tbl_key_eq,
            q_tbl_shift,
            q_tbl_last,

            map_pair,
            map_key_next,
            q_map_tbl,
            q_map_first,
            q_map_link,
            q_map_shift,
            q_map_last,
        }
    }

    fn configure_join(meta: &mut ConstraintSystem<F>, dst_col: Column<Advice>) -> JoinConfig<F> {
        let in_next = meta.advice_column();
        let low = meta.advice_column();
        let high = meta.advice_column();
        let val = meta.advice_column();
        for c in [in_next, low, high, val] {
            meta.enable_equality(c);
        }

        let q_lookup = meta.selector();
        let q_lookup_complex = meta.complex_selector();

        // lt_low: low < dst, lt_high: dst < high
        let lt_low = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| {
                let q = m.query_selector(q_lookup);
                let inx = m.query_advice(in_next, Rotation::cur());
                q * (Expression::Constant(F::ONE) - inx)
            },
            |m| m.query_advice(low, Rotation::cur()),
            |m| m.query_advice(dst_col, Rotation::cur()),
        );
        let lt_high = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| {
                let q = m.query_selector(q_lookup);
                let inx = m.query_advice(in_next, Rotation::cur());
                q * (Expression::Constant(F::ONE) - inx)
            },
            |m| m.query_advice(dst_col, Rotation::cur()),
            |m| m.query_advice(high, Rotation::cur()),
        );

        // Booleanize in_next and enforce if not in_next then lt's are true; also force val=0 when not in_next
        meta.create_gate("join membership/gap basic logic", |m| {
            let q = m.query_selector(q_lookup);
            let inx = m.query_advice(in_next, Rotation::cur());
            let one = Expression::Constant(F::ONE);

            let not_in = one.clone() - inx.clone();
            let v = m.query_advice(val, Rotation::cur());

            let low_ok = lt_low.is_lt(m, None);
            let high_ok = lt_high.is_lt(m, None);

            vec![
                // in_next boolean
                q.clone() * inx.clone() * (one.clone() - inx.clone()),
                // if not in -> must satisfy low<dst<high
                q.clone() * not_in.clone() * (one.clone() - low_ok),
                q.clone() * not_in.clone() * (one.clone() - high_ok),
                // if not in -> val must be 0
                q * not_in * v,
            ]
        });

        JoinConfig {
            in_next,
            low,
            high,
            val,
            q_lookup,
            q_lookup_complex,
            lt_low,
            lt_high,
        }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> GraphJoin5Config<F> {
        let instance = meta.instance_column();
        meta.enable_equality(instance);

        let out = meta.advice_column();
        meta.enable_equality(out);

        // ✅ allocate DISTINCT columns for each Ri
        let r: [[Column<Advice>; 2]; 5] =
            std::array::from_fn(|_| [meta.advice_column(), meta.advice_column()]);

        for i in 0..5 {
            meta.enable_equality(r[i][0]);
            meta.enable_equality(r[i][1]);
        }

        let join = [
            Self::configure_join(meta, r[0][1]),
            Self::configure_join(meta, r[1][1]),
            Self::configure_join(meta, r[2][1]),
            Self::configure_join(meta, r[3][1]),
        ];

        let agg = [
            Self::configure_agg(meta, r[4][0], r[4][1]),
            Self::configure_agg(meta, r[3][0], r[3][1]),
            Self::configure_agg(meta, r[2][0], r[2][1]),
            Self::configure_agg(meta, r[1][0], r[1][1]),
        ];

        let q_sum_first = meta.selector();
        let q_sum_accu = meta.selector();
        let sum = meta.advice_column();
        meta.enable_equality(sum);

        meta.create_gate("sum_first", |m| {
            let q = m.query_selector(q_sum_first);
            let s = m.query_advice(sum, Rotation::cur());
            let v = m.query_advice(join[0].val, Rotation::cur());
            vec![q * (s - v)]
        });
        meta.create_gate("sum_accu", |m| {
            let q = m.query_selector(q_sum_accu);
            let s_cur = m.query_advice(sum, Rotation::cur());
            let s_prev = m.query_advice(sum, Rotation::prev());
            let v = m.query_advice(join[0].val, Rotation::cur());
            vec![q * (s_cur - (s_prev + v))]
        });

        GraphJoin5Config {
            instance,
            r,
            join,
            agg,
            q_sum_first,
            q_sum_accu,
            sum,
            out,
        }
    }

    /// expose public output cell
    pub fn expose_public(
        &self,
        layouter: &mut impl Layouter<F>,
        cell: AssignedCell<F, F>,
        row: usize,
    ) -> Result<(), Error> {
        layouter.constrain_instance(cell.cell(), self.cfg.instance, row)
    }

    // ---------------- witness builder (host-side) ----------------

    fn sort_by_src(mut rows: Vec<[u64; 3]>) -> Vec<[u64; 3]> {
        rows.sort_by_key(|r| r[0]);
        rows
    }

    fn run_sum_by_src(sorted: &[[u64; 3]]) -> Vec<u64> {
        let mut out = vec![0u64; sorted.len()];
        let mut acc: u128 = 0;
        let mut prev: Option<u64> = None;
        for (i, r) in sorted.iter().enumerate() {
            let src = r[0];
            let v = r[2] as u128;
            if prev == Some(src) {
                acc += v;
            } else {
                acc = v;
            }
            out[i] = acc as u64;
            prev = Some(src);
        }
        out
    }

    fn emit_pairs(sorted: &[[u64; 3]], run: &[u64]) -> Vec<[u64; 2]> {
        let n = sorted.len();
        let mut out = vec![[PAD_KEY, PAD_VAL]; n];
        for i in 0..n {
            let cur = sorted[i][0];
            let next = if i + 1 < n { sorted[i + 1][0] } else { PAD_KEY };
            let is_last = next != cur;
            if is_last {
                out[i] = [cur, run[i]];
            }
        }
        out
    }

    fn build_tbl_from_emit(emit: &[[u64; 2]], n: usize) -> Vec<[u64; 2]> {
        // take all non-pad pairs (already in nondecreasing key order due to sorting+emit),
        // pack them to the top, then pad.
        let mut pairs: Vec<[u64; 2]> = emit.iter().copied().filter(|p| p[0] != PAD_KEY).collect();
        // already sorted by construction; but keep safe:
        pairs.sort_by_key(|p| p[0]);
        while pairs.len() < n {
            pairs.push([PAD_KEY, PAD_VAL]);
        }
        pairs.truncate(n);
        pairs
    }

    fn key_next_from_tbl(tbl: &[[u64; 2]]) -> Vec<u64> {
        let n = tbl.len();
        let mut out = vec![PAD_KEY; n];
        for i in 0..n {
            out[i] = if i + 1 < n { tbl[i + 1][0] } else { PAD_KEY };
        }
        out
    }

    fn map_from_tbl(tbl: &[[u64; 2]]) -> HashMap<u64, u64> {
        let mut m = HashMap::new();
        for [k, v] in tbl.iter().copied() {
            if k == PAD_KEY {
                continue;
            }
            m.insert(k, v);
        }
        m
    }

    fn gap_witness(keys_sorted_with_pad: &[u64], x: u64) -> (u64, u64, u64) {
        // returns (in, low, high)
        // keys_sorted_with_pad must include PAD_KEY at end, and also include 0 if you want that boundary.
        match keys_sorted_with_pad.binary_search(&x) {
            Ok(_) => (1, 0, PAD_KEY),
            Err(idx) => {
                // idx is first key > x
                let high = keys_sorted_with_pad[idx];
                let low = if idx == 0 {
                    0
                } else {
                    keys_sorted_with_pad[idx - 1]
                };
                (0, low, high)
            }
        }
    }

    // ---------------- assign ----------------

    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        r1: &[Edge],
        r2: &[Edge],
        r3: &[Edge],
        r4: &[Edge],
        r5: &[Edge],
    ) -> Result<AssignedCell<F, F>, Error> {
        // Load Lt chips used in config (IMPORTANT)
        // sort lt and table sort lt and join gap lts
        for a in self.cfg.agg.iter() {
            LtChip::<F, NUM_BYTES>::construct(a.lt_src_cur_next.clone()).load(layouter)?;
            LtChip::<F, NUM_BYTES>::construct(a.lt_tbl_key_cur_next.clone()).load(layouter)?;
        }
        for j in self.cfg.join.iter() {
            LtChip::<F, NUM_BYTES>::construct(j.lt_low.clone()).load(layouter)?;
            LtChip::<F, NUM_BYTES>::construct(j.lt_high.clone()).load(layouter)?;
        }

        // ---------- build host witnesses ----------
        let r: [&[Edge]; 5] = [r1, r2, r3, r4, r5];
        let n = r.iter().map(|x| x.len()).max().unwrap_or(0);
        if n == 0 {
            // trivial: output 0
            let cell = layouter.assign_region(
                || "out0",
                |mut region| {
                    let c = region.assign_advice(
                        || "out",
                        self.cfg.out,
                        0,
                        || Value::known(F::from(0)),
                    )?;
                    Ok(c)
                },
            )?;
            return Ok(cell);
        }

        // Build stage 5 (R5 -> T5): val = 1 per edge
        let r5_rows: Vec<[u64; 3]> = r5.iter().map(|e| [e.src, e.dst, 1u64]).collect();
        let r5_sorted = Self::sort_by_src(r5_rows.clone());
        let r5_run = Self::run_sum_by_src(&r5_sorted);
        let r5_emit = Self::emit_pairs(&r5_sorted, &r5_run);
        let t5_tbl = Self::build_tbl_from_emit(&r5_emit, r5.len());
        let t5_next = Self::key_next_from_tbl(&t5_tbl);
        let t5_map = Self::map_from_tbl(&t5_tbl);

        // helper: keys for gap proof (include 0 boundary and PAD)
        let mut t5_keys: Vec<u64> = t5_tbl
            .iter()
            .map(|p| p[0])
            .filter(|&k| k != PAD_KEY)
            .collect();
        t5_keys.push(0);
        t5_keys.push(PAD_KEY);
        t5_keys.sort();
        t5_keys.dedup();

        // Stage 4 join to T5, then aggregate -> T4
        let (
            r4_in,
            r4_low,
            r4_high,
            r4_val,
            r4_sorted,
            r4_run,
            r4_emit,
            t4_tbl,
            t4_next,
            t4_map,
            t4_keys,
        ) = {
            let mut rows = Vec::with_capacity(r4.len());
            let mut inx = vec![0u64; r4.len()];
            let mut low = vec![0u64; r4.len()];
            let mut high = vec![PAD_KEY; r4.len()];
            let mut val = vec![0u64; r4.len()];

            for (i, e) in r4.iter().enumerate() {
                let dst = e.dst;
                let (b, lo, hi) = Self::gap_witness(&t5_keys, dst);
                inx[i] = b;
                low[i] = lo;
                high[i] = hi;
                if b == 1 {
                    val[i] = *t5_map.get(&dst).unwrap_or(&0);
                } else {
                    val[i] = 0;
                }
                rows.push([e.src, e.dst, val[i]]);
            }

            let sorted = Self::sort_by_src(rows.clone());
            let run = Self::run_sum_by_src(&sorted);
            let emit = Self::emit_pairs(&sorted, &run);
            let tbl = Self::build_tbl_from_emit(&emit, r4.len());
            let kn = Self::key_next_from_tbl(&tbl);
            let mp = Self::map_from_tbl(&tbl);

            let mut keys: Vec<u64> = tbl.iter().map(|p| p[0]).filter(|&k| k != PAD_KEY).collect();
            keys.push(0);
            keys.push(PAD_KEY);
            keys.sort();
            keys.dedup();

            (inx, low, high, val, sorted, run, emit, tbl, kn, mp, keys)
        };

        // Stage 3 join to T4, aggregate -> T3
        let (
            r3_in,
            r3_low,
            r3_high,
            r3_val,
            r3_sorted,
            r3_run,
            r3_emit,
            t3_tbl,
            t3_next,
            t3_map,
            t3_keys,
        ) = {
            let mut rows = Vec::with_capacity(r3.len());
            let mut inx = vec![0u64; r3.len()];
            let mut low = vec![0u64; r3.len()];
            let mut high = vec![PAD_KEY; r3.len()];
            let mut val = vec![0u64; r3.len()];

            for (i, e) in r3.iter().enumerate() {
                let dst = e.dst;
                let (b, lo, hi) = Self::gap_witness(&t4_keys, dst);
                inx[i] = b;
                low[i] = lo;
                high[i] = hi;
                if b == 1 {
                    val[i] = *t4_map.get(&dst).unwrap_or(&0);
                } else {
                    val[i] = 0;
                }
                rows.push([e.src, e.dst, val[i]]);
            }

            let sorted = Self::sort_by_src(rows.clone());
            let run = Self::run_sum_by_src(&sorted);
            let emit = Self::emit_pairs(&sorted, &run);
            let tbl = Self::build_tbl_from_emit(&emit, r3.len());
            let kn = Self::key_next_from_tbl(&tbl);
            let mp = Self::map_from_tbl(&tbl);

            let mut keys: Vec<u64> = tbl.iter().map(|p| p[0]).filter(|&k| k != PAD_KEY).collect();
            keys.push(0);
            keys.push(PAD_KEY);
            keys.sort();
            keys.dedup();

            (inx, low, high, val, sorted, run, emit, tbl, kn, mp, keys)
        };

        // Stage 2 join to T3, aggregate -> T2
        let (
            r2_in,
            r2_low,
            r2_high,
            r2_val,
            r2_sorted,
            r2_run,
            r2_emit,
            t2_tbl,
            t2_next,
            t2_map,
            t2_keys,
        ) = {
            let mut rows = Vec::with_capacity(r2.len());
            let mut inx = vec![0u64; r2.len()];
            let mut low = vec![0u64; r2.len()];
            let mut high = vec![PAD_KEY; r2.len()];
            let mut val = vec![0u64; r2.len()];

            for (i, e) in r2.iter().enumerate() {
                let dst = e.dst;
                let (b, lo, hi) = Self::gap_witness(&t3_keys, dst);
                inx[i] = b;
                low[i] = lo;
                high[i] = hi;
                if b == 1 {
                    val[i] = *t3_map.get(&dst).unwrap_or(&0);
                } else {
                    val[i] = 0;
                }
                rows.push([e.src, e.dst, val[i]]);
            }

            let sorted = Self::sort_by_src(rows.clone());
            let run = Self::run_sum_by_src(&sorted);
            let emit = Self::emit_pairs(&sorted, &run);
            let tbl = Self::build_tbl_from_emit(&emit, r2.len());
            let kn = Self::key_next_from_tbl(&tbl);
            let mp = Self::map_from_tbl(&tbl);

            let mut keys: Vec<u64> = tbl.iter().map(|p| p[0]).filter(|&k| k != PAD_KEY).collect();
            keys.push(0);
            keys.push(PAD_KEY);
            keys.sort();
            keys.dedup();

            (inx, low, high, val, sorted, run, emit, tbl, kn, mp, keys)
        };

        // Stage 1 join to T2, final sum
        let (r1_in, r1_low, r1_high, r1_val, answer) = {
            let mut inx = vec![0u64; r1.len()];
            let mut low = vec![0u64; r1.len()];
            let mut high = vec![PAD_KEY; r1.len()];
            let mut val = vec![0u64; r1.len()];

            let mut total: u128 = 0;
            for (i, e) in r1.iter().enumerate() {
                let dst = e.dst;
                let (b, lo, hi) = Self::gap_witness(&t2_keys, dst);
                inx[i] = b;
                low[i] = lo;
                high[i] = hi;
                if b == 1 {
                    val[i] = *t2_map.get(&dst).unwrap_or(&0);
                } else {
                    val[i] = 0;
                }
                total += val[i] as u128;
            }
            (inx, low, high, val, total as u64)
        };

        // ---------- circuit assignment (single big region) ----------
        let cfg = &self.cfg;

        let out_cell = layouter.assign_region(
            || "graph_join5_witness",
            |mut region| {
                // assign base R1..R5
                let all_edges = [&r1, &r2, &r3, &r4, &r5];
                for (k, rel) in all_edges.iter().enumerate() {
                    for (i, e) in rel.iter().enumerate() {
                        region.assign_advice(
                            || "src",
                            cfg.r[k][0],
                            i,
                            || Value::known(F::from(e.src)),
                        )?;
                        region.assign_advice(
                            || "dst",
                            cfg.r[k][1],
                            i,
                            || Value::known(F::from(e.dst)),
                        )?;
                    }
                    // optional: pad unused rows up to n to keep things defined (not strictly needed if you never query them)
                }

                // assign join witnesses (R1..R4)
                let join_w = [
                    (&r1_in, &r1_low, &r1_high, &r1_val, r1.len()),
                    (&r2_in, &r2_low, &r2_high, &r2_val, r2.len()),
                    (&r3_in, &r3_low, &r3_high, &r3_val, r3.len()),
                    (&r4_in, &r4_low, &r4_high, &r4_val, r4.len()),
                ];

                for k in 0..4 {
                    let (inx, low, high, val, len) = join_w[k];
                    let jc = &cfg.join[k];

                    let lt_low_chip = LtChip::<F, NUM_BYTES>::construct(jc.lt_low.clone());
                    let lt_high_chip = LtChip::<F, NUM_BYTES>::construct(jc.lt_high.clone());

                    for i in 0..len {
                        jc.q_lookup.enable(&mut region, i)?;
                        jc.q_lookup_complex.enable(&mut region, i)?;

                        region.assign_advice(
                            || "in_next",
                            jc.in_next,
                            i,
                            || Value::known(F::from(inx[i])),
                        )?;
                        region.assign_advice(
                            || "low",
                            jc.low,
                            i,
                            || Value::known(F::from(low[i])),
                        )?;
                        region.assign_advice(
                            || "high",
                            jc.high,
                            i,
                            || Value::known(F::from(high[i])),
                        )?;
                        region.assign_advice(
                            || "val",
                            jc.val,
                            i,
                            || Value::known(F::from(val[i])),
                        )?;

                        // these are only enforced when (1-in_next) by the gate expression inside lt config
                        lt_low_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(low[i])),
                            Value::known(F::from(all_edges[k][i].dst)),
                        )?;
                        lt_high_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(all_edges[k][i].dst)),
                            Value::known(F::from(high[i])),
                        )?;
                    }
                }

                // function to assign an agg stage (sorted+run_sum+emit+tbl)
                let mut assign_agg_stage = |a: &AggConfig<F>,
                                            base_src: Column<Advice>,
                                            base_dst: Column<Advice>,
                                            base_len: usize,
                                            sorted_rows: &Vec<[u64; 3]>,
                                            run: &Vec<u64>,
                                            emit: &Vec<[u64; 2]>,
                                            tbl: &Vec<[u64; 2]>,
                                            key_next: &Vec<u64>,
                                            val_in_vec: Option<&Vec<u64>>|
                 -> Result<(), Error> {
                    // val_in
                    if let Some(vals) = val_in_vec {
                        for i in 0..base_len {
                            region.assign_advice(
                                || "val_in",
                                a.val_in,
                                i,
                                || Value::known(F::from(vals[i])),
                            )?;
                        }
                    }

                    // sorted triple + sentinel row (for same_next)
                    for i in 0..base_len {
                        region.assign_advice(
                            || "sorted_src",
                            a.sorted[0],
                            i,
                            || Value::known(F::from(sorted_rows[i][0])),
                        )?;
                        region.assign_advice(
                            || "sorted_dst",
                            a.sorted[1],
                            i,
                            || Value::known(F::from(sorted_rows[i][1])),
                        )?;
                        region.assign_advice(
                            || "sorted_val",
                            a.sorted[2],
                            i,
                            || Value::known(F::from(sorted_rows[i][2])),
                        )?;
                    }
                    // sentinel at row=base_len
                    region.assign_advice(
                        || "sorted_src_s",
                        a.sorted[0],
                        base_len,
                        || Value::known(F::from(PAD_KEY)),
                    )?;
                    region.assign_advice(
                        || "sorted_dst_s",
                        a.sorted[1],
                        base_len,
                        || Value::known(F::from(PAD_KEY)),
                    )?;
                    region.assign_advice(
                        || "sorted_val_s",
                        a.sorted[2],
                        base_len,
                        || Value::known(F::from(0u64)),
                    )?;

                    // run_sum
                    for i in 0..base_len {
                        region.assign_advice(
                            || "run_sum",
                            a.run_sum,
                            i,
                            || Value::known(F::from(run[i])),
                        )?;
                    }

                    // emit pairs
                    for i in 0..base_len {
                        region.assign_advice(
                            || "emit_k",
                            a.emit_pair[0],
                            i,
                            || Value::known(F::from(emit[i][0])),
                        )?;
                        region.assign_advice(
                            || "emit_v",
                            a.emit_pair[1],
                            i,
                            || Value::known(F::from(emit[i][1])),
                        )?;
                    }

                    // tbl pairs + key_next
                    for i in 0..base_len {
                        region.assign_advice(
                            || "tbl_k",
                            a.tbl_pair[0],
                            i,
                            || Value::known(F::from(tbl[i][0])),
                        )?;
                        region.assign_advice(
                            || "tbl_v",
                            a.tbl_pair[1],
                            i,
                            || Value::known(F::from(tbl[i][1])),
                        )?;
                        region.assign_advice(
                            || "tbl_kn",
                            a.tbl_key_next,
                            i,
                            || Value::known(F::from(key_next[i])),
                        )?;
                    }

                    // map_pair has length base_len + 1 (we already use row=base_len as sentinel elsewhere)
                    if base_len > 0 {
                        // row 0 = (0,0)
                        region.assign_advice(
                            || "map_k0",
                            a.map_pair[0],
                            0,
                            || Value::known(F::from(0u64)),
                        )?;
                        region.assign_advice(
                            || "map_v0",
                            a.map_pair[1],
                            0,
                            || Value::known(F::from(0u64)),
                        )?;

                        // rows 1..=base_len : copy tbl_pair[0..base_len-1]
                        for i in 0..base_len {
                            region.assign_advice(
                                || "map_k",
                                a.map_pair[0],
                                i + 1,
                                || Value::known(F::from(tbl[i][0])),
                            )?;
                            region.assign_advice(
                                || "map_v",
                                a.map_pair[1],
                                i + 1,
                                || Value::known(F::from(tbl[i][1])),
                            )?;
                        }

                        // map_key_next[0..base_len-1] = map_key[i+1], map_key_next[base_len] = PAD_KEY
                        for i in 0..base_len {
                            let nextk = if i + 1 <= base_len {
                                // always true here
                                if i + 1 == base_len {
                                    PAD_KEY
                                } else {
                                    tbl[i][0]
                                } // careful: map_key[i+1] is tbl[i][0]
                            } else {
                                PAD_KEY
                            };
                            // Easier: define directly by map_key shift:
                            // map_key_next[i] = map_key[i+1]
                            let v = if i == 0 { tbl[0][0] } else { tbl[i][0] };
                            region.assign_advice(
                                || "map_kn",
                                a.map_key_next,
                                i,
                                || Value::known(F::from(v)),
                            )?;
                        }
                        region.assign_advice(
                            || "map_kn_last",
                            a.map_key_next,
                            base_len,
                            || Value::known(F::from(PAD_KEY)),
                        )?;
                    } else {
                        // base_len == 0: just one row
                        region.assign_advice(
                            || "map_k0",
                            a.map_pair[0],
                            0,
                            || Value::known(F::from(0u64)),
                        )?;
                        region.assign_advice(
                            || "map_v0",
                            a.map_pair[1],
                            0,
                            || Value::known(F::from(0u64)),
                        )?;
                        region.assign_advice(
                            || "map_kn0",
                            a.map_key_next,
                            0,
                            || Value::known(F::from(PAD_KEY)),
                        )?;
                    }

                    // table rows 0..=base_len
                    for i in 0..=base_len {
                        a.q_map_tbl.enable(&mut region, i)?;
                    }
                    a.q_map_first.enable(&mut region, 0)?;
                    for i in 0..base_len {
                        a.q_map_link.enable(&mut region, i)?;
                        a.q_map_shift.enable(&mut region, i)?;
                    }
                    a.q_map_last.enable(&mut region, base_len)?;

                    // enable perms
                    for i in 0..base_len {
                        a.perm_sort.q_perm1.enable(&mut region, i)?;
                        a.perm_sort.q_perm2.enable(&mut region, i)?;

                        a.perm_tbl.q_perm1.enable(&mut region, i)?;
                        a.perm_tbl.q_perm2.enable(&mut region, i)?;
                    }

                    // sortedness gates: sorted triple and tbl keys
                    for i in 0..base_len.saturating_sub(1) {
                        a.q_sort.enable(&mut region, i)?;
                        a.q_tbl_sort.enable(&mut region, i)?;
                        a.q_tbl_shift.enable(&mut region, i)?;
                    }
                    if base_len > 0 {
                        a.q_tbl_last.enable(&mut region, base_len - 1)?;
                    }

                    // group run_sum gates
                    if base_len > 0 {
                        a.q_first.enable(&mut region, 0)?;
                        a.q_emit.enable(&mut region, 0)?;
                    }
                    for i in 1..base_len {
                        a.q_accu.enable(&mut region, i)?;
                        a.q_emit.enable(&mut region, i)?;
                    }

                    // assign IsZero helpers (same_prev, same_next, eqs)
                    let iz_same_prev_chip = IsZeroChip::construct(a.iz_same_prev.clone());
                    let iz_same_next_chip = IsZeroChip::construct(a.iz_same_next.clone());
                    let iz_src_eq_chip = IsZeroChip::construct(a.iz_src_eq.clone());
                    let iz_tbl_eq_chip = IsZeroChip::construct(a.iz_tbl_key_eq.clone());

                    // assign LtChip for sortedness checks
                    let lt_src_chip = LtChip::<F, NUM_BYTES>::construct(a.lt_src_cur_next.clone());
                    let lt_tbl_chip =
                        LtChip::<F, NUM_BYTES>::construct(a.lt_tbl_key_cur_next.clone());

                    for i in 0..base_len.saturating_sub(1) {
                        lt_src_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(sorted_rows[i][0])),
                            Value::known(F::from(sorted_rows[i + 1][0])),
                        )?;
                        lt_tbl_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(tbl[i][0])),
                            Value::known(F::from(tbl[i + 1][0])),
                        )?;
                    }

                    // same_prev: i>=1
                    for i in 1..base_len {
                        let diff = F::from(sorted_rows[i][0]) - F::from(sorted_rows[i - 1][0]);
                        iz_same_prev_chip.assign(&mut region, i, Value::known(diff))?;
                    }
                    // same_next: i=0..base_len-1 (needs sentinel)
                    for i in 0..base_len {
                        let next_src = if i + 1 < base_len {
                            sorted_rows[i + 1][0]
                        } else {
                            PAD_KEY
                        };
                        let diff = F::from(next_src) - F::from(sorted_rows[i][0]);
                        iz_same_next_chip.assign(&mut region, i, Value::known(diff))?;
                    }

                    // src_eq for sortedness check: uses next-cur on rows 0..base_len-2
                    for i in 0..base_len.saturating_sub(1) {
                        let diff = F::from(sorted_rows[i + 1][0]) - F::from(sorted_rows[i][0]);
                        iz_src_eq_chip.assign(&mut region, i, Value::known(diff))?;
                    }
                    // tbl_eq similarly
                    for i in 0..base_len.saturating_sub(1) {
                        let diff = F::from(tbl[i + 1][0]) - F::from(tbl[i][0]);
                        iz_tbl_eq_chip.assign(&mut region, i, Value::known(diff))?;
                    }

                    Ok(())
                };

                // Stage5 agg[0] uses R5 columns, val_in must be 1 vector
                let r5_vals = vec![1u64; r5.len()];
                assign_agg_stage(
                    &cfg.agg[0],
                    cfg.r[4][0],
                    cfg.r[4][1],
                    r5.len(),
                    &r5_sorted,
                    &r5_run,
                    &r5_emit,
                    &t5_tbl,
                    &t5_next,
                    Some(&r5_vals),
                )?;

                // Stage4 agg[1] uses R4 columns and join[3].val as val_in
                assign_agg_stage(
                    &cfg.agg[1],
                    cfg.r[3][0],
                    cfg.r[3][1],
                    r4.len(),
                    &r4_sorted,
                    &r4_run,
                    &r4_emit,
                    &t4_tbl,
                    &t4_next,
                    Some(&r4_val),
                )?;

                // Stage3 agg[2]
                assign_agg_stage(
                    &cfg.agg[2],
                    cfg.r[2][0],
                    cfg.r[2][1],
                    r3.len(),
                    &r3_sorted,
                    &r3_run,
                    &r3_emit,
                    &t3_tbl,
                    &t3_next,
                    Some(&r3_val),
                )?;

                // Stage2 agg[3]
                assign_agg_stage(
                    &cfg.agg[3],
                    cfg.r[1][0],
                    cfg.r[1][1],
                    r2.len(),
                    &r2_sorted,
                    &r2_run,
                    &r2_emit,
                    &t2_tbl,
                    &t2_next,
                    Some(&r2_val),
                )?;

                // ---------- lookups for membership + gap + mapping ----------
                // We implement them exactly like your Q3 pattern:
                // - membership: (in_next * dst) in tbl_key
                // - gap pair: (1-in_next)*low in tbl_key AND (1-in_next)*high in tbl_key_next
                // - mapping: (in_next*dst, in_next*val) exists in (tbl_key, tbl_val)

                // helper to add lookups per join step against a given agg table
                let mut add_join_lookups =
                    |step_idx: usize, table: &AggConfig<F>, rel_idx: usize, rel_len: usize| {
                        let j = cfg.join[step_idx].clone();

                        // membership
                        meta::lookup_any_placeholder::<F>(); // (no-op, just to keep structure clear)
                        Ok::<(), Error>(())
                    };

                // NOTE: Halo2 lookup definitions must be in `configure`, not here.
                // So we only *enable selectors* here (the lookups are already wired in configure below).
                //
                // Therefore: we enable q_lookup_complex on all rows of each join relation.
                for i in 0..r1.len() {
                    cfg.join[0].q_lookup_complex.enable(&mut region, i)?;
                }
                for i in 0..r2.len() {
                    cfg.join[1].q_lookup_complex.enable(&mut region, i)?;
                }
                for i in 0..r3.len() {
                    cfg.join[2].q_lookup_complex.enable(&mut region, i)?;
                }
                for i in 0..r4.len() {
                    cfg.join[3].q_lookup_complex.enable(&mut region, i)?;
                }

                // Final sum over R1: sum[i]=sum[i-1]+val[i]
                let mut running: u128 = 0;
                for i in 0..r1.len() {
                    running += r1_val[i] as u128;
                    region.assign_advice(
                        || "sum",
                        cfg.sum,
                        i,
                        || Value::known(F::from(running as u64)),
                    )?;
                }
                if r1.len() > 0 {
                    cfg.q_sum_first.enable(&mut region, 0)?;
                }
                for i in 1..r1.len() {
                    cfg.q_sum_accu.enable(&mut region, i)?;
                }

                // output cell = sum[last]
                let out_row = r1.len() - 1;
                let out_cell = region.assign_advice(
                    || "out",
                    cfg.out,
                    out_row,
                    || Value::known(F::from(answer)),
                )?;

                Ok(out_cell)
            },
        )?;

        Ok(out_cell)
    }
}

// ---------------- IMPORTANT: lookups must be defined in configure ----------------
// We attach them by implementing Circuit::configure below using meta.lookup_any.

impl<F: Field + Ord> Circuit<F> for GraphJoin5Circuit<F> {
    type Config = GraphJoin5Config<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let mut cfg = GraphJoin5Chip::<F>::configure(meta);

        // Convenience aliases:
        // tables: agg[0]=T5 from R5, agg[1]=T4, agg[2]=T3, agg[3]=T2
        let t5 = cfg.agg[0].clone();
        let t4 = cfg.agg[1].clone();
        let t3 = cfg.agg[2].clone();
        let t2 = cfg.agg[3].clone();

        // For each join step k, join[k] references next table (k=0 uses T2, k=1 uses T3, k=2 uses T4, k=3 uses T5)

        // Helper to add membership+gap+mapping lookups:
        let mut add_join =
            |k: usize, rel_dst: Column<Advice>, j: JoinConfig<F>, t: AggConfig<F>| {
                // gap pair: (1-in)*low in key  AND  (1-in)*high in key_next
                meta.lookup_any(format!("gap pair step {}", k), move |m| {
                    let q_in = m.query_selector(j.q_lookup_complex);
                    let inx = m.query_advice(j.in_next, Rotation::cur());
                    let gate = q_in * (Expression::Constant(F::ONE) - inx);

                    let low = m.query_advice(j.low, Rotation::cur());
                    let high = m.query_advice(j.high, Rotation::cur());

                    let q_tbl = m.query_selector(t.q_map_tbl);
                    let key = m.query_advice(t.map_pair[0], Rotation::cur());
                    let keyn = m.query_advice(t.map_key_next, Rotation::cur());

                    vec![
                        (gate.clone() * low, q_tbl.clone() * key),
                        (gate * high, q_tbl * keyn),
                    ]
                });

                // mapping: (in*dst, val) exists in (map_key, map_val)
                // when in=0, val is forced 0 by your join gate, so tuple becomes (0,0) which exists at row0
                meta.lookup_any(format!("map step {}", k), move |m| {
                    let q_in = m.query_selector(j.q_lookup_complex);
                    let inx = m.query_advice(j.in_next, Rotation::cur());

                    let dst = m.query_advice(rel_dst, Rotation::cur());
                    let v = m.query_advice(j.val, Rotation::cur());

                    let q_tbl = m.query_selector(t.q_map_tbl);
                    let tk = m.query_advice(t.map_pair[0], Rotation::cur());
                    let tv = m.query_advice(t.map_pair[1], Rotation::cur());

                    vec![
                        (q_in.clone() * inx.clone() * dst, q_tbl.clone() * tk),
                        (q_in * v, q_tbl * tv),
                    ]
                });
            };

        // R1 joins to T2
        add_join(0, cfg.r[0][1], cfg.join[0].clone(), t2);
        // R2 joins to T3
        add_join(1, cfg.r[1][1], cfg.join[1].clone(), t3);
        // R3 joins to T4
        add_join(2, cfg.r[2][1], cfg.join[2].clone(), t4);
        // R4 joins to T5
        add_join(3, cfg.r[3][1], cfg.join[3].clone(), t5);

        cfg
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = GraphJoin5Chip::construct(config);
        let out_cell = chip.assign(
            &mut layouter,
            &self.r1,
            &self.r2,
            &self.r3,
            &self.r4,
            &self.r5,
        )?;
        chip.expose_public(&mut layouter, out_cell, 0)?;
        Ok(())
    }
}

// dummy helper only to keep the file compiling if you copy blocks around
mod meta {
    #[allow(dead_code)]
    pub fn lookup_any_placeholder<F>() {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::graph_data_processing::read_edges;

    use halo2_proofs::dev::MockProver;
    use halo2curves::pasta::Fp;

    #[test]
    fn test_1() {
        let base_path = "/home2/binbin/PoneglyphDB/src/graph_data/facebook";

        let mut r1 = read_edges(&format!("{}/R1.tsv", base_path)).unwrap();
        let mut r2 = read_edges(&format!("{}/R2.tsv", base_path)).unwrap();
        let mut r3 = read_edges(&format!("{}/R3.tsv", base_path)).unwrap();
        let mut r4 = read_edges(&format!("{}/R4.tsv", base_path)).unwrap();
        let mut r5 = read_edges(&format!("{}/R5.tsv", base_path)).unwrap();

        // // take first 1000
        // r1.truncate(1000);
        // r2.truncate(1000);
        // r3.truncate(1000);
        // r4.truncate(1000);
        // r5.truncate(1000);

        // public input = COUNT(*). We compute it with the same DP as the circuit witness uses.
        // (You can also compute by brute force for small sizes.)
        let cnt = dp_count(&r1, &r2, &r3, &r4, &r5);

        let circuit = GraphJoin5Circuit::<Fp> {
            r1,
            r2,
            r3,
            r4,
            r5,
            _marker: PhantomData,
        };

        let public_input = vec![Fp::from(cnt)];
        let k = 16;

        let prover = MockProver::run(k, &circuit, vec![public_input]).unwrap();
        prover.assert_satisfied();
    }

    fn dp_count(r1: &[Edge], r2: &[Edge], r3: &[Edge], r4: &[Edge], r5: &[Edge]) -> u64 {
        use std::collections::HashMap;

        // T5
        let mut t5 = HashMap::<u64, u64>::new();
        for e in r5 {
            *t5.entry(e.src).or_insert(0) += 1;
        }

        // T4
        let mut t4 = HashMap::<u64, u64>::new();
        for e in r4 {
            let v = *t5.get(&e.dst).unwrap_or(&0);
            *t4.entry(e.src).or_insert(0) += v;
        }

        // T3
        let mut t3m = HashMap::<u64, u64>::new();
        for e in r3 {
            let v = *t4.get(&e.dst).unwrap_or(&0);
            *t3m.entry(e.src).or_insert(0) += v;
        }

        // T2
        let mut t2m = HashMap::<u64, u64>::new();
        for e in r2 {
            let v = *t3m.get(&e.dst).unwrap_or(&0);
            *t2m.entry(e.src).or_insert(0) += v;
        }

        // answer
        let mut ans: u128 = 0;
        for e in r1 {
            ans += *t2m.get(&e.dst).unwrap_or(&0) as u128;
        }
        ans as u64
    }
}
