use halo2_proofs::halo2curves::ff::PrimeField;
use halo2_proofs::plonk::Expression;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};

// IMPORTANT: use the real dataset Edge type
use crate::data::graph_data_processing::Edge;

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

// Use 8 bytes for robustness on real datasets; adjust if you know your ID width.
const NUM_BYTES: usize = 8;

// Reserve key=0 for the dummy map row
const SHIFT_ID: u64 = 1;

// Sentinel/pad key goes last in ASC order
const PAD_KEY: u64 = u64::MAX;
const PAD_VAL: u64 = 0;

/// Test hook, off in every benchmark path: when set, the prover moves one
/// joinable root tuple to the residual side and re-reduces the two children
/// around it, so the partition still passes the Conservation Check and every
/// other check in the file, and only condition (4) can catch it. This is
/// exactly the cheat a residual-side-only argument misses, so the negative
/// test in this module is what shows the Cardinality Preservation Check is not
/// vacuous.
pub static HIDE_ONE_CLEAN_TUPLE: AtomicBool = AtomicBool::new(false);

/// Test hook, off in every benchmark path: when set, the prover skips the
/// semijoin reduction entirely and declares every real tuple clean, leaving the
/// residual section of every partition empty. Conservation still holds and both
/// channels of condition (4) then agree trivially, so this is exactly the escape
/// that Pairwise Consistency has to close, and the third direction of the test
/// in this module is what shows condition (3) closes it.
pub static MARK_ALL_CLEAN: AtomicBool = AtomicBool::new(false);

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

/// ---------- Aggregator config (same pattern as GraphJoin5) ----------
///
/// Two multiplicity channels ride the same group-by: `*_cln` is the clean
/// channel, everything else is the input channel of `g_sql1_obj.rs`.
#[derive(Clone, Debug)]
struct AggConfig<F: Field + Ord> {
    // Input tuple: (src, dst, val, val_cln)
    val_in: Column<Advice>,
    val_in_cln: Column<Advice>,

    // Sorted triple columns, plus the clean channel's value
    sorted: [Column<Advice>; 3],
    sorted_cln: Column<Advice>,

    // perm: (src, dst, val_in, val_in_cln) <-> (sorted, sorted_cln)
    perm_sort: PermAnyConfig,

    // sortedness check on sorted[0] (src)
    q_sort: Selector,
    q_sort_last: Selector,
    lt_src_cur_next: LtConfig<F, NUM_BYTES>,
    iz_src_eq: IsZeroConfig<F>,

    // group helpers on sorted src
    q_first: Selector,
    q_accu: Selector,
    q_emit: Selector,
    run_sum: Column<Advice>,
    run_sum_cln: Column<Advice>,
    iz_same_prev: IsZeroConfig<F>,
    iz_same_next: IsZeroConfig<F>,

    // emitted padded group rows (key,val,val_cln) over rows of sorted
    emit_pair: [Column<Advice>; 2],
    emit_cln: Column<Advice>,

    // table (key,val,val_cln) padded, permuted from the emitted rows
    tbl_pair: [Column<Advice>; 2],
    tbl_cln: Column<Advice>,
    // DEAD, kept on purpose. Both gap lookups read `map_key_next`, which has its
    // own shift/last gates, so nothing consumes `tbl_key_next` and it could be
    // deleted without changing the satisfying set. It is inherited input-channel
    // machinery that `g_sql1_obj.rs` carries identically, so it cancels in the
    // pairwise cost delta this file is measured by; removing it here alone would
    // make that delta report a saving the one-pass gate did not make.
    tbl_key_next: Column<Advice>,
    perm_tbl: PermAnyConfig,

    // prove tbl_pair keys are sorted and tbl_key_next is the "next" key
    q_tbl_sort: Selector,
    lt_tbl_key_cur_next: LtConfig<F, NUM_BYTES>,
    iz_tbl_key_eq: IsZeroConfig<F>,

    q_tbl_shift: Selector, // enforce tbl_key_next[i] == tbl_key[i+1]
    q_tbl_last: Selector,  // enforce tbl_key_next[last] == PAD_KEY

    // map table used by joins (has dummy row 0)
    map_pair: [Column<Advice>; 2], // (key,val) with row0=(0,0), rows 1..=n copy tbl
    map_cln: Column<Advice>,       // the clean channel of the same table, row0=0
    map_key_next: Column<Advice>,  // next(key) for map_pair[0]

    q_map_tbl: Selector,   // enable table rows 0..=n (for lookup gating)
    q_map_first: Selector, // enforce (map_pair, map_cln)[0] == (0,0,0)
    q_map_link: Selector,  // enforce map row i+1 == tbl row i
    q_map_shift: Selector, // enforce map_key_next[i] == map_key[i+1]
    q_map_last: Selector,  // enforce map_key_next[last] == PAD_KEY
}

/// ---------- Join config (same pattern as GraphJoin5) ----------
#[derive(Clone, Debug)]
struct JoinConfig<F: Field + Ord> {
    // per-row membership flag: dst exists in next table?
    in_next: Column<Advice>,
    // gap witnesses if in_next == 0
    low: Column<Advice>,
    high: Column<Advice>,
    // attached values from next table (if in_next==1 else 0), one per channel
    val: Column<Advice>,
    val_cln: Column<Advice>,

    // selectors
    q_lookup: Selector,
    q_lookup_complex: Selector,

    // LT checks for low < dst and dst < high when not in_next
    lt_low: LtConfig<F, NUM_BYTES>,
    lt_high: LtConfig<F, NUM_BYTES>,
}

/// ---------- Main circuit config ----------
#[derive(Clone, Debug)]
pub struct Path3OrdConfig<F: Field + Ord> {
    instance: Column<Instance>,

    // base edges r1,r2,r3: [src, dst] (all equal to Edge table)
    r: [[Column<Advice>; 2]; 3],
    // r1, r2, r3 are three occurrences of ONE table: pin them row-wise equal
    q_self_join: Selector,

    // join steps: r1->T2, r2->T3
    join: [JoinConfig<F>; 2],

    // aggregators: agg[0]=r3->T3, agg[1]=r2->T2
    agg: [AggConfig<F>; 2],

    // ---------------- Cardinality Preservation Check ----------------
    // clean indicator per base row of r1, r2, r3
    cflag: [Column<Advice>; 3],

    // partition side of each Conservation Check: (src, dst, flag), laid out
    // as [R^c rows | R^r rows]
    part: [[Column<Advice>; 3]; 3],
    perm_cons: [PermAnyConfig; 3],
    q_cln_flag: [Selector; 3], // rows of R^c: flag == 1
    q_res_flag: [Selector; 3], // rows of R^r: flag == 0

    // ---------------- condition (3), Pairwise Consistency ----------------
    // one complex selector per relation, enabled over exactly the clean rows
    // [0, |R^c|) of that relation's partition group. These are the lookup input
    // gates, so they cannot be the simple q_cln_flag selectors above, and the
    // same selector serves as the table gate of the opposite direction: the four
    // lookups run between the partition key columns themselves, with no
    // intermediate key table to forge.
    q_pw_cln: [Selector; 3],

    // the clean indicator of r1 / r2 must satisfy the query's local predicate
    q_cln_pred: Selector,

    // ordering checks
    q_ord1: Selector,              // enable lt_ab on r1 rows
    q_ord2: Selector,              // enable lt_bc on r2 rows
    lt_ab: LtConfig<F, NUM_BYTES>, // r1.src < r1.dst
    lt_bc: LtConfig<F, NUM_BYTES>, // r2.src < r2.dst

    // glue gates (soundness) + filtered values
    q_r3_one: Selector,    // enforce agg0.val_in = 1, agg0.val_in_cln = c_3
    q_r2_filter: Selector, // enforce agg1.val_in = join1.val * lt_bc (both channels)
    q_r1_filter: Selector, // enforce fval_r1 = join0.val * lt_ab (both channels)

    fval_r1: Column<Advice>,     // filtered contribution per r1 row
    fval_r1_cln: Column<Advice>, // the same on the clean channel

    // final sums, one per channel
    q_sum_first: Selector,
    q_sum_accu: Selector,
    sum: Column<Advice>,
    sum_cln: Column<Advice>,

    // condition (4): the two join cardinalities agree
    q_card_eq: Selector,

    // output constrained to equal sum at chosen row
    out: Column<Advice>,
    q_out: Selector,
}

#[derive(Clone)]
pub struct Path3OrdCircuit<F: Field + Ord> {
    pub edges: Vec<Edge>,
    pub _marker: PhantomData<F>,
}

impl<F: Field + Ord> Default for Path3OrdCircuit<F> {
    fn default() -> Self {
        Self {
            edges: vec![],
            _marker: PhantomData,
        }
    }
}

pub struct Path3OrdChip<F: Field + Ord> {
    cfg: Path3OrdConfig<F>,
}

impl<F: Field + Ord> Path3OrdChip<F> {
    pub fn construct(cfg: Path3OrdConfig<F>) -> Self {
        Self { cfg }
    }

    // ---------------- helpers (host-side) ----------------

    fn sort_by_src(mut rows: Vec<[u64; 4]>) -> Vec<[u64; 4]> {
        // stable tie-breaking doesn't matter for correctness; (src, dst, val, val_cln)
        // just needs to be a permutation
        rows.sort_by_key(|r| r[0]);
        rows
    }

    /// Per-group running sums of the two channels over the key-sorted rows.
    ///
    /// COMPLETENESS LIMIT, not a soundness hole: the accumulators are u128 and
    /// the assigned cells are `acc as u64`, so past a group sum of 2^64 the
    /// witness is truncated while the in-circuit gate `run_sum_accu` adds in F
    /// with no reduction. The two then disagree and the gate simply REJECTS;
    /// there is no assignment in which the truncation and the field sum agree on
    /// a wrong value, so a prover cannot steer it. The same holds for the root
    /// sums (`running as u64`) and for the `saturating_mul` in the two filter
    /// witnesses, which contradict q_r2_filter / q_r1_filter rather than satisfy
    /// them. Widening the host arithmetic to F would raise the honest bound; it
    /// is deliberately not done here because no dataset in the paper is within
    /// 2^64 of it and the fix would touch every witness path in the file.
    fn run_sum_by_src(sorted: &[[u64; 4]]) -> (Vec<u64>, Vec<u64>) {
        let mut out = vec![0u64; sorted.len()];
        let mut out_cln = vec![0u64; sorted.len()];
        let mut acc: u128 = 0;
        let mut acc_cln: u128 = 0;
        let mut prev: Option<u64> = None;
        for (i, r) in sorted.iter().enumerate() {
            let src = r[0];
            if prev == Some(src) {
                acc += r[2] as u128;
                acc_cln += r[3] as u128;
            } else {
                acc = r[2] as u128;
                acc_cln = r[3] as u128;
            }
            out[i] = acc as u64;
            out_cln[i] = acc_cln as u64;
            prev = Some(src);
        }
        (out, out_cln)
    }

    /// One (key, sum_all, sum_cln) at the last row of each group, PAD elsewhere.
    fn emit_pairs(sorted: &[[u64; 4]], run: &[u64], run_cln: &[u64]) -> Vec<[u64; 3]> {
        let n = sorted.len();
        let mut out = vec![[PAD_KEY, PAD_VAL, PAD_VAL]; n];
        for i in 0..n {
            let cur = sorted[i][0];
            let next = if i + 1 < n { sorted[i + 1][0] } else { PAD_KEY };
            let is_last = next != cur;
            if is_last {
                out[i] = [cur, run[i], run_cln[i]];
            }
        }
        out
    }

    fn build_tbl_from_emit(emit: &[[u64; 3]], n: usize) -> Vec<[u64; 3]> {
        let mut rows: Vec<[u64; 3]> = emit.iter().copied().filter(|p| p[0] != PAD_KEY).collect();
        rows.sort_by_key(|p| p[0]);
        while rows.len() < n {
            rows.push([PAD_KEY, PAD_VAL, PAD_VAL]);
        }
        rows.truncate(n);
        rows
    }

    fn key_next_from_tbl(tbl: &[[u64; 3]]) -> Vec<u64> {
        let n = tbl.len();
        let mut out = vec![PAD_KEY; n];
        for i in 0..n {
            out[i] = if i + 1 < n { tbl[i + 1][0] } else { PAD_KEY };
        }
        out
    }

    /// key -> (sigma_all, sigma_cln), for the parent side of the edge.
    fn map_from_tbl(tbl: &[[u64; 3]]) -> HashMap<u64, (u64, u64)> {
        let mut m = HashMap::new();
        for [k, v, v_cln] in tbl.iter().copied() {
            if k == PAD_KEY {
                continue;
            }
            m.insert(k, (v, v_cln));
        }
        m
    }

    fn gap_witness(keys_sorted_with_pad: &[u64], x: u64) -> (u64, u64, u64) {
        // returns (in, low, high)
        match keys_sorted_with_pad.binary_search(&x) {
            Ok(_) => (1, 0, PAD_KEY),
            Err(idx) => {
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

    /// The clean side of the partition: the fully reduced instance.
    ///
    /// For a path the reduction is one bottom-up pass followed by one top-down
    /// pass. Bottom-up a tuple is alive iff its input-channel multiplicity is
    /// nonzero, i.e. iff it passes its own ordering predicate and its join key
    /// reaches a nonempty subtree. Top-down a tuple stays only if some kept
    /// parent tuple joins to it. What is left is exactly the set of tuples
    /// extendable to a full join witness, which is the honest prover's R^c;
    /// everything else goes to R^r.
    ///
    /// `hide_one` is the test hook. It drops one clean root tuple and
    /// re-reduces the two children around it, so the three partitions are still
    /// mutually consistent and only condition (4) can see the loss.
    fn clean_indicators(edges: &[(u64, u64)], hide_one: bool) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let n = edges.len();

        // input channel, bottom-up: sigma_all_3 = outdegree, sigma_all_2 = T2
        let mut outdeg: HashMap<u64, u64> = HashMap::new();
        for &(s, _) in edges.iter() {
            *outdeg.entry(s).or_insert(0) += 1;
        }
        let mut mu2 = vec![0u128; n];
        let mut sig2: HashMap<u64, u128> = HashMap::new();
        for (i, &(s, d)) in edges.iter().enumerate() {
            if s < d {
                mu2[i] = *outdeg.get(&d).unwrap_or(&0) as u128;
            }
            *sig2.entry(s).or_insert(0) += mu2[i];
        }
        let mut mu1 = vec![0u128; n];
        for (i, &(s, d)) in edges.iter().enumerate() {
            if s < d {
                mu1[i] = *sig2.get(&d).unwrap_or(&0);
            }
        }

        // top-down: the root keeps its alive tuples, then a child tuple stays
        // only if its own key is the join key of a kept parent tuple
        let mut cln1 = vec![0u64; n];
        for i in 0..n {
            cln1[i] = (mu1[i] > 0) as u64;
        }
        if hide_one {
            if let Some(i) = (0..n).find(|&i| cln1[i] == 1) {
                cln1[i] = 0;
            }
        }

        let d1: HashSet<u64> = (0..n)
            .filter(|&i| cln1[i] == 1)
            .map(|i| edges[i].1)
            .collect();
        let mut cln2 = vec![0u64; n];
        for i in 0..n {
            cln2[i] = (mu2[i] > 0 && d1.contains(&edges[i].0)) as u64;
        }

        let d2: HashSet<u64> = (0..n)
            .filter(|&i| cln2[i] == 1)
            .map(|i| edges[i].1)
            .collect();
        let mut cln3 = vec![0u64; n];
        for i in 0..n {
            // r3 is a leaf with no predicate, so aliveness is vacuous there
            cln3[i] = d2.contains(&edges[i].0) as u64;
        }

        (cln1, cln2, cln3)
    }

    /// The partition side of one Conservation Check: the relation's rows
    /// reordered as [R^c rows | R^r rows] with the pinned flag appended.
    /// Returns the rows and |R^c|.
    fn partition_rows(edges: &[(u64, u64)], flags: &[u64]) -> (Vec<[u64; 3]>, usize) {
        let mut rows: Vec<[u64; 3]> = Vec::with_capacity(edges.len());
        for (i, &(s, d)) in edges.iter().enumerate() {
            if flags[i] == 1 {
                rows.push([s, d, 1]);
            }
        }
        let n_cln = rows.len();
        for (i, &(s, d)) in edges.iter().enumerate() {
            if flags[i] == 0 {
                rows.push([s, d, 0]);
            }
        }
        (rows, n_cln)
    }

    // ---------------- configure gadgets ----------------

    fn configure_agg(
        meta: &mut ConstraintSystem<F>,
        src_col: Column<Advice>,
        dst_col: Column<Advice>,
    ) -> AggConfig<F> {
        // input val columns, one per channel
        let val_in = meta.advice_column();
        let val_in_cln = meta.advice_column();
        meta.enable_equality(val_in);
        meta.enable_equality(val_in_cln);

        // sorted triple + the clean channel's value on the same sorted row
        let sorted = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let sorted_cln = meta.advice_column();
        for c in sorted {
            meta.enable_equality(c);
        }
        meta.enable_equality(sorted_cln);

        // permutation: (src, dst, val_in, val_in_cln) <-> (sorted, sorted_cln).
        // Both channels ride ONE shuffle, so the clean channel pays a column
        // and nothing else here.
        let q_perm_in = meta.complex_selector();
        let q_perm_out = meta.complex_selector();
        let perm_sort = PermAnyChip::configure(
            meta,
            q_perm_in,
            q_perm_out,
            vec![src_col, dst_col, val_in, val_in_cln],
            vec![sorted[0], sorted[1], sorted[2], sorted_cln],
        );

        // sortedness on sorted[0] <= sorted[0]_next (uses sentinel row at n)
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

        // The comparison out of the LAST real row reads the sentinel row at n,
        // and nothing above pins that cell: nondecreasing is satisfied by
        // sorted[0][n] == sorted[0][n-1] just as well as by the honest PAD_KEY.
        // With the sentinel equal to the last real key, iz_same_next below sees
        // "same group" on row n-1, the emit gate writes (PAD_KEY, PAD_VAL)
        // instead of the group pair, and the whole max-key group silently
        // disappears from the table the next join level reads. Consumers of that
        // key then prove NON-membership against a gap that really is in the
        // forged table, so the aggregate comes out short while every other
        // constraint stays satisfied. Requiring the last comparison to be STRICT
        // forces is_last = 1 on row n-1, so the final group is always emitted.
        // This reuses lt_src_cur_next, whose own gate is already gated on
        // q_sort, and costs one selector: no advice column, no new lookup.
        let q_sort_last = meta.selector();
        meta.create_gate("sorted src: last row precedes the sentinel", |m| {
            let q = m.query_selector(q_sort_last);
            let lt = lt_src_cur_next.is_lt(m, None);
            vec![q * (lt - Expression::Constant(F::ONE))]
        });

        // group-by on sorted src, sum sorted[2] and sorted_cln
        let q_first = meta.selector();
        let q_accu = meta.selector();
        let q_emit = meta.selector();

        let run_sum = meta.advice_column();
        let run_sum_cln = meta.advice_column();
        meta.enable_equality(run_sum);
        meta.enable_equality(run_sum_cln);

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
            let rs_cln = m.query_advice(run_sum_cln, Rotation::cur());
            let v_cln = m.query_advice(sorted_cln, Rotation::cur());
            vec![q.clone() * (rs - v), q * (rs_cln - v_cln)]
        });

        meta.create_gate("run_sum_accu", |m| {
            let q = m.query_selector(q_accu);
            let same = iz_same_prev.expr(); // 1 if same group
            let rs_cur = m.query_advice(run_sum, Rotation::cur());
            let rs_prev = m.query_advice(run_sum, Rotation::prev());
            let v = m.query_advice(sorted[2], Rotation::cur());
            let rs_cur_cln = m.query_advice(run_sum_cln, Rotation::cur());
            let rs_prev_cln = m.query_advice(run_sum_cln, Rotation::prev());
            let v_cln = m.query_advice(sorted_cln, Rotation::cur());
            vec![
                q.clone() * (rs_cur - (same.clone() * rs_prev + v)),
                q * (rs_cur_cln - (same * rs_prev_cln + v_cln)),
            ]
        });

        // emit padded (key, val, val_cln)
        let emit_pair = [meta.advice_column(), meta.advice_column()];
        let emit_cln = meta.advice_column();
        for c in emit_pair {
            meta.enable_equality(c);
        }
        meta.enable_equality(emit_cln);

        meta.create_gate("emit group pair at last row of group", |m| {
            let q = m.query_selector(q_emit);
            let one = Expression::Constant(F::ONE);
            let same_next = iz_same_next.expr();
            let is_last = one.clone() - same_next;
            let not_last = one - is_last.clone();

            let cur_key = m.query_advice(sorted[0], Rotation::cur());
            let cur_sum = m.query_advice(run_sum, Rotation::cur());
            let cur_sum_cln = m.query_advice(run_sum_cln, Rotation::cur());

            let out_k = m.query_advice(emit_pair[0], Rotation::cur());
            let out_v = m.query_advice(emit_pair[1], Rotation::cur());
            let out_v_cln = m.query_advice(emit_cln, Rotation::cur());

            let pad_k = Expression::Constant(F::from(PAD_KEY));
            let pad_v = Expression::Constant(F::from(PAD_VAL));

            vec![
                q.clone() * (out_k - (is_last.clone() * cur_key + not_last.clone() * pad_k)),
                q.clone()
                    * (out_v - (is_last.clone() * cur_sum + not_last.clone() * pad_v.clone())),
                q * (out_v_cln - (is_last * cur_sum_cln + not_last * pad_v)),
            ]
        });

        // Build a padded table (key, val, val_cln) that must be a permutation
        // of the emitted rows
        let tbl_pair = [meta.advice_column(), meta.advice_column()];
        let tbl_cln = meta.advice_column();
        let tbl_key_next = meta.advice_column();
        for c in tbl_pair {
            meta.enable_equality(c);
        }
        meta.enable_equality(tbl_cln);
        meta.enable_equality(tbl_key_next);

        let q_tbl_in = meta.complex_selector();
        let q_tbl_out = meta.complex_selector();
        let perm_tbl = PermAnyChip::configure(
            meta,
            q_tbl_in,
            q_tbl_out,
            vec![emit_pair[0], emit_pair[1], emit_cln],
            vec![tbl_pair[0], tbl_pair[1], tbl_cln],
        );

        // prove tbl_pair[0] sorted (only 0..n-2 are enabled in assignment)
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

        // ---- map table (dummy row 0) ----
        let map_pair = [meta.advice_column(), meta.advice_column()];
        let map_cln = meta.advice_column();
        let map_key_next = meta.advice_column();
        for c in map_pair {
            meta.enable_equality(c);
        }
        meta.enable_equality(map_cln);
        meta.enable_equality(map_key_next);

        let q_map_tbl = meta.complex_selector();
        let q_map_first = meta.selector();
        let q_map_link = meta.selector();
        let q_map_shift = meta.selector();
        let q_map_last = meta.selector();

        // the dummy row is (0, 0, 0): it is what a parent whose key occurs in
        // no child tuple looks up, and it makes both channels default to 0
        meta.create_gate("map_first_is_zero", |m| {
            let q = m.query_selector(q_map_first);
            let k0 = m.query_advice(map_pair[0], Rotation::cur());
            let v0 = m.query_advice(map_pair[1], Rotation::cur());
            let v0_cln = m.query_advice(map_cln, Rotation::cur());
            vec![q.clone() * k0, q.clone() * v0, q * v0_cln]
        });

        // link: map[i+1] == tbl[i]
        meta.create_gate("map_link_tbl_shift", |m| {
            let q = m.query_selector(q_map_link);
            let mk_next = m.query_advice(map_pair[0], Rotation::next());
            let mv_next = m.query_advice(map_pair[1], Rotation::next());
            let mv_next_cln = m.query_advice(map_cln, Rotation::next());
            let tk_cur = m.query_advice(tbl_pair[0], Rotation::cur());
            let tv_cur = m.query_advice(tbl_pair[1], Rotation::cur());
            let tv_cur_cln = m.query_advice(tbl_cln, Rotation::cur());
            vec![
                q.clone() * (mk_next - tk_cur),
                q.clone() * (mv_next - tv_cur),
                q * (mv_next_cln - tv_cur_cln),
            ]
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
            val_in_cln,
            sorted,
            sorted_cln,
            perm_sort,
            q_sort,
            q_sort_last,
            lt_src_cur_next,
            iz_src_eq,

            q_first,
            q_accu,
            q_emit,
            run_sum,
            run_sum_cln,
            iz_same_prev,
            iz_same_next,

            emit_pair,
            emit_cln,
            tbl_pair,
            tbl_cln,
            tbl_key_next,
            perm_tbl,

            q_tbl_sort,
            lt_tbl_key_cur_next,
            iz_tbl_key_eq,
            q_tbl_shift,
            q_tbl_last,

            map_pair,
            map_cln,
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
        let val_cln = meta.advice_column();
        for c in [in_next, low, high, val, val_cln] {
            meta.enable_equality(c);
        }

        let q_lookup = meta.selector();
        let q_lookup_complex = meta.complex_selector();

        // lt_low: low < dst, lt_high: dst < high (enabled only when not in_next)
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

        meta.create_gate("join membership/gap basic logic", |m| {
            let q = m.query_selector(q_lookup);
            let inx = m.query_advice(in_next, Rotation::cur());
            let one = Expression::Constant(F::ONE);

            let not_in = one.clone() - inx.clone();
            let v = m.query_advice(val, Rotation::cur());
            let v_cln = m.query_advice(val_cln, Rotation::cur());

            let low_ok = lt_low.is_lt(m, None);
            let high_ok = lt_high.is_lt(m, None);

            vec![
                // in_next boolean
                q.clone() * inx.clone() * (one.clone() - inx.clone()),
                // if not in -> must satisfy low<dst<high
                q.clone() * not_in.clone() * (one.clone() - low_ok),
                q.clone() * not_in.clone() * (one.clone() - high_ok),
                // if not in -> both channels default to 0 (this is sigma_j(v) = 0)
                q.clone() * not_in.clone() * v,
                q * not_in * v_cln,
            ]
        });

        JoinConfig {
            in_next,
            low,
            high,
            val,
            val_cln,
            q_lookup,
            q_lookup_complex,
            lt_low,
            lt_high,
        }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> Path3OrdConfig<F> {
        let instance = meta.instance_column();
        meta.enable_equality(instance);

        let r: [[Column<Advice>; 2]; 3] =
            std::array::from_fn(|_| [meta.advice_column(), meta.advice_column()]);
        for i in 0..3 {
            meta.enable_equality(r[i][0]);
            meta.enable_equality(r[i][1]);
        }

        // join[0]: r1.dst joins to T2; join[1]: r2.dst joins to T3
        let join = [
            Self::configure_join(meta, r[0][1]),
            Self::configure_join(meta, r[1][1]),
        ];

        // agg[0]=r3->T3 ; agg[1]=r2->T2
        let agg = [
            Self::configure_agg(meta, r[2][0], r[2][1]),
            Self::configure_agg(meta, r[1][0], r[1][1]),
        ];

        // ---------------- Conservation Check, one per tree node ----------------
        // The clean indicator has to be pinned to the partition, or a prover
        // could mark a residual row clean, inflate the clean channel and make
        // condition (4) vacuous. So it is bound by condition (1) itself: the
        // permutation carries the indicator as an extra column, the partition
        // side is laid out as [R^c rows | R^r rows] and holds the constant 1 on
        // the clean rows against 0 on the residual rows. The multiset equality
        // then forces the base-row indicator to mark exactly the occurrences
        // that went to R^c, and makes it boolean.
        //
        // r1, r2, r3 are three occurrences of the same Edge table, so each gets
        // its own indicator and its own partition.
        let cflag: [Column<Advice>; 3] = std::array::from_fn(|_| meta.advice_column());
        let part: [[Column<Advice>; 3]; 3] = std::array::from_fn(|_| {
            [
                meta.advice_column(),
                meta.advice_column(),
                meta.advice_column(),
            ]
        });
        let perm_cons: [PermAnyConfig; 3] = std::array::from_fn(|i| {
            let q_in = meta.complex_selector();
            let q_out = meta.complex_selector();
            PermAnyChip::configure(
                meta,
                q_in,
                q_out,
                vec![r[i][0], r[i][1], cflag[i]],
                part[i].to_vec(),
            )
        });
        let q_cln_flag: [Selector; 3] = std::array::from_fn(|_| meta.selector());
        let q_res_flag: [Selector; 3] = std::array::from_fn(|_| meta.selector());
        for i in 0..3 {
            let q_c = q_cln_flag[i];
            let q_r = q_res_flag[i];
            let flag_col = part[i][2];
            meta.create_gate("clean indicator on the partition side", move |m| {
                let qc = m.query_selector(q_c);
                let qr = m.query_selector(q_r);
                let f = m.query_advice(flag_col, Rotation::cur());
                vec![qc * (f.clone() - Expression::Constant(F::ONE)), qr * f]
            });
        }

        // ---------------- Pairwise Consistency, condition (3) ----------------
        // pi_K_ij(R_i^c) == pi_K_ij(R_j^c) on every join-tree edge, as two
        // mutual Membership Checks per edge over the CLEAN sections of the two
        // partition groups. Conservation already ties those tuples to the base
        // relation, so part[i][key] restricted to rows [0, |R_i^c|) is exactly
        // pi_K(R_i^c); reading the base column r[i][key] instead would only
        // prove membership in R_i, which is not condition (3).
        //
        // The input gates are fresh COMPLEX selectors: a simple selector may not
        // appear in a lookup expression, so q_cln_flag cannot be reused here.
        // Each one is also the table gate of the opposite direction on its edge.
        //
        // Earlier versions of this file routed every direction through an
        // intermediate advice column holding the deduplicated key set of the
        // relation on the other end. Nothing in the circuit bound such a column
        // to the relation it claimed to enumerate, so a prover could set the
        // table r1 looks into to pi_dst(R1^c) and the table r2 looks into to
        // pi_src(R2^c) and satisfy both directions for an ARBITRARY partition:
        // condition (3) was vacuous, and the all-clean escape it exists to close
        // was still open. Looking the two clean key columns up in each other
        // leaves no free advice, so there is nothing left to forge, and the two
        // containments together are the set equality condition (3) asks for.
        //
        // A lookup input is 0 on every row where its selector is off, and the
        // table side is 0 on those rows too, so a bare one-column containment
        // has 0 in the table unconditionally. The comment this replaces argued
        // that SHIFT_ID makes every real key nonzero, but that shift is host
        // side: no gate constrains part[i][*] or r[i][*] to be nonzero, so a
        // prover could smuggle ONE 0-keyed tuple into R_i^c and have both
        // directions of condition (9) accept it vacuously, without re-running
        // any fixpoint and without changing the declared block sizes.
        //
        // The fix is to carry the gate itself as the first component of the
        // looked-up tuple. A gated-off input row is then (0, 0), which the
        // gated-off table rows still supply for free, while an ENABLED input row
        // is (1, key) and can only be matched by an ENABLED table row (1, key).
        // Containment is now over exactly the real clean keys, whatever their
        // value, and it costs no advice column: the extra component is the
        // selector expression that was already in the argument.
        let q_pw_cln: [Selector; 3] = std::array::from_fn(|_| meta.complex_selector());

        let mut pw_edge = |name: &'static str,
                           q_in: Selector,
                           in_col: Column<Advice>,
                           q_t: Selector,
                           tbl_col: Column<Advice>| {
            meta.lookup_any(name, move |m| {
                let qi = m.query_selector(q_in);
                let qt = m.query_selector(q_t);
                let lhs = qi.clone() * m.query_advice(in_col, Rotation::cur());
                let rhs = qt.clone() * m.query_advice(tbl_col, Rotation::cur());
                vec![(qi, qt), (lhs, rhs)]
            });
        };

        // edge r1.dst = r2.src
        pw_edge(
            "pw: r1^c dst in r2^c src",
            q_pw_cln[0],
            part[0][1],
            q_pw_cln[1],
            part[1][0],
        );
        pw_edge(
            "pw: r2^c src in r1^c dst",
            q_pw_cln[1],
            part[1][0],
            q_pw_cln[0],
            part[0][1],
        );

        // edge r2.dst = r3.src
        pw_edge(
            "pw: r2^c dst in r3^c src",
            q_pw_cln[1],
            part[1][1],
            q_pw_cln[2],
            part[2][0],
        );
        pw_edge(
            "pw: r3^c src in r2^c dst",
            q_pw_cln[2],
            part[2][0],
            q_pw_cln[1],
            part[1][1],
        );

        // Ordering checks
        let q_ord1 = meta.selector();
        let q_ord2 = meta.selector();

        // The three copies of Edge are ONE relation.
        //
        // r1, r2 and r3 are the same table in the query, but they are three
        // independent pairs of free advice columns. Before this gate nothing in
        // the circuit related them: the only cross-copy tie was the Pairwise
        // Consistency containment on the clean blocks, which is set equality of
        // KEYS, not tuple equality of relations. A prover could therefore leave
        // r1 and r2 honest and fill r3 with n copies of one (src, dst) pair. The
        // leaf gate q_r3_one forces val_in == 1 on every r3 row, so T3[src]
        // becomes n, the sortedness gates are satisfied because all keys are
        // equal, and the whole OBJ argument then certifies the honest reduction
        // of THAT forged three-relation instance while the public COUNT comes
        // out arbitrary.
        //
        // Because this circuit is a self-join of a single Edge table, and the
        // assignment writes the same (src, dst) into all three copies on every
        // row, the tie is a row-wise equality, not a shuffle: four degree-2
        // constraints under one selector, no advice column and no lookup. It
        // does NOT bind the relation to a commitment; that needs an instance /
        // wrapper change outside this file.
        let q_self_join = meta.selector();
        meta.create_gate("self-join: r1, r2, r3 are the same Edge row", |m| {
            let q = m.query_selector(q_self_join);
            let s0 = m.query_advice(r[0][0], Rotation::cur());
            let d0 = m.query_advice(r[0][1], Rotation::cur());
            let s1 = m.query_advice(r[1][0], Rotation::cur());
            let d1 = m.query_advice(r[1][1], Rotation::cur());
            let s2 = m.query_advice(r[2][0], Rotation::cur());
            let d2 = m.query_advice(r[2][1], Rotation::cur());
            vec![
                q.clone() * (s0 - s1.clone()),
                q.clone() * (d0 - d1.clone()),
                q.clone() * (s1 - s2),
                q * (d1 - d2),
            ]
        });

        let lt_ab = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_ord1),
            |m| m.query_advice(r[0][0], Rotation::cur()),
            |m| m.query_advice(r[0][1], Rotation::cur()),
        );
        let lt_bc = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_ord2),
            |m| m.query_advice(r[1][0], Rotation::cur()),
            |m| m.query_advice(r[1][1], Rotation::cur()),
        );

        // ---- the clean part must satisfy the query's local predicates ----
        //
        // The WHERE clause a<b<c reaches the circuit only as the two Lt flags
        // lt_ab and lt_bc, and those were read in exactly three places, all of
        // them inside a multiplicity product (q_r3_one, q_r2_filter,
        // q_r1_filter). Nothing constrained cflag against them. That made
        // conditions (9) and (10) pin R^c to the semijoin-reduced instance of
        // the UNFILTERED edge relation instead of the query's: a prover could
        // add to R_1^c every tuple with a >= b whose dst still has a clean
        // partner in R_2^c, and symmetrically for R_2^c. Every gate, shuffle and
        // lookup stayed satisfied and the public COUNT was unchanged, because
        // (10) is one-sided. Concretely sum_cln = sum holds for ANY pointwise
        // superset of the honest clean flags: sum_cln is a sum of nonnegative
        // terms that is monotone in the flags and bounded above by sum, so
        // inflation cannot move it. Condition (10) can only catch DEFLATION of
        // R^c, so the predicate has to be bound to the indicator directly, and
        // condition (9) alone bounds inflation only up to the filter-blind
        // semijoin fixpoint.
        //
        // Two degree-3 constraints under one selector, enabled on the same rows
        // as q_ord1/q_ord2: a clean r1 occurrence must have a<b, a clean r2
        // occurrence must have b<c. r3 is a leaf with no predicate of its own.
        // The honest witness already satisfies this, since cln1[i] and cln2[i]
        // are computed from mu1[i] > 0 and mu2[i] > 0 and both multiplicities
        // are zero unless the row passes its own predicate.
        let q_cln_pred = meta.selector();
        meta.create_gate("cp: clean rows satisfy the local predicate", |m| {
            let q = m.query_selector(q_cln_pred);
            let one = Expression::Constant(F::ONE);
            let c1 = m.query_advice(cflag[0], Rotation::cur());
            let c2 = m.query_advice(cflag[1], Rotation::cur());
            let ab = lt_ab.is_lt(m, None);
            let bc = lt_bc.is_lt(m, None);
            vec![q.clone() * c1 * (one.clone() - ab), q * c2 * (one - bc)]
        });

        // filtered contribution per r1 row, one per channel
        let fval_r1 = meta.advice_column();
        let fval_r1_cln = meta.advice_column();
        meta.enable_equality(fval_r1);
        meta.enable_equality(fval_r1_cln);

        // soundness glue
        let q_r3_one = meta.selector();
        let q_r2_filter = meta.selector();
        let q_r1_filter = meta.selector();

        // r3 is a leaf with no predicate: val_in == 1 on the input channel and
        // the bound clean indicator on the clean channel
        meta.create_gate("r3 val_in is 1", |m| {
            let q = m.query_selector(q_r3_one);
            let v = m.query_advice(agg[0].val_in, Rotation::cur());
            let v_cln = m.query_advice(agg[0].val_in_cln, Rotation::cur());
            let c = m.query_advice(cflag[2], Rotation::cur());
            vec![
                q.clone() * (v - Expression::Constant(F::ONE)),
                q * (v_cln - c),
            ]
        });

        // r2: agg[1].val_in == join[1].val * lt_bc, and on the clean channel
        // the same product with the clean indicator of r2 multiplied in
        meta.create_gate("r2 val_in filtered by b<c", |m| {
            let q = m.query_selector(q_r2_filter);
            let raw = m.query_advice(join[1].val, Rotation::cur());
            let raw_cln = m.query_advice(join[1].val_cln, Rotation::cur());
            let bc = lt_bc.is_lt(m, None);
            let vin = m.query_advice(agg[1].val_in, Rotation::cur());
            let vin_cln = m.query_advice(agg[1].val_in_cln, Rotation::cur());
            let c = m.query_advice(cflag[1], Rotation::cur());
            vec![
                q.clone() * (vin - raw * bc.clone()),
                q * (vin_cln - c * raw_cln * bc),
            ]
        });

        // r1: fval_r1 == join[0].val * lt_ab, same shape on the clean channel
        meta.create_gate("r1 filtered contribution", |m| {
            let q = m.query_selector(q_r1_filter);
            let raw = m.query_advice(join[0].val, Rotation::cur());
            let raw_cln = m.query_advice(join[0].val_cln, Rotation::cur());
            let ab = lt_ab.is_lt(m, None);
            let fv = m.query_advice(fval_r1, Rotation::cur());
            let fv_cln = m.query_advice(fval_r1_cln, Rotation::cur());
            let c = m.query_advice(cflag[0], Rotation::cur());
            vec![
                q.clone() * (fv - raw * ab.clone()),
                q * (fv_cln - c * raw_cln * ab),
            ]
        });

        // sum both channels over the root relation
        let q_sum_first = meta.selector();
        let q_sum_accu = meta.selector();
        let sum = meta.advice_column();
        let sum_cln = meta.advice_column();
        meta.enable_equality(sum);
        meta.enable_equality(sum_cln);

        meta.create_gate("sum_first", |m| {
            let q = m.query_selector(q_sum_first);
            let s = m.query_advice(sum, Rotation::cur());
            let v = m.query_advice(fval_r1, Rotation::cur());
            let s_cln = m.query_advice(sum_cln, Rotation::cur());
            let v_cln = m.query_advice(fval_r1_cln, Rotation::cur());
            vec![q.clone() * (s - v), q * (s_cln - v_cln)]
        });

        meta.create_gate("sum_accu", |m| {
            let q = m.query_selector(q_sum_accu);
            let s_cur = m.query_advice(sum, Rotation::cur());
            let s_prev = m.query_advice(sum, Rotation::prev());
            let v = m.query_advice(fval_r1, Rotation::cur());
            let s_cur_cln = m.query_advice(sum_cln, Rotation::cur());
            let s_prev_cln = m.query_advice(sum_cln, Rotation::prev());
            let v_cln = m.query_advice(fval_r1_cln, Rotation::cur());
            vec![
                q.clone() * (s_cur - (s_prev + v)),
                q * (s_cur_cln - (s_prev_cln + v_cln)),
            ]
        });

        // condition (4): |R_1^c |X| R_2^c |X| R_3^c| == |R_1 |X| R_2 |X| R_3|
        let q_card_eq = meta.selector();
        meta.create_gate("cp: cardinality preservation", |m| {
            let q = m.query_selector(q_card_eq);
            let s = m.query_advice(sum, Rotation::cur());
            let s_cln = m.query_advice(sum_cln, Rotation::cur());
            vec![q * (s - s_cln)]
        });

        // output constrained equal to sum (enable q_out at the chosen row)
        let out = meta.advice_column();
        meta.enable_equality(out);
        let q_out = meta.selector();
        meta.create_gate("out equals sum", |m| {
            let q = m.query_selector(q_out);
            let o = m.query_advice(out, Rotation::cur());
            let s = m.query_advice(sum, Rotation::cur());
            vec![q * (o - s)]
        });

        Path3OrdConfig {
            instance,
            r,
            q_self_join,
            join,
            agg,
            cflag,
            part,
            perm_cons,
            q_cln_flag,
            q_res_flag,
            q_pw_cln,
            q_cln_pred,
            q_ord1,
            q_ord2,
            lt_ab,
            lt_bc,
            q_r3_one,
            q_r2_filter,
            q_r1_filter,
            fval_r1,
            fval_r1_cln,
            q_sum_first,
            q_sum_accu,
            sum,
            sum_cln,
            q_card_eq,
            out,
            q_out,
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

    // ---------------- assignment helpers ----------------

    fn assign_agg_stage(
        region: &mut Region<'_, F>,
        a: &AggConfig<F>,
        n: usize,
        sorted_rows: &Vec<[u64; 4]>,
        run: &Vec<u64>,
        run_cln: &Vec<u64>,
        emit: &Vec<[u64; 3]>,
        tbl: &Vec<[u64; 3]>,
        key_next: &Vec<u64>,
        val_in_vec: &Vec<u64>,
        val_in_cln_vec: &Vec<u64>,
    ) -> Result<(), Error> {
        // val_in, one column per channel
        for i in 0..n {
            region.assign_advice(
                || "val_in",
                a.val_in,
                i,
                || Value::known(F::from(val_in_vec[i])),
            )?;
            region.assign_advice(
                || "val_in_cln",
                a.val_in_cln,
                i,
                || Value::known(F::from(val_in_cln_vec[i])),
            )?;
        }

        // sorted rows + sentinel row at n
        for i in 0..n {
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
            region.assign_advice(
                || "sorted_val_cln",
                a.sorted_cln,
                i,
                || Value::known(F::from(sorted_rows[i][3])),
            )?;
        }
        region.assign_advice(
            || "sorted_src_s",
            a.sorted[0],
            n,
            || Value::known(F::from(PAD_KEY)),
        )?;
        region.assign_advice(
            || "sorted_dst_s",
            a.sorted[1],
            n,
            || Value::known(F::from(PAD_KEY)),
        )?;
        region.assign_advice(|| "sorted_val_s", a.sorted[2], n, || Value::known(F::ZERO))?;
        region.assign_advice(
            || "sorted_val_cln_s",
            a.sorted_cln,
            n,
            || Value::known(F::ZERO),
        )?;

        // run_sum, one per channel
        for i in 0..n {
            region.assign_advice(|| "run_sum", a.run_sum, i, || Value::known(F::from(run[i])))?;
            region.assign_advice(
                || "run_sum_cln",
                a.run_sum_cln,
                i,
                || Value::known(F::from(run_cln[i])),
            )?;
        }

        // emitted group rows
        for i in 0..n {
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
            region.assign_advice(
                || "emit_v_cln",
                a.emit_cln,
                i,
                || Value::known(F::from(emit[i][2])),
            )?;
        }

        // tbl rows + key_next
        for i in 0..n {
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
                || "tbl_v_cln",
                a.tbl_cln,
                i,
                || Value::known(F::from(tbl[i][2])),
            )?;
            region.assign_advice(
                || "tbl_kn",
                a.tbl_key_next,
                i,
                || Value::known(F::from(key_next[i])),
            )?;
        }

        // ---- map table: rows 0..=n ----
        // row0 = (0,0,0)
        region.assign_advice(|| "map_k0", a.map_pair[0], 0, || Value::known(F::ZERO))?;
        region.assign_advice(|| "map_v0", a.map_pair[1], 0, || Value::known(F::ZERO))?;
        region.assign_advice(|| "map_v0_cln", a.map_cln, 0, || Value::known(F::ZERO))?;

        // rows 1..=n copy tbl[0..n-1]
        for i in 0..n {
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
            region.assign_advice(
                || "map_v_cln",
                a.map_cln,
                i + 1,
                || Value::known(F::from(tbl[i][2])),
            )?;
        }

        // map_key_next[r] = map_key[r+1]
        // => for r=0..n-1: map_key_next[r] = tbl[r].key ; for r=n: PAD
        for r in 0..n {
            region.assign_advice(
                || "map_kn",
                a.map_key_next,
                r,
                || Value::known(F::from(tbl[r][0])),
            )?;
        }
        region.assign_advice(
            || "map_kn_last",
            a.map_key_next,
            n,
            || Value::known(F::from(PAD_KEY)),
        )?;

        // enable map selectors (table rows 0..=n)
        for i in 0..=n {
            a.q_map_tbl.enable(region, i)?;
        }
        a.q_map_first.enable(region, 0)?;
        for i in 0..n {
            a.q_map_link.enable(region, i)?;
            a.q_map_shift.enable(region, i)?;
        }
        a.q_map_last.enable(region, n)?;

        // enable perms
        for i in 0..n {
            a.perm_sort.q_perm1.enable(region, i)?;
            a.perm_sort.q_perm2.enable(region, i)?;
            a.perm_tbl.q_perm1.enable(region, i)?;
            a.perm_tbl.q_perm2.enable(region, i)?;
        }

        // sortedness on sorted src uses sentinel row -> enable 0..n-1
        for i in 0..n {
            a.q_sort.enable(region, i)?;
        }
        // and the comparison into the sentinel row must be strict, so that the
        // last group is always recognised as a group end and gets emitted
        if n > 0 {
            a.q_sort_last.enable(region, n - 1)?;
        }

        // tbl sortedness is only meaningful for 0..n-2
        for i in 0..n.saturating_sub(1) {
            if i + 1 < n {
                a.q_tbl_sort.enable(region, i)?;
                a.q_tbl_shift.enable(region, i)?;
            }
        }
        if n > 0 {
            a.q_tbl_last.enable(region, n - 1)?;
        }

        // group gates
        if n > 0 {
            a.q_first.enable(region, 0)?;
        }
        for i in 1..n {
            a.q_accu.enable(region, i)?;
        }
        for i in 0..n {
            a.q_emit.enable(region, i)?;
        }

        // assign helper chips
        let iz_same_prev_chip = IsZeroChip::construct(a.iz_same_prev.clone());
        let iz_same_next_chip = IsZeroChip::construct(a.iz_same_next.clone());
        let iz_src_eq_chip = IsZeroChip::construct(a.iz_src_eq.clone());
        let iz_tbl_eq_chip = IsZeroChip::construct(a.iz_tbl_key_eq.clone());

        let lt_src_chip = LtChip::<F, NUM_BYTES>::construct(a.lt_src_cur_next.clone());
        let lt_tbl_chip = LtChip::<F, NUM_BYTES>::construct(a.lt_tbl_key_cur_next.clone());

        // For sorted src: compare row i with row i+1; last compares to PAD sentinel
        for i in 0..n {
            let cur = sorted_rows[i][0];
            let next = if i + 1 < n {
                sorted_rows[i + 1][0]
            } else {
                PAD_KEY
            };

            lt_src_chip.assign(
                region,
                i,
                Value::known(F::from(cur)),
                Value::known(F::from(next)),
            )?;
            let diff_src = F::from(next) - F::from(cur);
            iz_src_eq_chip.assign(region, i, Value::known(diff_src))?;
        }

        // For tbl sortedness: only 0..n-2
        for i in 0..n.saturating_sub(1) {
            if i + 1 < n {
                lt_tbl_chip.assign(
                    region,
                    i,
                    Value::known(F::from(tbl[i][0])),
                    Value::known(F::from(tbl[i + 1][0])),
                )?;
                let diff_tbl = F::from(tbl[i + 1][0]) - F::from(tbl[i][0]);
                iz_tbl_eq_chip.assign(region, i, Value::known(diff_tbl))?;
            }
        }

        // same_prev (1..n-1)
        for i in 1..n {
            let diff = F::from(sorted_rows[i][0]) - F::from(sorted_rows[i - 1][0]);
            iz_same_prev_chip.assign(region, i, Value::known(diff))?;
        }
        // same_next (0..n-1), uses sentinel row at n
        for i in 0..n {
            let next_src = if i + 1 < n {
                sorted_rows[i + 1][0]
            } else {
                PAD_KEY
            };
            let diff = F::from(next_src) - F::from(sorted_rows[i][0]);
            iz_same_next_chip.assign(region, i, Value::known(diff))?;
        }

        Ok(())
    }

    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        edges_in: &[Edge],
    ) -> Result<AssignedCell<F, F>, Error> {
        let cfg = &self.cfg;
        let n = edges_in.len();

        // load LT tables
        // agg LTs
        for a in cfg.agg.iter() {
            LtChip::<F, NUM_BYTES>::construct(a.lt_src_cur_next.clone()).load(layouter)?;
            LtChip::<F, NUM_BYTES>::construct(a.lt_tbl_key_cur_next.clone()).load(layouter)?;
        }
        // join LTs
        for j in cfg.join.iter() {
            LtChip::<F, NUM_BYTES>::construct(j.lt_low.clone()).load(layouter)?;
            LtChip::<F, NUM_BYTES>::construct(j.lt_high.clone()).load(layouter)?;
        }
        // ordering LTs
        LtChip::<F, NUM_BYTES>::construct(cfg.lt_ab.clone()).load(layouter)?;
        LtChip::<F, NUM_BYTES>::construct(cfg.lt_bc.clone()).load(layouter)?;

        if n == 0 {
            // trivial output 0
            let cell = layouter.assign_region(
                || "out0",
                |mut region| {
                    let c = region.assign_advice(|| "out", cfg.out, 0, || Value::known(F::ZERO))?;
                    Ok(c)
                },
            )?;
            return Ok(cell);
        }

        // Shift IDs inside the circuit to reserve key=0 for dummy map row.
        // (Works whether Edge fields are u32/u64/etc.)
        let edges: Vec<(u64, u64)> = edges_in
            .iter()
            .map(|e| {
                let s = e.src as u64;
                let d = e.dst as u64;
                (s + SHIFT_ID, d + SHIFT_ID)
            })
            .collect();

        // ---------------- host DP witnesses ----------------

        // The prover's partition of each tree node into R^c and R^r. Read the
        // test hook once, here, so the whole witness below is consistent with
        // whichever partition is used.
        let tamper = HIDE_ONE_CLEAN_TUPLE.load(Ordering::Relaxed);
        let all_clean = MARK_ALL_CLEAN.load(Ordering::Relaxed);
        let (cln1, cln2, cln3) = if all_clean {
            // no reduction at all: every real tuple is declared clean and the
            // residual section of every partition stays empty
            (vec![1u64; n], vec![1u64; n], vec![1u64; n])
        } else {
            Self::clean_indicators(&edges, tamper)
        };
        let (part_r1, n_cln1) = Self::partition_rows(&edges, &cln1);
        let (part_r2, n_cln2) = Self::partition_rows(&edges, &cln2);
        let (part_r3, n_cln3) = Self::partition_rows(&edges, &cln3);

        // The Pairwise Consistency lookups need no witness of their own: both
        // sides of every direction are partition key columns that the
        // Conservation Check already assigns below.

        // Stage T3 from r3 (same edges), grouped by src: the input channel is
        // 1 per edge (no predicate on the leaf) and the clean channel is c_3.
        let r3_rows: Vec<[u64; 4]> = edges
            .iter()
            .enumerate()
            .map(|(i, (s, d))| [*s, *d, 1u64, cln3[i]])
            .collect();
        let r3_sorted = Self::sort_by_src(r3_rows.clone());
        let (r3_run, r3_run_cln) = Self::run_sum_by_src(&r3_sorted);
        let r3_emit = Self::emit_pairs(&r3_sorted, &r3_run, &r3_run_cln);
        let t3_tbl = Self::build_tbl_from_emit(&r3_emit, n);
        let t3_next = Self::key_next_from_tbl(&t3_tbl);
        let t3_map = Self::map_from_tbl(&t3_tbl);

        let mut t3_keys: Vec<u64> = t3_tbl
            .iter()
            .map(|p| p[0])
            .filter(|&k| k != PAD_KEY)
            .collect();
        t3_keys.push(0);
        t3_keys.push(PAD_KEY);
        t3_keys.sort();
        t3_keys.dedup();

        // Join r2.dst into T3 (both channels come out of the same lookup), then
        // filter by (r2.src<r2.dst) to form the two val_in of T2
        let mut r2_in = vec![0u64; n];
        let mut r2_low = vec![0u64; n];
        let mut r2_high = vec![PAD_KEY; n];
        let mut r2_join_val = vec![0u64; n];
        let mut r2_join_val_cln = vec![0u64; n];
        let mut r2_val_in = vec![0u64; n];
        let mut r2_val_in_cln = vec![0u64; n];

        for (i, (s, d)) in edges.iter().enumerate() {
            let dst_c = *d;
            let (b, lo, hi) = Self::gap_witness(&t3_keys, dst_c);
            r2_in[i] = b;
            r2_low[i] = lo;
            r2_high[i] = hi;

            let (raw, raw_cln) = if b == 1 {
                *t3_map.get(&dst_c).unwrap_or(&(0, 0))
            } else {
                (0, 0)
            };
            r2_join_val[i] = raw;
            r2_join_val_cln[i] = raw_cln;

            let bc = if s < d { 1u64 } else { 0u64 };
            r2_val_in[i] = raw.saturating_mul(bc);
            r2_val_in_cln[i] = cln2[i].saturating_mul(raw_cln).saturating_mul(bc);
        }

        // Build T2 by aggregating rows (src, dst, val_in, val_in_cln) by src
        let r2_rows: Vec<[u64; 4]> = edges
            .iter()
            .enumerate()
            .map(|(i, (s, d))| [*s, *d, r2_val_in[i], r2_val_in_cln[i]])
            .collect();
        let r2_sorted = Self::sort_by_src(r2_rows.clone());
        let (r2_run, r2_run_cln) = Self::run_sum_by_src(&r2_sorted);
        let r2_emit = Self::emit_pairs(&r2_sorted, &r2_run, &r2_run_cln);
        let t2_tbl = Self::build_tbl_from_emit(&r2_emit, n);
        let t2_next = Self::key_next_from_tbl(&t2_tbl);
        let t2_map = Self::map_from_tbl(&t2_tbl);

        let mut t2_keys: Vec<u64> = t2_tbl
            .iter()
            .map(|p| p[0])
            .filter(|&k| k != PAD_KEY)
            .collect();
        t2_keys.push(0);
        t2_keys.push(PAD_KEY);
        t2_keys.sort();
        t2_keys.dedup();

        // Join r1.dst into T2
        let mut r1_in = vec![0u64; n];
        let mut r1_low = vec![0u64; n];
        let mut r1_high = vec![PAD_KEY; n];
        let mut r1_join_val = vec![0u64; n];
        let mut r1_join_val_cln = vec![0u64; n];
        let mut r1_fval = vec![0u64; n];
        let mut r1_fval_cln = vec![0u64; n];

        for (i, (s, d)) in edges.iter().enumerate() {
            let b = *d;
            let (inn, lo, hi) = Self::gap_witness(&t2_keys, b);
            r1_in[i] = inn;
            r1_low[i] = lo;
            r1_high[i] = hi;

            let (raw, raw_cln) = if inn == 1 {
                *t2_map.get(&b).unwrap_or(&(0, 0))
            } else {
                (0, 0)
            };
            r1_join_val[i] = raw;
            r1_join_val_cln[i] = raw_cln;

            let ab = if s < d { 1u64 } else { 0u64 };
            r1_fval[i] = raw.saturating_mul(ab);
            r1_fval_cln[i] = cln1[i].saturating_mul(raw_cln).saturating_mul(ab);
        }

        // ---------------- circuit assignment ----------------

        let out_cell = layouter.assign_region(
            || "path3_ordered",
            |mut region| {
                // base tables r1,r2,r3 all equal to edges (self-join)
                for (i, (s, d)) in edges.iter().enumerate() {
                    // the three copies are one relation, row by row
                    cfg.q_self_join.enable(&mut region, i)?;
                    // r1
                    region.assign_advice(
                        || "r1_src",
                        cfg.r[0][0],
                        i,
                        || Value::known(F::from(*s)),
                    )?;
                    region.assign_advice(
                        || "r1_dst",
                        cfg.r[0][1],
                        i,
                        || Value::known(F::from(*d)),
                    )?;
                    // r2
                    region.assign_advice(
                        || "r2_src",
                        cfg.r[1][0],
                        i,
                        || Value::known(F::from(*s)),
                    )?;
                    region.assign_advice(
                        || "r2_dst",
                        cfg.r[1][1],
                        i,
                        || Value::known(F::from(*d)),
                    )?;
                    // r3
                    region.assign_advice(
                        || "r3_src",
                        cfg.r[2][0],
                        i,
                        || Value::known(F::from(*s)),
                    )?;
                    region.assign_advice(
                        || "r3_dst",
                        cfg.r[2][1],
                        i,
                        || Value::known(F::from(*d)),
                    )?;
                }

                // ---- Conservation Check per tree node: R == R^c U R^r ----
                // The base row carries the clean indicator, the partition side
                // carries the same rows as [R^c | R^r] with the flag pinned to
                // 1 then 0, and the shuffle ties the two together. This is what
                // binds the indicator used by the clean channel below.
                let cons: [(&Vec<u64>, &Vec<[u64; 3]>, usize); 3] = [
                    (&cln1, &part_r1, n_cln1),
                    (&cln2, &part_r2, n_cln2),
                    (&cln3, &part_r3, n_cln3),
                ];
                for idx in 0..3 {
                    let (flags, part_rows, n_cln) = cons[idx];
                    for i in 0..n {
                        region.assign_advice(
                            || "cflag",
                            cfg.cflag[idx],
                            i,
                            || Value::known(F::from(flags[i])),
                        )?;
                        for j in 0..3 {
                            region.assign_advice(
                                || "part",
                                cfg.part[idx][j],
                                i,
                                || Value::known(F::from(part_rows[i][j])),
                            )?;
                        }
                        cfg.perm_cons[idx].q_perm1.enable(&mut region, i)?;
                        cfg.perm_cons[idx].q_perm2.enable(&mut region, i)?;
                    }
                    for i in 0..n_cln {
                        cfg.q_cln_flag[idx].enable(&mut region, i)?;
                        // the gate of the Pairwise Consistency lookups of this
                        // relation, over exactly the same clean row range: input
                        // side of its own directions, table side of the opposite
                        // ones
                        cfg.q_pw_cln[idx].enable(&mut region, i)?;
                    }
                    for i in n_cln..n {
                        cfg.q_res_flag[idx].enable(&mut region, i)?;
                    }
                }

                // join[1] for r2 -> T3
                {
                    let jc = &cfg.join[1];
                    let lt_low_chip = LtChip::<F, NUM_BYTES>::construct(jc.lt_low.clone());
                    let lt_high_chip = LtChip::<F, NUM_BYTES>::construct(jc.lt_high.clone());

                    for i in 0..n {
                        jc.q_lookup.enable(&mut region, i)?;
                        jc.q_lookup_complex.enable(&mut region, i)?;

                        region.assign_advice(
                            || "r2_in",
                            jc.in_next,
                            i,
                            || Value::known(F::from(r2_in[i])),
                        )?;
                        region.assign_advice(
                            || "r2_low",
                            jc.low,
                            i,
                            || Value::known(F::from(r2_low[i])),
                        )?;
                        region.assign_advice(
                            || "r2_high",
                            jc.high,
                            i,
                            || Value::known(F::from(r2_high[i])),
                        )?;
                        region.assign_advice(
                            || "r2_val",
                            jc.val,
                            i,
                            || Value::known(F::from(r2_join_val[i])),
                        )?;
                        region.assign_advice(
                            || "r2_val_cln",
                            jc.val_cln,
                            i,
                            || Value::known(F::from(r2_join_val_cln[i])),
                        )?;

                        let dst = edges[i].1;
                        lt_low_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(r2_low[i])),
                            Value::known(F::from(dst)),
                        )?;
                        lt_high_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(dst)),
                            Value::known(F::from(r2_high[i])),
                        )?;
                    }
                }

                // join[0] for r1 -> T2
                {
                    let jc = &cfg.join[0];
                    let lt_low_chip = LtChip::<F, NUM_BYTES>::construct(jc.lt_low.clone());
                    let lt_high_chip = LtChip::<F, NUM_BYTES>::construct(jc.lt_high.clone());

                    for i in 0..n {
                        jc.q_lookup.enable(&mut region, i)?;
                        jc.q_lookup_complex.enable(&mut region, i)?;

                        region.assign_advice(
                            || "r1_in",
                            jc.in_next,
                            i,
                            || Value::known(F::from(r1_in[i])),
                        )?;
                        region.assign_advice(
                            || "r1_low",
                            jc.low,
                            i,
                            || Value::known(F::from(r1_low[i])),
                        )?;
                        region.assign_advice(
                            || "r1_high",
                            jc.high,
                            i,
                            || Value::known(F::from(r1_high[i])),
                        )?;
                        region.assign_advice(
                            || "r1_val",
                            jc.val,
                            i,
                            || Value::known(F::from(r1_join_val[i])),
                        )?;
                        region.assign_advice(
                            || "r1_val_cln",
                            jc.val_cln,
                            i,
                            || Value::known(F::from(r1_join_val_cln[i])),
                        )?;

                        let dst = edges[i].1;
                        lt_low_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(r1_low[i])),
                            Value::known(F::from(dst)),
                        )?;
                        lt_high_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(dst)),
                            Value::known(F::from(r1_high[i])),
                        )?;
                    }
                }

                // ordering selectors + LT witnesses
                {
                    let lt_ab_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_ab.clone());
                    let lt_bc_chip = LtChip::<F, NUM_BYTES>::construct(cfg.lt_bc.clone());

                    for i in 0..n {
                        cfg.q_ord1.enable(&mut region, i)?;
                        cfg.q_ord2.enable(&mut region, i)?;
                        // and, on the same rows, the clean indicator of r1 and
                        // r2 must respect the predicate those Lt flags carry
                        cfg.q_cln_pred.enable(&mut region, i)?;

                        let (s, d) = edges[i];

                        // r1.src < r1.dst
                        lt_ab_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(s)),
                            Value::known(F::from(d)),
                        )?;
                        // r2.src < r2.dst
                        lt_bc_chip.assign(
                            &mut region,
                            i,
                            Value::known(F::from(s)),
                            Value::known(F::from(d)),
                        )?;
                    }
                }

                // glue selectors + filtered values, one per channel
                for i in 0..n {
                    cfg.q_r3_one.enable(&mut region, i)?;
                    cfg.q_r2_filter.enable(&mut region, i)?;
                    cfg.q_r1_filter.enable(&mut region, i)?;

                    region.assign_advice(
                        || "fval_r1",
                        cfg.fval_r1,
                        i,
                        || Value::known(F::from(r1_fval[i])),
                    )?;
                    region.assign_advice(
                        || "fval_r1_cln",
                        cfg.fval_r1_cln,
                        i,
                        || Value::known(F::from(r1_fval_cln[i])),
                    )?;
                }

                // agg stage 0: r3 -> T3 (val_in all ones, clean channel = c_3)
                let r3_val_in = vec![1u64; n];
                Self::assign_agg_stage(
                    &mut region,
                    &cfg.agg[0],
                    n,
                    &r3_sorted,
                    &r3_run,
                    &r3_run_cln,
                    &r3_emit,
                    &t3_tbl,
                    &t3_next,
                    &r3_val_in,
                    &cln3,
                )?;

                // agg stage 1: r2 -> T2 (val_in = join_val * (b<c), both channels)
                Self::assign_agg_stage(
                    &mut region,
                    &cfg.agg[1],
                    n,
                    &r2_sorted,
                    &r2_run,
                    &r2_run_cln,
                    &r2_emit,
                    &t2_tbl,
                    &t2_next,
                    &r2_val_in,
                    &r2_val_in_cln,
                )?;

                // final sums over both channels
                let mut running: u128 = 0;
                let mut running_cln: u128 = 0;
                for i in 0..n {
                    running += r1_fval[i] as u128;
                    running_cln += r1_fval_cln[i] as u128;
                    region.assign_advice(
                        || "sum",
                        cfg.sum,
                        i,
                        || Value::known(F::from(running as u64)),
                    )?;
                    region.assign_advice(
                        || "sum_cln",
                        cfg.sum_cln,
                        i,
                        || Value::known(F::from(running_cln as u64)),
                    )?;
                }
                cfg.q_sum_first.enable(&mut region, 0)?;
                for i in 1..n {
                    cfg.q_sum_accu.enable(&mut region, i)?;
                }

                // condition (4): the two join cardinalities agree
                cfg.q_card_eq.enable(&mut region, n - 1)?;
                if !tamper && !all_clean {
                    debug_assert_eq!(
                        running, running_cln,
                        "cardinality preservation: |R^c join| != |R join|"
                    );
                }

                // output at last row, constrained equal to sum via q_out
                let out_row = n - 1;
                cfg.q_out.enable(&mut region, out_row)?;
                let out_cell = region.assign_advice(
                    || "out",
                    cfg.out,
                    out_row,
                    || Value::known(F::from(running as u64)),
                )?;
                Ok(out_cell)
            },
        )?;

        Ok(out_cell)
    }
}

// ---------------- IMPORTANT: join lookups are defined in Circuit::configure ----------------

/// Full constraint-system setup for `Path3OrdCircuit`.
/// Extracted verbatim so the circuit and any wrapper that embeds it
/// (see `crate::inline_bind`) configure IDENTICAL constraints -- the
/// chip's own `configure` alone is NOT sufficient here.
pub fn configure_path3ord_full<F: Field + Ord>(
    meta: &mut ConstraintSystem<F>,
) -> Path3OrdConfig<F> {
    let cfg = Path3OrdChip::<F>::configure(meta);

    // convenience
    let t3 = cfg.agg[0].clone(); // table from r3
    let t2 = cfg.agg[1].clone(); // table from r2

    // r1 joins to T2
    {
        let j = cfg.join[0].clone();
        let rel_dst = cfg.r[0][1];
        let t = t2.clone();

        // gap lookup: (1-in)*low in key AND (1-in)*high in key_next
        meta.lookup_any("gap r1->t2", move |m| {
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

        // map lookup: (in*dst, val, val_cln) exists in the child table.
        // ONE lookup carries both channels. When in=0 the join gate forces
        // val=val_cln=0 so the tuple is (0,0,0), which exists at map row0.
        meta.lookup_any("map r1->t2", move |m| {
            let q_in = m.query_selector(j.q_lookup_complex);
            let inx = m.query_advice(j.in_next, Rotation::cur());

            let dst = m.query_advice(rel_dst, Rotation::cur());
            let v = m.query_advice(j.val, Rotation::cur());
            let v_cln = m.query_advice(j.val_cln, Rotation::cur());

            let q_tbl = m.query_selector(t.q_map_tbl);
            let tk = m.query_advice(t.map_pair[0], Rotation::cur());
            let tv = m.query_advice(t.map_pair[1], Rotation::cur());
            let tv_cln = m.query_advice(t.map_cln, Rotation::cur());

            vec![
                (q_in.clone() * inx.clone() * dst, q_tbl.clone() * tk),
                (q_in.clone() * v, q_tbl.clone() * tv),
                (q_in * v_cln, q_tbl * tv_cln),
            ]
        });
    }

    // r2 joins to T3
    {
        let j = cfg.join[1].clone();
        let rel_dst = cfg.r[1][1];
        let t = t3.clone();

        meta.lookup_any("gap r2->t3", move |m| {
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

        meta.lookup_any("map r2->t3", move |m| {
            let q_in = m.query_selector(j.q_lookup_complex);
            let inx = m.query_advice(j.in_next, Rotation::cur());

            let dst = m.query_advice(rel_dst, Rotation::cur());
            let v = m.query_advice(j.val, Rotation::cur());
            let v_cln = m.query_advice(j.val_cln, Rotation::cur());

            let q_tbl = m.query_selector(t.q_map_tbl);
            let tk = m.query_advice(t.map_pair[0], Rotation::cur());
            let tv = m.query_advice(t.map_pair[1], Rotation::cur());
            let tv_cln = m.query_advice(t.map_cln, Rotation::cur());

            vec![
                (q_in.clone() * inx.clone() * dst, q_tbl.clone() * tk),
                (q_in.clone() * v, q_tbl.clone() * tv),
                (q_in * v_cln, q_tbl * tv_cln),
            ]
        });
    }

    cfg
}

impl<F: Field + Ord> Circuit<F> for Path3OrdCircuit<F> {
    type Config = Path3OrdConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        configure_path3ord_full(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = Path3OrdChip::construct(config);
        let out_cell = chip.assign(&mut layouter, &self.edges)?;
        chip.expose_public(&mut layouter, out_cell, 0)?;
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
        let params_path = &crate::paths::param_file(17);
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

    fn dp_expected(edges: &[Edge]) -> u64 {
        // multiplicity-aware DP:
        // T3[c] = outdegree(c)
        // T2[b] = Σ_{(b->c)} [b<c] * T3[c]
        // ans   = Σ_{(a->b)} [a<b] * T2[b]
        let mut outdeg: HashMap<u64, u64> = HashMap::new();
        for e in edges {
            let s = e.src as u64;
            *outdeg.entry(s).or_insert(0) += 1;
        }

        let mut t2: HashMap<u64, u128> = HashMap::new();
        for e in edges {
            let b = e.src as u64;
            let c = e.dst as u64;
            if b < c {
                let v = *outdeg.get(&c).unwrap_or(&0) as u128;
                *t2.entry(b).or_insert(0) += v;
            }
        }

        let mut ans: u128 = 0;
        for e in edges {
            let a = e.src as u64;
            let b = e.dst as u64;
            if a < b {
                ans += *t2.get(&b).unwrap_or(&0);
            }
        }
        ans as u64
    }

    #[test]
    #[ignore = "inherited heavy end-to-end proof; the fast check is test_cardinality_preservation"]
    fn test() {
        let base_path = &crate::paths::graph_dir();

        let mut edges = read_edges(&format!("{}/wiki/wiki_Vote.txt", base_path)).unwrap();
        // let mut edges =
        //     read_edges(&format!("{}/facebook/facebook_combined.txt", base_path)).unwrap();
        // let mut edges =
        //     read_edges_csv(&format!("{}/last/lastfm_asia_edges.csv", base_path)).unwrap();

        // edges.truncate(100);

        let cnt = dp_expected(&edges);

        let circuit = Path3OrdCircuit::<Fp> {
            edges,
            _marker: PhantomData,
        };

        let public_input = vec![Fp::from(cnt)];
        let k = 17;

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
            let proof_path = &crate::paths::proof_file("wiki_proof_q1");
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
    }

    /// The maximum gate degree of the whole constraint system. Every soundness
    /// patch in this file has to stay inside the degree the one-pass gate
    /// already pays for: a rise would double every FFT in the prover and cost
    /// far more than the patch is worth.
    ///
    /// 7 is set by the `map r*->t*` lookups, whose input `q * in_next * dst` is
    /// degree 3 against a degree-2 table (halo2 charges 2 + input + table). The
    /// widest GATE is q_r1_filter / q_r2_filter at degree 4, so the self-join
    /// gate (2), the clean-predicate gate (3) and the tagged Pairwise
    /// Consistency lookups (2 + 2 + 2 = 6) are all strictly below the bound.
    #[test]
    fn test_max_gate_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        let mut cs = ConstraintSystem::<Fp>::default();
        let _ = <Path3OrdCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
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
        assert!(
            degree <= 7,
            "the maximum gate degree rose to {}, so a soundness patch is costing \
             more than the argument it protects",
            degree
        );
    }

    /// Fast correctness check of the Pairwise Consistency and Cardinality
    /// Preservation Checks: a truncated slice of the real graph under
    /// MockProver, which verifies every gate, shuffle and lookup of the circuit
    /// without paying for a real proof. Three directions:
    ///
    ///   1. the honest, fully reduced partition is accepted,
    ///   2. hiding one joinable root tuple is rejected by condition (4),
    ///   3. the all-clean partition, which condition (4) accepts for free, is
    ///      rejected by condition (3).
    #[test]
    fn test_cardinality_preservation() {
        // a slice small enough for MockProver but large enough that the
        // reduction actually drops tuples on all three tree nodes (76 of 3000
        // edges survive on r1, 46 on r2, 799 on r3)
        const N_EDGES: usize = 3000;
        let k = 13;

        let base_path = &crate::paths::graph_dir();
        let mut edges = read_edges(&format!("{}/wiki/wiki_Vote.txt", base_path))
            .unwrap_or_else(|e| panic!("graph dataset not readable: {}", e));
        edges.truncate(N_EDGES);
        assert!(!edges.is_empty(), "graph dataset is empty");

        let cnt = dp_expected(&edges);
        assert!(
            cnt > 0,
            "the slice has no 3-path, the test would be vacuous"
        );

        // Non-vacuity of the third direction below. Condition (3) rejects the
        // all-clean partition only if the slice really has dangling tuples, so
        // check on the host that the reduction drops rows on every tree node and
        // that the unreduced key sets of at least one join-tree edge differ.
        {
            let shifted: Vec<(u64, u64)> = edges
                .iter()
                .map(|e| (e.src as u64 + SHIFT_ID, e.dst as u64 + SHIFT_ID))
                .collect();
            let (c1, c2, c3) = Path3OrdChip::<Fp>::clean_indicators(&shifted, false);
            let live = |c: &Vec<u64>| c.iter().filter(|&&x| x == 1).count();
            assert!(
                live(&c1) < shifted.len() && live(&c2) < shifted.len() && live(&c3) < shifted.len(),
                "the slice has no dangling tuple, condition (3) would accept the \
                 all-clean partition and the third direction would be vacuous: \
                 |R1^c|={} |R2^c|={} |R3^c|={} of {}",
                live(&c1),
                live(&c2),
                live(&c3),
                shifted.len()
            );
            let all_src: HashSet<u64> = shifted.iter().map(|&(s, _)| s).collect();
            let all_dst: HashSet<u64> = shifted.iter().map(|&(_, d)| d).collect();
            assert!(
                all_src != all_dst,
                "pi_src(Edge) == pi_dst(Edge) on this slice, so the all-clean \
                 partition satisfies Pairwise Consistency and the third direction \
                 would be vacuous"
            );
        }

        let circuit = Path3OrdCircuit::<Fp> {
            edges,
            _marker: PhantomData,
        };
        let public_input = vec![Fp::from(cnt)];

        let prover = MockProver::run(k, &circuit, vec![public_input.clone()]).unwrap();
        prover.assert_satisfied();

        // Second direction: the same witness with one joinable root tuple
        // hidden in the residual side, and the two children re-reduced around
        // it so that Conservation, Non-Membership and Pairwise Consistency all
        // still hold and the input channel, hence the public COUNT, is
        // unchanged. Only condition (4) can see this, so the circuit must now
        // reject, and it must reject through the cardinality equality.
        HIDE_ONE_CLEAN_TUPLE.store(true, Ordering::Relaxed);
        let tampered = MockProver::run(k, &circuit, vec![public_input.clone()]).unwrap();
        let verdict = tampered.verify();
        HIDE_ONE_CLEAN_TUPLE.store(false, Ordering::Relaxed);

        let failures = verdict.expect_err("condition (4) accepted a hidden joinable tuple");
        println!(
            "direction 2 rejected by: {:?}",
            failures
                .iter()
                .map(|f| {
                    let d = format!("{:?}", f);
                    let one: String = d.split_whitespace().collect::<Vec<_>>().join(" ");
                    one.chars().take(110).collect::<String>()
                })
                .collect::<HashSet<String>>()
        );
        // Every failure has to be the cardinality equality. The tampered
        // witness satisfies the three Conservation Checks and every other
        // constraint of the circuit, including the public COUNT, so condition
        // (4) is the only thing between a verifier and a partition that has
        // quietly dropped a join witness.
        assert!(
            !failures.is_empty()
                && failures
                    .iter()
                    .all(|f| format!("{:?}", f).contains("cardinality preservation")),
            "expected the Cardinality Preservation Check to be the only failure: {:?}",
            failures
        );

        // Third direction: the escape condition (3) closes. The prover skips the
        // reduction and declares every real tuple clean. Conservation still
        // holds, and with R^c = R both channels of condition (4) compute the
        // same number on every row, so the cardinality equality is satisfied for
        // free and cannot see that the partition is not the reduced instance.
        // Pairwise Consistency does see it: the slice has tuples whose join key
        // has no partner, so the key sets of an edge differ.
        MARK_ALL_CLEAN.store(true, Ordering::Relaxed);
        let unreduced = MockProver::run(k, &circuit, vec![public_input]).unwrap();
        let verdict = unreduced.verify();
        MARK_ALL_CLEAN.store(false, Ordering::Relaxed);

        let failures = verdict.expect_err("condition (3) accepted the all-clean partition");
        let names: Vec<String> = failures.iter().map(|f| format!("{:?}", f)).collect();
        println!(
            "direction 3 rejected by: {:?}",
            names
                .iter()
                .map(|n| {
                    let one: String = n.split_whitespace().collect::<Vec<_>>().join(" ");
                    one.chars().take(110).collect::<String>()
                })
                .collect::<HashSet<String>>()
        );
        assert!(
            failures.iter().any(|f| matches!(
                f,
                VerifyFailure::Lookup { name, .. } if name.starts_with("pw: ")
            )),
            "expected a Pairwise Consistency lookup to reject the all-clean \
             partition, got: {:?}",
            names
        );
        // The all-clean partition also violates the predicate gate, because the
        // slice has 46 edges with src >= dst that it declares clean. That is a
        // second, independent reason to reject, and it is what shows the
        // predicate gate is not vacuous on real data. It does not weaken the
        // evidence above: the four Pairwise Consistency lookups still fire on
        // their own rows, so condition (3) is still doing the work it exists for.
        assert!(
            names
                .iter()
                .any(|f| f.contains("clean rows satisfy the local predicate")),
            "expected the clean part of the all-clean partition to violate the \
             local predicate a<b<c as well: {:?}",
            names
        );

        // and condition (4) really is blind to this partition, which is why
        // condition (3) has to exist
        assert!(
            names
                .iter()
                .all(|f| !f.contains("cardinality preservation")),
            "the all-clean partition was expected to satisfy condition (4): {:?}",
            names
        );
    }
}
