use ff::Field;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Expression, Instance, Selector},
    poly::Rotation,
};
use halo2curves::pasta::Fp;

use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::graph_sql::g_sql4_obj::{IndexedViewChip, IndexedViewConfig};
use std::collections::HashMap;

/// Sentinel vertex id for dummy/padding rows; never a real vertex.
pub const PAD: u64 = u64::MAX;
/// Real vertex ids are shifted by +1 at assignment time, so no live cell is
/// ever 0.  Without this, the all-zero tail rows of the (ungated) view tables
/// would satisfy a fabricated row `(0, 0, 0)`.  Same convention as
/// `g_sql4_obj::SHIFT_ID`.
const SHIFT_ID: u64 = 1;
/// Maximum chain length supported (GQ2/GQ4 join 4 edge instances).
pub const MAX_LEVELS: usize = 4;
/// Byte width of the ordering comparisons.  Matches the `NUM_BYTES` of
/// `g_sql1_obj`/`g_sql4_obj`, so the baseline pays the same per-comparison
/// price as the VPJoin circuits it is compared against.
const NUM_BYTES: usize = 8;

#[derive(Clone, Debug)]
pub struct PoneBaselineConfig {
    // The Edge relation enters through two indexed sorted views -- the SAME
    // `IndexedViewChip` the VPJoin graph circuits use for their bag
    // materializations (`g_sql4_obj`'s `in_by_dst`/`out_by_src`), so the
    // baseline's per-row join machinery is the per-row machinery of a
    // verified binary join, not a bare membership check.
    view_in: IndexedViewConfig<Fp>,  // keyed by dst, val = src
    view_out: IndexedViewConfig<Fp>, // keyed by src, val = dst
    /// Indexed view over level u's output table (keyed by `last`, val =
    /// `start`), the left lookup table of level u + 1.
    view_lvl: Vec<IndexedViewConfig<Fp>>,
    // Levels 2..=levels: materialized intermediates (index t = level - 2).
    p_start: Vec<Column<Advice>>,
    p_mid: Vec<Column<Advice>>,
    p_last: Vec<Column<Advice>>,
    p_dummy: Vec<Column<Advice>>,
    /// Witness inverse of (start - PAD); forces counted rows off the sentinel.
    p_inv: Vec<Column<Advice>>,
    /// View indices carried by each row: `p_i` selects the parent row inside
    /// the left view's key group, `p_j` the edge inside `view_out`'s.
    p_i: Vec<Column<Advice>>,
    p_j: Vec<Column<Advice>>,
    q_level: Vec<Selector>,
    q_close: Vec<Selector>,
    /// Ascending-vertex predicates (see [`orders`]).  `lt_ml[t]` checks
    /// `mid < last` for the edge level `t` appends; `lt_sm` additionally
    /// checks `start < mid` on level 0, whose row holds the whole two-edge
    /// path.  Configured for every level so the constraint system is
    /// identical across queries -- the extrapolation divides two timings of
    /// the same shape, so `configure` must not depend on `levels`.
    lt_sm: LtConfig<Fp, NUM_BYTES>,
    lt_ml: Vec<LtConfig<Fp, NUM_BYTES>>,
    q_ord: Vec<Selector>,
    // COUNT(*) accumulator over the final level.
    final_dummy: Column<Advice>,
    acc: Column<Advice>,
    q_acc0: Selector,
    q_acc: Selector,
    instance: Column<Instance>,
}

/// PoneglyphDB-style binary-join chain over Edge.
///
/// `levels` = number of Edge instances joined (3 for GQ1/GQ3, 4 for
/// GQ2/GQ4); `cyclic` enforces the closure `last = start` on counted final
/// rows (GQ3/GQ4).  `capacities[t]` is the materialization capacity of level
/// `t + 2` (`capacities.len() == levels - 1`); slots beyond the true
/// intermediate size hold PAD dummies.
#[derive(Clone, Debug, Default)]
pub struct PoneBaselineCircuit {
    pub edges: Vec<(u64, u64)>,
    pub levels: usize,
    pub cyclic: bool,
    pub capacities: Vec<usize>,
    /// Test-only adversarial knob: marks this many spare final-level dummy
    /// slots as counted PAD rows; the dummy-discipline gate must reject it.
    pub test_inflate_count: usize,
    /// Test-only adversarial knob: witness the UNORDERED walks, so counted
    /// rows violate the ascending predicates.  The ordering gates must reject
    /// it -- otherwise those gates are decorative and the baseline is silently
    /// still proving the weaker "count every walk" query.
    pub test_drop_ordering: bool,
    /// Test-only adversarial knob: mis-index the first real row's `p_j` by 1,
    /// so it references a view occurrence that does not carry its value.  The
    /// indexed lookups must reject it -- otherwise they degenerate to bare
    /// membership and rows are not bound to specific occurrences.
    pub test_scramble_indices: bool,
}

/// Is the ascending-vertex predicate enforced when level index `t` appends its
/// edge?
///
/// All four queries constrain the first `levels` vertices to ascend and leave
/// the last one free (it is unconstrained for the paths, and pinned to the
/// start by the closure gate for the cycles):
///
///   GQ1  a<b<c, d free        GQ3  a<b<c, closes to a
///   GQ2  a<b<c<d, e free      GQ4  a<b<c<d, closes to a
///
/// Level `t` appends the edge `mid -> last`, so it owns the predicate
/// `mid < last`; level 0 additionally owns `start < mid`, since its row
/// carries the whole two-edge path.  The final level (`t == levels - 2`) owns
/// no predicate, which is exactly the "last vertex is free" rule above.
pub fn orders(t: usize, levels: usize) -> bool {
    t + 2 < levels
}

/// Enumerate the true intermediates |P_2|, ..., |P_levels| and the final
/// count (closed walks for cyclic queries).  Used to size the anchor
/// execution and to fill witnesses.
///
/// The ordering predicates of [`orders`] are applied here, so the rows this
/// returns are exactly the rows the circuit's gates accept.
pub fn enumerate_paths(
    edges: &[(u64, u64)],
    levels: usize,
    cyclic: bool,
) -> (Vec<Vec<(u64, u64, u64)>>, u64) {
    enumerate_paths_inner(edges, levels, cyclic, true)
}

/// `enumerate_paths` with the ordering predicates switchable.
///
/// `ordered = false` reproduces the pre-ordering behaviour (every walk, no
/// ascending constraint) and exists so a test can hand the circuit rows its
/// own gates must reject; nothing outside the test knob uses it.
fn enumerate_paths_inner(
    edges: &[(u64, u64)],
    levels: usize,
    cyclic: bool,
    ordered: bool,
) -> (Vec<Vec<(u64, u64, u64)>>, u64) {
    use std::collections::HashMap;
    let orders = |t: usize, levels: usize| ordered && orders(t, levels);
    let mut out_adj: HashMap<u64, Vec<u64>> = HashMap::new();
    for &(s, d) in edges {
        out_adj.entry(s).or_default().push(d);
    }
    // entry t corresponds to P_{t+2}: rows (start, mid, last), mid = join key
    let mut lvls: Vec<Vec<(u64, u64, u64)>> = Vec::new();
    let ord0 = orders(0, levels);
    let mut cur: Vec<(u64, u64, u64)> = edges
        .iter()
        .filter(|&&(a, b)| !ord0 || a < b)
        .flat_map(|&(a, b)| {
            out_adj
                .get(&b)
                .into_iter()
                .flatten()
                .filter(move |&&c| !ord0 || b < c)
                .map(move |&c| (a, b, c))
        })
        .collect();
    lvls.push(cur.clone());
    while lvls.len() < levels - 1 {
        let ord = orders(lvls.len(), levels);
        cur = cur
            .iter()
            .flat_map(|&(a, _m, l)| {
                out_adj
                    .get(&l)
                    .into_iter()
                    .flatten()
                    .filter(move |&&c| !ord || l < c)
                    .map(move |&c| (a, l, c))
            })
            .collect();
        lvls.push(cur.clone());
    }
    let count = if cyclic {
        lvls.last()
            .unwrap()
            .iter()
            .filter(|&&(a, _, l)| l == a)
            .count() as u64
    } else {
        lvls.last().unwrap().len() as u64
    };
    (lvls, count)
}

/// Sizes `|P_1| ..= |P_levels|` of the TRUE (unpadded) intermediates, counted
/// without materializing any of them.
///
/// `enumerate_paths` builds every row, which is impossible at full dataset
/// scale (|P_3| is 202M rows on wiki).  Sizing the anchor only needs the
/// counts, and they satisfy a cheap O(|E|) per level recurrence: if `c_t[v]`
/// is the number of `P_t` rows ending at `v`, then
/// `c_{t+1}[w] = sum over edges (v,w) of c_t[v]`, and `|P_t| = sum_v c_t[v]`.
///
/// Counts are `u128` for type-compatibility with `circuit_rows`, which is
/// shared with the fully padded sizing where the public bound `m^4` (6.1e19
/// on facebook, 1.2e20 on wiki) does overflow `u64`.  The true counts
/// themselves stay small: `|P_4|` peaks at 9.1e9 on wiki.
pub fn level_sizes(edges: &[(u64, u64)], levels: usize) -> Vec<u128> {
    use std::collections::HashMap;
    let mut sizes: Vec<u128> = vec![edges.len() as u128];
    // c[v] = number of P_t rows ending at v.  Seeded with P_1, carrying
    // level 0's `start < mid` predicate.
    let mut c: HashMap<u64, u128> = HashMap::new();
    for &(s, d) in edges {
        if !orders(0, levels) || s < d {
            *c.entry(d).or_insert(0) += 1;
        }
    }
    while sizes.len() < levels {
        // Appending edge (v, w) builds P_{t+2} from P_{t+1}, so this step is
        // owned by level index t = sizes.len() - 1.
        let ord = orders(sizes.len() - 1, levels);
        let mut next: HashMap<u64, u128> = HashMap::new();
        for &(v, w) in edges {
            if ord && v >= w {
                continue;
            }
            match c.get(&v) {
                Some(&n) if n != 0 => *next.entry(w).or_insert(0) += n,
                _ => {}
            }
        }
        sizes.push(next.values().sum());
        c = next;
    }
    sizes
}

/// Row count of the circuit whose Edge table holds `edges` rows and whose
/// materialized levels have `capacities`.
///
/// `synthesize` places the edge views and every level in ONE region, each in
/// its OWN column group, all starting at row 0, so the region height is the
/// MAXIMUM of the per-group heights, not their sum.  The tallest group needs
/// two extra rows: an indexed view over a group of height h occupies h + 1
/// input rows (the appended PAD entry) plus the chip's own sentinel row, and
/// the final level's accumulator writes to row `cap`.  Using the maximum
/// matters twice over: it is what the layout actually costs, and it keeps the
/// anchor and the padded circuit on the same formula, which is what makes
/// their ratio a clean domain scaling.
pub fn circuit_rows(edges: u128, capacities: &[u128]) -> u128 {
    capacities.iter().copied().fold(edges, u128::max) + 2
}

impl Circuit<Fp> for PoneBaselineCircuit {
    type Config = PoneBaselineConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            edges: Vec::new(),
            levels: self.levels,
            cyclic: self.cyclic,
            capacities: self.capacities.clone(),
            test_inflate_count: self.test_inflate_count,
            test_drop_ordering: self.test_drop_ordering,
            test_scramble_indices: self.test_scramble_indices,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let n_lvls = MAX_LEVELS - 1; // configure the maximum; unused levels stay empty

        // Edge views and level views.  Configured unconditionally so the
        // constraint system is identical for every query -- the extrapolation
        // divides two timings of the same shape, so `configure` must not
        // depend on `levels`.
        let view_in = IndexedViewChip::<Fp>::configure(meta);
        let view_out = IndexedViewChip::<Fp>::configure(meta);
        let view_lvl: Vec<IndexedViewConfig<Fp>> = (0..n_lvls - 1)
            .map(|_| IndexedViewChip::<Fp>::configure(meta))
            .collect();

        let mut p_start = Vec::new();
        let mut p_mid = Vec::new();
        let mut p_last = Vec::new();
        let mut p_dummy = Vec::new();
        let mut p_inv = Vec::new();
        let mut p_i = Vec::new();
        let mut p_j = Vec::new();
        let mut q_level = Vec::new();
        let mut q_close = Vec::new();
        for _ in 0..n_lvls {
            p_start.push(meta.advice_column());
            p_mid.push(meta.advice_column());
            p_last.push(meta.advice_column());
            let d = meta.advice_column();
            meta.enable_equality(d);
            p_dummy.push(d);
            p_inv.push(meta.advice_column());
            p_i.push(meta.advice_column());
            p_j.push(meta.advice_column());
            q_level.push(meta.complex_selector());
            q_close.push(meta.selector());
        }

        // Ordering predicates.  The selector is separate from `q_level` so a
        // level can materialize rows without owning a predicate (the final
        // level always does).
        let q_ord: Vec<Selector> = (0..n_lvls).map(|_| meta.selector()).collect();
        // `Selector`/`Column` are Copy; the Vecs holding them are not, so pull
        // the elements out before they cross into the `move` closures.
        let (q0, s0, m0) = (q_ord[0], p_start[0], p_mid[0]);
        let lt_sm = LtChip::<Fp, NUM_BYTES>::configure(
            meta,
            move |m| m.query_selector(q0),
            move |m| m.query_advice(s0, Rotation::cur()),
            move |m| m.query_advice(m0, Rotation::cur()),
        );
        let mut lt_ml: Vec<LtConfig<Fp, NUM_BYTES>> = Vec::new();
        for t in 0..n_lvls {
            let (q, mid, last) = (q_ord[t], p_mid[t], p_last[t]);
            lt_ml.push(LtChip::<Fp, NUM_BYTES>::configure(
                meta,
                move |m| m.query_selector(q),
                move |m| m.query_advice(mid, Rotation::cur()),
                move |m| m.query_advice(last, Rotation::cur()),
            ));
        }

        for t in 0..n_lvls {
            // A counted row must satisfy its level's ascending predicates.
            // Dummy rows are exempt: they hold PAD in every vertex column, so
            // PAD < PAD is false and requiring the predicate would reject
            // every padded slot.
            meta.create_gate("ascending vertices", |m| {
                let q = m.query_selector(q_ord[t]);
                let d = m.query_advice(p_dummy[t], Rotation::cur());
                let one = Expression::Constant(Fp::ONE);
                let live = q * (one.clone() - d);
                let mut cs = vec![live.clone() * (one.clone() - lt_ml[t].is_lt(m, None))];
                if t == 0 {
                    cs.push(live * (one - lt_sm.is_lt(m, None)));
                }
                cs
            });
        }

        for t in 0..n_lvls {
            // Dummy flag is boolean; dummy rows hold the PAD sentinel, and
            // counted rows must NOT hold it: (1 - d) * ((start - PAD) * inv - 1)
            // = 0 forces start != PAD whenever d = 0 (inv witnesses the
            // inverse), so sentinel rows can never inflate the count.
            meta.create_gate("dummy discipline", |m| {
                let q = m.query_selector(q_level[t]);
                let d = m.query_advice(p_dummy[t], Rotation::cur());
                let start = m.query_advice(p_start[t], Rotation::cur());
                let inv = m.query_advice(p_inv[t], Rotation::cur());
                let one = Expression::Constant(Fp::ONE);
                let pad = Expression::Constant(Fp::from(PAD));
                vec![
                    q.clone() * d.clone() * (one.clone() - d.clone()),
                    q.clone() * d.clone() * (start.clone() - pad.clone()),
                    q * (one.clone() - d) * ((start - pad) * inv - one),
                ]
            });

            // Left lookup, indexed: (mid, i, start) must be a (key, idx, val)
            // row of the left view -- the Edge-by-dst view for level 0, the
            // previous level's view otherwise.  Carrying the index binds the
            // row to one specific parent occurrence instead of bare
            // membership, exactly as `g_sql4_obj`'s t12/t34 rows do against
            // in_by_dst/out_by_src.  Table side ungated, as there: unassigned
            // tail rows are all-zero, and no live row can be (0, 0, 0)
            // because ids are SHIFT_ID-shifted.
            {
                let left = if t == 0 { &view_in } else { &view_lvl[t - 1] };
                let (lk, lidx, lv) = (left.sorted_key, left.idx, left.sorted_val);
                let (ql, s, mm, pi) = (q_level[t], p_start[t], p_mid[t], p_i[t]);
                meta.lookup_any("left parent (indexed)", move |m| {
                    let q = m.query_selector(ql);
                    vec![
                        (
                            q.clone() * m.query_advice(mm, Rotation::cur()),
                            m.query_advice(lk, Rotation::cur()),
                        ),
                        (
                            q.clone() * m.query_advice(pi, Rotation::cur()),
                            m.query_advice(lidx, Rotation::cur()),
                        ),
                        (
                            q * m.query_advice(s, Rotation::cur()),
                            m.query_advice(lv, Rotation::cur()),
                        ),
                    ]
                });
            }

            // Right lookup, indexed: (mid, j, last) in the Edge-by-src view.
            {
                let (rk, ridx, rv) = (view_out.sorted_key, view_out.idx, view_out.sorted_val);
                let (ql, mm, l, pj) = (q_level[t], p_mid[t], p_last[t], p_j[t]);
                meta.lookup_any("right edge (indexed)", move |m| {
                    let q = m.query_selector(ql);
                    vec![
                        (
                            q.clone() * m.query_advice(mm, Rotation::cur()),
                            m.query_advice(rk, Rotation::cur()),
                        ),
                        (
                            q.clone() * m.query_advice(pj, Rotation::cur()),
                            m.query_advice(ridx, Rotation::cur()),
                        ),
                        (
                            q * m.query_advice(l, Rotation::cur()),
                            m.query_advice(rv, Rotation::cur()),
                        ),
                    ]
                });
            }

            // Cycle closure on counted rows: last = start (enabled by
            // synthesize only on the final level of a cyclic query).
            meta.create_gate("cycle closure", |m| {
                let q = m.query_selector(q_close[t]);
                let d = m.query_advice(p_dummy[t], Rotation::cur());
                let one = Expression::Constant(Fp::ONE);
                let last = m.query_advice(p_last[t], Rotation::cur());
                let start = m.query_advice(p_start[t], Rotation::cur());
                vec![q * (one - d) * (last - start)]
            });
        }

        // COUNT(*) accumulator: acc[0] = 0; acc[i+1] = acc[i] + (1 - final_dummy[i]).
        // final_dummy is copy-constrained to the final level's dummy column.
        let final_dummy = meta.advice_column();
        meta.enable_equality(final_dummy);
        let acc = meta.advice_column();
        meta.enable_equality(acc);
        let instance = meta.instance_column();
        meta.enable_equality(instance);
        let q_acc0 = meta.selector();
        let q_acc = meta.selector();
        meta.create_gate("acc init", |m| {
            let q = m.query_selector(q_acc0);
            vec![q * m.query_advice(acc, Rotation::cur())]
        });
        meta.create_gate("acc step", |m| {
            let q = m.query_selector(q_acc);
            let d = m.query_advice(final_dummy, Rotation::cur());
            let a_cur = m.query_advice(acc, Rotation::cur());
            let a_next = m.query_advice(acc, Rotation::next());
            let one = Expression::Constant(Fp::ONE);
            vec![q * (a_next - a_cur - (one - d))]
        });

        PoneBaselineConfig {
            view_in,
            view_out,
            view_lvl,
            p_start,
            p_mid,
            p_last,
            p_dummy,
            p_inv,
            p_i,
            p_j,
            q_level,
            q_close,
            lt_sm,
            lt_ml,
            q_ord,
            final_dummy,
            acc,
            q_acc0,
            q_acc,
            instance,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        assert!(self.levels >= 3 && self.levels <= MAX_LEVELS);
        assert_eq!(self.capacities.len(), self.levels - 1);
        let (lvls, _count) = enumerate_paths_inner(
            &self.edges,
            self.levels,
            self.cyclic,
            !self.test_drop_ordering,
        );

        // ---- host-side witness preparation --------------------------------
        let m = self.edges.len();
        let sh = |v: u64| v + SHIFT_ID;

        // Edge views: (key, val, eid), ids shifted, one PAD entry appended so
        // dummy rows have a (PAD, 0, PAD) target.
        let mut vin_rows: Vec<(u64, u64, u64)> = self
            .edges
            .iter()
            .enumerate()
            .map(|(e, &(s, d))| (sh(d), sh(s), e as u64))
            .collect();
        vin_rows.push((PAD, PAD, m as u64));
        let mut vout_rows: Vec<(u64, u64, u64)> = self
            .edges
            .iter()
            .enumerate()
            .map(|(e, &(s, d))| (sh(s), sh(d), e as u64))
            .collect();
        vout_rows.push((PAD, PAD, m as u64));

        // Assigned slot values per level: (start, mid, last, real), shifted.
        let mut slots: Vec<Vec<(u64, u64, u64, bool)>> = Vec::new();
        for t in 0..(self.levels - 1) {
            let is_final = t == self.levels - 2;
            let cap = self.capacities[t];
            assert!(
                lvls[t].len() <= cap,
                "level {} true size {} exceeds capacity {}",
                t + 2,
                lvls[t].len(),
                cap
            );
            let mut v = Vec::with_capacity(cap);
            for i in 0..cap {
                v.push(match lvls[t].get(i) {
                    Some(&(a, b, c)) => {
                        // On the final level of a cyclic query, only closed
                        // walks are counted; open walks occupy their padded
                        // slots as dummies (their capacity is still charged,
                        // as in PoneglyphDB's padded intermediates).
                        let keep = !(is_final && self.cyclic) || c == a;
                        if keep {
                            (sh(a), sh(b), sh(c), true)
                        } else {
                            (PAD, PAD, PAD, false)
                        }
                    }
                    None => {
                        let inflate = is_final && i < lvls[t].len() + self.test_inflate_count;
                        (PAD, PAD, PAD, inflate)
                    }
                });
            }
            slots.push(v);
        }

        // Level views (left lookup tables of the NEXT level): the level's
        // assigned slots verbatim, keyed by `last` with `start` as payload,
        // plus the appended PAD entry.
        let mut vlvl_rows: Vec<Vec<(u64, u64, u64)>> = Vec::new();
        for t in 0..(self.levels - 2) {
            let mut v: Vec<(u64, u64, u64)> = slots[t]
                .iter()
                .enumerate()
                .map(|(i, &(sa, _sb, sc, _))| (sc, sa, i as u64))
                .collect();
            v.push((PAD, PAD, slots[t].len() as u64));
            vlvl_rows.push(v);
        }

        // (key, val) -> idx maps replicating the views' own sort (by key,
        // then eid), so each row can witness WHICH occurrence it uses.
        fn view_index(rows: &[(u64, u64, u64)]) -> HashMap<u64, Vec<(u64, u64)>> {
            let mut sorted = rows.to_vec();
            sorted.sort_by_key(|&(k, _, eid)| (k, eid));
            let mut map: HashMap<u64, Vec<(u64, u64)>> = HashMap::new();
            let mut idx = 0u64;
            let mut prev: Option<u64> = None;
            for (k, v, _) in sorted {
                idx = if prev == Some(k) { idx + 1 } else { 0 };
                prev = Some(k);
                map.entry(k).or_default().push((v, idx));
            }
            map
        }
        fn find_idx(map: &HashMap<u64, Vec<(u64, u64)>>, key: u64, val: u64, what: &str) -> u64 {
            map.get(&key)
                .and_then(|g| g.iter().find(|&&(v, _)| v == val))
                .unwrap_or_else(|| panic!("{}: ({}, {}) not found in view", what, key, val))
                .1
        }
        let vin_map = view_index(&vin_rows);
        let vout_map = view_index(&vout_rows);
        let vlvl_maps: Vec<HashMap<u64, Vec<(u64, u64)>>> =
            vlvl_rows.iter().map(|r| view_index(r)).collect();

        // ---- chips and their fixed tables ---------------------------------
        // The u8 range table backing every ordering comparison.  Its lookup is
        // ungated, so it must exist even for the levels that own no predicate.
        let sm_chip = LtChip::<Fp, NUM_BYTES>::construct(config.lt_sm);
        sm_chip.load(&mut layouter)?;
        let ml_chips: Vec<LtChip<Fp, NUM_BYTES>> = config
            .lt_ml
            .iter()
            .map(|c| LtChip::<Fp, NUM_BYTES>::construct(*c))
            .collect();
        for c in &ml_chips {
            c.load(&mut layouter)?;
        }
        let vin_chip = IndexedViewChip::<Fp>::construct(config.view_in.clone());
        let vout_chip = IndexedViewChip::<Fp>::construct(config.view_out.clone());
        let vlvl_chips: Vec<IndexedViewChip<Fp>> = config
            .view_lvl
            .iter()
            .map(|c| IndexedViewChip::<Fp>::construct(c.clone()))
            .collect();
        vin_chip.load(&mut layouter)?;
        vout_chip.load(&mut layouter)?;
        for c in &vlvl_chips {
            c.load(&mut layouter)?;
        }

        let final_cell = layouter.assign_region(
            || "pone baseline chain",
            |mut region| {
                // Views first: the edge relation twice (by dst, by src) and
                // each materialized level's table, in exactly the multiset the
                // slot assignment below uses.
                vin_chip.assign(&mut region, vin_rows.len(), &vin_rows)?;
                vout_chip.assign(&mut region, vout_rows.len(), &vout_rows)?;
                for (u, chip) in vlvl_chips.iter().enumerate() {
                    match vlvl_rows.get(u) {
                        Some(rows) => chip.assign(&mut region, rows.len(), rows)?,
                        // Level absent for this query shape; sentinel only.
                        None => chip.assign(&mut region, 0, &[])?,
                    }
                }

                let mut final_acc_cell = None;
                for t in 0..(self.levels - 1) {
                    let is_final = t == self.levels - 2;
                    let cap = self.capacities[t];
                    if is_final {
                        config.q_acc0.enable(&mut region, 0)?;
                        region.assign_advice(
                            || "acc0",
                            config.acc,
                            0,
                            || Value::known(Fp::ZERO),
                        )?;
                    }
                    let mut acc = 0u64;
                    let ord = orders(t, self.levels);
                    let left_map = if t == 0 { &vin_map } else { &vlvl_maps[t - 1] };
                    for i in 0..cap {
                        config.q_level[t].enable(&mut region, i)?;
                        if ord {
                            config.q_ord[t].enable(&mut region, i)?;
                        }
                        let (a, b, c, real) = slots[t][i];
                        let d = if real { 0u64 } else { 1u64 };
                        // View indices: which parent occurrence and which edge
                        // occurrence this row uses.  Dummy (and inflated-PAD)
                        // rows use index 0, whose (PAD, 0, PAD) target exists
                        // in every view via the appended PAD entry.
                        let (vi, vj) = if real && a != PAD {
                            (
                                find_idx(left_map, b, a, "left parent"),
                                find_idx(&vout_map, b, c, "right edge"),
                            )
                        } else {
                            (0, 0)
                        };
                        // Adversarial knob: mis-index the first real row; the
                        // indexed lookup must reject the proof.
                        let vj = if self.test_scramble_indices && t == 0 && real && i == 0 {
                            vj + 1
                        } else {
                            vj
                        };
                        let inv = if d == 0 {
                            Fp::from(a) - Fp::from(PAD)
                        } else {
                            Fp::ZERO
                        };
                        let inv = Option::<Fp>::from(inv.invert()).unwrap_or(Fp::ZERO);
                        // Comparison witnesses. Assigned on every level, not
                        // just the ones that own a predicate: the diff bytes'
                        // range lookup is ungated, so the cells it reads must
                        // hold a value.
                        ml_chips[t].assign(
                            &mut region,
                            i,
                            Value::known(Fp::from(b)),
                            Value::known(Fp::from(c)),
                        )?;
                        if t == 0 {
                            sm_chip.assign(
                                &mut region,
                                i,
                                Value::known(Fp::from(a)),
                                Value::known(Fp::from(b)),
                            )?;
                        }
                        region.assign_advice(
                            || "p_inv",
                            config.p_inv[t],
                            i,
                            || Value::known(inv),
                        )?;
                        region.assign_advice(
                            || "p_start",
                            config.p_start[t],
                            i,
                            || Value::known(Fp::from(a)),
                        )?;
                        region.assign_advice(
                            || "p_mid",
                            config.p_mid[t],
                            i,
                            || Value::known(Fp::from(b)),
                        )?;
                        region.assign_advice(
                            || "p_last",
                            config.p_last[t],
                            i,
                            || Value::known(Fp::from(c)),
                        )?;
                        region.assign_advice(
                            || "p_i",
                            config.p_i[t],
                            i,
                            || Value::known(Fp::from(vi)),
                        )?;
                        region.assign_advice(
                            || "p_j",
                            config.p_j[t],
                            i,
                            || Value::known(Fp::from(vj)),
                        )?;
                        let d_cell = region.assign_advice(
                            || "p_dummy",
                            config.p_dummy[t],
                            i,
                            || Value::known(Fp::from(d)),
                        )?;
                        if is_final {
                            if self.cyclic {
                                config.q_close[t].enable(&mut region, i)?;
                            }
                            config.q_acc.enable(&mut region, i)?;
                            let fd_cell = region.assign_advice(
                                || "final_dummy",
                                config.final_dummy,
                                i,
                                || Value::known(Fp::from(d)),
                            )?;
                            region.constrain_equal(d_cell.cell(), fd_cell.cell())?;
                            let next = acc + (1 - d);
                            let cell = region.assign_advice(
                                || "acc",
                                config.acc,
                                i + 1,
                                || Value::known(Fp::from(next)),
                            )?;
                            acc = next;
                            if i + 1 == cap {
                                final_acc_cell = Some(cell);
                            }
                        }
                    }
                }
                Ok(final_acc_cell.expect("final accumulator"))
            },
        )?;
        layouter.constrain_instance(final_cell.cell(), config.instance, 0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;

    /// Two undirected triangles sharing an edge, directed both ways.
    fn tiny_graph() -> Vec<(u64, u64)> {
        let und = [(1u64, 2u64), (1, 3), (2, 3), (2, 4), (3, 4)];
        und.iter().flat_map(|&(a, b)| [(a, b), (b, a)]).collect()
    }

    /// Independent brute force over edge occurrences: walks
    /// `v_0 -> ... -> v_levels` whose first `levels` vertices strictly ascend,
    /// optionally closing back to `v_0`.  Written from the query definition,
    /// not from `enumerate_paths`, so agreement is real evidence.
    fn brute(edges: &[(u64, u64)], levels: usize, cyclic: bool) -> u64 {
        fn rec(
            edges: &[(u64, u64)],
            path: &mut Vec<u64>,
            levels: usize,
            cyclic: bool,
            n: &mut u64,
        ) {
            if path.len() == levels + 1 {
                if !cyclic || path[levels] == path[0] {
                    *n += 1;
                }
                return;
            }
            let last = *path.last().unwrap();
            for &(s, d) in edges {
                if s != last {
                    continue;
                }
                // the ascending predicate covers vertices 0..levels-1, i.e. it
                // constrains d only while d is not the final vertex
                if path.len() < levels && d <= last {
                    continue;
                }
                path.push(d);
                rec(edges, path, levels, cyclic, n);
                path.pop();
            }
        }
        let mut n = 0;
        let starts: std::collections::BTreeSet<u64> = edges.iter().map(|&(s, _)| s).collect();
        for s in starts {
            let mut path = vec![s];
            rec(edges, &mut path, levels, cyclic, &mut n);
        }
        n
    }

    fn check(edges: &[(u64, u64)], levels: usize, cyclic: bool, k: u32) -> u64 {
        let (lvls, count) = enumerate_paths(edges, levels, cyclic);
        assert_eq!(
            count,
            brute(edges, levels, cyclic),
            "levels={} cyclic={}: enumerate_paths disagrees with brute force",
            levels,
            cyclic
        );
        let capacities: Vec<usize> = lvls.iter().map(|l| l.len().max(1)).collect();
        let circuit = PoneBaselineCircuit {
            edges: edges.to_vec(),
            levels,
            cyclic,
            capacities,
            ..Default::default()
        };
        let prover = MockProver::run(k, &circuit, vec![vec![Fp::from(count)]]).unwrap();
        assert_eq!(
            prover.verify(),
            Ok(()),
            "levels={} cyclic={}: constraints unsatisfied at count {}",
            levels,
            cyclic,
            count
        );
        count
    }

    /// The four query shapes, each against brute force and the real gates.
    #[test]
    fn counts_match_bruteforce_under_ordering() {
        let e = tiny_graph();
        // GQ1 3-path, GQ3 triangle, GQ2 4-path, GQ4 4-cycle
        let gq1 = check(&e, 3, false, 11);
        let gq3 = check(&e, 3, true, 11);
        let gq2 = check(&e, 4, false, 11);
        let gq4 = check(&e, 4, true, 11);
        // tiny_graph is two triangles {1,2,3} and {2,3,4} sharing edge 2-3,
        // taken in both directions.  With A<B<C each triangle is counted once.
        assert_eq!(gq3, 2, "expected exactly the two undirected triangles");
        println!(
            "tiny_graph: gq1={} gq2={} gq3={} gq4={}",
            gq1, gq2, gq3, gq4
        );
    }

    /// The ordering GATES must have teeth, not just the enumeration.
    ///
    /// Every other test feeds witnesses that already satisfy the predicates,
    /// so a decorative gate would pass them all.  Here the circuit witnesses
    /// the unordered walks; the gates must reject.
    #[test]
    fn ordering_gates_reject_unordered_rows() {
        let edges = tiny_graph();
        for (levels, cyclic) in [(3usize, false), (3, true), (4, false), (4, true)] {
            let (lvls, count) = enumerate_paths_inner(&edges, levels, cyclic, false);
            let capacities: Vec<usize> = lvls.iter().map(|l| l.len().max(1)).collect();
            let circuit = PoneBaselineCircuit {
                edges: edges.clone(),
                levels,
                cyclic,
                capacities,
                test_drop_ordering: true,
                ..Default::default()
            };
            let prover = MockProver::run(12, &circuit, vec![vec![Fp::from(count)]]).unwrap();
            assert!(
                prover.verify().is_err(),
                "levels={} cyclic={}: unordered rows were ACCEPTED -- the ordering \
                 gates are not constraining anything",
                levels,
                cyclic
            );
        }
    }

    /// The indexed lookups must bind rows to specific view occurrences, not
    /// just to members: a row pointing at the wrong index must be rejected.
    #[test]
    fn scrambled_view_index_rejected() {
        let edges = tiny_graph();
        let (lvls, count) = enumerate_paths(&edges, 3, true);
        let capacities: Vec<usize> = lvls.iter().map(|l| l.len().max(1)).collect();
        let circuit = PoneBaselineCircuit {
            edges,
            levels: 3,
            cyclic: true,
            capacities,
            test_scramble_indices: true,
            ..Default::default()
        };
        let prover = MockProver::run(12, &circuit, vec![vec![Fp::from(count)]]).unwrap();
        assert!(
            prover.verify().is_err(),
            "a mis-indexed row was ACCEPTED -- the indexed lookups are not \
             binding rows to occurrences"
        );
    }

    /// The baseline must compute the SAME query as VPJoin, so its counts have
    /// to equal the ground truth the paper's own harness uses.  Without the
    /// ordering predicates the baseline counted every walk and these disagree
    /// by orders of magnitude.
    #[test]
    fn counts_match_the_vpjoin_ground_truth() {
        use crate::bench_queries::{count_gq1, count_gq2, count_gq3, count_gq4};
        use crate::data::graph_data_processing::Edge;

        // A directed graph with genuine cycles (the shipped LastFM/Facebook
        // files store every edge as src<dst and so contain none at all).
        let raw: Vec<(u64, u64)> = vec![
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 1),
            (1, 3),
            (3, 6),
            (6, 1),
            (2, 4),
            (4, 5),
            (5, 2),
            (5, 1),
            (6, 2),
            (4, 2),
            (3, 1),
            (2, 5),
            (1, 6),
            (3, 5),
            (5, 6),
            (2, 6),
            (1, 4),
        ];
        let edges: Vec<Edge> = raw.iter().map(|&(s, d)| Edge { src: s, dst: d }).collect();

        for (name, levels, cyclic, truth) in [
            ("gq1", 3, false, count_gq1(&edges)),
            ("gq2", 4, false, count_gq2(&edges)),
            ("gq3", 3, true, count_gq3(&edges)),
            ("gq4", 4, true, count_gq4(&edges)),
        ] {
            let (_, got) = enumerate_paths(&raw, levels, cyclic);
            assert_eq!(
                got, truth,
                "{}: baseline {} vs VPJoin ground truth {}",
                name, got, truth
            );
            assert!(truth > 0, "{}: degenerate test, ground truth is 0", name);
            println!("{}: {} (matches VPJoin ground truth)", name, truth);
        }
    }

    #[test]
    fn counted_pad_row_rejected() {
        // Adversarial witness: a spare dummy slot marked as a counted PAD
        // row must be rejected by the dummy-discipline gate, even though it
        // would pass every lookup and the closure gate.
        let edges = tiny_graph();
        let (lvls, count) = enumerate_paths(&edges, 3, true);
        let circuit = PoneBaselineCircuit {
            edges,
            levels: 3,
            cyclic: true,
            capacities: vec![lvls[0].len(), lvls[1].len() + 4],
            test_inflate_count: 1,
            ..Default::default()
        };
        let prover = MockProver::run(12, &circuit, vec![vec![Fp::from(count + 1)]]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// The O(|E|)-per-level counter must agree with brute-force enumeration,
    /// since the anchor is sized from the counter but filled from the
    /// enumerator: a mismatch would silently over- or under-run a capacity.
    #[test]
    fn level_sizes_match_enumeration() {
        let edges = tiny_graph();
        for levels in 3..=MAX_LEVELS {
            let sizes = level_sizes(&edges, levels);
            let (lvls, _) = enumerate_paths(&edges, levels, false);
            assert_eq!(sizes.len(), levels);
            assert_eq!(sizes[0], edges.len() as u128, "|P_1| is the edge count");
            for (t, l) in lvls.iter().enumerate() {
                assert_eq!(sizes[t + 1], l.len() as u128, "|P_{}|", t + 2);
            }
        }
    }

    /// The levels share one region in disjoint column groups, so the height is
    /// the max group height plus two rows (appended PAD entry + view
    /// sentinel) -- not the sum.
    #[test]
    fn circuit_rows_is_the_max_group_height() {
        assert_eq!(circuit_rows(10, &[100, 40]), 102);
        assert_eq!(circuit_rows(500, &[100, 40]), 502);
        assert_eq!(circuit_rows(10, &[]), 12);
    }

    #[test]
    fn wrong_count_rejected() {
        let edges = tiny_graph();
        let (lvls, count) = enumerate_paths(&edges, 3, true);
        let circuit = PoneBaselineCircuit {
            edges,
            levels: 3,
            cyclic: true,
            capacities: vec![lvls[0].len(), lvls[1].len()],
            ..Default::default()
        };
        let prover = MockProver::run(9, &circuit, vec![vec![Fp::from(count + 1)]]).unwrap();
        assert!(prover.verify().is_err());
    }
}
