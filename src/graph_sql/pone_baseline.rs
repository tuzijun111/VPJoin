//! PoneglyphDB-style baseline circuits for the graph queries (GQ1--GQ4).
//!
//! The released PoneglyphDB artifact provides circuits only for TPC-H, so
//! this module implements its execution strategy for the graph workloads:
//! the k-way self-join over `Edge(src,dst)` is decomposed into a chain of
//! binary joins, and **every intermediate result is materialized in the
//! witness** at a fixed capacity.  With capacities set to the true
//! intermediate sizes this is the runnable *anchor* execution of the
//! estimation methodology (Section 8.1 of the paper); padding the capacities
//! to worst-case bounds gives the fully oblivious PoneglyphDB configuration,
//! whose proving time is extrapolated from the anchor
//! (see `src/bin/pone_graph_bench.rs`).
//!
//! Level t materializes the paths that use t edges:
//!   P_1 = Edge;   P_t = P_{t-1} |x| Edge  on  P_{t-1}.last = Edge.src.
//! Each materialized row of P_t carries (start, mid, last), where mid is the
//! join key, and is verified with the same lookup-argument machinery as the
//! intra-cluster binary-join gates:
//!   (start, mid)  must appear in the P_{t-1} table   (left lookup), and
//!   (mid, last)   must appear in the Edge table      (right lookup).
//! For cyclic queries (GQ3/GQ4) the final level keeps exactly the closed
//! walks, enforced by the closure gate `last = start` on counted rows.
//! Dummy rows hold a PAD sentinel present in every lookup table, so all
//! capacities are fixed, data-independent circuit parameters, exactly as in
//! PoneglyphDB.  COUNT(*) is accumulated over non-dummy final rows and
//! exposed as the public instance.
//!
//! Like PoneglyphDB, the baseline enforces per-row membership and join-key
//! consistency of every materialized intermediate; the ordering predicates
//! of the SQL queries (src1 < src2 < ...) add O(1) range checks per row and
//! are omitted here, which only makes the baseline cheaper -- the resulting
//! extrapolated estimates remain conservative in PoneglyphDB's favor.
//!
//! Scope (cost anchor). The constraint system checks membership, join-key
//! consistency, cycle closure, non-sentinel-ness of counted rows, and the
//! count accumulation.  It does NOT bind the counted multiset to the full
//! enumeration (a prover could duplicate a genuine row into spare capacity
//! or mark genuine rows as dummies); PoneglyphDB's full gates close this
//! with additional sorted-merge/counting machinery.  Omitting it only makes
//! the baseline cheaper, so the measured anchors and the extrapolated
//! estimates remain conservative in PoneglyphDB's favor.

use ff::Field;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error, Expression, Instance, Selector,
    },
    poly::Rotation,
};
use halo2curves::pasta::Fp;

/// Sentinel vertex id for dummy/padding rows; never a real vertex.
pub const PAD: u64 = u64::MAX;
/// Maximum chain length supported (GQ2/GQ4 join 4 edge instances).
pub const MAX_LEVELS: usize = 4;

#[derive(Clone, Debug)]
pub struct PoneBaselineConfig {
    // Edge table (P_1).
    e_src: Column<Advice>,
    e_dst: Column<Advice>,
    q_edge: Selector,
    // Levels 2..=levels: materialized intermediates (index t = level - 2).
    p_start: Vec<Column<Advice>>,
    p_mid: Vec<Column<Advice>>,
    p_last: Vec<Column<Advice>>,
    p_dummy: Vec<Column<Advice>>,
    /// Witness inverse of (start - PAD); forces counted rows off the sentinel.
    p_inv: Vec<Column<Advice>>,
    q_level: Vec<Selector>,
    q_level_tbl: Vec<Selector>,
    q_close: Vec<Selector>,
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
}

/// Enumerate the true intermediates |P_2|, ..., |P_levels| and the final
/// count (closed walks for cyclic queries).  Used to size the anchor
/// execution and to fill witnesses.
pub fn enumerate_paths(
    edges: &[(u64, u64)],
    levels: usize,
    cyclic: bool,
) -> (Vec<Vec<(u64, u64, u64)>>, u64) {
    use std::collections::HashMap;
    let mut out_adj: HashMap<u64, Vec<u64>> = HashMap::new();
    for &(s, d) in edges {
        out_adj.entry(s).or_default().push(d);
    }
    // entry t corresponds to P_{t+2}: rows (start, mid, last), mid = join key
    let mut lvls: Vec<Vec<(u64, u64, u64)>> = Vec::new();
    let mut cur: Vec<(u64, u64, u64)> = edges
        .iter()
        .flat_map(|&(a, b)| {
            out_adj
                .get(&b)
                .into_iter()
                .flatten()
                .map(move |&c| (a, b, c))
        })
        .collect();
    lvls.push(cur.clone());
    while lvls.len() < levels - 1 {
        cur = cur
            .iter()
            .flat_map(|&(a, _m, l)| {
                out_adj
                    .get(&l)
                    .into_iter()
                    .flatten()
                    .map(move |&c| (a, l, c))
            })
            .collect();
        lvls.push(cur.clone());
    }
    let count = if cyclic {
        lvls.last().unwrap().iter().filter(|&&(a, _, l)| l == a).count() as u64
    } else {
        lvls.last().unwrap().len() as u64
    };
    (lvls, count)
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
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let e_src = meta.advice_column();
        let e_dst = meta.advice_column();
        let q_edge = meta.complex_selector();

        let n_lvls = MAX_LEVELS - 1; // configure the maximum; unused levels stay empty
        let mut p_start = Vec::new();
        let mut p_mid = Vec::new();
        let mut p_last = Vec::new();
        let mut p_dummy = Vec::new();
        let mut p_inv = Vec::new();
        let mut q_level = Vec::new();
        let mut q_level_tbl = Vec::new();
        let mut q_close = Vec::new();
        for _ in 0..n_lvls {
            p_start.push(meta.advice_column());
            p_mid.push(meta.advice_column());
            p_last.push(meta.advice_column());
            let d = meta.advice_column();
            meta.enable_equality(d);
            p_dummy.push(d);
            p_inv.push(meta.advice_column());
            q_level.push(meta.complex_selector());
            q_level_tbl.push(meta.complex_selector());
            q_close.push(meta.selector());
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

            // Left lookup: (start, mid) in the previous level (Edge if t = 0).
            meta.lookup_any("left component", |m| {
                let q = m.query_selector(q_level[t]);
                let a = m.query_advice(p_start[t], Rotation::cur());
                let b = m.query_advice(p_mid[t], Rotation::cur());
                let (tq, ta, tb) = if t == 0 {
                    (
                        m.query_selector(q_edge),
                        m.query_advice(e_src, Rotation::cur()),
                        m.query_advice(e_dst, Rotation::cur()),
                    )
                } else {
                    (
                        m.query_selector(q_level_tbl[t - 1]),
                        m.query_advice(p_start[t - 1], Rotation::cur()),
                        m.query_advice(p_last[t - 1], Rotation::cur()),
                    )
                };
                vec![(q.clone() * a, tq.clone() * ta), (q * b, tq * tb)]
            });

            // Right lookup: (mid, last) in the Edge table.
            meta.lookup_any("right component", |m| {
                let q = m.query_selector(q_level[t]);
                let b = m.query_advice(p_mid[t], Rotation::cur());
                let c = m.query_advice(p_last[t], Rotation::cur());
                let tq = m.query_selector(q_edge);
                let ts = m.query_advice(e_src, Rotation::cur());
                let td = m.query_advice(e_dst, Rotation::cur());
                vec![(q.clone() * b, tq.clone() * ts), (q * c, tq * td)]
            });

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
            e_src,
            e_dst,
            q_edge,
            p_start,
            p_mid,
            p_last,
            p_dummy,
            p_inv,
            q_level,
            q_level_tbl,
            q_close,
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
        let (lvls, _count) = enumerate_paths(&self.edges, self.levels, self.cyclic);

        let final_cell = layouter.assign_region(
            || "pone baseline chain",
            |mut region| {
                // Edge table with one PAD sentinel row.
                for (i, &(s, d)) in self
                    .edges
                    .iter()
                    .chain(std::iter::once(&(PAD, PAD)))
                    .enumerate()
                {
                    config.q_edge.enable(&mut region, i)?;
                    region.assign_advice(|| "e_src", config.e_src, i, || Value::known(Fp::from(s)))?;
                    region.assign_advice(|| "e_dst", config.e_dst, i, || Value::known(Fp::from(d)))?;
                }

                let mut final_acc_cell = None;
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
                    if is_final {
                        config.q_acc0.enable(&mut region, 0)?;
                        region.assign_advice(|| "acc0", config.acc, 0, || Value::known(Fp::ZERO))?;
                    }
                    let mut acc = 0u64;
                    for i in 0..cap {
                        config.q_level[t].enable(&mut region, i)?;
                        config.q_level_tbl[t].enable(&mut region, i)?;
                        // On the final level of a cyclic query, only closed
                        // walks are counted; open walks occupy their padded
                        // slots as dummies (their capacity is still charged,
                        // as in PoneglyphDB's padded intermediates).
                        let (a, b, c, real) = match lvls[t].get(i) {
                            Some(&(a, b, c)) => {
                                let keep = !(is_final && self.cyclic) || c == a;
                                if keep {
                                    (a, b, c, true)
                                } else {
                                    (PAD, PAD, PAD, false)
                                }
                            }
                            None => {
                                let inflate =
                                    is_final && i < lvls[t].len() + self.test_inflate_count;
                                (PAD, PAD, PAD, inflate)
                            }
                        };
                        let d = if real { 0u64 } else { 1u64 };
                        let inv = if d == 0 {
                            Fp::from(a) - Fp::from(PAD)
                        } else {
                            Fp::ZERO
                        };
                        let inv = Option::<Fp>::from(inv.invert()).unwrap_or(Fp::ZERO);
                        region.assign_advice(|| "p_inv", config.p_inv[t], i, || Value::known(inv))?;
                        region.assign_advice(|| "p_start", config.p_start[t], i, || Value::known(Fp::from(a)))?;
                        region.assign_advice(|| "p_mid", config.p_mid[t], i, || Value::known(Fp::from(b)))?;
                        region.assign_advice(|| "p_last", config.p_last[t], i, || Value::known(Fp::from(c)))?;
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
                    // Sentinel row on the table side, so the next level's
                    // dummy rows have a PAD target for their left lookup.
                    let i = cap;
                    config.q_level_tbl[t].enable(&mut region, i)?;
                    region.assign_advice(|| "p_start pad", config.p_start[t], i, || Value::known(Fp::from(PAD)))?;
                    region.assign_advice(|| "p_mid pad", config.p_mid[t], i, || Value::known(Fp::from(PAD)))?;
                    region.assign_advice(|| "p_last pad", config.p_last[t], i, || Value::known(Fp::from(PAD)))?;
                    region.assign_advice(|| "p_dummy pad", config.p_dummy[t], i, || Value::known(Fp::ONE))?;
                    region.assign_advice(|| "p_inv pad", config.p_inv[t], i, || Value::known(Fp::ZERO))?;
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

    #[test]
    fn triangle_count_matches_bruteforce() {
        let edges = tiny_graph();
        let (lvls, count) = enumerate_paths(&edges, 3, true);
        // each undirected triangle yields 6 directed closed walks
        assert_eq!(count, 12);
        let capacities = vec![lvls[0].len() + 3, lvls[1].len() + 5];
        let circuit = PoneBaselineCircuit {
            edges,
            levels: 3,
            cyclic: true,
            capacities,
            ..Default::default()
        };
        let prover = MockProver::run(9, &circuit, vec![vec![Fp::from(count)]]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn path_count_matches_bruteforce() {
        let edges = tiny_graph();
        let (lvls, count) = enumerate_paths(&edges, 3, false);
        let capacities = vec![lvls[0].len(), lvls[1].len()];
        let circuit = PoneBaselineCircuit {
            edges,
            levels: 3,
            cyclic: false,
            capacities,
            ..Default::default()
        };
        let prover = MockProver::run(9, &circuit, vec![vec![Fp::from(count)]]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn four_cycle_count_matches_bruteforce() {
        let edges = tiny_graph();
        let (lvls, count) = enumerate_paths(&edges, 4, true);
        let capacities = vec![lvls[0].len(), lvls[1].len(), lvls[2].len() + 2];
        let circuit = PoneBaselineCircuit {
            edges,
            levels: 4,
            cyclic: true,
            capacities,
            ..Default::default()
        };
        let prover = MockProver::run(10, &circuit, vec![vec![Fp::from(count)]]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
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
        };
        let prover = MockProver::run(9, &circuit, vec![vec![Fp::from(count + 1)]]).unwrap();
        assert!(prover.verify().is_err());
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
