
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::g_sql4_obj;
use crate::data::graph_data_processing::Edge;
use crate::inline_bind::{assign_bind_cells, configure_bind, tie_columns, BindConfig};

/// Committed input columns: Edge(src, dst).
pub const NC: usize = 2;

pub struct BoundGq4 {
    // -- identical to g_sql4_obj::MyCircuit --
    pub edges: Vec<Edge>,
    /// One pad knob, inherited from `g_sql4_obj::MyCircuit`: GQ4 materializes a
    /// single bag read in two column roles.
    pub pad_extra: usize,
    // -- binding additions --
    pub columns: Vec<Vec<u64>>,
    pub x: Fp,
}

impl Circuit<Fp> for BoundGq4 {
    type Config = (g_sql4_obj::Cycle4OrderedConfig<Fp>, BindConfig);
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            edges: Vec::new(),
            pad_extra: self.pad_extra,
            columns: Vec::new(),
            x: self.x,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        // `Cycle4OrderedChip::configure` IS the full baseline configure for GQ4
        // (`MyCircuit::configure` calls nothing else), and it already
        // enable_equality's e_src / e_dst, which is what `tie_columns` needs.
        let q = g_sql4_obj::Cycle4OrderedChip::<Fp>::configure(meta);
        let b = configure_bind(meta, NC);
        (q, b)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let chip = g_sql4_obj::Cycle4OrderedChip::construct(config.0);
        let (out, edge_cells) =
            chip.assign_with_edge_cells(&mut layouter, self.edges.clone(), self.pad_extra)?;
        chip.expose_public(&mut layouter, out, 0)?;
        // The binding evaluates its own columns at x, then those columns are
        // copy-constrained cell-by-cell to the Edge cells THIS proof witnesses.
        // Without the tie the binding would establish only that some columns
        // match Commit(D); with it, the exposed evaluation is an evaluation of
        // this proof's witness, which is what Appendix A requires.
        let bind_cells = assign_bind_cells(&mut layouter, &config.1, &self.columns, self.x)?;
        tie_columns(&mut layouter, &bind_cells, &edge_cells)?;
        Ok(())
    }
}

#[cfg(test)]
mod gq4_bound_to_witness_tests {
    use super::*;
    use crate::bench_queries::count_gq4;
    use crate::inline_bind::bind_instance;
    use halo2_proofs::dev::{MockProver, VerifyFailure};

    /// A 4-cycle-bearing edge list small enough for k = 12.
    /// 1->2->3->4->1 is the cycle a<b<c<d; 1->3->5->1 gives a second one.
    fn edges() -> Vec<Edge> {
        [
            (1u64, 2u64),
            (2, 3),
            (3, 4),
            (4, 1),
            (1, 3),
            (2, 4),
            (3, 5),
            (5, 1),
        ]
        .into_iter()
        .map(|(src, dst)| Edge { src, dst })
        .collect()
    }

    /// The committed columns must carry the SAME encoding the circuit
    /// witnesses. GQ4 stores `src + SHIFT_ID` / `dst + SHIFT_ID` in `e_src` /
    /// `e_dst`, so that node id 0 stays free for the base table's dummy row 0
    /// and for the padded bag rows. A commitment over RAW ids cannot be tied to
    /// that witness by equality, which is why this does not call
    /// `bench_queries::edge_columns`.
    fn cols(e: &[Edge]) -> Vec<Vec<u64>> {
        const SHIFT_ID: u64 = 1;
        vec![
            e.iter().map(|x| x.src + SHIFT_ID).collect(),
            e.iter().map(|x| x.dst + SHIFT_ID).collect(),
        ]
    }

    /// The honest prover: the committed columns ARE the circuit's edges.
    #[test]
    fn gq4_binding_accepts_the_committed_witness() {
        let e = edges();
        let cnt = count_gq4(&e);
        assert!(cnt > 0, "the edge list has no 4-cycle, the test would be weak");
        let x = Fp::from(7u64);
        let c = cols(&e);
        let circuit = BoundGq4 {
            edges: e.clone(),
            pad_extra: 0,
            columns: c.clone(),
            x,
        };
        let inst = vec![vec![Fp::from(cnt)], bind_instance(&c, x)];
        MockProver::run(12, &circuit, inst).unwrap().assert_satisfied();
    }

    /// THE POINT OF THE WHOLE EXERCISE. The prover answers the query over one
    /// edge list while claiming the evaluation of a DIFFERENT one -- exactly the
    /// substitution Threat (i) is about. The two lists are the same multiset,
    /// so the query's own constraints and its COUNT are untouched; only the
    /// row-by-row tie between the binding's data cells and `e_src` / `e_dst`
    /// can see the difference. Without the tie this verified, because the
    /// binding was over a private copy of the columns.
    #[test]
    fn gq4_binding_rejects_a_witness_that_is_not_the_committed_data() {
        let e = edges();
        let x = Fp::from(7u64);
        let honest = cols(&e);

        // Same length, same multiset per column, two rows swapped: a shuffle or
        // multiset argument would miss this; a random-point evaluation does not.
        let mut tampered = e.clone();
        tampered.swap(0, 2);
        assert_eq!(
            count_gq4(&tampered),
            count_gq4(&e),
            "the swap must not change the answer, or the rejection would prove nothing"
        );

        let circuit = BoundGq4 {
            edges: tampered,
            pad_extra: 0,
            columns: honest.clone(),
            x,
        };
        let inst = vec![vec![Fp::from(count_gq4(&e))], bind_instance(&honest, x)];
        let verdict = MockProver::run(12, &circuit, inst).unwrap().verify();
        let failures = verdict.expect_err(
            "a witness differing from the committed columns must not verify",
        );
        // And it must fail FOR THE RIGHT REASON. Every failure is a broken copy
        // constraint of the tie -- on `e_src` / `e_dst` at the two swapped rows
        // and on the binding's data columns opposite them. Nothing else in the
        // circuit objects, which is precisely why the tie is what discharges
        // Threat (i): drop it and this witness verifies.
        assert!(
            failures
                .iter()
                .all(|f| matches!(f, VerifyFailure::Permutation { .. })),
            "expected only broken copy constraints, got {:?}",
            failures
        );
    }
}
