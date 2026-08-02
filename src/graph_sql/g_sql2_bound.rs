

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::g_sql2_obj;
use crate::data::graph_data_processing::Edge;
use crate::inline_bind::{assign_bind_cells, configure_bind, tie_columns, BindConfig};

/// Committed input columns: Edge(src, dst).
pub const NC: usize = 2;

pub struct BoundGq2 {
    // -- identical to g_sql2_obj::GraphPath4OrderCircuit --
    pub edges: Vec<Edge>,
    // -- binding additions --
    pub columns: Vec<Vec<u64>>,
    pub x: Fp,
}

impl Circuit<Fp> for BoundGq2 {
    type Config = (g_sql2_obj::GraphPath4OrderConfig<Fp>, BindConfig);
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            edges: Vec::new(),
            columns: Vec::new(),
            x: self.x,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        // The FULL baseline configure -- see module docs.
        let q = g_sql2_obj::configure_path4order_full::<Fp>(meta);
        let b = configure_bind(meta, NC);
        (q, b)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let chip = g_sql2_obj::GraphPath4OrderChip::construct(config.0);
        let (out, edge_cells) = chip.assign_with_edge_cells(&mut layouter, &self.edges)?;
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
mod bound_to_witness_tests {
    use super::*;
    use crate::inline_bind::bind_instance;
    use halo2_proofs::dev::MockProver;

    fn edges() -> Vec<Edge> {
        [(1u64, 2u64), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5), (1, 4)]
            .into_iter()
            .map(|(src, dst)| Edge { src, dst })
            .collect()
    }

    /// The committed columns must carry the SAME encoding the circuit
    /// witnesses. Unlike gq1, `g_sql2_obj` writes RAW `e.src` / `e.dst` into
    /// `r[k]` -- its `SHIFT_ID` is applied only inside the masked Pairwise
    /// Consistency key columns `pw_src` / `pw_dst`, never to the base relation.
    /// So the committed columns here are exactly what
    /// `bench_queries::edge_columns` publishes.
    fn cols(e: &[Edge]) -> Vec<Vec<u64>> {
        vec![
            e.iter().map(|x| x.src).collect(),
            e.iter().map(|x| x.dst).collect(),
        ]
    }

    /// The honest prover: the committed columns ARE the circuit's edges.
    #[test]
    fn gq2_binding_accepts_the_committed_witness() {
        let e = edges();
        let x = Fp::from(7u64);
        let c = cols(&e);
        let circuit = BoundGq2 { edges: e.clone(), columns: c.clone(), x };
        let inst = vec![
            vec![Fp::from(crate::bench_queries::count_gq2(&e))],
            bind_instance(&c, x),
        ];
        MockProver::run(13, &circuit, inst).unwrap().assert_satisfied();
    }

    /// THE POINT OF THE WHOLE EXERCISE. The prover answers the query over one
    /// edge list while claiming the evaluation of a DIFFERENT one -- exactly
    /// the substitution Threat (i) is about. Without the tie both were private
    /// copies and this verified: the binding proved that some columns matched
    /// Commit(D), never that the query's witness did.
    #[test]
    fn gq2_binding_rejects_a_witness_that_is_not_the_committed_data() {
        let e = edges();
        let x = Fp::from(7u64);
        let honest = cols(&e);

        // Same length, same multiset per column, one row swapped: a shuffle or
        // multiset argument would miss this; a random-point evaluation does not.
        let mut tampered = e.clone();
        tampered.swap(0, 2);

        let circuit = BoundGq2 { edges: tampered, columns: honest.clone(), x };
        let inst = vec![
            vec![Fp::from(crate::bench_queries::count_gq2(&e))],
            bind_instance(&honest, x),
        ];
        let verdict = MockProver::run(13, &circuit, inst).unwrap().verify();
        assert!(
            verdict.is_err(),
            "a witness differing from the committed columns must not verify"
        );
    }
}
