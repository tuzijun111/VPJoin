

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::g_sql3_obj;
use crate::data::graph_data_processing::Edge;
use crate::inline_bind::{assign_bind_cells, configure_bind, tie_columns, BindConfig};

/// Committed input columns: Edge(src, dst).
pub const NC: usize = 2;

pub struct BoundGq3 {
    // -- identical to g_sql3_obj::MyCircuit --
    pub edges: Vec<Edge>,
    pub bag1_pad_extra: usize,
    pub bag2_pad_extra: usize,
    // -- binding additions --
    pub columns: Vec<Vec<u64>>,
    pub x: Fp,
}

impl Circuit<Fp> for BoundGq3 {
    type Config = (g_sql3_obj::TrianglePathCloserConfig<Fp>, BindConfig);
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            edges: Vec::new(),
            bag1_pad_extra: self.bag1_pad_extra,
            bag2_pad_extra: self.bag2_pad_extra,
            columns: Vec::new(),
            x: self.x,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        // The whole baseline circuit lives in the chip's `configure` for gq3
        // (unlike gq1/gq2, which add lookups on top of theirs), so this IS the
        // full baseline constraint system. `out_by_src.in_key` / `.in_val`, the
        // committed Edge columns, already have equality enabled there.
        let q = g_sql3_obj::TrianglePathCloserChip::<Fp>::configure(meta);
        let b = configure_bind(meta, NC);
        (q, b)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let chip = g_sql3_obj::TrianglePathCloserChip::construct(config.0);
        let (out, edge_cells) = chip.assign_with_edge_cells(
            &mut layouter,
            self.edges.clone(),
            self.bag1_pad_extra,
            self.bag2_pad_extra,
        )?;
        chip.expose_public(&mut layouter, out, 0)?;
        // The binding evaluates its own columns at x, then those columns are
        // copy-constrained cell-by-cell to the Edge cells THIS proof witnesses
        // (the OutBySrc view's input columns, shifted by SHIFT_ID). Without the
        // tie the binding would establish only that some columns match
        // Commit(D); with it, the exposed evaluation is an evaluation of this
        // proof's witness, which is what Appendix A requires and what
        // discharges Threat (i).
        let bind_cells = assign_bind_cells(&mut layouter, &config.1, &self.columns, self.x)?;
        tie_columns(&mut layouter, &bind_cells, &edge_cells)?;
        Ok(())
    }
}

#[cfg(test)]
mod gq3_bound_to_witness_tests {
    use super::*;
    use crate::inline_bind::bind_instance;
    use halo2_proofs::dev::MockProver;

    /// A small directed graph with triangles a<b<c, so the count is nonzero and
    /// the clean instance is not empty.
    fn edges() -> Vec<Edge> {
        [
            (1u64, 2u64),
            (2, 3),
            (3, 1),
            (1, 3),
            (2, 4),
            (4, 1),
            (1, 4),
            (3, 4),
            (4, 2),
        ]
        .into_iter()
        .map(|(src, dst)| Edge { src, dst })
        .collect()
    }

    /// The committed columns must carry the SAME encoding the circuit
    /// witnesses. gq3 shifts node ids by SHIFT_ID = 1 (so 0 stays free for the
    /// gadgets' dummy row) before they reach the OutBySrc view, so a commitment
    /// over raw ids could not be tied to the witness by equality. This matches
    /// `bench_queries::edge_columns`, restated locally so the test still says
    /// what the encoding is.
    fn cols(e: &[Edge]) -> Vec<Vec<u64>> {
        const SHIFT_ID: u64 = 1;
        vec![
            e.iter().map(|x| x.src + SHIFT_ID).collect(),
            e.iter().map(|x| x.dst + SHIFT_ID).collect(),
        ]
    }

    const K: u32 = 14;
    const BAG1_PAD: usize = 3;
    const BAG2_PAD: usize = 2;

    /// The honest prover: the committed columns ARE the circuit's edges.
    #[test]
    fn binding_accepts_the_committed_witness() {
        let e = edges();
        let x = Fp::from(7u64);
        let c = cols(&e);
        let cnt = crate::bench_queries::count_gq3(&e);
        assert!(cnt > 0, "the fixture has no triangle, so the test is weak");

        let circuit = BoundGq3 {
            edges: e.clone(),
            bag1_pad_extra: BAG1_PAD,
            bag2_pad_extra: BAG2_PAD,
            columns: c.clone(),
            x,
        };
        let inst = vec![vec![Fp::from(cnt)], bind_instance(&c, x)];
        MockProver::run(K, &circuit, inst).unwrap().assert_satisfied();
    }

    /// THE POINT OF THE WHOLE EXERCISE. The prover answers the query over one
    /// edge list while claiming the evaluation of a DIFFERENT one -- exactly
    /// the substitution Threat (i) is about. With the binding's data columns
    /// merely a private copy, both were unrelated and this verified: the
    /// binding proved that some columns matched Commit(D), never that the
    /// query's witness did.
    #[test]
    fn binding_rejects_a_witness_that_is_not_the_committed_data() {
        let e = edges();
        let x = Fp::from(7u64);
        let honest = cols(&e);

        // Same length, same multiset per column, two rows swapped: a shuffle or
        // multiset argument would miss this; a random-point evaluation does not.
        // The swap also leaves the triangle count untouched, so the public
        // output is the honest one and only the binding can reject.
        let mut tampered = e.clone();
        tampered.swap(0, 2);

        let cnt = crate::bench_queries::count_gq3(&e);
        assert_eq!(
            cnt,
            crate::bench_queries::count_gq3(&tampered),
            "the swap must not change the answer, or the public output alone \
             would reject and the binding would not be under test"
        );

        // Control: the SAME swapped edge list, committed honestly, verifies. So
        // nothing about the swapped witness is inherently unsatisfiable and the
        // rejection below can only come from the mismatch with the commitment.
        let swapped_cols = cols(&tampered);
        let control = BoundGq3 {
            edges: tampered.clone(),
            bag1_pad_extra: BAG1_PAD,
            bag2_pad_extra: BAG2_PAD,
            columns: swapped_cols.clone(),
            x,
        };
        MockProver::run(
            K,
            &control,
            vec![vec![Fp::from(cnt)], bind_instance(&swapped_cols, x)],
        )
        .unwrap()
        .assert_satisfied();

        let circuit = BoundGq3 {
            edges: tampered,
            bag1_pad_extra: BAG1_PAD,
            bag2_pad_extra: BAG2_PAD,
            columns: honest.clone(),
            x,
        };
        let inst = vec![vec![Fp::from(cnt)], bind_instance(&honest, x)];
        let verdict = MockProver::run(K, &circuit, inst).unwrap().verify();
        let failures = verdict
            .expect_err("a witness differing from the committed columns must not verify");
        // And it must reject through the TIE, not through some incidental gate:
        // the failing cells are the OutBySrc input columns at the two swapped
        // rows and the binding's data columns opposite them.
        assert!(
            failures
                .iter()
                .any(|f| format!("{:?}", f).contains("Equality constraint not satisfied")),
            "the circuit rejected, but not through the copy constraints that tie \
             the binding to the query's witness: {:?}",
            failures
        );
    }
}
