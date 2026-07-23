//! **Bound GQ3** = baseline GQ3 (see [`super::g_sql3_obj`], file
//! `g_sql3_obj.rs`) plus the inlined witness-binding check of Appendix A.
//!
//! See `src/sql/q3_bound.rs` for the map of the full pipeline. GQ3's baseline
//! `configure` IS a thin chip call (unlike GQ1/GQ2), so the chip's configure
//! is the correct thing to reuse here. Enforced by
//! `inline_bind::tests::bound_circuits_are_supersets_of_their_base`.
//!
//! The pad knobs size the materialized bags exactly as in the baseline (see
//! `bench_queries::graph_pads` for the DP / RJS / legacy regimes); the bound
//! circuit must be built with the SAME pads as the baseline it is compared to.

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::g_sql3_obj;
use crate::data::graph_data_processing::Edge;
use crate::inline_bind::{assign_bind, configure_bind, BindConfig};

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
        let out = chip.assign(
            &mut layouter,
            self.edges.clone(),
            self.bag1_pad_extra,
            self.bag2_pad_extra,
        )?;
        chip.expose_public(&mut layouter, out, 0)?;
        assign_bind(&mut layouter, &config.1, &self.columns, self.x)?;
        Ok(())
    }
}
