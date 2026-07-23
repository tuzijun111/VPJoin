//! **Bound GQ2** = baseline GQ2 (see [`super::g_sql2_obj`], file
//! `g_sql2_obj.rs`) plus the inlined witness-binding check of Appendix A.
//!
//! See `src/sql/q3_bound.rs` for the map of the full pipeline. Like GQ1, this
//! circuit's baseline `configure` adds lookup arguments beyond its chip's
//! (2 of them), so this file calls
//! [`g_sql2_obj::configure_path4order_full`] -- the same function the baseline
//! circuit calls -- rather than the chip's configure alone. Enforced by
//! `inline_bind::tests::bound_circuits_are_supersets_of_their_base`.

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::g_sql2_obj;
use crate::data::graph_data_processing::Edge;
use crate::inline_bind::{assign_bind, configure_bind, BindConfig};

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
        let out = chip.assign(&mut layouter, &self.edges)?;
        chip.expose_public(&mut layouter, out, 0)?;
        assign_bind(&mut layouter, &config.1, &self.columns, self.x)?;
        Ok(())
    }
}
