

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
