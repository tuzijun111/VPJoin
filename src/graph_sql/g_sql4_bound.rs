
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::g_sql4_obj;
use crate::data::graph_data_processing::Edge;
use crate::inline_bind::{assign_bind, configure_bind, BindConfig};

/// Committed input columns: Edge(src, dst).
pub const NC: usize = 2;

pub struct BoundGq4 {
    // -- identical to g_sql4_obj::MyCircuit --
    pub edges: Vec<Edge>,
    pub bag1_pad_extra: usize,
    pub bag2_pad_extra: usize,
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
            bag1_pad_extra: self.bag1_pad_extra,
            bag2_pad_extra: self.bag2_pad_extra,
            columns: Vec::new(),
            x: self.x,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
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
