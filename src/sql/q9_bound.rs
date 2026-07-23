
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::q9_obj;
use crate::inline_bind::{assign_bind, configure_bind, BindConfig};

/// Committed input columns:
/// part 2 + supplier 2 + nation 2 + orders 2 + partsupp 2 (packed key, cost)
/// + lineitem 6.
pub const NC: usize = 16;

pub struct BoundQ9 {
    // -- identical to q9_obj::MyCircuit --
    pub part: Vec<Vec<u64>>,
    pub supplier: Vec<Vec<u64>>,
    pub nation: Vec<Vec<u64>>,
    pub orders: Vec<Vec<u64>>,
    pub partsupp: Vec<Vec<u64>>,
    pub lineitem: Vec<Vec<u64>>,
    pub cond_hash: u64,
    // -- binding additions --
    pub columns: Vec<Vec<u64>>,
    pub x: Fp,
}

impl Circuit<Fp> for BoundQ9 {
    type Config = (q9_obj::TestCircuitConfig<Fp>, BindConfig);
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            part: Vec::new(),
            supplier: Vec::new(),
            nation: Vec::new(),
            orders: Vec::new(),
            partsupp: Vec::new(),
            lineitem: Vec::new(),
            cond_hash: 0,
            columns: Vec::new(),
            x: self.x,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let q = q9_obj::TestChip::<Fp>::configure(meta);
        let b = configure_bind(meta, NC);
        (q, b)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let chip = q9_obj::TestChip::construct(config.0);
        let out = chip.assign(
            &mut layouter,
            self.part.clone(),
            self.supplier.clone(),
            self.nation.clone(),
            self.orders.clone(),
            self.partsupp.clone(),
            self.lineitem.clone(),
            self.cond_hash,
        )?;
        chip.expose_public(&mut layouter, out, 0)?;
        assign_bind(&mut layouter, &config.1, &self.columns, self.x)?;
        Ok(())
    }
}
