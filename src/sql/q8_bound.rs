

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::q8_obj;
use crate::inline_bind::{assign_bind, configure_bind, BindConfig};

/// Committed input columns:
/// region 2 + nation 3 + customer 2 + orders 3 + part 2 + supplier 2 + lineitem 5.
pub const NC: usize = 19;

pub struct BoundQ8 {
    // -- identical to q8_obj::MyCircuit --
    pub region: Vec<Vec<u64>>,
    pub nation: Vec<Vec<u64>>,
    pub customer: Vec<Vec<u64>>,
    pub orders: Vec<Vec<u64>>,
    pub part: Vec<Vec<u64>>,
    pub supplier: Vec<Vec<u64>>,
    pub lineitem: Vec<Vec<u64>>,
    pub cond_nation_hash: u64,
    pub const_region_name_hash: u64,
    pub const_part_type_hash: u64,
    // -- binding additions --
    pub columns: Vec<Vec<u64>>,
    pub x: Fp,
}

impl Circuit<Fp> for BoundQ8 {
    type Config = (q8_obj::TestCircuitConfig<Fp>, BindConfig);
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            region: Vec::new(),
            nation: Vec::new(),
            customer: Vec::new(),
            orders: Vec::new(),
            part: Vec::new(),
            supplier: Vec::new(),
            lineitem: Vec::new(),
            cond_nation_hash: 0,
            const_region_name_hash: 0,
            const_part_type_hash: 0,
            columns: Vec::new(),
            x: self.x,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let q = q8_obj::TestChip::<Fp>::configure(meta);
        let b = configure_bind(meta, NC);
        (q, b)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let chip = q8_obj::TestChip::construct(config.0);
        let out = chip.assign(
            &mut layouter,
            self.region.clone(),
            self.nation.clone(),
            self.customer.clone(),
            self.orders.clone(),
            self.part.clone(),
            self.supplier.clone(),
            self.lineitem.clone(),
            self.cond_nation_hash,
            self.const_region_name_hash,
            self.const_part_type_hash,
        )?;
        chip.expose_public(&mut layouter, out, 0)?;
        assign_bind(&mut layouter, &config.1, &self.columns, self.x)?;
        Ok(())
    }
}
