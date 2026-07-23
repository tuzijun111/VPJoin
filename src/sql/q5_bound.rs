use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::q5_obj;
use crate::inline_bind::{assign_bind, configure_bind, BindConfig};

/// Committed input columns:
/// customer 2 + orders 3 + lineitem 4 + supplier 2 + nation 3 + region 2.
pub const NC: usize = 16;

pub struct BoundQ5 {
    // -- identical to q5_obj::MyCircuit --
    pub customer: Vec<Vec<u64>>,
    pub orders: Vec<Vec<u64>>,
    pub lineitem: Vec<Vec<u64>>,
    pub supplier: Vec<Vec<u64>>,
    pub nation: Vec<Vec<u64>>,
    pub region: Vec<Vec<u64>>,
    pub europe_hash: u64,
    pub start_ts: u64,
    pub end_ts: u64,
    pub nr_pad_extra: usize,
    pub co_pad_extra: usize,
    pub ls_pad_extra: usize,
    // -- binding additions --
    pub columns: Vec<Vec<u64>>,
    pub x: Fp,
}

impl Circuit<Fp> for BoundQ5 {
    type Config = (q5_obj::Q5Config<Fp>, BindConfig);
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            customer: Vec::new(),
            orders: Vec::new(),
            lineitem: Vec::new(),
            supplier: Vec::new(),
            nation: Vec::new(),
            region: Vec::new(),
            europe_hash: 0,
            start_ts: 0,
            end_ts: 0,
            nr_pad_extra: 0,
            co_pad_extra: 0,
            ls_pad_extra: 0,
            columns: Vec::new(),
            x: self.x,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let q = q5_obj::Q5Chip::<Fp>::configure(meta);
        let b = configure_bind(meta, NC);
        (q, b)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let chip = q5_obj::Q5Chip::construct(config.0);
        let out = chip.assign(
            &mut layouter,
            self.customer.clone(),
            self.orders.clone(),
            self.lineitem.clone(),
            self.supplier.clone(),
            self.nation.clone(),
            self.region.clone(),
            self.europe_hash,
            self.start_ts,
            self.end_ts,
            self.nr_pad_extra,
            self.co_pad_extra,
            self.ls_pad_extra,
        )?;
        chip.expose_public(&mut layouter, out, 0)?;
        assign_bind(&mut layouter, &config.1, &self.columns, self.x)?;
        Ok(())
    }
}
