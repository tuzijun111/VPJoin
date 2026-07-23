//! **Bound Q18** = baseline Q18 (see [`super::q18_obj`], file `q18_obj.rs`)
//! plus the inlined witness-binding check of Appendix A.
//!
//! See `q3_bound.rs` for the map of the full pipeline; this file must differ
//! from the baseline ONLY by the binding columns/gates (guard test:
//! `inline_bind::tests::bound_circuits_are_supersets_of_their_base`).

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::q18_obj;
use crate::inline_bind::{assign_bind, configure_bind, BindConfig};

/// Committed input columns: customer 2 + orders 4 + lineitem 2.
pub const NC: usize = 8;

pub struct BoundQ18 {
    // -- identical to q18_obj::MyCircuit --
    pub customer: Vec<Vec<u64>>,
    pub orders: Vec<Vec<u64>>,
    pub lineitem: Vec<Vec<u64>>,
    pub threshold: u64,
    // -- binding additions --
    pub columns: Vec<Vec<u64>>,
    pub x: Fp,
}

impl Circuit<Fp> for BoundQ18 {
    type Config = (q18_obj::Q18Config<Fp>, BindConfig);
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            customer: Vec::new(),
            orders: Vec::new(),
            lineitem: Vec::new(),
            threshold: 0,
            columns: Vec::new(),
            x: self.x,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let q = q18_obj::Q18Chip::<Fp>::configure(meta);
        let b = configure_bind(meta, NC);
        (q, b)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let chip = q18_obj::Q18Chip::construct(config.0);
        let out = chip.assign(
            &mut layouter,
            self.customer.clone(),
            self.orders.clone(),
            self.lineitem.clone(),
            self.threshold,
        )?;
        chip.expose_public(&mut layouter, out, 0)?;
        assign_bind(&mut layouter, &config.1, &self.columns, self.x)?;
        Ok(())
    }
}
