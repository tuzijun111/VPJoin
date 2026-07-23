//! **Bound Q3** = baseline Q3 (see [`super::q3_obj`], file `q3_obj.rs`) plus
//! the inlined witness-binding check of Appendix A.
//!
//! The full pipeline and where each piece lives:
//!
//!   1. data commitment       -> `crate::column_commit::commit_column_vectors`
//!                               (one hiding Pedersen commitment per input
//!                               column; outside the circuit, published once)
//!   2. proof binding         -> `crate::column_commit::binding_challenge`
//!                               (Fiat-Shamir challenge `x` over the
//!                               commitments AND the query proof; outside)
//!   3. witness check         -> `crate::inline_bind::{configure_bind,
//!                               assign_bind}`: Horner-evaluates each witness
//!                               input column at `x` INSIDE this circuit and
//!                               exposes the evaluations publicly
//!   4. column openings       -> `crate::column_commit::open_column_vectors`
//!                               (IPA openings the verifier checks against
//!                               the circuit's exposed evaluations; outside)
//!
//! This file must differ from the baseline circuit ONLY by the binding
//! columns/gates and the second instance column `[x, v_0..v_{NC-1}]`.
//! The guard test
//! `inline_bind::tests::bound_circuits_are_supersets_of_their_base`
//! enforces that structurally.

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::q3_obj;
use crate::inline_bind::{assign_bind, configure_bind, BindConfig};

/// Committed input columns: customer 2 + orders 4 + lineitem 4.
pub const NC: usize = 10;

pub struct BoundQ3 {
    // -- identical to q3_obj::MyCircuit --
    pub customer: Vec<Vec<u64>>,
    pub orders: Vec<Vec<u64>>,
    pub lineitem: Vec<Vec<u64>>,
    pub condition: [u64; 2],
    // -- binding additions --
    /// The query's input columns, in witness order
    /// (see `bench_queries::TpchInput::columns`).
    pub columns: Vec<Vec<u64>>,
    /// Fiat-Shamir challenge from the commitments and the query proof.
    pub x: Fp,
}

impl Circuit<Fp> for BoundQ3 {
    type Config = (q3_obj::TestCircuitConfig<Fp>, BindConfig);
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            customer: Vec::new(),
            orders: Vec::new(),
            lineitem: Vec::new(),
            condition: Default::default(),
            columns: Vec::new(),
            x: self.x,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        // EXACTLY the baseline circuit's constraints (q3_obj::MyCircuit's
        // configure is a thin call to TestChip::configure) ...
        let q = q3_obj::TestChip::<Fp>::configure(meta);
        // ... plus the binding gates, in the SAME constraint system.
        let b = configure_bind(meta, NC);
        (q, b)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        // Baseline synthesize, verbatim:
        let chip = q3_obj::TestChip::construct(config.0);
        let out = chip.assign(
            &mut layouter,
            self.customer.clone(),
            self.orders.clone(),
            self.lineitem.clone(),
            self.condition,
        )?;
        chip.expose_public(&mut layouter, out, 0)?;
        // Binding addition:
        assign_bind(&mut layouter, &config.1, &self.columns, self.x)?;
        Ok(())
    }
}
