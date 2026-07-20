//! In-circuit input binding to the published database commitment (Appendix A).
//!
//! Complements `src/commitment.rs` (the protocol-level, across-proof binding):
//! this module supplies the missing *in-circuit* consistency step -- proving,
//! inside a circuit, that the input columns a query consumes are exactly the
//! committed database tables.  It is additive: no existing query circuit is
//! modified; the binding is realized as a standalone circuit whose cost is
//! measured separately and added to a query's cost (composing the same
//! columns into a query circuit enforces identical constraints at the same
//! per-cell price).
//!
//! Construction. The committed table is placed in *fixed* columns.  Halo2's
//! key generation commits every fixed column with `Blind::default()`, so the
//! per-column IPA commitments inside the verifying key are deterministic
//! functions of the data; the prover publishes exactly these points at setup
//! (recomputed standalone via [`column_commitments`]) and any verifier checks
//! that the published points equal the VK's fixed commitments.  The circuit
//! then constrains, row by row, `advice[j] = fixed[j]` under a selector: the
//! advice columns -- the form in which query circuits consume inputs -- are
//! thereby bound to the committed data.  A cheating prover feeding different
//! inputs must either break the equality gate or present a different VK,
//! which no longer matches the published commitments.
//!
//! Relationship to `src/commitment.rs`: that module binds each *proof* to one
//! published Pedersen commitment of the whole canonical vector; this module
//! binds the *circuit inputs* to per-column commitments reproduced in the VK.
//! Together they realize the full workflow of Appendix A; the two costs are
//! reported separately by their respective benches.

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error, Fixed, Selector,
    },
    poly::{
        commitment::{Blind, Params},
        ipa::commitment::ParamsIPA,
        EvaluationDomain, Rotation,
    },
};
use halo2curves::group::Curve;
use halo2curves::pasta::{vesta, EqAffine, Fp};

/// Maximum number of attributes supported (TPC-H lineitem has 16).
pub const MAX_COLS: usize = 16;

#[derive(Clone, Debug)]
pub struct InputBindingConfig {
    pub fixed: Vec<Column<Fixed>>,
    pub advice: Vec<Column<Advice>>,
    pub q: Selector,
}

/// Binds `rows` (the query-side input, assigned as advice) to the committed
/// table (assigned as fixed columns).
#[derive(Clone, Debug, Default)]
pub struct InputBindingCircuit {
    /// The table, row-major; all rows must have the same length <= MAX_COLS.
    pub rows: Vec<Vec<u64>>,
    pub cols: usize,
}

impl Circuit<Fp> for InputBindingCircuit {
    type Config = InputBindingConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            rows: Vec::new(),
            cols: self.cols,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let fixed: Vec<Column<Fixed>> = (0..MAX_COLS).map(|_| meta.fixed_column()).collect();
        let advice: Vec<Column<Advice>> = (0..MAX_COLS).map(|_| meta.advice_column()).collect();
        let q = meta.selector();

        meta.create_gate("advice input equals committed fixed table", |m| {
            let q = m.query_selector(q);
            (0..MAX_COLS)
                .map(|j| {
                    q.clone()
                        * (m.query_advice(advice[j], Rotation::cur())
                            - m.query_fixed(fixed[j], Rotation::cur()))
                })
                .collect::<Vec<_>>()
        });

        InputBindingConfig { fixed, advice, q }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "bind input to committed table",
            |mut region| {
                for (i, row) in self.rows.iter().enumerate() {
                    config.q.enable(&mut region, i)?;
                    for j in 0..MAX_COLS {
                        let v = row.get(j).copied().unwrap_or(0);
                        region.assign_fixed(
                            || "committed cell",
                            config.fixed[j],
                            i,
                            || Value::known(Fp::from(v)),
                        )?;
                        region.assign_advice(
                            || "input cell",
                            config.advice[j],
                            i,
                            || Value::known(Fp::from(v)),
                        )?;
                    }
                }
                Ok(())
            },
        )
    }
}

/// The per-column commitments the prover publishes at setup: identical, by
/// construction, to the fixed-column commitments inside the verifying key
/// (`vk.fixed_commitments()`), which is what an external verifier checks.
pub fn column_commitments(
    params: &ParamsIPA<vesta::Affine>,
    rows: &[Vec<u64>],
    k: u32,
) -> Vec<EqAffine> {
    let domain: EvaluationDomain<Fp> = EvaluationDomain::new(1, k);
    (0..MAX_COLS)
        .map(|j| {
            let mut poly = domain.empty_lagrange();
            for (i, row) in rows.iter().enumerate() {
                poly[i] = Fp::from(row.get(j).copied().unwrap_or(0));
            }
            params.commit_lagrange(&poly, Blind::default()).to_affine()
        })
        .collect()
}

/// Smallest circuit size accommodating `rows` plus Halo2's blinding rows.
pub fn min_k_rows(rows: usize) -> u32 {
    let mut k = 4u32;
    while (1usize << k) < rows + 8 {
        k += 1;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::plonk::keygen_vk;
    use halo2_proofs::poly::commitment::ParamsProver; // provides ParamsIPA::new for tests

    fn tiny_rows() -> Vec<Vec<u64>> {
        (0..20u64).map(|i| vec![i, i * 2 + 1, 7 * i]).collect()
    }

    #[test]
    fn binding_accepts_honest_input() {
        let circuit = InputBindingCircuit {
            rows: tiny_rows(),
            cols: 3,
        };
        let prover = MockProver::run(6, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn binding_rejects_tampered_input() {
        // Tampering is simulated by making the advice deviate from fixed:
        // the circuit assigns advice == fixed from `rows`, so we model a
        // cheating prover with a variant circuit whose advice differs.
        #[derive(Clone, Default)]
        struct Tampered(InputBindingCircuit);
        impl Circuit<Fp> for Tampered {
            type Config = InputBindingConfig;
            type FloorPlanner = SimpleFloorPlanner;
            fn without_witnesses(&self) -> Self {
                Tampered(self.0.without_witnesses())
            }
            fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
                InputBindingCircuit::configure(meta)
            }
            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fp>,
            ) -> Result<(), Error> {
                layouter.assign_region(
                    || "tampered",
                    |mut region| {
                        for (i, row) in self.0.rows.iter().enumerate() {
                            config.q.enable(&mut region, i)?;
                            for j in 0..MAX_COLS {
                                let v = row.get(j).copied().unwrap_or(0);
                                region.assign_fixed(
                                    || "committed cell",
                                    config.fixed[j],
                                    i,
                                    || Value::known(Fp::from(v)),
                                )?;
                                // one advice cell deviates from the commitment
                                let w = if i == 3 && j == 1 { v + 1 } else { v };
                                region.assign_advice(
                                    || "input cell",
                                    config.advice[j],
                                    i,
                                    || Value::known(Fp::from(w)),
                                )?;
                            }
                        }
                        Ok(())
                    },
                )
            }
        }
        let bad = Tampered(InputBindingCircuit {
            rows: tiny_rows(),
            cols: 3,
        });
        let prover = MockProver::run(6, &bad, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn published_commitments_match_vk() {
        let k = 6;
        let rows = tiny_rows();
        let circuit = InputBindingCircuit {
            rows: rows.clone(),
            cols: 3,
        };
        let params = ParamsIPA::<vesta::Affine>::new(k);
        let vk = keygen_vk(&params, &circuit).unwrap();
        let published = column_commitments(&params, &rows, k);
        assert_eq!(&vk.fixed_commitments()[..MAX_COLS], &published[..]);
    }
}
