use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::q3_obj;
use crate::inline_bind::{assign_bind_cells, configure_bind, tie_columns, BindConfig};

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
        // The ten committed input columns are equality-enabled there, which is
        // what `tie_columns` needs below.
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
        // Baseline synthesize, verbatim apart from also collecting the cells of
        // the committed input columns:
        let chip = q3_obj::TestChip::construct(config.0);
        let (out, witness_cells) = chip.assign_with_input_cells(
            &mut layouter,
            self.customer.clone(),
            self.orders.clone(),
            self.lineitem.clone(),
            self.condition,
        )?;
        chip.expose_public(&mut layouter, out, 0)?;
        // Binding addition. The Horner gates evaluate the binding's own data
        // columns at x, and `tie_columns` then copy-constrains those columns
        // cell-by-cell to the very cells the query witnesses -- customer
        // [c_mktsegment, c_custkey], orders [o_orderdate, o_shippriority,
        // o_custkey, o_orderkey] and lineitem [l_orderkey, l_extendedprice,
        // l_discount, l_shipdate], in `TpchInput::columns` order and at the
        // committed row positions. Untied, the binding would establish only
        // that SOME columns match Commit(D); tied, the exposed evaluation is an
        // evaluation of THIS proof's witness, which is what Appendix A requires
        // and what discharges Threat (i).
        let bind_cells = assign_bind_cells(&mut layouter, &config.1, &self.columns, self.x)?;
        tie_columns(&mut layouter, &bind_cells, &witness_cells)?;
        Ok(())
    }
}

#[cfg(test)]
mod q3_bound_to_witness_tests {
    use super::*;
    use crate::bench_queries::{
        date_to_timestamp, scale_by_1000, string_to_u64, TpchInput,
    };
    use crate::data::data_processing as dp;
    use crate::inline_bind::bind_instance;
    use halo2_proofs::dev::{MockProver, VerifyFailure};
    use std::collections::HashSet;

    /// The TPC-H degree the Q3 benchmarks use.
    const K: u32 = 16;

    /// A truncated slice of the real tables, with EXACTLY the projections
    /// `bench_queries::tpch_inputs("q3", ..)` applies -- the same attributes in
    /// the same order, so `TpchInput::columns` below is the real publication and
    /// not a restatement of it. Truncated because the full 60K-row lineitem
    /// leaves no headroom for three MockProver runs.
    fn slice() -> TpchInput {
        const N_CUST: usize = 300;
        const N_ORD: usize = 2000;
        const N_LINE: usize = 8000;

        let customer: Vec<Vec<u64>> =
            dp::customer_read_records_from_file(&crate::paths::data_file("customer.tbl"))
                .expect("customer.tbl")
                .iter()
                .take(N_CUST)
                .map(|r| vec![string_to_u64(&r.c_mktsegment), r.c_custkey])
                .collect();
        let orders: Vec<Vec<u64>> =
            dp::orders_read_records_from_file(&crate::paths::data_file("orders.tbl"))
                .expect("orders.tbl")
                .iter()
                .take(N_ORD)
                .map(|r| {
                    vec![
                        date_to_timestamp(&r.o_orderdate),
                        r.o_shippriority,
                        r.o_custkey,
                        r.o_orderkey,
                    ]
                })
                .collect();
        let lineitem: Vec<Vec<u64>> =
            dp::lineitem_read_records_from_file(&crate::paths::data_file("lineitem.tbl"))
                .expect("lineitem.tbl")
                .iter()
                .take(N_LINE)
                .map(|r| {
                    vec![
                        r.l_orderkey,
                        scale_by_1000(r.l_extendedprice),
                        scale_by_1000(r.l_discount),
                        date_to_timestamp(&r.l_shipdate),
                    ]
                })
                .collect();

        assert!(
            !customer.is_empty() && !orders.is_empty() && !lineitem.is_empty(),
            "dataset files not found under {}",
            crate::paths::data_file("customer.tbl")
        );

        TpchInput::Q3 {
            customer,
            orders,
            lineitem,
            condition: [string_to_u64("HOUSEHOLD"), date_to_timestamp("1995-03-25")],
        }
    }

    fn parts(input: &TpchInput) -> (Vec<Vec<u64>>, Vec<Vec<u64>>, Vec<Vec<u64>>, [u64; 2]) {
        match input {
            TpchInput::Q3 {
                customer,
                orders,
                lineitem,
                condition,
            } => (
                customer.clone(),
                orders.clone(),
                lineitem.clone(),
                *condition,
            ),
            _ => unreachable!("slice() builds a Q3 input"),
        }
    }

    fn circuit(input: &TpchInput, columns: Vec<Vec<u64>>, x: Fp) -> BoundQ3 {
        let (customer, orders, lineitem, condition) = parts(input);
        BoundQ3 {
            customer,
            orders,
            lineitem,
            condition,
            columns,
            x,
        }
    }

    /// The slice must actually reduce, or the tie would be tested over a query
    /// whose clean instance is empty.
    fn clean_orders(input: &TpchInput) -> usize {
        let (customer, orders, lineitem, condition) = parts(input);
        let ckeys: HashSet<u64> = customer
            .iter()
            .filter(|c| c[0] == condition[0])
            .map(|c| c[1])
            .collect();
        let lkeys: HashSet<u64> = lineitem
            .iter()
            .filter(|l| l[3] > condition[1])
            .map(|l| l[0])
            .collect();
        orders
            .iter()
            .filter(|o| o[0] < condition[1] && ckeys.contains(&o[2]) && lkeys.contains(&o[3]))
            .count()
    }

    /// The honest prover: the committed columns ARE the three relations the
    /// query witnesses. This is also the ENCODING check -- the chip stores every
    /// committed attribute verbatim (`F::from(customer[i][j])` and friends) at
    /// the committed row position, so the copy constraints hold on honest input.
    /// Were any column shifted, packed or reordered on its way into the witness,
    /// this test would fail loudly with equality-constraint errors.
    #[test]
    fn binding_accepts_the_committed_witness() {
        let input = slice();
        let x = Fp::from(7u64);
        let cols = input.columns();
        assert_eq!(cols.len(), NC, "Q3 publishes {} columns", NC);
        assert!(
            clean_orders(&input) > 0,
            "the slice reduces to nothing, so the test is weak"
        );

        let c = circuit(&input, cols.clone(), x);
        let inst = vec![vec![Fp::from(1u64)], bind_instance(&cols, x)];
        MockProver::run(K, &c, inst).unwrap().assert_satisfied();
    }

    /// THE POINT OF THE WHOLE EXERCISE. The prover answers the query over one
    /// lineitem table while claiming the evaluation of a DIFFERENT one -- the
    /// substitution Threat (i) is about. With the binding's data columns a
    /// private copy, both were unrelated and this verified: the binding proved
    /// that some columns matched Commit(D), never that the query's witness did.
    ///
    /// The perturbation is a row swap, so every column keeps its multiset and
    /// the public output stays the honest 1: a shuffle argument would miss it,
    /// and only the row-by-row tie can reject.
    #[test]
    fn binding_rejects_a_witness_that_is_not_the_committed_data() {
        let honest = slice();
        let x = Fp::from(7u64);
        let honest_cols = honest.columns();

        let (customer, orders, mut lineitem, condition) = parts(&honest);
        let j = (1..lineitem.len())
            .find(|&j| lineitem[j] != lineitem[0])
            .expect("every lineitem row is identical, so no swap is visible");
        lineitem.swap(0, j);
        let tampered = TpchInput::Q3 {
            customer,
            orders,
            lineitem,
            condition,
        };

        // Control: the SAME swapped table, committed honestly, verifies. So
        // nothing about the swapped witness is inherently unsatisfiable and the
        // rejection below can only come from the mismatch with the commitment.
        let swapped_cols = tampered.columns();
        assert_ne!(
            swapped_cols, honest_cols,
            "the swap left the published columns unchanged"
        );
        let control = circuit(&tampered, swapped_cols.clone(), x);
        MockProver::run(
            K,
            &control,
            vec![vec![Fp::from(1u64)], bind_instance(&swapped_cols, x)],
        )
        .unwrap()
        .assert_satisfied();

        let c = circuit(&tampered, honest_cols.clone(), x);
        let inst = vec![vec![Fp::from(1u64)], bind_instance(&honest_cols, x)];
        let failures = MockProver::run(K, &c, inst)
            .unwrap()
            .verify()
            .expect_err("a witness differing from the committed columns must not verify");
        // And it must reject through the TIE, not through some incidental gate:
        // the failing cells are the lineitem input columns at the two swapped
        // rows and the binding's data columns opposite them, so BOTH regions
        // have to appear among the permutation failures.
        let perm: Vec<String> = failures
            .iter()
            .filter(|f| matches!(f, VerifyFailure::Permutation { .. }))
            .map(|f| format!("{:?}", f))
            .collect();
        assert!(
            perm.iter().any(|f| f.contains("'witness'"))
                && perm.iter().any(|f| f.contains("'inlined witness binding'")),
            "the circuit rejected, but not through the copy constraints that tie \
             the binding to the query's witness: {:?}",
            failures
        );
    }
}
