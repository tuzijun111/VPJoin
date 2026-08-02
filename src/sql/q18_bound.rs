

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::q18_obj;
use crate::inline_bind::{assign_bind_cells, configure_bind, tie_columns, BindConfig};

/// Committed input columns: customer 2 + orders 4 + lineitem 2.
///
/// In `bench_queries::TpchInput::columns()` order, which for Q18 is
/// `transpose(&[customer, orders, lineitem])`:
///
/// | j | column                          | chip cell            |
/// |---|---------------------------------|----------------------|
/// | 0 | `string_to_u64(c_name)`         | `config.customer[0]` |
/// | 1 | `c_custkey`                     | `config.customer[1]` |
/// | 2 | `o_orderkey`                    | `config.orders[0]`   |
/// | 3 | `o_custkey`                     | `config.orders[1]`   |
/// | 4 | `date_to_timestamp(o_orderdate)`| `config.orders[2]`   |
/// | 5 | `scale_by_1000(o_totalprice)`   | `config.orders[3]`   |
/// | 6 | `l_orderkey`                    | `config.lineitem[0]` |
/// | 7 | `l_quantity`                    | `config.lineitem[1]` |
///
/// The projections are host side (`string_to_u64`, `date_to_timestamp`,
/// `scale_by_1000`) and happen BEFORE the circuit sees the table, so what the
/// commitment publishes and what the chip witnesses are the same u64. The chip
/// applies no further encoding to these columns.
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
        // The height the binding lays out: the longest committed column, which
        // is what `assign_bind_cells` zero-extends every column to. The chip
        // zero-extends its three relations to the same height so the tie is
        // row-for-row over the whole binding.
        let bind_rows = if self.columns.len() == NC {
            self.columns
                .iter()
                .map(|c| c.len())
                .max()
                .unwrap_or(0)
                .max(1)
        } else {
            // No witness to bind (`without_witnesses`, cost probes).
            0
        };
        let (out, witness_cells) = chip.assign_with_input_cells(
            &mut layouter,
            self.customer.clone(),
            self.orders.clone(),
            self.lineitem.clone(),
            self.threshold,
            bind_rows,
        )?;
        chip.expose_public(&mut layouter, out, 0)?;
        // The binding evaluates its own columns at x, then those columns are
        // copy-constrained cell-by-cell to the base-table cells THIS proof
        // witnesses. Without the tie the binding would establish only that some
        // columns match Commit(D); with it, the exposed evaluation is an
        // evaluation of this proof's witness, which is what Appendix A requires
        // and what discharges Threat (i).
        let bind_cells = assign_bind_cells(&mut layouter, &config.1, &self.columns, self.x)?;
        tie_columns(&mut layouter, &bind_cells, &witness_cells)?;
        Ok(())
    }
}

#[cfg(test)]
mod q18_bound_to_witness_tests {
    use super::*;
    use crate::inline_bind::bind_instance;
    use halo2_proofs::dev::MockProver;

    /// TPC-H's own degree for Q18.
    const K: u32 = 16;
    const THRESHOLD: u64 = 300;

    // A three-customer / three-order / six-lineitem instance. Every lineitem
    // names a real order and every order a real customer, so the semijoin
    // reduction leaves every tuple clean and the Cardinality Preservation Check
    // balances. Order 10 sums to 350 and order 30 to 450, both above the
    // HAVING parameter, so the result table is not empty and the ORDER BY
    // ladder has two real rows to compare; order 20 sums to 190 and is
    // filtered, so the HAVING branch is exercised in both directions.
    fn customer() -> Vec<Vec<u64>> {
        vec![vec![11, 1], vec![22, 2], vec![33, 3]]
    }
    fn orders() -> Vec<Vec<u64>> {
        vec![
            vec![10, 1, 5000, 700],
            vec![20, 2, 5001, 900],
            vec![30, 3, 5002, 800],
        ]
    }
    fn lineitem() -> Vec<Vec<u64>> {
        vec![
            vec![10, 200],
            vec![10, 150],
            vec![20, 100],
            vec![20, 90],
            vec![30, 400],
            vec![30, 50],
        ]
    }

    /// The committed columns, exactly as `bench_queries::TpchInput::columns()`
    /// builds them for Q18: `transpose(&[customer, orders, lineitem])`, i.e.
    /// each table's attributes in projection order, one column per attribute.
    /// The three relations have different heights, which is why the columns do
    /// too; the binding zero-extends them to the tallest.
    fn cols(c: &[Vec<u64>], o: &[Vec<u64>], l: &[Vec<u64>]) -> Vec<Vec<u64>> {
        let mut out = Vec::with_capacity(NC);
        for t in [c, o, l] {
            let w = t.first().map(|r| r.len()).unwrap_or(0);
            for j in 0..w {
                out.push(t.iter().map(|r| r[j]).collect());
            }
        }
        assert_eq!(out.len(), NC);
        out
    }

    fn circuit(
        c: Vec<Vec<u64>>,
        o: Vec<Vec<u64>>,
        l: Vec<Vec<u64>>,
        columns: Vec<Vec<u64>>,
        x: Fp,
    ) -> BoundQ18 {
        BoundQ18 {
            customer: c,
            orders: o,
            lineitem: l,
            threshold: THRESHOLD,
            columns,
            x,
        }
    }

    /// The honest prover: the committed columns ARE the circuit's base tables.
    ///
    /// This also pins the ENCODING. Q18 stores every one of the eight committed
    /// attributes verbatim -- no shift, no packing, no hash -- so the honest
    /// prover's tie is between equal values. Had any column been transformed on
    /// the way into the witness, this test would fail loudly with an equality
    /// error, which is how the graph round caught gq1/gq3's `+ SHIFT_ID`.
    #[test]
    fn binding_accepts_the_committed_witness() {
        let (c, o, l) = (customer(), orders(), lineitem());
        let x = Fp::from(7u64);
        let cols = cols(&c, &o, &l);
        let circuit = circuit(c, o, l, cols.clone(), x);
        let inst = vec![vec![Fp::from(1u64)], bind_instance(&cols, x)];
        MockProver::run(K, &circuit, inst).unwrap().assert_satisfied();
    }

    /// THE POINT OF THE WHOLE EXERCISE. The prover answers the query over one
    /// customer table while claiming the evaluation of a DIFFERENT one --
    /// exactly the substitution Threat (i) is about. With the binding's data
    /// columns merely a private copy, the two were unrelated and this verified:
    /// the binding proved that some columns matched Commit(D), never that the
    /// query's witness did.
    #[test]
    fn binding_rejects_a_witness_that_is_not_the_committed_data() {
        let (c, o, l) = (customer(), orders(), lineitem());
        let x = Fp::from(7u64);
        let honest = cols(&c, &o, &l);

        // Same multiset of customer rows, two of them swapped. Every part of
        // Q18 that reads customer does so through a lookup or a permutation, so
        // the answer, the result table and both One-Pass conditions are
        // untouched: only a row-by-row tie can see this.
        let mut tampered = c.clone();
        tampered.swap(0, 2);
        assert_ne!(tampered, c);

        // Control: the SAME swapped table, committed honestly, verifies. So
        // nothing about the swapped witness is inherently unsatisfiable, and
        // the rejection below can only come from the mismatch with the
        // commitment.
        let swapped_cols = cols(&tampered, &o, &l);
        let control = circuit(
            tampered.clone(),
            o.clone(),
            l.clone(),
            swapped_cols.clone(),
            x,
        );
        MockProver::run(
            K,
            &control,
            vec![vec![Fp::from(1u64)], bind_instance(&swapped_cols, x)],
        )
        .unwrap()
        .assert_satisfied();

        let circuit = circuit(tampered, o, l, honest.clone(), x);
        let inst = vec![vec![Fp::from(1u64)], bind_instance(&honest, x)];
        let failures = MockProver::run(K, &circuit, inst)
            .unwrap()
            .verify()
            .expect_err("a witness differing from the committed columns must not verify");
        // And it must reject through the TIE, not through some incidental gate:
        // the failing cells are the customer columns at the two swapped rows and
        // the binding's data columns opposite them.
        assert!(
            failures
                .iter()
                .any(|f| format!("{:?}", f).contains("Equality constraint not satisfied")),
            "the circuit rejected, but not through the copy constraints that tie \
             the binding to the query's witness: {:?}",
            failures
        );
    }
}
