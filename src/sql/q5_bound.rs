use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::q5_obj;
use crate::inline_bind::{assign_bind_cells, configure_bind, tie_columns, BindConfig};

/// Committed input columns:
/// customer 2 + orders 3 + lineitem 4 + supplier 2 + nation 3 + region 2.
pub const NC: usize = 16;

/// The five committed columns Q5 does NOT witness verbatim, with the relation
/// whose length they run over. `q5_obj::assign_with_input_cells` writes
/// `value + 1` into these, because 0 is the "no match" sentinel of the three
/// tuple lookups and TPC-H has a real nationkey 0 and a real regionkey 0.
///
/// ```text
///    1  c_nationkey   |customer|
///   10  s_nationkey   |supplier|
///   11  n_nationkey   |nation|
///   13  n_regionkey   |nation|
///   14  r_regionkey   |region|
/// ```
///
/// The publication (`bench_queries::TpchInput::columns` for Q5) commits the
/// SAME shifted encoding, so all sixteen columns are tied by direct copy
/// constraints; anything deriving Q5's public evaluations must build them from
/// `columns()`, never from a raw transpose of the tables.
const SHIFTED_COLUMNS: [usize; 5] = [1, 10, 11, 13, 14];

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
        // The FULL baseline constraint system; it already equality-enables the
        // sixteen committed input columns.
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
        let (out, input_cells) = chip.assign_with_input_cells(
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

        // The binding evaluates its own columns at x, then those columns are
        // copy-constrained cell-by-cell to the input cells THIS proof witnesses.
        // Without the tie the binding would establish only that some columns
        // match Commit(D); with it, the exposed evaluation is an evaluation of
        // this proof's witness, which is what Appendix A requires and what
        // discharges Threat (i).
        let bind_cells = assign_bind_cells(&mut layouter, &config.1, &self.columns, self.x)?;

        // The publication now carries Q5's shifted key encoding
        // (`TpchInput::columns`), so every committed column is tied to the
        // query's own cell by a direct copy constraint, exactly like the graph
        // circuits. The former `raw = shifted - 1` adapter gate is gone.
        tie_columns(&mut layouter, &bind_cells, &input_cells)?;
        Ok(())
    }
}

#[cfg(test)]
mod q5_bound_to_witness_tests {
    use super::*;
    use crate::bench_queries::{tpch_inputs, Privacy, TpchInput};
    use crate::inline_bind::bind_instance;
    use halo2_proofs::dev::MockProver;

    const K: u32 = 16;

    /// Real TPC-H tables, truncated so three MockProver runs fit k=16. The
    /// dimension tables stay whole because Q5's tuple lookups need every
    /// referenced key present.
    fn input(swap_lineitem: bool) -> TpchInput {
        let TpchInput::Q5 {
            customer,
            orders,
            mut lineitem,
            supplier,
            nation,
            region,
            europe_hash,
            start_ts,
            end_ts,
            ..
        } = tpch_inputs("q5", Privacy::Rjs)
        else {
            unreachable!("tpch_inputs(\"q5\") returns Q5")
        };
        let orders: Vec<Vec<u64>> = orders.into_iter().take(2000).collect();
        lineitem.truncate(8000);
        if swap_lineitem {
            lineitem.swap(0, 1);
        }
        TpchInput::Q5 {
            customer,
            orders,
            lineitem,
            supplier,
            nation,
            region,
            europe_hash,
            start_ts,
            end_ts,
            nr_pad_extra: 0,
            co_pad_extra: 0,
            ls_pad_extra: 0,
        }
    }

    fn circuit(inp: &TpchInput, columns: Vec<Vec<u64>>, x: Fp) -> BoundQ5 {
        let TpchInput::Q5 {
            customer,
            orders,
            lineitem,
            supplier,
            nation,
            region,
            europe_hash,
            start_ts,
            end_ts,
            ..
        } = inp
        else {
            unreachable!()
        };
        BoundQ5 {
            customer: customer.clone(),
            orders: orders.clone(),
            lineitem: lineitem.clone(),
            supplier: supplier.clone(),
            nation: nation.clone(),
            region: region.clone(),
            europe_hash: *europe_hash,
            start_ts: *start_ts,
            end_ts: *end_ts,
            nr_pad_extra: 0,
            co_pad_extra: 0,
            ls_pad_extra: 0,
            columns,
            x,
        }
    }

    /// The publication must shift exactly [`SHIFTED_COLUMNS`] by one and carry
    /// every other column verbatim. This is the drift the tied full run hit:
    /// a raw transpose of the tables fed the public evaluations while the
    /// circuit bound the shifted encoding, and `column_indices` masked it
    /// because Q8/Q9 publish those same raw keys.
    #[test]
    fn publication_shifts_exactly_the_key_columns() {
        let inp = input(false);
        let cols = inp.columns();
        let TpchInput::Q5 {
            customer,
            orders,
            lineitem,
            supplier,
            nation,
            region,
            ..
        } = &inp
        else {
            unreachable!()
        };
        let mut raw: Vec<Vec<u64>> = Vec::new();
        for t in [customer, orders, lineitem, supplier, nation, region] {
            for j in 0..t[0].len() {
                raw.push(t.iter().map(|r| r[j]).collect());
            }
        }
        assert_eq!(raw.len(), NC);
        for j in 0..NC {
            if SHIFTED_COLUMNS.contains(&j) {
                let shifted: Vec<u64> = raw[j].iter().map(|v| v + 1).collect();
                assert_eq!(
                    cols[j], shifted,
                    "column {j} must be published with the +1 key encoding"
                );
            } else {
                assert_eq!(cols[j], raw[j], "column {j} must be published verbatim");
            }
        }
    }

    /// The committed columns come from `TpchInput::columns()` itself, so the
    /// five shifted key columns are never restated here: if the publication's
    /// encoding and the circuit's ever diverge again, this test fails.
    #[test]
    fn q5_binding_accepts_the_committed_witness() {
        let inp = input(false);
        let x = Fp::from(7u64);
        let cols = inp.columns();
        let inst = vec![vec![Fp::from(1u64)], bind_instance(&cols, x)];
        MockProver::run(K, &circuit(&inp, cols, x), inst)
            .unwrap()
            .assert_satisfied();
    }

    /// Answering over a swapped lineitem while committing the honest columns.
    /// The swap preserves every column's multiset and leaves the public output
    /// at the constant 1 -- and Q5's output is ALWAYS 1, so the instance can
    /// never reject a data substitution. Only the row-by-row tie can.
    #[test]
    fn q5_binding_rejects_a_witness_that_is_not_the_committed_data() {
        let honest = input(false).columns();
        let x = Fp::from(7u64);
        let tampered = input(true);

        // Control: the swapped witness committed honestly to ITSELF verifies,
        // so it is not inherently unsatisfiable and only the tie can reject.
        let own = tampered.columns();
        let inst_ok = vec![vec![Fp::from(1u64)], bind_instance(&own, x)];
        MockProver::run(K, &circuit(&tampered, own, x), inst_ok)
            .unwrap()
            .assert_satisfied();

        let inst = vec![vec![Fp::from(1u64)], bind_instance(&honest, x)];
        let verdict = MockProver::run(K, &circuit(&tampered, honest, x), inst)
            .unwrap()
            .verify();
        assert!(
            verdict.is_err(),
            "a witness differing from the committed columns must not verify"
        );
    }
}
