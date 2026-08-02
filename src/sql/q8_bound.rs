

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::q8_obj;
use crate::inline_bind::{assign_bind_cells, configure_bind, tie_columns, BindConfig};

/// Committed input columns:
/// region 2 + nation 3 + customer 2 + orders 3 + part 2 + supplier 2 + lineitem 5.
///
/// This is exactly the width `bench_queries::TpchInput::columns()` produces for
/// q8, and `q8_obj` returns one cell vector per entry in the same order.
pub const NC: usize = q8_obj::NUM_COMMITTED;

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
        // The nineteen committed columns are the chip's own base-table columns,
        // and `q8_obj::TestChip::configure` already calls `enable_equality` on
        // every one of them, so `tie_columns` below can copy-constrain them.
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
        let (out, witness_cells) = chip.assign_with_input_cells(
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
        // The binding evaluates its own columns at x, then those columns are
        // copy-constrained cell by cell to the base-table cells THIS proof
        // witnesses. Without the tie the binding would establish only that SOME
        // columns match Commit(D), proved alongside the query rather than about
        // it; with it, the exposed evaluation is an evaluation of this proof's
        // witness, which is what Appendix A requires and what discharges
        // Threat (i).
        let bind_cells = assign_bind_cells(&mut layouter, &config.1, &self.columns, self.x)?;
        tie_columns(&mut layouter, &bind_cells, &witness_cells)?;
        Ok(())
    }
}

#[cfg(test)]
mod q8_bound_to_witness_tests {
    use super::*;
    use crate::bench_queries::{scale_by_1000, string_to_u64_trim, year_from_date};
    use crate::data::data_processing as dp;
    use crate::inline_bind::bind_instance;
    use halo2_proofs::dev::MockProver;

    /// The TPC-H degree of the acyclic queries (`bench_queries::degree_for`).
    const K: u32 = 16;

    /// A slice small enough to keep three MockProver runs cheap and large enough
    /// that every relation still contributes: the five small tables are taken
    /// whole (every orders row has to find its customer and every customer its
    /// nation, because both attach lookups run on all rows), orders and lineitem
    /// are truncated. Same shape as `q8_obj::tests::test_cardinality_preservation`.
    const N_ORD: usize = 3000;
    const N_LINE: usize = 8000;

    struct Tables {
        region: Vec<Vec<u64>>,
        nation: Vec<Vec<u64>>,
        customer: Vec<Vec<u64>>,
        orders: Vec<Vec<u64>>,
        part: Vec<Vec<u64>>,
        supplier: Vec<Vec<u64>>,
        lineitem: Vec<Vec<u64>>,
    }

    /// The per-table attribute projections `bench_queries::run_tpch` uses for
    /// q8, restated here so the test states the encoding it is asserting rather
    /// than inheriting it silently.
    fn tables() -> Tables {
        let region = dp::region_read_records_from_cvs(&crate::paths::data_file("region.cvs"))
            .expect("region.cvs")
            .iter()
            .map(|r| vec![r.r_regionkey, string_to_u64_trim(&r.r_name)])
            .collect::<Vec<_>>();
        let nation = dp::nation_read_records_from_file(&crate::paths::data_file("nation.tbl"))
            .expect("nation.tbl")
            .iter()
            .map(|r| {
                vec![
                    r.n_nationkey,
                    r.n_regionkey,
                    string_to_u64_trim(&r.n_name),
                ]
            })
            .collect::<Vec<_>>();
        let customer = dp::customer_read_records_from_file(&crate::paths::data_file("customer.tbl"))
            .expect("customer.tbl")
            .iter()
            .map(|r| vec![r.c_custkey, r.c_nationkey])
            .collect::<Vec<_>>();
        let orders = dp::orders_read_records_from_file(&crate::paths::data_file("orders.tbl"))
            .expect("orders.tbl")
            .iter()
            .take(N_ORD)
            .map(|r| vec![r.o_orderkey, r.o_custkey, year_from_date(&r.o_orderdate)])
            .collect::<Vec<_>>();
        let part = dp::part_read_records_from_file(&crate::paths::data_file("part.tbl"))
            .expect("part.tbl")
            .iter()
            .map(|r| vec![r.p_partkey, string_to_u64_trim(&r.p_type)])
            .collect::<Vec<_>>();
        let supplier = dp::supplier_read_records_from_file(&crate::paths::data_file("supplier.tbl"))
            .expect("supplier.tbl")
            .iter()
            .map(|r| vec![r.s_suppkey, r.s_nationkey])
            .collect::<Vec<_>>();
        let lineitem = dp::lineitem_read_records_from_file(&crate::paths::data_file("lineitem.tbl"))
            .expect("lineitem.tbl")
            .iter()
            .take(N_LINE)
            .map(|r| {
                vec![
                    r.l_orderkey,
                    r.l_partkey,
                    r.l_suppkey,
                    scale_by_1000(r.l_extendedprice),
                    scale_by_1000(r.l_discount),
                ]
            })
            .collect::<Vec<_>>();

        for (name, t) in [
            ("region", &region),
            ("nation", &nation),
            ("customer", &customer),
            ("orders", &orders),
            ("part", &part),
            ("supplier", &supplier),
            ("lineitem", &lineitem),
        ] {
            assert!(!t.is_empty(), "table `{}` loaded as EMPTY", name);
        }

        Tables { region, nation, customer, orders, part, supplier, lineitem }
    }

    /// `bench_queries::transpose` over the q8 tables, in the FIXED order
    /// `run_tpch` uses: region, nation, customer, orders, part, supplier,
    /// lineitem. Column j of the result is committed column j, and its row i is
    /// the i-th row of the corresponding base table -- no shift, no packing, no
    /// hashing on top of the projection, and no reordering. That is the encoding
    /// `q8_obj::assign_with_input_cells` returns cells for.
    fn cols(t: &Tables) -> Vec<Vec<u64>> {
        let mut out: Vec<Vec<u64>> = Vec::new();
        for tbl in [
            &t.region,
            &t.nation,
            &t.customer,
            &t.orders,
            &t.part,
            &t.supplier,
            &t.lineitem,
        ] {
            let width = tbl.first().map(|r| r.len()).unwrap_or(0);
            for j in 0..width {
                out.push(tbl.iter().map(|r| r[j]).collect());
            }
        }
        assert_eq!(out.len(), NC, "the committed column count moved");
        out
    }

    fn circuit(t: &Tables, columns: Vec<Vec<u64>>, x: Fp) -> BoundQ8 {
        BoundQ8 {
            region: t.region.clone(),
            nation: t.nation.clone(),
            customer: t.customer.clone(),
            orders: t.orders.clone(),
            part: t.part.clone(),
            supplier: t.supplier.clone(),
            lineitem: t.lineitem.clone(),
            cond_nation_hash: string_to_u64_trim("EGYPT"),
            const_region_name_hash: string_to_u64_trim("MIDDLE EAST"),
            const_part_type_hash: string_to_u64_trim("PROMO BRUSHED COPPER"),
            columns,
            x,
        }
    }

    /// The honest prover: the committed columns ARE the tables the query circuit
    /// witnesses, so every copy constraint of the tie holds and the whole
    /// circuit is satisfied.
    #[test]
    fn binding_accepts_the_committed_witness() {
        let t = tables();
        let x = Fp::from(7u64);
        let c = cols(&t);

        let inst = vec![vec![Fp::from(1u64)], bind_instance(&c, x)];
        MockProver::run(K, &circuit(&t, c.clone(), x), inst)
            .unwrap()
            .assert_satisfied();
    }

    /// THE POINT OF THE WHOLE EXERCISE. The prover answers the query over one
    /// database while claiming the evaluation of a DIFFERENT one -- exactly the
    /// substitution Threat (i) is about. With the binding's data columns merely a
    /// private copy, the two were unrelated and this verified: the binding proved
    /// that some columns matched Commit(D), never that the query's witness did.
    #[test]
    fn binding_rejects_a_witness_that_is_not_the_committed_data() {
        let honest = tables();
        let x = Fp::from(7u64);
        let honest_cols = cols(&honest);

        // Two part rows that both fail the p_type predicate, swapped: the
        // multiset of every committed column is untouched, the join is untouched
        // and the public output (the constant 1 this family exposes) is
        // untouched, so nothing but the binding can tell the two databases
        // apart. A shuffle or multiset argument would miss this; an evaluation
        // at a random point does not.
        let ptype = string_to_u64_trim("PROMO BRUSHED COPPER");
        let dull: Vec<usize> = honest
            .part
            .iter()
            .enumerate()
            .filter(|(_, r)| r[1] != ptype)
            .map(|(i, _)| i)
            .take(2)
            .collect();
        assert_eq!(dull.len(), 2, "need two non-matching part rows to swap");
        let (i, j) = (dull[0], dull[1]);
        assert_ne!(
            honest.part[i][0], honest.part[j][0],
            "the swap has to actually change a committed column"
        );

        let mut tampered = tables();
        tampered.part.swap(i, j);
        let tampered_cols = cols(&tampered);
        assert_ne!(
            tampered_cols[10], honest_cols[10],
            "committed column 10 is p_partkey; the swap must move it"
        );

        // Control: the SAME swapped database, committed honestly, verifies. So
        // nothing about the swapped witness is inherently unsatisfiable, and the
        // rejection below can only come from the mismatch with the commitment.
        MockProver::run(
            K,
            &circuit(&tampered, tampered_cols.clone(), x),
            vec![vec![Fp::from(1u64)], bind_instance(&tampered_cols, x)],
        )
        .unwrap()
        .assert_satisfied();

        // The cheat: prove over the swapped database, publish the honest one.
        let verdict = MockProver::run(
            K,
            &circuit(&tampered, honest_cols.clone(), x),
            vec![vec![Fp::from(1u64)], bind_instance(&honest_cols, x)],
        )
        .unwrap()
        .verify();
        let failures =
            verdict.expect_err("a witness differing from the committed columns must not verify");
        println!(
            "swapped part rows {} and {}: {} failures",
            i,
            j,
            failures.len()
        );
        for f in failures.iter() {
            println!("  {:?}", f);
        }
        // And it must reject through the TIE, not through some incidental gate:
        // the failing cells are the chip's part columns at the two swapped rows
        // and the binding's data columns opposite them.
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
