
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::q9_obj;
use crate::inline_bind::{assign_bind_cells, configure_bind, tie_columns, BindConfig};

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
        // The whole baseline circuit lives in the chip's `configure`, so this IS
        // the full baseline constraint system. The six base relations' advice
        // columns -- the committed ones -- already have equality enabled there,
        // which is what lets `tie_columns` copy-constrain them.
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
        let (out, input_cells) = chip.assign_with_input_cells(
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
        // The binding evaluates its own columns at x, then those columns are
        // copy-constrained cell-by-cell to the base-relation cells THIS proof
        // witnesses. Without the tie the binding would establish only that some
        // columns match Commit(D); with it, the exposed evaluation is an
        // evaluation of this proof's witness, which is what Appendix A requires
        // and what discharges Threat (i).
        let bind_cells = assign_bind_cells(&mut layouter, &config.1, &self.columns, self.x)?;
        tie_columns(&mut layouter, &bind_cells, &input_cells)?;
        Ok(())
    }
}

#[cfg(test)]
mod q9_bound_to_witness_tests {
    use super::*;
    use crate::bench_queries::{scale_by_1000, string_to_u64, year_from_date};
    use crate::data::data_processing as dp;
    use crate::inline_bind::bind_instance;
    use halo2_proofs::dev::MockProver;

    /// The TPC-H degree. The binding lives in columns of its own, so
    /// `SimpleFloorPlanner` overlays its region on the query's rows rather than
    /// stacking it after them, and the bound circuit is no taller than the base.
    const K: u32 = 16;

    // The slice of `test_cardinality_preservation` in `q9_obj.rs`, which is
    // known to satisfy every gate, lookup and shuffle of the query: supplier,
    // nation and partsupp whole, part / orders / lineitem truncated. It also
    // keeps the six relations RAGGED (25 .. 8000 rows), which is the case the
    // tie has to get right -- the binding covers max_i |R_i| rows and
    // zero-extends the shorter columns.
    const N_PART: usize = 1600;
    const N_ORD: usize = 4000;
    const N_LINE: usize = 8000;

    struct Db {
        part: Vec<Vec<u64>>,
        supplier: Vec<Vec<u64>>,
        nation: Vec<Vec<u64>>,
        orders: Vec<Vec<u64>>,
        partsupp: Vec<Vec<u64>>,
        lineitem: Vec<Vec<u64>>,
        cond_hash: u64,
    }

    /// The publication's columns: `TpchInput::Q9::columns()`, i.e. `transpose`
    /// over [part, supplier, nation, orders, partsupp, lineitem] in that fixed
    /// table order. Restated locally so the test still says what the encoding
    /// is; the loader below applies exactly the projections `bench_queries`
    /// applies, so committed value == witnessed value cell for cell.
    fn cols(db: &Db) -> Vec<Vec<u64>> {
        let tables: [&Vec<Vec<u64>>; 6] = [
            &db.part,
            &db.supplier,
            &db.nation,
            &db.orders,
            &db.partsupp,
            &db.lineitem,
        ];
        let mut out: Vec<Vec<u64>> = Vec::new();
        for t in tables {
            let width = t.first().map(|r| r.len()).unwrap_or(0);
            for j in 0..width {
                out.push(t.iter().map(|r| r[j]).collect());
            }
        }
        assert_eq!(out.len(), NC, "q9 publishes {} columns", NC);
        out
    }

    fn load() -> Db {
        let part: Vec<Vec<u64>> =
            dp::part_read_records_from_file(&crate::paths::data_file("part.tbl"))
                .expect("part.tbl")
                .iter()
                .take(N_PART)
                .map(|r| vec![r.p_partkey, string_to_u64(&r.p_name)])
                .collect();
        let supplier: Vec<Vec<u64>> =
            dp::supplier_read_records_from_file(&crate::paths::data_file("supplier.tbl"))
                .expect("supplier.tbl")
                .iter()
                .map(|r| vec![r.s_suppkey, r.s_nationkey])
                .collect();
        let nation: Vec<Vec<u64>> =
            dp::nation_read_records_from_file(&crate::paths::data_file("nation.tbl"))
                .expect("nation.tbl")
                .iter()
                .map(|r| vec![r.n_nationkey, string_to_u64(&r.n_name)])
                .collect();
        let orders: Vec<Vec<u64>> =
            dp::orders_read_records_from_file(&crate::paths::data_file("orders.tbl"))
                .expect("orders.tbl")
                .iter()
                .take(N_ORD)
                .map(|r| vec![r.o_orderkey, year_from_date(&r.o_orderdate)])
                .collect();
        // PS_SHIFT is private to `q9_obj`; `bench_queries` mirrors it with the
        // same value, which is the constant the publication uses.
        const PS_SHIFT: u64 = 1u64 << 20;
        let partsupp: Vec<Vec<u64>> =
            dp::partsupp_read_records_from_file(&crate::paths::data_file("partsupp.tbl"))
                .expect("partsupp.tbl")
                .iter()
                .map(|r| {
                    vec![
                        r.ps_partkey * PS_SHIFT + r.ps_suppkey,
                        scale_by_1000(r.ps_supplycost),
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
                        r.l_partkey,
                        r.l_suppkey,
                        r.l_quantity,
                        scale_by_1000(r.l_extendedprice),
                        scale_by_1000(r.l_discount),
                    ]
                })
                .collect();

        // `string_to_u64("green")` matches no part name on this slice, which
        // would leave the clean instance empty and the check vacuous. This is
        // the name of part 848, so two part rows pass the predicate and the
        // reduced instance is nonempty on all six relations.
        let cond_hash = string_to_u64("orange olive puff midnight almond");
        let kept = part.iter().filter(|r| r[1] == cond_hash).count();
        assert!(kept > 0, "the slice keeps no part, so the query is vacuous");

        Db { part, supplier, nation, orders, partsupp, lineitem, cond_hash }
    }

    fn circuit(db: &Db, columns: Vec<Vec<u64>>, x: Fp) -> BoundQ9 {
        BoundQ9 {
            part: db.part.clone(),
            supplier: db.supplier.clone(),
            nation: db.nation.clone(),
            orders: db.orders.clone(),
            partsupp: db.partsupp.clone(),
            lineitem: db.lineitem.clone(),
            cond_hash: db.cond_hash,
            columns,
            x,
        }
    }

    /// The honest prover: the committed columns ARE the circuit's input tables.
    #[test]
    fn binding_accepts_the_committed_witness() {
        let db = load();
        let x = Fp::from(7u64);
        let c = cols(&db);
        let inst = vec![vec![Fp::from(1u64)], bind_instance(&c, x)];
        MockProver::run(K, &circuit(&db, c.clone(), x), inst)
            .unwrap()
            .assert_satisfied();
    }

    /// THE POINT OF THE WHOLE EXERCISE. The prover answers the query over one
    /// database while claiming the evaluation of a DIFFERENT one -- exactly the
    /// substitution Threat (i) is about. With the binding's data columns merely
    /// a private copy, the two were unrelated and this verified: the binding
    /// proved that some columns matched Commit(D), never that the query's
    /// witness did.
    #[test]
    fn binding_rejects_a_witness_that_is_not_the_committed_data() {
        let db = load();
        let x = Fp::from(7u64);
        let honest = cols(&db);

        // Two lineitem rows swapped: same length, same multiset per column, so a
        // shuffle or multiset argument would miss it; a random-point evaluation
        // does not. The query answer is a constant 1 and the aggregation is
        // order-independent, so the public output is untouched and only the
        // binding can reject.
        let mut tampered = load();
        let (i, j) = (0usize, 1usize);
        tampered.lineitem.swap(i, j);
        assert_ne!(
            db.lineitem[i], db.lineitem[j],
            "the two swapped rows are identical, so the witness is unchanged \
             and the test is vacuous"
        );

        // Control: the SAME swapped database, committed honestly, verifies. So
        // nothing about the swapped witness is inherently unsatisfiable and the
        // rejection below can only come from the mismatch with the commitment.
        let swapped = cols(&tampered);
        MockProver::run(
            K,
            &circuit(&tampered, swapped.clone(), x),
            vec![vec![Fp::from(1u64)], bind_instance(&swapped, x)],
        )
        .unwrap()
        .assert_satisfied();

        let inst = vec![vec![Fp::from(1u64)], bind_instance(&honest, x)];
        let verdict = MockProver::run(K, &circuit(&tampered, honest, x), inst)
            .unwrap()
            .verify();
        let failures =
            verdict.expect_err("a witness differing from the committed columns must not verify");
        // And it must reject through the TIE, not through some incidental gate:
        // the failing cells are the lineitem input columns at the two swapped
        // rows and the binding's data columns opposite them.
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
