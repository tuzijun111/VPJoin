use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error, Expression, Instance, Selector},
    poly::Rotation,
};
use halo2curves::pasta::Fp;

use ff::Field;

/// Binding columns and gates added to a host circuit's constraint system.
#[derive(Clone, Debug)]
pub struct BindConfig {
    data: Vec<Column<Advice>>,
    acc: Vec<Column<Advice>>,
    pow: Column<Advice>,
    xcol: Column<Advice>,
    /// Second instance column: `[x, v_0 .. v_{NC-1}]`.
    pub instance: Column<Instance>,
    q_first: Selector,
    q_step: Selector,
}

/// Add the binding columns and gates for `nc` committed columns.
///
/// Call this from the host circuit's `configure`, after the query's own
/// `configure`, so both sets of gates share one constraint system.
pub fn configure_bind(meta: &mut ConstraintSystem<Fp>, nc: usize) -> BindConfig {
    let data: Vec<Column<Advice>> = (0..nc).map(|_| meta.advice_column()).collect();
    let acc: Vec<Column<Advice>> = (0..nc).map(|_| meta.advice_column()).collect();
    let pow = meta.advice_column();
    let xcol = meta.advice_column();
    for c in acc.iter() {
        meta.enable_equality(*c);
    }
    meta.enable_equality(xcol);
    let instance = meta.instance_column();
    meta.enable_equality(instance);

    let q_first = meta.selector();
    let q_step = meta.selector();

    meta.create_gate("bind: first row", |m| {
        let q = m.query_selector(q_first);
        let one = Expression::Constant(Fp::ONE);
        let mut cs = vec![q.clone() * (m.query_advice(pow, Rotation::cur()) - one)];
        for j in 0..nc {
            cs.push(
                q.clone()
                    * (m.query_advice(acc[j], Rotation::cur())
                        - m.query_advice(data[j], Rotation::cur())),
            );
        }
        cs
    });

    meta.create_gate("bind: horner step", |m| {
        let q = m.query_selector(q_step);
        let x_cur = m.query_advice(xcol, Rotation::cur());
        let x_next = m.query_advice(xcol, Rotation::next());
        let pow_cur = m.query_advice(pow, Rotation::cur());
        let pow_next = m.query_advice(pow, Rotation::next());
        let mut cs = vec![
            q.clone() * (x_next - x_cur.clone()),
            q.clone() * (pow_next.clone() - pow_cur * x_cur),
        ];
        for j in 0..nc {
            cs.push(
                q.clone()
                    * (m.query_advice(acc[j], Rotation::next())
                        - m.query_advice(acc[j], Rotation::cur())
                        - m.query_advice(data[j], Rotation::next()) * pow_next.clone()),
            );
        }
        cs
    });

    BindConfig {
        data,
        acc,
        pow,
        xcol,
        instance,
        q_first,
        q_step,
    }
}

/// Assign the binding region and expose `[x, v_0 .. v_{NC-1}]` publicly.
///
/// `columns` are the query's input columns, zero-extended to a common height;
/// `x` is the Fiat-Shamir challenge derived from the published commitments and
/// the query proof.
pub fn assign_bind(
    layouter: &mut impl Layouter<Fp>,
    config: &BindConfig,
    columns: &[Vec<u64>],
    x: Fp,
) -> Result<(), Error> {
    let nc = config.data.len();
    assert_eq!(columns.len(), nc, "expected {} columns", nc);
    let rows = columns.iter().map(|c| c.len()).max().unwrap_or(0).max(1);

    let cells = layouter.assign_region(
        || "inlined witness binding",
        |mut region| {
            let mut acc = vec![Fp::ZERO; nc];
            let mut pow = Fp::ONE;
            let mut out = Vec::new();

            for i in 0..rows {
                if i == 0 {
                    config.q_first.enable(&mut region, 0)?;
                } else {
                    config.q_step.enable(&mut region, i - 1)?;
                }
                region.assign_advice(|| "x", config.xcol, i, || Value::known(x))?;
                if i > 0 {
                    pow *= x;
                }
                region.assign_advice(|| "pow", config.pow, i, || Value::known(pow))?;

                for j in 0..nc {
                    let v = columns[j].get(i).copied().unwrap_or(0);
                    let fv = Fp::from(v);
                    region.assign_advice(|| "data", config.data[j], i, || Value::known(fv))?;
                    if i == 0 {
                        acc[j] = fv;
                    } else {
                        acc[j] += fv * pow;
                    }
                    let cell =
                        region.assign_advice(|| "acc", config.acc[j], i, || Value::known(acc[j]))?;
                    if i == rows - 1 {
                        out.push(cell);
                    }
                }
            }
            Ok(out)
        },
    )?;

    // instance[0] is x; instance[1 + j] is the evaluation of column j.
    for (j, cell) in cells.into_iter().enumerate() {
        layouter.constrain_instance(cell.cell(), config.instance, 1 + j)?;
    }
    Ok(())
}

/// The public instance for the binding column: `[x, v_0 .. v_{NC-1}]`.
pub fn bind_instance(columns: &[Vec<u64>], x: Fp) -> Vec<Fp> {
    let mut out = vec![x];
    for col in columns {
        let mut acc = Fp::ZERO;
        let mut pow = Fp::ONE;
        for (i, v) in col.iter().enumerate() {
            if i > 0 {
                pow *= x;
            }
            acc += Fp::from(*v) * pow;
        }
        out.push(acc);
    }
    out
}

// ---------------------------------------------------------------------------
// The bound circuits themselves live in separate files, one per query,
// NEXT TO their baselines, so the two can be diffed side by side:
//
//   baseline                        bound (baseline + inlined check)
//   src/sql/q3_obj.rs           <-> src/sql/q3_bound.rs
//   src/sql/q5_obj.rs           <-> src/sql/q5_bound.rs
//   src/sql/q8_obj.rs           <-> src/sql/q8_bound.rs
//   src/sql/q9_obj.rs           <-> src/sql/q9_bound.rs
//   src/sql/q18_obj.rs          <-> src/sql/q18_bound.rs
//   src/graph_sql/g_sql1_obj.rs <-> src/graph_sql/g_sql1_bound.rs
//   src/graph_sql/g_sql2_obj.rs <-> src/graph_sql/g_sql2_bound.rs
//   src/graph_sql/g_sql3_obj.rs <-> src/graph_sql/g_sql3_bound.rs
//   src/graph_sql/g_sql4_obj.rs <-> src/graph_sql/g_sql4_bound.rs
//
// Re-exported here so existing call sites keep working.
// ---------------------------------------------------------------------------

pub use crate::graph_sql::{
    g_sql1_bound::BoundGq1, g_sql2_bound::BoundGq2, g_sql3_bound::BoundGq3,
    g_sql4_bound::BoundGq4,
};
pub use crate::sql::{
    q18_bound::BoundQ18, q3_bound::BoundQ3, q5_bound::BoundQ5, q8_bound::BoundQ8,
    q9_bound::BoundQ9,
};

use crate::data::graph_data_processing::Edge;
use halo2_proofs::circuit::SimpleFloorPlanner;
use halo2_proofs::plonk::Circuit;
// ---------------------------------------------------------------------------
// Proving both variants from one loaded input
// ---------------------------------------------------------------------------

use crate::bench_queries::TpchInput;
use halo2_proofs::{
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof},
    poly::{
        ipa::{
            commitment::{IPACommitmentScheme, ParamsIPA},
            multiopen::ProverIPA,
            strategy::SingleStrategy,
        },
        VerificationStrategy,
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use halo2curves::pasta::vesta;
use rand::rngs::OsRng;
use std::time::Instant;

/// Outcome of proving one circuit variant.  Every proof is also verified.
pub struct Proved {
    pub prove_s: f64,
    pub verify_s: f64,
    pub bytes: usize,
    pub proof: Vec<u8>,
}

fn prove_one<C: Circuit<Fp>>(
    params: &ParamsIPA<vesta::Affine>,
    circuit: &C,
    instances: &[&[Fp]],
) -> Proved {
    let pk = keygen_for(params, circuit);
    prove_with(params, &pk, circuit, instances)
}

/// Build the proving key once, so it can be reused across repetitions.
///
/// Keygen is deliberately separated from proving: it is untimed but expensive,
/// and running it immediately before each timed `create_proof` would warm
/// caches by a different amount for each circuit (the bound circuit's keygen
/// is larger), systematically biasing the comparison.
pub fn keygen_for<C: Circuit<Fp>>(
    params: &ParamsIPA<vesta::Affine>,
    circuit: &C,
) -> halo2_proofs::plonk::ProvingKey<vesta::Affine> {
    let vk = keygen_vk(params, circuit).expect("keygen_vk");
    keygen_pk(params, vk, circuit).expect("keygen_pk")
}

/// Prove with a pre-built proving key. Only `create_proof` is timed.
pub fn prove_with<C: Circuit<Fp>>(
    params: &ParamsIPA<vesta::Affine>,
    pk: &halo2_proofs::plonk::ProvingKey<vesta::Affine>,
    circuit: &C,
    instances: &[&[Fp]],
) -> Proved {
    let t = Instant::now();
    let mut tr = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    create_proof::<IPACommitmentScheme<_>, ProverIPA<_>, _, _, _, _>(
        params,
        pk,
        std::slice::from_ref(circuit),
        &[instances],
        OsRng,
        &mut tr,
    )
    .expect("create_proof");
    let proof = tr.finalize();
    let prove_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let strategy = SingleStrategy::new(params);
    let mut rt = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    assert!(
        verify_proof(params, pk.get_vk(), strategy, &[instances], &mut rt).is_ok(),
        "verification failed"
    );
    let verify_s = t.elapsed().as_secs_f64();

    Proved { prove_s, verify_s, bytes: proof.len(), proof }
}

/// Prove the query circuit exactly as the submission runs it.
pub fn prove_plain(params: &ParamsIPA<vesta::Affine>, input: &TpchInput) -> Proved {
    use crate::sql::{q18_obj, q3_obj, q5_obj, q8_obj, q9_obj};
    use std::marker::PhantomData;
    let one = [Fp::from(1u64)];
    let inst: &[&[Fp]] = &[&one];
    match input {
        TpchInput::Q3 { customer, orders, lineitem, condition } => prove_one(
            params,
            &q3_obj::MyCircuit::<Fp> {
                customer: customer.clone(),
                orders: orders.clone(),
                lineitem: lineitem.clone(),
                condition: *condition,
                _marker: PhantomData,
            },
            inst,
        ),
        TpchInput::Q5 {
            customer, orders, lineitem, supplier, nation, region,
            europe_hash, start_ts, end_ts, nr_pad_extra, co_pad_extra, ls_pad_extra,
        } => prove_one(
            params,
            &q5_obj::MyCircuit::<Fp> {
                customer: customer.clone(), orders: orders.clone(), lineitem: lineitem.clone(),
                supplier: supplier.clone(), nation: nation.clone(), region: region.clone(),
                europe_hash: *europe_hash, start_ts: *start_ts, end_ts: *end_ts,
                nr_pad_extra: *nr_pad_extra, co_pad_extra: *co_pad_extra,
                ls_pad_extra: *ls_pad_extra,
                _marker: PhantomData,
            },
            inst,
        ),
        TpchInput::Q8 {
            region, nation, customer, orders, part, supplier, lineitem,
            cond_nation_hash, const_region_name_hash, const_part_type_hash,
        } => prove_one(
            params,
            &q8_obj::MyCircuit::<Fp> {
                region: region.clone(), nation: nation.clone(), customer: customer.clone(),
                orders: orders.clone(), part: part.clone(), supplier: supplier.clone(),
                lineitem: lineitem.clone(),
                cond_nation_hash: *cond_nation_hash,
                const_region_name_hash: *const_region_name_hash,
                const_part_type_hash: *const_part_type_hash,
                _marker: PhantomData,
            },
            inst,
        ),
        TpchInput::Q9 { part, supplier, nation, orders, partsupp, lineitem, cond_hash } => {
            prove_one(
                params,
                &q9_obj::MyCircuit::<Fp> {
                    part: part.clone(), supplier: supplier.clone(), nation: nation.clone(),
                    orders: orders.clone(), partsupp: partsupp.clone(),
                    lineitem: lineitem.clone(), cond_hash: *cond_hash,
                    _marker: PhantomData,
                },
                inst,
            )
        }
        TpchInput::Q18 { customer, orders, lineitem, threshold } => prove_one(
            params,
            &q18_obj::MyCircuit::<Fp> {
                customer: customer.clone(), orders: orders.clone(), lineitem: lineitem.clone(),
                threshold: *threshold,
                _marker: PhantomData,
            },
            inst,
        ),
    }
}

/// Prove the SAME query circuit with the binding check inlined into it.
pub fn prove_bound(
    params: &ParamsIPA<vesta::Affine>,
    input: &TpchInput,
    x: Fp,
    bind_pub: &[Fp],
) -> Proved {
    let one = [Fp::from(1u64)];
    let inst: &[&[Fp]] = &[&one, bind_pub];
    let columns = input.columns();
    match input {
        TpchInput::Q3 { customer, orders, lineitem, condition } => prove_one(
            params,
            &BoundQ3 {
                customer: customer.clone(), orders: orders.clone(), lineitem: lineitem.clone(),
                condition: *condition, columns, x,
            },
            inst,
        ),
        TpchInput::Q5 {
            customer, orders, lineitem, supplier, nation, region,
            europe_hash, start_ts, end_ts, nr_pad_extra, co_pad_extra, ls_pad_extra,
        } => prove_one(
            params,
            &BoundQ5 {
                customer: customer.clone(), orders: orders.clone(), lineitem: lineitem.clone(),
                supplier: supplier.clone(), nation: nation.clone(), region: region.clone(),
                europe_hash: *europe_hash, start_ts: *start_ts, end_ts: *end_ts,
                nr_pad_extra: *nr_pad_extra, co_pad_extra: *co_pad_extra,
                ls_pad_extra: *ls_pad_extra,
                columns, x,
            },
            inst,
        ),
        TpchInput::Q8 {
            region, nation, customer, orders, part, supplier, lineitem,
            cond_nation_hash, const_region_name_hash, const_part_type_hash,
        } => prove_one(
            params,
            &BoundQ8 {
                region: region.clone(), nation: nation.clone(), customer: customer.clone(),
                orders: orders.clone(), part: part.clone(), supplier: supplier.clone(),
                lineitem: lineitem.clone(),
                cond_nation_hash: *cond_nation_hash,
                const_region_name_hash: *const_region_name_hash,
                const_part_type_hash: *const_part_type_hash,
                columns, x,
            },
            inst,
        ),
        TpchInput::Q9 { part, supplier, nation, orders, partsupp, lineitem, cond_hash } => {
            prove_one(
                params,
                &BoundQ9 {
                    part: part.clone(), supplier: supplier.clone(), nation: nation.clone(),
                    orders: orders.clone(), partsupp: partsupp.clone(),
                    lineitem: lineitem.clone(), cond_hash: *cond_hash,
                    columns, x,
                },
                inst,
            )
        }
        TpchInput::Q18 { customer, orders, lineitem, threshold } => prove_one(
            params,
            &BoundQ18 {
                customer: customer.clone(), orders: orders.clone(), lineitem: lineitem.clone(),
                threshold: *threshold, columns, x,
            },
            inst,
        ),
    }
}

/// Edge columns (src, dst) of a graph -- the columns the layer commits.
pub fn edge_columns(edges: &[Edge]) -> Vec<Vec<u64>> {
    vec![
        edges.iter().map(|e| e.src).collect(),
        edges.iter().map(|e| e.dst).collect(),
    ]
}

/// Prove a graph query exactly as the submission runs it.
pub fn prove_graph_base(
    params: &ParamsIPA<vesta::Affine>,
    query: &str,
    edges: &[Edge],
    pads: (usize, usize),
    cnt: u64,
) -> Proved {
    use crate::graph_sql::{g_sql1_obj, g_sql2_obj, g_sql3_obj, g_sql4_obj};
    use std::marker::PhantomData;
    let out = [Fp::from(cnt)];
    let inst: &[&[Fp]] = &[&out];
    match query {
        "gq1" => prove_one(
            params,
            &g_sql1_obj::Path3OrdCircuit::<Fp> { edges: edges.to_vec(), _marker: PhantomData },
            inst,
        ),
        "gq2" => prove_one(
            params,
            &g_sql2_obj::GraphPath4OrderCircuit::<Fp> {
                edges: edges.to_vec(),
                _marker: PhantomData,
            },
            inst,
        ),
        "gq3" => prove_one(
            params,
            &g_sql3_obj::MyCircuit::<Fp> {
                edges: edges.to_vec(),
                bag1_pad_extra: pads.0,
                bag2_pad_extra: pads.1,
                _marker: PhantomData,
            },
            inst,
        ),
        // GQ4 materializes ONE bag read in two column roles, so it takes a
        // single pad. `pads` keeps its pair shape for GQ3 and Q5; only .0 is used.
        "gq4" => prove_one(
            params,
            &g_sql4_obj::MyCircuit::<Fp> {
                edges: edges.to_vec(),
                pad_extra: pads.0,
                _marker: PhantomData,
            },
            inst,
        ),
        other => panic!("unknown graph query {}", other),
    }
}

/// Prove the same graph query with the binding check inlined.
pub fn prove_graph_bound(
    params: &ParamsIPA<vesta::Affine>,
    query: &str,
    edges: &[Edge],
    pads: (usize, usize),
    cnt: u64,
    x: Fp,
    bind_pub: &[Fp],
) -> Proved {
    let columns = edge_columns(edges);
    let out = [Fp::from(cnt)];
    let inst: &[&[Fp]] = &[&out, bind_pub];
    match query {
        "gq1" => prove_one(
            params,
            &BoundGq1 { edges: edges.to_vec(), columns, x },
            inst,
        ),
        "gq2" => prove_one(
            params,
            &BoundGq2 { edges: edges.to_vec(), columns, x },
            inst,
        ),
        "gq3" => prove_one(
            params,
            &BoundGq3 {
                edges: edges.to_vec(),
                bag1_pad_extra: pads.0,
                bag2_pad_extra: pads.1,
                columns,
                x,
            },
            inst,
        ),
        "gq4" => prove_one(
            params,
            &BoundGq4 {
                edges: edges.to_vec(),
                pad_extra: pads.0,
                columns,
                x,
            },
            inst,
        ),
        other => panic!("unknown graph query {}", other),
    }
}

// ---------------------------------------------------------------------------
// Structural guard
// ---------------------------------------------------------------------------

/// Gate / lookup / column counts of a circuit's constraint system.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct CsShape {
    pub gates: usize,
    pub lookups: usize,
    pub advice: usize,
    pub fixed: usize,
}

/// Configure `C` into a fresh constraint system and report its shape.
///
/// `configure` never sees witness data, so for a given circuit this is
/// IDENTICAL across datasets -- which is why the per-column proving work added
/// by the binding cannot legitimately differ between datasets at the same `k`.
pub fn cs_shape<C: Circuit<Fp>>() -> CsShape {
    let mut cs = ConstraintSystem::<Fp>::default();
    C::configure(&mut cs);
    CsShape {
        gates: cs.gates().len(),
        lookups: cs.lookups().len(),
        advice: cs.num_advice_columns(),
        fixed: cs.num_fixed_columns(),
    }
}

/// Assert the bound circuit's constraint system is a strict SUPERSET of the
/// base circuit's.
///
/// Without this, a wrapper that forgets part of the original `configure` (as
/// happened for GQ1/GQ2, whose circuits add lookup arguments on top of their
/// chip's `configure`) silently proves a weaker statement -- and shows up as
/// *negative* measured overhead, because the missing lookups were expensive.
pub fn assert_bound_superset<B: Circuit<Fp>, D: Circuit<Fp>>(label: &str) {
    let b = cs_shape::<B>();
    let d = cs_shape::<D>();
    assert!(
        d.gates >= b.gates && d.advice > b.advice,
        "{}: bound circuit is not a superset of the base circuit.\n  \
         base : {:?}\n  bound: {:?}\n  \
         The wrapper's `configure` must reproduce the base circuit's \
         constraints exactly, then add the binding columns.",
        label,
        b,
        d
    );
    assert_eq!(
        d.lookups, b.lookups,
        "{}: binding adds no lookups, so the counts must match ({:?} vs {:?})",
        label, b, d
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_sql::{g_sql1_obj, g_sql2_obj, g_sql3_obj, g_sql4_obj};
    use crate::sql::{q18_obj, q3_obj, q5_obj, q8_obj, q9_obj};

    /// Every bound circuit must contain everything its base circuit contains.
    #[test]
    fn bound_circuits_are_supersets_of_their_base() {
        assert_bound_superset::<q3_obj::MyCircuit<Fp>, BoundQ3>("q3");
        assert_bound_superset::<q5_obj::MyCircuit<Fp>, BoundQ5>("q5");
        assert_bound_superset::<q8_obj::MyCircuit<Fp>, BoundQ8>("q8");
        assert_bound_superset::<q9_obj::MyCircuit<Fp>, BoundQ9>("q9");
        assert_bound_superset::<q18_obj::MyCircuit<Fp>, BoundQ18>("q18");
        assert_bound_superset::<g_sql1_obj::Path3OrdCircuit<Fp>, BoundGq1>("gq1");
        assert_bound_superset::<g_sql2_obj::GraphPath4OrderCircuit<Fp>, BoundGq2>("gq2");
        assert_bound_superset::<g_sql3_obj::MyCircuit<Fp>, BoundGq3>("gq3");
        assert_bound_superset::<g_sql4_obj::MyCircuit<Fp>, BoundGq4>("gq4");
    }

    /// Advice-column growth per query: the binding adds 2*NC+2 columns, and
    /// prover cost is roughly proportional to column count at a fixed degree.
    /// This gives an expected in-circ as a fraction of the base proof.
    #[test]
    fn expected_incirc_fraction() {
        fn row<B: Circuit<Fp>, D: Circuit<Fp>>(name: &str, base_s: f64) {
            let b = cs_shape::<B>();
            let d = cs_shape::<D>();
            let frac = (d.advice - b.advice) as f64 / b.advice as f64;
            println!(
                "{:<12} advice {:>3} -> {:>3} (+{:>2}, {:>5.1}%)   base {:>5.1}s  => expected in-circ ~{:.2}s",
                name, b.advice, d.advice, d.advice - b.advice, 100.0 * frac, base_s, frac * base_s
            );
        }
        row::<q3_obj::MyCircuit<Fp>, BoundQ3>("q3", 24.2);
        row::<q8_obj::MyCircuit<Fp>, BoundQ8>("q8", 18.1);
        row::<q9_obj::MyCircuit<Fp>, BoundQ9>("q9", 20.1);
        row::<q18_obj::MyCircuit<Fp>, BoundQ18>("q18", 12.6);
        row::<g_sql2_obj::GraphPath4OrderCircuit<Fp>, BoundGq2>("gq2", 96.5);
    }

    /// Post-compression gate cost: `compress_selectors` runs inside keygen and
    /// repacks ALL selectors into fixed columns, rewriting every gate that used
    /// a simple selector. Adding selectors can therefore change the expressions
    /// of the HOST circuit's gates, and the quotient evaluation pays that cost
    /// on every one of the 2^ext_k rows.
    #[test]
    fn post_compression_gate_cost() {
        use halo2_proofs::poly::commitment::ParamsProver;
        use halo2_proofs::poly::ipa::commitment::ParamsIPA;
        use halo2curves::pasta::vesta;

        fn nodes(e: &halo2_proofs::plonk::Expression<Fp>) -> usize {
            use halo2_proofs::plonk::Expression::*;
            match e {
                Sum(a, b) | Product(a, b) => 1 + nodes(a) + nodes(b),
                Negated(a) | Scaled(a, _) => 1 + nodes(a),
                _ => 1,
            }
        }
        fn probe<C: Circuit<Fp>>(name: &str, params: &ParamsIPA<vesta::Affine>, c: &C) {
            let vk = halo2_proofs::plonk::keygen_vk(params, c).expect("keygen_vk");
            let cs = vk.cs();
            let total: usize = cs
                .gates()
                .iter()
                .flat_map(|g| g.polynomials().iter())
                .map(nodes)
                .sum();
            let n: usize = cs.gates().iter().map(|g| g.polynomials().len()).sum();
            println!(
                "{:<11} POST-compression: gates={:<3} constraints={:<4} expr_nodes={:<6} fixed={:<3} degree={}",
                name,
                cs.gates().len(),
                n,
                total,
                cs.num_fixed_columns(),
                cs.degree()
            );
        }

        let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(9);
        probe("gq1 base", &params, &g_sql1_obj::Path3OrdCircuit::<Fp>::default());
        probe("gq1 bound", &params, &BoundGq1 {
            edges: vec![],
            columns: vec![vec![], vec![]], x: Fp::ZERO,
        });
    }

    /// The quotient-polynomial degree drives the EXTENDED domain size, which
    /// is the single largest factor in prover cost. If bound < base here, the
    /// bound circuit legitimately proves faster despite having more columns.
    #[test]
    fn extended_domain_factor() {
        fn probe<C: Circuit<Fp>>(name: &str) {
            let mut cs = ConstraintSystem::<Fp>::default();
            C::configure(&mut cs);
            let deg = cs.degree();
            let factor = ((deg - 1) as u64).next_power_of_two();
            println!(
                "{:<10} degree={:<3} blinding={:<3} extended_factor={}",
                name,
                deg,
                cs.blinding_factors(),
                factor
            );
        }
        probe::<g_sql1_obj::Path3OrdCircuit<Fp>>("gq1 base");
        probe::<BoundGq1>("gq1 bound");
        probe::<g_sql2_obj::GraphPath4OrderCircuit<Fp>>("gq2 base");
        probe::<BoundGq2>("gq2 bound");
        probe::<q18_obj::MyCircuit<Fp>>("q18 base");
        probe::<BoundQ18>("q18 bound");
    }

    /// Print the prover-cost-relevant `ConstraintSystem` numbers for every
    /// base/bound pair.  Cheap: only `configure` runs, nothing is proved.
    #[test]
    fn print_domain_numbers() {
        use halo2_proofs::poly::EvaluationDomain;

        fn dump<C: Circuit<Fp>>(label: &str, k: u32) {
            let mut cs = ConstraintSystem::<Fp>::default();
            C::configure(&mut cs);
            let degree = cs.degree();
            let dom = EvaluationDomain::<Fp>::new(degree as u32, k);
            let max_gate_degree = cs
                .gates()
                .iter()
                .flat_map(|g| g.polynomials().iter().map(|p| p.degree()))
                .max()
                .unwrap_or(0);
            let perm_cols = cs.permutation().get_columns().len();
            let chunk = degree - 2;
            let perm_chunks = (perm_cols + chunk - 1) / chunk;
            println!(
                "{:>10} | deg {:>2} | ext_k {:>2} (x{:>2}) | blind {:>2} | \
                 maxgate {:>2} | adv {:>3} | fix {:>2} | sel {:>3} | inst {} | \
                 lookups {:>3} | advq {:>3} | perm_cols {:>3} -> {} chunks",
                label,
                degree,
                dom.extended_k(),
                1 << (dom.extended_k() - k),
                cs.blinding_factors(),
                max_gate_degree,
                cs.num_advice_columns(),
                cs.num_fixed_columns(),
                cs.num_selectors(),
                cs.num_instance_columns(),
                cs.lookups().len(),
                cs.advice_queries().len(),
                perm_cols,
                perm_chunks,
            );
        }

        let k = 17;
        dump::<g_sql1_obj::Path3OrdCircuit<Fp>>("gq1 base", k);
        dump::<BoundGq1>("gq1 bound", k);
        dump::<g_sql2_obj::GraphPath4OrderCircuit<Fp>>("gq2 base", k);
        dump::<BoundGq2>("gq2 bound", k);
        dump::<g_sql3_obj::MyCircuit<Fp>>("gq3 base", k);
        dump::<BoundGq3>("gq3 bound", k);
        dump::<g_sql4_obj::MyCircuit<Fp>>("gq4 base", k);
        dump::<BoundGq4>("gq4 bound", k);
        dump::<q3_obj::MyCircuit<Fp>>("q3 base", k);
        dump::<BoundQ3>("q3 bound", k);
        dump::<q5_obj::MyCircuit<Fp>>("q5 base", k);
        dump::<BoundQ5>("q5 bound", k);
        dump::<q8_obj::MyCircuit<Fp>>("q8 base", k);
        dump::<BoundQ8>("q8 bound", k);
        dump::<q9_obj::MyCircuit<Fp>>("q9 base", k);
        dump::<BoundQ9>("q9 bound", k);
        dump::<q18_obj::MyCircuit<Fp>>("q18 base", k);
        dump::<BoundQ18>("q18 bound", k);
    }

    /// The added constraint work is the same for every dataset of a query.
    #[test]
    fn binding_overhead_is_data_independent() {
        let base = cs_shape::<g_sql2_obj::GraphPath4OrderCircuit<Fp>>();
        let bound = cs_shape::<BoundGq2>();
        println!("gq2 base : {:?}", base);
        println!("gq2 bound: {:?}", bound);
        assert_eq!(bound.gates - base.gates, 2);
        assert_eq!(bound.advice - base.advice, 6);
        assert_eq!(bound.lookups, base.lookups);
    }
}

// ---------------------------------------------------------------------------
// Fair paired measurement
// ---------------------------------------------------------------------------

/// Prove `base` and `bound` `reps` times each, alternating which goes first,
/// and return the raw `(base_s, bound_s)` per repetition.
///
/// Fairness measures, all of which matter because the binding cost is only a
/// few percent of the query proof:
///  * each proving key is built ONCE, outside the timed region, so a timed
///    proof is never preceded by a different amount of untimed keygen work
///    (the bound circuit's keygen is larger, which otherwise warms caches in
///    its favour);
///  * one proof of EACH circuit is run and discarded first, so allocator
///    arenas, page tables and the thread pool are warm before anything counts;
///  * the order alternates, so any residual drift cancels across repetitions.
pub fn paired_runs<B: Circuit<Fp>, D: Circuit<Fp>>(
    params: &ParamsIPA<vesta::Affine>,
    base: &B,
    base_inst: &[&[Fp]],
    bound: &D,
    bound_inst: &[&[Fp]],
    reps: usize,
) -> Vec<(f64, f64)> {
    // Measure each circuit in ISOLATION: build its proving key, run one
    // discarded warm-up proof, take the timed proof, then DROP the key before
    // touching the other circuit.
    //
    // Keeping both keys resident at once is not neutral: they are multi-GB
    // (extended cosets at 2^ext_k), and whichever is allocated second gets
    // better page placement -- worth about 4s out of 60 here, which is several
    // times the effect being measured. Building the two keys in the opposite
    // order flips the sign of the result, which is how that was diagnosed.
    fn timed<C: Circuit<Fp>>(
        params: &ParamsIPA<vesta::Affine>,
        c: &C,
        inst: &[&[Fp]],
    ) -> f64 {
        let pk = keygen_for(params, c);
        let _warm = prove_with(params, &pk, c, inst);
        let t = prove_with(params, &pk, c, inst).prove_s;
        drop(pk);
        t
    }

    // Process-level burn-in. A per-circuit warm-up is not enough: the control
    // experiment (base vs base) shows the first TWO measurements of a process
    // are inflated (64.3s, 61.7s, then settling near 61s), so the drift spans
    // more than one proof. Run and discard a couple of full measurements
    // before anything counts.
    let burn: usize = std::env::var("VPJOIN_BURNIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    for _ in 0..burn {
        let _ = timed(params, base, base_inst);
    }

    let mut out = Vec::with_capacity(reps);
    for i in 0..reps {
        // Alternate which circuit is measured first, so any residual drift
        // cancels across repetitions.
        if i % 2 == 0 {
            let b = timed(params, base, base_inst);
            let d = timed(params, bound, bound_inst);
            out.push((b, d));
        } else {
            let d = timed(params, bound, bound_inst);
            let b = timed(params, base, base_inst);
            out.push((b, d));
        }
    }
    out
}

/// Paired base/bound timings for a graph query.
pub fn graph_paired(
    params: &ParamsIPA<vesta::Affine>,
    query: &str,
    edges: &[Edge],
    pads: (usize, usize),
    cnt: u64,
    x: Fp,
    bind_pub: &[Fp],
    reps: usize,
) -> Vec<(f64, f64)> {
    use crate::graph_sql::{g_sql1_obj, g_sql2_obj, g_sql3_obj, g_sql4_obj};
    use std::marker::PhantomData;

    let columns = edge_columns(edges);
    let out = [Fp::from(cnt)];
    let bi: &[&[Fp]] = &[&out];
    let di: &[&[Fp]] = &[&out, bind_pub];
    let e = edges.to_vec();

    match query {
        "gq1" => paired_runs(
            params,
            &g_sql1_obj::Path3OrdCircuit::<Fp> { edges: e.clone(), _marker: PhantomData },
            bi,
            &BoundGq1 { edges: e, columns, x },
            di,
            reps,
        ),
        "gq2" => paired_runs(
            params,
            &g_sql2_obj::GraphPath4OrderCircuit::<Fp> { edges: e.clone(), _marker: PhantomData },
            bi,
            &BoundGq2 { edges: e, columns, x },
            di,
            reps,
        ),
        "gq3" => paired_runs(
            params,
            &g_sql3_obj::MyCircuit::<Fp> {
                edges: e.clone(),
                bag1_pad_extra: pads.0,
                bag2_pad_extra: pads.1,
                _marker: PhantomData,
            },
            bi,
            &BoundGq3 {
                edges: e,
                bag1_pad_extra: pads.0,
                bag2_pad_extra: pads.1,
                columns,
                x,
            },
            di,
            reps,
        ),
        "gq4" => paired_runs(
            params,
            &g_sql4_obj::MyCircuit::<Fp> {
                edges: e.clone(),
                pad_extra: pads.0,
                _marker: PhantomData,
            },
            bi,
            &BoundGq4 {
                edges: e,
                pad_extra: pads.0,
                columns,
                x,
            },
            di,
            reps,
        ),
        other => panic!("unknown graph query {}", other),
    }
}

/// Paired base/bound timings for a TPC-H query.
pub fn tpch_paired(
    params: &ParamsIPA<vesta::Affine>,
    input: &TpchInput,
    x: Fp,
    bind_pub: &[Fp],
    reps: usize,
) -> Vec<(f64, f64)> {
    use crate::sql::{q18_obj, q3_obj, q5_obj, q8_obj, q9_obj};
    use std::marker::PhantomData;

    let one = [Fp::from(1u64)];
    let bi: &[&[Fp]] = &[&one];
    let di: &[&[Fp]] = &[&one, bind_pub];
    let columns = input.columns();

    match input {
        TpchInput::Q3 { customer, orders, lineitem, condition } => paired_runs(
            params,
            &q3_obj::MyCircuit::<Fp> {
                customer: customer.clone(), orders: orders.clone(),
                lineitem: lineitem.clone(), condition: *condition, _marker: PhantomData,
            },
            bi,
            &BoundQ3 {
                customer: customer.clone(), orders: orders.clone(),
                lineitem: lineitem.clone(), condition: *condition, columns, x,
            },
            di,
            reps,
        ),
        TpchInput::Q5 {
            customer, orders, lineitem, supplier, nation, region,
            europe_hash, start_ts, end_ts, nr_pad_extra, co_pad_extra, ls_pad_extra,
        } => paired_runs(
            params,
            &q5_obj::MyCircuit::<Fp> {
                customer: customer.clone(), orders: orders.clone(), lineitem: lineitem.clone(),
                supplier: supplier.clone(), nation: nation.clone(), region: region.clone(),
                europe_hash: *europe_hash, start_ts: *start_ts, end_ts: *end_ts,
                nr_pad_extra: *nr_pad_extra, co_pad_extra: *co_pad_extra,
                ls_pad_extra: *ls_pad_extra, _marker: PhantomData,
            },
            bi,
            &BoundQ5 {
                customer: customer.clone(), orders: orders.clone(), lineitem: lineitem.clone(),
                supplier: supplier.clone(), nation: nation.clone(), region: region.clone(),
                europe_hash: *europe_hash, start_ts: *start_ts, end_ts: *end_ts,
                nr_pad_extra: *nr_pad_extra, co_pad_extra: *co_pad_extra,
                ls_pad_extra: *ls_pad_extra, columns, x,
            },
            di,
            reps,
        ),
        TpchInput::Q8 {
            region, nation, customer, orders, part, supplier, lineitem,
            cond_nation_hash, const_region_name_hash, const_part_type_hash,
        } => paired_runs(
            params,
            &q8_obj::MyCircuit::<Fp> {
                region: region.clone(), nation: nation.clone(), customer: customer.clone(),
                orders: orders.clone(), part: part.clone(), supplier: supplier.clone(),
                lineitem: lineitem.clone(), cond_nation_hash: *cond_nation_hash,
                const_region_name_hash: *const_region_name_hash,
                const_part_type_hash: *const_part_type_hash, _marker: PhantomData,
            },
            bi,
            &BoundQ8 {
                region: region.clone(), nation: nation.clone(), customer: customer.clone(),
                orders: orders.clone(), part: part.clone(), supplier: supplier.clone(),
                lineitem: lineitem.clone(), cond_nation_hash: *cond_nation_hash,
                const_region_name_hash: *const_region_name_hash,
                const_part_type_hash: *const_part_type_hash, columns, x,
            },
            di,
            reps,
        ),
        TpchInput::Q9 { part, supplier, nation, orders, partsupp, lineitem, cond_hash } => {
            paired_runs(
                params,
                &q9_obj::MyCircuit::<Fp> {
                    part: part.clone(), supplier: supplier.clone(), nation: nation.clone(),
                    orders: orders.clone(), partsupp: partsupp.clone(),
                    lineitem: lineitem.clone(), cond_hash: *cond_hash, _marker: PhantomData,
                },
                bi,
                &BoundQ9 {
                    part: part.clone(), supplier: supplier.clone(), nation: nation.clone(),
                    orders: orders.clone(), partsupp: partsupp.clone(),
                    lineitem: lineitem.clone(), cond_hash: *cond_hash, columns, x,
                },
                di,
                reps,
            )
        }
        TpchInput::Q18 { customer, orders, lineitem, threshold } => paired_runs(
            params,
            &q18_obj::MyCircuit::<Fp> {
                customer: customer.clone(), orders: orders.clone(),
                lineitem: lineitem.clone(), threshold: *threshold, _marker: PhantomData,
            },
            bi,
            &BoundQ18 {
                customer: customer.clone(), orders: orders.clone(),
                lineitem: lineitem.clone(), threshold: *threshold, columns, x,
            },
            di,
            reps,
        ),
    }
}

/// Control experiment: measure the BASE circuit against ITSELF with the exact
/// same harness (two proving keys, warm-up, alternation).
///
/// The true difference is zero by construction, so whatever this reports is
/// pure harness bias. If a base-vs-base pair shows the same negative offset as
/// a base-vs-bound pair, the offset is positional, not circuit work.
pub fn graph_selftest(
    params: &ParamsIPA<vesta::Affine>,
    query: &str,
    edges: &[Edge],
    pads: (usize, usize),
    cnt: u64,
    reps: usize,
) -> Vec<(f64, f64)> {
    use crate::graph_sql::{g_sql1_obj, g_sql2_obj, g_sql3_obj, g_sql4_obj};
    use std::marker::PhantomData;

    let out = [Fp::from(cnt)];
    let bi: &[&[Fp]] = &[&out];
    let e = edges.to_vec();

    match query {
        "gq1" => paired_runs(
            params,
            &g_sql1_obj::Path3OrdCircuit::<Fp> { edges: e.clone(), _marker: PhantomData },
            bi,
            &g_sql1_obj::Path3OrdCircuit::<Fp> { edges: e, _marker: PhantomData },
            bi,
            reps,
        ),
        "gq2" => paired_runs(
            params,
            &g_sql2_obj::GraphPath4OrderCircuit::<Fp> { edges: e.clone(), _marker: PhantomData },
            bi,
            &g_sql2_obj::GraphPath4OrderCircuit::<Fp> { edges: e, _marker: PhantomData },
            bi,
            reps,
        ),
        "gq3" => paired_runs(
            params,
            &g_sql3_obj::MyCircuit::<Fp> {
                edges: e.clone(), bag1_pad_extra: pads.0, bag2_pad_extra: pads.1,
                _marker: PhantomData,
            },
            bi,
            &g_sql3_obj::MyCircuit::<Fp> {
                edges: e, bag1_pad_extra: pads.0, bag2_pad_extra: pads.1,
                _marker: PhantomData,
            },
            bi,
            reps,
        ),
        "gq4" => paired_runs(
            params,
            &g_sql4_obj::MyCircuit::<Fp> {
                edges: e.clone(), pad_extra: pads.0,
                _marker: PhantomData,
            },
            bi,
            &g_sql4_obj::MyCircuit::<Fp> {
                edges: e, pad_extra: pads.0,
                _marker: PhantomData,
            },
            bi,
            reps,
        ),
        other => panic!("unknown graph query {}", other),
    }
}

/// Control experiment for a TPC-H query: prove the BASE circuit against
/// ITSELF with the identical harness. The true difference is zero, so whatever
/// this reports is pure harness bias.
pub fn tpch_selftest(
    params: &ParamsIPA<vesta::Affine>,
    input: &TpchInput,
    reps: usize,
) -> Vec<(f64, f64)> {
    use crate::sql::{q18_obj, q3_obj, q5_obj, q8_obj, q9_obj};
    use std::marker::PhantomData;

    let one = [Fp::from(1u64)];
    let bi: &[&[Fp]] = &[&one];

    macro_rules! pair {
        ($ctor:expr) => {{
            let a = $ctor;
            let b = $ctor;
            paired_runs(params, &a, bi, &b, bi, reps)
        }};
    }

    match input {
        TpchInput::Q3 { customer, orders, lineitem, condition } => pair!(q3_obj::MyCircuit::<Fp> {
            customer: customer.clone(),
            orders: orders.clone(),
            lineitem: lineitem.clone(),
            condition: *condition,
            _marker: PhantomData,
        }),
        TpchInput::Q5 {
            customer, orders, lineitem, supplier, nation, region,
            europe_hash, start_ts, end_ts, nr_pad_extra, co_pad_extra, ls_pad_extra,
        } => pair!(q5_obj::MyCircuit::<Fp> {
            customer: customer.clone(), orders: orders.clone(), lineitem: lineitem.clone(),
            supplier: supplier.clone(), nation: nation.clone(), region: region.clone(),
            europe_hash: *europe_hash, start_ts: *start_ts, end_ts: *end_ts,
            nr_pad_extra: *nr_pad_extra, co_pad_extra: *co_pad_extra,
            ls_pad_extra: *ls_pad_extra, _marker: PhantomData,
        }),
        TpchInput::Q8 {
            region, nation, customer, orders, part, supplier, lineitem,
            cond_nation_hash, const_region_name_hash, const_part_type_hash,
        } => pair!(q8_obj::MyCircuit::<Fp> {
            region: region.clone(), nation: nation.clone(), customer: customer.clone(),
            orders: orders.clone(), part: part.clone(), supplier: supplier.clone(),
            lineitem: lineitem.clone(), cond_nation_hash: *cond_nation_hash,
            const_region_name_hash: *const_region_name_hash,
            const_part_type_hash: *const_part_type_hash, _marker: PhantomData,
        }),
        TpchInput::Q9 { part, supplier, nation, orders, partsupp, lineitem, cond_hash } => {
            pair!(q9_obj::MyCircuit::<Fp> {
                part: part.clone(), supplier: supplier.clone(), nation: nation.clone(),
                orders: orders.clone(), partsupp: partsupp.clone(),
                lineitem: lineitem.clone(), cond_hash: *cond_hash, _marker: PhantomData,
            })
        }
        TpchInput::Q18 { customer, orders, lineitem, threshold } => pair!(q18_obj::MyCircuit::<Fp> {
            customer: customer.clone(), orders: orders.clone(),
            lineitem: lineitem.clone(), threshold: *threshold, _marker: PhantomData,
        }),
    }
}
