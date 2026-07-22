//! The witness-equality check **inlined into the query circuit**, as
//! Appendix A describes it.
//!
//! `column_commit::BoundColumnsCircuit` proves the same statement as a
//! *separate* circuit, which measures the accumulation work but charges it to
//! its own (small) domain.  Here the check instead lives in the query
//! circuit's own region system: the same `ConstraintSystem` gains the binding
//! advice columns and gates alongside every gate the query already has, and a
//! single proof establishes both the query result and that the witnessed input
//! columns equal the published commitments.
//!
//! That makes the marginal cost measurable the honest way: prove the query
//! circuit, prove the query circuit *plus* the inlined check, and subtract.
//! Both run at the same degree over the same witness, so the difference is
//! exactly what the binding costs inside the query.
//!
//! The check itself is the Horner accumulation from `column_commit`:
//! `f_j(x) = sum_i col_j[i] * x^i` is evaluated in-circuit and exposed on a
//! second instance column, to be matched against the IPA openings of the
//! published per-column commitments.

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
// Query circuits with the check inlined
// ---------------------------------------------------------------------------

use halo2_proofs::circuit::SimpleFloorPlanner;
use halo2_proofs::plonk::Circuit;
use std::marker::PhantomData;

/// Generate a wrapper circuit that runs a query chip and the binding gates in
/// ONE constraint system, so a single proof covers both.
macro_rules! bound_query {
    (
        $name:ident, $chipty:ty, $cfgty:ty, $nc:expr,
        fields { $($f:ident : $ty:ty),* $(,)? },
        assign ($chipvar:ident, $lay:ident, $me:ident) $body:block
    ) => {
        pub struct $name {
            $(pub $f: $ty,)*
            /// The query's input columns, in witness order.
            pub columns: Vec<Vec<u64>>,
            /// Fiat-Shamir challenge from the commitments and the query proof.
            pub x: Fp,
        }

        impl Circuit<Fp> for $name {
            type Config = ($cfgty, BindConfig);
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                Self {
                    $($f: Default::default(),)*
                    columns: Vec::new(),
                    x: self.x,
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
                // The query's own gates and the binding gates share ONE
                // constraint system -- this is what "inlined" means.
                let q = <$chipty>::configure(meta);
                let b = configure_bind(meta, $nc);
                (q, b)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fp>,
            ) -> Result<(), Error> {
                let $chipvar = <$chipty>::construct(config.0);
                let $me = self;
                let $lay = &mut layouter;
                let out = $body;
                $chipvar.expose_public($lay, out, 0)?;
                assign_bind($lay, &config.1, &self.columns, self.x)?;
                Ok(())
            }
        }
    };
}

bound_query!(
    BoundQ3,
    crate::sql::q3_obj::TestChip<Fp>, crate::sql::q3_obj::TestCircuitConfig<Fp>, 10,
    fields {
        customer: Vec<Vec<u64>>, orders: Vec<Vec<u64>>, lineitem: Vec<Vec<u64>>,
        condition: [u64; 2],
    },
    assign (chip, lay, me) {
        chip.assign(lay, me.customer.clone(), me.orders.clone(), me.lineitem.clone(), me.condition)?
    }
);

bound_query!(
    BoundQ5,
    crate::sql::q5_obj::Q5Chip<Fp>, crate::sql::q5_obj::Q5Config<Fp>, 16,
    fields {
        customer: Vec<Vec<u64>>, orders: Vec<Vec<u64>>, lineitem: Vec<Vec<u64>>,
        supplier: Vec<Vec<u64>>, nation: Vec<Vec<u64>>, region: Vec<Vec<u64>>,
        europe_hash: u64, start_ts: u64, end_ts: u64,
        nr_pad_extra: usize, co_pad_extra: usize, ls_pad_extra: usize,
    },
    assign (chip, lay, me) {
        chip.assign(
            lay, me.customer.clone(), me.orders.clone(), me.lineitem.clone(),
            me.supplier.clone(), me.nation.clone(), me.region.clone(),
            me.europe_hash, me.start_ts, me.end_ts,
            me.nr_pad_extra, me.co_pad_extra, me.ls_pad_extra,
        )?
    }
);

bound_query!(
    BoundQ8,
    crate::sql::q8_obj::TestChip<Fp>, crate::sql::q8_obj::TestCircuitConfig<Fp>, 19,
    fields {
        region: Vec<Vec<u64>>, nation: Vec<Vec<u64>>, customer: Vec<Vec<u64>>,
        orders: Vec<Vec<u64>>, part: Vec<Vec<u64>>, supplier: Vec<Vec<u64>>,
        lineitem: Vec<Vec<u64>>,
        cond_nation_hash: u64, const_region_name_hash: u64, const_part_type_hash: u64,
    },
    assign (chip, lay, me) {
        chip.assign(
            lay, me.region.clone(), me.nation.clone(), me.customer.clone(),
            me.orders.clone(), me.part.clone(), me.supplier.clone(), me.lineitem.clone(),
            me.cond_nation_hash, me.const_region_name_hash, me.const_part_type_hash,
        )?
    }
);

bound_query!(
    BoundQ9,
    crate::sql::q9_obj::TestChip<Fp>, crate::sql::q9_obj::TestCircuitConfig<Fp>, 16,
    fields {
        part: Vec<Vec<u64>>, supplier: Vec<Vec<u64>>, nation: Vec<Vec<u64>>,
        orders: Vec<Vec<u64>>, partsupp: Vec<Vec<u64>>, lineitem: Vec<Vec<u64>>,
        cond_hash: u64,
    },
    assign (chip, lay, me) {
        chip.assign(
            lay, me.part.clone(), me.supplier.clone(), me.nation.clone(),
            me.orders.clone(), me.partsupp.clone(), me.lineitem.clone(), me.cond_hash,
        )?
    }
);

bound_query!(
    BoundQ18,
    crate::sql::q18_obj::Q18Chip<Fp>, crate::sql::q18_obj::Q18Config<Fp>, 8,
    fields {
        customer: Vec<Vec<u64>>, orders: Vec<Vec<u64>>, lineitem: Vec<Vec<u64>>,
        threshold: u64,
    },
    assign (chip, lay, me) {
        chip.assign(lay, me.customer.clone(), me.orders.clone(), me.lineitem.clone(), me.threshold)?
    }
);

// Graph queries: the committed input is Edge(src, dst) -- 2 columns.

use crate::data::graph_data_processing::Edge;

macro_rules! bound_graph {
    ($name:ident, $chipty:ty, $cfgty:ty, $configure:expr, $assign:expr) => {
        pub struct $name {
            pub edges: Vec<Edge>,
            pub bag1_pad_extra: usize,
            pub bag2_pad_extra: usize,
            pub columns: Vec<Vec<u64>>,
            pub x: Fp,
        }

        impl Circuit<Fp> for $name {
            type Config = ($cfgty, BindConfig);
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                Self {
                    edges: Vec::new(),
                    bag1_pad_extra: self.bag1_pad_extra,
                    bag2_pad_extra: self.bag2_pad_extra,
                    columns: Vec::new(),
                    x: self.x,
                }
            }

            fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
                // MUST match the original circuit's configure exactly. For
                // GQ1/GQ2 the chip's own `configure` is NOT sufficient: the
                // circuit adds further lookup arguments on top of it, and
                // omitting them would prove a strictly weaker statement (and
                // measure as *negative* overhead).
                let q = $configure(meta);
                let b = configure_bind(meta, 2);
                (q, b)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fp>,
            ) -> Result<(), Error> {
                let chip = <$chipty>::construct(config.0);
                let lay = &mut layouter;
                #[allow(clippy::redundant_closure_call)]
                let out = ($assign)(&chip, lay, self)?;
                chip.expose_public(lay, out, 0)?;
                assign_bind(lay, &config.1, &self.columns, self.x)?;
                Ok(())
            }
        }
    };
}

bound_graph!(
    BoundGq1,
    crate::graph_sql::g_sql1_obj::Path3OrdChip<Fp>,
    crate::graph_sql::g_sql1_obj::Path3OrdConfig<Fp>,
    crate::graph_sql::g_sql1_obj::configure_path3ord_full::<Fp>,
    |chip: &crate::graph_sql::g_sql1_obj::Path3OrdChip<Fp>, lay: &mut _, me: &BoundGq1| chip
        .assign(lay, &me.edges)
);

bound_graph!(
    BoundGq2,
    crate::graph_sql::g_sql2_obj::GraphPath4OrderChip<Fp>,
    crate::graph_sql::g_sql2_obj::GraphPath4OrderConfig<Fp>,
    crate::graph_sql::g_sql2_obj::configure_path4order_full::<Fp>,
    |chip: &crate::graph_sql::g_sql2_obj::GraphPath4OrderChip<Fp>, lay: &mut _, me: &BoundGq2| chip
        .assign(lay, &me.edges)
);

bound_graph!(
    BoundGq3,
    crate::graph_sql::g_sql3_obj::TrianglePathCloserChip<Fp>,
    crate::graph_sql::g_sql3_obj::TrianglePathCloserConfig<Fp>,
    <crate::graph_sql::g_sql3_obj::TrianglePathCloserChip<Fp>>::configure,
    |chip: &crate::graph_sql::g_sql3_obj::TrianglePathCloserChip<Fp>, lay: &mut _, me: &BoundGq3| {
        chip.assign(lay, me.edges.clone(), me.bag1_pad_extra, me.bag2_pad_extra)
    }
);

bound_graph!(
    BoundGq4,
    crate::graph_sql::g_sql4_obj::Cycle4OrderedChip<Fp>,
    crate::graph_sql::g_sql4_obj::Cycle4OrderedConfig<Fp>,
    <crate::graph_sql::g_sql4_obj::Cycle4OrderedChip<Fp>>::configure,
    |chip: &crate::graph_sql::g_sql4_obj::Cycle4OrderedChip<Fp>, lay: &mut _, me: &BoundGq4| {
        chip.assign(lay, me.edges.clone(), me.bag1_pad_extra, me.bag2_pad_extra)
    }
);

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
        "gq4" => prove_one(
            params,
            &g_sql4_obj::MyCircuit::<Fp> {
                edges: edges.to_vec(),
                bag1_pad_extra: pads.0,
                bag2_pad_extra: pads.1,
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
            &BoundGq1 {
                edges: edges.to_vec(),
                bag1_pad_extra: 0,
                bag2_pad_extra: 0,
                columns,
                x,
            },
            inst,
        ),
        "gq2" => prove_one(
            params,
            &BoundGq2 {
                edges: edges.to_vec(),
                bag1_pad_extra: 0,
                bag2_pad_extra: 0,
                columns,
                x,
            },
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
                bag1_pad_extra: pads.0,
                bag2_pad_extra: pads.1,
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
    let pk_b = keygen_for(params, base);
    let pk_d = keygen_for(params, bound);

    // Discarded warm-up of both circuits.
    let _ = prove_with(params, &pk_b, base, base_inst);
    let _ = prove_with(params, &pk_d, bound, bound_inst);

    let mut out = Vec::with_capacity(reps);
    for i in 0..reps {
        if i % 2 == 0 {
            let b = prove_with(params, &pk_b, base, base_inst).prove_s;
            let d = prove_with(params, &pk_d, bound, bound_inst).prove_s;
            out.push((b, d));
        } else {
            let d = prove_with(params, &pk_d, bound, bound_inst).prove_s;
            let b = prove_with(params, &pk_b, base, base_inst).prove_s;
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
            &BoundGq1 { edges: e, bag1_pad_extra: 0, bag2_pad_extra: 0, columns, x },
            di,
            reps,
        ),
        "gq2" => paired_runs(
            params,
            &g_sql2_obj::GraphPath4OrderCircuit::<Fp> { edges: e.clone(), _marker: PhantomData },
            bi,
            &BoundGq2 { edges: e, bag1_pad_extra: 0, bag2_pad_extra: 0, columns, x },
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
                bag1_pad_extra: pads.0,
                bag2_pad_extra: pads.1,
                _marker: PhantomData,
            },
            bi,
            &BoundGq4 {
                edges: e,
                bag1_pad_extra: pads.0,
                bag2_pad_extra: pads.1,
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
