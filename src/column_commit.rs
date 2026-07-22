//! Per-column database commitments in the query circuits' own domain
//! (Appendix A), with an in-circuit check that a query's witness columns
//! equal the committed data.
//!
//! This supersedes the monolithic single-vector layer of `src/commitment.rs`
//! on every axis that matters here:
//!
//! * **Same parameters as the query circuits.** Every TPC-H circuit in this
//!   repository runs at `k = 16` (`param16`), and the largest table
//!   (`lineitem`, 60,175 rows) fits in `2^16`.  Committing each column over
//!   that same evaluation domain means the commitment layer reuses the
//!   circuits' existing `ParamsIPA` verbatim -- no separate large setup.
//! * **Hiding.** Each column is committed with a fresh random blinder,
//!   `C_j = Commit(col_j; r_j)`.  (Halo2's *fixed*-column commitments use
//!   `Blind::default() = Blind(1)`, a constant, so they are deterministic
//!   functions of the data and therefore NOT hiding -- unsuitable for
//!   publishing a commitment to a private database.)
//! * **1:1 with circuit columns.** A query's witness column corresponds to
//!   exactly one published commitment, so the consistency check needs no
//!   reindexing (in the flattened layout, column `j` of a relation lives at
//!   strided positions and would need a permutation argument).
//! * **Selective + cheap.** A query binds only the columns it actually
//!   reads, and each opening is over `2^16` rather than a database-wide
//!   domain.
//!
//! # Binding mechanism
//!
//! Columns are committed in the **coefficient** basis: column `j` becomes
//! `f_j(X) = sum_i col_j[i] X^i`, and `C_j = Commit(f_j; r_j)` is published
//! once at setup.
//!
//! At query time a public challenge `x` is derived by Fiat--Shamir from the
//! published commitments, the layout, and the query proof.  Then:
//!
//! 1. the prover opens each committed column at `x` with an IPA opening
//!    proof, revealing `v_j = f_j(x)`; and
//! 2. the query circuit itself evaluates its own witness column at `x`
//!    (a Horner accumulation, `O(rows)` constraints per column) and exposes
//!    `v_j` as a public instance value.
//!
//! Both sides therefore claim the same `v_j`.  Since two distinct columns
//! agree at a uniformly random `x` only with probability `deg/|F|`, matching
//! `v_j` proves the circuit's witness column equals the committed column,
//! except with negligible probability.  This is exactly the
//! "witness columns equal the committed data" check, and it composes with an
//! unmodified query circuit by adding the accumulator columns alongside the
//! input columns it already has.
//!
//! [`PlainTableCircuit`] (inputs only) and [`BoundTableCircuit`] (inputs plus
//! the binding accumulator) are separate circuits so their costs can be
//! measured against each other; the difference is the additional cost of the
//! commitment layer.

use ff::{Field, PrimeField};
use halo2_proofs::{
    arithmetic::eval_polynomial,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error, Expression, Instance, Selector,
    },
    poly::{
        commitment::{Blind, Params, ParamsProver, MSM},
        ipa::{
            commitment::{create_proof as ipa_open, verify_proof as ipa_verify, ParamsIPA},
            msm::MSMIPA,
        },
        EvaluationDomain, Rotation,
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, Transcript, TranscriptRead,
        TranscriptReadBuffer, TranscriptWrite, TranscriptWriterBuffer,
    },
};
use halo2curves::group::Curve;
use halo2curves::pasta::{vesta, EqAffine, Fp};
use rand_core::RngCore;

/// Published per-column commitments plus the prover-only blinders.
#[derive(Clone, Debug)]
pub struct ColumnCommitments {
    /// One published commitment per column (public).
    pub points: Vec<EqAffine>,
    /// Per-column blinders (PRIVATE -- prover only; publishing them destroys
    /// the hiding property).
    blinders: Vec<Fp>,
    pub k: u32,
    pub rows: usize,
    pub cols: usize,
}

impl ColumnCommitments {
    /// Serialized size of the published commitments, in bytes.
    pub fn published_bytes(&self) -> usize {
        self.points.len() * 32
    }
}

/// Smallest circuit degree holding `rows` plus Halo2's blinding rows.
pub fn min_k_rows(rows: usize) -> u32 {
    let mut k = 4u32;
    while (1usize << k) < rows + 8 {
        k += 1;
    }
    k
}

/// Build the coefficient polynomial of one column VECTOR (zero-padded).
fn vec_poly(
    domain: &EvaluationDomain<Fp>,
    col: &[u64],
    k: u32,
) -> halo2_proofs::poly::Polynomial<Fp, halo2_proofs::poly::Coeff> {
    let n = 1usize << k;
    let mut coeffs = vec![Fp::ZERO; n];
    for (i, v) in col.iter().enumerate() {
        coeffs[i] = Fp::from(*v);
    }
    domain.coeff_from_vec(coeffs)
}

/// Commit a set of column vectors (possibly from different tables, of
/// different lengths) with fresh random blinders.  This is the per-QUERY
/// entry point: pass exactly the columns the query reads.
pub fn commit_column_vectors(
    params: &ParamsIPA<vesta::Affine>,
    cols: &[Vec<u64>],
    k: u32,
    mut rng: impl RngCore,
) -> ColumnCommitments {
    assert_eq!(params.n(), 1u64 << k, "params.k must equal k");
    let domain = EvaluationDomain::<Fp>::new(1, k);
    let mut points = Vec::with_capacity(cols.len());
    let mut blinders = Vec::with_capacity(cols.len());
    for col in cols {
        assert!(col.len() <= 1usize << k, "column longer than domain");
        let poly = vec_poly(&domain, col, k);
        let r = Fp::random(&mut rng);
        points.push(params.commit(&poly, Blind(r)).to_affine());
        blinders.push(r);
    }
    ColumnCommitments {
        points,
        blinders,
        k,
        rows: cols.iter().map(|c| c.len()).max().unwrap_or(0),
        cols: cols.len(),
    }
}

/// Evaluations `v_j = f_j(x)` of a set of column vectors.
pub fn vector_evaluations(cols: &[Vec<u64>], k: u32, x: Fp) -> Vec<Fp> {
    let domain = EvaluationDomain::<Fp>::new(1, k);
    cols.iter()
        .map(|c| eval_polynomial(&vec_poly(&domain, c, k), x))
        .collect()
}

/// IPA opening proofs of committed column vectors at `x`.
pub fn open_column_vectors(
    params: &ParamsIPA<vesta::Affine>,
    commitments: &ColumnCommitments,
    cols: &[Vec<u64>],
    x: Fp,
    mut rng: impl RngCore,
) -> Vec<Vec<u8>> {
    let domain = EvaluationDomain::<Fp>::new(1, commitments.k);
    cols.iter()
        .enumerate()
        .map(|(j, col)| {
            let poly = vec_poly(&domain, col, commitments.k);
            let v = eval_polynomial(&poly, x);
            let mut transcript =
                Blake2bWrite::<Vec<u8>, EqAffine, Challenge255<EqAffine>>::init(vec![]);
            transcript.write_scalar(v).unwrap();
            ipa_open(
                params,
                &mut rng,
                &mut transcript,
                &poly,
                Blind(commitments.blinders[j]),
                x,
            )
            .unwrap();
            transcript.finalize()
        })
        .collect()
}

fn column_poly(
    domain: &EvaluationDomain<Fp>,
    table: &[Vec<u64>],
    j: usize,
    k: u32,
) -> halo2_proofs::poly::Polynomial<Fp, halo2_proofs::poly::Coeff> {
    let n = 1usize << k;
    let mut coeffs = vec![Fp::ZERO; n];
    for (i, row) in table.iter().enumerate() {
        coeffs[i] = Fp::from(row.get(j).copied().unwrap_or(0));
    }
    domain.coeff_from_vec(coeffs)
}

/// Commit every column of `table` with a fresh random blinder (hiding).
///
/// `params.k` must equal `k`; the query circuits' own parameters can be used
/// directly when their degree matches (or downsized to it).
pub fn commit_columns(
    params: &ParamsIPA<vesta::Affine>,
    table: &[Vec<u64>],
    k: u32,
    mut rng: impl RngCore,
) -> ColumnCommitments {
    assert_eq!(params.n(), 1u64 << k, "params.k must equal k");
    let cols = table.first().map(|r| r.len()).unwrap_or(0);
    let domain = EvaluationDomain::<Fp>::new(1, k);
    let mut points = Vec::with_capacity(cols);
    let mut blinders = Vec::with_capacity(cols);
    for j in 0..cols {
        let poly = column_poly(&domain, table, j, k);
        let r = Fp::random(&mut rng);
        points.push(params.commit(&poly, Blind(r)).to_affine());
        blinders.push(r);
    }
    ColumnCommitments {
        points,
        blinders,
        k,
        rows: table.len(),
        cols,
    }
}

/// Derive the public binding challenge `x` by Fiat--Shamir from the published
/// commitments, the table shape, and the query proof.
pub fn binding_challenge(
    commitments: &ColumnCommitments,
    query_proof: &[u8],
) -> Fp {
    let mut transcript =
        Blake2bWrite::<Vec<u8>, EqAffine, Challenge255<EqAffine>>::init(vec![]);
    for p in &commitments.points {
        transcript.common_point(*p).unwrap();
    }
    transcript
        .common_scalar(Fp::from(commitments.rows as u64))
        .unwrap();
    transcript
        .common_scalar(Fp::from(commitments.cols as u64))
        .unwrap();
    for chunk in query_proof.chunks(16) {
        let mut buf = [0u8; 16];
        buf[..chunk.len()].copy_from_slice(chunk);
        transcript
            .common_scalar(Fp::from_u128(u128::from_le_bytes(buf)))
            .unwrap();
    }
    transcript
        .common_scalar(Fp::from(query_proof.len() as u64))
        .unwrap();
    *transcript.squeeze_challenge_scalar::<()>()
}

/// The claimed evaluations `v_j = f_j(x)` of every committed column.
pub fn column_evaluations(table: &[Vec<u64>], k: u32, x: Fp) -> Vec<Fp> {
    let cols = table.first().map(|r| r.len()).unwrap_or(0);
    let domain = EvaluationDomain::<Fp>::new(1, k);
    (0..cols)
        .map(|j| eval_polynomial(&column_poly(&domain, table, j, k), x))
        .collect()
}

/// Prover side: IPA opening proofs of every committed column at `x`.
pub fn open_columns(
    params: &ParamsIPA<vesta::Affine>,
    commitments: &ColumnCommitments,
    table: &[Vec<u64>],
    x: Fp,
    mut rng: impl RngCore,
) -> Vec<Vec<u8>> {
    let domain = EvaluationDomain::<Fp>::new(1, commitments.k);
    (0..commitments.cols)
        .map(|j| {
            let poly = column_poly(&domain, table, j, commitments.k);
            let v = eval_polynomial(&poly, x);
            let mut transcript =
                Blake2bWrite::<Vec<u8>, EqAffine, Challenge255<EqAffine>>::init(vec![]);
            transcript.write_scalar(v).unwrap();
            ipa_open(
                params,
                &mut rng,
                &mut transcript,
                &poly,
                Blind(commitments.blinders[j]),
                x,
            )
            .unwrap();
            transcript.finalize()
        })
        .collect()
}

/// Verifier side: check every opening against the published commitment, and
/// that the revealed evaluations equal `expected` (the values the circuit
/// exposed as public instance).
pub fn verify_column_openings(
    params: &ParamsIPA<vesta::Affine>,
    points: &[EqAffine],
    x: Fp,
    expected: &[Fp],
    proofs: &[Vec<u8>],
) -> bool {
    if points.len() != proofs.len() || points.len() != expected.len() {
        return false;
    }
    for (j, proof) in proofs.iter().enumerate() {
        let mut transcript =
            Blake2bRead::<&[u8], EqAffine, Challenge255<EqAffine>>::init(&proof[..]);
        let v = match transcript.read_scalar() {
            Ok(v) => v,
            Err(_) => return false,
        };
        if v != expected[j] {
            return false; // circuit's claimed evaluation != committed column's
        }
        let mut msm = MSMIPA::new(params);
        msm.append_term(Fp::ONE, points[j].into());
        match ipa_verify(params, msm, &mut transcript, x, v) {
            Ok(guard) => {
                if !guard.use_challenges().check() {
                    return false;
                }
            }
            Err(_) => return false,
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Circuits
// ---------------------------------------------------------------------------

/// Maximum supported attribute count (TPC-H `lineitem` has 16).
pub const MAX_COLS: usize = 16;

/// Baseline: a query circuit merely *consuming* the table as witness columns,
/// with no binding to any commitment.  Used as the cost reference: the delta
/// against [`BoundTableCircuit`] is the additional cost of the commitment
/// layer.
#[derive(Clone, Debug, Default)]
pub struct PlainTableCircuit {
    pub table: Vec<Vec<u64>>,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct PlainConfig {
    data: Vec<Column<Advice>>,
}

impl Circuit<Fp> for PlainTableCircuit {
    type Config = PlainConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            table: Vec::new(),
            cols: self.cols,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let data: Vec<Column<Advice>> = (0..MAX_COLS).map(|_| meta.advice_column()).collect();
        PlainConfig { data }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "table",
            |mut region| {
                for (i, row) in self.table.iter().enumerate() {
                    for j in 0..MAX_COLS {
                        region.assign_advice(
                            || "cell",
                            config.data[j],
                            i,
                            || Value::known(Fp::from(row.get(j).copied().unwrap_or(0))),
                        )?;
                    }
                }
                Ok(())
            },
        )
    }
}

/// Bound: the same input columns, plus an in-circuit Horner evaluation of
/// every column at the public challenge `x`, exposed as public instance
/// values.  Checking those values against the IPA openings of the published
/// commitments proves the witness columns equal the committed data.
///
/// Instance layout: `[x, v_0, v_1, ..., v_{MAX_COLS-1}]`.
#[derive(Clone, Debug, Default)]
pub struct BoundTableCircuit {
    pub table: Vec<Vec<u64>>,
    pub cols: usize,
    pub x: Fp,
}

#[derive(Clone, Debug)]
pub struct BoundConfig {
    data: Vec<Column<Advice>>,
    acc: Vec<Column<Advice>>,
    pow: Column<Advice>,
    xcol: Column<Advice>,
    instance: Column<Instance>,
    q_first: Selector,
    q_step: Selector,
}

impl Circuit<Fp> for BoundTableCircuit {
    type Config = BoundConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            table: Vec::new(),
            cols: self.cols,
            x: self.x,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let data: Vec<Column<Advice>> = (0..MAX_COLS).map(|_| meta.advice_column()).collect();
        let acc: Vec<Column<Advice>> = (0..MAX_COLS).map(|_| meta.advice_column()).collect();
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

        // Row 0: pow = 1 and acc_j = data_j (the X^0 term).
        meta.create_gate("first row", |m| {
            let q = m.query_selector(q_first);
            let one = Expression::Constant(Fp::ONE);
            let mut cs = vec![q.clone() * (m.query_advice(pow, Rotation::cur()) - one)];
            for j in 0..MAX_COLS {
                cs.push(
                    q.clone()
                        * (m.query_advice(acc[j], Rotation::cur())
                            - m.query_advice(data[j], Rotation::cur())),
                );
            }
            cs
        });

        // Row i -> i+1:  x is constant, pow' = pow * x,
        // acc_j' = acc_j + data_j' * pow'.
        meta.create_gate("horner step", |m| {
            let q = m.query_selector(q_step);
            let x_cur = m.query_advice(xcol, Rotation::cur());
            let x_next = m.query_advice(xcol, Rotation::next());
            let pow_cur = m.query_advice(pow, Rotation::cur());
            let pow_next = m.query_advice(pow, Rotation::next());
            let mut cs = vec![
                q.clone() * (x_next - x_cur.clone()),
                q.clone() * (pow_next.clone() - pow_cur * x_cur),
            ];
            for j in 0..MAX_COLS {
                cs.push(
                    q.clone()
                        * (m.query_advice(acc[j], Rotation::next())
                            - m.query_advice(acc[j], Rotation::cur())
                            - m.query_advice(data[j], Rotation::next()) * pow_next.clone()),
                );
            }
            cs
        });

        BoundConfig {
            data,
            acc,
            pow,
            xcol,
            instance,
            q_first,
            q_step,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let (x_cell, finals) = layouter.assign_region(
            || "table + horner",
            |mut region| {
                let n = self.table.len();
                assert!(n > 0, "empty table");
                let mut pow = Fp::ONE;
                let mut accs = vec![Fp::ZERO; MAX_COLS];
                let mut last_cells = Vec::new();
                let mut x0 = None;
                for i in 0..n {
                    if i == 0 {
                        config.q_first.enable(&mut region, i)?;
                    } else {
                        pow *= self.x;
                    }
                    if i + 1 < n {
                        config.q_step.enable(&mut region, i)?;
                    }
                    region.assign_advice(|| "pow", config.pow, i, || Value::known(pow))?;
                    let xc = region.assign_advice(
                        || "x",
                        config.xcol,
                        i,
                        || Value::known(self.x),
                    )?;
                    if i == 0 {
                        x0 = Some(xc);
                    }
                    let row = &self.table[i];
                    for j in 0..MAX_COLS {
                        let d = Fp::from(row.get(j).copied().unwrap_or(0));
                        region.assign_advice(|| "cell", config.data[j], i, || Value::known(d))?;
                        if i == 0 {
                            accs[j] = d;
                        } else {
                            accs[j] += d * pow;
                        }
                        let cell = region.assign_advice(
                            || "acc",
                            config.acc[j],
                            i,
                            || Value::known(accs[j]),
                        )?;
                        if i + 1 == n {
                            last_cells.push(cell);
                        }
                    }
                }
                Ok((x0.unwrap(), last_cells))
            },
        )?;
        // x is public (instance row 0), and the per-column evaluations are
        // the public outputs the verifier matches against the openings.
        layouter.constrain_instance(x_cell.cell(), config.instance, 0)?;
        for (j, cell) in finals.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, 1 + j)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Per-QUERY circuits, const-generic over the number of bound columns NC.
// A query binds columns drawn from several tables (of different lengths);
// each column has its own Horner accumulator against the same challenge x.
// ---------------------------------------------------------------------------

/// Baseline per-query circuit: consumes the query's input columns as advice,
/// with no binding.  The cost reference for [`BoundColumnsCircuit`].
#[derive(Clone, Debug)]
pub struct PlainColumnsCircuit<const NC: usize> {
    /// Exactly the column vectors the query reads (zero-padded to a common
    /// row count internally).
    pub columns: Vec<Vec<u64>>,
}

impl<const NC: usize> Default for PlainColumnsCircuit<NC> {
    fn default() -> Self {
        Self { columns: Vec::new() }
    }
}

#[derive(Clone, Debug)]
pub struct PlainColumnsConfig {
    data: Vec<Column<Advice>>,
}

impl<const NC: usize> Circuit<Fp> for PlainColumnsCircuit<NC> {
    type Config = PlainColumnsConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let data: Vec<Column<Advice>> = (0..NC).map(|_| meta.advice_column()).collect();
        PlainColumnsConfig { data }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        assert_eq!(self.columns.len(), NC);
        let rows = self.columns.iter().map(|c| c.len()).max().unwrap_or(1);
        layouter.assign_region(
            || "columns",
            |mut region| {
                for i in 0..rows {
                    for (j, col) in self.columns.iter().enumerate() {
                        region.assign_advice(
                            || "cell",
                            config.data[j],
                            i,
                            || Value::known(Fp::from(col.get(i).copied().unwrap_or(0))),
                        )?;
                    }
                }
                Ok(())
            },
        )
    }
}

/// Bound per-query circuit: the same input columns plus one Horner
/// accumulator per column at the public challenge `x`, exposing the
/// evaluations as public outputs.  Instance layout: `[x, v_0, ..., v_{NC-1}]`.
#[derive(Clone, Debug)]
pub struct BoundColumnsCircuit<const NC: usize> {
    pub columns: Vec<Vec<u64>>,
    pub x: Fp,
}

impl<const NC: usize> Default for BoundColumnsCircuit<NC> {
    fn default() -> Self {
        Self { columns: Vec::new(), x: Fp::ZERO }
    }
}

#[derive(Clone, Debug)]
pub struct BoundColumnsConfig {
    data: Vec<Column<Advice>>,
    acc: Vec<Column<Advice>>,
    pow: Column<Advice>,
    xcol: Column<Advice>,
    instance: Column<Instance>,
    q_first: Selector,
    q_step: Selector,
}

impl<const NC: usize> Circuit<Fp> for BoundColumnsCircuit<NC> {
    type Config = BoundColumnsConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            columns: Vec::new(),
            x: self.x,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let data: Vec<Column<Advice>> = (0..NC).map(|_| meta.advice_column()).collect();
        let acc: Vec<Column<Advice>> = (0..NC).map(|_| meta.advice_column()).collect();
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

        meta.create_gate("first row", |m| {
            let q = m.query_selector(q_first);
            let one = Expression::Constant(Fp::ONE);
            let mut cs = vec![q.clone() * (m.query_advice(pow, Rotation::cur()) - one)];
            for j in 0..NC {
                cs.push(
                    q.clone()
                        * (m.query_advice(acc[j], Rotation::cur())
                            - m.query_advice(data[j], Rotation::cur())),
                );
            }
            cs
        });

        meta.create_gate("horner step", |m| {
            let q = m.query_selector(q_step);
            let x_cur = m.query_advice(xcol, Rotation::cur());
            let x_next = m.query_advice(xcol, Rotation::next());
            let pow_cur = m.query_advice(pow, Rotation::cur());
            let pow_next = m.query_advice(pow, Rotation::next());
            let mut cs = vec![
                q.clone() * (x_next - x_cur.clone()),
                q.clone() * (pow_next.clone() - pow_cur * x_cur),
            ];
            for j in 0..NC {
                cs.push(
                    q.clone()
                        * (m.query_advice(acc[j], Rotation::next())
                            - m.query_advice(acc[j], Rotation::cur())
                            - m.query_advice(data[j], Rotation::next()) * pow_next.clone()),
                );
            }
            cs
        });

        BoundColumnsConfig {
            data,
            acc,
            pow,
            xcol,
            instance,
            q_first,
            q_step,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        assert_eq!(self.columns.len(), NC);
        let rows = self.columns.iter().map(|c| c.len()).max().unwrap_or(1).max(1);
        let (x_cell, finals) = layouter.assign_region(
            || "columns + horner",
            |mut region| {
                let mut pow = Fp::ONE;
                let mut accs = vec![Fp::ZERO; NC];
                let mut last_cells = Vec::new();
                let mut x0 = None;
                for i in 0..rows {
                    if i == 0 {
                        config.q_first.enable(&mut region, i)?;
                    } else {
                        pow *= self.x;
                    }
                    if i + 1 < rows {
                        config.q_step.enable(&mut region, i)?;
                    }
                    region.assign_advice(|| "pow", config.pow, i, || Value::known(pow))?;
                    let xc = region.assign_advice(
                        || "x",
                        config.xcol,
                        i,
                        || Value::known(self.x),
                    )?;
                    if i == 0 {
                        x0 = Some(xc);
                    }
                    for j in 0..NC {
                        let d = Fp::from(self.columns[j].get(i).copied().unwrap_or(0));
                        region.assign_advice(|| "cell", config.data[j], i, || Value::known(d))?;
                        if i == 0 {
                            accs[j] = d;
                        } else {
                            accs[j] += d * pow;
                        }
                        let cell = region.assign_advice(
                            || "acc",
                            config.acc[j],
                            i,
                            || Value::known(accs[j]),
                        )?;
                        if i + 1 == rows {
                            last_cells.push(cell);
                        }
                    }
                }
                Ok((x0.unwrap(), last_cells))
            },
        )?;
        layouter.constrain_instance(x_cell.cell(), config.instance, 0)?;
        for (j, cell) in finals.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, 1 + j)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;
    use rand::rngs::OsRng;

    fn tiny_table() -> Vec<Vec<u64>> {
        (0..40u64).map(|i| vec![i, i * 3 + 1, 7, i % 5]).collect()
    }

    #[test]
    fn in_circuit_evaluation_matches_committed_column() {
        let table = tiny_table();
        let k = min_k_rows(table.len());
        let params = ParamsIPA::<vesta::Affine>::new(k);
        let commitments = commit_columns(&params, &table, k, OsRng);

        let x = binding_challenge(&commitments, b"query-proof-pi");
        let expected = column_evaluations(&table, k, x);

        // The circuit's public outputs are exactly those evaluations.
        let circuit = BoundTableCircuit {
            table: table.clone(),
            cols: 4,
            x,
        };
        // instance layout is [x, v_0, ..., v_{MAX_COLS-1}]; columns beyond
        // the table's width are zero-padded, so their accumulators are 0.
        let mut instance = vec![x];
        instance.extend_from_slice(&expected);
        instance.resize(1 + MAX_COLS, Fp::ZERO);
        let prover = MockProver::run(k, &circuit, vec![instance]).unwrap();
        assert_eq!(prover.verify(), Ok(()));

        // And the openings of the published commitments agree with them.
        let proofs = open_columns(&params, &commitments, &table, x, OsRng);
        assert!(verify_column_openings(
            &params,
            &commitments.points,
            x,
            &expected,
            &proofs
        ));
    }

    #[test]
    fn tampered_column_is_rejected() {
        let table = tiny_table();
        let k = min_k_rows(table.len());
        let params = ParamsIPA::<vesta::Affine>::new(k);
        let commitments = commit_columns(&params, &table, k, OsRng);
        let x = binding_challenge(&commitments, b"pi");

        // A prover whose witness differs in one cell produces a different
        // evaluation, so it cannot match the committed column's opening.
        let mut tampered = table.clone();
        tampered[3][1] += 1;
        let bad = column_evaluations(&tampered, k, x);
        let proofs = open_columns(&params, &commitments, &table, x, OsRng);
        assert!(!verify_column_openings(
            &params,
            &commitments.points,
            x,
            &bad,
            &proofs
        ));
    }

    #[test]
    fn per_query_columns_roundtrip() {
        // Columns from "different tables" (different lengths) bound in one
        // circuit, as a query does.
        let cols: Vec<Vec<u64>> = vec![
            (0..40u64).collect(),                 // fact-table column
            (0..40u64).map(|i| i * 3 + 1).collect(),
            (0..7u64).map(|i| 100 + i).collect(), // small dimension column
        ];
        let k = min_k_rows(40);
        let params = ParamsIPA::<vesta::Affine>::new(k);
        let commitments = commit_column_vectors(&params, &cols, k, OsRng);
        let x = binding_challenge(&commitments, b"pi");
        let evals = vector_evaluations(&cols, k, x);

        let circuit = BoundColumnsCircuit::<3> {
            columns: cols.clone(),
            x,
        };
        let mut instance = vec![x];
        instance.extend_from_slice(&evals);
        let prover = MockProver::run(k, &circuit, vec![instance]).unwrap();
        assert_eq!(prover.verify(), Ok(()));

        let proofs = open_column_vectors(&params, &commitments, &cols, x, OsRng);
        assert!(verify_column_openings(
            &params,
            &commitments.points,
            x,
            &evals,
            &proofs
        ));
    }

    #[test]
    fn commitment_point_is_independent_of_k() {
        // The IPA generators are position-indexed and k-independent, so a
        // zero-padded column commits to the SAME point at any k that holds
        // it.  This is what lets every table be committed in the query
        // circuit's domain (k = 16, set by lineitem) without changing the
        // published value -- only the cost of opening it changes.
        let table = tiny_table();
        let small = min_k_rows(table.len());
        let large = small + 3;
        let p_small = ParamsIPA::<vesta::Affine>::new(small);
        let p_large = ParamsIPA::<vesta::Affine>::new(large);

        let r = Fp::from(987654321u64);
        let d_small = EvaluationDomain::<Fp>::new(1, small);
        let d_large = EvaluationDomain::<Fp>::new(1, large);
        let c_small = p_small
            .commit(&column_poly(&d_small, &table, 1, small), Blind(r))
            .to_affine();
        let c_large = p_large
            .commit(&column_poly(&d_large, &table, 1, large), Blind(r))
            .to_affine();
        assert_eq!(c_small, c_large);
    }

    #[test]
    fn commitments_are_hiding() {
        // Two commitments to the SAME table differ, because each uses a fresh
        // random blinder -- unlike Halo2's fixed-column commitments, which use
        // the constant Blind(1) and are therefore reproducible from the data.
        let table = tiny_table();
        let k = min_k_rows(table.len());
        let params = ParamsIPA::<vesta::Affine>::new(k);
        let a = commit_columns(&params, &table, k, OsRng);
        let b = commit_columns(&params, &table, k, OsRng);
        assert_ne!(a.points[0], b.points[0]);
    }
}
