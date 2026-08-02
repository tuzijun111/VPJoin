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

/// A set of per-column commitments plus the prover-only blinders.
///
/// This is a VIEW, not a publication: the sound way to obtain one is
/// [`DatabaseCommitment::view`], which selects the columns a query reads out of
/// the single commitment published at Setup. Building one directly with
/// [`commit_column_vectors`] draws fresh blinders and so produces a commitment
/// that is only meaningful for the query at hand -- see that function's note.
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

/// `Commit(D)`: the single commitment published during Setup, over EVERY column
/// of the database, fixed before any query is issued.
///
/// This is the difference between binding each proof to one database and merely
/// binding it to itself, and it is why the points and the blinders are drawn
/// once, here. A query binds by SELECTING the columns it reads
/// ([`DatabaseCommitment::view`]), never by committing again, so two queries
/// that read the same column open the same published point under the same
/// blinder. A prover therefore cannot answer one query from `D` and another
/// from `D' != D`.
///
/// Re-committing per query with a fresh blinder gives up exactly that: the
/// published points differ between queries, so nothing relates the two answers
/// to one database, whatever each proof establishes internally.
#[derive(Clone, Debug)]
pub struct DatabaseCommitment {
    /// One published commitment per column of `D`, in canonical layout order.
    /// Public, and fixed for the lifetime of the database.
    points: Vec<EqAffine>,
    /// Per-column blinders (PRIVATE). Fixed together with the points: a query
    /// reuses the blinder of every column it opens instead of drawing one.
    blinders: Vec<Fp>,
    /// One domain for the whole database, so a column's commitment does not
    /// depend on which query happens to read it.
    k: u32,
    /// Committed length of each column, kept for the published record.
    lens: Vec<usize>,
}

impl DatabaseCommitment {
    pub fn points(&self) -> &[EqAffine] {
        &self.points
    }
    pub fn cols(&self) -> usize {
        self.points.len()
    }
    pub fn k(&self) -> u32 {
        self.k
    }
    /// Bytes published ONCE at Setup, for the whole database. This is not a
    /// per-query cost: a query publishes no commitment of its own.
    pub fn published_bytes(&self) -> usize {
        self.points.len() * 32
    }

    /// The columns query `Q` reads, as a view on the published commitment.
    ///
    /// Selects; never re-commits. The returned [`ColumnCommitments`] carries the
    /// published points and their original blinders, so every downstream step
    /// ([`open_column_vectors`], [`verify_column_openings`]) opens the
    /// commitment that was fixed before any query ran.
    pub fn view(&self, idx: &[usize]) -> ColumnCommitments {
        let mut points = Vec::with_capacity(idx.len());
        let mut blinders = Vec::with_capacity(idx.len());
        let mut rows = 0usize;
        for &j in idx {
            assert!(
                j < self.points.len(),
                "column {} is outside the published database commitment ({} columns)",
                j,
                self.points.len()
            );
            points.push(self.points[j]);
            blinders.push(self.blinders[j]);
            rows = rows.max(self.lens[j]);
        }
        ColumnCommitments {
            points,
            blinders,
            k: self.k,
            rows,
            cols: idx.len(),
        }
    }
}

/// Publish `Commit(D)`: one hiding commitment per column of the WHOLE database,
/// drawn once, before any query is issued.
///
/// `cols` is the canonical column layout of `D`, so the index of a column here
/// is the index a query passes to [`DatabaseCommitment::view`]. Every column
/// shares one domain `k`, which must therefore hold the longest column.
pub fn commit_database(
    params: &ParamsIPA<vesta::Affine>,
    cols: &[Vec<u64>],
    k: u32,
    mut rng: impl RngCore,
) -> DatabaseCommitment {
    assert_eq!(params.n(), 1u64 << k, "params.k must equal k");
    let domain = EvaluationDomain::<Fp>::new(1, k);
    let mut points = Vec::with_capacity(cols.len());
    let mut blinders = Vec::with_capacity(cols.len());
    let mut lens = Vec::with_capacity(cols.len());
    for col in cols {
        assert!(
            col.len() <= 1usize << k,
            "column of length {} does not fit the database domain 2^{}",
            col.len(),
            k
        );
        let poly = vec_poly(&domain, col, k);
        let r = Fp::random(&mut rng);
        points.push(params.commit(&poly, Blind(r)).to_affine());
        blinders.push(r);
        lens.push(col.len());
    }
    DatabaseCommitment {
        points,
        blinders,
        k,
        lens,
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
/// different lengths) with FRESH random blinders.
///
/// COST MEASUREMENT ONLY. Because the blinders are redrawn on every call, two
/// invocations over the same column produce different published points, so a
/// proof bound this way is tied to a commitment created for it rather than to a
/// database fixed in advance. It measures what the binding layer costs, and
/// nothing about cross-query consistency. Use [`commit_database`] once and
/// [`DatabaseCommitment::view`] per query for the construction the paper
/// describes.
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
///
/// ORDERING, which the random-point argument depends on: `x` must be
/// unpredictable at the moment the witness polynomial it tests is fixed. That
/// holds only when `query_proof` is a proof that ALREADY commits that witness.
/// An empty transcript makes `x` a function of the commitments alone, i.e. a
/// constant the prover knows before choosing its witness, and then agreement at
/// `x` says nothing: a prover can pick any `W != D` with `W(x) = D(x)`. See
/// [`binding_challenge_db`], which refuses that case outright.
pub fn binding_challenge(
    commitments: &ColumnCommitments,
    query_proof: &[u8],
) -> Fp {
    assert!(
        !query_proof.is_empty(),
        "binding challenge derived from an empty transcript: x would then be a \
         function of the published commitments alone, i.e. a public constant \
         known before the witness is chosen, and agreement at x would certify \
         nothing. Bind to a real query proof."
    );
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

/// Derive the binding challenge from the WHOLE published `Commit(D)`, the
/// columns this query opens, and the query proof.
///
/// Absorbing every published point, not only the subset the query reads, is
/// what ties the challenge to one database: a prover who republished the
/// database would get a different `x` even for a query whose own columns were
/// unchanged, so the challenge cannot be reused across two publications.
///
/// Panics if `query_proof` is empty. That is not defensive tidiness: `x` has to
/// be unpredictable when the witness polynomial is fixed, so it must come from
/// a proof that already commits that witness. Deriving it from nothing yields a
/// public constant, and a prover can then choose any `W != D` agreeing with `D`
/// at that one point.
pub fn binding_challenge_db(db: &DatabaseCommitment, idx: &[usize], query_proof: &[u8]) -> Fp {
    assert!(
        !query_proof.is_empty(),
        "the binding challenge must come from a query proof that already commits \
         the witness being bound; an empty transcript makes x a public constant \
         the prover can choose its witness against"
    );
    let mut transcript = Blake2bWrite::<Vec<u8>, EqAffine, Challenge255<EqAffine>>::init(vec![]);
    // (1) the whole publication, in canonical order
    for p in &db.points {
        transcript.common_point(*p).unwrap();
    }
    transcript
        .common_scalar(Fp::from(db.points.len() as u64))
        .unwrap();
    transcript.common_scalar(Fp::from(db.k as u64)).unwrap();
    // (2) which columns this query opens
    for &j in idx {
        transcript.common_scalar(Fp::from(j as u64)).unwrap();
    }
    transcript
        .common_scalar(Fp::from(idx.len() as u64))
        .unwrap();
    // (3) the query proof, which is what fixes the witness before x exists
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

/// Derive the binding challenge WITHOUT a prior proof, from the publication,
/// the columns this query opens, and a query-identifying context.
///
/// ORDERING, stated plainly. `x` is then a public value the prover can compute
/// before it commits the tied proof's witness, so the random-point argument
/// does not hold on its own terms. That is NOT a regression from deriving `x`
/// from a separate base proof: nothing constrains that proof's witness to equal
/// the tied proof's, so it too left the prover free to choose its witness after
/// learning `x`. The base proof was Fiat--Shamir entropy, never a binding, and
/// charging a whole extra query proof for it obscured the gap rather than
/// closing it.
///
/// Closing it requires a challenge drawn from the tied proof's OWN first-phase
/// commitments, which this backend cannot express -- see the note on
/// [`BoundColumnsCircuit`] for why the evaluations could then no longer be
/// instance values.
pub fn binding_challenge_public(db: &DatabaseCommitment, idx: &[usize], context: &[u8]) -> Fp {
    let mut transcript = Blake2bWrite::<Vec<u8>, EqAffine, Challenge255<EqAffine>>::init(vec![]);
    for p in &db.points {
        transcript.common_point(*p).unwrap();
    }
    transcript
        .common_scalar(Fp::from(db.points.len() as u64))
        .unwrap();
    transcript.common_scalar(Fp::from(db.k as u64)).unwrap();
    for &j in idx {
        transcript.common_scalar(Fp::from(j as u64)).unwrap();
    }
    transcript
        .common_scalar(Fp::from(idx.len() as u64))
        .unwrap();
    for chunk in context.chunks(16) {
        let mut buf = [0u8; 16];
        buf[..chunk.len()].copy_from_slice(chunk);
        transcript
            .common_scalar(Fp::from_u128(u128::from_le_bytes(buf)))
            .unwrap();
    }
    transcript
        .common_scalar(Fp::from(context.len() as u64))
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
///
/// KNOWN GAP, and why the obvious repair does not work. `x` arrives as an
/// instance, so it is fixed before this proof commits its witness, and the
/// random-point argument wants the opposite order. The natural fix is to make
/// `x` a second-phase challenge (`ConstraintSystem::challenge_usable_after`,
/// which this backend does provide) so that it is squeezed from the first-phase
/// advice commitments. That cannot be completed here:
///
///  * the evaluations `v_j` would then depend on `x`, so they could no longer be
///    instance values -- `create_proof` commits and absorbs the whole instance
///    before it enters the phase loop, i.e. before any advice commitment exists
///    and before any phase challenge is squeezed;
///  * no API hands a squeezed challenge back to the caller, and the opening of
///    the published column has to happen at that same `x`, outside the circuit;
///  * binding against Halo2's own multiopen point instead (which is correctly
///    ordered, after every advice commitment) fails on the blinding rows: an
///    advice polynomial carries fresh randomness in its unusable rows, so it
///    does not agree with the committed column polynomial at any point.
///
/// Closing this needs a commit-and-prove linkage between the published
/// commitment and the circuit's witness, not a rearrangement of this circuit.
/// [`binding_challenge_db`] enforces the half that is enforceable: the challenge
/// must come from a proof that has already committed a witness.
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

    /// Columns of one database, as `commit_database` takes them.
    fn tiny_db() -> Vec<Vec<u64>> {
        vec![
            (0..40u64).collect(),
            (0..40u64).map(|i| i * 3 + 1).collect(),
            (0..40u64).map(|i| i % 5).collect(),
            (0..25u64).map(|i| i * i).collect(),
        ]
    }

    /// THE cross-query property, and the one a per-query commitment does not
    /// have: two queries reading the same column must open the SAME published
    /// point. Otherwise nothing stops a prover answering one from `D` and the
    /// other from `D' != D`.
    #[test]
    fn two_queries_share_one_published_commitment() {
        let cols = tiny_db();
        let k = min_k_rows(cols.iter().map(|c| c.len()).max().unwrap());
        let params = ParamsIPA::<vesta::Affine>::new(k);
        let db = commit_database(&params, &cols, k, OsRng);

        // Q_a reads columns {0, 2}; Q_b reads {2, 3}. Column 2 is shared, and
        // it sits at a DIFFERENT offset in each view (last in `a`, first in
        // `b`): a view is indexed by the query's own column order, so the
        // published identity has to be compared through the database index.
        let a = db.view(&[0, 2]);
        let b = db.view(&[2, 3]);
        assert_eq!(
            a.points[1], b.points[0],
            "the shared column must open to the same published point in both queries"
        );
        assert_ne!(
            a.points[0], b.points[1],
            "distinct columns must stay distinct"
        );

        // Re-committing per query is exactly what loses that.
        let ca = commit_column_vectors(&params, &[cols[2].clone()], k, OsRng);
        let cb = commit_column_vectors(&params, &[cols[2].clone()], k, OsRng);
        assert_ne!(
            ca.points[0], cb.points[0],
            "fresh blinders must give different points -- this is the behaviour \
             `commit_database` exists to replace, so if it ever stops holding the \
             contrast this test documents is gone"
        );
    }

    /// A view opens against the published commitment, with the blinder that was
    /// drawn at Setup rather than a fresh one.
    #[test]
    fn a_view_opens_against_the_published_commitment() {
        let cols = tiny_db();
        let k = min_k_rows(cols.iter().map(|c| c.len()).max().unwrap());
        let params = ParamsIPA::<vesta::Affine>::new(k);
        let db = commit_database(&params, &cols, k, OsRng);

        let idx = [1usize, 3];
        let view = db.view(&idx);
        let picked: Vec<Vec<u64>> = idx.iter().map(|&j| cols[j].clone()).collect();

        let x = binding_challenge_db(&db, &idx, b"a query proof that commits the witness");
        let evals = vector_evaluations(&picked, k, x);
        let openings = open_column_vectors(&params, &view, &picked, x, OsRng);
        assert!(
            verify_column_openings(&params, &view.points, x, &evals, &openings),
            "openings must verify against the points published at Setup"
        );
    }

    /// The challenge is a function of the WHOLE publication, so a prover who
    /// republishes the database cannot carry a challenge over to it.
    #[test]
    fn the_challenge_covers_the_whole_publication() {
        let cols = tiny_db();
        let k = min_k_rows(cols.iter().map(|c| c.len()).max().unwrap());
        let params = ParamsIPA::<vesta::Affine>::new(k);
        let proof = b"a query proof that commits the witness";

        let db1 = commit_database(&params, &cols, k, OsRng);
        // A different database, differing ONLY in a column this query never reads.
        let mut other = cols.clone();
        other[3][0] += 1;
        let db2 = commit_database(&params, &other, k, OsRng);

        let idx = [0usize, 1];
        assert_ne!(
            binding_challenge_db(&db1, &idx, proof),
            binding_challenge_db(&db2, &idx, proof),
            "the challenge must depend on every published column, not just the \
             ones this query opens"
        );
    }

    /// An empty transcript would make `x` a constant the prover knows before it
    /// fixes its witness, which is precisely when the random-point argument
    /// stops holding.
    #[test]
    #[should_panic(expected = "already commits the witness")]
    fn a_challenge_from_nothing_is_refused() {
        let cols = tiny_db();
        let k = min_k_rows(cols.iter().map(|c| c.len()).max().unwrap());
        let params = ParamsIPA::<vesta::Affine>::new(k);
        let db = commit_database(&params, &cols, k, OsRng);
        let _ = binding_challenge_db(&db, &[0], &[]);
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
