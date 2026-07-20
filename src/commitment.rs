//! Database-commitment layer (Appendix A of the paper).
//!
//! This module is deliberately **additive**: it does not modify any existing
//! query circuit or gate.  It implements the end-to-end commitment workflow so
//! that its cost can be measured separately and *added* to any query's
//! proving/verification time:
//!
//! 1. **Canonicalization** ([`canonicalize`]): the database is serialized into
//!    a single, fixed coefficient vector with a deterministic layout that is
//!    shared across all queries (relations sorted by name, rows in file order,
//!    attributes flattened row-major).  The layout is injective, which is what
//!    the binding property of the commitment requires.
//!
//! 2. **Setup commitment** ([`commit_db`]): a single Pedersen vector
//!    commitment `Commit(D) = sum_i d_i * G_i + r * W` over the Vesta curve,
//!    computed with the *same* IPA parameters/generators used by the query
//!    circuits (`ParamsIPA<EqAffine>`), so no new cryptographic assumption is
//!    introduced.  `Commit(D)` is published once at setup.
//!
//! 3. **Per-query binding** ([`bind_query_proof`] / [`verify_binding`]): for
//!    every query proof `pi`, the prover produces an IPA opening proof of the
//!    committed polynomial `f_D` at a Fiat--Shamir challenge point derived
//!    from `(Commit(D), pi)`.  The verifier checks the opening against the
//!    *published* `Commit(D)`, which yields the across-proof binding to a
//!    stable, externally published commitment that Halo2's internal
//!    within-proof commitments do not provide by themselves.
//!
//! The revealed opening value `v = f_D(x)` is a single random linear
//! combination of the committed data, exactly as described in Appendix A
//! ("claimed evaluations of `f_D(X)` at challenge points").  Masking this
//! evaluation (e.g., by opening a blinded low-degree extension instead) is an
//! orthogonal extension and does not change the costs measured here.
//!
//! Scope: this layer establishes the *across-proof* binding (each query proof
//! is tied, via Fiat--Shamir, to an opening of the one published
//! `Commit(D)`, so it was produced by a party who knows the committed
//! database).  The complementary in-circuit consistency step -- tying the
//! query circuit's witness columns back to `f_D` -- is what the pre-existing
//! prototype `src/data/database_commitment.rs` explores (per-dataset
//! fixed-column commitments checked by an in-circuit lookup); the two
//! modules use different cell encodings and serve different measurements.
//!
//! See `src/bin/commitment_bench.rs` for the cost-measurement harness.

use ff::{Field, PrimeField};
use halo2_proofs::{
    arithmetic::eval_polynomial,
    poly::{
        commitment::{Blind, Params, ParamsProver, MSM},
        ipa::{
            commitment::{create_proof, verify_proof, ParamsIPA},
            msm::MSMIPA,
        },
        EvaluationDomain,
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, Transcript, TranscriptRead,
        TranscriptReadBuffer, TranscriptWrite, TranscriptWriterBuffer,
    },
};
use halo2curves::pasta::{vesta, EqAffine, Fp};
use rand_core::RngCore;

/// Layout metadata of one relation inside the canonical coefficient vector.
#[derive(Clone, Debug)]
pub struct RelationLayout {
    pub name: String,
    /// Index of the relation's first cell in the coefficient vector.
    pub offset: usize,
    pub rows: usize,
    pub cols: usize,
}

/// A database in canonical (committable) form.
#[derive(Clone, Debug)]
pub struct CanonicalDb {
    /// Flattened cells, padded with zeros to length `1 << k`.
    pub coeffs: Vec<Fp>,
    pub layout: Vec<RelationLayout>,
    /// log2 of the padded vector length; must not exceed the params' `k`.
    pub k: u32,
}

/// The published database commitment (a single Vesta point).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DbCommitment {
    pub point: EqAffine,
}

/// A per-query binding proof: an IPA opening of `f_D` at the Fiat--Shamir
/// challenge derived from `(Commit(D), query proof)`.
#[derive(Clone, Debug)]
pub struct DbOpening {
    pub proof: Vec<u8>,
}

/// Smallest `k` such that `total_cells` fit into a length-`2^k` vector.
pub fn min_k(total_cells: usize) -> u32 {
    let mut k = 0u32;
    while (1usize << k) < total_cells.max(1) {
        k += 1;
    }
    k
}

/// Serialize `relations` (name, rows of u64 attribute values) into the
/// canonical coefficient vector.
///
/// The layout is deterministic and shared across queries: relations are
/// sorted by name, and each relation occupies a contiguous row-major block.
/// Every attribute value becomes one field element, so the encoding is
/// injective given the (public) layout.
pub fn canonicalize(relations: &[(String, Vec<Vec<u64>>)], k: u32) -> CanonicalDb {
    let mut sorted: Vec<&(String, Vec<Vec<u64>>)> = relations.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    for pair in sorted.windows(2) {
        assert_ne!(pair[0].0, pair[1].0, "duplicate relation name {}", pair[0].0);
    }

    let mut coeffs: Vec<Fp> = Vec::new();
    let mut layout: Vec<RelationLayout> = Vec::new();
    for (name, rows) in sorted {
        let cols = rows.first().map(|r| r.len()).unwrap_or(0);
        layout.push(RelationLayout {
            name: name.clone(),
            offset: coeffs.len(),
            rows: rows.len(),
            cols,
        });
        for row in rows {
            assert_eq!(row.len(), cols, "ragged rows in relation {}", name);
            for &cell in row {
                coeffs.push(Fp::from(cell));
            }
        }
    }
    let n = 1usize << k;
    assert!(
        coeffs.len() <= n,
        "database has {} cells but the padded domain holds only {} (k = {})",
        coeffs.len(),
        n,
        k
    );
    coeffs.resize(n, Fp::ZERO);
    CanonicalDb { coeffs, layout, k }
}

/// Compute the published Pedersen vector commitment `Commit(D)`.
///
/// `params` must satisfy `params.k == db.k` exactly (the underlying
/// `commit`/`create_proof` calls assert vector length `== params.n`).  Larger
/// query-circuit params can be reused after `Params::downsize(db.k)`, which
/// keeps the same generator prefix from the transparent setup.
pub fn commit_db(params: &ParamsIPA<vesta::Affine>, db: &CanonicalDb, blind: Fp) -> DbCommitment {
    assert_eq!(
        params.n(),
        1u64 << db.k,
        "params.k must equal db.k (downsize larger params first)"
    );
    let domain = EvaluationDomain::<Fp>::new(1, db.k);
    let poly = domain.coeff_from_vec(db.coeffs.clone());
    let point = params.commit(&poly, Blind(blind));
    use halo2curves::group::Curve;
    DbCommitment {
        point: point.to_affine(),
    }
}

/// Absorb the query proof and the public layout into the transcript so that
/// the squeezed opening point is bound to
/// `(Commit(D), layout, query proof)` by Fiat--Shamir.
///
/// Bytes are chunked into 16-byte little-endian words (injective together
/// with the trailing length element).  These are `common_*` absorptions: they
/// enter the transcript hash but are not serialized into the opening proof;
/// the verifier re-absorbs the same public values.  Absorbing the layout ties
/// the binding to the published schema/layout metadata, not only to the
/// flattened coefficient vector.
fn absorb_binding<T: Transcript<EqAffine, Challenge255<EqAffine>>>(
    transcript: &mut T,
    commitment: &DbCommitment,
    layout: &[RelationLayout],
    query_proof: &[u8],
) -> std::io::Result<()> {
    transcript.common_point(commitment.point)?;
    for rel in layout {
        for chunk in rel.name.as_bytes().chunks(16) {
            let mut buf = [0u8; 16];
            buf[..chunk.len()].copy_from_slice(chunk);
            transcript.common_scalar(Fp::from_u128(u128::from_le_bytes(buf)))?;
        }
        transcript.common_scalar(Fp::from(rel.name.len() as u64))?;
        transcript.common_scalar(Fp::from(rel.offset as u64))?;
        transcript.common_scalar(Fp::from(rel.rows as u64))?;
        transcript.common_scalar(Fp::from(rel.cols as u64))?;
    }
    transcript.common_scalar(Fp::from(layout.len() as u64))?;
    for chunk in query_proof.chunks(16) {
        let mut buf = [0u8; 16];
        buf[..chunk.len()].copy_from_slice(chunk);
        transcript.common_scalar(Fp::from_u128(u128::from_le_bytes(buf)))?;
    }
    transcript.common_scalar(Fp::from(query_proof.len() as u64))?;
    Ok(())
}

/// Prover side: bind `query_proof` to the published commitment by opening
/// `f_D` at the Fiat--Shamir challenge point.  `blind` must be the blinding
/// scalar used in [`commit_db`].
pub fn bind_query_proof(
    params: &ParamsIPA<vesta::Affine>,
    db: &CanonicalDb,
    blind: Fp,
    commitment: &DbCommitment,
    query_proof: &[u8],
    mut rng: impl RngCore,
) -> DbOpening {
    assert_eq!(
        params.n(),
        1u64 << db.k,
        "params.k must equal db.k (downsize larger params first)"
    );
    let domain = EvaluationDomain::<Fp>::new(1, db.k);
    let poly = domain.coeff_from_vec(db.coeffs.clone());

    let mut transcript =
        Blake2bWrite::<Vec<u8>, EqAffine, Challenge255<EqAffine>>::init(vec![]);
    absorb_binding(&mut transcript, commitment, &db.layout, query_proof).unwrap();
    let x = *transcript.squeeze_challenge_scalar::<()>();
    let v = eval_polynomial(&poly, x);
    transcript.write_scalar(v).unwrap();
    create_proof(params, &mut rng, &mut transcript, &poly, Blind(blind), x).unwrap();
    DbOpening {
        proof: transcript.finalize(),
    }
}

/// Verifier side: check that `opening` binds `query_proof` to the published
/// `Commit(D)`.  `layout` is the public layout metadata published together
/// with the commitment at setup.  Returns `true` iff the opening verifies.
pub fn verify_binding(
    params: &ParamsIPA<vesta::Affine>,
    commitment: &DbCommitment,
    layout: &[RelationLayout],
    query_proof: &[u8],
    opening: &DbOpening,
) -> bool {
    let mut transcript =
        Blake2bRead::<&[u8], EqAffine, Challenge255<EqAffine>>::init(&opening.proof[..]);
    if absorb_binding(&mut transcript, commitment, layout, query_proof).is_err() {
        return false;
    }
    let x = *transcript.squeeze_challenge_scalar::<()>();
    let v = match transcript.read_scalar() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let mut msm = MSMIPA::new(params);
    msm.append_term(Fp::ONE, commitment.point.into());
    match verify_proof(params, msm, &mut transcript, x, v) {
        Ok(guard) => guard.use_challenges().check(),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    fn tiny_db() -> Vec<(String, Vec<Vec<u64>>)> {
        vec![
            (
                "orders".to_string(),
                (0..40u64).map(|i| vec![i, i * 3 + 1, 7]).collect(),
            ),
            (
                "customer".to_string(),
                (0..25u64).map(|i| vec![i, 1000 + i]).collect(),
            ),
        ]
    }

    #[test]
    fn commit_and_bind_roundtrip() {
        let db = canonicalize(&tiny_db(), 8);
        let params = ParamsIPA::<vesta::Affine>::new(8);
        let blind = Fp::from(123456789u64);
        let commitment = commit_db(&params, &db, blind);

        let query_proof = vec![42u8; 1000]; // stand-in for a query proof
        let opening = bind_query_proof(&params, &db, blind, &commitment, &query_proof, OsRng);
        assert!(verify_binding(&params, &commitment, &db.layout, &query_proof, &opening));
    }

    #[test]
    fn binding_rejects_wrong_query_proof() {
        let db = canonicalize(&tiny_db(), 8);
        let params = ParamsIPA::<vesta::Affine>::new(8);
        let blind = Fp::from(5u64);
        let commitment = commit_db(&params, &db, blind);

        let opening = bind_query_proof(&params, &db, blind, &commitment, b"proof-A", OsRng);
        // The same opening must not verify against a different query proof:
        // the Fiat--Shamir point changes, so the claimed evaluation fails.
        assert!(!verify_binding(&params, &commitment, &db.layout, b"proof-B", &opening));
    }

    #[test]
    fn binding_rejects_wrong_database() {
        let params = ParamsIPA::<vesta::Affine>::new(8);
        let blind = Fp::from(5u64);

        let db = canonicalize(&tiny_db(), 8);
        let commitment = commit_db(&params, &db, blind);

        // A prover who switched to a different database cannot open the
        // published commitment.
        let mut other = tiny_db();
        other[0].1[0][1] += 1;
        let db2 = canonicalize(&other, 8);
        let opening = bind_query_proof(&params, &db2, blind, &commitment, b"pi", OsRng);
        assert!(!verify_binding(&params, &commitment, &db2.layout, b"pi", &opening));
    }

    #[test]
    fn canonical_layout_is_deterministic() {
        let a = canonicalize(&tiny_db(), 8);
        let mut reversed = tiny_db();
        reversed.reverse();
        let b = canonicalize(&reversed, 8);
        assert_eq!(a.coeffs, b.coeffs); // order-independent (sorted by name)
        assert_eq!(a.layout[0].name, "customer");
    }
}
