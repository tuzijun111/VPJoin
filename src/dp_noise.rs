//! One-sided DP noise for intra-cluster padding capacities (Appendix G.2).
//!
//! Rust twin of `dp/noise_generator.py` (the documented reference
//! implementation), provided so the DP-guided padding mode can obtain its
//! capacities directly at witness-generation time: call [`dp_join_capacity`]
//! for each intra-cluster (bag) join and use the returned `capacity` to size
//! that bag's witness columns, in place of a hard-coded constant.  The module
//! is additive: no existing circuit is modified; a circuit adopts it simply
//! by taking the returned capacity as its size parameter.
//!
//! The mechanism in use is [`dp_join_capacity_public_tau`]: ONE release per
//! bag, with noise calibrated to a PUBLIC upper bound on the join size's
//! global sensitivity, supplied by the caller's declared DP policy (Q5:
//! P = {lineitem}, sensitivity 1; graph queries: a per-dataset degree cap).
//!
//! Sensitivity.  For `A |x| B` with `f = sum_k freq_A(k) freq_B(k)`, under
//! the standard neighboring definition (change one tuple of ONE relation), a
//! change to A moves the count by at most `Freq(B)` and a change to B by at
//! most `Freq(A)`.  For SELF-joins the aliased instances are views of the
//! SAME stored relation, so one stored-tuple change perturbs both sides at
//! once and the contributions ADD: for the wedge count
//! `sum_v indeg(v) * outdeg(v)`, inserting one edge (u, v) changes the count
//! by `outdeg(v) + indeg(u)`, i.e. up to 2x the degree cap.  Callers must
//! pass the summed bound (see `graph_pads`).
//!
//! Every released value is a single draw and the noise is always >= 0, so
//! the released capacity never underestimates the true cardinality and no
//! valid tuple is dropped.  When one query makes several releases over the
//! same private relation, split the budget across them by basic composition
//! before calling this.
//!
//! [`dp_join_capacity`] (two-stage: noisy frequency thresholds, then a size
//! release scaled by them) is retained for comparison tests only; its two
//! location shifts multiply, giving a capacity quadratic in `1/epsilon`.
//!
//! The unit mechanism is the clamped continuous one-sided Laplace with
//! location `mu = 1 + ln(1/(2 delta))/eps` and scale `1/eps`, which is
//! exactly `(eps, delta)`-DP for a sensitivity-1 query (gap mass
//! `Pr[Z <= 1] = delta`, clamped atom `Pr[Z <= 0] = delta e^{-eps}`);
//! arbitrary sensitivity follows by rescaling (post-processing).

use rand::Rng;

/// A DP-released padding capacity together with the released thresholds.
/// `capacity`, `tau_a`, `tau_b` are jointly `(epsilon, delta)`-DP; nothing
/// else about the join may be published.
#[derive(Clone, Debug)]
pub struct DpCapacity {
    /// Integer upper bound on the true join size (>= true size, always).
    pub capacity: u64,
    pub tau_a: f64,
    pub tau_b: f64,
    /// The stage-2 join-size sensitivity used: `max(tau_a, tau_b)`.
    pub sensitivity: f64,
    /// Recorded for labeling only; does not change the sensitivity (the
    /// per-instance neighboring model gives `max` for self-joins too).
    pub self_join: bool,
}

fn unit_location(epsilon: f64, delta: f64) -> f64 {
    assert!(epsilon > 0.0, "epsilon must be positive");
    assert!(delta > 0.0 && delta < 0.5, "delta must lie in (0, 1/2)");
    1.0 + (1.0 / (2.0 * delta)).ln() / epsilon
}

/// One draw of non-negative `(epsilon, delta)`-DP one-sided Laplace noise for
/// a query with the given global sensitivity.
pub fn one_sided_laplace(
    epsilon: f64,
    delta: f64,
    sensitivity: f64,
    rng: &mut impl Rng,
) -> f64 {
    assert!(sensitivity > 0.0, "sensitivity must be positive");
    let mu = unit_location(epsilon, delta);
    let b = 1.0 / epsilon;
    // Inverse-CDF sampling of Laplace(mu, b).
    let u: f64 = rng.gen_range(-0.5..0.5);
    let z = mu - b * u.signum() * (1.0 - 2.0 * u.abs()).ln();
    sensitivity * z.max(0.0)
}

/// Single-release capacity for one intra-cluster join `A |x| B`, calibrated
/// to a PUBLIC bound on the maximum join-key frequency.
///
/// This is the cheaper alternative to [`dp_join_capacity`].  That function
/// certifies the private frequencies with their own one-sided releases, so
/// the budget splits three ways and the two shifts MULTIPLY:
///
/// ```text
///   dp_join_capacity:  capacity ~ m + (freq + mu_i) * mu_i  = m + freq*mu_i + mu_i^2
///   this function:     capacity ~ m + tau * mu
/// ```
///
/// with `mu = 1 + ln(1/(2 delta))/epsilon`.  The `mu_i^2` term is
/// data-independent and quadratic in `1/epsilon`; at the paper's default
/// budget it dominates everything else, which is why the two-stage variant
/// pads a 2,287-row bag to over half a million rows.  Spending the whole
/// budget on ONE release makes the cost linear in both `tau` and `1/epsilon`.
///
/// The price is that `tau` must be PUBLIC: an upper bound on the join size's
/// global sensitivity that is fixed before the data is seen.  Where it comes
/// from is the caller's declared DP policy: when the private set P contains
/// only a fact relation whose tuples each contribute at most one row (Q5
/// under P = {lineitem}), tau = 1 exactly; for the graph self-joins it is
/// derived from a declared per-dataset degree cap.  A declared cap is
/// strictly weaker than a key/PK-FK constraint (it bounds multiplicity
/// without pinning the join size), but it IS an assumption -- if the data
/// violates it the release can under-provision, so callers assert their cap
/// against the instance.
pub fn dp_join_capacity_public_tau(
    join_size: u64,
    tau: u64,
    epsilon: f64,
    delta: f64,
    rng: &mut impl Rng,
) -> DpCapacity {
    let noise = one_sided_laplace(epsilon, delta, tau.max(1) as f64, rng);
    DpCapacity {
        capacity: join_size + noise.ceil() as u64,
        tau_a: tau as f64,
        tau_b: tau as f64,
        sensitivity: tau.max(1) as f64,
        self_join: false,
    }
}

/// Single-release capacity where the frequency bound `tau` is MEASURED from
/// relations OUTSIDE the protected set P and released exactly.
///
/// [`dp_join_capacity_public_tau`] needs `tau` fixed before the data is
/// seen; [`dp_frequency_bound`] pays a second release to certify a private
/// frequency.  This is the third case: when the join-key fan-out lives
/// entirely in unprotected relations (Q5 under P = {customer, supplier}:
/// the custkey fan-out is in orders, the suppkey fan-out in lineitem), the
/// statistic "max tuples sharing a key" is identical on every neighboring
/// instance, i.e. 0-sensitive w.r.t. P.  Releasing it exactly costs no
/// budget, and `Delta <= tau` holds unconditionally on every neighbor --
/// no truncation, no declared cap, no under-provisioning risk.  The whole
/// budget then goes to the one size release, exactly as in the public-tau
/// variant, to which this delegates.
pub fn dp_join_capacity_unprotected_tau(
    join_size: u64,
    tau: u64,
    epsilon: f64,
    delta: f64,
    rng: &mut impl Rng,
) -> DpCapacity {
    dp_join_capacity_public_tau(join_size, tau, epsilon, delta, rng)
}

/// One-sided DP upper bound on a maximum join-key frequency.
///
/// The statistic "largest number of tuples sharing a join key" has global
/// sensitivity 1: adding or removing one tuple moves any single key's count
/// by at most 1, hence moves the maximum by at most 1.  The noise is
/// one-sided, so the returned bound is `>= max_freq` with certainty and can
/// stand in for the private frequency when calibrating a later release.
///
/// Use this when no PUBLIC bound on the frequency is available.  It costs a
/// second release out of the budget, and the two location shifts then
/// multiply in the final capacity (`(freq + mu) * mu`), so a public bound is
/// cheaper where one can be defended -- see
/// [`dp_join_capacity_public_tau`].
pub fn dp_frequency_bound(
    max_freq: u64,
    epsilon: f64,
    delta: f64,
    rng: &mut impl Rng,
) -> f64 {
    max_freq as f64 + one_sided_laplace(epsilon, delta, 1.0, rng)
}

/// Release a DP padding capacity for one intra-cluster join `A |x| B`.
///
/// * `join_size`   -- true cardinality `m = |A |x| B|` (private).
/// * `max_freq_a`, `max_freq_b` -- true maximum join-key frequencies
///   (private).
/// * `epsilon`, `delta` -- TOTAL budget for this capacity release, split
///   equally across the three internal releases (basic composition).
/// * `self_join`   -- recorded for labeling only; the sensitivity is
///   `max(tau_a, tau_b)` regardless (per-instance neighboring model).
pub fn dp_join_capacity(
    join_size: u64,
    max_freq_a: u64,
    max_freq_b: u64,
    epsilon: f64,
    delta: f64,
    self_join: bool,
    rng: &mut impl Rng,
) -> DpCapacity {
    let (eps_i, del_i) = (epsilon / 3.0, delta / 3.0);

    let tau_a = max_freq_a as f64 + one_sided_laplace(eps_i, del_i, 1.0, rng);
    let tau_b = max_freq_b as f64 + one_sided_laplace(eps_i, del_i, 1.0, rng);
    // Join-size sensitivity: a single-tuple change to A moves the count by
    // <= Freq(B) and to B by <= Freq(A), so the global sensitivity is
    // max(Freq(A), Freq(B)) -> max(tau_a, tau_b) with the released bounds.
    let sens = tau_a.max(tau_b).max(1.0);

    let size_noise = one_sided_laplace(eps_i, del_i, sens, rng);
    DpCapacity {
        capacity: join_size + size_noise.ceil() as u64,
        tau_a,
        tau_b,
        sensitivity: sens,
        self_join,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// The single-release variant must be strictly cheaper than the
    /// two-stage one at the same total budget, and the gap must widen as
    /// epsilon shrinks (the two-stage cost carries a `mu^2` term).
    #[test]
    fn single_release_beats_two_stage_and_gap_grows_as_epsilon_shrinks() {
        let (m, fa, fb, delta) = (2287u64, 10u64, 1u64, 5e-6);
        let mut prev_ratio = 0.0f64;
        for &eps in &[1.0f64, 0.5, 0.1, 0.05] {
            let two: Vec<u64> = (0u64..400)
                .map(|s| {
                    let mut r = StdRng::seed_from_u64(s);
                    dp_join_capacity(m, fa, fb, eps, delta, false, &mut r).capacity
                })
                .collect();
            let one: Vec<u64> = (0u64..400)
                .map(|s| {
                    let mut r = StdRng::seed_from_u64(s);
                    // tau declared public, generously above the true freq
                    dp_join_capacity_public_tau(m, 16, eps, delta, &mut r).capacity
                })
                .collect();
            let med = |mut v: Vec<u64>| {
                v.sort_unstable();
                v[v.len() / 2] as f64
            };
            let (t, o) = (med(two), med(one));
            assert!(o < t, "eps={}: single {} not cheaper than two-stage {}", eps, o, t);
            let ratio = t / o;
            assert!(
                ratio > prev_ratio,
                "eps={}: gap {:.1}x did not widen over {:.1}x",
                eps,
                ratio,
                prev_ratio
            );
            prev_ratio = ratio;
        }
    }

    /// One-sidedness must survive: the capacity never under-provisions.
    #[test]
    fn single_release_never_underprovisions() {
        let mut r = StdRng::seed_from_u64(3);
        for _ in 0..2000 {
            let c = dp_join_capacity_public_tau(63, 1024, 0.1, 1e-5, &mut r).capacity;
            assert!(c >= 63);
        }
    }

    #[test]
    fn noise_is_one_sided() {
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..5000 {
            assert!(one_sided_laplace(0.1, 1e-5, 5.0, &mut rng) >= 0.0);
        }
    }

    #[test]
    fn gap_mass_bounded_by_delta() {
        // Empirically check Pr[Z <= 1] (the distinguishing event of the
        // clamped continuous mechanism) stays around delta.
        let (eps, delta) = (0.5, 1e-3);
        let mu = unit_location(eps, delta);
        let b = 1.0 / eps;
        let mut rng = StdRng::seed_from_u64(11);
        let trials = 400_000;
        let mut hits = 0u64;
        for _ in 0..trials {
            let u: f64 = rng.gen_range(-0.5..0.5);
            let z = mu - b * u.signum() * (1.0 - 2.0 * u.abs()).ln();
            if z <= 1.0 {
                hits += 1;
            }
        }
        let gap = hits as f64 / trials as f64;
        assert!(gap <= delta * 1.5 + 3.0 / (trials as f64).sqrt());
    }

    #[test]
    fn capacity_upper_bounds_true_size() {
        let mut rng = StdRng::seed_from_u64(3);
        for self_join in [false, true] {
            for _ in 0..300 {
                let m = rng.gen_range(0..1_000_000u64);
                let out = dp_join_capacity(m, 7, 4, 0.1, 1e-5, self_join, &mut rng);
                assert!(out.capacity >= m);
                assert!(out.tau_a >= 7.0 && out.tau_b >= 4.0);
                // sensitivity is max(tau_a, tau_b), the same for self-joins.
                assert!((out.sensitivity - out.tau_a.max(out.tau_b)).abs() < 1e-9);
            }
        }
    }
}
