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
//! Mechanism (two stages, matching the paper and the Python reference):
//!  1. release one-sided noisy truncation thresholds `tau_a`, `tau_b` for the
//!     per-relation maximum join-key frequencies (global sensitivity 1 each);
//!  2. release the capacity with one-sided noise calibrated to the released
//!     thresholds: `max(tau_a, tau_b)` for joins of distinct relations, or
//!     `tau_a + tau_b` for SELF-joins (e.g. Edge |x| Edge), where one record
//!     change affects both sides.
//!
//! Every released value is a single draw; the total cost is
//! `(eps1 + eps2 + eps3, delta1 + delta2 + delta3)` by basic composition.
//! The noise is always >= 0, so the released capacity never underestimates
//! the true cardinality and no valid tuple is dropped.  When one query
//! materializes several bags over the same base relations, divide the
//! query's overall budget by the number of bags before calling this.
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
    /// The stage-2 sensitivity actually used (max or sum of the thresholds).
    pub sensitivity: f64,
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

/// Release a DP padding capacity for one intra-cluster join `A |x| B`.
///
/// * `join_size`   -- true cardinality `m = |A |x| B|` (private).
/// * `max_freq_a`, `max_freq_b` -- true maximum join-key frequencies
///   (private).
/// * `epsilon`, `delta` -- TOTAL budget for this capacity release, split
///   equally across the three internal releases (basic composition).
/// * `self_join`   -- set when `A` and `B` are instances of the same base
///   relation; the stage-2 sensitivity is then `tau_a + tau_b`.
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
    let sens = if self_join { tau_a + tau_b } else { tau_a.max(tau_b) }.max(1.0);

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
                if self_join {
                    assert!(out.sensitivity >= out.tau_a + out.tau_b - 1e-9);
                }
            }
        }
    }
}
