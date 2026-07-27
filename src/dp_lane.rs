use crate::bench_queries::Privacy;

/// The released capacity a laned circuit may PIN IN ITS VERIFYING KEY, or
/// `None` when it must fall back to filling every lane to `lane_rows`.
///
/// A selector is a fixed column, so every row range a lane is gated on is
/// committed at keygen and is part of what the verifier holds. Stopping the
/// last lane at the capacity therefore discloses the capacity, and the only
/// question is whether that number is already public:
///
///   * `Dp`     -> `Some`. The capacity IS the release: the harness prints it,
///                 the verifier is told it, and the DP guarantee is precisely
///                 about publishing this number. Pinning it in the vk discloses
///                 nothing the release did not.
///   * `Rjs`    -> `Some`. The capacity equals the true bag size, and revealing
///                 the join size is what that regime is. Nothing is left to
///                 protect.
///   * `Legacy` -> `None`. Its pad is a public CONSTANT, so
///                 `capacity = true_size + constant`: a capacity in the vk
///                 would PIN THE TRUE BAG SIZE, the one statistic the padding
///                 exists to hide. Filling the lanes discloses only the lane
///                 count `ceil(capacity / lane_rows)`, which is what this
///                 regime disclosed before the short last lane existed.
///
/// `VPJOIN_FULL_LANES=1` forces `None` in every regime. That is the A/B knob
/// for measuring what the short last lane is worth: one binary, one witness,
/// one lane count, differing only in whether the last lane stops at the
/// capacity or runs to `lane_rows`.
pub fn vk_released_capacity(privacy: Privacy, capacity: usize) -> Option<usize> {
    if std::env::var("VPJOIN_FULL_LANES")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return None;
    }
    match privacy {
        // `max(1)` mirrors the clamp the circuits apply to the capacity their
        // own witness implies. An empty bag is a legal input: a graph whose
        // edges all run high id to low id survives no A<B pre-filter, so the
        // wedge is empty and the release is 0. `lane_live_rows` requires the
        // capacity to tile the lanes with a non-empty last lane, so handing it
        // a bare 0 aborts a circuit that the full-lane layout builds happily.
        Privacy::Dp { .. } | Privacy::Rjs => Some(capacity.max(1)),
        Privacy::Legacy => None,
    }
}

/// Circuit geometry the released capacities imply, before anything is proved.
#[derive(Clone, Debug)]
pub struct DpLanePlan {
    /// "q5", "gq3", "gq4".
    pub query: String,
    /// "tpch-60K" for Q5, the graph name for GQ3/GQ4.
    pub dataset: String,
    /// Degree the circuit is proved at. PINNED at the Revealing-Join-Size
    /// value: the DP release is absorbed by lanes, not by a bigger domain.
    pub k: u32,
    /// Rows one lane hosts.
    pub lane_rows: usize,
    /// Lane count, one entry per laned bag. GQ3 lanes one of its two bags (the
    /// other is the public-size edge relation) and GQ4 lanes the ONE relation
    /// its two column roles share, so both carry a single entry; Q5 carries one
    /// per laned intermediate.
    pub lanes: Vec<usize>,
    /// Released capacity per laned bag, i.e. true size + DP pad.
    pub capacity: Vec<usize>,
    /// True (non-private) size of each laned bag, for reference only.
    pub true_size: Vec<usize>,
    /// `pad_extra` per materialized intermediate, in the circuit's own order.
    pub pads: Vec<usize>,
}

fn join(v: &[usize]) -> String {
    v.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join("/")
}

impl DpLanePlan {
    /// Lane counts, one per laned bag, e.g. "8" or "4/4".
    pub fn lanes_str(&self) -> String {
        join(&self.lanes)
    }

    /// Released capacities, one per laned bag.
    pub fn capacity_str(&self) -> String {
        join(&self.capacity)
    }

    /// True bag sizes, one per laned bag.
    pub fn true_size_str(&self) -> String {
        join(&self.true_size)
    }

    /// Pads, one per materialized intermediate.
    pub fn pads_str(&self) -> String {
        join(&self.pads)
    }
}

/// One completed measurement: keys built once, then `prove_s.len()` proofs.
#[derive(Clone, Debug)]
pub struct DpLaneRun {
    pub plan: DpLanePlan,
    /// `keygen_vk` + `keygen_pk`, outside the timed repetitions.
    pub keygen_s: f64,
    /// One entry per repetition.
    pub prove_s: Vec<f64>,
    /// Mean over the repetitions; every proof is verified.
    pub verify_s: f64,
    pub proof_bytes: usize,
}

impl DpLaneRun {
    pub fn prove_mean(&self) -> f64 {
        self.prove_s.iter().sum::<f64>() / self.prove_s.len() as f64
    }

    pub fn prove_min(&self) -> f64 {
        self.prove_s.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn prove_max(&self) -> f64 {
        self.prove_s
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The regime-to-layout mapping IS the privacy decision behind the short
    /// last lane, and until now nothing tested it.
    ///
    /// A `Some` capacity ends up in a selector enable range and therefore in
    /// the verifying key. That is sound exactly where the capacity is a genuine
    /// public release (Dp) or where the join size is revealed by construction
    /// (Rjs), and unsound under Legacy, whose pad is a fixed public constant:
    /// there `capacity = true_size + constant` would pin the true bag size.
    #[test]
    fn legacy_never_puts_a_capacity_in_the_key() {
        let dp = Privacy::Dp { epsilon: 0.1, delta: 1e-5 };
        assert_eq!(vk_released_capacity(dp, 1000), Some(1000));
        assert_eq!(vk_released_capacity(Privacy::Rjs, 1000), Some(1000));
        assert_eq!(
            vk_released_capacity(Privacy::Legacy, 1000),
            None,
            "Legacy's pad is a public constant, so a capacity in the key would \
             pin the true bag size"
        );
    }

    /// An empty bag is a legal input: a graph whose edges all run from a high
    /// id to a low one survives no `A<B` pre-filter, so the wedge is empty and
    /// the release is 0. `lane_live_rows` needs the capacity to tile the lanes
    /// with a non-empty last lane, so a bare 0 would abort a circuit that the
    /// full-lane layout builds happily.
    #[test]
    fn an_empty_release_is_clamped_to_one_row() {
        let dp = Privacy::Dp { epsilon: 0.1, delta: 1e-5 };
        assert_eq!(vk_released_capacity(dp, 0), Some(1));
        assert_eq!(vk_released_capacity(Privacy::Rjs, 0), Some(1));
        assert_eq!(vk_released_capacity(Privacy::Legacy, 0), None);
    }
}
