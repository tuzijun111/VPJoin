use crate::bench_queries::Privacy;

pub fn vk_released_capacity(privacy: Privacy, capacity: usize) -> Option<usize> {
    if std::env::var("VPJOIN_FULL_LANES")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return None;
    }
    match privacy {
        Privacy::Dp { .. } | Privacy::Rjs => Some(capacity.max(1)),
        // Oblivious capacities are a public function of the input length, so
        // like Legacy's declared constants they are not a release the verifying
        // key has to carry.
        Privacy::Legacy | Privacy::Oblivious => None,
    }
}

/// Circuit geometry the released capacities imply, before anything is proved.
#[derive(Clone, Debug)]
pub struct DpLanePlan {
    pub query: String,

    pub dataset: String,

    pub k: u32,

    pub lane_rows: usize,

    pub lanes: Vec<usize>,

    pub capacity: Vec<usize>,

    pub true_size: Vec<usize>,

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

    #[test]
    fn legacy_never_puts_a_capacity_in_the_key() {
        let dp = Privacy::Dp {
            epsilon: 0.1,
            delta: 1e-5,
        };
        assert_eq!(vk_released_capacity(dp, 1000), Some(1000));
        assert_eq!(vk_released_capacity(Privacy::Rjs, 1000), Some(1000));
        assert_eq!(
            vk_released_capacity(Privacy::Legacy, 1000),
            None,
            "Legacy's pad is a public constant, so a capacity in the key would \
             pin the true bag size"
        );
    }

    #[test]
    fn an_empty_release_is_clamped_to_one_row() {
        let dp = Privacy::Dp {
            epsilon: 0.1,
            delta: 1e-5,
        };
        assert_eq!(vk_released_capacity(dp, 0), Some(1));
        assert_eq!(vk_released_capacity(Privacy::Rjs, 0), Some(1));
        assert_eq!(vk_released_capacity(Privacy::Legacy, 0), None);
    }
}
