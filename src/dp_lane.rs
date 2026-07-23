

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
    /// Lane count, one entry per laned bag (GQ4 lanes two bags).
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
