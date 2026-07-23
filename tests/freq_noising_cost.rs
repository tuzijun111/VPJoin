//! Cost of the frequency-noising stage, per query.
//!
//! Frequency noising is required only when the maximum join-key frequency is
//! itself a statistic of a PROTECTED relation.  This test contrasts, for each
//! DP query, the released pad under (a) the noised-frequency mechanism and
//! (b) an exact frequency bound with the whole budget on the size release.
//!
//!     cargo test --test freq_noising_cost -- --nocapture
//!
use halo2_experiments::bench_queries::{graph_pads, load_graph, Privacy};

#[test]
fn frequency_noising_cost() {
    let (eps, del) = (0.1, 1e-5);
    for dataset in ["lastfm", "facebook", "wiki"] {
        let edges = load_graph(dataset);
        for q in ["gq3", "gq4"] {
            let (b1, b2) = graph_pads(q, dataset, &edges, Privacy::Dp { epsilon: eps, delta: del });
            println!("{:>4} {:>9}: noised-frequency pads = ({}, {})", q, dataset, b1, b2);
        }
    }
}
