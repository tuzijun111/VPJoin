//! Plan-only check of the Q5 DP release under P = {customer, supplier}:
//! prints, for each epsilon of the paper sweep, the released pads, the
//! LS pipeline height n = ls_true + ls_pad, the lane count the DP lane
//! circuit will use (LANE_ROWS = 63,000), and the degree the single-column
//! circuit would need instead.  No proving; run with
//!
//!     cargo test --test q5_dp_pads -- --nocapture
//!
use halo2_experiments::bench_queries::{degree_for, q5_pads, Privacy};

#[test]
fn q5_dp_pad_plan() {
    const LANE_ROWS: usize = 63_000;
    let ls_true = 63usize; // circuit-derived true bag size (see q5_pads log)
    println!(
        "{:>6} | {:>8} | {:>8} | {:>9} | {:>5} | {:>12}",
        "eps", "co_pad", "ls_pad", "n_ls", "lanes", "1-col degree"
    );
    for &eps in &[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0] {
        let privacy = Privacy::Dp { epsilon: eps, delta: 1e-5 };
        let (_, co_pad, ls_pad) = q5_pads(privacy);
        let n = ls_true + ls_pad;
        let lanes = n.div_ceil(LANE_ROWS);
        let k = degree_for("q5", "tpch", privacy);
        println!(
            "{:>6} | {:>8} | {:>8} | {:>9} | {:>5} | {:>12}",
            eps, co_pad, ls_pad, n, lanes, k
        );
    }
}
