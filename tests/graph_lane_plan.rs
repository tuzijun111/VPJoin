//! Lane plan for the cyclic graph queries: for each (query, dataset, epsilon),
//! the degree the Revealing-Join-Size configuration needs, the DP-released
//! pads, the degree the DP configuration would need WITHOUT lanes, and the
//! number of lanes that host the same capacity at the RJS degree instead.
//!
//!     cargo test --test graph_lane_plan -- --nocapture
//!
//! The per-lane row budget comes from the circuits themselves
//! (`g_sql3_obj_dp::lane_rows_for`, `g_sql4_obj_dp::lane_rows_for`) rather than
//! being restated here: GQ3 reserves 5 range-table load regions and GQ4
//! reserves 3, so one hard-coded slack would misreport both. GQ4 lanes its two
//! bags independently and so gets two lane counts; GQ3's Bag2 is the Edge
//! relation, whose size is public, so only its Bag1 is laned.
use halo2_experiments::bench_queries::{bag_stats, degree_for, graph_pads, load_graph, Privacy};
use halo2_experiments::graph_sql::{g_sql3_obj_dp, g_sql4_obj_dp};

#[test]
fn graph_lane_plan() {
    let delta = 1e-5;
    println!(
        "\n{:>4} {:>9} {:>6} {:>6} {:>10} | {:>11} {:>11} | {:>5} {:>10} {:>7} {:>7}",
        "q",
        "dataset",
        "eps",
        "k_rjs",
        "true_bag",
        "padded_bag1",
        "padded_bag2",
        "k_dp",
        "lane_rows",
        "lanes1",
        "lanes2"
    );
    for dataset in ["lastfm", "facebook", "wiki"] {
        let edges = load_graph(dataset);
        for q in ["gq3", "gq4"] {
            let s = bag_stats(q, &edges);
            let k_rjs = degree_for(q, dataset, Privacy::Rjs);
            // Usable rows per lane at the RJS degree, straight from the circuit
            // that would host them.
            let lane_rows = if q == "gq3" {
                g_sql3_obj_dp::lane_rows_for(k_rjs)
            } else {
                g_sql4_obj_dp::lane_rows_for(k_rjs)
            };
            for eps in [0.1, 0.01] {
                let privacy = Privacy::Dp { epsilon: eps, delta };
                let (p1, p2) = graph_pads(q, dataset, &edges, privacy);
                let n1 = s.bag1_size as usize + p1;
                let n2 = s.bag2_size as usize + p2;
                // GQ3 lanes Bag1 only; GQ4 lanes both bags independently.
                let lanes1 = g_sql3_obj_dp::lanes_for(n1, lane_rows);
                let lanes2 = if q == "gq3" {
                    0
                } else {
                    g_sql4_obj_dp::lanes_for(n2, lane_rows)
                };
                println!(
                    "{:>4} {:>9} {:>6} {:>6} {:>10} | {:>11} {:>11} | {:>5} {:>10} {:>7} {:>7}",
                    q,
                    dataset,
                    eps,
                    k_rjs,
                    s.bag1_size,
                    n1,
                    n2,
                    degree_for(q, dataset, privacy),
                    lane_rows,
                    lanes1,
                    lanes2
                );
            }
        }
    }
}
