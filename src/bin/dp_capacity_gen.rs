//! Generate DP padding capacities for the intra-cluster (bag) joins of a
//! DP-mode query, ready to be used as the circuit size parameters.
//!
//! Usage:
//!   cargo run --release --bin dp_capacity_gen -- \
//!       <epsilon> <delta> <n_bags> \
//!       <join_size> <max_freq_a> <max_freq_b> <self_join:0|1> [... one triple+flag per bag]
//!
//! Example (GQ4 on LastFM: two Edge|x|Edge bags, total budget (0.1, 1e-5)):
//!   cargo run --release --bin dp_capacity_gen -- 0.1 1e-5 2 \
//!       120000 7 4 1 \
//!       118000 7 4 1
//!
//! The total budget is divided equally across the n_bags capacity releases
//! (basic composition); each line of output is one bag's released capacity.
//! Feed the printed `capacity` values to the DP-mode circuits as their
//! intra-cluster column capacities (replacing the previously hard-coded
//! constants).  Only the printed DP-safe fields may be made public.

use halo2_experiments::dp_noise::dp_join_capacity;
use rand::rngs::OsRng;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 7 || (args.len() - 3) % 4 != 0 {
        eprintln!(
            "usage: dp_capacity_gen <epsilon> <delta> <n_bags> \
             [<join_size> <max_freq_a> <max_freq_b> <self_join:0|1>]+"
        );
        std::process::exit(1);
    }
    let epsilon: f64 = args[0].parse().expect("epsilon");
    let delta: f64 = args[1].parse().expect("delta");
    let n_bags: usize = args[2].parse().expect("n_bags");
    let bags: Vec<&[String]> = args[3..].chunks(4).collect();
    assert_eq!(bags.len(), n_bags, "expected {} bag descriptions", n_bags);

    let (eps_bag, del_bag) = (epsilon / n_bags as f64, delta / n_bags as f64);
    println!(
        "total budget ({}, {}) split over {} bag(s): ({}, {}) each",
        epsilon, delta, n_bags, eps_bag, del_bag
    );
    let mut rng = OsRng;
    for (i, bag) in bags.iter().enumerate() {
        let join_size: u64 = bag[0].parse().expect("join_size");
        let mf_a: u64 = bag[1].parse().expect("max_freq_a");
        let mf_b: u64 = bag[2].parse().expect("max_freq_b");
        let self_join = bag[3] == "1";
        let out = dp_join_capacity(join_size, mf_a, mf_b, eps_bag, del_bag, self_join, &mut rng);
        println!(
            "bag {}: capacity = {}  (tau_a = {:.1}, tau_b = {:.1}, sensitivity = {:.1}, self_join = {})",
            i, out.capacity, out.tau_a, out.tau_b, out.sensitivity, out.self_join
        );
    }
}
