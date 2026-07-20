//! Generate DP padding capacities for the intra-cluster (bag) joins of a
//! DP-mode query, ready to be used as the circuit size parameters.
//!
//! The first argument is a comma-separated list of epsilon values, so a
//! single run produces the whole privacy--efficiency sweep reported in the
//! paper (Figures for "various epsilon"): the default sweep is
//! 0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10 with delta = 1e-5.
//!
//! Usage:
//!   cargo run --release --bin dp_capacity_gen -- \
//!       <eps_list> <delta> <n_bags> \
//!       [<join_size> <max_freq_a> <max_freq_b> <self_join:0|1>]+
//!
//! where <eps_list> is one epsilon or a comma-separated list.
//!
//! Example -- Q5 (two bags over distinct relations), full sweep, delta=1e-5:
//!   cargo run --release --bin dp_capacity_gen -- \
//!       0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10 1e-5 2 \
//!       <co_size> <mfO> <mfC> 0 \
//!       <ls_size> <mfL> <mfS> 0
//!
//! Example -- GQ3/GQ4 (two Edge|x|Edge self-join bags) at the default eps:
//!   cargo run --release --bin dp_capacity_gen -- 0.1 1e-5 2 \
//!       120000 7 4 1  118000 7 4 1
//!
//! For each epsilon the TOTAL budget is divided equally across the n_bags
//! capacity releases (basic composition); each output row is one bag's
//! released capacity.  Feed the printed `capacity` values to the DP-mode
//! circuits as their intra-cluster column capacities (set the circuit's
//! `*_pad_extra` knob to `capacity - true_size`), or call
//! `halo2_experiments::dp_noise::dp_join_capacity` directly at
//! witness-generation time.  Only the printed DP-safe fields may be public.

use halo2_experiments::dp_noise::dp_join_capacity;
use rand::rngs::OsRng;

struct Bag {
    join_size: u64,
    mf_a: u64,
    mf_b: u64,
    self_join: bool,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 7 || (args.len() - 3) % 4 != 0 {
        eprintln!(
            "usage: dp_capacity_gen <eps_list> <delta> <n_bags> \
             [<join_size> <max_freq_a> <max_freq_b> <self_join:0|1>]+\n\
             <eps_list> is one epsilon or a comma-separated sweep, e.g. \
             0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10"
        );
        std::process::exit(1);
    }
    let epsilons: Vec<f64> = args[0]
        .split(',')
        .map(|s| s.trim().parse().expect("epsilon"))
        .collect();
    let delta: f64 = args[1].parse().expect("delta");
    let n_bags: usize = args[2].parse().expect("n_bags");
    let bags: Vec<Bag> = args[3..]
        .chunks(4)
        .map(|c| Bag {
            join_size: c[0].parse().expect("join_size"),
            mf_a: c[1].parse().expect("max_freq_a"),
            mf_b: c[2].parse().expect("max_freq_b"),
            self_join: c[3] == "1",
        })
        .collect();
    assert_eq!(bags.len(), n_bags, "expected {} bag descriptions", n_bags);

    println!(
        "delta = {}, {} bag(s); total budget split equally across bags per epsilon\n",
        delta, n_bags
    );
    let mut rng = OsRng;
    for &epsilon in &epsilons {
        let (eps_bag, del_bag) = (epsilon / n_bags as f64, delta / n_bags as f64);
        let mut total_pad: u64 = 0;
        println!("epsilon = {}  (per bag: eps = {:.5}, delta = {:.3e})", epsilon, eps_bag, del_bag);
        for (i, bag) in bags.iter().enumerate() {
            let out = dp_join_capacity(
                bag.join_size,
                bag.mf_a,
                bag.mf_b,
                eps_bag,
                del_bag,
                bag.self_join,
                &mut rng,
            );
            let pad = out.capacity.saturating_sub(bag.join_size);
            total_pad += pad;
            println!(
                "  bag {}: true = {:>10}  capacity = {:>12}  pad_extra = {:>12}  \
                 (tau_a = {:.1}, tau_b = {:.1}, sens = {:.1}, self_join = {})",
                i, bag.join_size, out.capacity, pad, out.tau_a, out.tau_b, out.sensitivity, out.self_join
            );
        }
        println!("  total pad rows across bags = {}\n", total_pad);
    }
}
