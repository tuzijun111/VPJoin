//! Independent measurement of how the laned graph circuits grow with the lane
//! count, and how that compares to raising the degree instead.
//!
//! Cost proxy: advice cells = advice_columns * 2^k, plus the lookup-argument
//! count, which is the other term that scales per lane.
//!
//!     cargo test --test lane_cost_probe -- --nocapture
//!
use halo2_experiments::graph_sql::{g_sql3_obj_dp, g_sql4_obj_dp};
use halo2_proofs::halo2curves::pasta::Fp;
use halo2_proofs::plonk::{Circuit, ConstraintSystem};

fn gq3_shape(c: usize) -> (usize, usize) {
    g_sql3_obj_dp::set_config_lanes(c);
    let mut cs = ConstraintSystem::<Fp>::default();
    <g_sql3_obj_dp::MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
    (cs.num_advice_columns(), cs.lookups().len())
}

fn gq4_shape(c1: usize, c2: usize) -> (usize, usize) {
    g_sql4_obj_dp::set_config_lanes(c1, c2);
    let mut cs = ConstraintSystem::<Fp>::default();
    <g_sql4_obj_dp::MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
    (cs.num_advice_columns(), cs.lookups().len())
}

#[test]
fn lane_cost_probe() {
    println!("\nGQ3: advice columns / lookup args vs lane count");
    let (a1, l1) = gq3_shape(1);
    for c in [1usize, 2, 3, 4, 8, 44] {
        let (a, l) = gq3_shape(c);
        println!(
            "  c={:>3}  advice {:>5} (+{:>4}/lane)  lookups {:>5}  cells@2^18 {:>6.1}M",
            c,
            a,
            if c > 1 { (a - a1) / (c - 1) } else { 0 },
            l,
            (a as f64) * 262_144.0 / 1e6
        );
    }
    let _ = l1;
    println!("  unlaned at the DP degree: advice {} at 2^19 = {:.1}M cells", a1, (a1 as f64) * 524_288.0 / 1e6);

    println!("\nGQ4: advice columns / lookup args vs (bag1, bag2) lane counts");
    let (b1, _) = gq4_shape(1, 1);
    for (c1, c2) in [(1usize, 1usize), (2, 2), (3, 2), (4, 3), (2, 4)] {
        let (a, l) = gq4_shape(c1, c2);
        println!(
            "  c1={} c2={}  advice {:>5}  lookups {:>5}  cells@2^18 {:>6.1}M",
            c1,
            c2,
            a,
            l,
            (a as f64) * 262_144.0 / 1e6
        );
    }
    println!(
        "  unlaned at the DP degree: advice {} at 2^20 = {:.1}M cells",
        b1,
        (b1 as f64) * 1_048_576.0 / 1e6
    );
}
