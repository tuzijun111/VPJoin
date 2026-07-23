//! Q5 padding under the mechanism the appendix describes VERBATIM:
//! every relation private (neighbors differ in one record of any relation),
//! each bag's maximum join-key frequencies noised by the same one-sided
//! mechanism before they calibrate the size release, and the total budget
//! split equally across the intra-cluster estimates by basic composition.
//!
//! Q5 has two intra-cluster bags (CO = orders |x| customer on custkey,
//! LS = lineitem |x| supplier on suppkey), so each bag gets (eps/2, delta/2)
//! and `dp_join_capacity` splits that three further ways internally
//! (tau_a, tau_b, size), i.e. eps/6 per release.
//!
//!     cargo test --test q5_appendix_mechanism -- --nocapture
//!
use halo2_experiments::bench_queries::{tpch_inputs, Privacy, TpchInput};
use halo2_experiments::dp_noise::dp_join_capacity;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use std::collections::HashMap;

fn max_freq(rows: &[Vec<u64>], idx: usize) -> u64 {
    let mut m: HashMap<u64, u64> = HashMap::new();
    for r in rows {
        *m.entry(r[idx]).or_insert(0) += 1;
    }
    m.values().copied().max().unwrap_or(0)
}

#[test]
fn q5_appendix_exact_mechanism() {
    let input = tpch_inputs("q5", Privacy::Rjs);
    let TpchInput::Q5 {
        customer,
        orders,
        lineitem,
        supplier,
        nation,
        region,
        europe_hash,
        start_ts,
        end_ts,
        ..
    } = input
    else {
        unreachable!()
    };

    let d = halo2_experiments::sql::q5_obj::q5_derive(
        &customer, &orders, &lineitem, &supplier, &nation, &region, europe_hash, start_ts, end_ts,
        0, 0,
    );
    let ls_true = d.ls_join_u64.len() as u64;

    // CO bag: date-filtered orders |x| customer on custkey.
    let orders_f: Vec<Vec<u64>> = orders
        .iter()
        .filter(|o| o[0] >= start_ts && o[0] < end_ts)
        .cloned()
        .collect();
    let cust_mult: HashMap<u64, u64> = {
        let mut m = HashMap::new();
        for c in &customer {
            *m.entry(c[0]).or_insert(0u64) += 1;
        }
        m
    };
    let co_true: u64 = orders_f
        .iter()
        .map(|o| cust_mult.get(&o[1]).copied().unwrap_or(0))
        .sum();
    let co_fa = max_freq(&orders_f, 1); // orders per custkey
    let co_fb = max_freq(&customer, 0); // customer rows per custkey

    // LS bag: lineitem |x| supplier on suppkey (region filter is on the
    // public catalogs, so it does not change the key frequencies).
    let ls_fa = max_freq(&lineitem, 1); // lineitem per suppkey
    let ls_fb = max_freq(&supplier, 0); // supplier rows per suppkey

    println!(
        "true sizes: CO {} (freq {} / {}), LS {} (freq {} / {})",
        co_true, co_fa, co_fb, ls_true, ls_fa, ls_fb
    );
    println!(
        "\n{:>6} | {:>12} | {:>12} | {:>12} | {:>12} | {:>9}",
        "eps", "tau_co", "co_cap", "tau_ls", "ls_cap", "1-col k"
    );

    let delta = 1e-5;
    for &eps in &[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0] {
        // basic composition over the two intra-cluster estimates
        let (eps_b, del_b) = (eps / 2.0, delta / 2.0);
        let mut rng = XorShiftRng::from_seed([7u8; 16]);
        let co = dp_join_capacity(co_true, co_fa, co_fb, eps_b, del_b, false, &mut rng);
        let ls = dp_join_capacity(ls_true, ls_fa, ls_fb, eps_b, del_b, false, &mut rng);
        let rows = (lineitem.len() as u64).max(co.capacity).max(ls.capacity);
        let k = (rows + 64).next_power_of_two().trailing_zeros();
        println!(
            "{:>6} | {:>12.1} | {:>12} | {:>12.1} | {:>12} | {:>9}",
            eps, co.sensitivity, co.capacity, ls.sensitivity, ls.capacity, k
        );
    }
}
