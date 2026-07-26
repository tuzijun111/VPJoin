//! Constraint-system cost of each query circuit.
//!
//! Every query has one circuit, `*_obj`, certifying the One-Pass OBJ: (7)
//! Conservation, (9) Pairwise Consistency and (10) Cardinality Preservation, the
//! last by the two-channel multiplicity count of `circuits::card_preserve`.
//! Nothing is proved and no witness is generated, so this runs in well under a
//! second and is the cheapest way to read off the constant factors the paper's
//! cost accounting quotes.
//!
//!   cargo run --release --bin obj_gate_cost
//!
//! Columns: advice / fixed / selector column counts, the maximum gate degree,
//! the number of custom gates and of individual constraint polynomials, and the
//! number of lookup and shuffle arguments.

use halo2_proofs::plonk::{Circuit, ConstraintSystem};
use halo2curves::pasta::Fp;

struct Cost {
    advice: usize,
    fixed: usize,
    selectors: usize,
    degree: usize,
    gates: usize,
    polys: usize,
    lookups: usize,
    shuffles: usize,
}

fn cost_of<C: Circuit<Fp>>() -> Cost {
    let mut cs = ConstraintSystem::<Fp>::default();
    let _ = C::configure(&mut cs);
    Cost {
        advice: cs.num_advice_columns(),
        fixed: cs.num_fixed_columns(),
        selectors: cs.num_selectors(),
        degree: cs.degree(),
        gates: cs.gates().len(),
        polys: cs.gates().iter().map(|g| g.polynomials().len()).sum(),
        lookups: cs.lookups().len(),
        shuffles: cs.shuffles().len(),
    }
}

fn header() {
    println!(
        "{:<6} {:<9} {:>6} {:>5} {:>4} {:>4} {:>6} {:>6} {:>8} {:>8}",
        "query", "variant", "advice", "fixed", "sel", "deg", "gates", "polys", "lookups", "shuffles"
    );
    println!("{}", "-".repeat(80));
}

fn line(query: &str, variant: &str, c: &Cost) {
    println!(
        "{:<6} {:<9} {:>6} {:>5} {:>4} {:>4} {:>6} {:>6} {:>8} {:>8}",
        query,
        variant,
        c.advice,
        c.fixed,
        c.selectors,
        c.degree,
        c.gates,
        c.polys,
        c.lookups,
        c.shuffles
    );
}

fn delta(query: &str, what: &str, old: &Cost, new: &Cost) {
    let d = |a: usize, b: usize| -> String {
        let x = b as i64 - a as i64;
        if x >= 0 {
            format!("+{}", x)
        } else {
            format!("{}", x)
        }
    };
    println!(
        "{:<6} {:<9} {:>6} {:>5} {:>4} {:>4} {:>6} {:>6} {:>8} {:>8}",
        query,
        what,
        d(old.advice, new.advice),
        d(old.fixed, new.fixed),
        d(old.selectors, new.selectors),
        d(old.degree, new.degree),
        d(old.gates, new.gates),
        d(old.polys, new.polys),
        d(old.lookups, new.lookups),
        d(old.shuffles, new.shuffles)
    );
    println!();
}

/// One row, for a query with no key-edge variant.
macro_rules! single {
    ($q:expr, $gen:ty) => {{
        line($q, "obj", &cost_of::<$gen>());
        println!();
    }};
}

fn main() {
    use halo2_experiments::graph_sql::*;
    use halo2_experiments::sql::*;

    header();

    single!("q3", q3_obj::MyCircuit<Fp>);
    single!("q5", q5_obj::MyCircuit<Fp>);
    single!("q8", q8_obj::MyCircuit<Fp>);
    single!("q9", q9_obj::MyCircuit<Fp>);
    single!("q18", q18_obj::MyCircuit<Fp>);

    single!("gq1", g_sql1_obj::Path3OrdCircuit<Fp>);
    single!("gq2", g_sql2_obj::GraphPath4OrderCircuit<Fp>);
    single!("gq3", g_sql3_obj::MyCircuit<Fp>);
    single!("gq4", g_sql4_obj::MyCircuit<Fp>);

    println!(
        "One row per query: what the constraint system fixes independently of the data. Rows and\n\
         constraint totals also scale with the relation sizes, which this report does not see."
    );
}
