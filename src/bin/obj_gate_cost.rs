//! Constraint-system cost of the One-Pass OBJ realizations.
//!
//! Every query has the general `*_obj` circuit, which certifies conditions (7),
//! (9) and (10) with the two-channel Cardinality Preservation Check. The five
//! TPC-H queries also have `*_obj_key`, which specializes that check on edges
//! whose child holds at most one tuple per join key, so the report is a pair
//! there and a single row for the graph queries.
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

/// A pair, for a query that also has a key-edge variant.
macro_rules! pair {
    ($q:expr, $gen:ty, $key:ty) => {{
        let g = cost_of::<$gen>();
        let k = cost_of::<$key>();
        line($q, "obj", &g);
        line($q, "obj_key", &k);
        delta($q, "key-saves", &g, &k);
    }};
}

fn main() {
    use halo2_experiments::graph_sql::*;
    use halo2_experiments::sql::*;

    header();

    pair!("q3", q3_obj::MyCircuit<Fp>, q3_obj_key::MyCircuit<Fp>);
    pair!("q5", q5_obj::MyCircuit<Fp>, q5_obj_key::MyCircuit<Fp>);
    pair!("q8", q8_obj::MyCircuit<Fp>, q8_obj_key::MyCircuit<Fp>);
    pair!("q9", q9_obj::MyCircuit<Fp>, q9_obj_key::MyCircuit<Fp>);
    pair!("q18", q18_obj::MyCircuit<Fp>, q18_obj_key::MyCircuit<Fp>);

    single!("gq1", g_sql1_obj::Path3OrdCircuit<Fp>);
    single!("gq2", g_sql2_obj::GraphPath4OrderCircuit<Fp>);
    single!("gq3", g_sql3_obj::MyCircuit<Fp>);
    single!("gq4", g_sql4_obj::MyCircuit<Fp>);

    println!(
        "The obj rows are the general One-Pass OBJ: conditions (7) Conservation, (9) Pairwise\n\
         Consistency and (10) Cardinality Preservation, with (10) certified by the two-channel\n\
         multiplicity count.\n\
         \n\
         The obj_key rows specialize (10) on key edges, where the child holds at most one tuple\n\
         per join key so every sigma is a bit: the per-key aggregation collapses and, when every\n\
         edge of the tree is such an edge, the clean channel disappears entirely. Absence\n\
         certification is NOT removed, which is why key-saves is about a quarter rather than all\n\
         of the check. GQ1, GQ2 and GQ4 have no key edge and GQ3 was not converted, so the graph\n\
         queries have a single row.\n\
         \n\
         Rows and constraint totals also scale with the relation sizes, which this report does not\n\
         see: it reports only what the constraint system fixes independently of the data."
    );
}
