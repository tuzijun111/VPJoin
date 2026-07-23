//! The GQ4 early ordering filters must not change the proved count.
//!
//! LastFM/Facebook store every edge with src<dst, so they contain no directed
//! 4-cycle at all and prove count 0 -- a test that passes even if the filter
//! wrongly drops real rows. These graphs are built to contain genuine cycles
//! AND rows the filter must discard, so a wrong filter changes the count.
use halo2_experiments::data::graph_data_processing::Edge;
use halo2_experiments::graph_sql::g_sql4_obj::MyCircuit;
use halo2_proofs::dev::MockProver;
use halo2curves::pasta::Fp;
use std::marker::PhantomData;

/// Independent brute force: 4-cycles A->B->C->D->A with A<B<C<D.
fn brute(edges: &[(u64, u64)]) -> u64 {
    let mut n = 0u64;
    for &(a, b) in edges {
        if a >= b { continue; }
        for &(b2, c) in edges {
            if b2 != b || b >= c { continue; }
            for &(c2, d) in edges {
                if c2 != c || c >= d { continue; }
                for &(d2, a2) in edges {
                    if d2 == d && a2 == a { n += 1; }
                }
            }
        }
    }
    n
}

fn run(name: &str, raw: &[(u64, u64)], k: u32) {
    let expect = brute(raw);
    let edges: Vec<Edge> = raw.iter().map(|&(s, d)| Edge { src: s, dst: d }).collect();
    let circuit = MyCircuit::<Fp> {
        edges,
        bag1_pad_extra: 0,
        bag2_pad_extra: 0,
        _marker: PhantomData,
    };
    let prover = MockProver::run(k, &circuit, vec![vec![Fp::from(expect)]])
        .unwrap_or_else(|e| panic!("{}: MockProver::run failed: {:?}", name, e));
    assert_eq!(prover.verify(), Ok(()), "{}: constraints unsatisfied (expected count {})", name, expect);
    println!("{:<22} edges={:<3} count={} OK", name, raw.len(), expect);
}

#[test]
fn filters_preserve_the_count() {
    // one clean 4-cycle 1->2->3->4->1
    run("single cycle", &[(1, 2), (2, 3), (3, 4), (4, 1)], 12);

    // two cycles sharing the 1->2->3 prefix
    run("two cycles", &[(1, 2), (2, 3), (3, 4), (4, 1), (3, 5), (5, 1)], 12);

    // cycles PLUS rows the filters must drop:
    //   (6,2): wedge (6,2,3) has A=6 >= B=2  -> Bag1 filter drops it
    //   (7,3): wedge (7,3,4) has A=7 >= B=3  -> Bag1 filter drops it
    //   Bag2 rows built on those same edges have C >= D -> Bag2 filter drops them
    run(
        "cycles + filtered rows",
        &[(1, 2), (2, 3), (3, 4), (4, 1), (3, 5), (5, 1), (6, 2), (7, 3), (8, 4)],
        13,
    );

    // a graph whose ONLY wedges are filtered out: no cycle, count 0
    run("all rows filtered", &[(9, 2), (2, 3), (8, 3), (7, 2)], 12);

    // denser: a 4-clique-ish mix with multiple cycles and reverse edges
    run(
        "dense mixed",
        &[
            (1, 2), (2, 3), (3, 4), (4, 1),
            (1, 3), (3, 6), (6, 1),
            (2, 4), (4, 5), (5, 2),
            (5, 1), (6, 2), (4, 2), (3, 1),
        ],
        14,
    );
}
