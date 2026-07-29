use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};
use crate::circuits::card_preserve::{
    assign_cp_agg, assign_cp_join, assign_cp_root, build_cp_stage, configure_cp_agg,
    configure_cp_join, configure_cp_root, wire_cp_edge, CpAggConfig, CpJoinConfig, CpRootConfig,
};
use crate::circuits::conserve_idx::{
    assign_conserve, assign_row_index, configure_conserve, configure_row_index, ConserveConfig,
    RowIndexConfig,
};

use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

const NUM_BYTES: usize = 5;
const MAX_SENTINEL: u64 = (1u64 << (8 * NUM_BYTES)) - 1; // 2^40-1

const SCALE: u64 = 1000;

const PAD_OK: u64 = MAX_SENTINEL; // pad orderkey (max)
const PAD_DATE: u64 = MAX_SENTINEL; // pad orderdate (max -> last when ASC)
const PAD_SHIP: u64 = MAX_SENTINEL; // pad shippriority (max)
const PAD_REV: u64 = 0; // pad revenue (min -> last when DESC)

/// Test hook: deselect one participating order and re-reduce its neighbours
/// around it, so that (1) and (2) still hold and only (3) can object.
pub static HIDE_ONE_CLEAN_TUPLE: AtomicBool = AtomicBool::new(false);

/// Test hook: select every row that passes its predicate, i.e. no reduction at
/// all. Both channels of (3) then coincide, so only (2) can object.
pub static MARK_ALL_CLEAN: AtomicBool = AtomicBool::new(false);

/// Test hook: additionally select one customer row that FAILS the mktsegment
/// predicate. Only the predicate half of (1) can object, which is the half the
/// revision added.
pub static SELECT_A_FILTERED_ROW: AtomicBool = AtomicBool::new(false);

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

#[derive(Clone, Debug)]
pub struct TestCircuitConfig<F: Field + Ord> {
    // one simple selector per relation for the gates, and one complex selector
    // over the same rows for the lookup expressions
    q_enable: Vec<Selector>, // 3: customer, orders, lineitem
    q_row: Vec<Selector>,    // 3: the same rows, usable inside lookups

    customer: Vec<Column<Advice>>, // 2: [c_mktsegment, c_custkey]
    orders: Vec<Column<Advice>>,   // 4: [o_orderdate, o_shippriority, o_custkey, o_orderkey]
    lineitem: Vec<Column<Advice>>, // 4: [l_orderkey, l_extendedprice, l_discount, l_shipdate]

    check: Vec<Column<Advice>>, // 3: the predicate bit b of each relation
    condition: Vec<Column<Advice>>, // 3: the query parameters

    // the query parameters are advice, so they are pinned per proof rather than
    // per row: constant down each column, and the two date columns equal
    q_cond_const: Vec<Selector>, // rows 0..len-2 of each condition column
    q_cond_link: Selector,       // row 0: condition[1] == condition[2]

    // ---------------- (1) Conservation Check ----------------
    // R^_i == R^_i^c U+ R^_i^r over the INDEXED relation: the committed row
    // position, the tuple and the indicator are conserved as one entry, so a
    // single permutation per relation places every occurrence on exactly one
    // side and binds `cflag` to the split the other two conditions read.
    row_idx: RowIndexConfig,
    cons: Vec<ConserveConfig>,  // [customer, orders, lineitem]
    cflag: Vec<Column<Advice>>, // the indicator c, one column per relation

    lt_compare_condition: Vec<LtConfig<F, NUM_BYTES>>,
    equal_condition: Vec<IsZeroConfig<F>>,

    instance: Column<Instance>,
    instance_test: Column<Advice>,

    // ---------------- (3) Cardinality Preservation Check ----------------
    // over the tree rooted at orders, with customer and lineitem as leaves
    cp_agg_c: CpAggConfig<F, NUM_BYTES>, // child customer, keyed by c_custkey
    cp_agg_l: CpAggConfig<F, NUM_BYTES>, // child lineitem, keyed by l_orderkey
    cp_join_c: CpJoinConfig<F, NUM_BYTES>, // orders -> customer
    cp_join_l: CpJoinConfig<F, NUM_BYTES>, // orders -> lineitem
    cp_root: CpRootConfig,
    q_cp_mu: Selector, // the two root product gates

    // ---------- the selector-masked lineitem key ----------
    l_key_sel: Column<Advice>, // c ? l_orderkey : PAD_OK

    // ---------- group-by over the sorted lineitem view ----------
    l_sorted: Vec<Column<Advice>>, // 3: [masked orderkey, extendedprice, discount]
    perm_lsort: PermAnyConfig,

    q_line: Selector,  // enable line_rev + emit + same_next
    q_first: Selector, // run_sum[0] = line_rev[0]
    q_accu: Selector,  // enable run_sum recurrence (needs prev)

    line_rev: Column<Advice>,
    run_sum: Column<Advice>,
    iz_same_prev: IsZeroConfig<F>, // cur_okey - prev_okey == 0 (only rows >=1)
    iz_same_next: IsZeroConfig<F>, // next_okey - cur_okey == 0 (rows 0..n-1)
    is_last: Column<Advice>,       // 1 - iz_same_next, materialized

    // l_sorted[0] is nondecreasing (rows 0..n-2, i.e. real consecutive pairs)
    q_lsort: Selector,
    q_lsort_sentinel: Selector, // row n: l_sorted[0] == PAD_OK
    lt_lsort_cur_next: LtConfig<F, NUM_BYTES>, // okey_cur < okey_next
    iz_lsort_eq: IsZeroConfig<F>, // okey_next - okey_cur == 0

    // ---------- emitted padded result (length = |lineitem|) ----------
    res_pad: Vec<Column<Advice>>, // [okey, odate, shippri, revenue]
    q_res_lookup: Selector,       // gates the input side of the attach lookup

    // ---------- ORDER BY proof (res_pad -> res_sorted) ----------
    res_sorted: Vec<Column<Advice>>, // same 4 cols
    perm_res: PermAnyConfig,
    q_sort_res: Selector, // rows 0..n-2

    lt_rev_next_cur: LtConfig<F, NUM_BYTES>, // rev_next < rev_cur
    lt_date_cur_next: LtConfig<F, NUM_BYTES>, // date_cur < date_next
    iz_rev_eq: IsZeroConfig<F>,              // rev_cur - rev_next == 0
    iz_date_eq: IsZeroConfig<F>,             // date_cur - date_next == 0
}

#[derive(Debug, Clone)]
pub struct TestChip<F: Field + Ord> {
    config: TestCircuitConfig<F>,
}

/// Everything the prover computes off-circuit for one proof. It is derived
/// from the three relations and the two query parameters alone, so the only
/// free choices in it are the three test hooks, and `test_answer_matches_sql`
/// checks the emitted answer against a direct evaluation of the query.
#[derive(Debug, Clone, Default)]
pub struct Witness {
    pub n_c: usize,
    pub n_o: usize,
    pub n_l: usize,
    /// the certified predicate bit b of each relation, per base row
    pub c_check: Vec<u64>,
    pub o_check: Vec<u64>,
    pub l_check: Vec<u64>,
    /// the selector bit c of each relation, per base row
    pub cln_c: Vec<u64>,
    pub cln_o: Vec<u64>,
    pub cln_l: Vec<u64>,
    /// `c ? l_orderkey : PAD_OK`, per lineitem row
    pub l_key_sel_u64: Vec<u64>,
    /// the sorted view of `(masked key, extendedprice, discount)`
    pub l_sorted_u64: Vec<[u64; 3]>,
    pub line_rev_u64: Vec<u64>,
    pub run_sum_u64: Vec<u64>,
    pub is_last_u64: Vec<u64>,
    /// one `(okey, odate, shippri, revenue)` per group, PAD elsewhere
    pub res_pad_u64: Vec<[u64; 4]>,
    /// the same rows under ORDER BY revenue DESC, o_orderdate ASC
    pub res_sorted_u64: Vec<[u64; 4]>,
    pub all_clean: bool,
    pub tamper: bool,
}

fn build_witness(
    customer: &[Vec<u64>],
    orders: &[Vec<u64>],
    lineitem: &[Vec<u64>],
    condition: [u64; 2],
) -> Witness {
    let n_c = customer.len();
    let n_o = orders.len();
    let n_l = lineitem.len();

    // ---------------- predicate bits b ----------------
    let c_check: Vec<u64> = customer
        .iter()
        .map(|c| (c[0] == condition[0]) as u64)
        .collect();
    let o_check: Vec<u64> = orders
        .iter()
        .map(|o| (o[0] < condition[1]) as u64)
        .collect();
    let l_check: Vec<u64> = lineitem
        .iter()
        .map(|l| (l[3] > condition[1]) as u64)
        .collect();

    // ---------------- selector bits c ----------------
    // The honest selection is the fully reduced instance of the
    // predicate-filtered query. The tree is customer <- orders -> lineitem, so
    // one bottom-up pass over the two leaves followed by one top-down pass
    // reaches the fixed point.
    let c_keys: HashSet<u64> = customer
        .iter()
        .zip(c_check.iter())
        .filter(|(_, &b)| b == 1)
        .map(|(c, _)| c[1])
        .collect();
    let l_okeys: HashSet<u64> = lineitem
        .iter()
        .zip(l_check.iter())
        .filter(|(_, &b)| b == 1)
        .map(|(l, _)| l[0])
        .collect();

    let mut cln_o: Vec<u64> = orders
        .iter()
        .zip(o_check.iter())
        .map(|(o, &b)| (b == 1 && c_keys.contains(&o[2]) && l_okeys.contains(&o[3])) as u64)
        .collect();

    // test hook only: no reduction at all, everything that passes its predicate
    // declared clean
    let all_clean = MARK_ALL_CLEAN.load(Ordering::Relaxed);
    if all_clean {
        cln_o = o_check.clone();
    }

    // test hook only: deselect one participating order and re-reduce the
    // neighbours around it, so (1) and (2) survive and only (3) objects
    let tamper = HIDE_ONE_CLEAN_TUPLE.load(Ordering::Relaxed);
    if tamper {
        if let Some(i) = cln_o.iter().position(|&f| f == 1) {
            cln_o[i] = 0;
        }
    }

    // push the surviving orders back down to the two leaves
    let cln_o_custkeys: HashSet<u64> = orders
        .iter()
        .zip(cln_o.iter())
        .filter(|(_, &f)| f == 1)
        .map(|(o, _)| o[2])
        .collect();
    let cln_o_orderkeys: HashSet<u64> = orders
        .iter()
        .zip(cln_o.iter())
        .filter(|(_, &f)| f == 1)
        .map(|(o, _)| o[3])
        .collect();

    let mut cln_c: Vec<u64> = if all_clean {
        c_check.clone()
    } else {
        customer
            .iter()
            .zip(c_check.iter())
            .map(|(c, &b)| (b == 1 && cln_o_custkeys.contains(&c[1])) as u64)
            .collect()
    };

    // test hook only: select a customer row that fails its predicate
    if SELECT_A_FILTERED_ROW.load(Ordering::Relaxed) {
        if let Some(i) = c_check.iter().position(|&b| b == 0) {
            cln_c[i] = 1;
        }
    }

    let cln_l: Vec<u64> = if all_clean {
        l_check.clone()
    } else {
        lineitem
            .iter()
            .zip(l_check.iter())
            .map(|(l, &b)| (b == 1 && cln_o_orderkeys.contains(&l[0])) as u64)
            .collect()
    };

    // ---------------- the aggregation witness ----------------
    // the masked key column, then its sorted view
    let l_key_sel_u64: Vec<u64> = (0..n_l)
        .map(|i| {
            if cln_l[i] == 1 {
                lineitem[i][0]
            } else {
                PAD_OK
            }
        })
        .collect();

    let mut l_sorted_u64: Vec<[u64; 3]> = (0..n_l)
        .map(|i| [l_key_sel_u64[i], lineitem[i][1], lineitem[i][2]])
        .collect();
    l_sorted_u64.sort_by_key(|r| r[0]);

    // orderkey -> (orderdate, shippriority), over the selected orders only,
    // which is what the attach lookup's table side holds
    let mut o_map: HashMap<u64, (u64, u64)> = HashMap::new();
    for (o, &f) in orders.iter().zip(cln_o.iter()) {
        if f == 1 {
            o_map.insert(o[3], (o[0], o[1]));
        }
    }

    let mut line_rev_u64: Vec<u64> = vec![0; n_l];
    let mut run_sum_u64: Vec<u64> = vec![0; n_l];
    let mut is_last_u64: Vec<u64> = vec![0; n_l];
    let mut res_pad_u64: Vec<[u64; 4]> = vec![[PAD_OK, PAD_DATE, PAD_SHIP, PAD_REV]; n_l];

    let mut acc: u128 = 0;
    let mut prev_ok: Option<u64> = None;

    for i in 0..n_l {
        let ok = l_sorted_u64[i][0];
        let ext = l_sorted_u64[i][1] as u128;
        let disc = l_sorted_u64[i][2] as u128;
        let lr = ext * ((SCALE as u128) - disc); // (scaled) revenue contribution
        line_rev_u64[i] = lr as u64;

        if prev_ok == Some(ok) {
            acc += lr;
        } else {
            acc = lr;
        }
        run_sum_u64[i] = acc as u64;

        // one past the last row sits the pinned sentinel, whose key is PAD_OK,
        // so a trailing run of deselected rows never ends a group and the last
        // real key always does
        let next_ok = if i + 1 < n_l {
            l_sorted_u64[i + 1][0]
        } else {
            PAD_OK
        };
        let is_last = next_ok != ok;
        is_last_u64[i] = is_last as u64;

        if is_last {
            let (od, sp) = o_map.get(&ok).copied().unwrap_or((0, 0));
            res_pad_u64[i] = [ok, od, sp, run_sum_u64[i]];
        }
        prev_ok = Some(ok);
    }

    // res_sorted: the emitted groups by (rev desc, odate asc), padded
    let mut groups: Vec<[u64; 4]> = res_pad_u64
        .iter()
        .copied()
        .filter(|r| r[0] != PAD_OK)
        .collect();
    groups.sort_by(|a, b| b[3].cmp(&a[3]).then(a[1].cmp(&b[1])));

    let mut res_sorted_u64: Vec<[u64; 4]> = Vec::with_capacity(n_l);
    res_sorted_u64.extend(groups.into_iter());
    while res_sorted_u64.len() < n_l {
        res_sorted_u64.push([PAD_OK, PAD_DATE, PAD_SHIP, PAD_REV]);
    }

    Witness {
        n_c,
        n_o,
        n_l,
        c_check,
        o_check,
        l_check,
        cln_c,
        cln_o,
        cln_l,
        l_key_sel_u64,
        l_sorted_u64,
        line_rev_u64,
        run_sum_u64,
        is_last_u64,
        res_pad_u64,
        res_sorted_u64,
        all_clean,
        tamper,
    }
}

impl<F: Field + Ord> TestChip<F> {
    pub fn construct(config: TestCircuitConfig<F>) -> Self {
        Self { config }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> TestCircuitConfig<F> {
        let instance = meta.instance_column();
        meta.enable_equality(instance);
        let instance_test = meta.advice_column();
        meta.enable_equality(instance_test);

        // A lookup argument may not read a simple selector, so every relation
        // carries a second, complex selector over exactly the same rows. Both
        // are fixed data determined by |R_i| alone.
        let q_enable = (0..3).map(|_| meta.selector()).collect::<Vec<_>>();
        let q_row = (0..3).map(|_| meta.complex_selector()).collect::<Vec<_>>();

        let customer = vec![meta.advice_column(), meta.advice_column()];
        let orders = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let lineitem = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();

        let condition = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let check = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();

        // -------- the query parameters are one choice per proof, not per row --------
        // Inherited from `q3_obj.rs`: the three condition columns are read as
        // the right-hand side of the predicate chips, one row per base tuple,
        // so without these gates a prover could lower the date on one orders
        // row and raise it on another and certify the answer to no single
        // query. Binding the per-proof choice to a public value still needs a
        // cell in the instance column, which is outside this file.
        let q_cond_const = (0..3).map(|_| meta.selector()).collect::<Vec<_>>();
        for j in 0..3 {
            let q = q_cond_const[j];
            let col = condition[j];
            meta.create_gate("query parameter constant down its column", move |m| {
                let q = m.query_selector(q);
                vec![
                    q * (m.query_advice(col, Rotation::cur())
                        - m.query_advice(col, Rotation::next())),
                ]
            });
        }
        let q_cond_link = meta.selector();
        {
            let c1 = condition[1];
            let c2 = condition[2];
            meta.create_gate("both date predicates read the same parameter", move |m| {
                let q = m.query_selector(q_cond_link);
                vec![
                    q * (m.query_advice(c1, Rotation::cur()) - m.query_advice(c2, Rotation::cur())),
                ]
            });
        }

        // ---------------- predicate chips: the bit b of each relation ----------------
        // IsZero for c_mktsegment == :1  => check[0]
        let is_zero_aux = meta.advice_column();
        let mut equal_condition = vec![];
        let iz = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_enable[0]),
            |m| {
                m.query_advice(customer[0], Rotation::cur())
                    - m.query_advice(condition[0], Rotation::cur())
            },
            is_zero_aux,
        );
        equal_condition.push(iz.clone());

        meta.create_gate("c_mktsegment == :1 => check0", |m| {
            let s = m.query_selector(q_enable[0]);
            let out = m.query_advice(check[0], Rotation::cur());
            vec![
                s.clone() * (iz.expr() * (out.clone() - Expression::Constant(F::ONE))),
                s * (Expression::Constant(F::ONE) - iz.expr()) * out,
            ]
        });

        // Lt for o_orderdate < :2 => check[1] (also booleanize check[1])
        let mut lt_compare_condition = vec![];
        let lt_o = LtChip::configure(
            meta,
            |m| m.query_selector(q_enable[1]),
            |m| m.query_advice(orders[0], Rotation::cur()),
            |m| m.query_advice(condition[1], Rotation::cur()),
        );
        meta.create_gate("o_orderdate < :2 => check1", |m| {
            let s = m.query_selector(q_enable[1]);
            let out = m.query_advice(check[1], Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                s.clone() * (lt_o.is_lt(m, None) - out.clone()),
                s * out.clone() * (one - out), // boolean
            ]
        });
        lt_compare_condition.push(lt_o);

        // Lt for :2 < l_shipdate => check[2] (also booleanize check[2])
        let lt_l = LtChip::configure(
            meta,
            |m| m.query_selector(q_enable[2]),
            |m| m.query_advice(condition[2], Rotation::cur()),
            |m| m.query_advice(lineitem[3], Rotation::cur()),
        );
        meta.create_gate(":2 < l_shipdate => check2", |m| {
            let s = m.query_selector(q_enable[2]);
            let out = m.query_advice(check[2], Rotation::cur());
            let one = Expression::Constant(F::ONE);
            vec![
                s.clone() * (lt_l.is_lt(m, None) - out.clone()),
                s * out.clone() * (one - out), // boolean
            ]
        });
        lt_compare_condition.push(lt_l);

        // ================= (1) Selector Check =================
        // One bit column per relation over the committed rows, and one gate per
        // relation that makes it a bit and confines it to the rows their
        // predicate keeps. That second half is what the earlier version got
        // structurally, by writing the selector into a column group that was
        // padded away wherever the predicate failed; here it is a constraint,
        // and it is the constraint the correctness argument needs to define
        // the input-side join over the filtered relations.
        let cflag = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for (idx, &c) in cflag.iter().enumerate() {
            let q = q_enable[idx];
            let b = check[idx];
            meta.create_gate("selector is a bit and implies its predicate", move |m| {
                let q = m.query_selector(q);
                let c = m.query_advice(c, Rotation::cur());
                let b = m.query_advice(b, Rotation::cur());
                let one = Expression::Constant(F::ONE);
                vec![
                    q.clone() * c.clone() * (one.clone() - c.clone()),
                    q * c * (one - b),
                ]
            });
        }

        // ================= (1) Conservation Check =================
        // One permutation argument per relation, between the indexed relation
        // R^_i and the concatenation of the two parts. The indices are distinct,
        // so R^_i is a set even though the relation is a bag, and that single
        // permutation already rules out an occurrence being fabricated, lost,
        // duplicated or counted on both sides: no Non-Membership Check is
        // needed, which is where this gate saves against a value-level split.
        //
        // The split covers every committed row, so the two parts fill |R_i|
        // exactly and there is no pad section.
        let row_idx = configure_row_index::<F>(meta);
        let cons: Vec<ConserveConfig> = vec![
            configure_conserve::<F>(meta, &row_idx, &customer, cflag[0]),
            configure_conserve::<F>(meta, &row_idx, &orders, cflag[1]),
            configure_conserve::<F>(meta, &row_idx, &lineitem, cflag[2]),
        ];

        // ================= (2) Pairwise Consistency Check =================
        // Two mutual Membership Checks per tree edge, each looking one clean
        // key column up directly in the adjacent clean key column, with no
        // intermediate table to bind. Both sides read
        //
        //     q_row * c(t) * (t[K] + 1),
        //
        // so a deselected row and a row past the relation both read 0, 0 is in
        // every table, and the containment is over the selected keys only. The
        // shift by one is what stops a real key of 0 from colliding with that
        // gated-off 0, which would let a selected row satisfy its membership
        // against nothing. It costs one addition per row.
        //
        // Standing assumption, unchanged from `q3_obj.rs`: nothing here binds
        // the base relations to a public commitment, the only instance cell
        // being the constant 1, so every check is an internal-consistency
        // check over prover-committed inputs.
        let mut pw_edge = |name: &'static str,
                           q_in: Selector,
                           c_in: Column<Advice>,
                           k_in: Column<Advice>,
                           q_tb: Selector,
                           c_tb: Column<Advice>,
                           k_tb: Column<Advice>| {
            meta.lookup_any(name, move |m| {
                let one = Expression::Constant(F::ONE);
                let lhs = m.query_selector(q_in)
                    * m.query_advice(c_in, Rotation::cur())
                    * (m.query_advice(k_in, Rotation::cur()) + one.clone());
                let rhs = m.query_selector(q_tb)
                    * m.query_advice(c_tb, Rotation::cur())
                    * (m.query_advice(k_tb, Rotation::cur()) + one);
                vec![(lhs, rhs)]
            });
        };

        // edge (orders, customer) on custkey
        pw_edge(
            "pw: orders.custkey in customer.custkey",
            q_row[1],
            cflag[1],
            orders[2],
            q_row[0],
            cflag[0],
            customer[1],
        );
        pw_edge(
            "pw: customer.custkey in orders.custkey",
            q_row[0],
            cflag[0],
            customer[1],
            q_row[1],
            cflag[1],
            orders[2],
        );

        // edge (orders, lineitem) on orderkey
        pw_edge(
            "pw: lineitem.orderkey in orders.orderkey",
            q_row[2],
            cflag[2],
            lineitem[0],
            q_row[1],
            cflag[1],
            orders[3],
        );
        pw_edge(
            "pw: orders.orderkey in lineitem.orderkey",
            q_row[1],
            cflag[1],
            orders[3],
            q_row[2],
            cflag[2],
            lineitem[0],
        );

        // ================= (3) Cardinality Preservation Check =================
        // One fixed column serves every Lt chip of the check, so the whole
        // check costs a single u8 range table.
        let cp_u8 = meta.fixed_column();

        // Children of the root. Both are leaves, so their two multiplicity
        // columns are columns the circuit already has: the predicate bit is the
        // input channel, mu^all = b, and the selector bit is the clean one,
        // mu^cln = c. The two agree wherever the row is deselected or fails its
        // predicate, which is the whole content of the leaf base case.
        let cp_agg_c = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            customer[1], // c_custkey
            check[0],
            cflag[0],
            MAX_SENTINEL,
        );
        let cp_agg_l = configure_cp_agg::<F, NUM_BYTES>(
            meta,
            cp_u8,
            lineitem[0], // l_orderkey
            check[2],
            cflag[2],
            MAX_SENTINEL,
        );

        // Parent side, on the rows of orders.
        let cp_join_c = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, orders[2]);
        let cp_join_l = configure_cp_join::<F, NUM_BYTES>(meta, cp_u8, orders[3]);
        wire_cp_edge(meta, &cp_join_c, &cp_agg_c, orders[2]);
        wire_cp_edge(meta, &cp_join_l, &cp_agg_l, orders[3]);

        // Root multiplicities and the single equality that compares the two
        // join cardinalities.
        let cp_root = configure_cp_root::<F>(meta);
        let q_cp_mu = meta.selector();
        {
            let s_all_c = cp_join_c.s_all;
            let s_cln_c = cp_join_c.s_cln;
            let s_all_l = cp_join_l.s_all;
            let s_cln_l = cp_join_l.s_cln;
            let pred_o = check[1];
            let cln_o = cflag[1];
            let mu_all = cp_root.mu_all;
            let mu_cln = cp_root.mu_cln;
            meta.create_gate("cp: root multiplicities over orders", move |m| {
                let q = m.query_selector(q_cp_mu);
                let all = m.query_advice(mu_all, Rotation::cur())
                    - m.query_advice(pred_o, Rotation::cur())
                        * m.query_advice(s_all_c, Rotation::cur())
                        * m.query_advice(s_all_l, Rotation::cur());
                let cln = m.query_advice(mu_cln, Rotation::cur())
                    - m.query_advice(cln_o, Rotation::cur())
                        * m.query_advice(s_cln_c, Rotation::cur())
                        * m.query_advice(s_cln_l, Rotation::cur());
                vec![q.clone() * all, q * cln]
            });
        }

        // ================= aggregation over the clean instance =================
        // The operators read the clean instance off the committed rows. The
        // group-by needs each clean orderkey to occupy one contiguous run of a
        // sorted view, and the deselected rows must not open runs of their own,
        // so the sorted view is over the selector-masked key: a deselected row
        // carries PAD_OK, joins the trailing PAD run and is never a group end,
        // hence emits nothing. Its extendedprice needs no masking, since its
        // contribution lands only in the PAD run's running sum, which is never
        // emitted.
        let l_key_sel = meta.advice_column();
        {
            let q_l = q_enable[2];
            let c = cflag[2];
            let k = lineitem[0];
            meta.create_gate("masked lineitem key: c ? l_orderkey : PAD", move |m| {
                let q = m.query_selector(q_l);
                let c = m.query_advice(c, Rotation::cur());
                let k = m.query_advice(k, Rotation::cur());
                let ks = m.query_advice(l_key_sel, Rotation::cur());
                let one = Expression::Constant(F::ONE);
                let pad = Expression::Constant(F::from(PAD_OK));
                vec![q * (ks - (c.clone() * k + (one - c) * pad))]
            });
        }

        let q_line = meta.selector();
        let q_first = meta.selector();
        let q_accu = meta.selector();
        let q_res_lookup = meta.complex_selector();
        let q_sort_res = meta.selector();

        // the sorted view carries only what the group-by reads
        let l_sorted = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();

        let line_rev = meta.advice_column();
        let run_sum = meta.advice_column();
        let is_last = meta.advice_column();

        let res_pad = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let res_sorted = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();

        // perm: (masked key, extendedprice, discount) <-> l_sorted
        let q_perm_l_in = meta.complex_selector();
        let q_perm_l_out = meta.complex_selector();
        let perm_lsort = PermAnyChip::configure(
            meta,
            q_perm_l_in,
            q_perm_l_out,
            vec![l_key_sel, lineitem[1], lineitem[2]],
            l_sorted.clone(),
        );

        // perm: res_pad <-> res_sorted
        let q_perm_r_in = meta.complex_selector();
        let q_perm_r_out = meta.complex_selector();
        let perm_res = PermAnyChip::configure(
            meta,
            q_perm_r_in,
            q_perm_r_out,
            res_pad.clone(),
            res_sorted.clone(),
        );

        let aux_same_prev = meta.advice_column();
        let aux_same_next = meta.advice_column();
        let aux_rev_eq = meta.advice_column();
        let aux_date_eq = meta.advice_column();

        let iz_same_prev = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_accu),
            |m| {
                m.query_advice(l_sorted[0], Rotation::cur())
                    - m.query_advice(l_sorted[0], Rotation::prev())
            },
            aux_same_prev,
        );

        let iz_same_next = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_line),
            |m| {
                m.query_advice(l_sorted[0], Rotation::next())
                    - m.query_advice(l_sorted[0], Rotation::cur())
            },
            aux_same_next,
        );

        // The group-end indicator, materialized. `iz_same_next.expr()` is a
        // degree-2 expression, and a lookup costs `2 + input_degree +
        // table_degree`: with the attach lookup's table side now gated by the
        // selector as well as by its row selector, reading the indicator as an
        // expression on the input side would push that lookup to degree 9 and
        // double every extended-domain FFT of the prover. One advice column and
        // one degree-3 gate keep the whole circuit at the degree 8 that
        // `q3_obj.rs` already carries.
        meta.create_gate("is_last = 1 - same_next", |m| {
            let q = m.query_selector(q_line);
            let one = Expression::Constant(F::ONE);
            vec![q * (m.query_advice(is_last, Rotation::cur()) - (one - iz_same_next.expr()))]
        });

        // ---- l_sorted is really sorted on the masked orderkey ----
        // The shuffle only proves l_sorted is a permutation of the masked
        // column; without an ordering constraint a prover could place one
        // orderkey in two non-adjacent runs and emit one res_pad row per run,
        // each carrying a partial revenue. Force nondecreasing, which makes
        // every key occupy exactly one contiguous run. The Lt chip shares the
        // u8 range column of the Cardinality Preservation Check.
        let q_lsort = meta.selector();
        let aux_lsort_eq = meta.advice_column();
        let iz_lsort_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_lsort),
            |m| {
                m.query_advice(l_sorted[0], Rotation::next())
                    - m.query_advice(l_sorted[0], Rotation::cur())
            },
            aux_lsort_eq,
        );
        let lt_lsort_cur_next = LtChip::<F, NUM_BYTES>::configure_with_u8(
            meta,
            cp_u8,
            |m| m.query_selector(q_lsort),
            |m| m.query_advice(l_sorted[0], Rotation::cur()),
            |m| m.query_advice(l_sorted[0], Rotation::next()),
        );
        meta.create_gate("l_sorted okey nondecreasing", |m| {
            let q = m.query_selector(q_lsort);
            let le = lt_lsort_cur_next.is_lt(m, None) + iz_lsort_eq.expr();
            vec![q * (le - Expression::Constant(F::ONE))]
        });

        // The group-boundary detector on the last real row reads l_sorted[0] at
        // row n, which neither the shuffle nor the nondecreasing gate touches.
        // Left free, a prover sets it equal to the last key; iz_same_next then
        // reports "same group" on row n-1, the emit gate writes PAD instead of
        // that group, and the group with the largest clean orderkey vanishes
        // from the answer with every other constraint still satisfied. Pin the
        // value, as `cp: sorted view sentinel is PAD` does in card_preserve.rs.
        let q_lsort_sentinel = meta.selector();
        {
            let key = l_sorted[0];
            meta.create_gate("l_sorted sentinel key is PAD", move |m| {
                let q = m.query_selector(q_lsort_sentinel);
                vec![
                    q * (m.query_advice(key, Rotation::cur())
                        - Expression::Constant(F::from(PAD_OK))),
                ]
            });
        }

        // res_sorted: [0]=okey,[1]=odate,[2]=ship,[3]=rev
        let iz_rev_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| {
                m.query_advice(res_sorted[3], Rotation::cur())
                    - m.query_advice(res_sorted[3], Rotation::next())
            },
            aux_rev_eq,
        );

        let iz_date_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| {
                m.query_advice(res_sorted[1], Rotation::cur())
                    - m.query_advice(res_sorted[1], Rotation::next())
            },
            aux_date_eq,
        );

        let lt_rev_next_cur = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| m.query_advice(res_sorted[3], Rotation::next()), // rev_next
            |m| m.query_advice(res_sorted[3], Rotation::cur()),  // rev_cur
        );

        let lt_date_cur_next = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort_res),
            |m| m.query_advice(res_sorted[1], Rotation::cur()), // date_cur
            |m| m.query_advice(res_sorted[1], Rotation::next()), // date_next
        );

        // line_rev = ext * (SCALE - disc)
        meta.create_gate("line_rev", |m| {
            let q = m.query_selector(q_line);
            let ext = m.query_advice(l_sorted[1], Rotation::cur());
            let disc = m.query_advice(l_sorted[2], Rotation::cur());
            let lr = m.query_advice(line_rev, Rotation::cur());
            let scale = Expression::Constant(F::from(SCALE));
            vec![q * (lr - ext * (scale - disc))]
        });

        // run_sum[0] = line_rev[0]
        meta.create_gate("run_sum_first", |m| {
            let q = m.query_selector(q_first);
            let rs = m.query_advice(run_sum, Rotation::cur());
            let lr = m.query_advice(line_rev, Rotation::cur());
            vec![q * (rs - lr)]
        });

        // run_sum[i] = same_prev * run_sum[i-1] + line_rev[i]
        meta.create_gate("run_sum_accu", |m| {
            let q = m.query_selector(q_accu);
            let same = iz_same_prev.expr(); // 1 if same group
            let rs_cur = m.query_advice(run_sum, Rotation::cur());
            let rs_prev = m.query_advice(run_sum, Rotation::prev());
            let lr = m.query_advice(line_rev, Rotation::cur());
            vec![q * (rs_cur - (same * rs_prev + lr))]
        });

        // emit group row only at the last row of each orderkey group
        meta.create_gate("emit_res_pad", |m| {
            let q = m.query_selector(q_line);
            let one = Expression::Constant(F::ONE);
            let is_last = m.query_advice(is_last, Rotation::cur()); // 1 if next != cur
            let not_last = one.clone() - is_last.clone();

            let cur_okey = m.query_advice(l_sorted[0], Rotation::cur());
            let rs = m.query_advice(run_sum, Rotation::cur());

            let out_okey = m.query_advice(res_pad[0], Rotation::cur());
            let out_date = m.query_advice(res_pad[1], Rotation::cur());
            let out_ship = m.query_advice(res_pad[2], Rotation::cur());
            let out_rev = m.query_advice(res_pad[3], Rotation::cur());

            let pad_ok = Expression::Constant(F::from(PAD_OK));
            let pad_date = Expression::Constant(F::from(PAD_DATE));
            let pad_ship = Expression::Constant(F::from(PAD_SHIP));
            let pad_rev = Expression::Constant(F::from(PAD_REV));

            vec![
                // okey / rev forced in both cases
                q.clone() * (out_okey - (is_last.clone() * cur_okey + not_last.clone() * pad_ok)),
                q.clone() * (out_rev - (is_last.clone() * rs + not_last.clone() * pad_rev)),
                // date/ship must be PAD when not last; last rows are constrained by lookup below
                q.clone() * not_last.clone() * (out_date - pad_date),
                q * not_last * (out_ship - pad_ship),
            ]
        });

        // Lookup: (orderkey, orderdate, shippriority) must name a SELECTED row
        // of orders. The table side spans the whole relation and is gated by
        // the selector, so a deselected orders row contributes (0, 0, 0), which
        // is what a gated-off input row reads; the shift by one keeps that
        // dummy tuple away from any real one.
        {
            let cln_o = cflag[1];
            let ok_col = orders[3];
            let od_col = orders[0];
            let sp_col = orders[1];
            let q_tbl = q_row[1];
            let rp = res_pad.clone();
            meta.lookup_any("attach orders attrs to res_pad", move |m| {
                let q_in = m.query_selector(q_res_lookup);
                let one = Expression::Constant(F::ONE);
                // the group-end indicator, on the same l_sorted rows
                let gate = q_in * m.query_advice(is_last, Rotation::cur());

                let tbl = m.query_selector(q_tbl) * m.query_advice(cln_o, Rotation::cur());

                vec![
                    (
                        gate.clone() * (m.query_advice(rp[0], Rotation::cur()) + one.clone()),
                        tbl.clone() * (m.query_advice(ok_col, Rotation::cur()) + one.clone()),
                    ),
                    (
                        gate.clone() * (m.query_advice(rp[1], Rotation::cur()) + one.clone()),
                        tbl.clone() * (m.query_advice(od_col, Rotation::cur()) + one.clone()),
                    ),
                    (
                        gate * (m.query_advice(rp[2], Rotation::cur()) + one.clone()),
                        tbl * (m.query_advice(sp_col, Rotation::cur()) + one),
                    ),
                ]
            });
        }

        // ORDER BY gate on res_sorted
        meta.create_gate("ORDER BY revenue DESC, o_orderdate ASC", |m| {
            let q = m.query_selector(q_sort_res);

            let rev_gt = lt_rev_next_cur.is_lt(m, None); // next < cur
            let rev_eq = iz_rev_eq.expr();

            let date_lt = lt_date_cur_next.is_lt(m, None); // cur < next
            let date_eq = iz_date_eq.expr();
            let date_le = date_lt + date_eq;

            vec![q * (rev_gt + rev_eq * date_le - Expression::Constant(F::ONE))]
        });

        TestCircuitConfig {
            q_enable,
            q_row,

            customer,
            orders,
            lineitem,

            check,
            condition,
            q_cond_const,
            q_cond_link,

            row_idx,
            cons,
            cflag,

            lt_compare_condition,
            equal_condition,

            instance,
            instance_test,

            cp_agg_c,
            cp_agg_l,
            cp_join_c,
            cp_join_l,
            cp_root,
            q_cp_mu,

            l_key_sel,

            l_sorted,
            perm_lsort,
            q_line,
            q_first,
            q_accu,
            line_rev,
            run_sum,
            iz_same_prev,
            iz_same_next,
            is_last,
            q_lsort,
            q_lsort_sentinel,
            lt_lsort_cur_next,
            iz_lsort_eq,

            res_pad,
            q_res_lookup,

            res_sorted,
            perm_res,
            q_sort_res,
            lt_rev_next_cur,
            lt_date_cur_next,
            iz_rev_eq,
            iz_date_eq,
        }
    }

    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        customer: Vec<Vec<u64>>,
        orders: Vec<Vec<u64>>,
        lineitem: Vec<Vec<u64>>,
        condition: [u64; 2],
    ) -> Result<AssignedCell<F, F>, Error> {
        // chips
        let equal_chip = IsZeroChip::construct(self.config.equal_condition[0].clone());

        let lt_o_chip = LtChip::construct(self.config.lt_compare_condition[0].clone());
        lt_o_chip.load(layouter)?;

        let lt_l_chip = LtChip::construct(self.config.lt_compare_condition[1].clone());
        lt_l_chip.load(layouter)?;

        // Every Lt chip of the Cardinality Preservation Check, and the sorted
        // lineitem view with it, shares one u8 fixed column, so a single load
        // covers them all.
        LtChip::<F, NUM_BYTES>::construct(self.config.cp_agg_c.lt_key_cur_next).load(layouter)?;

        let iz_same_prev_chip = IsZeroChip::construct(self.config.iz_same_prev.clone());
        let iz_same_next_chip = IsZeroChip::construct(self.config.iz_same_next.clone());
        let iz_lsort_eq_chip = IsZeroChip::construct(self.config.iz_lsort_eq.clone());
        let lt_lsort_chip =
            LtChip::<F, NUM_BYTES>::construct(self.config.lt_lsort_cur_next.clone());
        let iz_rev_eq_chip = IsZeroChip::construct(self.config.iz_rev_eq.clone());
        let iz_date_eq_chip = IsZeroChip::construct(self.config.iz_date_eq.clone());

        let lt_rev_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_rev_next_cur.clone());
        lt_rev_chip.load(layouter)?;
        let lt_date_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_date_cur_next.clone());
        lt_date_chip.load(layouter)?;

        let Witness {
            n_c,
            n_o,
            n_l,
            c_check,
            o_check,
            l_check,
            cln_c,
            cln_o,
            cln_l,
            l_key_sel_u64,
            l_sorted_u64,
            line_rev_u64,
            run_sum_u64,
            is_last_u64,
            res_pad_u64,
            res_sorted_u64,
            all_clean,
            tamper,
        } = build_witness(&customer, &orders, &lineitem, condition);

        layouter.assign_region(
            || "witness",
            |mut region| {
                // ---------------- base tables, predicate bits, selectors ----------------
                for i in 0..n_c {
                    self.config.q_enable[0].enable(&mut region, i)?;
                    self.config.q_row[0].enable(&mut region, i)?;
                    for j in 0..2 {
                        region.assign_advice(
                            || "customer",
                            self.config.customer[j],
                            i,
                            || Value::known(F::from(customer[i][j])),
                        )?;
                    }
                    region.assign_advice(
                        || "check0",
                        self.config.check[0],
                        i,
                        || Value::known(F::from(c_check[i])),
                    )?;
                    region.assign_advice(
                        || "cond0",
                        self.config.condition[0],
                        i,
                        || Value::known(F::from(condition[0])),
                    )?;
                    region.assign_advice(
                        || "cflag customer",
                        self.config.cflag[0],
                        i,
                        || Value::known(F::from(cln_c[i])),
                    )?;
                }

                for i in 0..n_o {
                    self.config.q_enable[1].enable(&mut region, i)?;
                    self.config.q_row[1].enable(&mut region, i)?;
                    for j in 0..4 {
                        region.assign_advice(
                            || "orders",
                            self.config.orders[j],
                            i,
                            || Value::known(F::from(orders[i][j])),
                        )?;
                    }
                    region.assign_advice(
                        || "check1",
                        self.config.check[1],
                        i,
                        || Value::known(F::from(o_check[i])),
                    )?;
                    region.assign_advice(
                        || "cond1",
                        self.config.condition[1],
                        i,
                        || Value::known(F::from(condition[1])),
                    )?;
                    region.assign_advice(
                        || "cflag orders",
                        self.config.cflag[1],
                        i,
                        || Value::known(F::from(cln_o[i])),
                    )?;
                }

                for i in 0..n_l {
                    self.config.q_enable[2].enable(&mut region, i)?;
                    self.config.q_row[2].enable(&mut region, i)?;
                    for j in 0..4 {
                        region.assign_advice(
                            || "lineitem",
                            self.config.lineitem[j],
                            i,
                            || Value::known(F::from(lineitem[i][j])),
                        )?;
                    }
                    region.assign_advice(
                        || "check2",
                        self.config.check[2],
                        i,
                        || Value::known(F::from(l_check[i])),
                    )?;
                    region.assign_advice(
                        || "cond2",
                        self.config.condition[2],
                        i,
                        || Value::known(F::from(condition[1])),
                    )?;
                    region.assign_advice(
                        || "cflag lineitem",
                        self.config.cflag[2],
                        i,
                        || Value::known(F::from(cln_l[i])),
                    )?;
                    region.assign_advice(
                        || "l_key_sel",
                        self.config.l_key_sel,
                        i,
                        || Value::known(F::from(l_key_sel_u64[i])),
                    )?;
                }

                // the query parameters: constant down each condition column,
                // and one shared date for the two date predicates
                for (idx, len) in [n_c, n_o, n_l].iter().enumerate() {
                    for i in 0..len.saturating_sub(1) {
                        self.config.q_cond_const[idx].enable(&mut region, i)?;
                    }
                }
                if n_o > 0 && n_l > 0 {
                    self.config.q_cond_link.enable(&mut region, 0)?;
                }

                // ---------------- predicate subchips ----------------
                for i in 0..n_c {
                    equal_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(customer[i][0]) - F::from(condition[0])),
                    )?;
                }
                for i in 0..n_o {
                    lt_o_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(orders[i][0])),
                        Value::known(F::from(condition[1])),
                    )?;
                }
                for i in 0..n_l {
                    lt_l_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(condition[1])),
                        Value::known(F::from(lineitem[i][3])),
                    )?;
                }

                // ===================== (1) CONSERVATION CHECK =====================
                // The indexed relation against the two parts. Nothing else has
                // to be assigned for it: the entries are the committed rows and
                // the indicator, both already written above.
                assign_row_index(&mut region, &self.config.row_idx, n_c.max(n_o).max(n_l))?;
                for (idx, (rows, flags)) in
                    [(&customer, &cln_c), (&orders, &cln_o), (&lineitem, &cln_l)]
                        .into_iter()
                        .enumerate()
                {
                    assign_conserve(&mut region, &self.config.cons[idx], rows, flags)?;
                }

                // ===================== CARDINALITY PRESERVATION CHECK =====================
                // Both children are leaves, so a child row's input-channel
                // multiplicity is its predicate bit and its clean-channel
                // multiplicity is its selector bit.
                let cp_rows_c: Vec<[u64; 3]> = (0..n_c)
                    .map(|i| [customer[i][1], c_check[i], cln_c[i]])
                    .collect();
                let cp_rows_l: Vec<[u64; 3]> = (0..n_l)
                    .map(|i| [lineitem[i][0], l_check[i], cln_l[i]])
                    .collect();

                let cp_stage_c = build_cp_stage(&cp_rows_c, MAX_SENTINEL);
                let cp_stage_l = build_cp_stage(&cp_rows_l, MAX_SENTINEL);

                assign_cp_agg(&mut region, &self.config.cp_agg_c, &cp_rows_c, &cp_stage_c)?;
                assign_cp_agg(&mut region, &self.config.cp_agg_l, &cp_rows_l, &cp_stage_l)?;

                // parent side, on the rows of orders
                let o_custkeys: Vec<u64> = orders.iter().map(|o| o[2]).collect();
                let o_orderkeys: Vec<u64> = orders.iter().map(|o| o[3]).collect();
                let fetched_c = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_c,
                    &o_custkeys,
                    &cp_stage_c,
                    MAX_SENTINEL,
                )?;
                let fetched_l = assign_cp_join(
                    &mut region,
                    &self.config.cp_join_l,
                    &o_orderkeys,
                    &cp_stage_l,
                    MAX_SENTINEL,
                )?;

                // root multiplicities and the equality between the two sums
                let cp_mu: Vec<(u64, u64)> = (0..n_o)
                    .map(|i| {
                        (
                            o_check[i] * fetched_c[i].0 * fetched_l[i].0,
                            cln_o[i] * fetched_c[i].1 * fetched_l[i].1,
                        )
                    })
                    .collect();
                for i in 0..n_o {
                    self.config.q_cp_mu.enable(&mut region, i)?;
                }
                let (cp_all, cp_cln) = assign_cp_root(&mut region, &self.config.cp_root, &cp_mu)?;
                if !tamper && !all_clean && !SELECT_A_FILTERED_ROW.load(Ordering::Relaxed) {
                    debug_assert_eq!(
                        cp_all, cp_cln,
                        "cardinality preservation: |R^c join| != |R^p join|"
                    );
                }

                // ===================== AGGREGATION =====================
                // the sorted view, plus the pinned PAD sentinel at row n_l
                for i in 0..n_l {
                    for j in 0..3 {
                        region.assign_advice(
                            || "l_sorted",
                            self.config.l_sorted[j],
                            i,
                            || Value::known(F::from(l_sorted_u64[i][j])),
                        )?;
                    }
                }
                for j in 0..3 {
                    let v = if j == 0 { PAD_OK } else { 0u64 };
                    region.assign_advice(
                        || "l_sorted_sentinel",
                        self.config.l_sorted[j],
                        n_l,
                        || Value::known(F::from(v)),
                    )?;
                }
                self.config.q_lsort_sentinel.enable(&mut region, n_l)?;

                for i in 0..n_l {
                    region.assign_advice(
                        || "line_rev",
                        self.config.line_rev,
                        i,
                        || Value::known(F::from(line_rev_u64[i])),
                    )?;
                    region.assign_advice(
                        || "run_sum",
                        self.config.run_sum,
                        i,
                        || Value::known(F::from(run_sum_u64[i])),
                    )?;
                    region.assign_advice(
                        || "is_last",
                        self.config.is_last,
                        i,
                        || Value::known(F::from(is_last_u64[i])),
                    )?;

                    let rp = res_pad_u64[i];
                    for j in 0..4 {
                        region.assign_advice(
                            || "res_pad",
                            self.config.res_pad[j],
                            i,
                            || Value::known(F::from(rp[j])),
                        )?;
                    }

                    let rs = res_sorted_u64[i];
                    for j in 0..4 {
                        region.assign_advice(
                            || "res_sorted",
                            self.config.res_sorted[j],
                            i,
                            || Value::known(F::from(rs[j])),
                        )?;
                    }
                }

                // permutation selectors: masked lineitem columns <-> l_sorted,
                // and res_pad <-> res_sorted
                for i in 0..n_l {
                    self.config.perm_lsort.q_perm1.enable(&mut region, i)?;
                    self.config.perm_lsort.q_perm2.enable(&mut region, i)?;
                    self.config.perm_res.q_perm1.enable(&mut region, i)?;
                    self.config.perm_res.q_perm2.enable(&mut region, i)?;
                }

                // line / accumulate / emit selectors
                if n_l > 0 {
                    self.config.q_first.enable(&mut region, 0)?;
                }
                for i in 0..n_l {
                    self.config.q_line.enable(&mut region, i)?;
                    self.config.q_res_lookup.enable(&mut region, i)?;
                }
                for i in 1..n_l {
                    self.config.q_accu.enable(&mut region, i)?;
                }
                for i in 0..n_l.saturating_sub(1) {
                    self.config.q_sort_res.enable(&mut region, i)?;
                }

                // same_prev only for i>=1
                for i in 1..n_l {
                    let diff = F::from(l_sorted_u64[i][0]) - F::from(l_sorted_u64[i - 1][0]);
                    iz_same_prev_chip.assign(&mut region, i, Value::known(diff))?;
                }
                // same_next for i=0..n-1 (needs the sentinel row assigned)
                for i in 0..n_l {
                    let next_ok = if i + 1 < n_l {
                        l_sorted_u64[i + 1][0]
                    } else {
                        PAD_OK
                    };
                    let diff = F::from(next_ok) - F::from(l_sorted_u64[i][0]);
                    iz_same_next_chip.assign(&mut region, i, Value::known(diff))?;
                }

                // sortedness of l_sorted[0]: real consecutive pairs only. Row
                // n_l is the pinned sentinel, which only iz_same_next reads.
                for i in 0..n_l.saturating_sub(1) {
                    self.config.q_lsort.enable(&mut region, i)?;
                    let cur = l_sorted_u64[i][0];
                    let next = l_sorted_u64[i + 1][0];
                    lt_lsort_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(cur)),
                        Value::known(F::from(next)),
                    )?;
                    iz_lsort_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(next) - F::from(cur)),
                    )?;
                }

                // ORDER BY helpers on res_sorted: rows 0..n-2
                for i in 0..n_l.saturating_sub(1) {
                    let rev_cur = res_sorted_u64[i][3];
                    let rev_next = res_sorted_u64[i + 1][3];
                    let date_cur = res_sorted_u64[i][1];
                    let date_next = res_sorted_u64[i + 1][1];

                    iz_rev_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(rev_cur) - F::from(rev_next)),
                    )?;
                    iz_date_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(date_cur) - F::from(date_next)),
                    )?;

                    lt_rev_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(rev_next)),
                        Value::known(F::from(rev_cur)),
                    )?;
                    lt_date_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(date_cur)),
                        Value::known(F::from(date_next)),
                    )?;
                }

                // public output
                let out = region.assign_advice(
                    || "instance_test",
                    self.config.instance_test,
                    0,
                    || Value::known(F::from(1)),
                )?;
                Ok(out)
            },
        )
    }

    pub fn expose_public(
        &self,
        layouter: &mut impl Layouter<F>,
        cell: AssignedCell<F, F>,
        row: usize,
    ) -> Result<(), Error> {
        layouter.constrain_instance(cell.cell(), self.config.instance, row)
    }
}

// ---------------- Circuit wrapper ----------------
pub struct MyCircuit<F> {
    pub customer: Vec<Vec<u64>>,
    pub orders: Vec<Vec<u64>>,
    pub lineitem: Vec<Vec<u64>>,
    pub condition: [u64; 2],
    pub _marker: PhantomData<F>,
}

impl<F: Copy + Default> Default for MyCircuit<F> {
    fn default() -> Self {
        Self {
            customer: Vec::new(),
            orders: Vec::new(),
            lineitem: Vec::new(),
            condition: [Default::default(); 2],
            _marker: PhantomData,
        }
    }
}

impl<F: Field + Ord> Circuit<F> for MyCircuit<F> {
    type Config = TestCircuitConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        TestChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = TestChip::construct(config);

        let out_cell = chip.assign(
            &mut layouter,
            self.customer.clone(),
            self.orders.clone(),
            self.lineitem.clone(),
            self.condition,
        )?;

        chip.expose_public(&mut layouter, out_cell, 0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::MyCircuit;
    use crate::data::data_processing;
    use chrono::{DateTime, NaiveDate, Utc};
    use halo2_proofs::dev::{MockProver, VerifyFailure};
    use std::marker::PhantomData;
    use std::sync::atomic::Ordering;

    use halo2_proofs::{
        plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit},
        poly::{
            commitment::Params,
            ipa::{
                commitment::{IPACommitmentScheme, ParamsIPA},
                multiopen::ProverIPA,
                strategy::SingleStrategy,
            },
            VerificationStrategy,
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
    };
    use halo2curves::pasta::{vesta, EqAffine, Fp};
    use rand::rngs::OsRng;
    use std::time::Instant;
    use std::{fs::File, io::Write, path::Path};

    fn generate_and_verify_proof<C: Circuit<Fp>>(
        circuit: C,
        public_input: &[Fp],
        proof_path: &str,
    ) {
        let params_path = &crate::paths::param_file(16);
        let mut fd = std::fs::File::open(&params_path).unwrap();
        let params = ParamsIPA::<vesta::Affine>::read(&mut fd).unwrap();

        let t0 = Instant::now();
        let vk = keygen_vk(&params, &circuit).expect("keygen_vk should not fail");
        println!("Time to generate vk {:?}", t0.elapsed());

        let t1 = Instant::now();
        let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk should not fail");
        println!("Time to generate pk {:?}", t1.elapsed());

        let mut rng = OsRng;
        let mut transcript = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);
        create_proof::<IPACommitmentScheme<_>, ProverIPA<_>, _, _, _, _>(
            &params,
            &pk,
            &[circuit],
            &[&[public_input]],
            &mut rng,
            &mut transcript,
        )
        .expect("proof generation should not fail");
        let proof = transcript.finalize();

        File::create(Path::new(proof_path))
            .expect("Failed to create proof file")
            .write_all(&proof)
            .expect("Failed to write proof");
        println!("Proof written to: {}", proof_path);

        let strategy = SingleStrategy::new(&params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        assert!(
            verify_proof(
                &params,
                pk.get_vk(),
                strategy,
                &[&[public_input]],
                &mut transcript
            )
            .is_ok(),
            "Proof verification failed"
        );
    }

    /// The three tamper hooks are process-wide statics and `cargo test` runs
    /// the tests of one binary in parallel, so every test that reads or writes
    /// them takes this lock first.
    static HOOKS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn lock_hooks() -> std::sync::MutexGuard<'static, ()> {
        HOOKS.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn string_to_u64(s: &str) -> u64 {
        let mut result = 0;
        for (i, c) in s.chars().enumerate() {
            result += (i as u64 + 1) * (c as u64);
        }
        result
    }
    fn scale_by_1000(x: f64) -> u64 {
        (1000.0 * x) as u64
    }
    fn date_to_timestamp(date_str: &str) -> u64 {
        match NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
            Ok(date) => {
                let datetime: DateTime<Utc> = DateTime::<Utc>::from_utc(date.and_hms(0, 0, 0), Utc);
                datetime.timestamp() as u64
            }
            Err(_) => 0,
        }
    }

    /// The truncated dataset slice the fast tests share: small enough for
    /// MockProver and for one real proof, large enough that the reduction
    /// actually drops tuples on all three relations.
    fn small_slice() -> (Vec<Vec<u64>>, Vec<Vec<u64>>, Vec<Vec<u64>>, [u64; 2]) {
        const N_CUST: usize = 300;
        const N_ORD: usize = 2000;
        const N_LINE: usize = 8000;

        let mut customer: Vec<Vec<u64>> = Vec::new();
        let mut orders: Vec<Vec<u64>> = Vec::new();
        let mut lineitem: Vec<Vec<u64>> = Vec::new();

        if let Ok(records) = data_processing::customer_read_records_from_file(
            &crate::paths::data_file("customer.tbl"),
        ) {
            customer = records
                .iter()
                .take(N_CUST)
                .map(|record| vec![string_to_u64(&record.c_mktsegment), record.c_custkey])
                .collect();
        }
        if let Ok(records) =
            data_processing::orders_read_records_from_file(&crate::paths::data_file("orders.tbl"))
        {
            orders = records
                .iter()
                .take(N_ORD)
                .map(|record| {
                    vec![
                        date_to_timestamp(&record.o_orderdate),
                        record.o_shippriority,
                        record.o_custkey,
                        record.o_orderkey,
                    ]
                })
                .collect();
        }
        if let Ok(records) = data_processing::lineitem_read_records_from_file(
            &crate::paths::data_file("lineitem.tbl"),
        ) {
            lineitem = records
                .iter()
                .take(N_LINE)
                .map(|record| {
                    vec![
                        record.l_orderkey,
                        scale_by_1000(record.l_extendedprice),
                        scale_by_1000(record.l_discount),
                        date_to_timestamp(&record.l_shipdate),
                    ]
                })
                .collect();
        }

        assert!(
            !customer.is_empty() && !orders.is_empty() && !lineitem.is_empty(),
            "dataset files not found under {}",
            crate::paths::data_file("customer.tbl")
        );

        let condition = [string_to_u64("HOUSEHOLD"), date_to_timestamp("1995-03-25")];

        (customer, orders, lineitem, condition)
    }

    /// The full dataset, one real proof or one MockProver run under
    /// `VPJOIN_MOCK=1`, exactly as `q3_obj.rs::test_1` does for the previous
    /// realization.
    #[test]
    #[ignore = "full-scale end-to-end run; the fast check is test_one_pass_conditions"]
    fn test_full() {
        let k = 16;

        let mut customer: Vec<Vec<u64>> = Vec::new();
        let mut orders: Vec<Vec<u64>> = Vec::new();
        let mut lineitem: Vec<Vec<u64>> = Vec::new();

        if let Ok(records) = data_processing::customer_read_records_from_file(
            &crate::paths::data_file("customer.tbl"),
        ) {
            customer = records
                .iter()
                .map(|record| vec![string_to_u64(&record.c_mktsegment), record.c_custkey])
                .collect();
        }
        if let Ok(records) =
            data_processing::orders_read_records_from_file(&crate::paths::data_file("orders.tbl"))
        {
            orders = records
                .iter()
                .map(|record| {
                    vec![
                        date_to_timestamp(&record.o_orderdate),
                        record.o_shippriority,
                        record.o_custkey,
                        record.o_orderkey,
                    ]
                })
                .collect();
        }
        if let Ok(records) = data_processing::lineitem_read_records_from_file(
            &crate::paths::data_file("lineitem.tbl"),
        ) {
            lineitem = records
                .iter()
                .map(|record| {
                    vec![
                        record.l_orderkey,
                        scale_by_1000(record.l_extendedprice),
                        scale_by_1000(record.l_discount),
                        date_to_timestamp(&record.l_shipdate),
                    ]
                })
                .collect();
        }

        let condition = [string_to_u64("HOUSEHOLD"), date_to_timestamp("1995-03-25")];

        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            condition,
            _marker: PhantomData,
        };

        let public_input = vec![Fp::from(1)];

        let mock = std::env::var("VPJOIN_MOCK")
            .map(|v| v == "1")
            .unwrap_or(false);

        if mock {
            let prover = MockProver::run(k, &circuit, vec![public_input]).unwrap();
            prover.assert_satisfied();
        } else {
            let proof_path = &crate::paths::proof_file("proof_obj_q3_new");
            generate_and_verify_proof(circuit, &public_input, proof_path);
        }
    }

    /// The real prover, not MockProver, on a truncated slice. MockProver checks
    /// every constraint but tolerates cells that are never read; this closes
    /// that gap by generating and verifying an actual proof.
    #[test]
    #[ignore = "real IPA proof, minutes in a debug build; run explicitly to check provability"]
    fn test_real_proof_small() {
        let (customer, orders, lineitem, condition) = small_slice();
        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            condition,
            _marker: PhantomData,
        };
        let t = Instant::now();
        generate_and_verify_proof(
            circuit,
            &[Fp::from(1)],
            &crate::paths::proof_file("proof_obj_q3_new_small"),
        );
        println!("real proof of the truncated slice took {:?}", t.elapsed());
    }

    /// Cost probe. Every condition of the revised gate is a degree-1 to
    /// degree-3 expression under its own selector, so none of them may raise
    /// the maximum gate degree above what `q3_obj.rs` already carries: a rise
    /// would double every FFT of the prover.
    #[test]
    fn test_max_gate_degree() {
        use halo2_proofs::plonk::ConstraintSystem;

        let mut cs = ConstraintSystem::<Fp>::default();
        let _ = <MyCircuit<Fp> as Circuit<Fp>>::configure(&mut cs);
        println!(
            "advice {} fixed {} selectors {} degree {} gates {} polys {} lookups {} shuffles {}",
            cs.num_advice_columns(),
            cs.num_fixed_columns(),
            cs.num_selectors(),
            cs.degree(),
            cs.gates().len(),
            cs.gates()
                .iter()
                .map(|g| g.polynomials().len())
                .sum::<usize>(),
            cs.lookups().len(),
            cs.shuffles().len(),
        );
        assert!(
            cs.degree() <= 8,
            "the maximum gate degree rose to {}",
            cs.degree()
        );
    }

    /// Fast correctness check of the three conditions: a truncated slice under
    /// MockProver, which verifies every gate, shuffle and lookup of the circuit
    /// without paying for a real proof, plus the two tampered witnesses that
    /// separate what (2) catches from what (3) catches.
    #[test]
    fn test_one_pass_conditions() {
        let _hooks = lock_hooks();
        let k = 15;
        let (customer, orders, lineitem, condition) = small_slice();

        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            condition,
            _marker: PhantomData,
        };

        let prover = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        prover.assert_satisfied();

        // One participating order deselected, its neighbours re-reduced around
        // it, so the Selector Check and Pairwise Consistency both still hold.
        // Only the Cardinality Preservation Check can see this.
        super::HIDE_ONE_CLEAN_TUPLE.store(true, Ordering::Relaxed);
        let tampered = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        let verdict = tampered.verify();
        super::HIDE_ONE_CLEAN_TUPLE.store(false, Ordering::Relaxed);

        let failures = verdict.expect_err("condition (3) accepted a hidden participating tuple");
        assert!(
            failures
                .iter()
                .any(|f| format!("{:?}", f).contains("cardinality preservation")),
            "the circuit rejected, but not through the Cardinality Preservation Check: {:?}",
            failures
        );

        // No reduction at all: every row that passes its predicate selected.
        // Both channels of (3) then agree row by row, so this is the escape
        // that Pairwise Consistency exists to close.
        super::MARK_ALL_CLEAN.store(true, Ordering::Relaxed);
        let unreduced = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        let verdict = unreduced.verify();
        super::MARK_ALL_CLEAN.store(false, Ordering::Relaxed);

        let failures = verdict.expect_err("condition (2) accepted an unreduced clean instance");
        assert!(
            failures.iter().any(|f| matches!(
                f,
                VerifyFailure::Lookup { name, .. } if name.starts_with("pw: ")
            )),
            "the circuit rejected, but not through a Pairwise Consistency lookup: {:?}",
            failures
        );
    }

    /// The emitted answer against a direct evaluation of Q3 over the same
    /// slice. This is what the restructured aggregation has to preserve: the
    /// sorted view now covers every lineitem row rather than the clean ones
    /// alone, and the group boundaries come from the selector-masked key, so
    /// nothing but this check rules out a deselected row opening a group of its
    /// own or a clean group being split by one.
    #[test]
    fn test_answer_matches_sql() {
        use std::collections::HashMap;
        let _hooks = lock_hooks();

        let (customer, orders, lineitem, condition) = small_slice();
        let (seg, d) = (condition[0], condition[1]);

        // c_custkey is the customer primary key, so a filtered order matches at
        // most one filtered customer. The group-by sums line revenue per
        // orderkey without a customer-multiplicity factor, exactly as
        // q3_obj.rs does, so the comparison below is meaningful only under that
        // key property. Assert it rather than assume it.
        let mut mult: HashMap<u64, u64> = HashMap::new();
        for c in customer.iter().filter(|c| c[0] == seg) {
            *mult.entry(c[1]).or_default() += 1;
        }
        assert!(
            mult.values().all(|&m| m <= 1),
            "the slice has a duplicate c_custkey, so Q3's SUM is not the plain per-orderkey sum"
        );

        let mut by_okey: HashMap<u64, (u64, u64, u128)> = HashMap::new();
        for o in orders.iter().filter(|o| o[0] < d) {
            if mult.get(&o[2]).copied().unwrap_or(0) == 0 {
                continue;
            }
            for l in lineitem.iter().filter(|l| l[0] == o[3] && l[3] > d) {
                let e = by_okey.entry(o[3]).or_insert((o[0], o[1], 0));
                e.2 += (l[1] as u128) * (1000u128 - l[2] as u128);
            }
        }

        let total = |r: &[u64; 4], s: &[u64; 4]| {
            s[3].cmp(&r[3]).then(r[1].cmp(&s[1])).then(r[0].cmp(&s[0]))
        };

        let mut expected: Vec<[u64; 4]> = by_okey
            .into_iter()
            .map(|(k, (od, sp, rev))| [k, od, sp, rev as u64])
            .collect();
        expected.sort_by(total);
        assert!(!expected.is_empty(), "the slice produced no answer rows");

        let w = super::build_witness(&customer, &orders, &lineitem, condition);
        let mut got: Vec<[u64; 4]> = w.res_sorted_u64[..expected.len()].to_vec();
        got.sort_by(total);

        assert_eq!(got, expected, "the answer differs from the query's");
        assert!(
            w.res_sorted_u64[expected.len()..]
                .iter()
                .all(|r| r[0] == super::PAD_OK),
            "a real group survived past the answer's length"
        );
    }

    /// The Conservation Check of condition (1). Moving one occurrence across
    /// the clean/residual boundary while leaving the flag pattern intact keeps
    /// every section size and every other constraint satisfied, so only the
    /// permutation argument can reject it.
    #[test]
    fn test_conservation_rejects_a_misplaced_occurrence() {
        let k = 15;
        let _hooks = lock_hooks();
        let (customer, orders, lineitem, condition) = small_slice();

        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            condition,
            _marker: PhantomData,
        };

        crate::circuits::conserve_idx::MISPLACE_ONE_OCCURRENCE.store(true, Ordering::Relaxed);
        let tampered = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        let verdict = tampered.verify();
        crate::circuits::conserve_idx::MISPLACE_ONE_OCCURRENCE.store(false, Ordering::Relaxed);

        let failures = verdict.expect_err("the Conservation Check accepted a misplaced occurrence");
        assert!(
            failures
                .iter()
                .any(|f| matches!(f, VerifyFailure::Shuffle { .. })),
            "the circuit rejected, but not through a Conservation permutation: {:?}",
            failures
        );
    }

    /// The predicate half of the Selector Check, which is what the revision
    /// added to the condition list. A row that fails its WHERE clause must not
    /// be selectable: without `c(t)(1 - b(t)) = 0` the input-side channel of
    /// (3) would count the raw join and a prover could certify a
    /// WHERE-violating row while every other check held.
    #[test]
    fn test_selector_implies_predicate() {
        let _hooks = lock_hooks();
        let k = 15;
        let (customer, orders, lineitem, condition) = small_slice();

        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            condition,
            _marker: PhantomData,
        };

        super::SELECT_A_FILTERED_ROW.store(true, Ordering::Relaxed);
        let tampered = MockProver::run(k, &circuit, vec![vec![Fp::from(1)]]).unwrap();
        let verdict = tampered.verify();
        super::SELECT_A_FILTERED_ROW.store(false, Ordering::Relaxed);

        let failures = verdict.expect_err("the Selector Check accepted a filtered-out row");
        assert!(
            failures.iter().any(|f| matches!(
                f,
                VerifyFailure::ConstraintNotSatisfied { constraint, .. }
                    if format!("{:?}", constraint).contains("implies its predicate")
            )),
            "the circuit rejected, but not through the Selector Check: {:?}",
            failures
        );
    }
}
