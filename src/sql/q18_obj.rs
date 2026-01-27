use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use halo2_proofs::{halo2curves::ff::PrimeField, plonk::Expression};
use std::marker::PhantomData;

use crate::chips::is_zero::{IsZeroChip, IsZeroConfig};
use crate::chips::less_than::{LtChip, LtConfig, LtInstruction};
use crate::chips::permutation_any::{PermAnyChip, PermAnyConfig};

const NUM_BYTES: usize = 5;
const MAX_SENTINEL: u64 = (1u64 << (8 * NUM_BYTES)) - 1;

// Padding tuned for ORDER BY (totalprice DESC, orderdate ASC)
const PAD_NAME: u64 = MAX_SENTINEL;
const PAD_CUST: u64 = MAX_SENTINEL;
const PAD_OKEY: u64 = MAX_SENTINEL;
const PAD_DATE: u64 = MAX_SENTINEL; // biggest => last when ASC
const PAD_TOTAL: u64 = 0; // smallest => last when DESC
const PAD_QSUM: u64 = 0;

pub trait Field: PrimeField<Repr = [u8; 32]> {}
impl<F> Field for F where F: PrimeField<Repr = [u8; 32]> {}

#[derive(Clone, Debug)]
pub struct Q18Config<F: Field + Ord> {
    // base tables
    customer: Vec<Column<Advice>>, // [name, custkey]
    orders: Vec<Column<Advice>>,   // [okey, custkey, date, total]
    lineitem: Vec<Column<Advice>>, // [okey, qty]

    // condition (:1)
    cond_thresh: Column<Advice>,

    // permutation: lineitem <-> l_sorted
    l_sorted: Vec<Column<Advice>>,
    perm_lsort: PermAnyConfig,

    // grouping helpers
    q_line: Selector,
    q_first: Selector,
    q_accu: Selector,

    run_sum: Column<Advice>,
    iz_same_prev: IsZeroConfig<F>,
    iz_same_next: IsZeroConfig<F>,

    // having: threshold < group_sum (only on last row)
    lt_thresh_sum: LtConfig<F, NUM_BYTES>,
    heavy: Column<Advice>, // boolean on last rows (else can be 0)

    // result (padded, length = n_lineitem rows)
    res_pad: Vec<Column<Advice>>, // [c_name, c_cust, okey, date, total, sum_qty]
    res_sorted: Vec<Column<Advice>>, // same
    perm_res: PermAnyConfig,

    // lookups to attach attributes (only when emit=1)
    q_lookup_ord: Selector,
    q_lookup_cust: Selector,
    q_tbl_orders: Selector,
    q_tbl_customer: Selector,

    // ORDER BY proof
    q_sort: Selector,
    lt_total_next_cur: LtConfig<F, NUM_BYTES>, // total_next < total_cur
    lt_date_cur_next: LtConfig<F, NUM_BYTES>,  // date_cur < date_next
    iz_total_eq: IsZeroConfig<F>,
    iz_date_eq: IsZeroConfig<F>,

    instance: Column<Instance>,
    instance_test: Column<Advice>,
}

#[derive(Clone, Debug)]
pub struct Q18Chip<F: Field + Ord> {
    pub config: Q18Config<F>,
}

impl<F: Field + Ord> Q18Chip<F> {
    pub fn construct(config: Q18Config<F>) -> Self {
        Self { config }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>) -> Q18Config<F> {
        let instance = meta.instance_column();
        meta.enable_equality(instance);
        let instance_test = meta.advice_column();
        meta.enable_equality(instance_test);

        // base tables
        let customer = vec![meta.advice_column(), meta.advice_column()];
        let orders = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let lineitem = vec![meta.advice_column(), meta.advice_column()];

        // condition
        let cond_thresh = meta.advice_column();

        // selectors
        let q_line = meta.selector();
        let q_first = meta.selector();
        let q_accu = meta.selector();

        let q_lookup_ord = meta.complex_selector();
        let q_lookup_cust = meta.complex_selector();
        let q_tbl_orders = meta.complex_selector();
        let q_tbl_customer = meta.complex_selector();

        let q_sort = meta.selector();

        // l_sorted
        let l_sorted = vec![meta.advice_column(), meta.advice_column()];

        // PermAny: lineitem <-> l_sorted
        let q_perm_l_in = meta.complex_selector();
        let q_perm_l_out = meta.complex_selector();
        let perm_lsort = PermAnyChip::configure(
            meta,
            q_perm_l_in,
            q_perm_l_out,
            lineitem.clone(),
            l_sorted.clone(),
        );

        // grouping columns
        let run_sum = meta.advice_column();

        // same_prev / same_next
        let aux_same_prev = meta.advice_column();
        let aux_same_next = meta.advice_column();

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

        // run_sum gates
        // run_sum[0] = qty[0]
        meta.create_gate("run_sum_first", |m| {
            let q = m.query_selector(q_first);
            let rs = m.query_advice(run_sum, Rotation::cur());
            let qty = m.query_advice(l_sorted[1], Rotation::cur());
            vec![q * (rs - qty)]
        });
        // run_sum[i] = same_prev * run_sum[i-1] + qty[i]
        meta.create_gate("run_sum_accu", |m| {
            let q = m.query_selector(q_accu);
            let same = iz_same_prev.expr();
            let rs_cur = m.query_advice(run_sum, Rotation::cur());
            let rs_prev = m.query_advice(run_sum, Rotation::prev());
            let qty = m.query_advice(l_sorted[1], Rotation::cur());
            vec![q * (rs_cur - (same * rs_prev + qty))]
        });

        // HAVING: heavy = (thresh < run_sum) only on last rows
        let heavy = meta.advice_column();
        meta.enable_equality(heavy);

        let lt_thresh_sum = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| {
                let q = m.query_selector(q_line);
                let one = Expression::Constant(F::ONE);
                let is_last = one - iz_same_next.expr(); // only last row enables
                q * is_last
            },
            |m| m.query_advice(cond_thresh, Rotation::cur()),
            |m| m.query_advice(run_sum, Rotation::cur()),
        );

        // constrain heavy == lt_thresh_sum on last rows, and heavy boolean
        meta.create_gate("heavy flag on last rows", |m| {
            let q = m.query_selector(q_line);
            let one = Expression::Constant(F::ONE);
            let is_last = one.clone() - iz_same_next.expr();

            let h = m.query_advice(heavy, Rotation::cur());
            let lt = lt_thresh_sum.is_lt(m, None);

            vec![
                q.clone() * is_last.clone() * (h.clone() - lt),
                q * is_last * h.clone() * (one - h),
            ]
        });

        // result columns
        let res_pad = (0..6).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let res_sorted = (0..6).map(|_| meta.advice_column()).collect::<Vec<_>>();

        // permutation res_pad <-> res_sorted
        let q_perm_r_in = meta.complex_selector();
        let q_perm_r_out = meta.complex_selector();
        let perm_res = PermAnyChip::configure(
            meta,
            q_perm_r_in,
            q_perm_r_out,
            res_pad.clone(),
            res_sorted.clone(),
        );

        // emit gate: only if emit = is_last * heavy
        meta.create_gate("emit res_pad rows (pad otherwise)", |m| {
            let q = m.query_selector(q_line);
            let one = Expression::Constant(F::ONE);
            let is_last = one.clone() - iz_same_next.expr();
            let h = m.query_advice(heavy, Rotation::cur());
            let emit = is_last * h;
            let not_emit = one.clone() - emit.clone();

            let cur_okey = m.query_advice(l_sorted[0], Rotation::cur());
            let sum_qty = m.query_advice(run_sum, Rotation::cur());

            // res_pad indices:
            // 0=c_name,1=c_cust,2=o_okey,3=o_date,4=o_total,5=sum_qty
            let out_name = m.query_advice(res_pad[0], Rotation::cur());
            let out_cust = m.query_advice(res_pad[1], Rotation::cur());
            let out_okey = m.query_advice(res_pad[2], Rotation::cur());
            let out_date = m.query_advice(res_pad[3], Rotation::cur());
            let out_total = m.query_advice(res_pad[4], Rotation::cur());
            let out_sum = m.query_advice(res_pad[5], Rotation::cur());

            let pad_name = Expression::Constant(F::from(PAD_NAME));
            let pad_cust = Expression::Constant(F::from(PAD_CUST));
            let pad_okey = Expression::Constant(F::from(PAD_OKEY));
            let pad_date = Expression::Constant(F::from(PAD_DATE));
            let pad_total = Expression::Constant(F::from(PAD_TOTAL));
            let pad_qsum = Expression::Constant(F::from(PAD_QSUM));

            vec![
                // we always set okey & sum based on emit; other cols forced to PAD when not emit
                q.clone() * (out_okey - (emit.clone() * cur_okey + not_emit.clone() * pad_okey)),
                q.clone() * (out_sum - (emit.clone() * sum_qty + not_emit.clone() * pad_qsum)),
                q.clone() * not_emit.clone() * (out_name - pad_name),
                q.clone() * not_emit.clone() * (out_cust - pad_cust),
                q.clone() * not_emit.clone() * (out_date - pad_date),
                q * not_emit * (out_total - pad_total),
            ]
        });

        // ---------- Tuple lookup into orders (only when emit=1) ----------
        // (o_okey, o_cust, o_date, o_total) ∈ orders
        meta.lookup_any("attach orders tuple", |m| {
            let q_in = m.query_selector(q_lookup_ord);
            let q_tbl = m.query_selector(q_tbl_orders);

            let one = Expression::Constant(F::ONE);
            let is_last = one.clone() - iz_same_next.expr();
            let h = m.query_advice(heavy, Rotation::cur());
            let emit = is_last * h;
            let gate = q_in * emit;

            vec![
                (
                    gate.clone() * m.query_advice(res_pad[2], Rotation::cur()),
                    q_tbl.clone() * m.query_advice(orders[0], Rotation::cur()),
                ), // okey
                (
                    gate.clone() * m.query_advice(res_pad[1], Rotation::cur()),
                    q_tbl.clone() * m.query_advice(orders[1], Rotation::cur()),
                ), // cust
                (
                    gate.clone() * m.query_advice(res_pad[3], Rotation::cur()),
                    q_tbl.clone() * m.query_advice(orders[2], Rotation::cur()),
                ), // date
                (
                    gate * m.query_advice(res_pad[4], Rotation::cur()),
                    q_tbl * m.query_advice(orders[3], Rotation::cur()),
                ), // total
            ]
        });

        // ---------- Tuple lookup into customer (only when emit=1) ----------
        // (c_custkey, c_name) ∈ customer
        meta.lookup_any("attach customer tuple", |m| {
            let q_in = m.query_selector(q_lookup_cust);
            let q_tbl = m.query_selector(q_tbl_customer);

            let one = Expression::Constant(F::ONE);
            let is_last = one.clone() - iz_same_next.expr();
            let h = m.query_advice(heavy, Rotation::cur());
            let emit = is_last * h;
            let gate = q_in * emit;

            vec![
                (
                    gate.clone() * m.query_advice(res_pad[1], Rotation::cur()),
                    q_tbl.clone() * m.query_advice(customer[1], Rotation::cur()),
                ), // custkey
                (
                    gate * m.query_advice(res_pad[0], Rotation::cur()),
                    q_tbl * m.query_advice(customer[0], Rotation::cur()),
                ), // name
            ]
        });

        // ---------- ORDER BY (o_totalprice DESC, o_orderdate ASC) on res_sorted ----------
        let aux_total_eq = meta.advice_column();
        let aux_date_eq = meta.advice_column();

        let iz_total_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort),
            |m| {
                m.query_advice(res_sorted[4], Rotation::cur())
                    - m.query_advice(res_sorted[4], Rotation::next())
            },
            aux_total_eq,
        );
        let iz_date_eq = IsZeroChip::configure(
            meta,
            |m| m.query_selector(q_sort),
            |m| {
                m.query_advice(res_sorted[3], Rotation::cur())
                    - m.query_advice(res_sorted[3], Rotation::next())
            },
            aux_date_eq,
        );

        let lt_total_next_cur = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort),
            |m| m.query_advice(res_sorted[4], Rotation::next()), // total_next
            |m| m.query_advice(res_sorted[4], Rotation::cur()),  // total_cur
        );
        let lt_date_cur_next = LtChip::<F, NUM_BYTES>::configure(
            meta,
            |m| m.query_selector(q_sort),
            |m| m.query_advice(res_sorted[3], Rotation::cur()), // date_cur
            |m| m.query_advice(res_sorted[3], Rotation::next()), // date_next
        );

        meta.create_gate("ORDER BY total DESC, date ASC", |m| {
            let q = m.query_selector(q_sort);

            let total_gt = lt_total_next_cur.is_lt(m, None); // next < cur
            let total_eq = iz_total_eq.expr();

            let date_lt = lt_date_cur_next.is_lt(m, None); // cur < next
            let date_eq = iz_date_eq.expr();
            let date_le = date_lt + date_eq;

            vec![q * (total_gt + total_eq * date_le - Expression::Constant(F::ONE))]
        });

        Q18Config {
            customer,
            orders,
            lineitem,
            cond_thresh,
            l_sorted,
            perm_lsort,
            q_line,
            q_first,
            q_accu,
            run_sum,
            iz_same_prev,
            iz_same_next,
            lt_thresh_sum,
            heavy,
            res_pad,
            res_sorted,
            perm_res,
            q_lookup_ord,
            q_lookup_cust,
            q_tbl_orders,
            q_tbl_customer,
            q_sort,
            lt_total_next_cur,
            lt_date_cur_next,
            iz_total_eq,
            iz_date_eq,
            instance,
            instance_test,
        }
    }

    pub fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        customer_u64: Vec<Vec<u64>>, // [name, custkey]
        orders_u64: Vec<Vec<u64>>,   // [okey, custkey, date, total]
        lineitem_u64: Vec<Vec<u64>>, // [okey, qty]
        threshold: u64,
    ) -> Result<AssignedCell<F, F>, Error> {
        // chips
        let iz_same_prev_chip = IsZeroChip::construct(self.config.iz_same_prev.clone());
        let iz_same_next_chip = IsZeroChip::construct(self.config.iz_same_next.clone());

        let lt_thresh_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_thresh_sum.clone());
        lt_thresh_chip.load(layouter)?;

        let iz_total_eq_chip = IsZeroChip::construct(self.config.iz_total_eq.clone());
        let iz_date_eq_chip = IsZeroChip::construct(self.config.iz_date_eq.clone());

        let lt_total_chip =
            LtChip::<F, NUM_BYTES>::construct(self.config.lt_total_next_cur.clone());
        lt_total_chip.load(layouter)?;
        let lt_date_chip = LtChip::<F, NUM_BYTES>::construct(self.config.lt_date_cur_next.clone());
        lt_date_chip.load(layouter)?;

        // prepare sorted lineitems
        let mut l_sorted = lineitem_u64.clone();
        l_sorted.sort_by_key(|r| r[0]); // by orderkey
        let n = l_sorted.len();

        // compute run_sum and heavy only on last row of group
        let mut run_sum_u64 = vec![0u64; n];
        let mut heavy_u64 = vec![0u64; n];

        let mut acc: u128 = 0;
        let mut prev_ok: Option<u64> = None;

        for i in 0..n {
            let ok = l_sorted[i][0];
            let qty = l_sorted[i][1] as u128;

            if prev_ok == Some(ok) {
                acc += qty;
            } else {
                acc = qty;
            }
            run_sum_u64[i] = acc as u64;

            let next_ok = if i + 1 < n { l_sorted[i + 1][0] } else { 0 };
            let is_last = next_ok != ok;

            if is_last && run_sum_u64[i] > threshold {
                heavy_u64[i] = 1;
            }
            prev_ok = Some(ok);
        }

        // maps for attachment
        use std::collections::HashMap;
        let mut orders_map: HashMap<u64, (u64, u64, u64)> = HashMap::new(); // okey -> (cust,date,total)
        for r in &orders_u64 {
            orders_map.insert(r[0], (r[1], r[2], r[3]));
        }
        let mut cust_map: HashMap<u64, u64> = HashMap::new(); // custkey -> name
        for r in &customer_u64 {
            cust_map.insert(r[1], r[0]);
        }

        // build res_pad (length = n)
        let mut res_pad = vec![[PAD_NAME, PAD_CUST, PAD_OKEY, PAD_DATE, PAD_TOTAL, PAD_QSUM]; n];
        for i in 0..n {
            // only last rows can emit
            let next_ok = if i + 1 < n { l_sorted[i + 1][0] } else { 0 };
            let is_last = next_ok != l_sorted[i][0];

            if is_last && heavy_u64[i] == 1 {
                let okey = l_sorted[i][0];
                let sumq = run_sum_u64[i];
                let (cust, date, total) = orders_map.get(&okey).copied().unwrap_or((0, 0, 0));
                let name = cust_map.get(&cust).copied().unwrap_or(0);
                res_pad[i] = [name, cust, okey, date, total, sumq];
            }
        }

        // build res_sorted (sort emitted rows by total desc, date asc)
        let mut emitted: Vec<[u64; 6]> = res_pad
            .iter()
            .copied()
            .filter(|r| r[2] != PAD_OKEY)
            .collect();
        emitted.sort_by(|a, b| b[4].cmp(&a[4]).then(a[3].cmp(&b[3])));

        let mut res_sorted = Vec::with_capacity(n);
        res_sorted.extend(emitted.into_iter());
        while res_sorted.len() < n {
            res_sorted.push([PAD_NAME, PAD_CUST, PAD_OKEY, PAD_DATE, PAD_TOTAL, PAD_QSUM]);
        }

        // assign region
        layouter.assign_region(
            || "Q18 witness",
            |mut region| {
                // base tables
                for i in 0..customer_u64.len() {
                    for j in 0..2 {
                        region.assign_advice(
                            || "customer",
                            self.config.customer[j],
                            i,
                            || Value::known(F::from(customer_u64[i][j])),
                        )?;
                    }
                    self.config.q_tbl_customer.enable(&mut region, i)?;
                }
                for i in 0..orders_u64.len() {
                    for j in 0..4 {
                        region.assign_advice(
                            || "orders",
                            self.config.orders[j],
                            i,
                            || Value::known(F::from(orders_u64[i][j])),
                        )?;
                    }
                    self.config.q_tbl_orders.enable(&mut region, i)?;
                }
                for i in 0..lineitem_u64.len() {
                    for j in 0..2 {
                        region.assign_advice(
                            || "lineitem",
                            self.config.lineitem[j],
                            i,
                            || Value::known(F::from(lineitem_u64[i][j])),
                        )?;
                    }
                }

                // condition column (threshold) over n rows (like your Q3 condition)
                for i in 0..n {
                    region.assign_advice(
                        || "threshold",
                        self.config.cond_thresh,
                        i,
                        || Value::known(F::from(threshold)),
                    )?;
                }

                // l_sorted + sentinel row for same_next
                for i in 0..n {
                    for j in 0..2 {
                        region.assign_advice(
                            || "l_sorted",
                            self.config.l_sorted[j],
                            i,
                            || Value::known(F::from(l_sorted[i][j])),
                        )?;
                    }
                }
                // sentinel at row n (okey=0, qty=0) for same_next on last
                for j in 0..2 {
                    region.assign_advice(
                        || "l_sorted_sentinel",
                        self.config.l_sorted[j],
                        n,
                        || Value::known(F::from(0u64)),
                    )?;
                }

                // enable perm lineitem<->l_sorted
                for i in 0..n {
                    self.config.perm_lsort.q_perm1.enable(&mut region, i)?;
                    self.config.perm_lsort.q_perm2.enable(&mut region, i)?;
                }

                // assign run_sum and heavy
                for i in 0..n {
                    // selectors
                    self.config.q_line.enable(&mut region, i)?;
                    if i == 0 {
                        self.config.q_first.enable(&mut region, i)?;
                    }
                    if i >= 1 {
                        self.config.q_accu.enable(&mut region, i)?;
                    }

                    region.assign_advice(
                        || "run_sum",
                        self.config.run_sum,
                        i,
                        || Value::known(F::from(run_sum_u64[i])),
                    )?;
                    region.assign_advice(
                        || "heavy",
                        self.config.heavy,
                        i,
                        || Value::known(F::from(heavy_u64[i])),
                    )?;
                }

                // isZero assignments
                for i in 1..n {
                    let diff = F::from(l_sorted[i][0]) - F::from(l_sorted[i - 1][0]);
                    iz_same_prev_chip.assign(&mut region, i, Value::known(diff))?;
                }
                for i in 0..n {
                    let next_ok = if i + 1 < n { l_sorted[i + 1][0] } else { 0u64 };
                    let diff = F::from(next_ok) - F::from(l_sorted[i][0]);
                    iz_same_next_chip.assign(&mut region, i, Value::known(diff))?;
                }

                // Lt threshold < run_sum (only constrained on last rows due to q_enable in configure)
                for i in 0..n {
                    lt_thresh_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(threshold)),
                        Value::known(F::from(run_sum_u64[i])),
                    )?;
                }

                // assign res_pad and res_sorted
                for i in 0..n {
                    for j in 0..6 {
                        region.assign_advice(
                            || "res_pad",
                            self.config.res_pad[j],
                            i,
                            || Value::known(F::from(res_pad[i][j])),
                        )?;
                        region.assign_advice(
                            || "res_sorted",
                            self.config.res_sorted[j],
                            i,
                            || Value::known(F::from(res_sorted[i][j])),
                        )?;
                    }
                }

                // enable lookup selectors (safe to enable for all rows; gate uses emit inside lookup)
                for i in 0..n {
                    self.config.q_lookup_ord.enable(&mut region, i)?;
                    self.config.q_lookup_cust.enable(&mut region, i)?;
                }

                // perm res_pad<->res_sorted
                for i in 0..n {
                    self.config.perm_res.q_perm1.enable(&mut region, i)?;
                    self.config.perm_res.q_perm2.enable(&mut region, i)?;
                }

                // ORDER BY helper chips (rows 0..n-2)
                for i in 0..n.saturating_sub(1) {
                    self.config.q_sort.enable(&mut region, i)?;

                    let total_cur = res_sorted[i][4];
                    let total_next = res_sorted[i + 1][4];
                    let date_cur = res_sorted[i][3];
                    let date_next = res_sorted[i + 1][3];

                    iz_total_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(total_cur) - F::from(total_next)),
                    )?;
                    iz_date_eq_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(date_cur) - F::from(date_next)),
                    )?;

                    lt_total_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(total_next)),
                        Value::known(F::from(total_cur)),
                    )?;
                    lt_date_chip.assign(
                        &mut region,
                        i,
                        Value::known(F::from(date_cur)),
                        Value::known(F::from(date_next)),
                    )?;
                }

                // public output (same style as your Q3)
                let out = region.assign_advice(
                    || "instance_test",
                    self.config.instance_test,
                    0,
                    || Value::known(F::from(1u64)),
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

#[derive(Clone, Debug)]
pub struct MyCircuit<F: Field + Ord> {
    pub customer: Vec<Vec<u64>>,
    pub orders: Vec<Vec<u64>>,
    pub lineitem: Vec<Vec<u64>>,
    pub threshold: u64, // <-- :1 in TPCH Q18
    pub _marker: PhantomData<F>,
}

impl<F: Field + Ord> Default for MyCircuit<F> {
    fn default() -> Self {
        Self {
            customer: vec![],
            orders: vec![],
            lineitem: vec![],
            threshold: 0,
            _marker: PhantomData,
        }
    }
}

impl<F: Field + Ord> Circuit<F> for MyCircuit<F> {
    type Config = Q18Config<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        Q18Chip::<F>::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = Q18Chip::<F>::construct(config);

        let out_cell = chip.assign(
            &mut layouter,
            self.customer.clone(),
            self.orders.clone(),
            self.lineitem.clone(),
            self.threshold, // <-- pass :1
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
    use halo2_proofs::dev::MockProver;
    use halo2curves::pasta::{vesta, EqAffine, Fp};

    use halo2_proofs::{
        plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit},
        poly::{
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

    use halo2_proofs::poly::commitment::Params;
    use rand::rngs::OsRng;
    use std::marker::PhantomData;
    use std::time::Instant;
    use std::{fs::File, io::Write, path::Path};

    fn generate_and_verify_proof<C: Circuit<Fp>>(
        _k: u32,
        circuit: C,
        public_input: &[Fp],
        proof_path: &str,
    ) {
        let params_path = "/home2/binbin/PoneglyphDB/src/proof/param16";
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

    #[test]
    fn test_1() {
        let k = 16;

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
                    let datetime: DateTime<Utc> =
                        DateTime::<Utc>::from_utc(date.and_hms(0, 0, 0), Utc);
                    datetime.timestamp() as u64
                }
                Err(_) => 0,
            }
        }

        let customer_file_path = "/home2/binbin/PoneglyphDB/src/data/customer.tbl";
        let orders_file_path = "/home2/binbin/PoneglyphDB/src/data/orders.tbl";
        let lineitem_file_path = "/home2/binbin/PoneglyphDB/src/data/lineitem.tbl";

        let mut customer: Vec<Vec<u64>> = Vec::new();
        let mut orders: Vec<Vec<u64>> = Vec::new();
        let mut lineitem: Vec<Vec<u64>> = Vec::new();

        if let Ok(records) = data_processing::customer_read_records_from_file(customer_file_path) {
            // customer = [c_name_u64, c_custkey]
            customer = records
                .iter()
                .map(|record| vec![string_to_u64(&record.c_name), record.c_custkey])
                .collect();
        }

        if let Ok(records) = data_processing::orders_read_records_from_file(orders_file_path) {
            // IMPORTANT: match Q18Chip expectation:
            // orders = [o_orderkey, o_custkey, o_orderdate_ts, o_totalprice_scaled]
            orders = records
                .iter()
                .map(|record| {
                    vec![
                        record.o_orderkey,
                        record.o_custkey,
                        date_to_timestamp(&record.o_orderdate),
                        scale_by_1000(record.o_totalprice),
                    ]
                })
                .collect();
        }

        if let Ok(records) = data_processing::lineitem_read_records_from_file(lineitem_file_path) {
            // lineitem = [l_orderkey, l_quantity]
            lineitem = records
                .iter()
                .map(|record| vec![record.l_orderkey, record.l_quantity])
                .collect();
        }

        // TPCH Q18 typical parameter is 300
        let threshold: u64 = 300;

        let circuit = MyCircuit::<Fp> {
            customer,
            orders,
            lineitem,
            threshold,
            _marker: PhantomData,
        };

        let public_input = vec![Fp::from(1)];

        // let test = true;
        let test = false;

        if test {
            let prover = MockProver::run(k, &circuit, vec![public_input]).unwrap();
            prover.assert_satisfied();
        } else {
            let proof_path = "/home2/binbin/PoneglyphDB/src/proof/proof_obj_q18";
            generate_and_verify_proof(k, circuit, &public_input, proof_path);
        }
    }
}
