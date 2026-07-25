pub mod g_sql1_obj;
pub mod g_sql1_bound; // bound version: g_sql1_obj + inlined witness-binding check
pub mod pone_baseline; // additive PoneglyphDB-style binary-join-chain baseline for GQ1-GQ4
pub mod g_sql1_obj_old;
pub mod g_sql1_obj_test; // One-Pass OBJ with the Cardinality Preservation Check for condition (4)
pub mod g_sql2_obj;
pub mod g_sql2_bound;
pub mod g_sql2_obj_old;
pub mod g_sql2_obj_test; // One-Pass OBJ with the Cardinality Preservation Check for condition (4)
pub mod g_sql3_obj;
pub mod g_sql3_bound;
pub mod g_sql3_obj_dp; // multi-lane DP-padding variant of g_sql3_obj (k fixed at the RJS degree)
pub mod g_sql3_obj_old;
pub mod g_sql3_obj_test; // One-Pass OBJ with the Cardinality Preservation Check for condition (4)
pub mod g_sql4_obj;
pub mod g_sql4_bound;
pub mod g_sql4_obj_dp; // multi-lane DP-padding variant of g_sql4_obj (k fixed at the RJS degree)
pub mod g_sql4_obj_old;
pub mod g_sql4_obj_test; // One-Pass OBJ with the Cardinality Preservation Check for condition (4)
pub mod test;
