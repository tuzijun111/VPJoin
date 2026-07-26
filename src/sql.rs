pub mod q3_obj;
pub mod q3_bound; // bound version: q3_obj + inlined witness-binding check
pub mod q3_obj_key; // key-edge specialization of condition (10): sigma is a bit, so no aggregation and no clean channel

pub mod q5_obj;
pub mod q5_bound;
pub mod q5_obj_key; // key-edge specialization of condition (10): sigma is a bit, so no aggregation and no clean channel
pub mod q5_obj_dp; // multi-lane DP-padding variant of q5_obj (fixed k = 16)

pub mod q8_obj;
pub mod q8_bound;
pub mod q8_obj_key; // key-edge specialization of condition (10): sigma is a bit, so no aggregation and no clean channel

pub mod q9_obj;
pub mod q9_bound;
pub mod q9_obj_key; // key-edge specialization of condition (10): sigma is a bit, so no aggregation and no clean channel

pub mod q18_obj;
pub mod q18_bound;
pub mod q18_obj_key; // key-edge specialization of condition (10): sigma is a bit, so no aggregation and no clean channel

pub mod test;
pub mod test1;
