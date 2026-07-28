pub mod q3_bound;
pub mod q3_obj; // bound version: q3_obj + inlined witness-binding check
pub mod q3_obj_new; // revised One-Pass OBJ: selector bits on the committed rows

pub mod q5_bound;
pub mod q5_obj;
pub mod q5_obj_new; // revised One-Pass OBJ over the committed rows; capacities kept as a padding layer
pub mod q5_obj_dp; // multi-lane DP-padding variant of q5_obj (fixed k = 17)
pub mod q5_obj_dp_new; // the same lanes under the revised One-Pass OBJ

pub mod q8_bound;
pub mod q8_obj;
pub mod q8_obj_new; // revised One-Pass OBJ: selector bits on the committed rows

pub mod q9_bound;
pub mod q9_obj;
pub mod q9_obj_new; // revised One-Pass OBJ: selector bits on the committed rows

pub mod q18_bound;
pub mod q18_obj;
pub mod q18_obj_new; // revised One-Pass OBJ: selector bits on the committed rows

pub mod test;
pub mod test1;
