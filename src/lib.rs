pub mod chips;
pub mod circuits;
pub mod commitment; // additive database-commitment layer (Appendix A); no circuit changes
pub mod dp_noise; // additive DP capacity generation (Appendix G.2); Rust twin of dp/noise_generator.py
pub mod input_binding; // additive in-circuit input binding to the published commitments (Appendix A)
// pub mod gadgets;
pub mod data;
pub mod graph_data;
pub mod graph_sql;
pub mod is_zero;
pub mod sql;
