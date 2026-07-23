pub mod bench_queries; // additive single-command benchmark harness (revision)
pub mod chips;
pub mod circuits;
pub mod column_commit; // additive per-column commitments in the circuits' own domain (Appendix A)
pub mod commitment; // legacy monolithic single-vector layer (Appendix A, first version)
pub mod dp_lane; // plan/measurement records shared by the three DP lane circuits and dp_lane_bench
pub mod dp_noise; // additive DP capacity generation (Appendix G.2); Rust twin of dp/noise_generator.py
pub mod input_binding; // additive in-circuit input binding to the published commitments (Appendix A)
// pub mod gadgets;
pub mod data;
pub mod graph_data;
pub mod graph_sql;
pub mod inline_bind; // witness-equality check inlined into the query circuit (Appendix A)
pub mod is_zero;
pub mod paths; // repo-relative path resolution (no absolute paths in source)
pub mod sql;
