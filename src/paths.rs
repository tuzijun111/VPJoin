use std::path::PathBuf;

/// Root holding `data/`, `graph_data/` and `proof/` (the crate's own `src/`
/// unless `VPJOIN_DATA` overrides it).
pub fn src_root() -> PathBuf {
    if let Ok(p) = std::env::var("VPJOIN_DATA") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// A path relative to [`src_root`], e.g. `in_src("data/customer.tbl")`.
pub fn in_src(rel: &str) -> String {
    src_root().join(rel).to_string_lossy().into_owned()
}

/// A TPC-H table, e.g. `data_file("lineitem.tbl")`.
///
/// `VPJOIN_TABLES` overrides just the directory the `.tbl`/`.cvs` tables load
/// from, leaving the proof params and graph datasets where they are (under
/// [`src_root`]).  This is what a scaling sweep uses: it swaps the tables per
/// scale while `proof/param{k}` keeps resolving to the crate.  `VPJOIN_DATA`
/// still overrides the whole root (data + proof + graph) for the case where an
/// entire self-contained tree has been relocated.
pub fn data_file(name: &str) -> String {
    if let Ok(dir) = std::env::var("VPJOIN_TABLES") {
        return PathBuf::from(dir)
            .join(name)
            .to_string_lossy()
            .into_owned();
    }
    in_src(&format!("data/{}", name))
}

/// The directory holding the graph datasets.
pub fn graph_dir() -> String {
    in_src("graph_data")
}

/// A graph dataset, e.g. `graph_file("wiki/wiki_Vote.txt")`.
pub fn graph_file(rel: &str) -> String {
    in_src(&format!("graph_data/{}", rel))
}

/// The persisted Halo2 public parameters of degree `2^k`.
pub fn param_file(k: u32) -> String {
    in_src(&format!("proof/param{}", k))
}

/// A proof artifact under `proof/`, e.g. `proof_file("proof_obj_q3")`.
pub fn proof_file(name: &str) -> String {
    in_src(&format!("proof/{}", name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_relative_to_the_crate_and_well_formed() {
        assert!(src_root().is_absolute(), "resolved root must be absolute");
        assert!(data_file("lineitem.tbl").ends_with("src/data/lineitem.tbl"));
        assert!(param_file(16).ends_with("src/proof/param16"));
        assert!(graph_file("wiki/wiki_Vote.txt").ends_with("src/graph_data/wiki/wiki_Vote.txt"));
        assert!(proof_file("proof_obj_q3").ends_with("src/proof/proof_obj_q3"));
    }

    /// The files the query tests actually open must exist in a fresh checkout.
    #[test]
    fn the_data_the_tests_need_is_present() {
        for f in [
            "customer.tbl",
            "orders.tbl",
            "lineitem.tbl",
            "supplier.tbl",
            "nation.tbl",
            "part.tbl",
            "partsupp.tbl",
            "region.cvs",
        ] {
            assert!(
                std::path::Path::new(&data_file(f)).exists(),
                "missing TPC-H table {}",
                f
            );
        }
        for f in [
            "wiki/wiki_Vote.txt",
            "facebook/facebook_combined.txt",
            "last/lastfm_asia_edges.csv",
        ] {
            assert!(
                std::path::Path::new(&graph_file(f)).exists(),
                "missing graph dataset {}",
                f
            );
        }
        for k in [16u32, 17, 18] {
            assert!(
                std::path::Path::new(&param_file(k)).exists(),
                "missing params param{}",
                k
            );
        }
    }
}
