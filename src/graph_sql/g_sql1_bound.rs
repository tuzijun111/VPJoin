

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp;

use super::g_sql1_obj;
use crate::data::graph_data_processing::Edge;
use crate::inline_bind::{assign_bind_cells, configure_bind, tie_columns, BindConfig};

/// Committed input columns: Edge(src, dst).
pub const NC: usize = 2;

pub struct BoundGq1 {
    // -- identical to g_sql1_obj::Path3OrdCircuit --
    pub edges: Vec<Edge>,
    // -- binding additions --
    pub columns: Vec<Vec<u64>>,
    pub x: Fp,
}

impl Circuit<Fp> for BoundGq1 {
    type Config = (g_sql1_obj::Path3OrdConfig<Fp>, BindConfig);
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            edges: Vec::new(),
            columns: Vec::new(),
            x: self.x,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        // MUST be the FULL baseline configure (chip + its extra lookups), not
        // Path3OrdChip::configure alone -- see module docs.
        let q = g_sql1_obj::configure_path3ord_full::<Fp>(meta);
        let b = configure_bind(meta, NC);
        (q, b)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let chip = g_sql1_obj::Path3OrdChip::construct(config.0);
        let (out, edge_cells) = chip.assign_with_edge_cells(&mut layouter, &self.edges)?;
        chip.expose_public(&mut layouter, out, 0)?;
        // The binding evaluates its own columns at x, then those columns are
        // copy-constrained cell-by-cell to the Edge cells THIS proof witnesses.
        // Without the tie the binding would establish only that some columns
        // match Commit(D); with it, the exposed evaluation is an evaluation of
        // this proof's witness, which is what Appendix A requires.
        let bind_cells = assign_bind_cells(&mut layouter, &config.1, &self.columns, self.x)?;
        tie_columns(&mut layouter, &bind_cells, &edge_cells)?;
        Ok(())
    }
}
