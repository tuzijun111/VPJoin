use std::fs::File;
use std::io::{self, BufRead, BufReader};

#[derive(Clone, Debug)]
pub struct Edge {
    pub src: u64,
    pub dst: u64,
}

pub fn read_edges_tsv(path: &str) -> io::Result<Vec<Edge>> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);

    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let src: u64 = it.next().unwrap().parse().unwrap();
        let dst: u64 = it.next().unwrap().parse().unwrap();
        out.push(Edge { src, dst });
    }
    Ok(out)
}
