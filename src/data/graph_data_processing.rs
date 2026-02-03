use std::fs::File;
use std::io::{self, BufRead, BufReader};

#[derive(Clone, Debug)]
pub struct Edge {
    pub src: u64,
    pub dst: u64,
}

pub fn read_edges(path: &str) -> io::Result<Vec<Edge>> {
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

pub fn read_edges_csv(path: &str) -> io::Result<Vec<Edge>> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);

    let mut out = Vec::new();

    // We use .enumerate() to track the line number so we can skip the header
    for (index, line) in reader.lines().enumerate() {
        let line = line?;

        // Skip the header (line 0)
        if index == 0 {
            continue;
        }

        if line.trim().is_empty() {
            continue;
        }

        // CHANGE: Split by comma ',' instead of whitespace
        let mut it = line.split(',');

        // We use unwrap() here assuming the file format is always perfect.
        // In a real app, you might want better error handling.
        let src_str = it.next().unwrap().trim();
        let dst_str = it.next().unwrap().trim();

        let src: u64 = src_str.parse().expect("Failed to parse source node");
        let dst: u64 = dst_str.parse().expect("Failed to parse dest node");

        out.push(Edge { src, dst });
    }
    Ok(out)
}
