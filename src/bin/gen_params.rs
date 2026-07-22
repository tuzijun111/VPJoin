//! Generate and persist Halo2 IPA public parameters of a given degree.
//!
//! The repo ships param15..param19; the largest graph circuits need more.
//! Generating them once up front keeps them out of the measured runs.
//!
//! Usage:
//!   cargo run --release --bin gen_params -- 22 [23 ...]
//!
//! Writes `src/proof/param{k}` (honours `VPJOIN_DATA`).  Existing files of the
//! correct degree are left alone.

use halo2_proofs::poly::{
    commitment::{Params, ParamsProver},
    ipa::commitment::ParamsIPA,
};
use halo2curves::pasta::vesta;
use std::time::Instant;

fn main() {
    let ks: Vec<u32> = std::env::args()
        .skip(1)
        .map(|a| {
            a.parse()
                .unwrap_or_else(|_| panic!("not a degree: {}", a))
        })
        .collect();
    if ks.is_empty() {
        eprintln!("usage: gen_params <k> [k ...]   e.g. gen_params 22");
        std::process::exit(2);
    }

    for k in ks {
        let path = halo2_experiments::paths::param_file(k);
        if std::path::Path::new(&path).exists() {
            // Verify it really is the claimed degree before skipping.
            let mut fd = std::fs::File::open(&path).expect("open existing params");
            match ParamsIPA::<vesta::Affine>::read(&mut fd) {
                Ok(p) if p.k() == k => {
                    println!("param{} already present and valid: {}", k, path);
                    continue;
                }
                Ok(p) => panic!(
                    "{} exists but has degree 2^{}, expected 2^{}",
                    path,
                    p.k(),
                    k
                ),
                Err(e) => panic!("{} exists but is unreadable: {}", path, e),
            }
        }

        println!("generating 2^{} params ...", k);
        let t = Instant::now();
        let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(k);
        let gen_s = t.elapsed().as_secs_f64();

        if let Some(dir) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(dir).expect("create proof dir");
        }
        let t = Instant::now();
        let mut fd = std::fs::File::create(&path).expect("create params file");
        params.write(&mut fd).expect("write params");
        drop(fd);
        let write_s = t.elapsed().as_secs_f64();

        let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        println!(
            "wrote {} ({:.1} MiB) -- generate {:.1}s, write {:.1}s",
            path,
            bytes as f64 / (1024.0 * 1024.0),
            gen_s,
            write_s
        );
    }
}
