use halo2_proofs::arithmetic::{best_fft, Field};
use halo2_proofs::halo2curves::bn256::Fr;
use std::time::Instant;
use halo2_proofs::poly::EvaluationDomain;
use rand_core::OsRng;
use rayon::prelude::*;
#[test]
fn test_fft_cpu_gpu() {
    let now = Instant::now();
    let k = 24;
    let domain = EvaluationDomain::<Fr>::new(3, k);
    let rng = OsRng;
    let mut coeff = (0..(1 << k)).into_par_iter().map(|_| Fr::random(OsRng)).collect::<Vec<_>>();
    
    println!("Prepare for fft takes {:?}", now.elapsed());
    let now: Instant = Instant::now();
    best_fft(coeff.as_mut_slice(), domain.get_omega(), k);
    println!("fft cpu takes {:?}", now.elapsed());
}

