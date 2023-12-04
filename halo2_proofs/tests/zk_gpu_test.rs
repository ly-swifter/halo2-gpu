use halo2_proofs::arithmetic::{best_fft, Field, generate_twiddle_lookup_table, Group};
use halo2_proofs::halo2curves::bn256::Fr;
use halo2curves::FieldExt;
use std::time::Instant;
use halo2_proofs::poly::{Polynomial, EvaluationDomain, kzg::commitment::{ParamsKZG, commit_lagrange_gpu}, 
    Coeff, commitment::{ParamsProver, Blind, Params}};
use rand_core::OsRng;
use rayon::prelude::*;
use zk_gpu::api::{parallel_fft_api, parallel_coeff_to_extended_part_api};
use zk_gpu::threadpool::{Waiter, Worker};
use zk_gpu::gpulock::{LOCKER_1GPU, FFT_MEM, GpuInstance};
pub use halo2curves::{ CurveAffine, bn256::Bn256};
use std::sync::{Arc, Mutex};
use std::marker::PhantomData;

#[test]
fn test_fft_cpu_gpu() {
    use env_logger::Builder;
    use log::LevelFilter;

    Builder::new().filter(None, LevelFilter::Debug).parse_default_env().init();

    let now: Instant = Instant::now();
    let k: u32 = 24;
    let domain = EvaluationDomain::<Fr>::new(3, k);
    let mut coeff_cpu = (0..(1 << k)).into_par_iter().map(|_| Fr::random(OsRng)).collect::<Vec<_>>();
    let mut raw_gpu = coeff_cpu.clone();
    log::info!("firstline of original data {:?}", raw_gpu.as_mut_slice().as_mut_ptr() as *mut u64);
    let mut coeff_gpu: Mutex<Vec<Fr>> = Mutex::new(coeff_cpu.clone());
    unsafe {
        log::info!("firstline of casted data {:?}", std::mem::transmute::<&Vec<Fr>, *mut u64>(&coeff_gpu.get_mut().unwrap()));
    }

    let mut coeff_fil: Vec<Fr> = coeff_cpu.clone();

    println!("{:?}", coeff_cpu[0]);
    println!("Prepare for fft takes {:?}", now.elapsed());
    let now: Instant = Instant::now();
    best_fft(coeff_cpu.as_mut_slice(), domain.get_omega(), k);
    println!("fft cpu takes {:?}", now.elapsed());
    println!("n is {:?}", 1<<k);

    let twiddle = generate_twiddle_lookup_table(domain.get_omega(), k as u32, 10, true);
    let now: Instant = Instant::now();
    unsafe{
        //let tmp = coeff_gpu.as_mut_slice().as_mut_ptr() as *mut u64;
        //let tmp1 = std::slice::from_raw_parts(tmp, 4*5);
        //println!("data is {:?}", tmp1);
        //parallel_fft_api(0, coeff_gpu.as_mut_slice().as_mut_ptr() as *mut u64,(&(domain.get_omega()) as *const Fr) as *const u64, twiddle.as_slice().as_ptr() as *const u64,  1<<k, k.into(), twiddle.len() as u64);
        log::info!("firstline of original data {:?}", coeff_gpu.get_mut().unwrap()[0]);
        log::info!("firstline of casted data {:?}", std::mem::transmute::<*mut Fr, *mut u64>(coeff_gpu.get_mut().unwrap().as_mut_slice().as_mut_ptr()));
        log::info!("firstline of another casted data {:?}", coeff_fil.as_mut_slice().as_mut_ptr() as *mut u64);

        parallel_fft_api(
            0, 
            std::mem::transmute::<&Vec<Fr>, *mut u64>(&coeff_gpu.get_mut().unwrap()), 
            std::mem::transmute::<&Fr, *const u64>(&(domain.get_omega())),            twiddle.as_slice().as_ptr() as *const u64,  
            1<<k, 
            k.into(), 
            twiddle.len() as u64);

    }
    println!("fft gpu takes {:?}", now.elapsed());
    let now: Instant = Instant::now();
    //parallel_fft::<halo2_proofs::halo2curves::bn256::G1Affine>(coeff_fil.as_mut_slice(), &worker, &domain.get_omega(), k, 6);
    serial_fft::<halo2_proofs::halo2curves::bn256::G1Affine>(coeff_fil.as_mut_slice(), &domain.get_omega(), k);
    println!("fft filecoin parelle takes {:?}", now.elapsed());

    println!("cpu result: {:?}, gpu result: {:?}, filecoin fft reuslt: {:?}", coeff_cpu[0], coeff_gpu.get_mut().unwrap()[0], coeff_fil[0])
    //assert_eq!(coeff_cpu[0], coeff_gpu[0]);
}

fn computing_fft<'b, G: Group>(pool: &Worker, a: Arc<Mutex<Vec<G>>>, domain: Arc<EvaluationDomain<G>>, twiddle: Arc<Vec<Fr>>,) -> Waiter<u64> {
    pool.compute(move || 
    {
        let gpu_idx = LOCKER_1GPU.acquire_gpu(FFT_MEM.get_mem(domain.k()));
        let mut gpu_ret_code: u64;
        let now: Instant = Instant::now();
        unsafe { 
            gpu_ret_code = parallel_fft_api(
                gpu_idx, 
                //a.as_mut_ptr() as *mut u64,  
                //std::mem::transmute::<&Vec<G>, *mut u64>(a.get_mut().unwrap().as_mut_slice().as_mut_ptr()), 
                a.lock().unwrap().as_mut_slice().as_mut_ptr() as *mut u64,
                &(domain.get_omega()) as *const G::Scalar as *const u64,
                //std::mem::transmute::<&G::Scalar, *const u64>(&(domain.get_omega())), 
                twiddle.as_slice().as_ptr() as *const u64,  
                1<<domain.k(), 
                domain.k() as u64, 
                twiddle.len() as u64);
        }
        log::info!("One GPU Job finish takes {:?}", now.elapsed());

        LOCKER_1GPU.release_gpu(FFT_MEM.get_mem(domain.k()), gpu_idx);
        gpu_ret_code
    })
    
}

#[test]
fn test_fft_gpu_with_lock() {
    use env_logger::Builder;
    use log::LevelFilter;
    use rand::seq::SliceRandom;

    Builder::new().filter(None, LevelFilter::Debug).parse_default_env().init();

    let job_num = 100;
    let rounds = 4;
    let worker = Worker::new();

    let now: Instant = Instant::now();
    let k: u32 = 24;
    let domain = Arc::new(EvaluationDomain::<Fr>::new(3, k));
    let twiddle = Arc::new(generate_twiddle_lookup_table(domain.get_omega(), k as u32, 10, true));

    
    for round in 0..rounds {
        log::info!("ROUND {:?}", round);
        log::info!("Build input data");
        let now: Instant = Instant::now();
        let mut coeffs_gpu = Vec::new();
        let mut coeffs_cpu = Vec::new();


        (0..job_num).for_each(|_| {
            let mut coeff = (0..(1 << k)).into_par_iter().map(|_| Fr::random(OsRng)).collect::<Vec<_>>();
            coeffs_gpu.push(Arc::new(Mutex::new(coeff.clone())));
            coeffs_cpu.push(coeff);
        });
        let sample_idx = (0..job_num).collect::<Vec<_>>().choose(&mut rand::thread_rng()).unwrap().clone();
        log::info!("sample_idx is {:?} coeffs length is ", sample_idx);

        let mut coeff_sampe = coeffs_gpu[sample_idx].lock().unwrap().clone();
        log::info!("Build input data takes {:?}", now.elapsed());

        let now: Instant = Instant::now();
        let gpu_ret_codes = coeffs_gpu.iter().map(
            |coeff| {
                computing_fft(&worker, coeff.clone(), domain.clone(), twiddle.clone())
            }
        ).collect::<Vec<_>>();

        log::info!("GPU Job distribution takes {:?}", now.elapsed());
        let gpu_ret_codes = gpu_ret_codes.iter().map( |gpu_ret_code| {
            gpu_ret_code.wait()
        }).collect::<Vec<_>>();
        log::info!("GPU Job total finish takes {:?}", now.elapsed());

        assert_eq!(gpu_ret_codes, (0..job_num).map(|_| 0).collect::<Vec<u64>>());
        
        let now: Instant = Instant::now();
        for coeff in coeffs_cpu.as_mut_slice().iter_mut() {
            best_fft(coeff.as_mut_slice(), domain.get_omega(), k);
        }
        log::info!("CPU Job total finish takes {:?}", now.elapsed());

        (0..job_num).for_each(|idx| {
            assert_eq!(coeffs_cpu[idx], *(coeffs_gpu[idx].lock().unwrap()));
        });
    }
    
    log::info!("ALL TEST FINISHED SUCCESSFULLY!");


}

pub fn serial_fft<C: CurveAffine>(a: &mut [C::Scalar], omega: &C::Scalar, log_n: u32) {
    fn bitreverse(mut n: u32, l: u32) -> u32 {
        let mut r = 0;
        for _ in 0..l {
            r = (r << 1) | (n & 1);
            n >>= 1;
        }
        r
    }

    let n = a.len() as u32;
    assert_eq!(n, 1 << log_n);

    for k in 0..n {
        let rk = bitreverse(k, log_n);
        if k < rk {
            a.swap(rk as usize, k as usize);
        }
    }

    let mut m = 1;
    for _ in 0..log_n {
        let w_m = omega.pow_vartime(&[u64::from(n / (2 * m))]);

        let mut k = 0;
        while k < n {
            let mut w = C::Scalar::one();
            for j in 0..m {
                let mut t = a[(k + j + m) as usize];
                t *= w;
                let mut tmp = a[(k + j) as usize];
                tmp -= t;
                a[(k + j + m) as usize] = tmp;
                a[(k + j) as usize] += t;
                w *= w_m;
            }

            k += 2 * m;
        }

        m *= 2;
    }
}
/// Calculate the Fast Fourier Transform on the CPU (multithreaded).
///
/// The result is is written to the input `a`.
/// The number of threads used will be `2^log_threads`.
/// There must be more items to process than threads.
pub fn parallel_fft<C: CurveAffine>(
    a: &mut [C::Scalar],
    worker: &Worker,
    omega: &C::Scalar,
    log_n: u32,
    log_threads: u32,
) {
    assert!(log_n >= log_threads);

    let num_threads = 1 << log_threads;
    let log_new_n = log_n - log_threads;
    let mut tmp = vec![vec![C::Scalar::zero(); 1 << log_new_n]; num_threads];
    let new_omega = omega.pow_vartime(&[num_threads as u64]);

    worker.scope(0, |scope, _| {
        let a = &*a;

        for (j, tmp) in tmp.iter_mut().enumerate() {
            scope.execute(move || {
                // Shuffle into a sub-FFT
                let omega_j = omega.pow_vartime(&[j as u64]);
                let omega_step = omega.pow_vartime(&[(j as u64) << log_new_n]);

                let mut elt = C::Scalar::zero();
                for (i, tmp) in tmp.iter_mut().enumerate() {
                    for s in 0..num_threads {
                        let idx = (i + (s << log_new_n)) % (1 << log_n);
                        let mut t = a[idx];
                        t *= elt;
                        *tmp += t;
                        elt *= omega_step;
                    }
                    elt *= omega_j;
                }

                // Perform sub-FFT
                serial_fft::<C>(tmp, &new_omega, log_new_n);
            });
        }
    });

    // TODO: does this hurt or help?
    worker.scope(a.len(), |scope, chunk| {
        let tmp = &tmp;

        for (idx, a) in a.chunks_mut(chunk).enumerate() {
            scope.execute(move || {
                let mut idx = idx * chunk;
                let mask = (1 << log_threads) - 1;
                for a in a {
                    *a = tmp[idx & mask][idx >> log_threads];
                    idx += 1;
                }
            });
        }
    });

}

#[test]
fn test_msm_gpu_with_lock() {
    use halo2curves::bn256::Bn256;
    use halo2curves::pairing::Engine;

    use env_logger::Builder;
    use log::LevelFilter;
    let worker = Worker::new();

    Builder::new().filter(None, LevelFilter::Debug).parse_default_env().init();
    log::info!("MSM testing started");
    let k = 20;
    let rounds = 200;
    let params = ParamsKZG::<Bn256>::new(k);
    let domain: EvaluationDomain<Fr> = EvaluationDomain::new(1, k);
    log::info!("params has been created with k={:?}", params.k());

    let mut a_cpu = domain.empty_lagrange();

    a_cpu.par_iter_mut().for_each(|ele| *ele = Fr::random(OsRng));
    let alpha_cpu = Blind(Fr::random(OsRng));

    let now: Instant = Instant::now();
    let mut commited_cpu: Vec<<Bn256 as Engine>::G1> = Vec::new();
    let w_params = Arc::new(params.clone());
    for i in 0..rounds {
        //log::info!("running {:?} rounds", i);
        let cpu_res = params.commit_lagrange(&a_cpu, alpha_cpu);
        commited_cpu.push(cpu_res);
       
    }
    log::info!("commit_lagrange cpu takes {:?}", now.elapsed());

    
    let now: Instant = Instant::now();
    let commited_wait = (0..rounds).map(|_|
        commit_lagrange_gpu(&worker, w_params.clone(), a_cpu.clone())
    ).collect::<Vec<_>>();

    let commited_gpu = commited_wait.iter().map(|waiter| 
        waiter.wait().unwrap()
    ).collect::<Vec<_>>();
    log::info!("commit_lagrange gpu takes {:?}", now.elapsed());

    for i in 0..rounds {
        // log::info!("round {:?} verify", i);
        // log::info!("commited cpu is {:?}", commited_cpu[i]);
        // log::info!("commited gpu is {:?}", commited_gpu[i]);
        assert_eq!(commited_cpu[i], commited_gpu[i])
    }
    log::info!("ALL TESTING IS DONE!");
}


#[test]
fn test_coeff_to_extend_gpu() {

    use env_logger::Builder;
    use log::LevelFilter;

    Builder::new().filter(None, LevelFilter::Debug).parse_default_env().init();

    log::info!("Start coeff to extend testing");

    let now: Instant = Instant::now();
    let k: u32 = 17;
    let domain = EvaluationDomain::<Fr>::new(2, k);
    
    let mut poly_cpu = domain.empty_coeff();

    poly_cpu.par_iter_mut().for_each(|ele| *ele = Fr::random(OsRng));

    let mut poly_gpu = poly_cpu.clone();

    log::info!("Prepare data takes {:?}", now.elapsed());
    let now: Instant = Instant::now();
    let current_extended_omega = <halo2_proofs::halo2curves::bn256::G1Affine as CurveAffine>::ScalarExt::one();

    let poly_cpu_extend = domain.coeff_to_extended_part(poly_cpu, current_extended_omega);
    log::info!("coeff to extend cpu takes {:?}", now.elapsed());

    let twiddle = generate_twiddle_lookup_table(domain.get_omega(), k as u32, 10, true);
    let now: Instant = Instant::now();
    unsafe{
        parallel_coeff_to_extended_part_api(
            0, 
            (*poly_gpu).as_mut_ptr() as *mut u64,
            std::mem::transmute::<&Fr, *const u64>(&(domain.get_omega())),            
            twiddle.as_slice().as_ptr() as *const u64,  
            1<<k, 
            k.into(), 
            twiddle.len() as u64,
            &(Fr::ZETA * current_extended_omega) as *const Fr as *const u64
        );

    }
    
    log::info!("coeff to extend gpu takes {:?}", now.elapsed());
    let mut poly_gpu_extend = domain.empty_lagrange();

    poly_gpu_extend.par_iter_mut().enumerate().for_each(|(i, ele)| *ele = poly_gpu[i]);
  
    log::info!("cpu result: {:?}, gpu result: {:?}", poly_cpu_extend[0], poly_gpu_extend[0]);
    assert_eq!(*poly_cpu_extend, *poly_gpu_extend);
    log::info!("Complete coeff to extend testing");

}