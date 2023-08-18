mod mat;

use std::fmt::{Debug, Display};
use std::time::Instant;

use mat::{Mat, Number};

use rand::distributions::uniform::SampleUniform;
use rand::rngs::ThreadRng;
use rand::Rng;

fn random_mat<T: Number + SampleUniform>(rng: &mut ThreadRng, rows: usize, cols: usize) -> Mat<T> {
    let mut res = unsafe { Mat::uninit(rows, cols) };
    for i in 0..res.rows() {
        for j in 0..res.cols() {
            res[i][j] = rng.gen_range(T::from(0)..T::from(100));
        }
    }
    return res;
}

fn print_mat<T: Number + Display>(mat: &Mat<T>) {
    for i in 0..mat.rows() {
        for j in 0..mat.cols() {
            print!("{} ", mat[i][j]);
        }
        println!("");
    }
}

fn compare_mats<T: Number + Debug>(m1: &Mat<T>, m2: &Mat<T>) {
    assert_eq!(m1.rows(), m2.rows());
    assert_eq!(m1.cols(), m2.cols());
    for i in 0..m1.rows() {
        for j in 0..m2.cols() {
            assert_eq!(m1[i][j], m2[i][j]);
        }
    }
}

fn mat_test<T: Number + Debug + SampleUniform>(size: usize) {
    let mut rng = rand::thread_rng();
    let now = Instant::now();
    let m1 = random_mat::<T>(&mut rng, size, size);
    let m2 = random_mat::<T>(&mut rng, size, size);
    let elapsed = now.elapsed();
    println!(
        "Initializing {} took {:?}",
        std::any::type_name::<T>(),
        elapsed
    );

    let now = Instant::now();
    let res1 = mat::mul_normal(&m1, &m2);
    let elapsed = now.elapsed();
    println!("Normal mul took {:?}", elapsed);

    let now = Instant::now();
    let res2 = mat::mul_transposed(&m1, &m2);
    let elapsed = now.elapsed();
    println!("Transposed mul took {:?}", elapsed);

    let m2_trans = m2.get_transposed();
    let now = Instant::now();
    let res3 = mat::mul_with_transposed(&m1, &m2_trans);
    println!("Transposed without copy took {:?}", now.elapsed());

    let now = Instant::now();
    let res4 = mat::mul_unrolled(&m1, &m2);
    let elapsed = now.elapsed();
    println!("Unrolled mul took {:?}", elapsed);

    let now = Instant::now();
    let res5 = mat::mul_unrolleder(&m1, &m2);
    let elapsed = now.elapsed();
    println!("Unrolleder mul took {:?}", elapsed);
    compare_mats(&res1, &res2);
    compare_mats(&res1, &res3);
    compare_mats(&res1, &res4);
    compare_mats(&res1, &res5);
    println!("");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let size;
    if args.len() < 2 {
        size = 500;
    } else {
        size = args[1].parse().expect("Invalid size");
    }
    mat_test::<f32>(size);
    mat_test::<f64>(size);
    mat_test::<u8>(size);
    mat_test::<u16>(size);
    mat_test::<i16>(size);
    mat_test::<u32>(size);
    mat_test::<i32>(size);
    mat_test::<u64>(size);
    mat_test::<i64>(size);
    mat_test::<u128>(size);
    mat_test::<i128>(size);
}
