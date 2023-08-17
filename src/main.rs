mod mat;

use std::time::Instant;

use mat::Mat;

use rand::rngs::ThreadRng;
use rand::Rng;

fn random_mat(rng: &mut ThreadRng, rows: usize, cols: usize) -> Mat {
    let mut res = unsafe { Mat::uninit(rows, cols) };
    for i in 0..res.rows() {
        for j in 0..res.cols() {
            res[i][j] = rng.gen_range(0.0..1000.0);
        }
    }
    return res;
}

fn print_mat(mat: &Mat) {
    for i in 0..mat.rows() {
        for j in 0..mat.cols() {
            print!("{} ", mat[i][j]);
        }
        println!("");
    }
}

fn compare_mats(m1: &Mat, m2: &Mat) {
    assert_eq!(m1.rows(), m2.rows());
    assert_eq!(m1.cols(), m2.cols());
    for i in 0..m1.rows() {
        for j in 0..m2.cols() {
            assert_eq!(m1[i][j], m2[i][j]);
        }
    }
}

fn main() {
    let mut rng = rand::thread_rng();
    let dim = 1000;
    let now = Instant::now();
    let m1 = random_mat(&mut rng, dim, dim);
    let m2 = random_mat(&mut rng, dim, dim);
    let elapsed = now.elapsed();
    println!("Initializing took {:?}", elapsed);

    let now = Instant::now();
    let res1 = mat::mul_normal(&m1, &m2);
    let elapsed = now.elapsed();
    println!("Normal mul took {:?}", elapsed);

    let now = Instant::now();
    let res2 = mat::mul_transposed(&m1, &m2);
    let elapsed = now.elapsed();
    println!("Transposed mul took {:?}", elapsed);

    let now = Instant::now();
    let res3 = mat::mul_unrolled(&m1, &m2);
    let elapsed = now.elapsed();
    println!("Unrolled mul took {:?}", elapsed);
    compare_mats(&res1, &res2);
    compare_mats(&res1, &res3);
}
