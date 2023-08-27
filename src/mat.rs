use std::cmp::min;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul};

pub trait Number:
    Copy + From<u8> + PartialEq + PartialOrd + Sized + Mul<Self, Output = Self> + Add + AddAssign
{
}
impl<T> Number for T where
    T: Copy + From<u8> + PartialEq + PartialOrd + Mul<Self, Output = Self> + Add + AddAssign
{
}

pub struct Mat<T: Number> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Number> Mat<T> {
    pub fn rows(&self) -> usize {
        return self.rows;
    }
    pub fn cols(&self) -> usize {
        return self.cols;
    }

    pub fn zeros(rows: usize, cols: usize) -> Mat<T> {
        let buf_len = rows * cols;
        let data = vec![T::from(0); buf_len];
        return Mat { rows, cols, data };
    }

    pub unsafe fn uninit(rows: usize, cols: usize) -> Mat<T> {
        let buf_len = rows * cols;
        let mut data = Vec::with_capacity(buf_len);
        data.set_len(buf_len);
        return Mat { rows, cols, data };
    }
    pub fn identity(rows: usize, cols: usize) -> Mat<T> {
        let buf_len = rows * cols;
        let mut data = Vec::with_capacity(buf_len);
        for i in 0..rows {
            for j in 0..cols {
                if i == j {
                    data.push(T::from(1));
                } else {
                    data.push(T::from(0));
                }
            }
        }
        return Mat { rows, cols, data };
    }
    pub fn get_transposed(&self) -> Mat<T> {
        let mut res = Mat::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res[j][i] = self[i][j];
            }
        }
        return res;
    }
}

impl<'a, T: Number> Index<usize> for Mat<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        return &self.data[index * self.cols..index * self.cols + self.cols];
    }
}

impl<'a, T: Number> IndexMut<usize> for Mat<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return &mut self.data[index * self.cols..index * self.cols + self.cols];
    }
}

pub fn mul_normal<T: Number>(m1: &Mat<T>, m2: &Mat<T>) -> Mat<T> {
    assert_eq!(m1.cols, m2.rows);
    let mut res = unsafe { Mat::uninit(m1.rows, m2.cols) };

    for i in 0..res.rows {
        for j in 0..res.cols {
            let mut val = T::from(0);
            for k in 0..m1.cols {
                val += m1[i][k] * m2[k][j];
            }
            res[i][j] = val;
        }
    }
    return res;
}

pub fn mul_with_transposed<T: Number>(m1: &Mat<T>, m2: &Mat<T>) -> Mat<T> {
    assert_eq!(m1.cols, m2.cols);

    let mut res = unsafe { Mat::uninit(m1.rows, m2.cols) };
    for i in 0..res.rows {
        for j in 0..res.cols {
            let mut val = T::from(0);
            for k in 0..m1.cols {
                val += m1[i][k] * m2[j][k];
            }
            res[i][j] = val;
        }
    }
    return res;
}

pub fn mul_transposed<T: Number>(m1: &Mat<T>, m2: &Mat<T>) -> Mat<T> {
    assert_eq!(m1.cols, m2.rows);

    let transposed = m2.get_transposed();
    return mul_with_transposed(m1, &transposed);
}

fn unrolled_step<T: Number>(
    res: &mut Mat<T>,
    m1: &Mat<T>,
    m2: &Mat<T>,
    outer_i: usize,
    outer_j: usize,
    outer_k: usize,
    stride: usize,
) {
    for i in outer_i..min(outer_i + stride, res.rows) {
        for k in outer_k..min(outer_k + stride, m1.cols) {
            for j in outer_j..min(outer_j + stride, res.cols) {
                res[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
}

pub fn mul_unrolled<T: Number>(m1: &Mat<T>, m2: &Mat<T>) -> Mat<T> {
    assert_eq!(m1.cols, m2.rows);

    let mut res = Mat::zeros(m1.rows, m2.cols);

    const CACHE_LINE_SIZE: usize = 64;
    let size = std::mem::size_of::<T>();
    let vals_per_cache_line: usize = CACHE_LINE_SIZE / size;

    for i in (0..res.rows).step_by(vals_per_cache_line) {
        for j in (0..res.cols).step_by(vals_per_cache_line) {
            for k in (0..m1.cols).step_by(vals_per_cache_line) {
                unrolled_step(&mut res, m1, m2, i, j, k, vals_per_cache_line);
            }
        }
    }
    return res;
}
