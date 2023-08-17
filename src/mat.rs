use std::cmp::min;
use std::ops::{Index, IndexMut};

pub struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Mat {
    pub fn rows(&self) -> usize {
        return self.rows;
    }
    pub fn cols(&self) -> usize {
        return self.cols;
    }

    pub fn zeros(rows: usize, cols: usize) -> Mat {
        let buf_len = rows * cols;
        let data = vec![0.0; buf_len];
        return Mat { rows, cols, data };
    }

    pub unsafe fn uninit(rows: usize, cols: usize) -> Mat {
        let buf_len = rows * cols;
        let mut data = Vec::with_capacity(buf_len);
        data.set_len(buf_len);
        return Mat { rows, cols, data };
    }
    pub fn identity(rows: usize, cols: usize) -> Mat {
        let buf_len = rows * cols;
        let mut data = Vec::with_capacity(buf_len);
        for i in 0..rows {
            for j in 0..cols {
                if i == j {
                    data.push(1.0);
                } else {
                    data.push(0.0);
                }
            }
        }
        return Mat { rows, cols, data };
    }
    pub fn get_transposed(&self) -> Mat {
        let mut res = Mat::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res[j][i] = self[i][j];
            }
        }
        return res;
    }
}

pub fn mul_normal(m1: &Mat, m2: &Mat) -> Mat {
    assert_eq!(m1.cols, m2.rows);
    let mut res = unsafe { Mat::uninit(m1.rows, m2.cols) };

    for i in 0..res.rows {
        for j in 0..res.cols {
            let mut val = 0.0;
            for k in 0..m1.cols {
                val += m1[i][k] * m2[k][j];
            }
            res[i][j] = val;
        }
    }
    return res;
}

pub fn mul_transposed(m1: &Mat, m2: &Mat) -> Mat {
    assert_eq!(m1.cols, m2.rows);

    let mut res = unsafe { Mat::uninit(m1.rows, m2.cols) };
    let transposed = m2.get_transposed();

    for i in 0..res.rows {
        for j in 0..res.cols {
            let mut val = 0.0;
            for k in 0..m1.cols {
                val += m1[i][k] * transposed[j][k];
            }
            res[i][j] = val;
        }
    }
    return res;
}

pub fn mul_unrolled(m1: &Mat, m2: &Mat) -> Mat {
    assert_eq!(m1.cols, m2.rows);

    let mut res = Mat::zeros(m1.rows, m2.cols);

    const CACHE_LINE_SIZE: usize = 64;
    const VALS_PER_CACHE_LINE: usize = CACHE_LINE_SIZE / std::mem::size_of::<f64>();

    for i in (0..res.rows).step_by(VALS_PER_CACHE_LINE) {
        for j in (0..res.cols).step_by(VALS_PER_CACHE_LINE) {
            for k in (0..m1.cols).step_by(VALS_PER_CACHE_LINE) {
                for i2 in i..min(i + VALS_PER_CACHE_LINE, res.rows) {
                    for k2 in k..min(k + VALS_PER_CACHE_LINE, m1.cols) {
                        for j2 in j..min(j + VALS_PER_CACHE_LINE, res.cols) {
                            res[i2][j2] += m1[i2][k2] * m2[k2][j2];
                        }
                    }
                }
            }
        }
    }
    return res;
}

impl<'a> Index<usize> for Mat {
    type Output = [f64];

    fn index(&self, index: usize) -> &Self::Output {
        return &self.data[index * self.cols..index * self.cols + self.cols];
    }
}

impl<'a> IndexMut<usize> for Mat {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        return &mut self.data[index * self.cols..index * self.cols + self.cols];
    }
}
