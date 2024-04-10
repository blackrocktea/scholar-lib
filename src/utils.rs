
use nalgebra::DMatrix;
use rand::distributions::{Distribution, Uniform};

/// Generates a matrix with the specified dimensions and random values between -1 and 1.
pub(crate) fn gen_random_matrix(rows: usize, cols: usize) -> DMatrix<f64> {
    let elements = rows * cols;
    let range = Uniform::new_inclusive(-1.0, 1.0);
    DMatrix::from_iterator(