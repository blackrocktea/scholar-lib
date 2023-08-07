
use crate::dataset::Dataset;
use crate::utils::*;

use nalgebra::DMatrix;

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{fs, marker::PhantomData, path::Path};

/// A fully-connected neural network.
#[derive(Serialize, Deserialize)]
pub struct NeuralNet<A: Activation> {
    layers: Vec<DMatrix<f64>>,
    weights: Vec<DMatrix<f64>>,
    biases: Vec<DMatrix<f64>>,
    errors: Vec<DMatrix<f64>>,
    activation: PhantomData<A>,
}

impl<A: Activation + Serialize + DeserializeOwned> NeuralNet<A> {
    /// Creates a new `NeuralNet` with the given node configuration.
    ///
    /// Note that you must supply a type annotation so that it knows which
    /// [`Activation`](#trait.Activation) to use.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scholar::{NeuralNet, Sigmoid};
    ///