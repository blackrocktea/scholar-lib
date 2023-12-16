
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
    /// // Creates a neural network with two input nodes, a single hidden layer with two nodes,
    /// // and one output node
    /// let brain: NeuralNet<Sigmoid> = NeuralNet::new(&[2, 2, 1]);
    /// ```
    ///
    /// # Panics
    ///
    /// This function panics if the number of layers (i.e. the length of the given `node_counts`
    /// slice) is less than 2.
    pub fn new(node_counts: &[usize]) -> Self {
        let num_layers = node_counts.len();
        if num_layers < 2 {
            panic!(
                "not enough layers supplied (expected at least 2, found {})",
                num_layers
            );
        }

        Self {
            layers: node_counts.iter().map(|c| DMatrix::zeros(*c, 1)).collect(),
            weights: (1..num_layers)
                .map(|i| gen_random_matrix(node_counts[i], node_counts[i - 1]))
                .collect(),
            biases: node_counts
                .iter()
                .skip(1)
                .map(|c| gen_random_matrix(*c, 1))
                .collect(),
            errors: node_counts
                .iter()
                .skip(1)
                .map(|c| DMatrix::zeros(*c, 1))
                .collect(),
            activation: PhantomData,
        }
    }

    /// Creates a new `NeuralNet` from a valid file (those created using
    /// [`NeuralNet::save()`](#method.save)).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scholar::{NeuralNet, Sigmoid};
    ///
    /// let brain: NeuralNet<Sigmoid> = NeuralNet::from_file("brain.network")?;
    /// ```
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, LoadErr> {
        let file = fs::File::open(path)?;
        let decoded: NeuralNet<A> = bincode::deserialize_from(file)?;

        Ok(decoded)
    }

    /// Trains the network on the given `Dataset` for the given number of `iterations`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scholar::{Dataset, NeuralNet, Sigmoid};
    ///
    /// let dataset = Dataset::from_csv("iris.csv", false, 4)?;
    ///
    /// let mut brain: NeuralNet<Sigmoid> = NeuralNet::new(&[4, 10, 10, 1]);
    ///
    /// // Trains the network by iterating over the entire dataset 10,000 times. The last parameter
    /// // (the 'learning rate') dictates how quickly the network 'adapts to the dataset'
    /// brain.train(dataset, 10_000, 0.01);
    /// ```
    pub fn train(&mut self, mut training_dataset: Dataset, iterations: u64, learning_rate: f64) {
        let progress_bar = indicatif::ProgressBar::new(iterations);
        progress_bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("Training [{bar:30}] {percent:>3}% ETA: {eta}")
                .progress_chars("=> "),
        );

        // The progress bar is only updated every percentage progressed so as not to significantly
        // impact the speed of training
        let percentile = iterations / 100;

        for i in 1..iterations {
            training_dataset.shuffle();
            for (inputs, targets) in &training_dataset {
                let guesses = self.guess(inputs);
                self.backpropagate(&guesses, targets, learning_rate);
            }

            if i % percentile == 0 {
                progress_bar.inc(percentile);
            }
        }

        progress_bar.finish_and_clear();
    }

    /// Calculates the average cost of the network.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scholar::{Dataset, NeuralNet, Sigmoid};
    ///
    /// let dataset = Dataset::from_csv("iris.csv", false, 4)?;
    /// let (training_data, testing_data) = dataset.split(0.75);
    ///
    /// let mut brain: NeuralNet<Sigmoid> = NeuralNet::new(&[4, 10, 10, 1]);
    /// brain.train(training_data, 10_000, 0.01);
    ///
    /// let avg_cost = brain.test(testing_data);
    /// println!("Accuracy: {:.2}%", (1.0 - avg_cost) * 100.0);
    /// ```
    pub fn test(&mut self, testing_dataset: Dataset) -> f64 {
        let mut avg_cost = 0.0;
        for (inputs, targets) in &testing_dataset {
            let guesses = self.guess(inputs);
            // Iterates over each guess value, compares it to its target, and
            // sums the costs
            let cost_sum: f64 = guesses
                .iter()
                .zip(targets)