
use rand::seq::SliceRandom;

/// A tuple containing a vector of input values matched to a vector of their expected output values
type Row = (Vec<f64>, Vec<f64>);

/// A collection of input vectors matched with their expected output.
///
/// You can construct a `Dataset` manually like so:
///
/// ```rust
/// // Note that the inputs and target outputs are both vectors, even though the latter has just
/// // one element
/// let data = vec![
///     (vec![0.0, 0.0], vec![0.0]),
///     (vec![0.0, 1.0], vec![1.0]),
///     (vec![1.0, 0.0], vec![1.0]),
///     (vec![1.0, 1.0], vec![0.0]),
/// ];
///
/// let dataset = scholar::Dataset::from(data);
/// ```
#[derive(Debug)]
pub struct Dataset {
    data: Vec<Row>,
}

impl Dataset {
    /// Parses a `Dataset` from a CSV file.
    ///
    /// # Arguments
    ///
    /// * `file_path` - The path to the CSV file
    /// * `includes_headers` - Whether the CSV has a header row or not
    /// * `num_inputs` - The number of columns in the CSV that are designated as inputs (to a
    /// Machine Learning model)