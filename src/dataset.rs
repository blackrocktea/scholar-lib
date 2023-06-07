
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
    ///
    /// # Examples
    /// ```rust
    /// // Parses the first four columns of 'iris.csv' as inputs, and the remaining columns as
    /// // target outputs
    /// let dataset = scholar::Dataset::from_csv("iris.csv", false, 4);
    /// ```
    pub fn from_csv(
        file_path: impl AsRef<std::path::Path>,
        includes_headers: bool,
        num_inputs: usize,
    ) -> Result<Self, ParseCsvError> {
        use std::str::FromStr;

        let file = std::fs::File::open(file_path)?;
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(includes_headers)
            .from_reader(file);

        let data: Result<Vec<Row>, ParseCsvError> = reader
            .records()
            .map(|row| {
                // Catches a possible parsing error
                let row = row?;
                let row = row
                    .iter()
                    .map(|val| {
                        let val = val.trim();
                        f64::from_str(val)
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let mut inputs = row;
                // Splits the row into input and output vectors
                let outputs = inputs.split_off(num_inputs);
                Ok((inputs, outputs))
            })
            .collect();
        Ok(Dataset::from(data?))
    }

    /// Splits the dataset into two, with the size of each determined by the given `train_portion`.
    /// This is useful for separating it into training and testing segments.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let dataset = scholar::Dataset::from_csv("iris.csv", false, 4)?;
    ///
    /// // Randomly allocates 75% of the original dataset to `training_data`, and the rest
    /// // to `testing_data`
    /// let (training_data, testing_data) = dataset.split(0.75);
    /// ```
    ///
    /// # Panics
    ///
    /// This method panics if the given `train_portion` isn't between 0 and 1.
    pub fn split(mut self, train_portion: f64) -> (Self, Self) {
        if train_portion < 0.0 || train_portion > 1.0 {
            panic!(