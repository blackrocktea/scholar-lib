
use rand::seq::SliceRandom;

/// A tuple containing a vector of input values matched to a vector of their expected output values
type Row = (Vec<f64>, Vec<f64>);

/// A collection of input vectors matched with their expected output.
///
/// You can construct a `Dataset` manually like so: