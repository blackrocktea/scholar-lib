
use crate::dataset::Dataset;
use crate::utils::*;

use nalgebra::DMatrix;

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{fs, marker::PhantomData, path::Path};

/// A fully-connected neural network.
#[derive(Serialize, Deserialize)]