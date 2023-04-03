use scholar::{Dataset, NeuralNet, Sigmoid};

fn main() -> anyhow::Result<()> {
    let dataset = Dataset::from_csv("examples/iris.csv", false, 4)?;
    let (training_data, test