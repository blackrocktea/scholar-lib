use scholar::{Dataset, NeuralNet, Sigmoid};

fn main() -> anyhow::Result<()> {
    let dataset = Dataset::from_csv("examples/iri