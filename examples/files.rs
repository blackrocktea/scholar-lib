
use scholar::{Dataset, NeuralNet, Sigmoid};

fn main() -> anyhow::Result<()> {
    let data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![0.0]),
        (vec![1.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![1.0]),
    ];

    let dataset = Dataset::from(data);