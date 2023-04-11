use scholar::{Dataset, NeuralNet, Sigmoid};

fn main() -> anyhow::Result<()> {
    let dataset = Dataset::from_csv("examples/iris.csv", false, 4)?;
    let (training_data, testing_data) = dataset.split(0.75);

    let mut brain: NeuralNet<Sigmoid> = NeuralNet::new(&[4, 10, 10, 3]