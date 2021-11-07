use discrust_core::Discretizer;
use std::fs;

fn main() {
    let mut fare: Vec<f64> = Vec::new();
    let mut survived: Vec<f64> = Vec::new();
    let file =
        fs::read_to_string("resources/data.csv").expect("Something went wrong reading the file");
    for l in file.lines() {
        let split: Vec<f64> = l.split(",").map(|x| x.parse::<f64>().unwrap()).collect();
        fare.push(split[0]);
        survived.push(split[1]);
    }
    let w_ = vec![1.0; fare.len()];
    let mut disc = Discretizer::new(Some(5.0), Some(10), Some(0.001), Some(1.0), Some(1));
    let splits = disc.fit(&fare, &survived, &w_);
    println!("{:?}", splits);
    for i in 0..1 {
        println!("Val {}", i);
    }
}
