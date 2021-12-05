use criterion::{black_box, criterion_group, criterion_main, Criterion};

use discrust_core::Discretizer;
use std::fs;

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Fit Discretizer", |b| b.iter(|| {
        let mut fare: Vec<f64> = Vec::new();
        let mut survived: Vec<f64> = Vec::new();
        let file = fs::read_to_string("resources/data.csv")
            .expect("Something went wrong reading the file");
        for l in file.lines() {
            let split: Vec<f64> = l.split(",").map(|x| x.parse::<f64>().unwrap()).collect();
            fare.push(split[0]);
            survived.push(split[1]);
        }
        for _ in 0..5 {
            fare.extend(fare.to_vec());
            survived.extend(survived.to_vec());
        }
        let mut disc = Discretizer::new(Some(5.0), Some(10), Some(0.001), Some(1.0), Some(1));
        let w_ = vec![1.0; fare.len()];
        let splits = disc.fit(&fare, &survived, &w_, None).unwrap();
    }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);