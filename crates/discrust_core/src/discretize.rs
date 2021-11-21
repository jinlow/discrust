use crate::feature::Feature;
use crate::node::{Node, NodePtr};
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use std::collections::VecDeque;
use std::fmt;

impl std::convert::From<NotFittedError> for PyErr {
    fn from(err: NotFittedError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct NotFittedError;

impl fmt::Display for NotFittedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Calling a method that requires object to be fit, when `fit` has not been called."
        )
    }
}

pub struct Discretizer {
    min_obs: f64,
    max_bins: i64,
    min_iv: f64,
    min_pos: f64,
    pub mono: Option<i8>,
    root_node: NodePtr,
    pub splits_: Vec<f64>,
    pub feature: Option<Feature>,
}

impl Discretizer {
    pub fn new(
        min_obs: Option<f64>,
        max_bins: Option<i64>,
        min_iv: Option<f64>,
        min_pos: Option<f64>,
        mono: Option<i8>,
    ) -> Self {
        let min_obs = min_obs.unwrap_or(5.0);
        let max_bins = max_bins.unwrap_or(10);
        let min_iv = min_iv.unwrap_or(0.001);
        let min_pos = min_pos.unwrap_or(5.0);
        Discretizer {
            min_obs,
            max_bins,
            min_iv,
            min_pos,
            mono,
            root_node: None,
            splits_: Vec::new(),
            feature: None,
        }
    }

    pub fn fit(
        &mut self,
        x: &[f64],
        y: &[f64],
        w: &[f64],
        exceptions: Option<Vec<f64>>,
    ) -> Vec<f64> {
        // Reset the splits
        self.splits_ = Vec::new();
        let e = match exceptions {
            Some(v) => v,
            None => Vec::new(),
        };
        let feature = Feature::new(x, y, w, &e);
        let root_node = Node::new(
            &feature,
            Some(self.min_obs),
            Some(self.min_iv),
            Some(self.min_pos),
            self.mono,
            None,
            None,
            None,
            None,
        );

        self.root_node = Some(Box::new(root_node));
        let mut que = VecDeque::new();
        que.push_front(self.root_node.as_mut());
        let mut n_bins = 1;
        while que.len() > 0 {
            // If we are running this piece of code, the que is not empty, so
            // we can always safely unwrap.
            let mut node = que.pop_back().unwrap().unwrap();
            let info = node.find_best_split(&feature);
            // If this feature doesn't produce a valid
            // split, just continue this is a terminal node.
            let split = match info.split {
                Some(split) => split,
                None => continue,
            };
            n_bins += 1;
            if n_bins > self.max_bins {
                break;
            }

            // If monotonicity is None, then we can set it right
            // now based on the monotonicity of the best first
            // split.
            if self.mono.is_none() {
                let split_sign = if info.lhs_woe < info.rhs_woe { 1 } else { -1 };
                self.mono = Some(split_sign);
            }

            let split_idx = feature.vals_.iter().position(|&v| v > split);

            let lhs_node = Node::new(
                &feature,
                Some(self.min_obs),
                Some(self.min_iv),
                Some(self.min_pos),
                self.mono,
                info.lhs_woe,
                info.lhs_iv,
                Some(node.start),
                split_idx,
            );
            let rhs_node = Node::new(
                &feature,
                Some(self.min_obs),
                Some(self.min_iv),
                Some(self.min_pos),
                self.mono,
                info.rhs_woe,
                info.rhs_iv,
                split_idx,
                Some(node.stop),
            );

            // Add the split info here, after we use it, to avoid a move.
            node.split_info = info;

            node.left_node = Some(Box::new(lhs_node));
            node.right_node = Some(Box::new(rhs_node));
            que.push_front(node.left_node.as_mut());
            que.push_front(node.right_node.as_mut());
            self.splits_.push(split);
        }
        // Take ownership of feature for now.
        self.feature = Some(feature);
        self.splits_.push(-f64::INFINITY);
        self.splits_.push(f64::INFINITY);
        self.splits_.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.splits_.to_vec()
    }

    pub fn predict(&self, x: &[f64]) -> Result<Vec<f64>, NotFittedError> {
        let res: Vec<f64> = x
            .iter()
            .map(|v| self.predict_record(v))
            .collect::<Result<Vec<f64>, NotFittedError>>()?;
        Ok(res)
    }

    fn predict_record(&self, v: &f64) -> Result<f64, NotFittedError> {
        let mut node = self.root_node.as_ref().ok_or(NotFittedError)?;
        let w: f64;
        loop {
            if node.is_terminal() {
                w = node.woe.to_owned();
                break;
            }
            // If this is not a terminal node, split will always be
            // populated, which is why we are OK to use unwrap here.
            if v > &node.split_info.split.unwrap() {
                node = node.right_node.as_ref().unwrap();
            } else {
                node = node.left_node.as_ref().unwrap();
            }
        }
        Ok(w)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::fs;
    // [-inf, 6.95, 7.125, 7.7292, 10.4625, 15.1, 50.4958, 52.0, 73.5, 79.65, inf]
    #[test]
    fn test_discretizer() {
        let mut fare: Vec<f64> = Vec::new();
        let mut survived: Vec<f64> = Vec::new();
        let file = fs::read_to_string("resources/data.csv")
            .expect("Something went wrong reading the file");
        for l in file.lines() {
            let split: Vec<f64> = l.split(",").map(|x| x.parse::<f64>().unwrap()).collect();
            fare.push(split[0]);
            survived.push(split[1]);
        }
        let mut disc = Discretizer::new(Some(5.0), Some(10), Some(0.001), Some(1.0), Some(1));
        let w_ = vec![1.0; fare.len()];
        let splits = disc.fit(&fare, &survived, &w_, None);
        assert_eq!(
            splits,
            vec![
                -f64::INFINITY,
                6.95,
                7.125,
                7.7292,
                10.4625,
                15.1,
                50.4958,
                52.0,
                73.5,
                79.65,
                f64::INFINITY
            ]
        );
        // println!("{:?}", disc.predict(&fare));
    }

    #[test]
    fn test_discretizer_mono_n1() {
        let mut fare: Vec<f64> = Vec::new();
        let mut survived: Vec<f64> = Vec::new();
        let file = fs::read_to_string("resources/data.csv")
            .expect("Something went wrong reading the file");
        for l in file.lines() {
            let split: Vec<f64> = l.split(",").map(|x| x.parse::<f64>().unwrap()).collect();
            fare.push(split[0]);
            survived.push((split[1] == 0.0) as i64 as f64);
        }
        let mut disc = Discretizer::new(Some(5.0), Some(10), Some(0.001), Some(1.0), Some(-1));
        let w_ = vec![1.0; fare.len()];
        let splits = disc.fit(&fare, &survived, &w_, None);
        assert_eq!(
            splits,
            vec![
                -f64::INFINITY,
                6.95,
                7.125,
                7.7292,
                10.4625,
                15.1,
                50.4958,
                52.0,
                73.5,
                79.65,
                f64::INFINITY
            ]
        );
        // println!("{:?}", disc.predict(&fare));
    }

    #[test]
    fn test_discretizer_mono_none() {
        let mut fare: Vec<f64> = Vec::new();
        let mut survived: Vec<f64> = Vec::new();
        let file = fs::read_to_string("resources/data.csv")
            .expect("Something went wrong reading the file");
        for l in file.lines() {
            let split: Vec<f64> = l.split(",").map(|x| x.parse::<f64>().unwrap()).collect();
            fare.push(split[0]);
            survived.push(split[1]);
        }
        let mut disc = Discretizer::new(Some(5.0), Some(10), Some(0.001), Some(1.0), None);
        let w_ = vec![1.0; fare.len()];
        let splits = disc.fit(&fare, &survived, &w_, None);
        assert_eq!(
            splits,
            vec![
                -f64::INFINITY,
                6.95,
                7.125,
                7.7292,
                10.4625,
                15.1,
                50.4958,
                52.0,
                73.5,
                79.65,
                f64::INFINITY
            ]
        );
        // println!("{:?}", disc.predict(&fare));
    }
}
