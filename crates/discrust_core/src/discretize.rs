use crate::errors::DiscrustError;
use crate::feature::Feature;
use crate::node::{Node, NodePtr};
use crate::utils::nan_safe_compare;
use std::cmp::Ordering;
use std::collections::VecDeque;

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
        exception_values: Option<Vec<f64>>,
    ) -> Result<Vec<f64>, DiscrustError> {
        // Reset the splits
        self.splits_ = Vec::new();
        let e = match exception_values {
            Some(v) => v,
            None => Vec::new(),
        };
        let feature = Feature::new(x, y, w, &e)?;
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
        Ok(self.splits_.to_vec())
    }

    pub fn predict_woe(&self, x: &[f64]) -> Result<Vec<f64>, DiscrustError> {
        // First we check if this is an exception value, to do this, we need
        // to check if the value is present in the exception struct.
        let feature = self
            .feature
            .as_ref()
            .ok_or_else(|| DiscrustError::NotFitted)?;
        let res: Vec<f64> = x
            .iter()
            .map(|v| self.predict_record_woe(v, feature))
            .collect::<Result<Vec<f64>, DiscrustError>>()?;
        Ok(res)
    }

    pub fn predict_idx(&self, x: &[f64]) -> Result<Vec<i64>, DiscrustError> {
        // We don't need the first, value, as this will be negative infinity.
        let all_splits = &self.splits_.as_slice()[1..];
        let feature = self
            .feature
            .as_ref()
            .ok_or_else(|| DiscrustError::NotFitted)?;
        let res: Vec<i64> = x
            .iter()
            .map(|v| self.predict_record_idx(v, all_splits, feature))
            .collect::<Result<Vec<i64>, DiscrustError>>()?;
        Ok(res)
    }

    fn predict_record_idx(
        &self,
        v: &f64,
        all_splits: &[f64],
        feature: &Feature,
    ) -> Result<i64, DiscrustError> {
        // If it's an exception value, we return the index negative value.
        // We start this at -1. So we add 1, to the zero indexed result
        // of the `exception_idx` function.
        if let Some(i) = feature.exception_values.exception_idx(v) {
            return Ok(((i + 1) as i64) * -1);
        }
        let idx = all_splits
            .iter()
            // If the value is less than, or equal to the bin edge, we are in that
            // position bin.
            .position(|x| match nan_safe_compare(x, v) {
                Ordering::Greater => true,
                Ordering::Equal => true,
                _ => false,
            })
            .ok_or(DiscrustError::Prediction)?;
        Ok(idx as i64)
    }
    // -1, 4, 10
    fn predict_record_woe(&self, v: &f64, feature: &Feature) -> Result<f64, DiscrustError> {
        let excp_idx = feature.exception_values.exception_idx(v);
        if let Some(idx) = excp_idx {
            if feature.exception_values.totals_ct_[idx] == 0.0 {
                return Ok(0.0);
            }
            return Ok(feature.exception_values.woe_[idx]);
        }
        let mut node = self
            .root_node
            .as_ref()
            .ok_or_else(|| DiscrustError::NotFitted)?;
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
        let splits = disc.fit(&fare, &survived, &w_, None).unwrap();

        disc.predict_idx(&fare).unwrap();
        disc.predict_woe(&fare).unwrap();

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

        // Test predictions run without error.
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
        let splits = disc.fit(&fare, &survived, &w_, None).unwrap();
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
        let splits = disc.fit(&fare, &survived, &w_, None).unwrap();
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
    fn test_discretizer_mono_none_nan_excp() {
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
        fare[10] = f64::NAN;
        let splits = disc
            .fit(&fare, &survived, &w_, Some(vec![f64::NAN]))
            .unwrap();
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
