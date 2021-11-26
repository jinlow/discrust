use crate::feature::Feature;
use std::cmp::PartialEq;

#[derive(Debug, PartialEq)]
pub struct SplitInfo {
    pub split: Option<f64>,
    pub lhs_iv: Option<f64>,
    pub lhs_woe: Option<f64>,
    pub rhs_iv: Option<f64>,
    pub rhs_woe: Option<f64>,
}

impl SplitInfo {
    pub fn new(split: f64, lhs_iv: f64, lhs_woe: f64, rhs_iv: f64, rhs_woe: f64) -> Self {
        SplitInfo {
            split: Some(split),
            lhs_iv: Some(lhs_iv),
            lhs_woe: Some(lhs_woe),
            rhs_iv: Some(rhs_iv),
            rhs_woe: Some(rhs_woe),
        }
    }
    pub fn new_empty() -> Self {
        SplitInfo {
            split: None,
            lhs_iv: None,
            lhs_woe: None,
            rhs_iv: None,
            rhs_woe: None,
        }
    }
}

pub type NodePtr = Option<Box<Node>>;

#[derive(Debug)]
pub struct Node {
    min_obs: f64,
    min_iv: f64,
    min_pos: f64,
    mono: Option<i8>,
    pub woe: f64,
    pub iv: f64,
    pub start: usize,
    pub stop: usize,
    pub left_node: NodePtr,
    pub right_node: NodePtr,
    pub split_info: SplitInfo,
}

impl Node {
    pub fn new(
        feature: &Feature,
        min_obs: Option<f64>,
        min_iv: Option<f64>,
        min_pos: Option<f64>,
        mono: Option<i8>,
        woe: Option<f64>,
        iv: Option<f64>,
        start: Option<usize>,
        stop: Option<usize>,
    ) -> Self {
        let min_obs = min_obs.unwrap_or(5.0);
        let min_iv = min_iv.unwrap_or(0.001);
        let min_pos = min_pos.unwrap_or(5.0);
        let woe = woe.unwrap_or(0.0);
        let iv = iv.unwrap_or(0.0);
        let start = start.unwrap_or(0);
        let stop = stop.unwrap_or(feature.vals_.len());
        Node {
            min_obs,
            min_iv,
            min_pos,
            mono,
            woe,
            iv,
            start,
            stop,
            left_node: None,
            right_node: None,
            split_info: SplitInfo::new_empty(),
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.left_node.is_none() && self.right_node.is_none()
    }

    fn eval_values<'a>(&self, feature: &'a Feature) -> &'a [f64] {
        // We do not need to evaluate the last value, as this is not a
        // valid value becase there are no records greater than it.
        feature.vals_[self.start..(self.stop - 1)].as_ref()
    }

    pub fn find_best_split(&mut self, feature: &Feature) -> SplitInfo {
        // loop through all the unique levels
        // of the feature, identifying the split
        // that generates the maximum information
        // value
        let mut best_iv = 0.0;
        let mut best_lhs_iv = 0.0;
        let mut best_lhs_woe = 0.0;
        let mut best_rhs_iv = 0.0;
        let mut best_rhs_woe = 0.0;
        let mut best_split = -f64::INFINITY;

        for v in self.eval_values(feature) {
            let ((lhs_ct, lhs_ones), (rhs_ct, rhs_ones)) =
                feature.split_totals_ct_ones_ct(*v, self.start, self.stop);
            // Min response
            if (lhs_ones < self.min_pos) | (rhs_ones < self.min_pos) {
                continue;
            }

            // Min observations count
            if (lhs_ct < self.min_obs) | (rhs_ct < self.min_obs) {
                continue;
            }

            // Get information value for split.
            let ((lhs_iv, lhs_woe), (rhs_iv, rhs_woe)) =
                feature.split_iv_woe(*v, self.start, self.stop);

            let total_iv = lhs_iv + rhs_iv;
            if total_iv < self.min_iv {
                continue;
            }

            // Monotonicity check
            // We want to make sure the relationship between the
            // parent node and these two nodes is following the
            // monotonic requirements.
            // If a monotonicity of None was passed, then we will chose the
            // monotonicity of the best first split.
            let split_sign = if lhs_woe < rhs_woe { 1 } else { -1 };
            let check_mono = match self.mono {
                Some(v) => v,
                None => 0,
            };
            if check_mono != 0 {
                if check_mono == -1 {
                    if split_sign == 1 {
                        continue;
                    }
                } else {
                    if split_sign == -1 {
                        continue;
                    }
                }
            }
            // Collect best
            if total_iv > best_iv {
                best_iv = total_iv;
                best_split = *v;
                best_lhs_iv = lhs_iv;
                best_lhs_woe = lhs_woe;
                best_rhs_iv = rhs_iv;
                best_rhs_woe = rhs_woe;
            }
        }
        if best_iv == 0.0 {
            SplitInfo::new_empty()
        } else {
            let split_info = SplitInfo::new(
                best_split,
                best_lhs_iv,
                best_lhs_woe,
                best_rhs_iv,
                best_rhs_woe,
            );
            split_info
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::fs;
    #[test]
    fn test_find_best_split() {
        let x_ = vec![6.2375, 6.4375, 0.0, 0.0, 4.0125, 5.0, 6.45, 6.4958, 6.4958];
        let y_ = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let w_ = vec![1.0; x_.len()];
        let f = Feature::new(&x_, &y_, &w_, &Vec::new()).unwrap();
        let mut n = Node::new(
            &f,
            Some(1.0),
            None,
            Some(0.0),
            Some(1),
            None,
            None,
            None,
            None,
        );
        let comp_info = SplitInfo::new(
            6.2375,
            0.22001303079783097,
            -0.6286086594223742,
            0.3064140580738651,
            0.8754687373539001,
        );
        assert_eq!(n.find_best_split(&f), comp_info);
    }

    #[test]
    fn test_find_best_split_w_excp() {
        // Even with exception values the best split should be
        // the same
        let x_ = vec![
            -1.0, -1.0, 6.2375, 6.4375, 0.0, 100.0, 0.0, 4.0125, 5.0, 6.45, 6.4958, 6.4958,
        ];
        let y_ = vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let w_ = vec![1.0; x_.len()];

        let f = Feature::new(&x_, &y_, &w_, &vec![-1.0, 100.0]).unwrap();
        let mut n = Node::new(
            &f,
            Some(1.0),
            None,
            Some(0.0),
            Some(1),
            None,
            None,
            None,
            None,
        );
        println!("{:?}", f.exception_values);
        assert_eq!(n.find_best_split(&f).split.unwrap(), 6.2375);

        let f = Feature::new(&x_, &y_, &w_, &Vec::new()).unwrap();
        let mut n = Node::new(
            &f,
            Some(1.0),
            None,
            Some(0.0),
            Some(1),
            None,
            None,
            None,
            None,
        );
        println!("{:?}", f.exception_values);
        assert_ne!(n.find_best_split(&f).split.unwrap(), 6.2375);
    }

    #[test]
    fn test_file() {
        let mut fare: Vec<f64> = Vec::new();
        let mut survived: Vec<f64> = Vec::new();
        let file = fs::read_to_string("resources/data.csv")
            .expect("Something went wrong reading the file");
        for l in file.lines() {
            let split: Vec<f64> = l.split(",").map(|x| x.parse::<f64>().unwrap()).collect();
            fare.push(split[0]);
            survived.push(split[1]);
        }
        let w = vec![1.0; fare.len()];
        let f = Feature::new(&fare, &survived, &w, &Vec::new()).unwrap();
        let mut n = Node::new(
            &f,
            Some(1.0),
            None,
            Some(0.0),
            Some(1),
            None,
            None,
            Some(4),
            Some(30),
        );
        println!("{:?}", n.find_best_split(&f));
        let test_info = SplitInfo {
            split: Some(6.4375),
            lhs_iv: Some(f64::INFINITY),
            lhs_woe: Some(-f64::INFINITY),
            rhs_iv: Some(0.08392941911181274),
            rhs_woe: Some(-1.016803450354609),
        };
        assert_eq!(n.find_best_split(&f), test_info);
    }
}
