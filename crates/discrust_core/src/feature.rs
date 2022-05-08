use crate::utils::nan_safe_compare;
use crate::DiscrustError;
use std::{cmp::Ordering, collections::HashMap};

/// A Feature struct
/// This struct houses all of the aggregate information
/// for feature, and it's binary performance field.
/// It has functionality to utilize weights, and then
/// to compute information value and weight of evidence
/// for arbitrary ranges of the data.
#[derive(Debug)]
pub struct Feature {
    pub vals_: Vec<f64>,
    cuml_ones_ct_: Vec<f64>,
    cuml_zero_ct_: Vec<f64>,
    cuml_totals_ct_: Vec<f64>,
    total_ones_: f64,
    total_zero_: f64,
    pub exception_values_: ExceptionValues,
}

#[derive(Debug)]
pub struct ExceptionValues {
    pub vals_: Vec<f64>,
    pub ones_ct_: Vec<f64>,
    pub zero_ct_: Vec<f64>,
    pub totals_ct_: Vec<f64>,
    pub iv_: Vec<f64>,
    pub woe_: Vec<f64>,
}

impl ExceptionValues {
    fn new(exception_values: &[f64]) -> Self {
        let mut vals_ = exception_values.to_vec();
        vals_.sort_by(|i, j| nan_safe_compare(i, j));
        vals_.dedup();
        let vals_len = vals_.len();
        ExceptionValues {
            vals_,
            ones_ct_: vec![0.0; vals_len],
            zero_ct_: vec![0.0; vals_len],
            totals_ct_: vec![0.0; vals_len],
            iv_: vec![0.0; vals_len],
            woe_: vec![0.0; vals_len],
        }
    }
    // Get the index of an exception, if it's None
    // we know the exception does not exist.
    // This also works for Missing values.
    // This NA can be passed as a possible exception.
    pub fn exception_idx(&self, v: &f64) -> Option<usize> {
        self.vals_
            .iter()
            .position(|x| matches!(nan_safe_compare(x, v), Ordering::Equal))
    }

    // Add the values to the appropriate location in the exception
    // value vectors.
    fn update_exception_values(&mut self, idx: usize, w: &f64, y: &f64) {
        self.totals_ct_[idx] += w;
        self.ones_ct_[idx] += w * y;
        self.zero_ct_[idx] += w * ((y < &1.0) as i64 as f64);
    }

    fn calculate_iv_woe(&mut self, total_ones: f64, total_zero: f64) {
        for i in 0..self.vals_.len() {
            let ones_dist = self.ones_ct_[i] / total_ones;
            let zero_dist = self.zero_ct_[i] / total_zero;
            let woe = (ones_dist / zero_dist).ln();
            let iv = (ones_dist - zero_dist) * woe;
            self.woe_[i] = woe;
            self.iv_[i] = iv;
        }
    }

    pub fn to_hashmap(&self) -> HashMap<String, Vec<f64>> {
        let mut hmp = HashMap::new();
        // hmp.
        hmp.insert("vals_".to_string(), self.vals_.to_vec());
        hmp.insert("ones_ct_".to_string(), self.ones_ct_.to_vec());
        hmp.insert("zero_ct_".to_string(), self.zero_ct_.to_vec());
        hmp.insert("totals_ct_".to_string(), self.totals_ct_.to_vec());
        hmp.insert("iv_".to_string(), self.iv_.to_vec());
        hmp.insert("woe_".to_string(), self.woe_.to_vec());
        hmp
    }
}

impl Feature {
    /// Generate a new feature from a vector and it's
    /// binary performance. All vectors provided should be
    /// of the same length.
    ///
    /// # Arguments
    ///
    /// * `x` - A reference to a vector that will be used
    ///     discretized.
    /// * `y` - A reference to a vector of 1s (positive class)
    ///     and 0s (negative class).
    /// * `w` - A reference to a vector of weights. If the feature
    ///     should be unweighted, pass in a vector of 1s, `vec![1.0; y.len()]`.
    pub fn new(
        x: &[f64],
        y: &[f64],
        w: &[f64],
        exception_values: &[f64],
    ) -> Result<Self, DiscrustError> {
        // Make exception values.
        let mut exception_values_ = ExceptionValues::new(exception_values);

        // Define all of the stats we will use
        let mut vals_ = Vec::new();
        let mut cuml_ones_ct_ = Vec::new();
        let mut cuml_zero_ct_ = Vec::new();
        let mut cuml_totals_ct_ = Vec::new();
        // First we will get the index needed to sort the vector x.
        let mut sort_tuples: Vec<(usize, &f64)> = x.iter().enumerate().collect();
        let no_exceptions = exception_values.is_empty();
        if no_exceptions {
            // Now sort these tuples by the float values of x.
            sort_tuples.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
        } else {
            sort_tuples.sort_by(|a, b| nan_safe_compare(a.1, b.1));
            
        };
        let sort_index = sort_tuples.iter().map(|(i, _)| *i);

        let mut totals_idx = 0;
        let mut first_value = true;
        let mut x_ = f64::NAN;
        let mut y_;
        let mut w_;
        let mut total_ones_ = 0.0;
        let mut total_zero_ = 0.0;
        for i in sort_index {
            y_ = y[i];
            w_ = w[i];
            // Some error checking
            if y_.is_nan() {
                return Err(DiscrustError::ContainsNaN(String::from("y column")));
            }
            if w_.is_nan() {
                return Err(DiscrustError::ContainsNaN(String::from("weight column")));
            }
            if !no_exceptions {
                let e_idx = exception_values_.exception_idx(&x[i]);
                if x[i].is_nan() && e_idx == None {
                    return Err(DiscrustError::ContainsNaN(String::from(
                        "x column, but NaN is not an exception value",
                    )));
                }
                // If the value is equal to one of our exception_values_ update the exception_values_
                // and continue.
                if let Some(idx) = e_idx {
                    exception_values_.update_exception_values(idx, &w_, &y_);
                    if y_ == 1.0 {
                        total_ones_ += w_;
                    } else {
                        total_zero_ += w_;
                    }
                    continue;
                }
            }
            // If this is the first value, add to our vectors
            // Initializing them.
            if first_value {
                x_ = x[i];
                cuml_totals_ct_.push(w_);
                if y_ == 1.0 {
                    total_ones_ += w_;
                    cuml_ones_ct_.push(w_);
                    cuml_zero_ct_.push(0.0);
                } else {
                    total_zero_ += w_;
                    cuml_ones_ct_.push(0.0);
                    cuml_zero_ct_.push(w_);
                }
                vals_.push(x_);
            // If this is a new value, push, and cumulate
            // this won't panic, because we know these
            // vectors have values
            } else if x_ < x[i] {
                let t_last = cuml_totals_ct_[totals_idx];
                let o_last = cuml_ones_ct_[totals_idx];
                let z_last = cuml_zero_ct_[totals_idx];
                x_ = x[i];
                cuml_totals_ct_.push(w_ + t_last);
                if y_ == 1.0 {
                    total_ones_ += w_;
                    cuml_ones_ct_.push(w_ + o_last);
                    cuml_zero_ct_.push(z_last);
                } else {
                    total_zero_ += w_;
                    cuml_ones_ct_.push(o_last);
                    cuml_zero_ct_.push(w_ + z_last);
                }
                vals_.push(x_);
                totals_idx += 1;
            } else {
                cuml_totals_ct_[totals_idx] += w_;
                if y_ == 1.0 {
                    total_ones_ += w_;
                    cuml_ones_ct_[totals_idx] += w_;
                } else {
                    total_zero_ += w_;
                    cuml_zero_ct_[totals_idx] += w_;
                }
            }
            first_value = false;
        }
        exception_values_.calculate_iv_woe(total_ones_, total_zero_);

        Ok(Feature {
            vals_,
            cuml_ones_ct_,
            cuml_zero_ct_,
            cuml_totals_ct_,
            total_ones_,
            total_zero_,
            exception_values_,
        })
    }

    /// Split the feature and calculate information value
    /// and weight of evidence for the records bellow and
    /// above the split.
    pub fn split_iv_woe(
        &self,
        split_idx: usize,
        start: usize,
        stop: usize,
    ) -> ((f64, f64), (f64, f64)) {
        // vals_ is in sorted order, so we need to find
        // the first position of the record that is less
        // than our split_value
        // This means the split_idx, will be one after our actual
        // split value, thus the left hand side will include
        // the split value.
        let split_idx = split_idx + 1 + start;

        // Accumulate the left hand side.
        let lhs_zero_dist =
            sum_of_cuml_subarray(&self.cuml_zero_ct_, start, split_idx - 1) / self.total_zero_;
        let lhs_ones_dist =
            sum_of_cuml_subarray(&self.cuml_ones_ct_, start, split_idx - 1) / self.total_ones_;
        let lhs_woe = (lhs_ones_dist / lhs_zero_dist).ln();
        let lhs_iv = (lhs_ones_dist - lhs_zero_dist) * lhs_woe;

        // Accumulate the right hand side.
        let rhs_zero_dist =
            sum_of_cuml_subarray(&self.cuml_zero_ct_, split_idx, stop - 1) / self.total_zero_;
        let rhs_ones_dist =
            sum_of_cuml_subarray(&self.cuml_ones_ct_, split_idx, stop - 1) / self.total_ones_;
        let rhs_woe = (rhs_ones_dist / rhs_zero_dist).ln();
        let rhs_iv = (rhs_ones_dist - rhs_zero_dist) * rhs_woe;

        ((lhs_iv, lhs_woe), (rhs_iv, rhs_woe))
    }

    pub fn split_totals_ct_ones_ct(
        &self,
        split_idx: usize,
        start: usize,
        stop: usize,
    ) -> ((f64, f64), (f64, f64)) {
        let split_idx = split_idx + 1 + start;

        let lhs_ct = sum_of_cuml_subarray(&self.cuml_totals_ct_, start, split_idx - 1);
        let lhs_ones = sum_of_cuml_subarray(&self.cuml_ones_ct_, start, split_idx - 1);

        let rhs_ct = sum_of_cuml_subarray(&self.cuml_totals_ct_, split_idx, stop - 1);
        let rhs_ones = sum_of_cuml_subarray(&self.cuml_ones_ct_, split_idx, stop - 1);

        ((lhs_ct, lhs_ones), (rhs_ct, rhs_ones))
    }
}

fn sum_of_cuml_subarray(x: &[f64], start: usize, stop: usize) -> f64 {
    if start == 0 {
        x[stop]
    } else {
        x[stop] - x[start - 1]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_new_feature() {
        let x_ = vec![1.0, 1.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let y_ = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let w_ = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let _f = Feature::new(&x_, &y_, &w_, &Vec::new());
        assert!(true);
    }
    #[test]
    fn test_feature_fit() {
        let x_ = vec![1.0, 1.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let y_ = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let w_ = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let f = Feature::new(&x_, &y_, &w_, &Vec::new()).unwrap();
        assert_eq!(f.vals_, vec![1.0, 2.0, 3.0]);
        assert_eq!(f.cuml_totals_ct_, vec![2.0, 4.0, 8.0]);

        let x_ = vec![2.0, 2.0, 1.0, 1.0];
        let y_ = vec![1.0, 1.0, 1.0, 0.0];
        let w_ = vec![3.0, 3.0, 1.0, 1.0];
        let f = Feature::new(&x_, &y_, &w_, &Vec::new()).unwrap();
        assert_eq!(f.vals_, vec![1.0, 2.0]);
        assert_eq!(f.cuml_totals_ct_, vec![2.0, 8.0]);
        assert_eq!(f.cuml_ones_ct_, vec![1.0, 7.0]);
        // assert_eq!(
        //     f.cuml_ones_dist_,
        //     vec![1.0 / 7.0, (1.0 / 7.0) + (6.0 / 7.0)]
        // );
        // assert_eq!(f.cuml_zero_dist_, vec![1.0 / 1.0, 1.0]);
    }

    #[test]
    fn test_split_iv_woe() {
        let x_ = vec![6.2375, 6.4375, 0.0, 0.0, 4.0125, 5.0, 6.45, 6.4958, 6.4958];
        let y_ = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let w_ = vec![1.0; x_.len()];
        let f = Feature::new(&x_, &y_, &w_, &Vec::new()).unwrap();
        assert_eq!(
            // 0, 4, 5, (Split on 5.0)
            f.split_iv_woe(2, 0, f.vals_.len()),
            (
                // (0.022314355131420965, -0.2231435513142097),
                // (0.018232155679395495, 0.1823215567939548)
                (0.022314355131420965, -0.2231435513142097),
                (0.018232155679395456, 0.1823215567939546)
            )
        );

        // The same test but on a subset of the data
        assert_eq!(
            f.split_iv_woe(1, 1, 5),
            (
                // (0.011157177565710483, -0.2231435513142097),
                // (0.011157177565710483, -0.2231435513142097)
                (0.011157177565710483, -0.2231435513142097),
                (0.011157177565710483, -0.2231435513142097)
            )
        )
    }
    #[test]
    fn test_split_totals_ct_ones_ct() {
        let x_ = vec![6.2375, 6.4375, 0.0, 0.0, 4.0125, 5.0, 6.45, 6.4958, 6.4958];
        let y_ = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let w_ = vec![1.0; x_.len()];
        let f = Feature::new(&x_, &y_, &w_, &Vec::new()).unwrap();
        assert_eq!(
            f.split_totals_ct_ones_ct(2, 0, f.vals_.len()),
            ((4.0, 2.0), (5.0, 3.0))
        );

        // The same test but on a subset of the data
        assert_eq!(f.split_totals_ct_ones_ct(1, 1, 5), ((2.0, 1.0), (2.0, 1.0)))
    }
    #[test]
    fn test_accumulate() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let cuml_v: Vec<f64> = v
            .iter()
            .scan(0.0, |acc, &x| {
                *acc = *acc + x;
                Some(*acc)
            })
            .collect();
        let mut cuml_rl_v: Vec<f64> = v
            .iter()
            .rev()
            .scan(0.0, |acc, &x| {
                *acc = *acc + x;
                Some(*acc)
            })
            .collect();
        cuml_rl_v.reverse();

        assert_eq!(cuml_v, vec![1.0, 3.0, 6.0, 10.0]);
        assert_eq!(cuml_rl_v, vec![10.0, 9.0, 7.0, 4.0]);
    }
}
