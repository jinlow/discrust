use crate::{utils::nan_safe_compare, DiscrustError};
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
    ones_ct_: Vec<f64>,
    totals_ct_: Vec<f64>,
    ones_dist_: Vec<f64>,
    zero_dist_: Vec<f64>,
    pub exception_values: ExceptionValues,
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
            .position(|x| match nan_safe_compare(x, v) {
                Ordering::Equal => true,
                _ => false,
            })
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
        let mut exception_values = ExceptionValues::new(exception_values);

        // Define all of the stats we will use
        let mut vals_ = Vec::new();
        let mut ones_ct_ = Vec::new();
        let mut totals_ct_ = Vec::new();
        let mut ones_dist_ = Vec::new();
        let mut zero_dist_ = Vec::new();
        // First we will get the index needed to sort the vector x.
        let mut sort_tuples: Vec<(usize, &f64)> = x.iter().enumerate().collect();
        // Now sort these tuples by the float values of x.
        sort_tuples.sort_by(|a, b| nan_safe_compare(a.1, b.1));
        // Now that we have the tuples sorted, we only need the index, we will loop over
        // the data again here, to retrieve the sort index.
        // Maybe if we need to speed things up again, we can consider skipping this part and
        // and only use the the tuples? Here we just set it to the iterator, so it's
        // not actually consumed.
        let mut sort_index = sort_tuples.iter().map(|(i, _)| *i); // .collect();

        // Now loop over all columns, collecting aggregate statistics.
        let mut zero_ct: Vec<f64> = Vec::new();

        // Do this part in one pass for both.
        let mut total_ones = 0.0;
        let mut total_zero = 0.0;
        for (w_, y_) in w.iter().zip(y) {
            // Confirm there are no NaN values in the y_ variable, or
            // or the weight field.
            if y_.is_nan() {
                return Err(DiscrustError::ContainsNaN(String::from("y column")));
            }
            if w_.is_nan() {
                return Err(DiscrustError::ContainsNaN(String::from("weight column")));
            }

            total_ones += w_ * y_;
            // I am using greater than 0 here, because
            // floats only have partial equality.
            total_zero += w_ * ((y_ < &1.0) as i64 as f64);
        }

        // Now loop over the data collecting all relevant stats.
        // We grab the first item from the iterator, and start the sums
        // of all relevant fields.
        let mut init_idx = sort_index.next().unwrap_or(0);

        // Check if NaN is in the vector, but not an exception.
        // The NaN values will be at the beginning because it's sorted.
        if x[init_idx].is_nan() {
            match exception_values.exception_idx(&x[init_idx]) {
                None => {
                    return Err(DiscrustError::ContainsNaN(String::from(
                        "x column, but NaN is not an exception value",
                    )))
                }
                _ => (),
            }
        }

        // If this first value is an exception, we need to loop until
        // we find a non-exception value. Or consume the data. In which
        // case all values in the data are exception_values.
        // If the first value is not an exception, we just leave the
        // init index alone.
        if let Some(idx) = exception_values.exception_idx(&x[init_idx]) {
            exception_values.update_exception_values(idx, &w[init_idx], &y[init_idx]);
            // Start searching through the vector to find the first non-exception
            // value.
            loop {
                let i_op = sort_index.next();
                if i_op.is_none() {
                    break;
                }
                let i = i_op.unwrap();
                match exception_values.exception_idx(&x[i]) {
                    Some(idx) => {
                        exception_values.update_exception_values(idx, &w[i], &y[i]);
                    }
                    // If we have reached a point in the loop where the value
                    // is no longer an exception, update the init_idx
                    // and then break out of the loop.
                    None => {
                        init_idx = i;
                        break;
                    }
                };
            }
        };

        // TODO: At this point we run the risk of actually having gone through
        // all the values. Because of this, we need to add some check here
        // incase we have totally consumed sort_index.
        let mut x_ = x[init_idx];
        vals_.push(x_);
        totals_ct_.push(w[init_idx]);
        ones_ct_.push(w[init_idx] * y[init_idx]);
        zero_ct.push(w[init_idx] * ((y[init_idx] < 1.0) as i64 as f64));
        let mut totals_idx = 0;

        // This will start at the second element.
        for i in sort_index {
            // If the value is equal to one of our exception_values update the exception_values
            // and continue.
            if let Some(idx) = exception_values.exception_idx(&x[i]) {
                exception_values.update_exception_values(idx, &w[i], &y[i]);
                continue;
            }

            // if the value is greater than x_ we know we are at a new
            // value and can calculate the distributions, as well as increment
            // the totals_idx.
            if x_ < x[i] {
                // We update x_ to the new value.
                x_ = x[i];
                // We calculate the distribution values
                ones_dist_.push(ones_ct_[totals_idx] / total_ones);
                zero_dist_.push(zero_ct[totals_idx] / total_zero);

                // Update the values
                totals_ct_.push(w[i]);
                ones_ct_.push(w[i] * y[i]);
                zero_ct.push(w[i] * ((y[i] < 1.0) as i64 as f64));
                vals_.push(x_);
                totals_idx += 1;
            } else {
                // Otherwise just add this value to our current aggregations
                totals_ct_[totals_idx] += w[i];
                ones_ct_[totals_idx] += w[i] * y[i];
                zero_ct[totals_idx] += w[i] * ((y[i] < 1.0) as i64 as f64);
            }
        }
        // Finally add the very last value to our distribution columns
        ones_dist_.push(ones_ct_[totals_idx] / total_ones);
        zero_dist_.push(zero_ct[totals_idx] / total_zero);

        exception_values.calculate_iv_woe(total_ones, total_zero);

        Ok(Feature {
            vals_,
            ones_ct_,
            totals_ct_,
            ones_dist_,
            zero_dist_,
            exception_values,
        })
    }

    /// Split the feature and calculate information value
    /// and weight of evidence for the records bellow and
    /// above the split.
    pub fn split_iv_woe(
        &self,
        split_value: f64,
        start: usize,
        stop: usize,
    ) -> ((f64, f64), (f64, f64)) {
        // vals_ is in sorted order, so we need to find
        // the first position of the record that is less
        // than our split_value
        // This means the split_idx, will be one after our actual
        // split value, thus the left hand side will include
        // the split value.
        let split_idx = (&self.vals_[start..stop])
            .iter()
            .position(|&v| v > split_value)
            .unwrap_or(stop - start);

        // Accumulate the left hand side.
        let mut lhs_zero_dist = 0.0;
        let mut lhs_ones_dist = 0.0;
        for i in 0..(split_idx) {
            lhs_zero_dist += &self.zero_dist_[start..stop][i];
            lhs_ones_dist += &self.ones_dist_[start..stop][i];
        }
        let lhs_woe = (lhs_ones_dist / lhs_zero_dist).ln();
        let lhs_iv = (lhs_ones_dist - lhs_zero_dist) * lhs_woe;

        // Accumulate the right hand side.
        let mut rhs_zero_dist = 0.0;
        let mut rhs_ones_dist = 0.0;
        for i in (split_idx)..self.vals_[start..stop].len() {
            rhs_zero_dist += &self.zero_dist_[start..stop][i];
            rhs_ones_dist += &self.ones_dist_[start..stop][i];
        }
        let rhs_woe = (rhs_ones_dist / rhs_zero_dist).ln();
        let rhs_iv = (rhs_ones_dist - rhs_zero_dist) * rhs_woe;

        ((lhs_iv, lhs_woe), (rhs_iv, rhs_woe))
    }

    pub fn split_totals_ct_ones_ct(
        &self,
        split_value: f64,
        start: usize,
        stop: usize,
    ) -> ((f64, f64), (f64, f64)) {
        let split_idx = (&self.vals_[start..stop])
            .iter()
            .position(|&v| v > split_value)
            // Double check we should be doing this...
            .unwrap_or(stop - start);
        let mut lhs_ct = 0.0;
        let mut lhs_ones = 0.0;
        for i in 0..(split_idx) {
            lhs_ct += &self.totals_ct_[start..stop][i];
            lhs_ones += &self.ones_ct_[start..stop][i];
        }

        let mut rhs_ct = 0.0;
        let mut rhs_ones = 0.0;
        for i in (split_idx)..self.vals_[start..stop].len() {
            rhs_ct += &self.totals_ct_[start..stop][i];
            rhs_ones += &self.ones_ct_[start..stop][i];
        }
        ((lhs_ct, lhs_ones), (rhs_ct, rhs_ones))
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
        assert_eq!(f.totals_ct_, vec![2.0, 2.0, 4.0]);

        let x_ = vec![2.0, 2.0, 1.0, 1.0];
        let y_ = vec![1.0, 1.0, 1.0, 0.0];
        let w_ = vec![3.0, 3.0, 1.0, 1.0];
        let f = Feature::new(&x_, &y_, &w_, &Vec::new()).unwrap();
        assert_eq!(f.vals_, vec![1.0, 2.0]);
        assert_eq!(f.totals_ct_, vec![2.0, 6.0]);
        assert_eq!(f.ones_ct_, vec![1.0, 6.0]);
        assert_eq!(f.ones_dist_, vec![1.0 / 7.0, 6.0 / 7.0]);
        assert_eq!(f.zero_dist_, vec![1.0 / 1.0, 0.0]);
    }

    #[test]
    fn test_split_iv_woe() {
        let x_ = vec![6.2375, 6.4375, 0.0, 0.0, 4.0125, 5.0, 6.45, 6.4958, 6.4958];
        let y_ = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let w_ = vec![1.0; x_.len()];
        let f = Feature::new(&x_, &y_, &w_, &Vec::new()).unwrap();
        assert_eq!(
            f.split_iv_woe(5.0, 0, f.vals_.len()),
            (
                (0.022314355131420965, -0.2231435513142097),
                (0.018232155679395495, 0.1823215567939548)
            )
        );

        // The same test but on a subset of the data
        assert_eq!(
            f.split_iv_woe(5.0, 1, 5),
            (
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
            f.split_totals_ct_ones_ct(5.0, 0, f.vals_.len()),
            ((4.0, 2.0), (5.0, 3.0))
        );

        // The same test but on a subset of the data
        assert_eq!(
            f.split_totals_ct_ones_ct(5.0, 1, 5),
            ((2.0, 1.0), (2.0, 1.0))
        )
    }
}
