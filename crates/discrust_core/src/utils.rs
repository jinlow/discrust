use num::Float;
use std::cmp::Ordering;

pub fn nan_safe_compare<T: Float>(i: &T, j: &T) -> Ordering {
    return match (i.is_nan(), j.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => i.partial_cmp(j).unwrap(),
    };
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_compare_in_sort() {
        let mut v = vec![0.0, 100.0, 1.1, f64::NAN, 2.2, f64::NAN];
        v.sort_by(|i, j| nan_safe_compare(i, j));
        // Check the first two values are NaN
        assert!(v[0].is_nan());
        assert!(v[1].is_nan());
        // Then confirm everything else is sorted.
        assert_eq!(v[2..], vec![0.0, 1.1, 2.2, 100.0])
    }
}
