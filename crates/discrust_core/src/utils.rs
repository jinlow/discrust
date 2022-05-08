use num::Float;
use std::cmp::Ordering;

pub fn nan_safe_compare<T: Float>(i: &T, j: &T) -> Ordering {
    match i.partial_cmp(j) {
        Some(o) => o,
        None => match (i.is_nan(), j.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => Ordering::Equal,
        },
    }
}

/// Take a sorted array, and find the position
/// of the first value that is less than some target
/// value.
#[allow(dead_code)]
pub fn first_greater_than<T: std::cmp::PartialOrd>(x: &[T], v: &T) -> usize {
    let mut low = 0;
    let mut high = x.len();
    while low != high {
        let mid = (low + high) / 2;
        if x[mid] <= *v {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
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
    #[test]
    fn test_first_greater_than() {
        let v = vec![0, 2, 2, 3, 4, 6, 7];
        assert_eq!(1, first_greater_than(&v, &0));
        assert_eq!(1, first_greater_than(&v, &1));
        assert_eq!(3, first_greater_than(&v, &2));
        assert_eq!(5, first_greater_than(&v, &5));
        let i = (&v).iter().position(|&v| v > 2).unwrap();
        assert_eq!(3, i);
        let i = (&v).iter().position(|&v| v > 0).unwrap();
        assert_eq!(1, i);
        let i = (&v).iter().position(|&v| v > 5).unwrap();
        assert_eq!(5, i);
    }
}
