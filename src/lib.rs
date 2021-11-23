use discrust_core::Discretizer as CrateDiscretizer;
use ndarray::Ix1;
use numpy::{IntoPyArray, PyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Python;
use std::collections::HashMap;

// We need to pass subclass here, so that we
// can inherit from this class later.
#[pyclass(subclass)]
struct Discretizer {
    disc: CrateDiscretizer,
    pub splits_: Vec<f64>,
}

#[pymethods]
impl Discretizer {
    #[new]
    fn new(
        min_obs: Option<f64>,
        max_bins: Option<i64>,
        min_iv: Option<f64>,
        min_pos: Option<f64>,
        mono: Option<i8>,
    ) -> Self {
        Discretizer {
            disc: CrateDiscretizer::new(min_obs, max_bins, min_iv, min_pos, mono),
            splits_: Vec::new(),
        }
    }

    #[getter]
    pub fn exception_values_(&self) -> PyResult<HashMap<String, Vec<f64>>> {
        Ok(self
            .disc
            .feature
            .as_ref()
            .unwrap()
            .exception_values
            .to_hashmap())
    }

    #[getter]
    pub fn get_splits_(&self) -> PyResult<Vec<f64>> {
        Ok(self.splits_.to_vec())
    }

    #[setter]
    pub fn set_mono(&mut self, value: Option<i8>) -> PyResult<()> {
        self.disc.mono = value;
        Ok(())
    }

    pub fn fit(
        &mut self,
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
        w: Option<PyReadonlyArray1<f64>>,
        exception_values: Option<Vec<f64>>,
    ) -> PyResult<Vec<f64>> {
        let x = x.as_slice()?;
        let y = y.as_slice()?;
        let w_ = match w {
            Some(v) => v.to_vec(),
            // If a weight is provided this means we are always coping it...
            // Probably should change this so that it's accepting a value,
            // and just init with np.ones
            None => {
                let v = vec![1.0; y.len()];
                Ok(v)
            }
        }?;
        let splits = self.disc.fit(x, y, &w_, exception_values);
        self.splits_ = splits;
        // self.exception_values = Some(self.disc.feature.as_ref().unwrap().exception_values);
        Ok(self.splits_.to_vec())
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<f64>,
    ) -> PyResult<&'py PyArray<f64, Ix1>> {
        let x = x.as_slice()?;
        // I didn't want the underlying discrust_core crate to depend on
        // pyO3, so have to deal with the error custom here.
        match self.disc.predict(x) {
            Ok(v) => {
                let arr = v.into_pyarray(py);
                return Ok(arr);
            }
            Err(e) => return Err(PyValueError::new_err(e.to_string())),
        };
    }
}

#[pymodule]
fn discrust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Discretizer>()?;
    Ok(())
}
