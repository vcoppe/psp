//! This module defines an abstract representation of a PSP instance.

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PspInstance {
    pub nb_types: usize,
    pub nb_periods: usize,
    pub stocking: Vec<usize>,
    pub changeover: Vec<Vec<usize>>,
    pub demands: Vec<Vec<usize>>,
}
