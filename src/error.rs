use std::fmt;

/// Errors produced by witness-topology operations.
#[derive(Debug)]
pub enum TopologyError {
    /// No data points provided.
    EmptyData,
    /// Requested more landmarks than data points.
    InsufficientData { have: usize, need: usize },
    /// Dimension mismatch between points.
    DimensionMismatch { expected: usize, got: usize },
    /// Invalid parameter.
    InvalidParameter(String),
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TopologyError::EmptyData => write!(f, "no data points provided"),
            TopologyError::InsufficientData { have, need } => {
                write!(f, "insufficient data: have {have} points, need {need}")
            }
            TopologyError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            TopologyError::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl std::error::Error for TopologyError {}
