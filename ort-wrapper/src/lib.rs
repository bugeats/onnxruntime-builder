//! ORT Wrapper - ONNX Runtime Rust bindings with static linking
//!
//! Provides utilities for running ONNX models with various execution providers.
//!
//! # Feature Flags
//!
//! Accelerators are selected at compile time via Cargo features:
//! - `cuda` - Enable CUDA execution provider (Linux only)
//! - `coreml` - Enable CoreML execution provider (macOS only)
//! - (no features) - CPU-only execution
//!
//! # Static Linking
//!
//! This crate uses the `alternative-backend` feature of `ort`, which means we must
//! initialize the ONNX Runtime API manually via `init_ort()` before using any ort APIs.

use anyhow::{Context, Result};
use ort::{
    execution_providers::ExecutionProviderDispatch,
    session::{Session, SessionInputValue},
};
use std::path::Path;
use std::sync::Once;
use tracing::{debug, info, instrument, trace, warn};

// Import ExecutionProvider trait to access is_available() method
#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(feature = "coreml")]
use ort::execution_providers::CoreMLExecutionProvider;

// The ExecutionProvider trait is needed to call is_available()
#[cfg(any(feature = "cuda", feature = "coreml"))]
use ort::execution_providers::ExecutionProvider;

// =============================================================================
// Static ONNX Runtime API Initialization
// =============================================================================

// Declare the OrtGetApiBase function from the statically linked ONNX Runtime
extern "C" {
    fn OrtGetApiBase() -> *const ort::sys::OrtApiBase;
}

static ORT_INIT: Once = Once::new();

/// Initialize the ONNX Runtime API for static linking.
///
/// This must be called before using any `ort` APIs. It's safe to call multiple
/// times - subsequent calls are no-ops.
///
/// # Panics
///
/// Panics if the ONNX Runtime API cannot be initialized (e.g., version mismatch).
#[instrument(level = "info")]
pub fn init_ort() {
    ORT_INIT.call_once(|| {
        info!("Initializing ONNX Runtime API (first call)");
        unsafe {
            trace!("Calling OrtGetApiBase()");
            let api_base = OrtGetApiBase();
            if api_base.is_null() {
                panic!("OrtGetApiBase returned null");
            }
            trace!("OrtGetApiBase returned non-null pointer");

            trace!("Calling GetApi with version {}", ort::sys::ORT_API_VERSION);
            let get_api = (*api_base).GetApi;
            let api_ptr = get_api(ort::sys::ORT_API_VERSION);
            if api_ptr.is_null() {
                panic!(
                    "OrtApiBase::GetApi({}) returned null - version mismatch?",
                    ort::sys::ORT_API_VERSION
                );
            }
            trace!("GetApi returned non-null pointer");

            let api = *api_ptr;
            trace!("Setting API in ort crate");
            if !ort::set_api(api) {
                debug!("API was already set (race condition, harmless)");
            }
            info!("ONNX Runtime API initialized successfully");
        }
    });
}

/// Information about available execution providers
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    pub cpu_available: bool,
    pub cuda_available: bool,
    pub coreml_available: bool,
}

impl ProviderInfo {
    /// Check which execution providers are available
    ///
    /// Uses compile-time feature detection:
    /// - `cuda` feature: checks CUDA availability
    /// - `coreml` feature: checks CoreML availability
    #[instrument(level = "debug")]
    pub fn detect() -> Self {
        debug!("Detecting available execution providers");
        let cpu_available = true; // CPU is always available
        trace!("CPU provider: available (always)");

        // CUDA availability (when cuda feature is enabled)
        #[cfg(feature = "cuda")]
        let cuda_available = {
            trace!("Checking CUDA availability (feature enabled)");
            let available = CUDAExecutionProvider::default()
                .is_available()
                .unwrap_or(false);
            debug!(cuda_available = available, "CUDA detection complete");
            available
        };
        #[cfg(not(feature = "cuda"))]
        let cuda_available = {
            trace!("CUDA feature not enabled at compile time");
            false
        };

        // CoreML availability (when coreml feature is enabled)
        #[cfg(feature = "coreml")]
        let coreml_available = {
            trace!("Checking CoreML availability (feature enabled)");
            let available = CoreMLExecutionProvider::default()
                .is_available()
                .unwrap_or(false);
            debug!(coreml_available = available, "CoreML detection complete");
            available
        };
        #[cfg(not(feature = "coreml"))]
        let coreml_available = {
            trace!("CoreML feature not enabled at compile time");
            false
        };

        let info = ProviderInfo {
            cpu_available,
            cuda_available,
            coreml_available,
        };
        info!(
            cpu = info.cpu_available,
            cuda = info.cuda_available,
            coreml = info.coreml_available,
            "Provider detection complete"
        );
        info
    }

    /// Returns true if any GPU accelerator is available
    pub fn has_gpu(&self) -> bool {
        self.cuda_available || self.coreml_available
    }

    /// Get the name of the compiled-in accelerator (if any)
    pub fn accelerator_name() -> &'static str {
        #[cfg(feature = "cuda")]
        return "CUDA";
        #[cfg(feature = "coreml")]
        return "CoreML";
        #[cfg(not(any(feature = "cuda", feature = "coreml")))]
        return "None (CPU only)";
    }

    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        let mut providers = vec![];
        if self.cpu_available {
            providers.push("CPU");
        }
        if self.cuda_available {
            providers.push("CUDA");
        }
        if self.coreml_available {
            providers.push("CoreML");
        }
        format!("Available providers: {}", providers.join(", "))
    }
}

/// Execution provider preference for session creation
#[derive(Debug, Clone, Copy, Default)]
pub enum ProviderPreference {
    /// Use only CPU provider
    CpuOnly,
    /// Prefer GPU (CUDA on Linux, CoreML on macOS) with CPU fallback
    #[default]
    PreferGpu,
    /// Require the platform-specific GPU accelerator (fail if unavailable)
    RequireGpu,
}

/// Create an ONNX inference session with the specified provider preference
#[instrument(level = "info", skip(model_path), fields(model = ?model_path))]
pub fn create_session(model_path: &Path, preference: ProviderPreference) -> Result<Session> {
    info!(?preference, "Creating inference session");

    let info = ProviderInfo::detect();

    // Build list of execution providers based on preference
    debug!("Building execution provider list");
    let providers: Vec<ExecutionProviderDispatch> = match preference {
        ProviderPreference::CpuOnly => {
            debug!("Using CPU-only mode");
            vec![] // Empty = CPU only
        }
        ProviderPreference::PreferGpu => {
            debug!("Preferring GPU with CPU fallback");
            build_gpu_providers(&info)
        }
        ProviderPreference::RequireGpu => {
            if !info.has_gpu() {
                warn!("GPU required but not available");
                anyhow::bail!(
                    "{} execution provider required but not available",
                    ProviderInfo::accelerator_name()
                );
            }
            debug!("GPU required and available");
            build_gpu_providers(&info)
        }
    };

    debug!(provider_count = providers.len(), "Building session");
    let mut builder = Session::builder()?;

    for (i, ep) in providers.iter().enumerate() {
        trace!(index = i, "Adding execution provider");
        builder = builder.with_execution_providers([ep.clone()])?;
    }

    info!("Loading model from file");
    let session = builder
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load model from {:?}", model_path))?;

    info!(
        input_count = session.inputs().len(),
        output_count = session.outputs().len(),
        "Session created successfully"
    );

    Ok(session)
}

/// Build GPU execution providers based on compiled features and availability
fn build_gpu_providers(info: &ProviderInfo) -> Vec<ExecutionProviderDispatch> {
    #[allow(unused_mut)] // eps is only mutated when accelerator features are enabled
    let mut eps = Vec::new();

    // CUDA (when cuda feature is enabled)
    #[cfg(feature = "cuda")]
    if info.cuda_available {
        eps.push(CUDAExecutionProvider::default().build().into());
    }

    // CoreML (when coreml feature is enabled)
    #[cfg(feature = "coreml")]
    if info.coreml_available {
        eps.push(CoreMLExecutionProvider::default().build().into());
    }

    // Suppress unused variable warning when no accelerator features are enabled
    let _ = info;

    eps
}

/// Output from running inference, including raw tensor data for validation
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    /// Shape of each output tensor
    pub shapes: Vec<Vec<usize>>,
    /// Flattened data from each output tensor (f32 assumed)
    pub data: Vec<Vec<f32>>,
    /// Names of output tensors
    pub names: Vec<String>,
}

impl InferenceOutput {
    /// Validate that output tensors contain reasonable values
    #[instrument(level = "debug", skip(self))]
    pub fn validate(&self) -> Result<ValidationReport> {
        debug!("Validating {} output tensors", self.data.len());
        let mut report = ValidationReport::default();

        for (i, (data, name)) in self.data.iter().zip(&self.names).enumerate() {
            trace!(output_index = i, name = %name, len = data.len(), "Validating tensor");

            // Check for empty tensors
            if data.is_empty() {
                warn!(name = %name, "Empty tensor");
                report.warnings.push(format!("Output '{}' is empty", name));
                continue;
            }

            // Check for non-finite values
            let non_finite_count = data.iter().filter(|v| !v.is_finite()).count();
            if non_finite_count > 0 {
                warn!(
                    name = %name,
                    non_finite_count,
                    total = data.len(),
                    "Tensor contains non-finite values"
                );
                report.errors.push(format!(
                    "Output '{}' has {} non-finite values out of {}",
                    name,
                    non_finite_count,
                    data.len()
                ));
            }

            // Compute statistics
            let finite_values: Vec<f32> = data.iter().copied().filter(|v| v.is_finite()).collect();
            if finite_values.is_empty() {
                report.errors.push(format!("Output '{}' has no finite values", name));
                continue;
            }

            let min = finite_values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = finite_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = finite_values.iter().sum();
            let mean = sum / finite_values.len() as f32;

            // Variance calculation
            let variance: f32 = finite_values
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f32>()
                / finite_values.len() as f32;
            let std_dev = variance.sqrt();

            let stats = TensorStats {
                name: name.clone(),
                shape: self.shapes[i].clone(),
                min,
                max,
                mean,
                std_dev,
                zero_count: finite_values.iter().filter(|&&v| v == 0.0).count(),
                total_elements: data.len(),
            };

            debug!(
                name = %name,
                min = stats.min,
                max = stats.max,
                mean = stats.mean,
                std_dev = stats.std_dev,
                "Tensor statistics"
            );

            // Warn on suspicious patterns
            if stats.std_dev < 1e-10 {
                warn!(name = %name, "Tensor has near-zero variance (all same value?)");
                report.warnings.push(format!(
                    "Output '{}' has near-zero variance (all values ≈ {:.6})",
                    name, mean
                ));
            }

            if stats.zero_count == stats.total_elements {
                warn!(name = %name, "Tensor is all zeros");
                report.warnings.push(format!("Output '{}' is all zeros", name));
            }

            report.tensor_stats.push(stats);
        }

        report.is_valid = report.errors.is_empty();
        info!(
            is_valid = report.is_valid,
            error_count = report.errors.len(),
            warning_count = report.warnings.len(),
            "Validation complete"
        );
        Ok(report)
    }

    /// For classification models: apply softmax and return top-k predictions
    #[instrument(level = "debug", skip(self))]
    pub fn top_k_predictions(&self, k: usize) -> Vec<Vec<(usize, f32)>> {
        debug!(k, outputs = self.data.len(), "Computing top-k predictions");

        self.data
            .iter()
            .map(|logits| {
                let probabilities = softmax(logits);

                // Get top-k indices
                let mut indexed: Vec<(usize, f32)> =
                    probabilities.into_iter().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(k);

                trace!(
                    top_class = indexed.first().map(|(i, _)| *i),
                    top_prob = indexed.first().map(|(_, p)| *p),
                    "Top prediction"
                );
                indexed
            })
            .collect()
    }

    /// Check that softmax probabilities sum to approximately 1.0
    #[instrument(level = "debug", skip(self))]
    pub fn verify_softmax_sum(&self, tolerance: f32) -> Vec<(String, f32, bool)> {
        debug!(tolerance, "Verifying softmax sums");

        self.data
            .iter()
            .zip(&self.names)
            .map(|(logits, name)| {
                let probabilities = softmax(logits);
                let prob_sum: f32 = probabilities.iter().sum();
                let is_valid = (prob_sum - 1.0).abs() < tolerance;

                if !is_valid {
                    warn!(
                        name = %name,
                        sum = prob_sum,
                        expected = 1.0,
                        tolerance,
                        "Softmax sum outside tolerance"
                    );
                } else {
                    trace!(name = %name, sum = prob_sum, "Softmax sum valid");
                }

                (name.clone(), prob_sum, is_valid)
            })
            .collect()
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute softmax probabilities from logits (numerically stable)
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = logits.iter().map(|v| (v - max_logit).exp()).collect();
    let sum_exp: f32 = exp_values.iter().sum();
    exp_values.iter().map(|v| v / sum_exp).collect()
}

/// Pre-computed input data for inference
///
/// Useful for benchmarks where you want to create input data once and reuse it.
#[derive(Debug, Clone)]
pub struct PreparedInput {
    /// Shape of the input tensor (dynamic dims replaced with 1)
    pub shape: Vec<usize>,
    /// Flattened input data (deterministic pseudo-random values in [0, 1))
    pub data: Vec<f32>,
}

impl PreparedInput {
    /// Create prepared input from session input info at given index
    ///
    /// Replaces dynamic dimensions (-1) with 1.
    /// Generates deterministic pseudo-random data in [0, 1).
    pub fn from_session(session: &Session, input_index: usize) -> Self {
        let input = &session.inputs()[input_index];
        let shape_ref = input
            .dtype()
            .tensor_shape()
            .expect("Input should be a tensor");

        // Replace dynamic dims (-1) with 1
        let shape: Vec<usize> = shape_ref
            .iter()
            .map(|&d| if d < 0 { 1 } else { d as usize })
            .collect();

        let total_elements: usize = shape.iter().product();
        let data: Vec<f32> = (0..total_elements)
            .map(|i| (i as f32 * 0.01) % 1.0)
            .collect();

        Self { shape, data }
    }

    /// Create prepared input with a specific batch size (modifies first dimension)
    pub fn with_batch_size(&self, batch_size: usize) -> Self {
        let mut shape = self.shape.clone();
        if !shape.is_empty() {
            shape[0] = batch_size;
        }

        let total_elements: usize = shape.iter().product();
        let data: Vec<f32> = (0..total_elements)
            .map(|i| (i as f32 * 0.01) % 1.0)
            .collect();

        Self { shape, data }
    }

    /// Create an ORT tensor from this prepared input
    pub fn to_tensor(&self) -> ort::value::Tensor<f32> {
        ort::value::Tensor::from_array((self.shape.clone(), self.data.clone().into_boxed_slice()))
            .expect("Failed to create tensor")
    }
}

/// Statistics about a single output tensor
#[derive(Debug, Clone)]
pub struct TensorStats {
    pub name: String,
    pub shape: Vec<usize>,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std_dev: f32,
    pub zero_count: usize,
    pub total_elements: usize,
}

/// Report from validating inference output
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub tensor_stats: Vec<TensorStats>,
}

impl std::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Validation Report: {}", if self.is_valid { "PASS" } else { "FAIL" })?;

        if !self.errors.is_empty() {
            writeln!(f, "  Errors:")?;
            for err in &self.errors {
                writeln!(f, "    - {}", err)?;
            }
        }

        if !self.warnings.is_empty() {
            writeln!(f, "  Warnings:")?;
            for warn in &self.warnings {
                writeln!(f, "    - {}", warn)?;
            }
        }

        writeln!(f, "  Tensor Statistics:")?;
        for stats in &self.tensor_stats {
            writeln!(
                f,
                "    {}: shape={:?}, min={:.6}, max={:.6}, mean={:.6}, std={:.6}",
                stats.name, stats.shape, stats.min, stats.max, stats.mean, stats.std_dev
            )?;
        }

        Ok(())
    }
}

/// Run inference on a session with random input data
///
/// This is useful for testing that a model can actually execute.
/// Returns detailed output including shapes and raw data for validation.
/// Note: In ort v2, running inference requires mutable access to the session.
#[instrument(level = "info", skip(session))]
pub fn run_random_inference(session: &mut Session) -> Result<InferenceOutput> {
    info!("Preparing inference inputs");

    // Prepare all inputs using the PreparedInput helper
    let input_tensors: Vec<ort::value::Tensor<f32>> = (0..session.inputs().len())
        .map(|idx| {
            let name = session.inputs()[idx].name();
            trace!(index = idx, name = %name, "Processing input");

            let prepared = PreparedInput::from_session(session, idx);
            debug!(
                name = %name,
                shape = ?prepared.shape,
                elements = prepared.data.len(),
                "Input tensor shape"
            );

            trace!(name = %name, "Creating tensor");
            prepared.to_tensor()
        })
        .collect();

    // Convert tensors to SessionInputValue - ort v2 requires SessionInputValue for run()
    let input_values: Vec<SessionInputValue<'_>> = input_tensors
        .iter()
        .map(SessionInputValue::from)
        .collect();

    info!(input_count = input_values.len(), "Running inference");
    let start = std::time::Instant::now();

    // Session::run accepts &[SessionInputValue] via Into<SessionInputs>
    let outputs = session.run(input_values.as_slice())?;

    let elapsed = start.elapsed();
    info!(elapsed_ms = elapsed.as_millis(), "Inference complete");

    // Extract output data
    debug!("Extracting output tensors");
    let mut shapes = Vec::new();
    let mut data = Vec::new();
    let mut names = Vec::new();

    for (name, value) in outputs.iter() {
        trace!(name = %name, "Processing output");

        match value.try_extract_array::<f32>() {
            Ok(arr) => {
                let shape: Vec<usize> = arr.shape().to_vec();
                let flat_data: Vec<f32> = arr.iter().copied().collect();

                debug!(
                    name = %name,
                    shape = ?shape,
                    elements = flat_data.len(),
                    "Output tensor extracted"
                );

                shapes.push(shape);
                data.push(flat_data);
                names.push(name.to_string());
            }
            Err(e) => {
                warn!(name = %name, error = %e, "Failed to extract output as f32");
            }
        }
    }

    Ok(InferenceOutput { shapes, data, names })
}

/// Legacy function for backward compatibility - returns just shapes
#[instrument(level = "info", skip(session))]
pub fn run_random_inference_shapes(session: &mut Session) -> Result<Vec<Vec<i64>>> {
    let output = run_random_inference(session)?;
    Ok(output
        .shapes
        .into_iter()
        .map(|s| s.into_iter().map(|d| d as i64).collect())
        .collect())
}

/// Get the path to a test model
///
/// Looks for ONNX_TEST_MODEL environment variable first,
/// then falls back to a default location.
pub fn get_test_model_path() -> Result<std::path::PathBuf> {
    if let Ok(path) = std::env::var("ONNX_TEST_MODEL") {
        let path = std::path::PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
        anyhow::bail!("ONNX_TEST_MODEL path does not exist: {:?}", path);
    }

    // Check common locations
    let candidates = [
        "squeezenet1.0-7.onnx",
        "test-model.onnx",
        "../squeezenet1.0-7.onnx",
    ];

    for candidate in candidates {
        let path = std::path::PathBuf::from(candidate);
        if path.exists() {
            return Ok(path);
        }
    }

    anyhow::bail!(
        "No test model found. Set ONNX_TEST_MODEL environment variable or place a model in the current directory."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::info;

    #[test_log::test]
    fn test_provider_detection() {
        init_ort();
        let info = ProviderInfo::detect();
        assert!(info.cpu_available, "CPU should always be available");
        info!(
            accelerator = ProviderInfo::accelerator_name(),
            summary = %info.summary(),
            "Provider detection"
        );
    }
}
