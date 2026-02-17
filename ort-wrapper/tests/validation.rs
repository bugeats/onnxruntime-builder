//! Deep validation and stability tests
//!
//! Provider behavior, output analysis, determinism, and performance stability.
//! These tests verify inference correctness rather than demonstrating the API.

use ort_wrapper::{
    create_session, get_test_model_path, init, run_random_inference, ProviderInfo,
    ProviderPreference,
};
use std::time::Instant;
use test_log::test;
use tracing::{debug, info, info_span, warn};

fn setup() {
    init();
}

macro_rules! skip_test {
    ($reason:expr) => {{
        warn!(reason = $reason, "SKIPPING TEST");
        return;
    }};
}

macro_rules! require_test_model {
    () => {
        match get_test_model_path() {
            Ok(path) => {
                info!(model = ?path, "Using test model");
                path
            }
            Err(e) => {
                skip_test!(format!("No test model available: {}", e));
            }
        }
    };
}

/// Per-tensor stats: finiteness, min <= max, element counts
#[test]
fn test_output_validation() {
    let _span = info_span!("test_output_validation").entered();
    setup();

    let model_path = require_test_model!();

    let mut session =
        create_session(&model_path, ProviderPreference::CpuOnly).expect("Should create session");

    let output = run_random_inference(&mut session).expect("Inference should succeed");
    let report = output.validate().expect("Validation should not error");

    assert_eq!(
        report.tensor_stats.len(),
        output.data.len(),
        "Should have stats for all {} outputs, got {}",
        output.data.len(),
        report.tensor_stats.len()
    );

    for stats in &report.tensor_stats {
        info!(
            name = %stats.name,
            shape = ?stats.shape,
            min = stats.min,
            max = stats.max,
            mean = stats.mean,
            std_dev = stats.std_dev,
            zeros = stats.zero_count,
            total = stats.total_elements,
            "Tensor statistics"
        );

        assert!(
            stats.min.is_finite(),
            "Tensor '{}' min should be finite, got {}",
            stats.name,
            stats.min
        );
        assert!(
            stats.max.is_finite(),
            "Tensor '{}' max should be finite, got {}",
            stats.name,
            stats.max
        );
        assert!(
            stats.mean.is_finite(),
            "Tensor '{}' mean should be finite, got {}",
            stats.name,
            stats.mean
        );
        assert!(
            stats.min <= stats.max,
            "Tensor '{}' min ({}) should be <= max ({})",
            stats.name,
            stats.min,
            stats.max
        );
        assert!(
            stats.total_elements > 0,
            "Tensor '{}' should have elements",
            stats.name
        );
    }
}

/// Zero-density analysis and variance check
#[test]
fn test_output_not_trivial() {
    let _span = info_span!("test_output_not_trivial").entered();
    setup();

    let model_path = require_test_model!();

    let mut session =
        create_session(&model_path, ProviderPreference::CpuOnly).expect("Should create session");

    let output = run_random_inference(&mut session).expect("Inference should succeed");
    let report = output.validate().expect("Validation should not error");

    let has_variance = report.tensor_stats.iter().any(|s| s.std_dev > 1e-10);
    assert!(
        has_variance,
        "At least one output should have variance (not all same value). Stats: {:?}",
        report
            .tensor_stats
            .iter()
            .map(|s| (&s.name, s.std_dev))
            .collect::<Vec<_>>()
    );

    for stats in &report.tensor_stats {
        let zero_ratio = stats.zero_count as f64 / stats.total_elements as f64;

        assert!(
            stats.zero_count < stats.total_elements,
            "Output '{}' should not be ALL zeros ({}/{} are zero)",
            stats.name,
            stats.zero_count,
            stats.total_elements
        );

        if zero_ratio > 0.9 {
            warn!(
                name = %stats.name,
                zero_ratio = format!("{:.2}%", zero_ratio * 100.0),
                "Output is mostly zeros - may indicate inference issue"
            );
        }
    }
}

/// Bitwise consistency across 3 runs (same deterministic input)
#[test]
fn test_inference_determinism() {
    let _span = info_span!("test_inference_determinism").entered();
    setup();

    let model_path = require_test_model!();

    let mut session =
        create_session(&model_path, ProviderPreference::CpuOnly).expect("Should create session");

    const NUM_RUNS: usize = 3;
    let mut outputs = Vec::with_capacity(NUM_RUNS);

    for i in 0..NUM_RUNS {
        let output = run_random_inference(&mut session)
            .unwrap_or_else(|e| panic!("Inference run {} should succeed: {}", i + 1, e));
        outputs.push(output);
    }

    let baseline = &outputs[0];
    for (i, output) in outputs.iter().enumerate().skip(1) {
        assert_eq!(
            baseline.shapes, output.shapes,
            "Run {} shapes should match run 1",
            i + 1
        );
        assert_eq!(
            baseline.data.len(),
            output.data.len(),
            "Run {} data count should match run 1",
            i + 1
        );

        for (tensor_idx, (d1, d2)) in baseline.data.iter().zip(&output.data).enumerate() {
            assert_eq!(
                d1.len(),
                d2.len(),
                "Run {} tensor {} length should match",
                i + 1,
                tensor_idx
            );

            let mut max_diff: f32 = 0.0;
            let mut diff_count = 0usize;

            for (elem_idx, (v1, v2)) in d1.iter().zip(d2).enumerate() {
                let diff = (v1 - v2).abs();
                if diff > 1e-6 {
                    diff_count += 1;
                    if diff_count <= 5 {
                        debug!(
                            run = i + 1,
                            tensor = tensor_idx,
                            elem = elem_idx,
                            v1 = v1,
                            v2 = v2,
                            diff = diff,
                            "Value mismatch"
                        );
                    }
                }
                max_diff = max_diff.max(diff);
            }

            assert!(
                diff_count == 0,
                "Run {} tensor {}: {} elements differ (max diff: {}, tolerance: 1e-6)",
                i + 1,
                tensor_idx,
                diff_count,
                max_diff
            );
        }
    }
}

/// PreferGpu creates a session regardless of GPU availability (CPU fallback)
#[test]
fn test_prefer_gpu_fallback() {
    let _span = info_span!("test_prefer_gpu_fallback").entered();
    setup();

    let model_path = require_test_model!();

    let mut session = create_session(&model_path, ProviderPreference::PreferGpu)
        .expect("PreferGpu should always succeed");

    let output = run_random_inference(&mut session).expect("Inference should succeed");
    assert!(!output.shapes.is_empty(), "Should produce output");
}

/// RequireGpu fails with a clear error when no GPU is present
#[test]
fn test_require_gpu_error_semantics() {
    let _span = info_span!("test_require_gpu_error_semantics").entered();
    setup();

    let info = ProviderInfo::detect();
    if info.has_gpu() {
        skip_test!("GPU is available — cannot test failure path");
    }

    let model_path = require_test_model!();

    let result = create_session(&model_path, ProviderPreference::RequireGpu);
    assert!(result.is_err(), "RequireGpu should fail without GPU");

    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not available"),
        "Error should mention provider unavailability, got: {}",
        err
    );
}

/// Softmax probabilities sum to ~1.0, top-k predictions are sorted descending
#[test]
fn test_softmax_and_top_k() {
    let _span = info_span!("test_softmax_and_top_k").entered();
    setup();

    let model_path = require_test_model!();

    let mut session =
        create_session(&model_path, ProviderPreference::CpuOnly).expect("Should create session");

    let output = run_random_inference(&mut session).expect("Inference should succeed");

    let has_classification_output = output.shapes.iter().any(|s| s.len() == 2 && s[1] >= 100);
    if !has_classification_output {
        skip_test!("Model doesn't appear to be a classification model");
    }

    for (name, sum, valid) in &output.verify_softmax_sum(0.001) {
        assert!(
            *valid,
            "Softmax for '{}' should sum to ~1.0, got {}",
            name, sum
        );
    }

    for preds in &output.top_k_predictions(5) {
        assert!(!preds.is_empty(), "Should have predictions");
        for window in preds.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "Predictions should be sorted descending"
            );
        }
    }
}

/// Performance degradation detection over 10 iterations
#[test]
fn test_repeated_inference_stability() {
    let _span = info_span!("test_repeated_inference_stability").entered();
    setup();

    let model_path = require_test_model!();

    let mut session =
        create_session(&model_path, ProviderPreference::CpuOnly).expect("Should create session");

    const NUM_ITERATIONS: usize = 10;
    let mut timings = Vec::with_capacity(NUM_ITERATIONS);

    for i in 0..NUM_ITERATIONS {
        let start = Instant::now();
        let output = run_random_inference(&mut session)
            .unwrap_or_else(|e| panic!("Iteration {} failed: {}", i + 1, e));
        timings.push(start.elapsed());

        assert!(
            !output.shapes.is_empty(),
            "Iteration {} should have output",
            i + 1
        );
    }

    let times_ms: Vec<f64> = timings.iter().map(|t| t.as_secs_f64() * 1000.0).collect();
    let first_5_avg: f64 = times_ms[..5].iter().sum::<f64>() / 5.0;
    let last_5_avg: f64 = times_ms[5..].iter().sum::<f64>() / 5.0;
    let degradation_ratio = last_5_avg / first_5_avg;

    info!(
        first_5_avg_ms = format!("{:.2}", first_5_avg),
        last_5_avg_ms = format!("{:.2}", last_5_avg),
        degradation_ratio = format!("{:.2}x", degradation_ratio),
        "Performance stability check"
    );

    if degradation_ratio > 2.0 {
        warn!(
            degradation_ratio = format!("{:.2}x", degradation_ratio),
            "Significant performance degradation detected"
        );
    }
}
