//! Integration tests for ORT Wrapper
//!
//! These tests use test-log for tracing output. Set RUST_LOG to see traces:
//! ```
//! RUST_LOG=trace cargo test -- --nocapture
//! ```
//!
//! Timing information is included in spans for performance analysis.

use ort_wrapper::{
    create_session, get_test_model_path, init_ort, run_random_inference, ProviderInfo,
    ProviderPreference,
};
use std::time::Instant;
use test_log::test;
use tracing::{debug, info, info_span, warn};

// =============================================================================
// Test Setup Helpers
// =============================================================================

/// Initialize ORT and return timing info
fn setup() -> std::time::Duration {
    let start = Instant::now();
    init_ort();
    let elapsed = start.elapsed();
    info!(elapsed_ms = elapsed.as_millis(), "ORT initialization complete");
    elapsed
}

/// Helper to skip a test with proper tracing (instead of silent eprintln)
macro_rules! skip_test {
    ($reason:expr) => {{
        warn!(reason = $reason, "SKIPPING TEST");
        return;
    }};
}

/// Get test model or skip with traced warning
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

// =============================================================================
// Basic Linking Tests
// =============================================================================

/// This test passes if the binary links at all - verifies static linking works
#[test]
fn test_static_linking_works() {
    let _span = info_span!("test_static_linking_works").entered();
    let init_time = setup();

    // If we got here, static linking succeeded!
    let info = ProviderInfo::detect();
    assert!(info.cpu_available, "CPU provider should always be available");

    info!(
        init_ms = init_time.as_millis(),
        cpu = info.cpu_available,
        "Static linking verified"
    );
}

/// Test that provider detection works correctly
#[test]
fn test_provider_detection() {
    let _span = info_span!("test_provider_detection").entered();
    setup();

    let detect_start = Instant::now();
    let info = ProviderInfo::detect();
    let detect_time = detect_start.elapsed();

    // CPU is always available
    assert!(info.cpu_available, "CPU provider must always be available");

    info!(
        platform = std::env::consts::OS,
        accelerator = ProviderInfo::accelerator_name(),
        gpu_available = info.has_gpu(),
        detect_ms = detect_time.as_millis(),
        "Provider detection complete"
    );

    // Summary should work and contain CPU
    let summary = info.summary();
    assert!(
        summary.contains("CPU"),
        "Summary should mention CPU, got: {}",
        summary
    );
}

// =============================================================================
// CPU Inference Tests
// =============================================================================

/// Test CPU inference if a test model is available
#[test]
fn test_cpu_inference() {
    let _span = info_span!("test_cpu_inference").entered();
    setup();

    let model_path = require_test_model!();

    // Create session with timing
    let session_start = Instant::now();
    info!("Creating CPU session");
    let mut session = create_session(&model_path, ProviderPreference::CpuOnly)
        .expect("Should create CPU session");
    let session_time = session_start.elapsed();
    info!(session_ms = session_time.as_millis(), "Session created");

    // Verify session has inputs/outputs
    assert!(
        !session.inputs().is_empty(),
        "Model should have at least one input"
    );
    assert!(
        !session.outputs().is_empty(),
        "Model should have at least one output"
    );
    info!(
        inputs = session.inputs().len(),
        outputs = session.outputs().len(),
        "Session I/O verified"
    );

    // Run inference with timing
    let inference_start = Instant::now();
    info!("Running inference");
    let output = run_random_inference(&mut session).expect("CPU inference should succeed");
    let inference_time = inference_start.elapsed();
    info!(
        inference_ms = inference_time.as_millis(),
        output_count = output.shapes.len(),
        "Inference complete"
    );

    // Verify we got output
    assert!(!output.shapes.is_empty(), "Should have output shapes");
    assert!(!output.data.is_empty(), "Should have output data");
    assert_eq!(
        output.shapes.len(),
        output.data.len(),
        "Shape and data count must match"
    );

    // Validate output quality
    let validate_start = Instant::now();
    info!("Validating output");
    let report = output.validate().expect("Validation should not error");
    let validate_time = validate_start.elapsed();

    debug!(
        is_valid = report.is_valid,
        errors = ?report.errors,
        warnings = ?report.warnings,
        validate_ms = validate_time.as_millis(),
        "Validation complete"
    );

    assert!(
        report.is_valid,
        "Output should be valid. Errors: {:?}",
        report.errors
    );

    // Log timing summary
    info!(
        total_ms = (session_time + inference_time + validate_time).as_millis(),
        session_ms = session_time.as_millis(),
        inference_ms = inference_time.as_millis(),
        validate_ms = validate_time.as_millis(),
        "Test timing summary"
    );
}

/// Test output tensor validation in detail
#[test]
fn test_output_validation() {
    let _span = info_span!("test_output_validation").entered();
    setup();

    let model_path = require_test_model!();

    let mut session =
        create_session(&model_path, ProviderPreference::CpuOnly).expect("Should create session");

    let output = run_random_inference(&mut session).expect("Inference should succeed");
    let report = output.validate().expect("Validation should not error");

    // Check that we got statistics for all outputs
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

        // Basic sanity checks with detailed error messages
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

/// Test that inference produces non-trivial output (not all zeros)
#[test]
fn test_output_not_trivial() {
    let _span = info_span!("test_output_not_trivial").entered();
    setup();

    let model_path = require_test_model!();

    let mut session =
        create_session(&model_path, ProviderPreference::CpuOnly).expect("Should create session");

    let output = run_random_inference(&mut session).expect("Inference should succeed");
    let report = output.validate().expect("Validation should not error");

    // At least one output should have non-zero variance
    let outputs_with_variance: Vec<_> = report
        .tensor_stats
        .iter()
        .filter(|s| s.std_dev > 1e-10)
        .collect();

    info!(
        outputs_with_variance = outputs_with_variance.len(),
        total_outputs = report.tensor_stats.len(),
        "Variance check"
    );

    assert!(
        !outputs_with_variance.is_empty(),
        "At least one output should have variance (not all same value). Stats: {:?}",
        report
            .tensor_stats
            .iter()
            .map(|s| (&s.name, s.std_dev))
            .collect::<Vec<_>>()
    );

    // No output should be all zeros (for a real model with non-zero input)
    for stats in &report.tensor_stats {
        // Allow some tolerance - maybe 99% zeros is suspicious but not 100%
        let zero_ratio = stats.zero_count as f64 / stats.total_elements as f64;
        info!(
            name = %stats.name,
            zero_ratio = format!("{:.2}%", zero_ratio * 100.0),
            "Zero ratio"
        );

        assert!(
            stats.zero_count < stats.total_elements,
            "Output '{}' should not be ALL zeros ({}/{} are zero)",
            stats.name,
            stats.zero_count,
            stats.total_elements
        );

        // Warn if mostly zeros (might indicate a problem)
        if zero_ratio > 0.9 {
            warn!(
                name = %stats.name,
                zero_ratio = format!("{:.2}%", zero_ratio * 100.0),
                "Output is mostly zeros - may indicate inference issue"
            );
        }
    }
}

/// Test classification model output (softmax properties)
#[test]
fn test_classification_output_softmax() {
    let _span = info_span!("test_classification_output_softmax").entered();
    setup();

    let model_path = require_test_model!();

    let mut session =
        create_session(&model_path, ProviderPreference::CpuOnly).expect("Should create session");

    let output = run_random_inference(&mut session).expect("Inference should succeed");

    // Check if this looks like a classification model (1000 classes for ImageNet)
    let classification_outputs: Vec<_> = output
        .shapes
        .iter()
        .enumerate()
        .filter(|(_, s)| s.len() == 2 && s[1] >= 100)
        .collect();

    if classification_outputs.is_empty() {
        info!(
            shapes = ?output.shapes,
            "Model doesn't appear to be classification (no 2D output with >=100 classes)"
        );
        skip_test!("Model doesn't appear to be a classification model");
    }

    info!(
        classification_outputs = classification_outputs.len(),
        "Found classification outputs"
    );

    // Verify softmax sums to ~1.0
    let sums = output.verify_softmax_sum(0.001);
    for (name, sum, valid) in &sums {
        info!(name = %name, sum = sum, valid = valid, "Softmax sum");
        assert!(
            *valid,
            "Softmax for '{}' should sum to ~1.0 (within 0.001), got {} (diff: {})",
            name,
            sum,
            (sum - 1.0).abs()
        );
    }

    // Get top predictions and verify they're reasonable
    let predictions = output.top_k_predictions(5);
    for (i, preds) in predictions.iter().enumerate() {
        assert!(
            !preds.is_empty(),
            "Should have predictions for output {}",
            i
        );

        // Top probability should be positive
        let (top_class, top_prob) = preds[0];
        info!(
            output = i,
            top_class = top_class,
            top_prob = format!("{:.4}", top_prob),
            "Top prediction"
        );

        assert!(
            top_prob > 0.0,
            "Top probability should be positive, got {}",
            top_prob
        );
        assert!(
            top_prob <= 1.0,
            "Probability should be <= 1.0, got {}",
            top_prob
        );

        // Probabilities should be in descending order
        for (j, window) in preds.windows(2).enumerate() {
            assert!(
                window[0].1 >= window[1].1,
                "Probabilities should be sorted descending at position {}: {} >= {}",
                j,
                window[0].1,
                window[1].1
            );
        }
    }
}

/// Test that running inference twice produces consistent results
#[test]
fn test_inference_determinism() {
    let _span = info_span!("test_inference_determinism").entered();
    setup();

    let model_path = require_test_model!();

    let mut session =
        create_session(&model_path, ProviderPreference::CpuOnly).expect("Should create session");

    // Run inference multiple times with same (deterministic) input
    const NUM_RUNS: usize = 3;
    let mut outputs = Vec::with_capacity(NUM_RUNS);
    let mut timings = Vec::with_capacity(NUM_RUNS);

    for i in 0..NUM_RUNS {
        let start = Instant::now();
        let output = run_random_inference(&mut session)
            .unwrap_or_else(|e| panic!("Inference run {} should succeed: {}", i + 1, e));
        let elapsed = start.elapsed();

        info!(
            run = i + 1,
            elapsed_ms = elapsed.as_millis(),
            "Inference run complete"
        );

        timings.push(elapsed);
        outputs.push(output);
    }

    // Compare all runs against the first
    let baseline = &outputs[0];
    for (i, output) in outputs.iter().enumerate().skip(1) {
        // Shapes should be identical
        assert_eq!(
            baseline.shapes, output.shapes,
            "Run {} shapes should match run 1",
            i + 1
        );

        // Data should be identical (same input -> same output)
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

    // Report timing consistency
    let avg_ms: f64 = timings.iter().map(|t| t.as_secs_f64() * 1000.0).sum::<f64>() / NUM_RUNS as f64;
    let timing_strs: Vec<_> = timings
        .iter()
        .map(|t| format!("{}ms", t.as_millis()))
        .collect();

    info!(
        runs = NUM_RUNS,
        timings = ?timing_strs,
        avg_ms = format!("{:.2}", avg_ms),
        "Inference is deterministic"
    );
}

// =============================================================================
// GPU Inference Tests
// =============================================================================

/// Test GPU inference if available (CUDA on Linux, CoreML on macOS)
#[test]
fn test_gpu_inference() {
    let _span = info_span!("test_gpu_inference").entered();
    setup();

    let info = ProviderInfo::detect();
    if !info.has_gpu() {
        skip_test!(format!(
            "{} not available",
            ProviderInfo::accelerator_name()
        ));
    }

    let model_path = require_test_model!();

    info!(
        accelerator = ProviderInfo::accelerator_name(),
        "Creating GPU session"
    );

    let session_start = Instant::now();
    let mut session = create_session(&model_path, ProviderPreference::RequireGpu)
        .expect("Should create GPU session");
    let session_time = session_start.elapsed();

    info!(session_ms = session_time.as_millis(), "GPU session created");

    let inference_start = Instant::now();
    let output = run_random_inference(&mut session).expect("GPU inference should succeed");
    let inference_time = inference_start.elapsed();

    info!(
        inference_ms = inference_time.as_millis(),
        output_count = output.shapes.len(),
        "GPU inference complete"
    );

    assert!(!output.shapes.is_empty(), "Should have output shapes");

    let report = output.validate().expect("Validation should not error");
    assert!(
        report.is_valid,
        "GPU output should be valid: {:?}",
        report.errors
    );

    info!(
        total_ms = (session_time + inference_time).as_millis(),
        "GPU test complete"
    );
}

/// Test that GPU preference falls back to CPU when GPU unavailable
#[test]
fn test_gpu_fallback_to_cpu() {
    let _span = info_span!("test_gpu_fallback_to_cpu").entered();
    setup();

    let model_path = require_test_model!();

    let info = ProviderInfo::detect();
    info!(
        gpu_available = info.has_gpu(),
        "Testing PreferGpu fallback behavior"
    );

    // PreferGpu should work even without GPU (falls back to CPU)
    let start = Instant::now();
    let mut session = create_session(&model_path, ProviderPreference::PreferGpu)
        .expect("PreferGpu should succeed with CPU fallback");
    let session_time = start.elapsed();

    info!(
        session_ms = session_time.as_millis(),
        "Session created with PreferGpu"
    );

    let inference_start = Instant::now();
    let output = run_random_inference(&mut session).expect("Inference should succeed");
    let inference_time = inference_start.elapsed();

    assert!(!output.shapes.is_empty(), "Should have output shapes");

    info!(
        inference_ms = inference_time.as_millis(),
        shapes = ?output.shapes,
        "PreferGpu fallback test complete"
    );
}

/// Verify that RequireGpu fails gracefully when GPU is unavailable
#[test]
fn test_require_gpu_fails_without_gpu() {
    let _span = info_span!("test_require_gpu_fails_without_gpu").entered();
    setup();

    let info = ProviderInfo::detect();
    if info.has_gpu() {
        skip_test!("GPU is available - cannot test failure case");
    }

    let model_path = require_test_model!();

    info!("Testing RequireGpu failure when GPU unavailable");

    // Should fail with a clear error message
    let result = create_session(&model_path, ProviderPreference::RequireGpu);
    assert!(result.is_err(), "RequireGpu should fail without GPU");

    let err = result.unwrap_err().to_string();
    info!(error = %err, "Got expected error");

    assert!(
        err.contains("not available"),
        "Error should mention provider not available, got: {}",
        err
    );
}

// =============================================================================
// Performance / Stress Tests
// =============================================================================

/// Run multiple inferences to check for memory leaks or degradation
#[test]
fn test_repeated_inference_stability() {
    let _span = info_span!("test_repeated_inference_stability").entered();
    setup();

    let model_path = require_test_model!();

    let mut session =
        create_session(&model_path, ProviderPreference::CpuOnly).expect("Should create session");

    const NUM_ITERATIONS: usize = 10;
    let mut timings = Vec::with_capacity(NUM_ITERATIONS);

    info!(iterations = NUM_ITERATIONS, "Starting repeated inference test");

    for i in 0..NUM_ITERATIONS {
        let start = Instant::now();
        let output = run_random_inference(&mut session)
            .unwrap_or_else(|e| panic!("Iteration {} failed: {}", i + 1, e));
        let elapsed = start.elapsed();

        // Quick validation
        assert!(
            !output.shapes.is_empty(),
            "Iteration {} should have output",
            i + 1
        );

        timings.push(elapsed);

        if (i + 1) % 5 == 0 {
            info!(
                iteration = i + 1,
                elapsed_ms = elapsed.as_millis(),
                "Progress"
            );
        }
    }

    // Analyze timing stability
    let times_ms: Vec<f64> = timings.iter().map(|t| t.as_secs_f64() * 1000.0).collect();
    let avg_ms: f64 = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let min_ms = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = times_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance: f64 = times_ms.iter().map(|t| (t - avg_ms).powi(2)).sum::<f64>() / times_ms.len() as f64;
    let std_dev = variance.sqrt();

    info!(
        iterations = NUM_ITERATIONS,
        avg_ms = format!("{:.2}", avg_ms),
        min_ms = format!("{:.2}", min_ms),
        max_ms = format!("{:.2}", max_ms),
        std_dev_ms = format!("{:.2}", std_dev),
        "Timing statistics"
    );

    // Check for performance degradation (last 5 shouldn't be much slower than first 5)
    if NUM_ITERATIONS >= 10 {
        let first_5_avg: f64 = times_ms[..5].iter().sum::<f64>() / 5.0;
        let last_5_avg: f64 = times_ms[NUM_ITERATIONS - 5..].iter().sum::<f64>() / 5.0;
        let degradation_ratio = last_5_avg / first_5_avg;

        info!(
            first_5_avg_ms = format!("{:.2}", first_5_avg),
            last_5_avg_ms = format!("{:.2}", last_5_avg),
            degradation_ratio = format!("{:.2}x", degradation_ratio),
            "Performance stability check"
        );

        // Allow up to 2x slowdown (generous for test environment variability)
        if degradation_ratio > 2.0 {
            warn!(
                degradation_ratio = format!("{:.2}x", degradation_ratio),
                "Significant performance degradation detected"
            );
        }
    }
}
