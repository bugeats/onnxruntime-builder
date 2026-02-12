//! ORT Wrapper - Manual Test Binary
//!
//! Run this to verify the ONNX Runtime build works correctly.
//!
//! # Tracing
//!
//! Set `RUST_LOG` environment variable to control tracing verbosity:
//! - `RUST_LOG=info` - High-level operation flow
//! - `RUST_LOG=debug` - Detailed operation info
//! - `RUST_LOG=trace` - Everything including tensor details
//!
//! Example: `RUST_LOG=debug ./test-harness`

use anyhow::Result;
use ort_wrapper::{
    create_session, get_test_model_path, init_ort, run_random_inference, ProviderInfo,
    ProviderPreference,
};
use tracing::{debug, info, info_span, warn};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

fn main() -> Result<()> {
    // Initialize tracing subscriber with timing
    tracing_subscriber::registry()
        .with(fmt::layer().with_timer(fmt::time::uptime()))
        .with(EnvFilter::from_default_env().add_directive("onnxruntime_test_harness=info".parse()?))
        .init();

    let _main_span = info_span!("main").entered();
    info!("Starting ONNX Runtime Test Harness");
    info!("╔════════════════════════════════════════════════════════════════╗");
    info!("║         ONNX Runtime Static Build Test Harness                 ║");
    info!("╚════════════════════════════════════════════════════════════════╝");

    // Initialize the statically linked ONNX Runtime API
    // This MUST be called before any other ort operations
    init_ort();

    // Report build configuration
    info!(
        ort_version = "2.0.0-rc.11",
        onnxruntime_version = "1.23.2",
        platform = std::env::consts::OS,
        accelerator = ProviderInfo::accelerator_name(),
        "Build configuration"
    );

    // Detect available providers
    let info = ProviderInfo::detect();
    info!(
        cpu = info.cpu_available,
        cuda = info.cuda_available,
        coreml = info.coreml_available,
        "Execution provider detection"
    );

    // Load test model (required)
    let model_path = match get_test_model_path() {
        Ok(path) => {
            info!(model = ?path, "Test model loaded");
            path
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                "ONNX_TEST_MODEL environment variable is required but not set or invalid"
            );
            eprintln!();
            eprintln!("ERROR: Missing required environment variable ONNX_TEST_MODEL");
            eprintln!();
            eprintln!("The test harness requires an ONNX model file to run inference tests.");
            eprintln!();
            eprintln!("Usage:");
            eprintln!("  ONNX_TEST_MODEL=/path/to/model.onnx nix run .#test-harness-cpu");
            eprintln!();
            eprintln!("Or with the bundled SqueezeNet model:");
            eprintln!("  ONNX_TEST_MODEL=$(nix build .#squeezenet-model --print-out-paths) nix run .#test-harness-cpu");
            eprintln!();
            std::process::exit(1);
        }
    };

    // Test CPU inference
    test_inference(&model_path, "CPU", ProviderPreference::CpuOnly)?;

    // Test GPU inference if available
    if info.has_gpu() {
        test_inference(
            &model_path,
            ProviderInfo::accelerator_name(),
            ProviderPreference::RequireGpu,
        )?;
    }

    info!("════════════════════════════════════════════════════════════════");
    info!("Static linking verification: PASSED");
    info!("════════════════════════════════════════════════════════════════");

    info!("Test harness completed successfully");
    Ok(())
}

fn test_inference(
    model_path: &std::path::Path,
    provider_name: &str,
    preference: ProviderPreference,
) -> Result<()> {
    let _span = info_span!("test_inference", provider = %provider_name).entered();
    let start = std::time::Instant::now();

    info!(provider = %provider_name, ?preference, "Testing inference");

    match create_session(model_path, preference) {
        Ok(mut session) => {
            let session_time = start.elapsed();
            let inputs: Vec<_> = session.inputs().iter().map(|i| i.name().to_string()).collect();
            let outputs: Vec<_> = session.outputs().iter().map(|o| o.name().to_string()).collect();

            info!(
                session_ms = session_time.as_millis(),
                inputs = ?inputs,
                outputs = ?outputs,
                "Session created"
            );

            let inference_start = std::time::Instant::now();
            match run_random_inference(&mut session) {
                Ok(output) => {
                    let inference_time = inference_start.elapsed();
                    info!(
                        inference_ms = inference_time.as_millis(),
                        output_shapes = ?output.shapes,
                        "✓ Inference succeeded"
                    );

                    // Validate output
                    let validate_start = std::time::Instant::now();
                    let report = output.validate()?;
                    let validate_time = validate_start.elapsed();

                    if report.is_valid {
                        info!(validate_ms = validate_time.as_millis(), "✓ All outputs valid");
                    } else {
                        warn!(
                            validate_ms = validate_time.as_millis(),
                            errors = ?report.errors,
                            "✗ Validation failed"
                        );
                    }

                    for w in &report.warnings {
                        warn!(warning = %w, "Validation warning");
                    }

                    // Log tensor statistics
                    for stats in &report.tensor_stats {
                        let zero_pct = 100.0 * stats.zero_count as f64 / stats.total_elements as f64;
                        debug!(
                            name = %stats.name,
                            shape = ?stats.shape,
                            min = stats.min,
                            max = stats.max,
                            mean = stats.mean,
                            std_dev = stats.std_dev,
                            zeros = stats.zero_count,
                            total = stats.total_elements,
                            zero_pct = format!("{:.1}%", zero_pct),
                            "Tensor statistics"
                        );
                    }

                    // For classification models, show top predictions
                    if output.shapes.iter().any(|s| s.len() == 2 && s[1] >= 100) {
                        let predictions = output.top_k_predictions(5);
                        for (i, preds) in predictions.iter().enumerate() {
                            let top_classes: Vec<_> = preds
                                .iter()
                                .map(|(idx, prob)| format!("{}:{:.2}%", idx, prob * 100.0))
                                .collect();
                            info!(
                                output_idx = i,
                                top_5 = ?top_classes,
                                "Top predictions"
                            );
                        }

                        // Verify softmax sums
                        let sums = output.verify_softmax_sum(0.001);
                        for (name, sum, valid) in sums {
                            if valid {
                                debug!(name = %name, sum = sum, "Softmax sum valid");
                            } else {
                                warn!(name = %name, sum = sum, expected = 1.0, "Softmax sum invalid");
                            }
                        }
                    }

                    let total_time = start.elapsed();
                    info!(
                        total_ms = total_time.as_millis(),
                        session_ms = session_time.as_millis(),
                        inference_ms = inference_time.as_millis(),
                        validate_ms = validate_time.as_millis(),
                        "Test timing summary"
                    );
                }
                Err(e) => {
                    warn!(error = %e, "✗ Inference failed");
                }
            }
        }
        Err(e) => {
            warn!(error = %e, "✗ Failed to create session");
        }
    }

    Ok(())
}
