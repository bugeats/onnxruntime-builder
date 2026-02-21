//! Usage examples for downstream crates.
//!
//! Each test is a self-contained example. The crate auto-initializes
//! ONNX Runtime — no manual setup required.

use ort_wrapper::{ndarray, ProviderPreference, SessionBuilder};
use std::path::PathBuf;

fn test_model() -> Option<PathBuf> {
    ort_wrapper::get_test_model_path().ok()
}

/// One-shot inference: model path + ndarray in, Vec<ArrayD<f32>> out
#[test]
fn one_shot_infer() {
    let Some(model_path) = test_model() else { return };

    let input = ndarray::Array4::<f32>::zeros((1, 3, 224, 224));
    let outputs = ort_wrapper::infer(&model_path, &input).expect("inference");

    assert_eq!(outputs[0].shape(), &[1, 1000, 1, 1]);
}

/// SessionBuilder: load model, create input tensor, run, extract output
#[test]
fn session_builder() {
    let Some(model_path) = test_model() else { return };

    let mut session = SessionBuilder::from_file(&model_path)
        .with_provider(ProviderPreference::CpuOnly)
        .build()
        .expect("build session");

    // ort accepts ndarray types directly via Tensor::from_array
    let input = ndarray::Array4::<f32>::zeros((1, 3, 224, 224));
    let tensor = ort_wrapper::ort::value::Tensor::from_array(input).unwrap();

    let outputs = session
        .run(ort_wrapper::ort::inputs![tensor])
        .expect("run inference");

    let (_name, value) = outputs.iter().next().expect("at least one output");
    let output = value.try_extract_array::<f32>().expect("extract f32");
    assert_eq!(output.shape(), &[1, 1000, 1, 1]);
}

/// Load model from in-memory bytes
#[test]
fn session_from_memory() {
    let Some(model_path) = test_model() else { return };
    let model_bytes = std::fs::read(&model_path).expect("read model file");

    let session = SessionBuilder::from_memory(model_bytes)
        .with_provider(ProviderPreference::CpuOnly)
        .build()
        .expect("build from memory");

    assert!(!session.inputs().is_empty());
}

/// Runtime provider detection
#[test]
fn provider_detection() {
    let info = ort_wrapper::ProviderInfo::detect();

    assert!(info.cpu_available);
    // info.cuda_available / info.coreml_available reflect compiled features + hardware
    // info.has_gpu() — true if any accelerator is available
    // ProviderInfo::accelerator_name() — "CUDA", "CoreML", or "None (CPU only)"
}
