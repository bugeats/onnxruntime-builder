//! API usage examples for `ort_wrapper`
//!
//! Each test demonstrates a primary entry point. The crate auto-initializes
//! the ONNX Runtime API — no manual `init()` call required.

use ort_wrapper::{
    get_test_model_path, infer, ndarray, PreparedInput, ProviderInfo, ProviderPreference,
    SessionBuilder,
};
use std::path::PathBuf;

fn test_model() -> Option<PathBuf> {
    get_test_model_path().ok()
}

/// SessionBuilder::from_file — build a session, run inference via re-exported ort
#[test]
fn session_builder_from_file() {
    let Some(model_path) = test_model() else { return };

    let mut session = SessionBuilder::from_file(&model_path)
        .with_provider(ProviderPreference::CpuOnly)
        .build()
        .expect("build session");

    let prepared = PreparedInput::from_session(&session, 0);
    let tensor = prepared.to_tensor();
    let input = ort_wrapper::ort::session::SessionInputValue::from(&tensor);
    let outputs = session.run(std::slice::from_ref(&input)).unwrap();

    let (_name, value) = outputs.iter().next().expect("model should have output");
    let _array = value.try_extract_array::<f32>().expect("extract f32 output");
}

/// SessionBuilder::from_memory — load model bytes directly
#[test]
fn session_builder_from_memory() {
    let Some(model_path) = test_model() else { return };
    let model_bytes = std::fs::read(&model_path).expect("read model file");

    let session = SessionBuilder::from_memory(model_bytes)
        .with_provider(ProviderPreference::CpuOnly)
        .build()
        .expect("build session from memory");

    assert!(!session.inputs().is_empty());
}

/// infer() — one-shot: model path + ndarray in, Vec<ArrayD<f32>> out
#[test]
fn one_shot_infer() {
    let Some(model_path) = test_model() else { return };

    // SqueezeNet: 1×3×224×224 image tensor
    let input = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1, 3, 224, 224]), 0.5f32);
    let outputs = infer(&model_path, &input).expect("one-shot inference");

    assert!(!outputs.is_empty());
}

/// ProviderInfo — runtime discovery of available execution providers
#[test]
fn provider_discovery() {
    let info = ProviderInfo::detect();

    assert!(info.cpu_available);
    let _has_gpu: bool = info.has_gpu();
    let _name: &str = ProviderInfo::accelerator_name();
    let _summary: String = info.summary();
}
