//! Inference benchmarks for ONNX Runtime static build
//!
//! Note: Benchmarks use tracing for skip messages. These will be visible
//! if you set RUST_LOG=warn before running benchmarks.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use onnxruntime_test_harness::{
    create_session, get_test_model_path, init_ort, PreparedInput, ProviderInfo, ProviderPreference,
};
use ort::session::Session;
use tracing::warn;

/// Run a single inference pass using PreparedInput
fn run_inference(session: &mut Session, prepared: &PreparedInput) {
    let tensor = prepared.to_tensor();
    let input_value = ort::session::SessionInputValue::from(&tensor);
    let _ = session
        .run(std::slice::from_ref(&input_value))
        .expect("Inference should succeed");
}

fn cpu_inference_benchmark(c: &mut Criterion) {
    init_ort();

    let model_path = match get_test_model_path() {
        Ok(path) => path,
        Err(e) => {
            warn!(error = %e, "Skipping CPU benchmarks: no test model");
            return;
        }
    };

    let mut session = match create_session(&model_path, ProviderPreference::CpuOnly) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "Failed to create CPU session");
            return;
        }
    };

    let prepared = PreparedInput::from_session(&session, 0);

    let mut group = c.benchmark_group("cpu_inference");
    group.throughput(Throughput::Elements(1));

    group.bench_function("single", |b| {
        b.iter(|| run_inference(&mut session, &prepared))
    });

    group.finish();
}

fn gpu_inference_benchmark(c: &mut Criterion) {
    init_ort();

    let info = ProviderInfo::detect();
    if !info.has_gpu() {
        warn!(
            accelerator = ProviderInfo::accelerator_name(),
            "Skipping GPU benchmarks: not available"
        );
        return;
    }

    let model_path = match get_test_model_path() {
        Ok(path) => path,
        Err(e) => {
            warn!(error = %e, "Skipping GPU benchmarks: no test model");
            return;
        }
    };

    let mut session = match create_session(&model_path, ProviderPreference::RequireGpu) {
        Ok(s) => s,
        Err(e) => {
            warn!(
                accelerator = ProviderInfo::accelerator_name(),
                error = %e,
                "Failed to create GPU session"
            );
            return;
        }
    };

    let prepared = PreparedInput::from_session(&session, 0);

    // Use platform-specific benchmark group name
    let group_name = format!("{}_inference", ProviderInfo::accelerator_name().to_lowercase());
    let mut group = c.benchmark_group(&group_name);
    group.throughput(Throughput::Elements(1));

    group.bench_function("single", |b| {
        b.iter(|| run_inference(&mut session, &prepared))
    });

    group.finish();
}

fn batch_size_benchmark(c: &mut Criterion) {
    init_ort();

    let model_path = match get_test_model_path() {
        Ok(path) => path,
        Err(e) => {
            warn!(error = %e, "Skipping batch benchmarks: no test model");
            return;
        }
    };

    let mut session = match create_session(&model_path, ProviderPreference::CpuOnly) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "Failed to create session for batch benchmarks");
            return;
        }
    };

    let base_input = PreparedInput::from_session(&session, 0);

    let mut group = c.benchmark_group("batch_scaling");

    for batch_size in [1, 2, 4, 8].iter() {
        let prepared = base_input.with_batch_size(*batch_size);

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| b.iter(|| run_inference(&mut session, &prepared)),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    cpu_inference_benchmark,
    gpu_inference_benchmark,
    batch_size_benchmark
);

criterion_main!(benches);
