//! Build and linking smoke test
//!
//! Verifies that the binary links against ONNX Runtime and can initialize.

use ort_wrapper::{init, ProviderInfo};
use test_log::test;
use tracing::{info, info_span};

/// Passes if the binary links and ORT initializes — verifies static linking works
#[test]
fn test_static_linking_works() {
    let _span = info_span!("test_static_linking_works").entered();
    init();

    let info = ProviderInfo::detect();
    assert!(info.cpu_available, "CPU provider should always be available");

    info!(cpu = info.cpu_available, "Static linking verified");
}
