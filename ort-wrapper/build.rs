//! Build script for ONNX Runtime linking
//!
//! This build script takes FULL control of linking ONNX Runtime.
//! ort-sys is configured with `alternative-backend` feature which disables its linking.
//!
//! Required environment variables:
//! - ORT_LIB_LOCATION: Path to directory containing libonnxruntime.a
//!   Note: This name is an `ort` crate convention, not ours. See https://ort.pyke.io/setup/linking
//!
//! Optional environment variables:
//! - CUDNN_LIB_PATH: Path to cuDNN shared libraries (Linux with static cuda only)
//!
//! Feature flags:
//! - cuda: Static CUDA linking (libonnxruntime.a includes CUDA providers)
//! - cuda-dlopen: Hybrid mode — static core, CUDA providers load at runtime via dlopen
//! - coreml: Enable CoreML linking (macOS only)
//!
//! Nix integration:
//! - Automatically picks up library paths from NIX_LDFLAGS when building under Nix

use std::env;
use std::path::PathBuf;

/// Discover abseil static libraries from ORT_LIB_LOCATION at build time.
/// The directory is the single source of truth — no hardcoded list to maintain.
fn find_abseil_libs(lib_dir: &str) -> Vec<PathBuf> {
    let mut libs: Vec<PathBuf> = std::fs::read_dir(lib_dir)
        .unwrap_or_else(|e| panic!("Cannot read ORT_LIB_LOCATION '{}': {}", lib_dir, e))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            if name.starts_with("libabsl_") && name.ends_with(".a") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    libs.sort(); // Deterministic output for reproducible builds
    libs
}

fn main() {
    println!("cargo:rerun-if-env-changed=ORT_LIB_LOCATION");
    println!("cargo:rerun-if-env-changed=CUDNN_LIB_PATH");
    println!("cargo:rerun-if-env-changed=NIX_LDFLAGS");

    let lib_dir = env::var("ORT_LIB_LOCATION")
        .expect("ORT_LIB_LOCATION must be set to the ONNX Runtime lib directory");
    let lib_path = PathBuf::from(&lib_dir);

    let onnxruntime_lib = lib_path.join("libonnxruntime.a");
    if !onnxruntime_lib.exists() {
        panic!(
            "libonnxruntime.a not found at {:?}\n\
             Make sure ORT_LIB_LOCATION points to the lib/ directory from `nix build`",
            onnxruntime_lib
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir);

    // Nix buildInputs expose library paths via NIX_LDFLAGS
    if let Ok(nix_ldflags) = env::var("NIX_LDFLAGS") {
        for flag in nix_ldflags.split_whitespace() {
            if let Some(path) = flag.strip_prefix("-L") {
                println!("cargo:rustc-link-search=native={}", path);
            }
        }
    }

    // Static CUDA patching creates duplicate symbols — not needed in hybrid/dlopen mode
    #[cfg(all(feature = "cuda", not(feature = "cuda-dlopen")))]
    println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");

    println!("cargo:rustc-link-lib=static=onnxruntime");

    // Resolves __cudaRegisterLinkedBinary_* from separable compilation (-dc)
    // Only needed when CUDA objects are in the static archive
    #[cfg(all(feature = "cuda", not(feature = "cuda-dlopen")))]
    {
        let dlink_obj = lib_path.join("cuda_device_link.o");
        if dlink_obj.exists() {
            println!(
                "cargo:rustc-link-arg={}",
                dlink_obj.display()
            );
            println!("cargo:warning=Linking CUDA device link object: {:?}", dlink_obj);
        } else {
            println!("cargo:warning=CUDA device link object not found at {:?}", dlink_obj);
        }
    }

    #[cfg(target_os = "linux")]
    link_linux(&lib_path);

    #[cfg(target_os = "macos")]
    link_macos();

    let _ = lib_path;
}

#[cfg(target_os = "linux")]
fn link_linux(lib_path: &PathBuf) {
    println!("cargo:rustc-link-lib=stdc++");

    // Dependencies not merged into libonnxruntime.a — linked separately
    // Dynamic linking where nixpkgs lacks static variants
    println!("cargo:rustc-link-lib=protobuf");

    println!("cargo:rustc-link-lib=static=onnx");
    println!("cargo:rustc-link-lib=static=onnx_proto");

    link_abseil();

    println!("cargo:rustc-link-lib=re2");

    // clog is built into cpuinfo in nixpkgs
    println!("cargo:rustc-link-lib=static=cpuinfo");

    // Static CUDA: link CUDA libraries directly (not used in hybrid/dlopen mode)
    #[cfg(all(feature = "cuda", not(feature = "cuda-dlopen")))]
    link_cuda(lib_path);

    let _ = lib_path;
}

/// Link CUDA libraries statically. Only for fully-static CUDA builds —
/// hybrid mode loads these at runtime via the CUDA provider .so.
#[cfg(all(target_os = "linux", feature = "cuda", not(feature = "cuda-dlopen")))]
fn link_cuda(lib_path: &PathBuf) {
    // cuDNN: dynamic only — static linking has performance penalties
    let cudnn_bundled = lib_path.join("cudnn");
    if cudnn_bundled.exists() {
        println!("cargo:rustc-link-search=native={}", cudnn_bundled.display());
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            cudnn_bundled.display()
        );
    }

    if let Ok(cudnn_path) = env::var("CUDNN_LIB_PATH") {
        println!("cargo:rustc-link-search=native={}", cudnn_path);
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cudnn_path);
    }

    println!("cargo:rustc-link-lib=cudnn");

    // Static ORT still references these CUDA symbols
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=curand");
    println!("cargo:rustc-link-lib=cufft");
    println!("cargo:rustc-link-lib=cusparse");
}

#[cfg(target_os = "linux")]
fn link_abseil() {
    // Link from our built abseil, NOT system abseil.
    // Uses rustc-link-arg with --start-group/--end-group because cargo's
    // rustc-link-lib flags go in a fixed position that prevents us from
    // wrapping abseil's circular dependencies with linker groups.
    let lib_dir = env::var("ORT_LIB_LOCATION").unwrap();
    let libs = find_abseil_libs(&lib_dir);

    println!("cargo:rustc-link-arg=-Wl,--start-group");
    for lib_path in &libs {
        println!("cargo:rustc-link-arg={}", lib_path.display());
    }
    println!("cargo:rustc-link-arg=-Wl,--end-group");
}

#[cfg(target_os = "macos")]
fn link_macos() {
    println!("cargo:rustc-link-lib=c++");

    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // Dynamic linking where nixpkgs lacks static variants
    println!("cargo:rustc-link-lib=protobuf");
    // ONNX is already in libonnxruntime.a on macOS (unlike Linux)
    println!("cargo:rustc-link-lib=re2");

    link_abseil_macos();
}

#[cfg(target_os = "macos")]
fn link_abseil_macos() {
    // macOS linker resolves ordering automatically (no --start-group needed)
    let lib_dir = env::var("ORT_LIB_LOCATION").unwrap();
    for lib_path in &find_abseil_libs(&lib_dir) {
        let name = lib_path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("lib"))
            .expect("abseil library has unexpected filename format");
        println!("cargo:rustc-link-lib=static={}", name);
    }
}
