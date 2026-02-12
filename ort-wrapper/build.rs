//! Build script for ONNX Runtime linking
//!
//! This build script takes FULL control of linking ONNX Runtime.
//! ort-sys is configured with `alternative-backend` feature which disables its linking.
//!
//! Required environment variables:
//! - ORT_LIB_LOCATION: Path to directory containing libonnxruntime.a (static builds)
//!   Note: This name is an `ort` crate convention, not ours. See https://ort.pyke.io/setup/linking
//! - ORT_DYLIB_PATH: Path to directory containing libonnxruntime.so (cuda-dyn builds)
//!
//! Optional environment variables:
//! - CUDNN_LIB_PATH: Path to cuDNN shared libraries (Linux with cuda feature only)
//!
//! Feature flags:
//! - cuda: Enable static CUDA linking (Linux only)
//! - cuda-dyn: Enable dynamic CUDA linking via libonnxruntime.so (Linux only)
//! - coreml: Enable CoreML linking (macOS only)
//!
//! Nix integration:
//! - Automatically picks up library paths from NIX_LDFLAGS when building under Nix

use std::env;
use std::path::PathBuf;

// =============================================================================
// Dynamic CUDA Linking (cuda-dyn feature)
// =============================================================================

/// Dynamic linking path for CUDA builds using shared libonnxruntime.so
/// This is much simpler than static linking - the shared library handles
/// all internal dependencies (abseil, protobuf, CUDA providers, etc.)
#[cfg(all(target_os = "linux", feature = "cuda-dyn"))]
fn link_cuda_dynamic() {
    println!("cargo:rerun-if-env-changed=ORT_DYLIB_PATH");
    println!("cargo:rerun-if-env-changed=CUDNN_LIB_PATH");

    let lib_dir = env::var("ORT_DYLIB_PATH")
        .expect("ORT_DYLIB_PATH must be set for cuda-dyn feature");
    let lib_path = PathBuf::from(&lib_dir);

    // Verify libonnxruntime.so exists
    let dylib = lib_path.join("libonnxruntime.so");
    if !dylib.exists() {
        panic!(
            "libonnxruntime.so not found at {:?}\n\
             Make sure ORT_DYLIB_PATH points to the shared library directory",
            dylib
        );
    }

    // Add library search path and rpath for runtime discovery
    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);

    // Link the shared ONNX Runtime library
    println!("cargo:rustc-link-lib=dylib=onnxruntime");

    // cuDNN path for runtime
    let cudnn_path = lib_path.join("cudnn");
    if cudnn_path.exists() {
        println!("cargo:rustc-link-search=native={}", cudnn_path.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cudnn_path.display());
    }
    if let Ok(cudnn_env) = env::var("CUDNN_LIB_PATH") {
        println!("cargo:rustc-link-search=native={}", cudnn_env);
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cudnn_env);
    }

    // C++ standard library (required for ONNX Runtime C++ internals)
    println!("cargo:rustc-link-lib=stdc++");
}

// =============================================================================
// Abseil Libraries (shared between Linux and macOS)
// =============================================================================

/// Complete list of abseil static libraries required by ONNX Runtime
///
/// Abseil has complex interdependencies between its many small libraries.
/// This list is used by both Linux and macOS linking, though the linking
/// strategy differs (Linux uses --start-group/--end-group, macOS links directly).
const ABSEIL_LIBS: &[&str] = &[
    // Base utilities
    "absl_base",
    "absl_spinlock_wait",
    "absl_malloc_internal",
    "absl_raw_logging_internal",
    "absl_throw_delegate",
    "absl_strerror",
    "absl_log_severity",
    // String utilities
    "absl_strings",
    "absl_strings_internal",
    "absl_string_view",
    "absl_str_format_internal",
    "absl_cord",
    "absl_cord_internal",
    "absl_cordz_functions",
    "absl_cordz_handle",
    "absl_cordz_info",
    // Numeric
    "absl_int128",
    // Time
    "absl_time",
    "absl_time_zone",
    "absl_civil_time",
    // Synchronization
    "absl_synchronization",
    "absl_graphcycles_internal",
    "absl_kernel_timeout_internal",
    // Hash/Container
    "absl_hash",
    "absl_city",
    "absl_low_level_hash",
    "absl_hashtablez_sampler",
    "absl_raw_hash_set",
    // Debugging
    "absl_stacktrace",
    "absl_symbolize",
    "absl_debugging_internal",
    "absl_demangle_internal",
    "absl_demangle_rust",
    "absl_decode_rust_punycode",
    "absl_utf8_for_code_point",
    "absl_examine_stack",
    "absl_leak_check",
    // Logging (complete set)
    "absl_log_globals",
    "absl_log_entry",
    "absl_log_initialize",
    "absl_log_internal_check_op",
    "absl_log_internal_conditions",
    "absl_log_internal_fnmatch",
    "absl_log_internal_format",
    "absl_log_internal_globals",
    "absl_log_internal_log_sink_set",
    "absl_log_internal_message",
    "absl_log_internal_nullguard",
    "absl_log_internal_proto",
    "absl_log_sink",
    "absl_vlog_config_internal",
    // Status
    "absl_status",
    "absl_statusor",
    // Random
    "absl_random_distributions",
    "absl_random_internal_platform",
    "absl_random_internal_pool_urbg",
    "absl_random_internal_randen",
    "absl_random_internal_randen_hwaes",
    "absl_random_internal_randen_hwaes_impl",
    "absl_random_internal_randen_slow",
    "absl_random_internal_seed_material",
    "absl_random_seed_gen_exception",
    "absl_random_seed_sequences",
    // CRC
    "absl_crc32c",
    "absl_crc_cord_state",
    "absl_crc_cpu_detect",
    "absl_crc_internal",
    // Flags
    "absl_flags_commandlineflag",
    "absl_flags_commandlineflag_internal",
    "absl_flags_config",
    "absl_flags_internal",
    "absl_flags_marshalling",
    "absl_flags_private_handle_accessor",
    "absl_flags_program_name",
    "absl_flags_reflection",
    // Misc
    "absl_bad_optional_access",
    "absl_bad_variant_access",
    "absl_die_if_null",
    "absl_exponential_biased",
];

fn main() {
    // Dynamic CUDA linking takes a completely different path
    #[cfg(all(target_os = "linux", feature = "cuda-dyn"))]
    {
        link_cuda_dynamic();
        return;
    }

    // Re-run if these change
    println!("cargo:rerun-if-env-changed=ORT_LIB_LOCATION");
    println!("cargo:rerun-if-env-changed=CUDNN_LIB_PATH");
    println!("cargo:rerun-if-env-changed=NIX_LDFLAGS");

    let lib_dir = env::var("ORT_LIB_LOCATION")
        .expect("ORT_LIB_LOCATION must be set to the ONNX Runtime lib directory");
    let lib_path = PathBuf::from(&lib_dir);

    // Verify the library exists
    let onnxruntime_lib = lib_path.join("libonnxruntime.a");
    if !onnxruntime_lib.exists() {
        panic!(
            "libonnxruntime.a not found at {:?}\n\
             Make sure ORT_LIB_LOCATION points to the lib/ directory from `nix build`",
            onnxruntime_lib
        );
    }

    // Add our lib directory to the search path
    println!("cargo:rustc-link-search=native={}", lib_dir);

    // Parse Nix library paths from NIX_LDFLAGS
    // This allows us to find libraries from buildInputs (cpuinfo, abseil, etc.)
    if let Ok(nix_ldflags) = env::var("NIX_LDFLAGS") {
        for flag in nix_ldflags.split_whitespace() {
            if let Some(path) = flag.strip_prefix("-L") {
                println!("cargo:rustc-link-search=native={}", path);
            }
        }
    }

    // The unified archive has some duplicate symbols between the CUDA provider
    // and provider bridge (from patching them to be static). Allow this.
    #[cfg(feature = "cuda")]
    println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");

    // Link the unified static ONNX Runtime library
    // This contains all ONNX Runtime components (and CUDA provider if built with CUDA)
    println!("cargo:rustc-link-lib=static=onnxruntime");

    // Link CUDA device link object if it exists (only for CUDA builds)
    // This resolves __cudaRegisterLinkedBinary_* symbols from separable compilation
    #[cfg(feature = "cuda")]
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

    // Platform-specific linking
    #[cfg(target_os = "linux")]
    link_linux(&lib_path);

    #[cfg(target_os = "macos")]
    link_macos();

    // Suppress unused variable warning when not on Linux
    let _ = lib_path;
}

#[cfg(target_os = "linux")]
fn link_linux(lib_path: &PathBuf) {
    // ==========================================================================
    // C++ Standard Library
    // ==========================================================================
    println!("cargo:rustc-link-lib=stdc++");

    // ==========================================================================
    // ONNX Runtime External Dependencies
    // These are required by libonnxruntime.a but not merged into it
    // ==========================================================================

    // Protocol Buffers (serialization for ONNX models)
    // Note: Using dynamic linking - static protobuf not available in nixpkgs
    println!("cargo:rustc-link-lib=protobuf");

    // ONNX libraries (schema definitions and protobuf messages)
    // These are NOT merged into libonnxruntime.a and must be linked separately
    println!("cargo:rustc-link-lib=static=onnx");
    println!("cargo:rustc-link-lib=static=onnx_proto");

    // Abseil (Google's C++ utility library)
    // ONNX Runtime uses many abseil components
    link_abseil();

    // RE2 (Google's regex library)
    // Note: Using dynamic linking - static re2 not available in nixpkgs
    println!("cargo:rustc-link-lib=re2");

    // cpuinfo (CPU feature detection)
    // Note: clog is built into cpuinfo in nixpkgs, don't link separately
    println!("cargo:rustc-link-lib=static=cpuinfo");

    // ==========================================================================
    // CUDA Dependencies (only when cuda feature is enabled)
    // ==========================================================================
    #[cfg(feature = "cuda")]
    link_cuda(lib_path);

    // Suppress unused variable warning when cuda feature is not enabled
    #[cfg(not(feature = "cuda"))]
    let _ = lib_path;
}

/// Link CUDA libraries (only called when cuda feature is enabled)
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn link_cuda(lib_path: &PathBuf) {
    // cuDNN is dynamically linked (static has performance penalties)
    let cudnn_bundled = lib_path.join("cudnn");
    if cudnn_bundled.exists() {
        println!("cargo:rustc-link-search=native={}", cudnn_bundled.display());
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            cudnn_bundled.display()
        );
    }

    // Additional cuDNN path from environment
    if let Ok(cudnn_path) = env::var("CUDNN_LIB_PATH") {
        println!("cargo:rustc-link-search=native={}", cudnn_path);
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cudnn_path);
    }

    // cuDNN shared library
    println!("cargo:rustc-link-lib=cudnn");

    // CUDA runtime libraries
    // The static ONNX Runtime build may still need some CUDA symbols
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=curand");
    println!("cargo:rustc-link-lib=cufft");
    println!("cargo:rustc-link-lib=cusparse");
}

#[cfg(target_os = "linux")]
fn link_abseil() {
    // IMPORTANT: Link from our built abseil (lts_20240722), NOT system abseil.
    //
    // Abseil has complex interdependencies between its many small libraries.
    // With static linking, the linker processes archives left-to-right and only
    // keeps symbols that are currently referenced. This causes undefined symbol
    // errors when library A depends on library B, but A is linked before B.
    //
    // Cargo's rustc-link-lib flags go in a fixed position, before rustc-link-arg.
    // This means we can't wrap the libraries with --start-group/--end-group or
    // --whole-archive using the normal approach.
    //
    // Solution: Use rustc-link-arg for everything, giving us full control over
    // the link command. We use --start-group/--end-group to let the linker
    // iterate until all symbols are resolved.
    let lib_dir = env::var("ORT_LIB_LOCATION").unwrap();

    // Start a linker group to handle circular dependencies
    println!("cargo:rustc-link-arg=-Wl,--start-group");

    // Link each abseil library using explicit path
    for lib in ABSEIL_LIBS {
        let lib_path = format!("{}/lib{}.a", lib_dir, lib);
        println!("cargo:rustc-link-arg={}", lib_path);
    }

    // End the linker group
    println!("cargo:rustc-link-arg=-Wl,--end-group");
}

#[cfg(target_os = "macos")]
fn link_macos() {
    // C++ standard library (libc++ on macOS)
    println!("cargo:rustc-link-lib=c++");

    // Apple frameworks for CoreML
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // ONNX Runtime external dependencies
    // Note: Using dynamic linking - static protobuf not available in nixpkgs
    println!("cargo:rustc-link-lib=protobuf");
    // Note: ONNX is built internally and already in libonnxruntime.a
    // Note: Using dynamic linking for re2 - static not available in nixpkgs
    println!("cargo:rustc-link-lib=re2");

    // Abseil on macOS
    link_abseil_macos();
}

#[cfg(target_os = "macos")]
fn link_abseil_macos() {
    // macOS linker handles library ordering automatically (unlike Linux's ld.bfd)
    // so we can use simple rustc-link-lib directives
    for lib in ABSEIL_LIBS {
        println!("cargo:rustc-link-lib=static={}", lib);
    }
}
