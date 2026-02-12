{
  description = "Static ONNX Runtime v1.23.2 build with platform-specific accelerator support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      # Supported systems: Linux x86_64 (CUDA) and macOS aarch64 (CoreML)
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];

      forEachSystem =
        f:
        nixpkgs.lib.genAttrs systems (
          system:
          f {
            inherit system;
            pkgs = import nixpkgs {
              inherit system;
              config.allowUnfree = true; # Required for CUDA
              config.cudaSupport = system == "x86_64-linux";
            };
          }
        );

      onnxruntimeVersion = "1.23.2";

      # CUDA architectures: sm_80=Ampere, sm_86=Ampere consumer, sm_89=Ada, sm_90=Hopper
      # sm_100=Blackwell datacenter (B100/B200), sm_120=Blackwell consumer (RTX 50xx)
      # PTX forward compatibility means sm_90 works on Blackwell via JIT - but shared libs may not embed PTX
      cudaArchitectures = "90;120"; # Hopper + Blackwell consumer; use "80;86;89;90;120" for release

      # Platform detection
      mkPlatformInfo = system: {
        isLinux = system == "x86_64-linux";
        isDarwin = system == "aarch64-darwin";
      };

      # Compute accelerator flags from platform with optional overrides
      mkAcceleratorConfig =
        {
          system,
          enableCuda ? null,
          enableCoreml ? null,
        }:
        let
          inherit (mkPlatformInfo system) isLinux isDarwin;
          useCuda = (if enableCuda == null then isLinux else enableCuda) && isLinux;
          useCoreml = (if enableCoreml == null then isDarwin else enableCoreml) && isDarwin;
        in
        {
          inherit useCuda useCoreml;
          acceleratorSuffix =
            if useCuda then
              "-cuda"
            else if useCoreml then
              "-coreml"
            else
              "-cpu";
          acceleratorName =
            if useCuda then
              "CUDA"
            else if useCoreml then
              "CoreML"
            else
              "CPU only";
        };

      darwinFrameworks =
        pkgs: with pkgs.darwin.apple_sdk.frameworks; [
          Accelerate
          CoreML
          CoreVideo
          Foundation
          Metal
          MetalPerformanceShaders
        ];

      mkCudnnLibPath =
        { onnxruntime, cudaPackages }: "${onnxruntime}/lib/cudnn:${cudaPackages.cudnn.lib}/lib";

      # Shared vendored dependencies
      mkSharedDeps =
        pkgs:
        let
          inherit (pkgs) fetchFromGitHub fetchpatch applyPatches;
        in
        {
          mp11-src = fetchFromGitHub {
            name = "mp11-src";
            owner = "boostorg";
            repo = "mp11";
            tag = "boost-1.82.0";
            hash = "sha256-cLPvjkf2Au+B19PJNrUkTW/VPxybi1MpPxnIl4oo4/o=";
          };

          safeint-src = fetchFromGitHub {
            name = "safeint-src";
            owner = "dcleblanc";
            repo = "safeint";
            tag = "3.0.28";
            hash = "sha256-pjwjrqq6dfiVsXIhbBtbolhiysiFlFTnx5XcX77f+C0=";
          };

          onnx-src = applyPatches {
            name = "onnx-src";
            src = fetchFromGitHub {
              owner = "onnx";
              repo = "onnx";
              tag = "v1.18.0";
              hash = "sha256-UhtF+CWuyv5/Pq/5agLL4Y95YNP63W2BraprhRqJOag=";
            };
            patches = [
              (fetchpatch {
                url = "https://github.com/onnx/onnx/commit/595a069aaac07586f111681245bc808ee63551f8.patch";
                includes = [ "onnx/defs/schema.h" ];
                hash = "sha256-FFAJuJse4nmNT3ixvEdlqzbr3edY46SqEFv7z/oo6m0=";
              })
              (fetchpatch {
                url = "https://github.com/onnx/onnx/commit/6769c41ad64ebca0358da8c7211d2c6d0e627b2b.patch";
                hash = "sha256-VlTHs0om20kTNvSVQaasSsa5JROliQy4k9BECTsBtbU=";
              })
            ];
          };

          cutlass-src = fetchFromGitHub {
            name = "cutlass-src";
            owner = "NVIDIA";
            repo = "cutlass";
            tag = "v3.9.2";
            hash = "sha256-teziPNA9csYvhkG5t2ht8W8x5+1YGGbHm8VKx4JoxgI=";
          };

          dlpack-src = fetchFromGitHub {
            name = "dlpack-src";
            owner = "dmlc";
            repo = "dlpack";
            rev = "5c210da409e7f1e51ddf445134a4376fdbd70d7d";
            hash = "sha256-YqgzCyNywixebpHGx16tUuczmFS5pjCz5WjR89mv9eI=";
          };
        };

      mkOnnxruntime =
        {
          pkgs,
          system,
          enableCuda ? null,
          enableCoreml ? null,
          buildShared ? false, # Build shared library instead of static
        }:
        let
          inherit (pkgs)
            lib
            stdenv
            fetchFromGitHub
            fetchpatch
            fetchurl
            ;

          inherit (mkPlatformInfo system) isLinux isDarwin;
          accel = mkAcceleratorConfig { inherit system enableCuda enableCoreml; };
          inherit (accel) useCuda useCoreml acceleratorName;
          sharedDeps = mkSharedDeps pkgs;
          inherit (sharedDeps)
            mp11-src
            safeint-src
            onnx-src
            cutlass-src
            dlpack-src
            ;

          cudaPackages = pkgs.cudaPackages or { };
          effectiveStdenv = if useCuda then cudaPackages.backendStdenv else stdenv;

          # nvcc requires patched protobuf
          protobuf' =
            if useCuda then
              pkgs.protobuf.overrideAttrs (old: {
                patches = (old.patches or [ ]) ++ [
                  (fetchpatch {
                    name = "Workaround nvcc bug in message_lite.h";
                    url = "https://raw.githubusercontent.com/conda-forge/protobuf-feedstock/737a13ea0680484c08e8e0ab0144dab82c10c1b3/recipe/patches/0010-Workaround-nvcc-bug-in-message_lite.h.patch";
                    hash = "sha256-joK50Il4mrwIc6zuNW9gDIfOx9LuA4FlusJuzUf9kqI=";
                  })
                ];
              })
            else
              pkgs.protobuf;

        in
        effectiveStdenv.mkDerivation (finalAttrs: {
          # Package naming:
          # - onnxruntime-cpu: static CPU-only
          # - onnxruntime-cuda: static CUDA (default on Linux)
          # - onnxruntime-cuda-dyn: shared/dynamic CUDA
          # - onnxruntime-coreml: static CoreML (default on macOS)
          pname =
            "onnxruntime"
            + (if useCuda then "-cuda" else if useCoreml then "-coreml" else "-cpu")
            + (if buildShared then "-dyn" else "");
          version = onnxruntimeVersion;

          src = fetchFromGitHub {
            owner = "microsoft";
            repo = "onnxruntime";
            tag = "v${finalAttrs.version}";
            fetchSubmodules = true;
            hash = "sha256-hZ2L5+0Enkw4rGDKVpRECnKXP87w6Kbiyp6Fdxwt6hk=";
          };

          patches = [
            # GCC 15 compatibility
            (fetchpatch {
              url = "https://github.com/microsoft/onnxruntime/commit/d6e712c5b7b6260a61e54d1fe40107cf5366ee77.patch";
              hash = "sha256-FSuPybX8f2VoxvLhcYx4rdChaiK8bSUDR32sN3Efwfc=";
            })
            (fetchpatch {
              url = "https://github.com/microsoft/onnxruntime/commit/8ebd0bf1cf02414584d15d7244b07fa97d65ba02.patch";
              hash = "sha256-vX+kaFiNdmqWI91JELcLpoaVIHBb5EPbI7rCAMYAx04=";
            })
            (fetchurl {
              url = "https://gitlab.alpinelinux.org/alpine/aports/-/raw/462dfe0eb4b66948fe48de44545cc22bb64fdf9f/community/onnxruntime/0001-Remove-MATH_NO_EXCEPT-macro.patch";
              hash = "sha256-BdeGYevZExWWCuJ1lSw0Roy3h+9EbJgFF8qMwVxSn1A=";
            })
          ]
          ++ lib.optionals isLinux [
            ./patches/musl-execinfo.patch
            ./patches/musl-cstdint.patch
          ]
          # Static CUDA patches - only apply when building static library
          ++ lib.optionals (useCuda && !buildShared) [
            ./patches/cuda-static-provider.patch
            ./patches/providers-shared-static.patch
            ./patches/static-init-common.patch
            ./patches/static-init-bridge.patch
            ./patches/static-init-cpu.patch
            ./patches/static-init-provider.patch
            ./patches/static-datatype-loop.patch
            ./patches/static-tensorshape-loop.patch
          ];

          nativeBuildInputs = [
            pkgs.cmake
            pkgs.pkg-config
            pkgs.python3
            protobuf'
          ]
          ++ lib.optionals useCuda [
            cudaPackages.cuda_nvcc
            pkgs.removeReferencesTo
          ];

          buildInputs = [
            pkgs.eigen
            pkgs.glibcLocales
            pkgs.howard-hinnant-date
            pkgs.libpng
            pkgs.nlohmann_json
            pkgs.microsoft-gsl
            pkgs.zlib
          ]
          ++ lib.optionals (lib.meta.availableOn effectiveStdenv.hostPlatform pkgs.cpuinfo) [
            pkgs.cpuinfo
          ]
          ++ lib.optionals useCuda (
            with cudaPackages;
            [
              cuda_cccl
              libcublas
              libcublas.static
              libcurand
              libcurand.static
              libcusparse
              libcusparse.static
              libcufft
              libcufft.static
              cuda_cudart
              cudnn # No static version available
              cudnn-frontend
            ]
          )
          ++ lib.optionals isDarwin (darwinFrameworks pkgs ++ [ (pkgs.darwinMinVersionHook "13.3") ]);

          outputs = [
            "out"
            "dev"
          ];

          separateDebugInfo = true;
          enableParallelBuilding = true;

          cmakeDir = "../cmake";

          cmakeFlags = [
            (lib.cmakeBool "ABSL_ENABLE_INSTALL" true)
            (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
            (lib.cmakeBool "FETCHCONTENT_QUIET" false)
            (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_ABSEIL_CPP" "${pkgs.abseil-cpp_202407.src}")
            (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_DLPACK" "${dlpack-src}")
            (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_FLATBUFFERS" "${pkgs.flatbuffers_23.src}")
            (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_MP11" "${mp11-src}")
            (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_ONNX" "${onnx-src}")
            (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_RE2" "${pkgs.re2.src}")
            (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_SAFEINT" "${safeint-src}")
            (lib.cmakeFeature "FETCHCONTENT_TRY_FIND_PACKAGE_MODE" "ALWAYS")
            (lib.cmakeFeature "ONNX_CUSTOM_PROTOC_EXECUTABLE" (lib.getExe protobuf'))
            (lib.cmakeBool "onnxruntime_BUILD_SHARED_LIB" buildShared)
            (lib.cmakeBool "onnxruntime_BUILD_UNIT_TESTS" false)
            (lib.cmakeBool "onnxruntime_ENABLE_PYTHON" false)
            (lib.cmakeBool "onnxruntime_USE_FULL_PROTOBUF" false)
            (lib.cmakeBool "onnxruntime_USE_ROCM" false)
            (lib.cmakeBool "onnxruntime_USE_MIGRAPHX" false)
            (lib.cmakeBool "onnxruntime_USE_NCCL" false)
          ]
          ++ lib.optionals isLinux [
            "--compile-no-warning-as-error"
            (lib.cmakeBool "onnxruntime_ENABLE_LTO" false) # LTO incompatible with CUDA
          ]
          ++ lib.optionals useCuda [
            (lib.cmakeBool "onnxruntime_USE_CUDA" true)
            (lib.cmakeBool "onnxruntime_USE_COREML" false)
            (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_CUTLASS" "${cutlass-src}")
            (lib.cmakeFeature "onnxruntime_CUDNN_HOME" "${cudaPackages.cudnn}")
            (lib.cmakeFeature "CMAKE_CUDA_ARCHITECTURES" cudaArchitectures)
            (lib.cmakeFeature "onnxruntime_NVCC_THREADS" "1") # Avoid OOM
          ]
          # Static CUDA: define ORT_STATIC_PROVIDERS to skip conflicting template specializations
          ++ lib.optionals (useCuda && !buildShared) [
            "-DCMAKE_CXX_FLAGS=-DORT_STATIC_PROVIDERS=1"
          ]
          ++ lib.optionals useCoreml [
            (lib.cmakeBool "onnxruntime_USE_CUDA" false)
            (lib.cmakeBool "onnxruntime_USE_COREML" true)
            (lib.cmakeBool "onnxruntime_ENABLE_LTO" true)
          ]
          ++ lib.optionals (!useCuda && !useCoreml) [
            (lib.cmakeBool "onnxruntime_USE_CUDA" false)
            (lib.cmakeBool "onnxruntime_USE_COREML" false)
          ];

          env = lib.optionalAttrs isLinux {
            NIX_LDFLAGS = "-z,noexecstack";
          };

          postPatch = ''
            substituteInPlace cmake/libonnxruntime.pc.cmake.in \
              --replace-fail '$'{prefix}/@CMAKE_INSTALL_ @CMAKE_INSTALL_
          ''
          + lib.optionalString useCuda ''
            echo "find_package(cudnn_frontend REQUIRED)" > cmake/external/cudnn_frontend.cmake
          ''
          + ''
            substituteInPlace onnxruntime/core/platform/env.h --replace-fail \
              "GetRuntimePath() const { return PathString(); }" \
              "GetRuntimePath() const { return PathString(\"$out/lib/\"); }"
          '';

          postInstall = ''
            install -m644 -Dt $out/include \
              ../include/onnxruntime/core/framework/provider_options.h \
              ../include/onnxruntime/core/providers/cpu/cpu_provider_factory.h \
              ../include/onnxruntime/core/session/onnxruntime_*.h
          ''
          + lib.optionalString isLinux ''
            # Copy internal dependency libraries built via FETCHCONTENT
            mkdir -p $out/lib
            find _deps/onnx-build -name "libonnx*.a" -exec cp {} $out/lib/ \; 2>/dev/null || true
            find _deps/protobuf-build -name "libprotobuf*.a" -exec cp {} $out/lib/ \; 2>/dev/null || true
            find _deps/re2-build -name "libre2.a" -exec cp {} $out/lib/ \; 2>/dev/null || true
            find _deps/abseil_cpp-build -name "libabsl_*.a" -exec cp {} $out/lib/ \; 2>/dev/null || true
            find _deps/pytorch_cpuinfo-build -name "libcpuinfo*.a" -exec cp {} $out/lib/ \; 2>/dev/null || true
            find _deps/pytorch_clog-build -name "libclog*.a" -exec cp {} $out/lib/ \; 2>/dev/null || true
            find _deps/google_nsync-build -name "libnsync*.a" -exec cp {} $out/lib/ \; 2>/dev/null || true
          ''
          # Copy cuDNN shared libs for bundling (both static and shared CUDA builds)
          + lib.optionalString useCuda ''
            mkdir -p $out/lib/cudnn
            cp -L ${cudaPackages.cudnn.lib}/lib/libcudnn*.so* $out/lib/cudnn/ || true
          ''
          # Static CUDA build: device linking and archive merging
          + lib.optionalString (useCuda && !buildShared) ''
            # Device linking: ORT uses separable compilation (-dc) which requires nvcc -dlink
            DLINK_DIR=$(mktemp -d)
            pushd $DLINK_DIR
            ${pkgs.binutils}/bin/ar x $out/lib/libonnxruntime_providers_cuda.a
            CUDA_OBJS=$(find . -name "*.cu.o" -o -name "*cuda*.o" | grep -v dlink || true)
            if [ -n "$CUDA_OBJS" ]; then
              GENCODE_FLAGS=""
              FIRST_ARCH=""
              for arch in $(echo "${cudaArchitectures}" | tr ';' ' '); do
                [ -z "$FIRST_ARCH" ] && FIRST_ARCH="$arch"
                GENCODE_FLAGS="$GENCODE_FLAGS -gencode=arch=compute_$arch,code=sm_$arch"
              done
              ${cudaPackages.cuda_nvcc}/bin/nvcc -dlink -arch=sm_$FIRST_ARCH $GENCODE_FLAGS \
                -o cuda_device_link.o $CUDA_OBJS 2>&1 || true
              [ -f cuda_device_link.o ] && cp cuda_device_link.o $out/lib/
            fi
            popd
            rm -rf $DLINK_DIR

            # Merge all static libraries into unified archive
            pushd $out/lib
            cat > merge.mri << 'MRIEOF'
            CREATE libonnxruntime.a
            ADDLIB libonnxruntime_common.a
            ADDLIB libonnxruntime_flatbuffers.a
            ADDLIB libonnxruntime_framework.a
            ADDLIB libonnxruntime_graph.a
            ADDLIB libonnxruntime_lora.a
            ADDLIB libonnxruntime_mlas.a
            ADDLIB libonnxruntime_optimizer.a
            ADDLIB libonnxruntime_providers.a
            ADDLIB libonnxruntime_providers_shared.a
            ADDLIB libonnxruntime_providers_cuda.a
            ADDLIB libonnxruntime_session.a
            ADDLIB libonnxruntime_util.a
            SAVE
            END
            MRIEOF
            ${pkgs.binutils}/bin/ar -M < merge.mri
            [ -f cuda_device_link.o ] && ${pkgs.binutils}/bin/ar r libonnxruntime.a cuda_device_link.o
            ${pkgs.binutils}/bin/ranlib libonnxruntime.a
            rm merge.mri
            popd
          ''
          # CPU-only static: merge without CUDA provider libraries
          + lib.optionalString (isLinux && !useCuda && !buildShared) ''
            pushd $out/lib
            cat > merge.mri << 'MRIEOF'
            CREATE libonnxruntime.a
            ADDLIB libonnxruntime_common.a
            ADDLIB libonnxruntime_flatbuffers.a
            ADDLIB libonnxruntime_framework.a
            ADDLIB libonnxruntime_graph.a
            ADDLIB libonnxruntime_lora.a
            ADDLIB libonnxruntime_mlas.a
            ADDLIB libonnxruntime_optimizer.a
            ADDLIB libonnxruntime_providers.a
            ADDLIB libonnxruntime_session.a
            ADDLIB libonnxruntime_util.a
            SAVE
            END
            MRIEOF
            ${pkgs.binutils}/bin/ar -M < merge.mri
            ${pkgs.binutils}/bin/ranlib libonnxruntime.a
            rm merge.mri
            popd
          '';

          postFixup = lib.optionalString useCuda ''
            # Remove nvcc references from binaries (ELF files)
            find $out $dev -type f -exec remove-references-to -t "${lib.getBin cudaPackages.cuda_nvcc}" {} \; 2>/dev/null || true
            # Remove nvcc references from ALL text files in dev output
            find $dev -type f -exec grep -l "${lib.getBin cudaPackages.cuda_nvcc}" {} \; 2>/dev/null | while read f; do
              sed -i "s|${lib.getBin cudaPackages.cuda_nvcc}|/removed-nvcc-ref|g" "$f" 2>/dev/null || true
            done
          '';

          disallowedRequisites = lib.optionals useCuda [ (lib.getBin cudaPackages.cuda_nvcc) ];

          __structuredAttrs = true;

          requiredSystemFeatures = lib.optionals useCuda [ "big-parallel" ];

          passthru = {
            inherit
              isLinux
              isDarwin
              useCuda
              useCoreml
              buildShared
              ;
          };

          meta = {
            description = "ONNX Runtime library (${acceleratorName}, ${if buildShared then "dynamic" else "static"} linking)";
            homepage = "https://github.com/microsoft/onnxruntime";
            license = lib.licenses.mit;
            platforms = [ system ];
          };
        });

      mkSqueezeNet =
        pkgs:
        pkgs.fetchurl {
          url = "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-7.onnx";
          hash = "sha256-2V4hkeBW8ZEqm49gANo7nHgYRBuetIE3Azwq2/Y5i8g=";
        };

      # Shared env for ort-wrapper package and dev shells
      mkOrtWrapperEnv =
        {
          pkgs,
          system,
          onnxruntime,
          useCuda,
          useCoreml,
          squeezenet-model,
        }:
        let
          inherit (pkgs) lib;
          inherit (mkPlatformInfo system) isDarwin;
          cudaPackages = pkgs.cudaPackages or { };
          cudnnLibPath = mkCudnnLibPath { inherit onnxruntime cudaPackages; };
        in
        {
          buildInputs = [
            onnxruntime
            pkgs.protobuf
            pkgs.re2
            pkgs.cpuinfo
          ]
          ++ lib.optionals useCuda [
            cudaPackages.cudnn
            cudaPackages.cuda_cudart
            cudaPackages.libcublas
            cudaPackages.libcurand
            cudaPackages.libcusparse
            cudaPackages.libcufft
          ]
          ++ lib.optionals isDarwin (darwinFrameworks pkgs);

          envVars = {
            ORT_LIB_LOCATION = "${onnxruntime}/lib";
            ONNX_TEST_MODEL = squeezenet-model;
          };

          shellHook = lib.optionalString useCuda ''
            export LD_LIBRARY_PATH="${cudnnLibPath}:$LD_LIBRARY_PATH"
            export CARGO_BUILD_RUSTFLAGS="-C link-arg=-Wl,-rpath,${onnxruntime}/lib/cudnn"
          '';

          preCheck = lib.optionalString useCuda ''
            export LD_LIBRARY_PATH="${cudnnLibPath}:$LD_LIBRARY_PATH"
          '';

          cargoFeatures = lib.concatStringsSep "," (
            lib.optional useCuda "cuda" ++ lib.optional useCoreml "coreml"
          );

          inherit
            squeezenet-model
            cudnnLibPath
            useCuda
            useCoreml
            ;
        };

      mkOrtWrapper =
        {
          pkgs,
          system,
          squeezenet-model,
          enableCuda ? null,
          enableCoreml ? null,
        }:
        let
          inherit (pkgs) lib;
          accel = mkAcceleratorConfig { inherit system enableCuda enableCoreml; };
          inherit (accel)
            useCuda
            useCoreml
            acceleratorSuffix
            acceleratorName
            ;
          onnxruntime = mkOnnxruntime {
            inherit
              pkgs
              system
              enableCuda
              enableCoreml
              ;
          };
          env = mkOrtWrapperEnv {
            inherit
              pkgs
              system
              onnxruntime
              useCuda
              useCoreml
              squeezenet-model
              ;
          };
        in
        pkgs.rustPlatform.buildRustPackage {
          pname = "ort-wrapper" + acceleratorSuffix;
          version = "0.1.0";
          src = ./ort-wrapper;
          cargoLock.lockFile = ./ort-wrapper/Cargo.lock;

          cargoBuildFlags = lib.optionals (env.cargoFeatures != "") [
            "--features"
            env.cargoFeatures
          ];

          inherit (env.envVars) ORT_LIB_LOCATION ONNX_TEST_MODEL;
          nativeBuildInputs = [ pkgs.pkg-config ];
          inherit (env) buildInputs preCheck;
          doCheck = false;

          meta = {
            description = "ONNX Runtime wrapper with static linking (${acceleratorName})";
            platforms = [ system ];
            mainProgram = "ort-wrapper";
          };
        };

      # Pre-compute all shared values once per system
      mkSystemContext =
        { pkgs, system }:
        let
          inherit (pkgs) lib;
          inherit (mkPlatformInfo system) isLinux isDarwin;
          cudaPackages = pkgs.cudaPackages or { };
          squeezenet-model = mkSqueezeNet pkgs;

          defaultAccel = mkAcceleratorConfig { inherit system; };
          cpuAccel = mkAcceleratorConfig {
            inherit system;
            enableCuda = false;
            enableCoreml = false;
          };

          onnxruntime = mkOnnxruntime { inherit pkgs system; };
          onnxruntimeCpu = mkOnnxruntime {
            inherit pkgs system;
            enableCuda = false;
            enableCoreml = false;
          };
          # Shared library builds (for dynamic linking)
          onnxruntimeCudaDyn = mkOnnxruntime {
            inherit pkgs system;
            buildShared = true;
          };

          ortWrapperDefault = mkOrtWrapper { inherit pkgs system squeezenet-model; };
          ortWrapperCpu = mkOrtWrapper {
            inherit pkgs system squeezenet-model;
            enableCuda = false;
            enableCoreml = false;
          };

          # Dynamic CUDA wrapper (uses shared libonnxruntime.so)
          ortWrapperCudaDyn =
            if isLinux then
              pkgs.rustPlatform.buildRustPackage {
                pname = "ort-wrapper-cuda-dyn";
                version = "0.1.0";
                src = ./ort-wrapper;
                cargoLock.lockFile = ./ort-wrapper/Cargo.lock;

                cargoBuildFlags = [
                  "--features"
                  "cuda-dyn"
                ];

                ORT_DYLIB_PATH = "${onnxruntimeCudaDyn}/lib";
                ONNX_TEST_MODEL = squeezenet-model;
                nativeBuildInputs = [ pkgs.pkg-config ];
                buildInputs = [
                  onnxruntimeCudaDyn
                  pkgs.protobuf
                  pkgs.re2
                  cudaPackages.cudnn
                  cudaPackages.cuda_cudart
                  cudaPackages.libcublas
                  cudaPackages.libcurand
                  cudaPackages.libcusparse
                  cudaPackages.libcufft
                ];
                doCheck = false;

                meta = {
                  description = "ONNX Runtime wrapper with dynamic CUDA linking";
                  platforms = [ system ];
                  mainProgram = "ort-wrapper";
                };
              }
            else
              null;

          # Env config for dynamic CUDA builds
          envCudaDyn =
            if isLinux then
              let
                cudnnLibPath = mkCudnnLibPath {
                  onnxruntime = onnxruntimeCudaDyn;
                  inherit cudaPackages;
                };
              in
              {
                useCuda = true;
                useCoreml = false;
                inherit squeezenet-model cudnnLibPath;
                onnxruntimeLibPath = "${onnxruntimeCudaDyn}/lib";
                buildInputs = [
                  onnxruntimeCudaDyn
                  pkgs.protobuf
                  pkgs.re2
                  cudaPackages.cudnn
                  cudaPackages.cuda_cudart
                  cudaPackages.libcublas
                  cudaPackages.libcurand
                  cudaPackages.libcusparse
                  cudaPackages.libcufft
                ];
                envVars = {
                  ORT_DYLIB_PATH = "${onnxruntimeCudaDyn}/lib";
                  ONNX_TEST_MODEL = squeezenet-model;
                };
                shellHook = ''
                  export LD_LIBRARY_PATH="${onnxruntimeCudaDyn}/lib:${cudnnLibPath}:$LD_LIBRARY_PATH"
                '';
              }
            else
              null;

          envDefault = mkOrtWrapperEnv {
            inherit
              pkgs
              system
              onnxruntime
              squeezenet-model
              ;
            inherit (defaultAccel) useCuda useCoreml;
          };
          envCpu = mkOrtWrapperEnv {
            inherit pkgs system squeezenet-model;
            onnxruntime = onnxruntimeCpu;
            inherit (cpuAccel) useCuda useCoreml;
          };

          mkApp =
            { package, env }:
            let
              wrapper = pkgs.writeShellScriptBin "ort-wrapper" (
                lib.optionalString env.useCuda ''
                  export LD_LIBRARY_PATH="${env.cudnnLibPath}:''${LD_LIBRARY_PATH:-}"
                ''
                + ''
                  export ONNX_TEST_MODEL="${squeezenet-model}"
                  exec "${package}/bin/ort-wrapper" "$@"
                ''
              );
            in
            {
              type = "app";
              program = "${wrapper}/bin/ort-wrapper";
            };

          devPackages = [
            pkgs.cargo
            pkgs.rustc
            pkgs.rust-analyzer
            pkgs.pkg-config
          ];

          mkShellHook =
            { env, accel }:
            let
              featureHint =
                if env.useCuda then
                  " --features cuda"
                else if env.useCoreml then
                  " --features coreml"
                else
                  "";
            in
            env.shellHook
            + ''
              echo "ONNX Runtime development shell (${accel.acceleratorName})"
              echo "  ORT_LIB_LOCATION: $ORT_LIB_LOCATION"
              echo "  ONNX_TEST_MODEL: $ONNX_TEST_MODEL"
              echo ""
              echo "Commands:"
              echo "  cd ort-wrapper && cargo build${featureHint}    # Build"
              echo "  cd ort-wrapper && cargo test${featureHint}     # Run tests"
              echo "  cd ort-wrapper && cargo bench${featureHint}    # Run benchmarks"
            ''
            + lib.optionalString isLinux ''
              echo ""
              echo "For CPU-only development, use: nix develop .#cpu"
            '';

          mkDevShell =
            {
              name,
              env,
              accel,
            }:
            pkgs.mkShell {
              inherit name;
              packages = devPackages;
              inherit (env) buildInputs;
              inherit (env.envVars) ORT_LIB_LOCATION ONNX_TEST_MODEL;
              shellHook = mkShellHook { inherit env accel; };
            };
        in
        {
          inherit
            lib
            isLinux
            isDarwin
            cudaPackages
            squeezenet-model
            defaultAccel
            cpuAccel
            onnxruntime
            onnxruntimeCpu
            onnxruntimeCudaDyn
            ortWrapperDefault
            ortWrapperCpu
            ortWrapperCudaDyn
            envDefault
            envCpu
            envCudaDyn
            mkApp
            mkDevShell
            ;
        };

    in
    {
      packages = forEachSystem (
        { pkgs, system }:
        let
          ctx = mkSystemContext { inherit pkgs system; };
        in
        {
          default = ctx.onnxruntime;
          inherit (ctx) squeezenet-model;
        }
        // ctx.lib.optionalAttrs ctx.isLinux {
          onnxruntime-cpu = ctx.onnxruntimeCpu;
          onnxruntime-cuda-dyn = ctx.onnxruntimeCudaDyn;
          ort-wrapper-cuda = ctx.ortWrapperDefault;
          ort-wrapper-cpu = ctx.ortWrapperCpu;
          ort-wrapper-cuda-dyn = ctx.ortWrapperCudaDyn;
        }
        // ctx.lib.optionalAttrs ctx.isDarwin {
          ort-wrapper-coreml = ctx.ortWrapperDefault;
        }
      );

      checks = forEachSystem (
        { pkgs, system }:
        let
          ctx = mkSystemContext { inherit pkgs system; };
        in
        {
          ort-wrapper =
            pkgs.runCommand "ort-wrapper-check"
              {
                nativeBuildInputs = [ ctx.ortWrapperDefault ];
                ONNX_TEST_MODEL = ctx.squeezenet-model;
              }
              (
                ''
                  export HOME=$TMPDIR
                ''
                + ctx.lib.optionalString ctx.envDefault.useCuda ''
                  export LD_LIBRARY_PATH="${ctx.envDefault.cudnnLibPath}:$LD_LIBRARY_PATH"
                ''
                + ''
                  ort-wrapper
                  touch $out
                ''
              );
        }
      );

      devShells = forEachSystem (
        { pkgs, system }:
        let
          ctx = mkSystemContext { inherit pkgs system; };
        in
        {
          default = ctx.mkDevShell {
            name = "onnxruntime-dev";
            env = ctx.envDefault;
            accel = ctx.defaultAccel;
          };
        }
        // ctx.lib.optionalAttrs ctx.isLinux {
          cpu = ctx.mkDevShell {
            name = "onnxruntime-dev-cpu";
            env = ctx.envCpu;
            accel = ctx.cpuAccel;
          };
          cuda-dyn = pkgs.mkShell {
            name = "onnxruntime-dev-cuda-dyn";
            packages = [
              pkgs.cargo
              pkgs.rustc
              pkgs.rust-analyzer
              pkgs.pkg-config
            ];
            inherit (ctx.envCudaDyn) buildInputs;
            inherit (ctx.envCudaDyn.envVars) ORT_DYLIB_PATH ONNX_TEST_MODEL;
            shellHook =
              ctx.envCudaDyn.shellHook
              + ''
                echo "ONNX Runtime development shell (Dynamic CUDA)"
                echo "  ORT_DYLIB_PATH: $ORT_DYLIB_PATH"
                echo "  ONNX_TEST_MODEL: $ONNX_TEST_MODEL"
                echo ""
                echo "Commands:"
                echo "  cd ort-wrapper && cargo build --features cuda-dyn    # Build"
                echo "  cd ort-wrapper && cargo test --features cuda-dyn     # Run tests"
              '';
          };
        }
      );

      apps = forEachSystem (
        { pkgs, system }:
        let
          ctx = mkSystemContext { inherit pkgs system; };
        in
        ctx.lib.optionalAttrs ctx.isLinux {
          ort-wrapper-cuda = ctx.mkApp {
            package = ctx.ortWrapperDefault;
            env = ctx.envDefault;
          };
          ort-wrapper-cpu = ctx.mkApp {
            package = ctx.ortWrapperCpu;
            env = ctx.envCpu;
          };
          ort-wrapper-cuda-dyn =
            let
              wrapper = pkgs.writeShellScriptBin "ort-wrapper" ''
                export LD_LIBRARY_PATH="${ctx.envCudaDyn.onnxruntimeLibPath}:${ctx.envCudaDyn.cudnnLibPath}:''${LD_LIBRARY_PATH:-}"
                export ONNX_TEST_MODEL="${ctx.squeezenet-model}"
                exec "${ctx.ortWrapperCudaDyn}/bin/ort-wrapper" "$@"
              '';
            in
            {
              type = "app";
              program = "${wrapper}/bin/ort-wrapper";
            };
          # CPU for faster iteration
          default = ctx.mkApp {
            package = ctx.ortWrapperCpu;
            env = ctx.envCpu;
          };

          # Debug: auto-interrupt GDB backtrace
          # Usage: nix run .#debug-bt [seconds]
          debug-bt =
            let
              wrapper = pkgs.writeShellScriptBin "debug-ort-bt" ''
                DELAY=''${1:-2}
                export LD_LIBRARY_PATH="${ctx.envDefault.cudnnLibPath}:''${LD_LIBRARY_PATH:-}"
                export ONNX_TEST_MODEL="${ctx.squeezenet-model}"
                echo "=== Static CUDA Backtrace (auto-interrupt after ''${DELAY}s) ==="
                echo ""
                # Start GDB in background, get its PID, send SIGINT after delay
                ${pkgs.gdb}/bin/gdb \
                  -ex "set pagination off" \
                  -ex "run" \
                  -ex "bt" \
                  -ex "thread apply all bt" \
                  -ex "quit" \
                  "${ctx.ortWrapperDefault}/bin/ort-wrapper" &
                GDB_PID=$!
                sleep "$DELAY"
                kill -INT $GDB_PID 2>/dev/null || true
                wait $GDB_PID 2>/dev/null
              '';
            in
            {
              type = "app";
              program = "${wrapper}/bin/debug-ort-bt";
            };

          # Debug: interactive GDB session
          debug-gdb =
            let
              wrapper = pkgs.writeShellScriptBin "debug-ort-gdb" ''
                export LD_LIBRARY_PATH="${ctx.envDefault.cudnnLibPath}:''${LD_LIBRARY_PATH:-}"
                export ONNX_TEST_MODEL="${ctx.squeezenet-model}"
                echo "=== Static CUDA GDB Session ==="
                echo "When it hangs, press Ctrl+C then: bt, thread apply all bt"
                echo ""
                exec ${pkgs.gdb}/bin/gdb -ex "set pagination off" -ex run "${ctx.ortWrapperDefault}/bin/ort-wrapper"
              '';
            in
            {
              type = "app";
              program = "${wrapper}/bin/debug-ort-gdb";
            };

          # Build monitor: reliable build output capture
          # Usage: nix run .#build-cuda
          # Captures --print-build-logs and stderr to show real progress
          build-cuda =
            let
              wrapper = pkgs.writeShellScriptBin "build-cuda" ''
                echo "=== Building Static CUDA (with logs) ==="
                echo "Using: nix build .#ort-wrapper-cuda --print-build-logs 2>&1"
                echo ""
                exec ${pkgs.nix}/bin/nix build ${builtins.toString ./.}#ort-wrapper-cuda --print-build-logs 2>&1
              '';
            in
            {
              type = "app";
              program = "${wrapper}/bin/build-cuda";
            };
        }
        // ctx.lib.optionalAttrs ctx.isDarwin {
          ort-wrapper-coreml = ctx.mkApp {
            package = ctx.ortWrapperDefault;
            env = ctx.envDefault;
          };
          default = ctx.mkApp {
            package = ctx.ortWrapperDefault;
            env = ctx.envDefault;
          };
        }
      );

      # Debug tools for static CUDA hang investigation
      debug = forEachSystem (
        { pkgs, system }:
        let
          ctx = mkSystemContext { inherit pkgs system; };
        in
        ctx.lib.optionalAttrs ctx.isLinux {
          # Run static CUDA wrapper under GDB
          # Usage: nix run .#debug.gdb-cuda
          # When it hangs, press Ctrl+C then type "bt" for backtrace
          gdb-cuda =
            let
              wrapper = pkgs.writeShellScriptBin "debug-ort-gdb" ''
                export LD_LIBRARY_PATH="${ctx.envDefault.cudnnLibPath}:''${LD_LIBRARY_PATH:-}"
                export ONNX_TEST_MODEL="${ctx.squeezenet-model}"
                echo "=== Static CUDA Debug Session ==="
                echo "When it hangs, press Ctrl+C then type:"
                echo "  bt        - show backtrace"
                echo "  bt full   - show backtrace with locals"
                echo "  info threads - list all threads"
                echo "  thread N  - switch to thread N"
                echo ""
                exec ${pkgs.gdb}/bin/gdb -ex "set pagination off" -ex run --args "${ctx.ortWrapperDefault}/bin/ort-wrapper" "$@"
              '';
            in
            {
              type = "app";
              program = "${wrapper}/bin/debug-ort-gdb";
            };

          # Run static CUDA wrapper under strace to see syscalls
          # Usage: nix run .#debug.strace-cuda
          strace-cuda =
            let
              wrapper = pkgs.writeShellScriptBin "debug-ort-strace" ''
                export LD_LIBRARY_PATH="${ctx.envDefault.cudnnLibPath}:''${LD_LIBRARY_PATH:-}"
                export ONNX_TEST_MODEL="${ctx.squeezenet-model}"
                echo "=== Static CUDA Strace Session ==="
                echo "Tracing syscalls (futex calls highlighted)..."
                echo ""
                exec ${pkgs.strace}/bin/strace -f -e trace=futex,write,nanosleep,clock_nanosleep \
                  "${ctx.ortWrapperDefault}/bin/ort-wrapper" "$@" 2>&1 | head -500
              '';
            in
            {
              type = "app";
              program = "${wrapper}/bin/debug-ort-strace";
            };

          # Run static CUDA wrapper under perf to sample where it's spinning
          # Usage: nix run .#debug.perf-cuda
          perf-cuda =
            let
              wrapper = pkgs.writeShellScriptBin "debug-ort-perf" ''
                export LD_LIBRARY_PATH="${ctx.envDefault.cudnnLibPath}:''${LD_LIBRARY_PATH:-}"
                export ONNX_TEST_MODEL="${ctx.squeezenet-model}"
                echo "=== Static CUDA Perf Session ==="
                echo "Recording for 5 seconds (Ctrl+C to stop early)..."
                echo "Then run: perf report"
                echo ""
                ${pkgs.linuxPackages.perf}/bin/perf record -g --call-graph dwarf -o perf.data -- \
                  timeout 5 "${ctx.ortWrapperDefault}/bin/ort-wrapper" "$@" || true
                echo ""
                echo "Perf data saved to perf.data"
                echo "Run: ${pkgs.linuxPackages.perf}/bin/perf report -i perf.data"
              '';
            in
            {
              type = "app";
              program = "${wrapper}/bin/debug-ort-perf";
            };

          # Quick spin detector - samples /proc/pid/stack repeatedly
          spin-cuda =
            let
              wrapper = pkgs.writeShellScriptBin "debug-ort-spin" ''
                export LD_LIBRARY_PATH="${ctx.envDefault.cudnnLibPath}:''${LD_LIBRARY_PATH:-}"
                export ONNX_TEST_MODEL="${ctx.squeezenet-model}"
                echo "=== Static CUDA Spin Detector ==="
                echo "Starting wrapper in background, sampling stack..."
                echo ""
                "${ctx.ortWrapperDefault}/bin/ort-wrapper" "$@" &
                PID=$!
                sleep 0.5
                for i in 1 2 3 4 5; do
                  echo "--- Sample $i (PID $PID) ---"
                  cat /proc/$PID/stack 2>/dev/null || echo "(process exited)"
                  sleep 0.5
                done
                kill $PID 2>/dev/null || true
              '';
            in
            {
              type = "app";
              program = "${wrapper}/bin/debug-ort-spin";
            };

          # Auto-interrupt GDB - runs for N seconds then captures backtrace
          # Usage: nix run .#debug.bt-cuda [seconds]
          bt-cuda =
            let
              wrapper = pkgs.writeShellScriptBin "debug-ort-bt" ''
                DELAY=''${1:-2}
                export LD_LIBRARY_PATH="${ctx.envDefault.cudnnLibPath}:''${LD_LIBRARY_PATH:-}"
                export ONNX_TEST_MODEL="${ctx.squeezenet-model}"
                echo "=== Static CUDA Backtrace (auto-interrupt after ''${DELAY}s) ==="
                echo ""

                # Self-interrupt technique: fork a process that sends SIGINT after delay
                SELF=$$
                ( sleep "$DELAY" ; kill -INT $SELF ) &
                KILLER=$!

                # GDB will receive the SIGINT, stop the inferior, and print backtrace
                ${pkgs.gdb}/bin/gdb \
                  -ex "set pagination off" \
                  -ex "run" \
                  -ex "bt" \
                  -ex "thread apply all bt" \
                  -ex "quit" \
                  "${ctx.ortWrapperDefault}/bin/ort-wrapper"

                kill $KILLER 2>/dev/null || true
              '';
            in
            {
              type = "app";
              program = "${wrapper}/bin/debug-ort-bt";
            };
        }
      );

      formatter = forEachSystem ({ pkgs, ... }: pkgs.nixfmt-rfc-style);
    };
}
