# Vendored (with light trimming) from jaxlib/cpu in the jax-ml/jax repo at the
# jax-v0.9.0.1 tag, so that JAX-generated HLO relying on LAPACK (e.g.
# `jnp.linalg.inv`, `jnp.linalg.solve`) can run against our standalone PJRT
# CPU plugin. See jaxlib/cpu/cpu_kernels.cc upstream: "This file is not used
# by JAX itself, but exists to assist with running JAX-generated HLO code
# from outside of JAX."
#
# Not part of upstream XLA; added by call_jax_from_cpp.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ffi_helpers",
    hdrs = ["ffi_helpers.h"],
    deps = [
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:ffi",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "lapack_kernels",
    srcs = ["lapack_kernels.cc"],
    hdrs = ["lapack_kernels.h"],
    # -fexceptions matches jaxlib's own build of these sources.
    # -Wno-c++20-extensions is needed because XLA compiles with -Werror while
    # jaxlib does not, and lapack_kernels.cc captures structured bindings in
    # lambdas -- a C++20 feature that clang accepts as an extension under the
    # -std=c++17 both projects use. Suppressing the warning reproduces jaxlib's
    # effective behaviour without changing the language standard.
    copts = [
        "-fexceptions",
        "-Wno-c++20-extensions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":ffi_helpers",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:ffi",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform",
    ],
)

# Binds the FFI kernel function pointers directly to a system LAPACK/BLAS
# (see lapack_kernels_using_lapack.cc's own comment: "a C++ user should link
# against LAPACK directly").
cc_library(
    name = "lapack_kernels_using_lapack",
    srcs = ["lapack_kernels_using_lapack.cc"],
    deps = [":lapack_kernels"],
    alwayslink = 1,
)

# Registers the LAPACK-backed kernels above as FFI custom-call handlers for
# the "Host" platform. Linking this into a binary/shared library is enough:
# registration happens via static initializers.
cc_library(
    name = "cpu_kernels",
    srcs = ["cpu_kernels.cc"],
    deps = [
        ":lapack_kernels",
        ":lapack_kernels_using_lapack",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:ffi",
    ],
    alwayslink = 1,
)
