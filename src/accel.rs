//! ONNX Runtime execution-provider selection, shared by the embedder and the
//! reranker. An empty provider list makes fastembed use the CPU provider; with
//! the `gpu` feature we prepend CUDA. Accelerated builds require provider
//! registration to succeed; ONNX Runtime can still place unsupported nodes on
//! its default CPU provider.
use fastembed::ExecutionProviderDispatch;

#[cfg(any(
    all(feature = "gpu", feature = "directml"),
    all(feature = "gpu", feature = "coreml"),
    all(feature = "directml", feature = "coreml")
))]
compile_error!("features `gpu`, `directml`, and `coreml` are mutually exclusive");

#[cfg(feature = "gpu")]
pub fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    vec![ort::ep::CUDA::default().build().error_on_failure()]
}

#[cfg(feature = "directml")]
pub fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    vec![ort::ep::DirectML::default().build().error_on_failure()]
}

#[cfg(feature = "coreml")]
pub fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    vec![ort::ep::CoreML::default().build().error_on_failure()]
}

#[cfg(not(any(feature = "gpu", feature = "directml", feature = "coreml")))]
pub fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    Vec::new()
}

/// Accelerated builds prefer FP32 ONNX exports; CPU builds prefer quantized
/// exports optimized for the host architecture.
pub const fn prefers_fp32_onnx() -> bool {
    cfg!(any(
        feature = "gpu",
        feature = "directml",
        feature = "coreml"
    ))
}

/// Human-readable label of the configured provider. Session creation fails if
/// registration fails, while individual unsupported nodes may still use CPU.
#[cfg(feature = "gpu")]
pub fn provider_name() -> &'static str {
    "cuda (strict registration; per-node CPU fallback possible)"
}

#[cfg(feature = "directml")]
pub fn provider_name() -> &'static str {
    "directml (strict registration; per-node CPU fallback possible)"
}

#[cfg(feature = "coreml")]
pub fn provider_name() -> &'static str {
    "coreml (strict registration; per-node CPU fallback possible)"
}

#[cfg(not(any(feature = "gpu", feature = "directml", feature = "coreml")))]
pub fn provider_name() -> &'static str {
    "cpu"
}
