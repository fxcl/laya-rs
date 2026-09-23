//! Device resolution with the same fallback behaviour as the Python runtime.
//!
//! `laya/agent.py` picks cuda > mps > cpu when no device is requested, and falls back to
//! CPU (with a warning) when an explicitly requested GPU is unavailable or runs out of
//! memory. This module mirrors that, gated on the candle backend features.

use candle_core::Device;

use crate::LayaError;

/// What the caller asked for, before resolution.
#[derive(Clone, Debug, Default)]
pub enum DeviceRequest {
    /// No device given: auto-detect cuda > metal > cpu.
    #[default]
    Auto,
    /// An explicit device string such as "cpu", "cuda", "cuda:1" or "metal".
    Explicit(String),
}

/// A resolved device plus whether a fallback happened (for the warning).
#[derive(Clone, Debug)]
pub struct ResolvedDevice {
    pub device: Device,
    /// Set when an explicitly requested GPU was unavailable and CPU was used instead.
    pub fell_back_from: Option<String>,
    /// Why the requested device was rejected, when it was probed rather than merely absent.
    pub probe_error: Option<String>,
}

/// Run one trivial kernel on `device` to confirm it can actually execute.
fn probe(device: &Device) -> Result<(), String> {
    if device.is_cpu() {
        return Ok(());
    }
    let a = candle_core::Tensor::zeros((2usize, 2usize), candle_core::DType::F32, device)
        .map_err(|e| e.to_string())?;
    let b = a.matmul(&a).map_err(|e| e.to_string())?;
    b.sum_all().map_err(|e| e.to_string())?;
    Ok(())
}

/// Resolve a device request, mirroring the Python precedence and fallback.
pub fn resolve(request: &DeviceRequest) -> Result<ResolvedDevice, LayaError> {
    match request {
        DeviceRequest::Auto => Ok(ResolvedDevice {
            device: auto_device(),
            fell_back_from: None,
            probe_error: None,
        }),
        DeviceRequest::Explicit(name) => {
            let lower = name.trim().to_lowercase();
            let kind = lower.split(':').next().unwrap_or("");
            let available = match kind {
                "cpu" => true,
                "cuda" => cfg!(feature = "cuda") && cuda_available(),
                "metal" | "mps" => cfg!(feature = "metal") && metal_available(),
                other => {
                    return Err(LayaError::DeviceUnavailable(format!(
                        "unknown device {:?}; expected cpu, cuda or metal",
                        other
                    )))
                }
            };
            if available {
                let device = match kind {
                    "cpu" => Device::Cpu,
                    "cuda" => Device::new_cuda(ordinal(&lower)).map_err(device_err)?,
                    _ => Device::new_metal(ordinal(&lower)).map_err(device_err)?,
                };
                // A device can be created but still be unable to run candle's kernels
                // (observed on an AMD RX 6600 under Metal: "Failed to create pipeline").
                // Probe it with a trivial matmul so the failure surfaces as the usual
                // CPU fallback instead of an error halfway through the first forward pass.
                match probe(&device) {
                    Ok(()) => Ok(ResolvedDevice {
                        device,
                        fell_back_from: None,
                        probe_error: None,
                    }),
                    Err(why) => Ok(ResolvedDevice {
                        device: Device::Cpu,
                        fell_back_from: Some(name.clone()),
                        probe_error: Some(why),
                    }),
                }
            } else {
                // Python prints a warning and continues on CPU rather than failing.
                Ok(ResolvedDevice {
                    device: Device::Cpu,
                    fell_back_from: Some(name.clone()),
                    probe_error: None,
                })
            }
        }
    }
}

/// cuda > metal > cpu, exactly like the Python auto-detection order.
fn auto_device() -> Device {
    if cfg!(feature = "cuda") && cuda_available() {
        if let Ok(d) = Device::new_cuda(0) {
            if probe(&d).is_ok() {
                return d;
            }
        }
    }
    if cfg!(feature = "metal") && metal_available() {
        if let Ok(d) = Device::new_metal(0) {
            if probe(&d).is_ok() {
                return d;
            }
        }
    }
    Device::Cpu
}

fn ordinal(name: &str) -> usize {
    name.split(':')
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    Device::cuda_if_available(0).is_ok()
}

#[cfg(not(feature = "cuda"))]
fn cuda_available() -> bool {
    false
}

#[cfg(feature = "metal")]
fn metal_available() -> bool {
    Device::metal_if_available(0).is_ok()
}

#[cfg(not(feature = "metal"))]
fn metal_available() -> bool {
    false
}

fn device_err(e: candle_core::Error) -> LayaError {
    LayaError::DeviceUnavailable(e.to_string())
}

/// The warning the Python runtime prints when it has to fall back to CPU.
pub fn fallback_warning(from: &str, why: &str) -> String {
    format!(
        "\n[laya] Warning: could not place the model on {}, so it is running on CPU.\n  \
         Reason: {}\n  Inference will be roughly 10-15x slower (~200-500 ms rather than ~35 ms).\n",
        from, why
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_always_available() {
        let r = resolve(&DeviceRequest::Explicit("cpu".to_string())).unwrap();
        assert!(matches!(r.device, Device::Cpu));
        assert!(r.fell_back_from.is_none());
        assert!(r.probe_error.is_none());
    }

    #[test]
    fn probe_accepts_cpu() {
        assert!(probe(&Device::Cpu).is_ok());
    }

    #[test]
    fn auto_detect_never_fails() {
        let r = resolve(&DeviceRequest::Auto).unwrap();
        assert!(r.fell_back_from.is_none());
        // on a CPU-only build this is CPU; with a backend feature it may be a GPU
        assert!(matches!(r.device, Device::Cpu) || !r.device.is_cpu());
    }

    #[test]
    fn unknown_device_is_an_error() {
        let err = resolve(&DeviceRequest::Explicit("tpu".to_string())).unwrap_err();
        assert!(matches!(err, LayaError::DeviceUnavailable(_)));
    }

    #[test]
    fn unavailable_backend_falls_back_to_cpu() {
        // "cuda" without the cuda feature must fall back rather than fail
        if !cfg!(feature = "cuda") {
            let r = resolve(&DeviceRequest::Explicit("cuda".to_string())).unwrap();
            assert!(matches!(r.device, Device::Cpu));
            assert_eq!(r.fell_back_from.as_deref(), Some("cuda"));
        }
    }

    #[test]
    fn ordinal_parsing() {
        assert_eq!(ordinal("cuda"), 0);
        assert_eq!(ordinal("cuda:2"), 2);
        assert_eq!(ordinal("metal:1"), 1);
    }
}
