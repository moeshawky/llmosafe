//! Python bindings for llmosafe using PyO3.
//!
//! These bindings provide a Pythonic interface to the llmosafe safety primitives.
//! All functions are thread-safe and GIL-independent where possible.

use pyo3::prelude::*;
use pyo3::create_exception;
use llmosafe::llmosafe_body::ResourceGuard;
use llmosafe::llmosafe_kernel::{KernelError, Synapse};
use llmosafe::llmosafe_sifter::calculate_halo_signal;

// Custom exceptions
create_exception!(llmosafe, LLMOSafeError, pyo3::exceptions::PyException);
create_exception!(llmosafe, ResourceExhaustedError, LLMOSafeError);
create_exception!(llmosafe, CognitiveInstabilityError, LLMOSafeError);
create_exception!(llmosafe, BiasHaloDetectedError, LLMOSafeError);

/// Calculate the "halo signal" (bias score) for a given text.
#[pyfunction]
fn calculate_halo(text: &str) -> u16 {
    calculate_halo_signal(text)
}

/// Check if the current resource usage is within the specified ceiling.
#[pyfunction]
fn check_resources(ceiling_mb: u32) -> PyResult<i32> {
    let ceiling_bytes = (ceiling_mb as usize).saturating_mul(1024 * 1024);
    let guard = ResourceGuard::new(ceiling_bytes);
    
    match guard.check() {
        Ok(_) => Ok(0),
        Err(llmosafe::llmosafe_kernel::KernelError::ResourceExhaustion) => {
            Err(ResourceExhaustedError::new_err("Memory ceiling exceeded"))
        }
        Err(e) => Err(LLMOSafeError::new_err(e.to_string())),
    }
}

/// Get the current resource pressure as a percentage.
#[pyfunction]
fn get_resource_pressure(ceiling_mb: u32) -> u8 {
    let ceiling_bytes = (ceiling_mb as usize).saturating_mul(1024 * 1024);
    if ceiling_bytes == 0 {
        return 100;
    }
    let guard = ResourceGuard::new(ceiling_bytes);
    guard.pressure()
}

/// Check the stability of a synapse (cognitive state).
#[pyfunction]
fn get_stability(synapse_bits: u64) -> i32 {
    let synapse = Synapse::from_raw_u64(synapse_bits);
    match synapse.validate() {
        Ok(()) => 0,
        Err(KernelError::CognitiveInstability) => -2,
        Err(KernelError::BiasHaloDetected) => -3,
        Err(KernelError::DepthExceeded) => -1,
        Err(KernelError::HallucinationDetected) => -4,
        Err(KernelError::ResourceExhaustion) => -5,
        Err(KernelError::SelfMemoryExceeded) => -6,
        Err(KernelError::DeadlineExceeded) => -7,
    }
}

/// Get the current system CPU load percentage.
#[pyfunction]
fn get_system_cpu_load() -> u8 {
    ResourceGuard::system_cpu_load()
}

/// Get the environmental entropy score.
#[pyfunction]
fn get_environmental_entropy() -> u16 {
    use llmosafe::llmosafe_body::llmosafe_get_environmental_entropy;
    llmosafe_get_environmental_entropy()
}

/// Process a synapse through the cognitive safety pipeline.
#[pyfunction]
fn process_synapse(synapse_bits: u64) -> i32 {
    use llmosafe::llmosafe_memory::cognitive_memory::process_state_update;
    process_state_update(synapse_bits.into())
}

/// Python module definition for PyO3.
#[pymodule]
pub fn _llmosafe(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_halo, m)?)?;
    m.add_function(wrap_pyfunction!(check_resources, m)?)?;
    m.add_function(wrap_pyfunction!(get_resource_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(get_stability, m)?)?;
    m.add_function(wrap_pyfunction!(get_system_cpu_load, m)?)?;
    m.add_function(wrap_pyfunction!(get_environmental_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(process_synapse, m)?)?;
    
    m.add("LLMOSafeError", _py.get_type::<LLMOSafeError>())?;
    m.add("ResourceExhaustedError", _py.get_type::<ResourceExhaustedError>())?;
    m.add("CognitiveInstabilityError", _py.get_type::<CognitiveInstabilityError>())?;
    m.add("BiasHaloDetectedError", _py.get_type::<BiasHaloDetectedError>())?;
    
    Ok(())
}
