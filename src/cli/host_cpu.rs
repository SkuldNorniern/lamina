//! Whether the host can execute code built for a given x86-64 level.

use crate::LaminaError;
use lamina_platform::X86CpuLevel;

/// Features each x86-64 level adds over the one below it.
fn level_features(level: X86CpuLevel) -> &'static [&'static str] {
    match level {
        X86CpuLevel::V1 => &["sse2"],
        X86CpuLevel::V2 => &["sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "cmpxchg16b"],
        X86CpuLevel::V3 => &[
            "avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "lzcnt", "movbe",
        ],
        X86CpuLevel::V4 => &["avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"],
    }
}

#[cfg(target_arch = "x86_64")]
fn host_x86_features() -> Vec<&'static str> {
    let detected = [
        ("sse2", std::arch::is_x86_feature_detected!("sse2")),
        ("sse3", std::arch::is_x86_feature_detected!("sse3")),
        ("ssse3", std::arch::is_x86_feature_detected!("ssse3")),
        ("sse4.1", std::arch::is_x86_feature_detected!("sse4.1")),
        ("sse4.2", std::arch::is_x86_feature_detected!("sse4.2")),
        ("popcnt", std::arch::is_x86_feature_detected!("popcnt")),
        (
            "cmpxchg16b",
            std::arch::is_x86_feature_detected!("cmpxchg16b"),
        ),
        ("avx", std::arch::is_x86_feature_detected!("avx")),
        ("avx2", std::arch::is_x86_feature_detected!("avx2")),
        ("bmi1", std::arch::is_x86_feature_detected!("bmi1")),
        ("bmi2", std::arch::is_x86_feature_detected!("bmi2")),
        ("f16c", std::arch::is_x86_feature_detected!("f16c")),
        ("fma", std::arch::is_x86_feature_detected!("fma")),
        ("lzcnt", std::arch::is_x86_feature_detected!("lzcnt")),
        ("movbe", std::arch::is_x86_feature_detected!("movbe")),
        ("avx512f", std::arch::is_x86_feature_detected!("avx512f")),
        ("avx512bw", std::arch::is_x86_feature_detected!("avx512bw")),
        ("avx512cd", std::arch::is_x86_feature_detected!("avx512cd")),
        ("avx512dq", std::arch::is_x86_feature_detected!("avx512dq")),
        ("avx512vl", std::arch::is_x86_feature_detected!("avx512vl")),
    ];
    detected
        .into_iter()
        .filter(|(_, present)| *present)
        .map(|(name, _)| name)
        .collect()
}

#[cfg(not(target_arch = "x86_64"))]
fn host_x86_features() -> Vec<&'static str> {
    Vec::new()
}

/// Refuses a run whose x86-64 level the host cannot execute.
///
/// `what` names the thing that would not run, and `remedy` the way out; both are
/// quoted into the message so a JIT run and an AOT build read differently.
pub fn check_host_runs_level(
    level: X86CpuLevel,
    what: &str,
    remedy: &str,
) -> Result<(), LaminaError> {
    match level_refusal(&host_x86_features(), level, what, remedy) {
        Some(message) => Err(LaminaError::ValidationError(message)),
        None => Ok(()),
    }
}

fn level_refusal(host: &[&str], level: X86CpuLevel, what: &str, remedy: &str) -> Option<String> {
    let missing: Vec<&str> = X86CpuLevel::ALL
        .into_iter()
        .filter(|candidate| *candidate <= level)
        .flat_map(|candidate| level_features(candidate).iter().copied())
        .filter(|feature| !host.contains(feature))
        .collect();
    if missing.is_empty() {
        return None;
    }
    let highest = X86CpuLevel::ALL
        .into_iter()
        .take_while(|candidate| level_features(*candidate).iter().all(|f| host.contains(f)))
        .last();
    let supported = highest.map_or_else(|| "no x86-64 level".to_owned(), |level| level.to_string());
    Some(format!(
        "{what} cannot run {level} code: this host lacks {} (the level needs {}). This host supports {supported}; {remedy}",
        missing.join(", "),
        level.features(),
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn level_refusal_names_the_missing_features() {
        let v2_host: Vec<&str> = level_features(X86CpuLevel::V1)
            .iter()
            .chain(level_features(X86CpuLevel::V2))
            .copied()
            .collect();
        let refusal = level_refusal(&v2_host, X86CpuLevel::V3, "--jit", "pick a lower level")
            .expect("v3 needs AVX2");
        assert!(refusal.contains("x86-64-v3"), "{refusal}");
        assert!(refusal.contains("avx2"), "{refusal}");
        assert!(refusal.contains("supports x86-64-v2"), "{refusal}");
        assert!(refusal.contains("pick a lower level"), "{refusal}");
        assert!(level_refusal(&v2_host, X86CpuLevel::V2, "--jit", "x").is_none());
        assert!(level_refusal(&[], X86CpuLevel::V1, "--jit", "x").is_some());
    }
}
