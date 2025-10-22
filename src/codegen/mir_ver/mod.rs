pub mod arm;

use crate::Result;
use std::io::Write;

/// Generate AArch64 assembly from MIR for the requested host OS.
///
/// host_os: "macos" | "linux" | "windows"
pub fn generate_mir_to_aarch64<'a, W: Write>(
    module: &'a crate::mir::Module,
    writer: &mut W,
    host_os: &str,
) -> Result<()> {
    match host_os {
        "macos" | "darwin" => arm::aarch64::generate_mir_aarch64(module, writer, TargetOs::MacOs),
        "linux" => arm::aarch64::generate_mir_aarch64(module, writer, TargetOs::Linux),
        "windows" | "win" => arm::aarch64::generate_mir_aarch64(module, writer, TargetOs::Windows),
        "bsd" => arm::aarch64::generate_mir_aarch64(module, writer, TargetOs::BSD),
        _ => arm::aarch64::generate_mir_aarch64(module, writer, TargetOs::Linux),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetOs {
    MacOs,
    Linux,
    Windows,
    BSD,
}


