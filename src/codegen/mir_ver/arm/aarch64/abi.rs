use crate::codegen::mir_ver::TargetOs;

/// Return optional global directive and public label name for a function on the given OS.
pub fn public_symbol(func_name: &str, os: TargetOs) -> (Option<String>, String) {
    match os {
        TargetOs::MacOs => (
            Some(format!(".globl _{}", func_name)),
            if func_name == "main" {
                "_main".to_string()
            } else {
                format!("_{}", func_name)
            },
        ),
        TargetOs::Linux | TargetOs::Windows | TargetOs::BSD => {
            (Some(format!(".globl {}", func_name)), func_name.to_string())
        }
    }
}

/// Map well-known intrinsic/runtime names to platform symbol stubs.
pub fn call_stub(name: &str, os: TargetOs) -> Option<&'static str> {
    match (name, os) {
        ("print", TargetOs::MacOs) => Some("_printf"),
        ("print", _) => Some("printf"),
        ("malloc", TargetOs::MacOs) => Some("_malloc"),
        ("malloc", _) => Some("malloc"),
        ("dealloc", TargetOs::MacOs) => Some("_free"),
        ("dealloc", _) => Some("free"),
        _ => None,
    }
}
