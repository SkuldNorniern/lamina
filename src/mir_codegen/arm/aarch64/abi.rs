use crate::target::TargetOperatingSystem;

/// Return optional global directive and public label name for a function on the given OS.
pub fn public_symbol(func_name: &str, os: TargetOperatingSystem) -> (Option<String>, String) {
    match os {
        TargetOperatingSystem::MacOS => (
            Some(format!(".globl _{}", func_name)),
            if func_name == "main" {
                "_main".to_string()
            } else {
                format!("_{}", func_name)
            },
        ),
        // FEAT: TODO: need to mark each Symbol for each target OS
        TargetOperatingSystem::Linux
        | TargetOperatingSystem::Windows
        | TargetOperatingSystem::FreeBSD
        | TargetOperatingSystem::OpenBSD
        | TargetOperatingSystem::NetBSD
        | TargetOperatingSystem::DragonFly
        | TargetOperatingSystem::Redox
        | TargetOperatingSystem::Artery
        | TargetOperatingSystem::Unknown => {
            (Some(format!(".globl {}", func_name)), func_name.to_string())
        }
    }
}

/// Map well-known intrinsic/runtime names to platform symbol stubs.
pub fn call_stub(name: &str, os: TargetOperatingSystem) -> Option<&'static str> {
    match (name, os) {
        ("print", TargetOperatingSystem::MacOS) => Some("_printf"),
        ("print", _) => Some("printf"),
        ("malloc", TargetOperatingSystem::MacOS) => Some("_malloc"),
        ("malloc", _) => Some("malloc"),
        ("dealloc", TargetOperatingSystem::MacOS) => Some("_free"),
        ("dealloc", _) => Some("free"),
        _ => None,
    }
}
