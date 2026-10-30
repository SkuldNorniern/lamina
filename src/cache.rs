use std::{
    env, fs, io,
    path::{Path, PathBuf},
};

pub(crate) fn lamina_cache_dir() -> io::Result<PathBuf> {
    let base = match env::var_os("XDG_CACHE_HOME") {
        Some(dir) if Path::new(&dir).is_absolute() => PathBuf::from(dir),
        _ => env::var_os("HOME")
            .map(|home| PathBuf::from(home).join(".cache"))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "neither XDG_CACHE_HOME nor HOME is set",
                )
            })?,
    };
    let dir = base.join("lamina");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}
