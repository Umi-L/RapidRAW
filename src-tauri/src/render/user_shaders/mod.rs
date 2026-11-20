//! User shader passes: `.wgsl` fragment shaders from `<app data>/shaders` that run, in a
//! per-image stack, after every built-in adjustment. See `examples/README.md` for the
//! contract shown to users.

pub mod gpu;
pub mod parse;

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::time::SystemTime;

use serde::Serialize;
use tauri::{AppHandle, Manager};

use parse::{ParamSpec, ParsedShader, parse_user_shader};

static SHADERS_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Written once, when the folder is first created. Never overwritten afterwards.
const SEED_FILES: &[(&str, &str)] = &[
    ("README.md", include_str!("examples/README.md")),
    (
        "example_vignette.wgsl",
        include_str!("examples/example_vignette.wgsl"),
    ),
    (
        "example_glow.wgsl",
        include_str!("examples/example_glow.wgsl"),
    ),
];

/// Creates `<app data>/shaders` (seeding the examples on first run) and remembers it so
/// render paths can resolve files without an `AppHandle`.
pub fn init_shaders_dir(app_data_dir: &Path) -> std::io::Result<PathBuf> {
    let dir = app_data_dir.join("shaders");
    if !dir.exists() {
        std::fs::create_dir_all(&dir)?;
        for (name, contents) in SEED_FILES {
            std::fs::write(dir.join(name), contents)?;
        }
    }
    let _ = SHADERS_DIR.set(dir.clone());
    Ok(dir)
}

pub fn shaders_dir() -> Option<&'static Path> {
    SHADERS_DIR.get().map(PathBuf::as_path)
}

fn ensure_shaders_dir(app_handle: &AppHandle) -> Result<PathBuf, String> {
    if let Some(dir) = shaders_dir() {
        return Ok(dir.to_path_buf());
    }
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?;
    init_shaders_dir(&data_dir).map_err(|e| e.to_string())
}

/// Resolves a file name stored in the adjustments. Only plain `*.wgsl` names directly
/// inside the shaders folder are accepted, so a sidecar can't point anywhere else.
pub fn resolve_shader_file(file: &str) -> Option<PathBuf> {
    let name = Path::new(file);
    let mut components = name.components();
    let is_plain =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    let is_wgsl = name
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("wgsl"));
    if !(is_plain && is_wgsl) {
        return None;
    }
    shaders_dir().map(|dir| dir.join(name))
}

/// Identifies one version of a file on disk; a save changes it and triggers a re-parse
/// and recompile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FileStamp {
    modified: Option<SystemTime>,
    len: u64,
}

impl FileStamp {
    /// Opaque token the frontend compares to notice that a file changed on disk.
    pub fn token(&self) -> String {
        let nanos = self
            .modified
            .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        format!("{nanos}-{}", self.len)
    }
}

pub type ParseResult = Result<Arc<ParsedShader>, String>;

static PARSE_CACHE: LazyLock<Mutex<HashMap<PathBuf, (FileStamp, ParseResult)>>> =
    LazyLock::new(Default::default);

/// Reads and parses a shader file, reusing the cached result while the file is unchanged.
/// The outer error means the file couldn't be read; the inner one is a shader error.
pub fn load_shader(path: &Path) -> Result<(FileStamp, ParseResult), String> {
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let meta = std::fs::metadata(path)
        .map_err(|_| format!("{file_name} was not found in the shaders folder"))?;
    let stamp = FileStamp {
        modified: meta.modified().ok(),
        len: meta.len(),
    };
    {
        let cache = PARSE_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((cached_stamp, result)) = cache.get(path)
            && *cached_stamp == stamp
        {
            return Ok((stamp, result.clone()));
        }
    }
    let source = std::fs::read_to_string(path).map_err(|e| format!("{file_name}: {e}"))?;
    let result = parse_user_shader(&file_name, &source).map(Arc::new);
    PARSE_CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(path.to_path_buf(), (stamp, result.clone()));
    Ok((stamp, result))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UserShaderInfo {
    /// File name relative to the shaders folder; this is what adjustments store.
    pub file: String,
    pub name: String,
    pub description: String,
    pub params: Vec<ParamSpec>,
    pub error: Option<String>,
    /// Changes whenever the file changes on disk.
    pub revision: String,
}

#[tauri::command]
pub fn list_user_shaders(app_handle: AppHandle) -> Result<Vec<UserShaderInfo>, String> {
    let dir = ensure_shaders_dir(&app_handle)?;
    let mut shaders: Vec<UserShaderInfo> = std::fs::read_dir(&dir)
        .map_err(|e| e.to_string())?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("wgsl"))
        })
        .map(|path| {
            let file = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let name = path
                .file_stem()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| file.clone());
            let (description, params, error, revision) = match load_shader(&path) {
                Ok((stamp, Ok(shader))) => (
                    shader.description.clone(),
                    shader.params.clone(),
                    None,
                    stamp.token(),
                ),
                Ok((stamp, Err(error))) => (String::new(), Vec::new(), Some(error), stamp.token()),
                Err(error) => (String::new(), Vec::new(), Some(error), String::new()),
            };
            UserShaderInfo {
                file,
                name,
                description,
                params,
                error,
                revision,
            }
        })
        .collect();
    shaders.sort_by_key(|s| s.name.to_lowercase());
    Ok(shaders)
}

#[tauri::command]
pub fn open_user_shaders_folder(app_handle: AppHandle) -> Result<(), String> {
    let dir = ensure_shaders_dir(&app_handle)?;
    let program = if cfg!(target_os = "windows") {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    std::process::Command::new(program)
        .arg(&dir)
        .spawn()
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_rejects_paths_outside_the_folder() {
        let _ = SHADERS_DIR.set(PathBuf::from("shaders-root"));
        assert!(resolve_shader_file("glow.wgsl").is_some());
        assert!(resolve_shader_file("GLOW.WGSL").is_some());
        assert!(resolve_shader_file("../glow.wgsl").is_none());
        assert!(resolve_shader_file("sub/glow.wgsl").is_none());
        assert!(resolve_shader_file("C:/glow.wgsl").is_none());
        assert!(resolve_shader_file("/glow.wgsl").is_none());
        assert!(resolve_shader_file("glow.txt").is_none());
    }
}
