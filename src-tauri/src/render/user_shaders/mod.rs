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

/// The reference doc; owned by the app and rewritten on every launch so it stays current.
const README: &str = include_str!("examples/README.md");

/// Example shaders, each written once. Names already listed in `SEEDED_LIST` are never
/// written again, so a user's edits and deletions stick while new examples still arrive.
///
/// To add an example: drop `<name>_example.wgsl` into `examples/` and add its file name
/// to the list below. The `bundled_examples_parse` test then checks it automatically.
macro_rules! examples {
    ($($file:literal),* $(,)?) => {
        &[$(($file, include_str!(concat!("examples/", $file)))),*]
    };
}

const EXAMPLES: &[(&str, &str)] = examples![
    "vignette_example.wgsl",
    "glow_example.wgsl",
    "pixel_sort_example.wgsl",
    "duotone_example.wgsl",
    "chromatic_aberration_example.wgsl",
    "pixelize_example.wgsl",
    "dither_example.wgsl",
    "crt_example.wgsl",
    "halftone_example.wgsl",
];

const SEEDED_LIST: &str = ".seeded-examples";

/// Creates `<app data>/shaders`, refreshes the README, writes examples not seeded before,
/// and remembers the folder so render paths can resolve files without an `AppHandle`.
pub fn init_shaders_dir(app_data_dir: &Path) -> std::io::Result<PathBuf> {
    let dir = app_data_dir.join("shaders");
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("README.md"), README)?;

    let list_path = dir.join(SEEDED_LIST);
    let mut seeded: Vec<String> = std::fs::read_to_string(&list_path)
        .unwrap_or_default()
        .lines()
        .map(str::to_string)
        .collect();
    let mut changed = false;
    for (name, contents) in EXAMPLES {
        if seeded.iter().any(|s| s == name) {
            continue;
        }
        let path = dir.join(name);
        if !path.exists() {
            std::fs::write(&path, contents)?;
        }
        seeded.push(name.to_string());
        changed = true;
    }
    if changed {
        std::fs::write(&list_path, seeded.join("\n") + "\n")?;
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

#[cfg(test)]
mod probe_tmp {
    #[test]
    fn probe_real_sidecar() {
        let appdata = std::env::var("APPDATA").unwrap();
        let _ = super::SHADERS_DIR.set(std::path::PathBuf::from(format!("{appdata}/io.github.CyberTimon.RapidRAW/shaders")));
        let home = std::env::var("USERPROFILE").unwrap();
        let sidecar = format!("{home}/OneDrive/Pictures/Walk Oct 29 w mom/DSC02857.ARW.rrdata");
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(sidecar).unwrap()).unwrap();
        let adjustments = &meta["adjustments"];
        for file in ["vignette_example.wgsl", "glow_example.wgsl", "pixel_sort_example.wgsl"] {
            let path = super::resolve_shader_file(file).unwrap();
            let r = super::load_shader(&path).map(|(_, r)| r.map(|s| s.params.len()));
            eprintln!("PROBE {file}: {:?}", r);
        }
        let stack = super::gpu::UserPassStack::from_adjustments(adjustments, true);
        eprintln!("PROBE passes: {}", stack.passes.len());
        eprintln!("PROBE tool active: {}", crate::render::adjustment_utils::is_tool_active(adjustments, "shaders"));
    }
}
