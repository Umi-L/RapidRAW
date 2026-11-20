//! CPU-side handling of user shader files: build the full WGSL source (prelude + user code),
//! validate it with naga, and reflect `struct Params` into slider specs and a uniform layout.
//! Nothing here touches the GPU, so a broken file can never reach the device.

use serde::Serialize;

pub const PRELUDE: &str = include_str!("prelude.wgsl");

/// Binding line appended to the prelude when the user declares `struct Params`.
const PARAMS_BINDING: &str = "@group(0) @binding(3) var<uniform> params: Params;\n";

/// Names of globals the prelude declares; any other bound global in a user file is rejected.
const PRELUDE_GLOBALS: &[&str] = &["rr_prev", "rr_sampler", "rr", "params"];

pub const VERTEX_ENTRY: &str = "rr_vs";

#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ParamKind {
    Float,
    Int,
    Uint,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ParamSpec {
    pub name: String,
    pub label: String,
    pub kind: ParamKind,
    /// 1 for scalars, 2–4 for vectors.
    pub components: u32,
    pub min: f32,
    pub max: f32,
    pub step: f32,
    pub default: Vec<f32>,
    pub toggle: bool,
    /// Byte offset inside the `Params` uniform buffer.
    #[serde(skip)]
    pub offset: u32,
}

#[derive(Clone, Debug)]
pub struct ParsedShader {
    /// Prelude + params binding + user code, ready for `create_shader_module`.
    pub source: String,
    pub fragment_entry: String,
    pub description: String,
    pub params: Vec<ParamSpec>,
    /// Size of the `Params` struct in bytes (0 when the file has none).
    pub params_size: u32,
}

impl ParsedShader {
    /// Packs slider values (as stored in the adjustments JSON) into the `Params` uniform
    /// buffer layout. Missing or malformed values fall back to the annotation defaults.
    pub fn pack_params(&self, values: Option<&serde_json::Value>) -> Vec<u8> {
        let size = self.params_size.max(16).div_ceil(16) * 16;
        let mut bytes = vec![0u8; size as usize];
        for spec in &self.params {
            let value = values.and_then(|v| v.get(&spec.name));
            for i in 0..spec.components as usize {
                let v = match value {
                    Some(serde_json::Value::Array(items)) => items.get(i).and_then(|x| x.as_f64()),
                    Some(other) if spec.components == 1 => other
                        .as_f64()
                        .or_else(|| other.as_bool().map(|b| if b { 1.0 } else { 0.0 })),
                    _ => None,
                }
                .map(|x| x as f32)
                .unwrap_or(spec.default[i]);
                let word = match spec.kind {
                    ParamKind::Float => v.to_ne_bytes(),
                    ParamKind::Int => (v.round() as i32).to_ne_bytes(),
                    ParamKind::Uint => (v.round().max(0.0) as u32).to_ne_bytes(),
                };
                let at = (spec.offset as usize) + i * 4;
                bytes[at..at + 4].copy_from_slice(&word);
            }
        }
        bytes
    }
}

/// Parses and validates one user shader. `file_name` is only used in error messages.
pub fn parse_user_shader(file_name: &str, user_source: &str) -> Result<ParsedShader, String> {
    let code = strip_comments(user_source);
    let has_params = regex::Regex::new(r"\bstruct\s+Params\b")
        .expect("valid regex")
        .is_match(&code);

    let header = if has_params {
        format!("{PRELUDE}{PARAMS_BINDING}")
    } else {
        PRELUDE.to_string()
    };
    let header_lines = header.matches('\n').count() as u32;
    let source = format!("{header}{user_source}");

    let locate = |line: Option<u32>, message: String| match line {
        Some(line) if line > header_lines => {
            format!("{file_name}:{}: {message}", line - header_lines)
        }
        Some(_) => format!("{file_name}: {message} (conflicts with a RapidRAW built-in)"),
        None => format!("{file_name}: {message}"),
    };

    let module = naga::front::wgsl::parse_str(&source).map_err(|e| {
        let label = e
            .labels()
            .next()
            .map(|(_, text)| text.to_string())
            .filter(|text| !text.is_empty() && !e.message().contains(text.as_str()));
        let message = match label {
            Some(label) => format!("{} ({label})", e.message()),
            None => e.message().to_string(),
        };
        locate(e.location(&source).map(|l| l.line_number), message)
    })?;

    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .map_err(|e| {
        let mut message = e.as_inner().to_string();
        let mut cause = std::error::Error::source(e.as_inner());
        while let Some(inner) = cause {
            message.push_str(": ");
            message.push_str(&inner.to_string());
            cause = inner.source();
        }
        locate(e.location(&source).map(|l| l.line_number), message)
    })?;

    for (_, global) in module.global_variables.iter() {
        let name = global.name.as_deref().unwrap_or("");
        if global.binding.is_some() && !PRELUDE_GLOBALS.contains(&name) {
            return Err(format!(
                "{file_name}: `{name}` declares its own @group/@binding. Use `struct Params` for inputs instead."
            ));
        }
    }

    let fragments: Vec<_> = module
        .entry_points
        .iter()
        .filter(|ep| ep.stage == naga::ShaderStage::Fragment)
        .collect();
    let fragment = match fragments.as_slice() {
        [one] => *one,
        [] => {
            return Err(format!(
                "{file_name}: no @fragment function found. Add `@fragment fn main(in: RRInput) -> @location(0) vec4f`."
            ));
        }
        _ => {
            return Err(format!(
                "{file_name}: found {} @fragment functions; exactly one is allowed.",
                fragments.len()
            ));
        }
    };
    let returns_color =
        fragment.function.result.as_ref().is_some_and(|r| {
            matches!(r.binding, Some(naga::Binding::Location { location: 0, .. }))
        });
    if !returns_color {
        return Err(format!(
            "{file_name}: `{}` must return `@location(0) vec4f`.",
            fragment.name
        ));
    }

    let (params, params_size) = if has_params {
        reflect_params(file_name, &module, user_source)?
    } else {
        (Vec::new(), 0)
    };

    Ok(ParsedShader {
        source,
        fragment_entry: fragment.name.clone(),
        description: description(user_source),
        params,
        params_size,
    })
}

fn reflect_params(
    file_name: &str,
    module: &naga::Module,
    user_source: &str,
) -> Result<(Vec<ParamSpec>, u32), String> {
    let (members, span) = module
        .types
        .iter()
        .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
            (Some(name), naga::TypeInner::Struct { members, span }) if name == "Params" => {
                Some((members, *span))
            }
            _ => None,
        })
        .ok_or_else(|| format!("{file_name}: could not find `struct Params`."))?;

    let annotations = member_comments(user_source);
    let mut specs = Vec::with_capacity(members.len());
    for member in members {
        let name = member.name.clone().unwrap_or_default();
        let unsupported = || {
            format!(
                "{file_name}: Params field `{name}` has an unsupported type. Use f32, i32, u32, vec2f, vec3f or vec4f."
            )
        };
        let (kind, components) = match &module.types[member.ty].inner {
            naga::TypeInner::Scalar(scalar) if scalar.width == 4 => match scalar.kind {
                naga::ScalarKind::Float => (ParamKind::Float, 1),
                naga::ScalarKind::Sint => (ParamKind::Int, 1),
                naga::ScalarKind::Uint => (ParamKind::Uint, 1),
                _ => return Err(unsupported()),
            },
            naga::TypeInner::Vector { size, scalar }
                if scalar.kind == naga::ScalarKind::Float && scalar.width == 4 =>
            {
                (ParamKind::Float, *size as u32)
            }
            _ => return Err(unsupported()),
        };
        let comment = annotations
            .iter()
            .find(|(field, _)| *field == name)
            .map_or("", |(_, comment)| comment.as_str());
        specs.push(build_spec(name, kind, components, member.offset, comment));
    }
    Ok((specs, span))
}

fn build_spec(
    name: String,
    kind: ParamKind,
    components: u32,
    offset: u32,
    comment: &str,
) -> ParamSpec {
    let mut spec = ParamSpec {
        label: name.clone(),
        name,
        kind,
        components,
        min: 0.0,
        max: if kind == ParamKind::Float { 1.0 } else { 10.0 },
        step: if kind == ParamKind::Float { 0.01 } else { 1.0 },
        default: Vec::new(),
        toggle: false,
        offset,
    };
    let numbers = |args: &str| -> Vec<f32> {
        args.split(',')
            .filter_map(|part| part.trim().parse::<f32>().ok())
            .collect()
    };
    let mut defaults = Vec::new();
    for (key, args) in parse_annotations(comment) {
        let args = args.unwrap_or_default();
        match key.as_str() {
            "range" => {
                if let [min, max] = numbers(&args)[..] {
                    spec.min = min.min(max);
                    spec.max = min.max(max);
                }
            }
            "default" => defaults = numbers(&args),
            "step" => {
                if let Some(&step) = numbers(&args).first().filter(|s| **s > 0.0) {
                    spec.step = step;
                }
            }
            "label" => {
                let label = args.trim().trim_matches('"').trim();
                if !label.is_empty() {
                    spec.label = label.to_string();
                }
            }
            "toggle" => spec.toggle = true,
            _ => {}
        }
    }
    if spec.toggle {
        spec.min = 0.0;
        spec.max = 1.0;
        spec.step = 1.0;
    }
    let fallback = 0.0_f32.clamp(spec.min, spec.max);
    spec.default = (0..components as usize)
        .map(|i| match defaults.len() {
            0 => fallback,
            1 => defaults[0],
            _ => defaults.get(i).copied().unwrap_or(fallback),
        })
        .collect();
    spec
}

/// `@name` or `@name(args)` tokens from a trailing field comment. Parentheses inside
/// quotes (as in `@label("Radius (px)")`) are kept.
fn parse_annotations(comment: &str) -> Vec<(String, Option<String>)> {
    let chars: Vec<char> = comment.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '@' {
            i += 1;
            continue;
        }
        i += 1;
        let start = i;
        while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
            i += 1;
        }
        let key: String = chars[start..i].iter().collect();
        if key.is_empty() {
            continue;
        }
        let mut args = None;
        if i < chars.len() && chars[i] == '(' {
            i += 1;
            let arg_start = i;
            let mut in_quotes = false;
            while i < chars.len() && (in_quotes || chars[i] != ')') {
                if chars[i] == '"' {
                    in_quotes = !in_quotes;
                }
                i += 1;
            }
            args = Some(chars[arg_start..i.min(chars.len())].iter().collect());
            i += 1;
        }
        out.push((key.to_lowercase(), args));
    }
    out
}

/// Maps each field line inside `struct Params { ... }` to its trailing `//` comment.
fn member_comments(user_source: &str) -> Vec<(String, String)> {
    let header = regex::Regex::new(r"\bstruct\s+Params\s*\{").expect("valid regex");
    let field = regex::Regex::new(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:").expect("valid regex");
    let Some(found) = header.find(user_source) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    // The first line may hold a field after the brace; scan from the brace onwards.
    for line in user_source[found.end()..].lines() {
        let (code, comment) = match line.find("//") {
            Some(at) => (&line[..at], &line[at + 2..]),
            None => (line, ""),
        };
        for piece in code.split(',') {
            if let Some(caps) = field.captures(piece) {
                out.push((caps[1].to_string(), comment.to_string()));
            }
        }
        if code.contains('}') {
            break;
        }
    }
    out
}

fn description(user_source: &str) -> String {
    user_source
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix("//!"))
        .map(str::trim)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Removes `//` and `/* */` comments so text searches don't match commented-out code.
fn strip_comments(source: &str) -> String {
    let block = regex::Regex::new(r"(?s)/\*.*?\*/").expect("valid regex");
    block
        .replace_all(source, " ")
        .lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str =
        "@fragment\nfn main(in: RRInput) -> @location(0) vec4f {\n    return rr_color(in);\n}\n";

    #[test]
    fn minimal_shader_parses() {
        let parsed = parse_user_shader("min.wgsl", MINIMAL).unwrap();
        assert_eq!(parsed.fragment_entry, "main");
        assert!(parsed.params.is_empty());
        assert_eq!(parsed.params_size, 0);
    }

    #[test]
    fn bundled_examples_parse() {
        for (name, src) in [
            (
                "example_vignette.wgsl",
                include_str!("examples/example_vignette.wgsl"),
            ),
            (
                "example_glow.wgsl",
                include_str!("examples/example_glow.wgsl"),
            ),
        ] {
            let parsed = parse_user_shader(name, src).unwrap_or_else(|e| panic!("{e}"));
            assert!(!parsed.params.is_empty(), "{name} should expose params");
            assert!(
                !parsed.description.is_empty(),
                "{name} should have a description"
            );
        }
    }

    #[test]
    fn params_reflection_uses_uniform_layout() {
        let src = "struct Params {\n    a: f32, // @range(0, 2) @default(0.5) @label(\"Amount (px)\")\n    tint: vec3f, // @default(1, 0.9, 0.8)\n    n: i32, // @range(-5, 5) @default(2)\n    flag: u32, // @toggle @default(1)\n}\n\n@fragment\nfn main(in: RRInput) -> @location(0) vec4f {\n    return rr_color(in) * params.a + vec4f(params.tint, f32(params.n + i32(params.flag)));\n}\n";
        let parsed = parse_user_shader("p.wgsl", src).unwrap();
        let by_name = |n: &str| parsed.params.iter().find(|p| p.name == n).unwrap();

        let a = by_name("a");
        assert_eq!(
            (a.offset, a.min, a.max, a.default.clone()),
            (0, 0.0, 2.0, vec![0.5])
        );
        assert_eq!(a.label, "Amount (px)");

        let tint = by_name("tint");
        assert_eq!(tint.offset, 16, "vec3f aligns to 16 bytes");
        assert_eq!(tint.components, 3);
        assert_eq!(tint.default, vec![1.0, 0.9, 0.8]);

        let n = by_name("n");
        assert_eq!(n.offset, 28, "i32 packs into the vec3's tail");
        assert_eq!(
            (n.kind, n.min, n.max, n.step),
            (ParamKind::Int, -5.0, 5.0, 1.0)
        );

        let flag = by_name("flag");
        assert!(flag.toggle);
        assert_eq!(flag.kind, ParamKind::Uint);
        assert_eq!(flag.offset, 32);
        assert_eq!(
            parsed.params_size, 48,
            "struct span rounds up to its 16-byte alignment"
        );
    }

    #[test]
    fn pack_params_uses_values_then_defaults() {
        let src = "struct Params {\n    a: f32, // @default(0.5)\n    tint: vec3f, // @default(1, 2, 3)\n    n: i32,\n}\n@fragment\nfn main(in: RRInput) -> @location(0) vec4f {\n    return vec4f(params.tint * params.a, f32(params.n));\n}\n";
        let parsed = parse_user_shader("p.wgsl", src).unwrap();
        let values = serde_json::json!({ "tint": [4.0, 5.0], "n": 7.4 });
        let bytes = parsed.pack_params(Some(&values));
        assert_eq!(bytes.len(), 32);
        let f = |at: usize| f32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap());
        let i = |at: usize| i32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap());
        assert_eq!(f(0), 0.5, "missing value falls back to default");
        assert_eq!(
            (f(16), f(20), f(24)),
            (4.0, 5.0, 3.0),
            "short arrays fall back per component"
        );
        assert_eq!(i(28), 7);
    }

    #[test]
    fn errors_report_user_line_numbers() {
        let src = "@fragment\nfn main(in: RRInput) -> @location(0) vec4f {\n    let c = rr_color(in);\n    return colr;\n}\n";
        let err = parse_user_shader("bad.wgsl", src).unwrap_err();
        assert!(err.starts_with("bad.wgsl:4:"), "got: {err}");
        assert!(err.contains("colr"), "got: {err}");
    }

    #[test]
    fn rejects_user_bindings() {
        let src = "@group(1) @binding(0) var<uniform> mine: vec4f;\n@fragment\nfn main(in: RRInput) -> @location(0) vec4f {\n    return mine;\n}\n";
        let err = parse_user_shader("bind.wgsl", src).unwrap_err();
        assert!(err.contains("@group/@binding"), "got: {err}");
    }

    #[test]
    fn rejects_missing_or_multiple_fragments() {
        let none = "fn helper() -> f32 { return 1.0; }\n";
        assert!(
            parse_user_shader("none.wgsl", none)
                .unwrap_err()
                .contains("no @fragment")
        );

        let two = format!(
            "{MINIMAL}@fragment\nfn other(in: RRInput) -> @location(0) vec4f {{\n    return vec4f(1.0);\n}}\n"
        );
        assert!(
            parse_user_shader("two.wgsl", &two)
                .unwrap_err()
                .contains("exactly one")
        );
    }

    #[test]
    fn rejects_unsupported_param_types() {
        let src = "struct Params {\n    m: mat2x2f,\n}\n@fragment\nfn main(in: RRInput) -> @location(0) vec4f {\n    return vec4f(params.m[0], 0.0, 1.0);\n}\n";
        assert!(
            parse_user_shader("m.wgsl", src)
                .unwrap_err()
                .contains("unsupported type")
        );
    }

    #[test]
    fn main_shader_has_valid_user_encode_entry() {
        let source = crate::render::gpu::pipelines::high_precision_shader_source();
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        assert!(
            module
                .entry_points
                .iter()
                .any(|ep| ep.name == "user_encode")
        );
        assert!(module.entry_points.iter().any(|ep| ep.name == "main"));
    }

    #[test]
    fn commented_out_params_struct_is_ignored() {
        let src = format!("// struct Params {{ a: f32 }}\n{MINIMAL}");
        let parsed = parse_user_shader("c.wgsl", &src).unwrap();
        assert!(parsed.params.is_empty());
    }
}
