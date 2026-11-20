# RapidRAW shader passes

Every `.wgsl` file in this folder shows up in the **Shaders** section of the editor
under **Add pass**. Passes run after all other adjustments, top to bottom, and apply
to the preview, exports and thumbnails alike. Save a file, switch back to RapidRAW,
and the preview updates.

## Minimal shader

```wgsl
//! Lines starting with //! become the tooltip in the editor.

@fragment
fn main(in: RRInput) -> @location(0) vec4f {
    let c = rr_color(in);
    return vec4f(c.rgb * 1.2, c.a);
}
```

## Sliders

Declare a `struct Params`. Each field becomes a control; RapidRAW binds it as `params`.

```wgsl
struct Params {
    amount: f32,  // @range(0, 2) @default(0.5) @label("Amount")
    tint: vec3f,  // @range(0, 1) @default(1, 0.9, 0.8)
    invert: u32,  // @toggle
}
```

| Annotation | Meaning |
|---|---|
| `@range(min, max)` | Slider range (default 0–1) |
| `@default(v)` / `@default(x, y, z)` | Initial value(s) (default 0) |
| `@step(s)` | Slider step (default 0.01, or 1 for integers) |
| `@label("Text")` | Display name (default: the field name) |
| `@toggle` | Show a switch instead of a slider (value 0 or 1) |

Allowed field types: `f32`, `i32`, `u32`, `vec2f`, `vec3f`, `vec4f`.
Vectors get one slider per component.

## Colour

Colours are **linear light** with **sRGB / Rec.709 primaries**. 1.0 is SDR white;
in HDR mode values go up to `rr.hdr_headroom`. Wide-gamut colours can be negative.
Use `rr_to_srgb` / `rr_from_srgb` if you want to do maths on gamma-encoded values.

## Built-ins

| Name | Meaning |
|---|---|
| `rr.image_size: vec2f` | Size of the whole rendered image in pixels (the preview is smaller than an export) |
| `rr.scale: f32` | `min(image_size) / 1080`. Multiply pixel radii by this so preview and export match |
| `rr.hdr_headroom: f32` | 1.0 in SDR, above 1.0 in HDR |
| `rr.pass_index`, `rr.pass_count: u32` | This pass's position in the stack |
| `rr.is_preview`, `rr.is_raw: u32` | 1 when rendering the editor preview / a raw file |
| `rr.tile_origin`, `rr.tile_size: vec2f` | The tile being rendered (rarely needed) |
| `rr_px(in) -> vec2f` | Absolute pixel coordinate (pixel centre) |
| `rr_uv(in) -> vec2f` | 0–1 across the whole image |
| `rr_color(in) -> vec4f` | This pixel, from the previous pass |
| `rr_load(px: vec2i) -> vec4f` | Any pixel by absolute coordinate (clamped, see below) |
| `rr_sample(uv: vec2f) -> vec4f` | Bilinear sample by absolute UV (clamped, see below) |
| `rr_luma(rgb) -> f32` | Rec.709 luminance |
| `rr_to_srgb(rgb)`, `rr_from_srgb(rgb)` | sRGB transfer curve helpers |

## Rules and limits

- Exactly one `@fragment` function, returning `@location(0) vec4f`.
- Don't declare your own `@group` / `@binding` resources; use `struct Params`.
- Names starting with `rr_` / `RR` are reserved.
- **Reach:** images are rendered in 2048 px tiles that overlap by 128 px, so `rr_load`
  and `rr_sample` can only see about 128 px beyond the current tile. Further samples
  clamp, and very large radii can leave faint seams in full-size exports. Keep
  `radius * rr.scale` under about 120 px.
- A shader that loops forever can hang the GPU. Syntax and type errors are caught and
  shown in the editor instead.

The two `example_*.wgsl` files show sliders, toggles and neighbour sampling. RapidRAW
creates them once; feel free to edit or delete them.
