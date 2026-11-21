# RapidRAW shader passes

> RapidRAW rewrites this README on every launch. Keep your own notes in another file.

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
    region: i32,  // @mask @label("Apply to")
    mode: u32,    // @options("Soft", "Hard", "Wild") @default(1)
}
```

| Annotation | Meaning |
|---|---|
| `@range(min, max)` | Slider range (default 0–1, or 0–10 for integers) |
| `@default(v)` / `@default(x, y, z)` | Initial value(s) (default 0) |
| `@step(s)` | Slider step (default 0.01, or 1 for integers) |
| `@label("Text")` | Display name (default: the field name) |
| `@toggle` | Show a switch instead of a slider (value 0 or 1) |
| `@options("A", "B", ...)` | Show a dropdown instead of a slider (field must be `i32` or `u32`); the value is the chosen index, starting at 0 (`@default(1)` picks "B") |
| `@mask` | Show a mask picker (field must be `i32`); read it with `rr_mask` |

Allowed field types: `f32`, `i32`, `u32`, `vec2f`, `vec3f`, `vec4f`.
Vectors get one slider per component.

## Masks

A `@mask` field shows a dropdown of the image's masks, plus **Whole image**. Pass the
field to `rr_mask` to get that mask's coverage (0–1) at a pixel:

```wgsl
let coverage = rr_mask(params.region, vec2i(rr_px(in)));
return vec4f(mix(c.rgb, effect, coverage), c.a);
```

- **Whole image** gives 1 everywhere.
- A hidden or deleted mask gives 0 everywhere, so the effect switches off.
- The mask already includes its invert and opacity settings.
- Masks cover the whole image, so `rr_mask` works at any pixel, with no tile reach limit.

## Iterations

Some effects need to run many times, for example sorting pixels one step per run. Add a
directive line:

```wgsl
//! @iterations(32)              run 32 times
//! @iterations(steps)           run as many times as the `steps` Params field says
//! @iterations(steps, scaled)   …times rr.scale, so preview and export match
```

Each run reads the previous run's output via `rr_color` / `rr_load`; `rr.iteration`
counts from 0 to `rr.iteration_count - 1`. The count is capped at 1024. Each iteration
is a full pass over the image, so high counts make renders slower.

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
| `rr.iteration`, `rr.iteration_count: u32` | Which run of an `@iterations` pass this is (0 and 1 otherwise) |
| `rr.mask_count: u32` | Number of visible masks on the image |
| `rr.is_preview`, `rr.is_raw: u32` | 1 when rendering the editor preview / a raw file |
| `rr.tile_origin`, `rr.tile_size: vec2f` | The tile being rendered (rarely needed) |
| `rr_px(in) -> vec2f` | Absolute pixel coordinate (pixel centre) |
| `rr_uv(in) -> vec2f` | 0–1 across the whole image |
| `rr_color(in) -> vec4f` | This pixel, from the previous pass or iteration |
| `rr_load(px: vec2i) -> vec4f` | Any pixel by absolute coordinate (clamped, see below) |
| `rr_available(px: vec2i) -> bool` | Whether `rr_load(px)` returns that exact pixel rather than a clamped one |
| `rr_sample(uv: vec2f) -> vec4f` | Bilinear sample by absolute UV (clamped, see below) |
| `rr_mask(index: i32, px: vec2i) -> f32` | Coverage of the mask chosen in a `@mask` field |
| `rr_luma(rgb) -> f32` | Rec.709 luminance |
| `rr_to_srgb(rgb)`, `rr_from_srgb(rgb)` | sRGB transfer curve helpers |

## Rules and limits

- Exactly one `@fragment` function, returning `@location(0) vec4f`.
- Don't declare your own `@group` / `@binding` resources; use `struct Params`.
- Names starting with `rr_` / `RR` are reserved.
- **Reach:** images are rendered in 2048 px tiles that overlap by 128 px, so `rr_load`
  and `rr_sample` can only see about 128 px beyond the current tile. Further samples
  clamp, and very large radii can leave faint seams in full-size exports. Keep
  `radius * rr.scale` under about 120 px. The same applies to how far an iterated
  effect spreads: after N iterations of moving one pixel each, keep N under ~128 at full
  resolution to avoid seams.
- A shader that loops forever can hang the GPU. Syntax and type errors are caught and
  shown in the editor instead.

The `*_example.wgsl` files show sliders, toggles, neighbour sampling, masks and
iterations. RapidRAW writes each example once; feel free to edit or delete them.
