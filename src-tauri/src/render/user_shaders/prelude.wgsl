// ---- RapidRAW user shader prelude (prepended to every user .wgsl file) ----
// Colours are linear light with sRGB/Rec.709 primaries. 1.0 = SDR white; HDR goes above it.

struct RRUniforms {
    image_size: vec2<f32>,
    tile_origin: vec2<f32>,
    tile_size: vec2<f32>,
    scale: f32,
    hdr_headroom: f32,
    pass_index: u32,
    pass_count: u32,
    is_preview: u32,
    is_raw: u32,
}

struct RRInput {
    @builtin(position) frag: vec4<f32>,
}

@group(0) @binding(0) var rr_prev: texture_2d<f32>;
@group(0) @binding(1) var rr_sampler: sampler;
@group(0) @binding(2) var<uniform> rr: RRUniforms;

@vertex
fn rr_vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

fn rr_px(input: RRInput) -> vec2<f32> {
    return input.frag.xy + rr.tile_origin;
}

fn rr_uv(input: RRInput) -> vec2<f32> {
    return rr_px(input) / rr.image_size;
}

fn rr_color(input: RRInput) -> vec4<f32> {
    return textureLoad(rr_prev, vec2<i32>(input.frag.xy), 0);
}

fn rr_load(px: vec2<i32>) -> vec4<f32> {
    let local = clamp(px - vec2<i32>(rr.tile_origin), vec2<i32>(0), vec2<i32>(rr.tile_size) - 1);
    return textureLoad(rr_prev, local, 0);
}

fn rr_sample(uv: vec2<f32>) -> vec4<f32> {
    let local = clamp(uv * rr.image_size - rr.tile_origin, vec2<f32>(0.5), rr.tile_size - 0.5);
    return textureSampleLevel(rr_prev, rr_sampler, local / vec2<f32>(textureDimensions(rr_prev)), 0.0);
}

fn rr_luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn rr_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let c = max(linear, vec3<f32>(0.0));
    return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3<f32>(0.0031308));
}

fn rr_from_srgb(encoded: vec3<f32>) -> vec3<f32> {
    return select(pow((encoded + 0.055) / 1.055, vec3<f32>(2.4)), encoded / 12.92, encoded <= vec3<f32>(0.04045));
}
// ---- end of prelude ----
