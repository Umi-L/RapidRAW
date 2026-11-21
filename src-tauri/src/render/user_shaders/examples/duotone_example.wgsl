//! Duotone / tritone gradient map. Maps brightness onto a colour ramp.

struct Params {
    amount: f32,       // @range(0, 1) @default(1) @label("Amount")
    mode: u32,         // @options("Duotone", "Tritone") @default(0) @label("Mode")
    shadows: vec3f,    // @range(0, 1) @default(0.02, 0.03, 0.15) @label("Shadows")
    mids: vec3f,       // @range(0, 1) @default(0.6, 0.1, 0.25) @label("Mids (tritone)")
    highlights: vec3f, // @range(0, 1) @default(1, 0.8, 0.45) @label("Highlights")
    midpoint: f32,     // @range(0.05, 0.95) @default(0.5) @label("Midpoint")
    contrast: f32,     // @range(0.5, 2) @default(1) @label("Contrast")
    region: i32,       // @mask @label("Apply to")
}

@fragment
fn main(in: RRInput) -> @location(0) vec4f {
    let color = rr_color(in);

    // Work on perceptual brightness so the ramp feels even.
    var l = rr_luma(rr_to_srgb(color.rgb));
    l = clamp((l - 0.5) * params.contrast + 0.5, 0.0, 1.0);

    var mapped = mix(params.shadows, params.highlights, l);
    if (params.mode == 1u) {
        if (l < params.midpoint) {
            mapped = mix(params.shadows, params.mids, l / params.midpoint);
        } else {
            mapped = mix(params.mids, params.highlights, (l - params.midpoint) / (1.0 - params.midpoint));
        }
    }

    let coverage = rr_mask(params.region, vec2i(rr_px(in))) * params.amount;
    return vec4f(mix(color.rgb, mapped, coverage), color.a);
}
