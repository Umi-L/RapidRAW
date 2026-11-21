//! Ordered dithering. Reduces the number of tones with Bayer matrices or noise; optional two-colour output.

struct Params {
    method: u32,    // @options("Bayer 8x8", "Bayer 4x4", "Noise", "Threshold") @default(0) @label("Method")
    levels: f32,    // @range(2, 16) @step(1) @default(2) @label("Levels per channel")
    size: f32,      // @range(1, 16) @step(1) @default(2) @label("Dot size (px)")
    strength: f32,  // @range(0, 1) @default(1) @label("Dither strength")
    mono: u32,      // @toggle @default(1) @label("Two-colour (luma)")
    ink: vec3f,     // @range(0, 1) @default(0.02, 0.02, 0.03) @label("Ink")
    paper: vec3f,   // @range(0, 1) @default(0.95, 0.93, 0.86) @label("Paper")
    region: i32,    // @mask @label("Apply to")
}

// Bayer threshold in 0..1 for an n-bit matrix (bits = 2 -> 4x4, bits = 3 -> 8x8).
fn bayer(p: vec2u, bits: u32) -> f32 {
    var v = 0u;
    for (var i = 0u; i < bits; i++) {
        let xb = (p.x >> i) & 1u;
        let yb = (p.y >> i) & 1u;
        v = (v << 2u) | ((xb ^ yb) << 1u) | yb;
    }
    return (f32(v) + 0.5) / f32(1u << (2u * bits));
}

// Interleaved gradient noise (Jimenez): cheap and stable per pixel.
fn ign(p: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

@fragment
fn main(in: RRInput) -> @location(0) vec4f {
    let color = rr_color(in);
    let px = rr_px(in);

    let cell = max(round(params.size * rr.scale), 1.0);
    let p = floor(px / cell);

    var t = 0.5;
    switch (params.method) {
        case 0u: { t = bayer(vec2u(p), 3u); }
        case 1u: { t = bayer(vec2u(p), 2u); }
        case 2u: { t = ign(p); }
        default: { t = 0.5; }
    }
    // Strength blends from a plain threshold (0.5) to the full pattern.
    t = mix(0.5, t, params.strength);

    // Quantise gamma-encoded values so the steps look even.
    let encoded = clamp(rr_to_srgb(color.rgb), vec3f(0.0), vec3f(1.0));
    let steps = max(params.levels - 1.0, 1.0);

    var result = vec3f(0.0);
    if (params.mono == 1u) {
        let l = rr_luma(encoded);
        let q = floor(l * steps + t) / steps;
        result = mix(params.ink, params.paper, clamp(q, 0.0, 1.0));
    } else {
        result = rr_from_srgb(floor(encoded * steps + t) / steps);
    }

    let coverage = rr_mask(params.region, vec2i(px));
    return vec4f(mix(color.rgb, result, coverage), color.a);
}
