//! CRT monitor. Screen curvature, scanlines, phosphor stripes, glow and corner darkening.

struct Params {
    curvature: f32,  // @range(0, 1) @default(0.3) @label("Curvature")
    scanlines: f32,  // @range(0, 1) @default(0.5) @label("Scanlines")
    line_size: f32,  // @range(2, 12) @default(4) @label("Scanline height (px)")
    mask: f32,       // @range(0, 1) @default(0.4) @label("Phosphor mask")
    mask_size: f32,  // @range(1, 6) @step(1) @default(1) @label("Mask stripe (px)")
    glow: f32,       // @range(0, 1) @default(0.3) @label("Glow")
    vignette: f32,   // @range(0, 1) @default(0.4) @label("Corner darkening")
    region: i32,     // @mask @label("Apply to")
}

@fragment
fn main(in: RRInput) -> @location(0) vec4f {
    let color = rr_color(in);
    let px = rr_px(in);

    // Barrel distortion in -1..1 space.
    var p = rr_uv(in) * 2.0 - 1.0;
    p *= 1.0 + params.curvature * 0.25 * vec2f(p.y * p.y, p.x * p.x);
    let uv = p * 0.5 + 0.5;
    let inside = all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0));

    var rgb = rr_sample(uv).rgb;

    // Cheap glow: two rings of taps added back on top.
    if (params.glow > 0.0) {
        let r = 4.0 * rr.scale / rr.image_size;
        var sum = vec3f(0.0);
        for (var k = 0; k < 8; k++) {
            let a = f32(k) * 0.785398;
            let dir = vec2f(cos(a), sin(a));
            sum += rr_sample(uv + dir * r).rgb;
            sum += rr_sample(uv + dir * r * 2.0).rgb;
        }
        rgb += sum / 16.0 * params.glow;
    }

    // Scanlines: a 0..1 wave, normalised so the average brightness is kept.
    let period = max(params.line_size * rr.scale, 2.0);
    let wave = 0.5 + 0.5 * cos(6.2831853 * px.y / period);
    let line = 1.0 - params.scanlines * (1.0 - wave);
    rgb *= line / (1.0 - params.scanlines * 0.5);

    // RGB phosphor stripes (average weight of the mask is 0.5667).
    let stripe = max(round(params.mask_size * rr.scale), 1.0);
    let i = u32(floor(px.x / stripe)) % 3u;
    let lit = vec3f(f32(i == 0u), f32(i == 1u), f32(i == 2u));
    let m = mix(vec3f(0.35), vec3f(1.0), lit) / 0.5667;
    rgb *= mix(vec3f(1.0), m, params.mask);

    rgb *= 1.0 - params.vignette * smoothstep(0.6, 1.5, length(p));
    if (!inside) {
        rgb = vec3f(0.0);
    }

    let coverage = rr_mask(params.region, vec2i(px));
    return vec4f(mix(color.rgb, rgb, coverage), color.a);
}
