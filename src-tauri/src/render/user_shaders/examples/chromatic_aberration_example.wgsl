//! Chromatic aberration. Splits red and blue apart, radially from the centre or along an angle.

struct Params {
    amount: f32,   // @range(0, 30) @default(6) @label("Amount (px)")
    mode: u32,     // @options("Radial", "Linear") @default(0) @label("Mode")
    falloff: f32,  // @range(1, 4) @default(2) @label("Edge falloff (radial)")
    angle: f32,    // @range(0, 360) @step(1) @default(0) @label("Angle (linear)")
    region: i32,   // @mask @label("Apply to")
}

@fragment
fn main(in: RRInput) -> @location(0) vec4f {
    let color = rr_color(in);
    let px = rr_px(in);

    var shift = vec2f(0.0);
    if (params.mode == 0u) {
        // 0 at the centre, 1 at the corners.
        let half_size = 0.5 * rr.image_size;
        let d = (px - half_size) / length(half_size);
        shift = d * pow(max(length(d), 0.0001), params.falloff - 1.0);
    } else {
        let a = radians(params.angle);
        shift = vec2f(cos(a), sin(a));
    }
    // Pixel amounts are defined at 1080px; rr.scale keeps preview and export alike.
    shift *= params.amount * rr.scale;

    let r = rr_sample((px + shift) / rr.image_size).r;
    let b = rr_sample((px - shift) / rr.image_size).b;
    let result = vec3f(r, color.g, b);

    let coverage = rr_mask(params.region, vec2i(px));
    return vec4f(mix(color.rgb, result, coverage), color.a);
}
