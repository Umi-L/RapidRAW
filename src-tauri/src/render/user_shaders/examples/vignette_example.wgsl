//! Simple coloured vignette. Shows how `struct Params` fields become sliders.

struct Params {
    amount: f32,    // @range(0, 1) @default(0.5) @label("Amount")
    size: f32,      // @range(0.1, 1.5) @default(0.8) @label("Size")
    softness: f32,  // @range(0.01, 1) @default(0.5) @label("Softness")
    tint: vec3f,    // @range(0, 1) @default(0, 0, 0) @label("Tint")
    circular: u32,  // @toggle @default(1) @label("Circular")
}

@fragment
fn main(in: RRInput) -> @location(0) vec4f {
    let color = rr_color(in);

    // Centred coordinates; optionally correct for aspect ratio so the vignette is a circle.
    var p = rr_uv(in) * 2.0 - 1.0;
    if (params.circular == 1u) {
        p.x *= rr.image_size.x / rr.image_size.y;
        p /= max(rr.image_size.x / rr.image_size.y, 1.0);
    }

    let d = length(p);
    let falloff = smoothstep(params.size, params.size + params.softness, d);
    let weight = falloff * params.amount;
    return vec4f(mix(color.rgb, params.tint, weight), color.a);
}
