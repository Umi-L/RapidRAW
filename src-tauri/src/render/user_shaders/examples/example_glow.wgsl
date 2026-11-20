//! Soft highlight glow. Shows neighbour sampling with rr_load and resolution-independent radii.

struct Params {
    strength: f32,   // @range(0, 2) @default(0.6) @label("Strength")
    threshold: f32,  // @range(0, 2) @default(0.8) @label("Threshold")
    radius: f32,     // @range(1, 60) @default(20) @label("Radius")
    tint: vec3f,     // @range(0, 1) @default(1, 0.85, 0.7) @label("Tint")
}

@fragment
fn main(in: RRInput) -> @location(0) vec4f {
    let color = rr_color(in);
    let center = vec2i(rr_px(in));

    // Radius is defined at a 1080px short edge; rr.scale keeps preview and export alike.
    // Keep radius * rr.scale under ~120 px: tiles only see 128 px past their edges.
    let r = params.radius * rr.scale;

    var sum = vec3f(0.0);
    var weight = 0.0;
    for (var ring = 1; ring <= 4; ring++) {
        let dist = r * f32(ring) / 4.0;
        let w = 1.0 / f32(ring);
        for (var k = 0; k < 8; k++) {
            let angle = f32(k) * 0.785398 + f32(ring) * 0.4;
            let offset = vec2i(round(vec2f(cos(angle), sin(angle)) * dist));
            let s = rr_load(center + offset).rgb;
            sum += max(s - vec3f(params.threshold), vec3f(0.0)) * w;
            weight += w;
        }
    }

    let glow = sum / weight * params.tint;
    return vec4f(color.rgb + glow * params.strength, color.a);
}
