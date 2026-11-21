//! Halftone print. Mono newspaper dots or four-colour CMYK screens at classic angles.

struct Params {
    mode: u32,     // @options("Mono", "CMYK") @default(1) @label("Mode")
    size: f32,     // @range(3, 40) @default(10) @label("Dot spacing (px)")
    angle: f32,    // @range(0, 90) @step(1) @default(0) @label("Screen angle offset")
    sharp: f32,    // @range(0.5, 4) @default(1) @label("Edge softness (px)")
    ink: vec3f,    // @range(0, 1) @default(0.03, 0.03, 0.04) @label("Ink (mono)")
    paper: vec3f,  // @range(0, 1) @default(0.97, 0.95, 0.9) @label("Paper")
    amount: f32,   // @range(0, 1) @default(1) @label("Amount")
    region: i32,   // @mask @label("Apply to")
}

// One rotated dot screen at `px`. Returns the cell centre (in image pixels, for sampling
// the picture) and the distance from `px` to that centre.
struct Dot {
    centre: vec2f,
    dist: f32,
}

fn dot_screen(px: vec2f, angle: f32, cell: f32) -> Dot {
    let c = cos(angle);
    let s = sin(angle);
    // Rotate into screen space, find the cell, rotate its centre back to image space.
    let q = vec2f(c * px.x + s * px.y, -s * px.x + c * px.y);
    let centre_q = (floor(q / cell) + 0.5) * cell;
    var d: Dot;
    d.centre = vec2f(c * centre_q.x - s * centre_q.y, s * centre_q.x + c * centre_q.y);
    d.dist = length(q - centre_q);
    return d;
}

// Antialiased dot coverage; the dot radius grows with the ink density.
fn dot_cover(d: Dot, density: f32, cell: f32, aa: f32) -> f32 {
    let radius = cell * 0.7071 * sqrt(clamp(density, 0.0, 1.0));
    return clamp((radius - d.dist) / aa + 0.5, 0.0, 1.0);
}

// Linear RGB -> (c, m, y, k) ink densities, each 0..1.
fn cmyk_of(linear: vec3f) -> vec4f {
    let e = clamp(rr_to_srgb(linear), vec3f(0.0), vec3f(1.0));
    let k = 1.0 - max(e.r, max(e.g, e.b));
    let inv = 1.0 / max(1.0 - k, 0.0001);
    return vec4f((1.0 - e.r - k) * inv, (1.0 - e.g - k) * inv, (1.0 - e.b - k) * inv, k);
}

@fragment
fn main(in: RRInput) -> @location(0) vec4f {
    let color = rr_color(in);
    let px = rr_px(in);

    let cell = max(params.size * rr.scale, 2.0);
    let aa = params.sharp * rr.scale;
    let off = radians(params.angle);

    var result = vec3f(0.0);
    if (params.mode == 0u) {
        let d = dot_screen(px, off + 0.785398, cell);
        let sampled = clamp(rr_to_srgb(rr_sample(d.centre / rr.image_size).rgb), vec3f(0.0), vec3f(1.0));
        let cov = dot_cover(d, 1.0 - rr_luma(sampled), cell, aa);
        result = mix(params.paper, params.ink, cov);
    } else {
        // One screen per ink, each at its own angle to avoid moire.
        let dc = dot_screen(px, off + 0.261799, cell);  // cyan 15 deg
        let dm = dot_screen(px, off + 1.308997, cell);  // magenta 75 deg
        let dy = dot_screen(px, off, cell);             // yellow 0 deg
        let dk = dot_screen(px, off + 0.785398, cell);  // black 45 deg

        let cov_c = dot_cover(dc, cmyk_of(rr_sample(dc.centre / rr.image_size).rgb).x, cell, aa);
        let cov_m = dot_cover(dm, cmyk_of(rr_sample(dm.centre / rr.image_size).rgb).y, cell, aa);
        let cov_y = dot_cover(dy, cmyk_of(rr_sample(dy.centre / rr.image_size).rgb).z, cell, aa);
        let cov_k = dot_cover(dk, cmyk_of(rr_sample(dk.centre / rr.image_size).rgb).w, cell, aa);

        // Subtractive mixing: each ink filters the paper.
        let ink_c = rr_from_srgb(vec3f(0.0, 0.68, 0.94));
        let ink_m = rr_from_srgb(vec3f(0.93, 0.0, 0.55));
        let ink_y = rr_from_srgb(vec3f(1.0, 0.95, 0.0));
        let ink_k = rr_from_srgb(vec3f(0.1, 0.1, 0.1));
        result = params.paper;
        result *= mix(vec3f(1.0), ink_c, cov_c);
        result *= mix(vec3f(1.0), ink_m, cov_m);
        result *= mix(vec3f(1.0), ink_y, cov_y);
        result *= mix(vec3f(1.0), ink_k, cov_k);
    }

    let coverage = rr_mask(params.region, vec2i(px)) * params.amount;
    return vec4f(mix(color.rgb, result, coverage), color.a);
}
