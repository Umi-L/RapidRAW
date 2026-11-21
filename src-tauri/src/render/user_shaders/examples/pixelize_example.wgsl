//! Pixelize / mosaic. Blocks of flat colour with an optional grid.

struct Params {
    size: f32,         // @range(2, 64) @default(12) @label("Block size (px)")
    average: u32,      // @toggle @default(1) @label("Average block")
    gap: f32,          // @range(0, 0.5) @default(0) @label("Grid gap")
    gap_color: vec3f,  // @range(0, 1) @default(0, 0, 0) @label("Grid colour")
    region: i32,       // @mask @label("Apply to")
}

@fragment
fn main(in: RRInput) -> @location(0) vec4f {
    let color = rr_color(in);
    let px = rr_px(in);

    // Blocks are anchored to absolute pixels, so tiles line up seamlessly.
    let cell = max(params.size * rr.scale, 1.0);
    let index = floor(px / cell);
    let centre = (index + 0.5) * cell;

    var rgb = vec3f(0.0);
    if (params.average == 1u) {
        // 4x4 taps spread across the block.
        for (var j = 0; j < 4; j++) {
            for (var i = 0; i < 4; i++) {
                let o = (vec2f(f32(i), f32(j)) + 0.5) / 4.0 - 0.5;
                rgb += rr_sample((centre + o * cell) / rr.image_size).rgb;
            }
        }
        rgb /= 16.0;
    } else {
        rgb = rr_sample(centre / rr.image_size).rgb;
    }

    // Distance from the nearest block edge, as a fraction of the block.
    let local = px / cell - index;
    let edge = min(min(local.x, 1.0 - local.x), min(local.y, 1.0 - local.y));
    let in_gap = select(0.0, 1.0, edge < params.gap * 0.5);
    rgb = mix(rgb, params.gap_color, in_gap);

    let coverage = rr_mask(params.region, vec2i(px));
    return vec4f(mix(color.rgb, rgb, coverage), color.a);
}
