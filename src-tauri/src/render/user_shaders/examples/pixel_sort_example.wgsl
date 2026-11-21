//! Pixel sort: sorts runs of pixels by brightness, colour or channel, optionally only inside a mask.
//! Only pixels inside the mask whose brightness is between the thresholds move.
//! @iterations(streak, scaled)

// How it works: odd-even transposition sort. The pass runs once per iteration; each
// iteration every pixel pairs with a neighbour and the pair swaps if it's out of order.
// A pixel moves at most one step per iteration, so `streak` iterations fully sort runs up
// to `streak` pixels long (measured at 1080 px; `scaled` keeps preview and export alike).
// Runs longer than about 128 px at full resolution can show seams at 2048 px tile edges.

struct Params {
    region: i32,     // @mask @label("Mask")
    streak: f32,     // @range(1, 128) @default(48) @step(1) @label("Streak length")
    low: f32,        // @range(0, 1) @default(0.2) @label("Lower threshold")
    high: f32,       // @range(0, 4) @default(1) @label("Upper threshold")
    vertical: u32,   // @toggle @label("Vertical")
    sort_by: u32,    // @options("Brightness", "Hue", "Saturation", "Red", "Green", "Blue") @label("Sort by")
    order: u32,      // @options("Ascending", "Descending") @label("Order")
}

// The value pixels are ordered by, for the chosen "Sort by" mode.
fn sort_key(color: vec3f) -> f32 {
    let mx = max(color.r, max(color.g, color.b));
    let mn = min(color.r, min(color.g, color.b));
    let d = mx - mn;
    switch params.sort_by {
        case 1u: { // hue, 0..1
            if (d <= 0.0) { return 0.0; }
            var h: f32;
            if (mx == color.r) {
                h = (color.g - color.b) / d;
            } else if (mx == color.g) {
                h = 2.0 + (color.b - color.r) / d;
            } else {
                h = 4.0 + (color.r - color.g) / d;
            }
            return fract(h / 6.0);
        }
        case 2u: { return select(0.0, d / mx, mx > 0.0); } // saturation
        case 3u: { return color.r; }
        case 4u: { return color.g; }
        case 5u: { return color.b; }
        default: { return rr_luma(color); }
    }
}

// Whether the pixel at `px` takes part. The mask is fixed per position and both pixels of
// a swap pass the brightness test, so the set of sortable positions never changes.
fn sortable(px: vec2i, color: vec3f) -> bool {
    let l = rr_luma(color);
    return rr_mask(params.region, px) >= 0.5 && l >= params.low && l <= params.high;
}

@fragment
fn main(in: RRInput) -> @location(0) vec4f {
    let px = vec2i(rr_px(in));
    let here = rr_color(in);
    let vertical = params.vertical == 1u;
    let axis = select(vec2i(1, 0), vec2i(0, 1), vertical);
    let pos = select(px.x, px.y, vertical);

    // Even iterations pair (0,1)(2,3)…, odd iterations pair (1,2)(3,4)…
    let is_first = ((pos + i32(rr.iteration)) & 1) == 0;
    let other_px = select(px - axis, px + axis, is_first);
    if (!rr_available(other_px)) {
        return here;
    }
    let other = rr_load(other_px);
    if (!sortable(px, here.rgb) || !sortable(other_px, other.rgb)) {
        return here;
    }

    let first = sort_key(select(other, here, is_first).rgb);
    let second = sort_key(select(here, other, is_first).rgb);
    let out_of_order = select(first > second, first < second, params.order == 1u);
    return select(here, other, out_of_order);
}
