pub fn ln_add_exp(x: f32, y: f32) -> f32 {
    let diff = x - y;
    let result = if x == f32::NEG_INFINITY {
        y
    } else if y == f32::NEG_INFINITY {
        x
    } else if diff > 0. {
        x + (-diff).exp().ln_1p()
    } else {
        y + diff.exp().ln_1p()
    };
    if result.is_nan() {
        panic!("ln_add_exp({}, {}) = NaN", x, y);
    }
    result
}