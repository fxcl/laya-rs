use candle_core::{DType, Device, Module, Tensor};
fn bench<F: FnMut()>(n: usize, label: &str, mut f: F) {
    for _ in 0..3 {
        f();
    }
    let t = std::time::Instant::now();
    for _ in 0..n {
        f();
    }
    println!(
        "{:30}: {:8.2} ms/call",
        label,
        t.elapsed().as_secs_f64() * 1000.0 / n as f64
    );
}
fn main() {
    let dev = Device::Cpu;
    let (b, l, h, heads, hd) = (5usize, 117usize, 1024usize, 16usize, 64usize);
    let x = Tensor::randn(0f32, 1f32, (b, l, h), &dev).unwrap();
    let w = Tensor::ones(h, DType::F32, &dev).unwrap();
    let ln = candle_nn::LayerNorm::new_no_bias(w, 1e-5);
    bench(84, "layernorm (5,117,1024)", || {
        let _ = ln.forward(&x).unwrap();
    });
    let x2 = x.reshape((b * l, h)).unwrap();
    let wm = Tensor::randn(0f32, 1f32, (h, h), &dev).unwrap();
    bench(112, "linear (585,1024)@(1024,1024)", || {
        let _ = x2.matmul(&wm).unwrap();
    });
    let q = Tensor::randn(0f32, 1f32, (b, heads, l, hd), &dev).unwrap();
    let k = Tensor::randn(0f32, 1f32, (b, heads, l, hd), &dev).unwrap();
    let v = Tensor::randn(0f32, 1f32, (b, heads, l, hd), &dev).unwrap();
    bench(28, "attention QK^T+AV", || {
        let _ = q.matmul(&k.t().unwrap()).unwrap().matmul(&v).unwrap();
    });
    let s = Tensor::randn(0f32, 1f32, (b, heads, l, l), &dev).unwrap();
    bench(28, "softmax (5,16,117,117)", || {
        let _ = candle_nn::ops::softmax(&s, candle_core::D::Minus1).unwrap();
    });
    let w1 = Tensor::randn(0f32, 1f32, (h, 4 * h), &dev).unwrap();
    let w2 = Tensor::randn(0f32, 1f32, (4 * h, h), &dev).unwrap();
    bench(28, "FFN (585,1024)@(1024,4096)x2", || {
        let _ = x2.matmul(&w1).unwrap().matmul(&w2).unwrap();
    });
}
