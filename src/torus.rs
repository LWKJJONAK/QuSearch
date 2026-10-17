//! Spectral sums for the self-potential on d-dimensional tori, and a simulation of
//! Tulsi's walk on the L x L torus that is independent of python/make_data.py.
//!
//!   torus spectral > spectral.csv            self-potential A for d = 1, 2, 3
//!   torus walk     > tulsi_independent.csv   peak step and success probability
//!
//! The walk has four real coin amplitudes and one control qubit per vertex. One
//! iteration is one rank-one marked reflection followed by one local step.
use std::f64::consts::PI;

fn self_potential(l: usize, d: usize) -> f64 {
    // A = (1/L^d) sum_{k != 0} 1/lambda_k, lambda_k = 4 sum_i sin^2(pi k_i / L), with Kahan
    // summation. The sine form avoids cancellation at small k.
    let s: Vec<f64> = (0..l)
        .map(|p| (PI * p as f64 / l as f64).sin().powi(2))
        .collect();
    let n = l.pow(d as u32);
    let mut total = 0.0;
    let mut compensation = 0.0;
    for index in 1..n {
        let mut a = index;
        let mut den = 0.0;
        for _ in 0..d {
            den += s[a % l];
            a /= l;
        }
        let value = 1.0 / (4.0 * den);
        let y = value - compensation;
        let t = total + y;
        compensation = (t - total) - y;
        total = t;
    }
    total / n as f64
}

fn run(l: usize, a: f64, mark: usize) -> (usize, f64, f64) {
    let n = l * l;
    let c = 1.0 / (1.0 + 8.0 * a).sqrt();
    let s = (1.0 - c * c).sqrt();
    let mut parked = vec![0.0_f64; 4 * n];
    let mut walking = vec![1.0 / (2.0 * l as f64); 4 * n];
    let mut next = vec![0.0_f64; 4 * n];
    let period_scale = 2.0 * a + s * s / (4.0 * c * c);
    let predicted = 0.5 * PI * l as f64 * period_scale.sqrt();
    // Observation window: 1.5 times the two-level estimate of the first peak.
    let max_steps = (1.5 * predicted).ceil() as usize;
    let mut t_peak = 0;
    let mut p_peak = 1.0 / n as f64;
    let mut norm_error: f64 = 0.0;
    for t in 1..=max_steps {
        // R = I - 2 |delta_1, s, marked><delta_1, s, marked|,
        // with |delta_1> = -s |0> + c |1>, |s> = (1,1,1,1)/2.
        let at = 4 * mark;
        let projection = 0.5
            * (-s * parked[at..at + 4].iter().sum::<f64>()
                + c * walking[at..at + 4].iter().sum::<f64>());
        for j in 0..4 {
            parked[at + j] += s * projection;
            walking[at + j] -= c * projection;
        }
        // C = diag(-I, W), W = flip-flop shift * Grover coin.
        for a in &mut parked {
            *a = -*a;
        }
        for y in 0..l {
            for x in 0..l {
                let a = 4 * (y * l + x);
                let twice_mean = 0.5 * walking[a..a + 4].iter().sum::<f64>();
                next[4 * (y * l + (x + 1) % l) + 1] = twice_mean - walking[a];
                next[4 * (y * l + (x + l - 1) % l)] = twice_mean - walking[a + 1];
                next[4 * (((y + 1) % l) * l + x) + 3] = twice_mean - walking[a + 2];
                next[4 * (((y + l - 1) % l) * l + x) + 2] = twice_mean - walking[a + 3];
            }
        }
        std::mem::swap(&mut walking, &mut next);
        let p = parked[at..at + 4]
            .iter()
            .chain(walking[at..at + 4].iter())
            .map(|a| a * a)
            .sum();
        if p > p_peak {
            p_peak = p;
            t_peak = t;
        }
        if t == 1 || t == max_steps || t % 100 == 0 {
            let norm: f64 = parked.iter().chain(walking.iter()).map(|a| a * a).sum();
            norm_error = norm_error.max((norm - 1.0).abs());
        }
    }
    (t_peak, p_peak, norm_error)
}

fn main() {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "spectral".to_owned());
    if mode == "spectral" {
        println!("dimension,L,N,A");
        for (d, sizes) in [
            (1, vec![16, 32, 64, 128, 256, 512, 1024]),
            (2, vec![16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]),
            (3, vec![8, 16, 32, 64, 128]),
        ] {
            for l in sizes {
                let a = self_potential(l, d);
                if d == 1 {
                    let exact = (l * l - 1) as f64 / (12.0 * l as f64);
                    assert!((a - exact).abs() < 1e-9 * exact.max(1.0));
                }
                println!("{d},{l},{},{a:.12}", l.pow(d as u32));
            }
        }
    } else if mode == "walk" {
        println!("L,N,peak_step,peak_probability,norm_error");
        for l in [16, 32, 64, 128, 256] {
            let a = self_potential(l, 2);
            let (t, p, e) = run(l, a, 0);
            println!("{l},{},{t},{p:.12},{e:.4e}", l * l);
            assert!(e < 1e-9, "unitarity failure: {e}");
            if l == 16 {
                // Translation invariance: another marked vertex gives the same peak.
                let translated = run(l, a, 3 * l + 5);
                assert_eq!(translated.0, t);
                assert!((translated.1 - p).abs() < 1e-12, "translation mismatch");
            }
        }
    } else {
        panic!("use spectral or walk");
    }
}
