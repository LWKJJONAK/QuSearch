//! Finite-size check of the electrostatic bound on L x L tori, L = 5, 8, 12, 16.
//!
//! Verifies the Poisson equation, zero mean and energy identity of the potential, then
//! runs every marked input with position- and time-dependent local steps and queries and
//! asserts query invariance of F, the speed limit at every step and the endpoint estimate.
//! Prints one CSV row per size; any violated assertion aborts the program.
use std::f64::consts::PI;

fn green(l: usize) -> Vec<f64> {
    let n = l * l;
    let mut k = vec![0.0; n];
    for q in 0..l {
        for p in 0..l {
            if p == 0 && q == 0 {
                continue;
            }
            let lam = 4.0
                * ((PI * p as f64 / l as f64).sin().powi(2)
                    + (PI * q as f64 / l as f64).sin().powi(2));
            for y in 0..l {
                for x in 0..l {
                    k[y * l + x] +=
                        (2.0 * PI * (p * x + q * y) as f64 / l as f64).cos() / (n as f64 * lam);
                }
            }
        }
    }
    k
}

fn kernel_at(k: &[f64], l: usize, x: usize, m: usize) -> f64 {
    let dx = (x % l + l - m % l) % l;
    let dy = (x / l + l - m / l) % l;
    k[dy * l + dx]
}

fn forward(state: &[f64], l: usize, t: usize) -> Vec<f64> {
    let mut result = vec![0.0; state.len()];
    for x in 0..l * l {
        // Arbitrary onsite rotations, followed by conditional nearest-neighbour
        // shifts along alternating axes. Inhomogeneous and time-dependent.
        let angle = 0.173 * (x + 3 * t + 1) as f64 + 0.09 * ((x * t + 7) % 11) as f64;
        let (s, c) = angle.sin_cos();
        let a = c * state[2 * x] - s * state[2 * x + 1];
        let b = s * state[2 * x] + c * state[2 * x + 1];
        let plus = if t % 2 == 0 {
            (x / l) * l + (x % l + 1) % l
        } else {
            ((x / l + 1) % l) * l + x % l
        };
        let minus = if t % 2 == 0 {
            (x / l) * l + (x % l + l - 1) % l
        } else {
            ((x / l + l - 1) % l) * l + x % l
        };
        result[2 * plus] = a;
        result[2 * minus + 1] = b;
    }
    result
}

fn weighted_overlap(free: &[f64], marked: &[Vec<f64>], k: &[f64], l: usize) -> f64 {
    let n = l * l;
    let mut value = 0.0;
    for m in 0..n {
        for x in 0..n {
            for coin in 0..2 {
                value +=
                    free[2 * x + coin] * (k[0] - kernel_at(k, l, x, m)) * marked[m][2 * x + coin]
                        / n as f64;
            }
        }
    }
    value
}

fn norm_error(state: &[f64]) -> f64 {
    (state.iter().map(|x| x * x).sum::<f64>() - 1.0).abs()
}

fn main() {
    println!("L,N,A,S2,energy,laplace_error,mean_error,F_initial_error,max_step_fraction,max_commutator_fraction,endpoint_remainder,query_invariance_error,norm_error");
    for l in [5, 8, 12, 16] {
        let n = l * l;
        let k = green(l);
        let a = k[0];
        let s2 = k.iter().map(|x| x * x).sum::<f64>();
        let mean = k.iter().sum::<f64>();
        let mut energy = 0.0;
        let mut laplace_error: f64 = 0.0;
        for x in 0..n {
            let xp = (x / l) * l + (x % l + 1) % l;
            let xm = (x / l) * l + (x % l + l - 1) % l;
            let yp = ((x / l + 1) % l) * l + x % l;
            let ym = ((x / l + l - 1) % l) * l + x % l;
            energy += (k[x] - k[xp]).powi(2) + (k[x] - k[yp]).powi(2);
            let rhs = if x == 0 {
                1.0 - 1.0 / n as f64
            } else {
                -1.0 / n as f64
            };
            laplace_error =
                laplace_error.max((4.0 * k[x] - k[xp] - k[xm] - k[yp] - k[ym] - rhs).abs());
        }
        assert!(laplace_error < 1e-11 && mean.abs() < 1e-10 && (energy - a).abs() < 1e-11);
        // Deliberately localized initial state; free probability is not flat.
        let mut free = vec![0.0; 2 * n];
        free[0] = 1.0;
        let mut marked = vec![free.clone(); n];
        let initial = weighted_overlap(&free, &marked, &k, l);
        assert!((initial - a).abs() < 1e-12);
        let mut old = initial;
        let rate = (2.0 * a / n as f64).sqrt();
        let mut max_step_fraction: f64 = 0.0;
        let mut max_commutator_fraction: f64 = 0.0;
        let mut query_error: f64 = 0.0;
        let mut max_norm_error: f64 = 0.0;
        for t in 0..80 {
            // Check the query separately, before any propagation changes F.
            for m in 0..n {
                let (s, c) = (0.71 + 0.13 * t as f64).sin_cos();
                let aa = marked[m][2 * m];
                let bb = marked[m][2 * m + 1];
                marked[m][2 * m] = c * aa - s * bb;
                marked[m][2 * m + 1] = s * aa + c * bb;
                max_norm_error = max_norm_error.max(norm_error(&marked[m]));
            }
            query_error = query_error.max((weighted_overlap(&free, &marked, &k, l) - old).abs());
            assert!(query_error < 1e-12, "queries changed F: {query_error}");
            let next_free = forward(&free, l, t);
            let mut commutator_mean = 0.0;
            for m in 0..n {
                let weighted: Vec<f64> = (0..2 * n)
                    .map(|i| kernel_at(&k, l, i / 2, m) * free[i])
                    .collect();
                let evolved = forward(&weighted, l, t);
                let mut comm_norm2 = 0.0;
                for i in 0..2 * n {
                    let comm = kernel_at(&k, l, i / 2, m) * next_free[i] - evolved[i];
                    comm_norm2 += comm * comm;
                }
                commutator_mean += comm_norm2 / n as f64;
                marked[m] = forward(&marked[m], l, t);
                max_norm_error = max_norm_error.max(norm_error(&marked[m]));
            }
            max_norm_error = max_norm_error.max(norm_error(&next_free));
            assert!(
                max_norm_error < 1e-12,
                "state normalization changed: {max_norm_error}"
            );
            let cf = commutator_mean / (2.0 * a / n as f64);
            assert!(cf <= 1.0 + 1e-10, "mean commutator bound exceeded: {cf}");
            max_commutator_fraction = max_commutator_fraction.max(cf);
            free = next_free;
            let new = weighted_overlap(&free, &marked, &k, l);
            let sf = (new - old).abs() / rate;
            assert!(sf <= 1.0 + 1e-10, "speed limit exceeded: {sf}");
            max_step_fraction = max_step_fraction.max(sf);
            old = new;
        }
        let average_overlap = marked
            .iter()
            .map(|psi| free.iter().zip(psi).map(|(x, y)| x * y).sum::<f64>())
            .sum::<f64>()
            / n as f64;
        let endpoint = (old - a * average_overlap).abs();
        assert!(endpoint <= (s2 / n as f64).sqrt() + 1e-10);
        println!("{l},{n},{a:.12},{s2:.12},{energy:.12},{laplace_error:.3e},{:.3e},{:.3e},{max_step_fraction:.12},{max_commutator_fraction:.12},{endpoint:.12},{query_error:.3e},{max_norm_error:.3e}",mean.abs(),(initial-a).abs());
    }
}
