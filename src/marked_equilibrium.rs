//! Equilibrium potentials of marked sets on the L x L torus.
//!
//!   marked_equilibrium <dir>   writes marked_equilibrium.csv (one row per marked set),
//!                              marked_patterns.csv (coordinates) and marked_summary.csv
//!
//! Patterns: periodic subgrid, compact square cluster, one random vertex per window
//! ("jittered") and uniformly random k-subsets. See data/README.md for the columns.
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};

#[derive(Clone, Copy, Default)]
struct C {
    re: f64,
    im: f64,
}
impl C {
    fn mul(self, z: C) -> C {
        C {
            re: self.re * z.re - self.im * z.im,
            im: self.re * z.im + self.im * z.re,
        }
    }
}
fn fft(a: &mut [C], inverse: bool) {
    let n = a.len();
    assert!(n.is_power_of_two());
    let mut j = 0;
    for i in 1..n {
        let mut b = n >> 1;
        while j & b != 0 {
            j ^= b;
            b >>= 1;
        }
        j ^= b;
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let t = (if inverse { 2.0 } else { -2.0 }) * PI / len as f64;
        let root = C {
            re: t.cos(),
            im: t.sin(),
        };
        for start in (0..n).step_by(len) {
            let mut w = C { re: 1.0, im: 0.0 };
            for r in 0..len / 2 {
                let u = a[start + r];
                let v = w.mul(a[start + r + len / 2]);
                a[start + r] = C {
                    re: u.re + v.re,
                    im: u.im + v.im,
                };
                a[start + r + len / 2] = C {
                    re: u.re - v.re,
                    im: u.im - v.im,
                };
                w = w.mul(root);
            }
        }
        len *= 2;
    }
    if inverse {
        for z in a {
            z.re /= n as f64;
            z.im /= n as f64;
        }
    }
}
fn fft2(a: &mut [C], l: usize, inverse: bool) {
    for row in a.chunks_mut(l) {
        fft(row, inverse);
    }
    let mut col = vec![C::default(); l];
    for x in 0..l {
        for y in 0..l {
            col[y] = a[y * l + x];
        }
        fft(&mut col, inverse);
        for y in 0..l {
            a[y * l + x] = col[y];
        }
    }
}
struct Torus {
    l: usize,
    inv: Vec<f64>,
    green: Vec<f64>,
}
impl Torus {
    fn new(l: usize) -> Self {
        assert!(l >= 4 && l.is_power_of_two());
        let s: Vec<f64> = (0..l)
            .map(|i| (PI * i as f64 / l as f64).sin().powi(2))
            .collect();
        let mut inv = vec![0.0; l * l];
        for y in 0..l {
            for x in 0..l {
                if x + y != 0 {
                    inv[y * l + x] = 1.0 / (4.0 * (s[x] + s[y]));
                }
            }
        }
        let mut a: Vec<C> = inv.iter().map(|&re| C { re, im: 0.0 }).collect();
        fft2(&mut a, l, true);
        assert!(a.iter().map(|z| z.im.abs()).fold(0.0, f64::max) < 1e-10);
        Self {
            l,
            inv,
            green: a.iter().map(|z| z.re).collect(),
        }
    }
    fn displacement(&self, x: usize, y: usize) -> usize {
        ((x / self.l + self.l - y / self.l) % self.l) * self.l
            + (x % self.l + self.l - y % self.l) % self.l
    }
    fn neighbours(&self, x: usize) -> [usize; 4] {
        let l = self.l;
        let (a, b) = (x % l, x / l);
        [
            b * l + (a + 1) % l,
            b * l + (a + l - 1) % l,
            ((b + 1) % l) * l + a,
            ((b + l - 1) % l) * l + a,
        ]
    }
    fn potential(&self, sites: &[usize], rho: &[f64]) -> Vec<f64> {
        let mut a = vec![C::default(); self.l * self.l];
        for (&x, &r) in sites.iter().zip(rho) {
            a[x].re = r;
        }
        fft2(&mut a, self.l, false);
        for (z, &r) in a.iter_mut().zip(&self.inv) {
            z.re *= r;
            z.im *= r;
        }
        fft2(&mut a, self.l, true);
        assert!(a.iter().map(|z| z.im.abs()).fold(0.0, f64::max) < 1e-9);
        a.iter().map(|z| z.re).collect()
    }
}
fn solve_spd(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut low = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for h in 0..j {
                s -= low[i * n + h] * low[j * n + h];
            }
            if i == j {
                assert!(s > 0.0, "not positive definite: {i}, {s}");
                low[i * n + j] = s.sqrt();
            } else {
                low[i * n + j] = s / low[j * n + j];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= low[i * n + j] * x[j];
        }
        x[i] = s / low[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in i + 1..n {
            s -= low[j * n + i] * x[j];
        }
        x[i] = s / low[i * n + i];
    }
    x
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn index(&mut self, n: usize) -> usize {
        let bound = u64::MAX - u64::MAX % n as u64;
        loop {
            let v = self.next();
            if v < bound {
                return (v % n as u64) as usize;
            }
        }
    }
}
fn pattern(l: usize, q: usize, kind: &str, seed: u64) -> Vec<usize> {
    let mut rng = Rng(seed);
    let mut sites = Vec::new();
    let ell = l / q;
    match kind {
        "periodic" => {
            for y in 0..q {
                for x in 0..q {
                    sites.push(y * ell * l + x * ell);
                }
            }
        }
        "cluster" => {
            for y in 0..q {
                for x in 0..q {
                    sites.push(y * l + x);
                }
            }
        }
        "jittered" => {
            for y in 0..q {
                for x in 0..q {
                    sites.push((y * ell + rng.index(ell)) * l + x * ell + rng.index(ell));
                }
            }
        }
        "random" => {
            let mut pool: Vec<usize> = (0..l * l).collect();
            for i in 0..q * q {
                let j = i + rng.index(l * l - i);
                pool.swap(i, j);
            }
            sites.extend_from_slice(&pool[..q * q]);
        }
        _ => panic!("unknown pattern"),
    }
    sites.sort_unstable();
    assert_eq!(sites.len(), q * q);
    assert!(sites.windows(2).all(|p| p[0] < p[1]));
    sites
}
// Averaging a fixed tiling over all translations equals averaging all square windows.
fn one_window_fraction(l: usize, ell: usize, sites: &[usize]) -> f64 {
    let mut marked = vec![0usize; l * l];
    for &x in sites {
        marked[x] = 1;
    }
    let mut rows = vec![0usize; l * l];
    for y in 0..l {
        let mut count = (0..ell).map(|x| marked[y * l + x]).sum::<usize>();
        for x in 0..l {
            rows[y * l + x] = count;
            count = count + marked[y * l + (x + ell) % l] - marked[y * l + x];
        }
    }
    let mut one = 0usize;
    for x in 0..l {
        let mut count = (0..ell).map(|y| rows[y * l + x]).sum::<usize>();
        for y in 0..l {
            if count == 1 {
                one += 1;
            }
            count = count + rows[((y + ell) % l) * l + x] - rows[y * l + x];
        }
    }
    one as f64 / (l * l) as f64
}
#[derive(Clone)]
struct ResultRow {
    l: usize,
    k: usize,
    kind: String,
    a: f64,
    s2: f64,
    eta: f64,
    uniform_energy: f64,
    rmin: f64,
    rmax: f64,
    plateau: f64,
    mean: f64,
    poisson: f64,
    energy: f64,
    aref_error: f64,
    sref_error: f64,
    f1: f64,
}
fn evaluate(t: &Torus, sites: &[usize], kind: &str) -> (ResultRow, Vec<f64>) {
    let k = sites.len();
    let n = t.l * t.l;
    assert!(k < n);
    let mat: Vec<f64> = (0..k * k)
        .map(|i| t.green[t.displacement(sites[i / k], sites[i % k])])
        .collect();
    let uniform_energy = mat.iter().sum::<f64>() / (k * k) as f64;
    let q = solve_spd(&mat, &vec![1.0; k]);
    let den = q.iter().sum::<f64>();
    let a = 1.0 / den;
    let rho: Vec<f64> = q.iter().map(|x| x / den).collect();
    let rmin = rho.iter().copied().fold(f64::INFINITY, f64::min);
    let rmax = rho.iter().copied().fold(0.0, f64::max);
    assert!(a > 0.0 && rmin >= 1.0 / n as f64 - 1e-10);
    assert!((rho.iter().sum::<f64>() - 1.0).abs() < 1e-11);
    assert!(uniform_energy >= a - 1e-10);
    let field = t.potential(sites, &rho);
    let plateau = sites
        .iter()
        .map(|&x| (field[x] - a).abs())
        .fold(0.0, f64::max);
    let mean = field.iter().sum::<f64>().abs() / n as f64;
    let s2 = field.iter().map(|x| x * x).sum::<f64>();
    let eta = s2.sqrt() / ((n as f64).sqrt() * a);
    let mut target = vec![-1.0 / n as f64; n];
    for (&x, &r) in sites.iter().zip(&rho) {
        target[x] += r;
    }
    let mut poisson: f64 = 0.0;
    let mut e = 0.0;
    for x in 0..n {
        let adj = t.neighbours(x);
        let lap = 4.0 * field[x] - adj.iter().map(|&y| field[y]).sum::<f64>();
        poisson = poisson.max((lap - target[x]).abs());
        e += (field[x] - field[adj[0]]).powi(2) + (field[x] - field[adj[2]]).powi(2);
    }
    let energy = (e - a).abs();
    assert!(plateau < 1e-9 && mean < 1e-10 && poisson < 1e-9 && energy < 1e-8 * a.max(1.0));
    assert!(field.iter().all(|&x| x <= a + 1e-9), "maximum principle");
    (
        ResultRow {
            l: t.l,
            k,
            kind: kind.into(),
            a,
            s2,
            eta,
            uniform_energy,
            rmin,
            rmax,
            plateau,
            mean,
            poisson,
            energy,
            aref_error: 0.0,
            sref_error: 0.0,
            f1: 0.0,
        },
        field,
    )
}
fn independent_dirichlet() {
    let t = Torus::new(8);
    let n = 64;
    for (q, kind) in [
        (1, "periodic"),
        (2, "periodic"),
        (2, "random"),
        (4, "random"),
        (4, "cluster"),
    ] {
        let sites = pattern(8, q, kind, 20261001);
        let (r, field) = evaluate(&t, &sites, kind);
        let mut marked = vec![false; n];
        for &x in &sites {
            marked[x] = true;
        }
        let unmarked: Vec<usize> = (0..n).filter(|&x| !marked[x]).collect();
        let m = unmarked.len();
        let mut indices = vec![usize::MAX; n];
        for (i, &x) in unmarked.iter().enumerate() {
            indices[x] = i;
        }
        let mut a = vec![0.0; m * m];
        for (i, &x) in unmarked.iter().enumerate() {
            a[i * m + i] = 4.0;
            for y in t.neighbours(x) {
                if !marked[y] {
                    a[i * m + indices[y]] = -1.0;
                }
            }
        }
        let h = solve_spd(&a, &vec![4.0; m]);
        let ah = h.iter().sum::<f64>() / (4 * n * n) as f64;
        let err = unmarked
            .iter()
            .enumerate()
            .map(|(i, &x)| (h[i] - 4.0 * n as f64 * (r.a - field[x])).abs())
            .fold(0.0, f64::max);
        assert!((r.a - ah).abs() < 1e-11 && err < 1e-9);
        eprintln!(
            "independent Dirichlet L=8 k={} {kind}: A_error={:.3e}, hit_error={err:.3e}",
            q * q,
            (r.a - ah).abs()
        );
    }
}
fn median(v: &[f64]) -> f64 {
    let mut a = v.to_vec();
    a.sort_by(f64::total_cmp);
    let m = a.len() / 2;
    if a.len() % 2 == 0 {
        (a[m - 1] + a[m]) / 2.0
    } else {
        a[m]
    }
}
fn lower(r: &ResultRow, p: f64, sharp: bool) -> f64 {
    let n = (r.l * r.l) as f64;
    let alpha = 1.0 - (1.0 - p).sqrt() - (r.k as f64 / n).sqrt();
    let denom = if sharp {
        (r.a / 2.0).sqrt()
    } else {
        (2.0 * r.a).sqrt()
    };
    ((alpha * n.sqrt() * r.a - r.s2.sqrt()) / denom).max(0.0)
}
fn main() -> std::io::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "output/data".into());
    create_dir_all(&dir)?;
    independent_dirichlet();
    let mut raw = BufWriter::new(File::create(format!("{dir}/marked_equilibrium.csv"))?);
    let mut loc = BufWriter::new(File::create(format!("{dir}/marked_patterns.csv"))?);
    writeln!(raw,"pattern,L,N,k,rep,seed,A,S2,eta,scaled_A_4pi_k_over_log_Nk,uniform_energy,uniform_over_equilibrium,rho_min,rho_max,t_hit,sqrt_t_hit,lower_p_two_thirds,lower_p_one,periodic_sharp_p_two_thirds,plateau_error,mean_error,poisson_error,energy_error,periodic_A_error,periodic_S2_error,one_window_fraction")?;
    writeln!(loc, "pattern,L,k,rep,seed,x,y")?;
    let mut all: Vec<ResultRow> = Vec::new();
    let mut quotients: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    for l in [64, 128, 256, 512, 1024] {
        let start = std::time::Instant::now();
        let t = Torus::new(l);
        for q in [1, 2, 4, 8, 16] {
            let k = q * q;
            let ell = l / q;
            for kind in ["periodic", "cluster", "jittered", "random"] {
                if q == 1 && kind != "periodic" {
                    continue;
                }
                let reps = match kind {
                    "random" => {
                        if l == 1024 {
                            4
                        } else {
                            8
                        }
                    }
                    "jittered" => {
                        if l == 1024 {
                            2
                        } else {
                            4
                        }
                    }
                    _ => 1,
                };
                for rep in 0..reps {
                    let seed = 20261001 + l as u64 * 10000 + k as u64 * 20 + rep as u64;
                    let sites = pattern(l, q, kind, seed);
                    let (mut r, _) = evaluate(&t, &sites, kind);
                    r.f1 = one_window_fraction(l, ell, &sites);
                    if kind == "periodic" {
                        assert!((r.f1 - 1.0).abs() < 1e-14);
                        let (aq, sq) = *quotients.entry(ell).or_insert_with(|| {
                            let z = Torus::new(ell);
                            (z.green[0], z.green.iter().map(|x| x * x).sum::<f64>())
                        });
                        r.aref_error = (r.a - aq / k as f64).abs();
                        r.sref_error = (r.s2 - sq / k as f64).abs();
                        assert!(r.aref_error < 1e-10 && r.sref_error < 1e-8 * sq.max(1.0));
                    }
                    let n = (l * l) as f64;
                    let hit = 4.0 * n * r.a;
                    let scaled = 4.0 * PI * k as f64 * r.a / (n / k as f64).ln();
                    let sharp = if kind == "periodic" {
                        lower(&r, 2.0 / 3.0, true)
                    } else {
                        0.0
                    };
                    writeln!(raw,"{kind},{l},{},{k},{rep},{seed},{:.16e},{:.16e},{:.12e},{scaled:.12e},{:.16e},{:.12e},{:.12e},{:.12e},{hit:.12e},{:.12e},{:.12e},{:.12e},{sharp:.12e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.12e}",l*l,r.a,r.s2,r.eta,r.uniform_energy,r.uniform_energy/r.a,r.rmin,r.rmax,hit.sqrt(),lower(&r,2.0/3.0,false),lower(&r,1.0,false),r.plateau,r.mean,r.poisson,r.energy,r.aref_error,r.sref_error,r.f1)?;
                    for &x in &sites {
                        writeln!(loc, "{kind},{l},{k},{rep},{seed},{},{}", x % l, x / l)?;
                    }
                    all.push(r);
                }
            }
        }
        raw.flush()?;
        loc.flush()?;
        eprintln!(
            "completed L={l}; cumulative patterns={}; elapsed {:.2}s",
            all.len(),
            start.elapsed().as_secs_f64()
        );
    }
    let mut groups: BTreeMap<(usize, usize, String), Vec<&ResultRow>> = BTreeMap::new();
    for r in &all {
        groups
            .entry((r.l, r.k, r.kind.clone()))
            .or_default()
            .push(r);
    }
    let mut summary = BufWriter::new(File::create(format!("{dir}/marked_summary.csv"))?);
    writeln!(summary,"L,N,k,N_over_k,pattern,replicates,A_median,A_min,A_max,scaled_A_median,eta_median,eta_max,uniform_over_equilibrium_median,t_hit_median,lower_p_two_thirds_median,lower_p_one_median,positive_two_thirds_count,one_window_fraction_median")?;
    for ((l, k, kind), rs) in groups {
        let av: Vec<f64> = rs.iter().map(|r| r.a).collect();
        let et: Vec<f64> = rs.iter().map(|r| r.eta).collect();
        let uv: Vec<f64> = rs.iter().map(|r| r.uniform_energy / r.a).collect();
        let lb: Vec<f64> = rs.iter().map(|r| lower(r, 2.0 / 3.0, false)).collect();
        let f1: Vec<f64> = rs.iter().map(|r| r.f1).collect();
        let lb1: Vec<f64> = rs.iter().map(|r| lower(r, 1.0, false)).collect();
        let n = l * l;
        let am = median(&av);
        writeln!(summary,"{l},{n},{k},{},{kind},{},{am:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{},{:.12e}",n/k,rs.len(),av.iter().copied().fold(f64::INFINITY,f64::min),av.iter().copied().fold(0.0,f64::max),4.0*PI*k as f64*am/(n as f64/k as f64).ln(),median(&et),et.iter().copied().fold(0.0,f64::max),median(&uv),4.0*n as f64*am,median(&lb),median(&lb1),lb.iter().filter(|&&v|v>0.0).count(),median(&f1))?;
    }
    summary.flush()?;
    eprintln!(
        "{} marked sets: all checks passed",
        all.len()
    );
    Ok(())
}
